from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

import distributed_dispatch as dispatch
import one_vs_three
import run_stage05_ab as ab
import run_stage05_fidelity as fidelity
import run_stage05_formal as formal
import run_stage05_winner_refine_distributed as common_dispatch
from toml_utils import write_toml_file


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
INTERACTIVE_REMOTE_PYTHON_HELPER = REPO_ROOT / 'scripts' / 'start_interactive_remote_python.ps1'
ROUND_KIND_FORMAL_1V3 = 'formal_1v3'
DISPATCH_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA_VERSION = 1
DEFAULT_COARSE_ITERS = 1
DEFAULT_EXTRA_ITERS = 1
DEFAULT_CLOSE_STDERR_MULT = 1.0
DEFAULT_CLOSE_EXTRA_ROUNDS = 3
DEFAULT_CLOSE_MAX_EXTRA_ROUNDS = 5
REMOTE_INTERACTIVE_TASK_NAME_PREFIX = 'MahjongAI-WinnerRefine-'
FORMAL_1V3_SEED_START_BASE = 10000
MANUAL_FORMAL_1V3_CONFIG_KEY = 'formal_1v3_config'
MANUAL_FORMAL_1V3_MODE = 'manual_shortlist'

POINT_VALUES = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)
RANK_VALUES = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

WorkerSpec = dispatch.WorkerSpec
ActiveTask = dispatch.ActiveTask
JsonTaskLaunchSpec = dispatch.JsonTaskLaunchSpec


def quote_ps(value: str) -> str:
    return dispatch.quote_ps(value)


def path_to_scp_remote(path: str | Path) -> str:
    return dispatch.path_to_scp_remote(path)


def dispatch_root_for_run(run_dir: Path) -> Path:
    return run_dir / 'distributed' / 'formal_1v3_dispatch'


def dispatch_state_path_for_run(run_dir: Path) -> Path:
    return dispatch_root_for_run(run_dir) / 'dispatch_state.json'


def dispatch_control_path_for_run(run_dir: Path) -> Path:
    return dispatch_root_for_run(run_dir) / 'dispatch_control.json'


def load_dispatch_state(path: Path) -> dict[str, Any]:
    return fidelity.load_json(path)


def write_dispatch_state(path: Path, payload: dict[str, Any]) -> None:
    payload['updated_at'] = fidelity.ts_now()
    fidelity.atomic_write_json(path, payload)


def dispatch_state_is_completed(dispatch_state: dict[str, Any]) -> bool:
    return (
        str(dispatch_state.get('stage') or '') == 'completed'
        or str(dispatch_state.get('status') or '') == 'completed'
    )


def map_repo_path_to_remote(local_path: str | Path, *, remote_repo: str | Path) -> Path:
    resolved = Path(local_path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f'path {resolved} is outside repo root {REPO_ROOT}') from exc
    return Path(remote_repo) / relative


def run_remote_powershell(
    worker: WorkerSpec,
    *,
    script: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ['ssh']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.append(worker.host or common_dispatch.DEFAULT_REMOTE_HOST)
    command.append(script)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if capture_output else subprocess.DEVNULL,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def ensure_remote_parent_dir(worker: WorkerSpec, remote_path: Path) -> None:
    completed = run_remote_powershell(
        worker,
        script=f"New-Item -ItemType Directory -Force -Path {quote_ps(str(remote_path.parent))} | Out-Null",
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'failed to create remote directory for {remote_path}: {completed.stdout.strip()}'
        )


def copy_local_file_to_remote(worker: WorkerSpec, *, local_path: Path, remote_path: Path) -> None:
    ensure_remote_parent_dir(worker, remote_path)
    command = ['scp']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.extend([str(local_path), f'{worker.host}:{path_to_scp_remote(remote_path)}'])
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'failed to copy {local_path} to {remote_path}: {completed.stdout.strip()}'
        )


def build_formal_candidate(
    checkpoint_type: str,
    candidate_payload: dict[str, Any],
    *,
    protocol_arm: str,
) -> fidelity.CandidateSpec:
    checkpoint_path = str(candidate_payload.get('path', '') or '').strip()
    if not checkpoint_path:
        raise RuntimeError(f'formal checkpoint-pack candidate `{checkpoint_type}` is missing a checkpoint path')
    return fidelity.CandidateSpec(
        arm_name=checkpoint_type,
        scheduler_profile='formal',
        curriculum_profile='formal',
        weight_profile='formal',
        window_profile='formal',
        cfg_overrides={},
        meta={
            'stage': ROUND_KIND_FORMAL_1V3,
            'checkpoint_type': checkpoint_type,
            'checkpoint_path': checkpoint_path,
            'protocol_arm': protocol_arm,
        },
    )


def _update_tensor_hash(hasher: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode('utf-8'))
    hasher.update(str(tuple(value.shape)).encode('utf-8'))
    hasher.update(value.numpy().tobytes())


def checkpoint_inference_fingerprint(checkpoint_path: Path) -> str:
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f'checkpoint at {checkpoint_path} is not a valid state dict payload')
    saved_cfg = state.get('config')
    if not isinstance(saved_cfg, dict):
        raise RuntimeError(f'checkpoint at {checkpoint_path} is missing saved config')
    resnet_cfg = saved_cfg.get('resnet')
    control_cfg = saved_cfg.get('control')
    if not isinstance(resnet_cfg, dict) or not isinstance(control_cfg, dict):
        raise RuntimeError(f'checkpoint at {checkpoint_path} is missing architecture config')

    hasher = hashlib.sha256()
    hasher.update(str(control_cfg.get('version', 1)).encode('utf-8'))
    hasher.update(str(resnet_cfg.get('conv_channels')).encode('utf-8'))
    hasher.update(str(resnet_cfg.get('num_blocks')).encode('utf-8'))

    for module_name in ('mortal', 'policy_net'):
        module_state = state.get(module_name)
        if not isinstance(module_state, dict):
            raise RuntimeError(f'checkpoint at {checkpoint_path} is missing `{module_name}` weights')
        hasher.update(module_name.encode('utf-8'))
        for tensor_name in sorted(module_state):
            tensor = module_state[tensor_name]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f'checkpoint at {checkpoint_path} has non-tensor entry `{module_name}.{tensor_name}`'
                )
            hasher.update(tensor_name.encode('utf-8'))
            _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def dedupe_formal_shortlist_candidates(
    shortlist_checkpoint_types: list[str],
    shortlist_candidates: dict[str, Any],
) -> dict[str, Any]:
    representative_by_fingerprint: dict[str, str] = {}
    representative_for_checkpoint: dict[str, str] = {}
    groups: dict[str, dict[str, Any]] = {}
    unique_checkpoint_types: list[str] = []

    for checkpoint_type in shortlist_checkpoint_types:
        candidate_payload = shortlist_candidates.get(checkpoint_type)
        if not isinstance(candidate_payload, dict):
            raise RuntimeError(f'formal shortlist is missing candidate payload for `{checkpoint_type}`')
        checkpoint_path = Path(str(candidate_payload.get('path', '') or '')).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'formal shortlist checkpoint for `{checkpoint_type}` does not exist: {checkpoint_path}'
            )
        fingerprint = checkpoint_inference_fingerprint(checkpoint_path)
        representative = representative_by_fingerprint.get(fingerprint)
        if representative is None:
            representative = checkpoint_type
            representative_by_fingerprint[fingerprint] = representative
            unique_checkpoint_types.append(representative)
            groups[representative] = {
                'representative': representative,
                'members': [checkpoint_type],
                'fingerprint': fingerprint,
                'checkpoint_path': str(checkpoint_path),
            }
        else:
            groups[representative]['members'].append(checkpoint_type)
        representative_for_checkpoint[checkpoint_type] = representative

    duplicate_checkpoint_types = [
        checkpoint_type
        for checkpoint_type in shortlist_checkpoint_types
        if representative_for_checkpoint[checkpoint_type] != checkpoint_type
    ]
    return {
        'unique_checkpoint_types': list(unique_checkpoint_types),
        'representative_for_checkpoint': dict(representative_for_checkpoint),
        'groups': {
            representative: {
                'representative': representative,
                'members': list(group['members']),
                'fingerprint': str(group['fingerprint']),
                'checkpoint_path': str(group['checkpoint_path']),
            }
            for representative, group in groups.items()
        },
        'duplicate_checkpoint_types': duplicate_checkpoint_types,
    }


def load_frozen_formal_config_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    snapshot = result.get('config_snapshot')
    if not isinstance(snapshot, dict):
        raise RuntimeError(
            'formal result is missing frozen config snapshot; rerun formal_train before formal_1v3'
        )
    frozen_base_cfg = snapshot.get('base_cfg')
    frozen_1v3_cfg = snapshot.get('base_1v3_cfg')
    if not isinstance(frozen_base_cfg, dict) or not isinstance(frozen_1v3_cfg, dict):
        raise RuntimeError(
            'formal config snapshot is incomplete; rerun formal_train before formal_1v3'
        )
    return {
        'snapshot': deepcopy(snapshot),
        'base_cfg': deepcopy(frozen_base_cfg),
        'base_1v3_cfg': deepcopy(frozen_1v3_cfg),
    }


def normalize_manual_shortlist_candidates(raw_candidates: Any) -> dict[str, dict[str, Any]]:
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_candidates, dict):
        iterator = raw_candidates.items()
    elif isinstance(raw_candidates, list):
        iterator = []
        for entry in raw_candidates:
            if not isinstance(entry, dict):
                raise RuntimeError('manual formal_1v3 candidates must be a dict or a list of dict entries')
            arm_name = str(entry.get('arm_name') or entry.get('checkpoint_type') or '').strip()
            if not arm_name:
                raise RuntimeError('manual formal_1v3 candidate entry is missing `arm_name`')
            iterator.append((arm_name, entry))
    else:
        raise RuntimeError('manual formal_1v3 candidates must be a dict or a list')

    for raw_name, raw_payload in iterator:
        arm_name = str(raw_name or '').strip()
        if not arm_name:
            raise RuntimeError('manual formal_1v3 candidate arm name cannot be empty')
        if not isinstance(raw_payload, dict):
            raise RuntimeError(f'manual formal_1v3 candidate `{arm_name}` payload must be a dict')
        payload = dict(raw_payload)
        checkpoint_path = str(payload.get('path') or payload.get('checkpoint_path') or '').strip()
        if not checkpoint_path:
            raise RuntimeError(f'manual formal_1v3 candidate `{arm_name}` is missing checkpoint path')
        payload['path'] = checkpoint_path
        items.append((arm_name, payload))

    normalized: dict[str, dict[str, Any]] = {}
    for arm_name, payload in items:
        if arm_name in normalized:
            raise RuntimeError(f'duplicate manual formal_1v3 candidate `{arm_name}`')
        normalized[arm_name] = payload
    if not normalized:
        raise RuntimeError('manual formal_1v3 shortlist is empty')
    return normalized


def load_manual_formal_context(
    run_dir: Path,
    *,
    state_path: Path,
    state: dict[str, Any],
    manual_config: dict[str, Any],
) -> dict[str, Any]:
    shortlist_candidates = normalize_manual_shortlist_candidates(manual_config.get('candidates'))
    protocol_arm = str(manual_config.get('protocol_arm') or '').strip()
    if not protocol_arm:
        raise RuntimeError('manual formal_1v3 config is missing protocol arm')
    current_primary_protocol_arm = (
        str(manual_config.get('current_primary_protocol_arm') or '').strip() or protocol_arm
    )
    frozen_config = load_frozen_formal_config_snapshot(manual_config)
    shortlist_checkpoint_types = list(
        manual_config.get('shortlist_checkpoint_types')
        or shortlist_candidates.keys()
    )
    checkpoint_pack_types = list(
        manual_config.get('checkpoint_pack_types')
        or shortlist_checkpoint_types
    )
    checkpoint_dedupe = dedupe_formal_shortlist_candidates(
        shortlist_checkpoint_types,
        shortlist_candidates,
    )
    unique_checkpoint_types = list(checkpoint_dedupe['unique_checkpoint_types'])
    candidates = [
        build_formal_candidate(
            checkpoint_type,
            shortlist_candidates[checkpoint_type],
            protocol_arm=protocol_arm,
        )
        for checkpoint_type in unique_checkpoint_types
    ]
    publish_canonical_aliases = bool(manual_config.get('publish_canonical_aliases', False))
    pending_canonical_alias_targets = (
        [str(target) for target in manual_config.get('pending_canonical_alias_targets') or [] if str(target)]
        if publish_canonical_aliases
        else []
    )
    return {
        'run_name': run_dir.name,
        'run_dir': run_dir,
        'state_path': state_path,
        'state': state,
        'config_snapshot': frozen_config['snapshot'],
        'base_cfg': frozen_config['base_cfg'],
        'base_1v3_cfg': frozen_config['base_1v3_cfg'],
        'protocol_arm': protocol_arm,
        'current_primary_protocol_arm': current_primary_protocol_arm,
        'shortlist_candidates': shortlist_candidates,
        'shortlist_checkpoint_types': shortlist_checkpoint_types,
        'unique_shortlist_checkpoint_types': unique_checkpoint_types,
        'checkpoint_pack_types': checkpoint_pack_types,
        'checkpoint_dedupe': checkpoint_dedupe,
        'pending_canonical_alias_targets': pending_canonical_alias_targets,
        'publish_canonical_aliases': publish_canonical_aliases,
        'candidates': candidates,
        'candidate_index': {candidate.arm_name: candidate for candidate in candidates},
        'candidate_order': {
            candidate.arm_name: index
            for index, candidate in enumerate(candidates)
        },
    }


def load_formal_context(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing state.json under {run_dir}')
    state = fidelity.load_json(state_path)
    manual_config = state.get(MANUAL_FORMAL_1V3_CONFIG_KEY)
    if isinstance(manual_config, dict):
        mode = str(manual_config.get('mode') or MANUAL_FORMAL_1V3_MODE).strip()
        if mode != MANUAL_FORMAL_1V3_MODE:
            raise RuntimeError(f'unsupported manual formal_1v3 mode `{mode}`')
        return load_manual_formal_context(
            run_dir,
            state_path=state_path,
            state=state,
            manual_config=manual_config,
        )
    formal_state = state.get('formal')
    if not isinstance(formal_state, dict) or formal_state.get('status') != 'completed':
        raise RuntimeError('formal checkpoint pack is missing; formal_1v3 requires a completed formal train stage')
    result = formal_state.get('result')
    if not isinstance(result, dict):
        raise RuntimeError('formal result payload is missing')
    shortlist_candidates = result.get('candidates')
    if not isinstance(shortlist_candidates, dict):
        raise RuntimeError('formal result is missing checkpoint-pack candidates')
    protocol_arm = str(
        result.get('selected_protocol_arm')
        or state.get('final_conclusion', {}).get('p1_protocol_winner')
        or ''
    ).strip()
    if not protocol_arm:
        raise RuntimeError('formal result is missing selected protocol arm')
    current_primary_protocol_arm = str(
        result.get('current_primary_protocol_arm')
        or formal.CURRENT_PRIMARY_PROTOCOL_ARM
        or ''
    ).strip()
    frozen_config = load_frozen_formal_config_snapshot(result)
    shortlist_checkpoint_types = list(
        formal_state.get('shortlist_checkpoint_types')
        or result.get('shortlist_checkpoint_types')
        or formal.FORMAL_SHORTLIST_CHECKPOINT_TYPES
    )
    checkpoint_dedupe = dedupe_formal_shortlist_candidates(
        shortlist_checkpoint_types,
        shortlist_candidates,
    )
    unique_checkpoint_types = list(checkpoint_dedupe['unique_checkpoint_types'])
    candidates = [
        build_formal_candidate(
            checkpoint_type,
            shortlist_candidates[checkpoint_type],
            protocol_arm=protocol_arm,
        )
        for checkpoint_type in unique_checkpoint_types
    ]
    return {
        'run_name': run_dir.name,
        'run_dir': run_dir,
        'state_path': state_path,
        'state': state,
        'config_snapshot': frozen_config['snapshot'],
        'base_cfg': frozen_config['base_cfg'],
        'base_1v3_cfg': frozen_config['base_1v3_cfg'],
        'protocol_arm': protocol_arm,
        'current_primary_protocol_arm': current_primary_protocol_arm,
        'shortlist_candidates': shortlist_candidates,
        'shortlist_checkpoint_types': shortlist_checkpoint_types,
        'unique_shortlist_checkpoint_types': unique_checkpoint_types,
        'checkpoint_pack_types': list(shortlist_checkpoint_types),
        'checkpoint_dedupe': checkpoint_dedupe,
        'pending_canonical_alias_targets': formal.resolve_pending_canonical_alias_targets(state),
        'publish_canonical_aliases': True,
        'candidates': candidates,
        'candidate_index': {candidate.arm_name: candidate for candidate in candidates},
        'candidate_order': {
            candidate.arm_name: index
            for index, candidate in enumerate(candidates)
        },
    }


def materialize_manual_formal_1v3_state(
    *,
    run_dir: Path,
    manifest_path: Path,
) -> Path:
    manifest = fidelity.load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f'manual formal_1v3 manifest must be a JSON object: {manifest_path}')
    manual_config = dict(manifest)
    mode = str(manual_config.get('mode') or MANUAL_FORMAL_1V3_MODE).strip()
    if mode != MANUAL_FORMAL_1V3_MODE:
        raise RuntimeError(f'unsupported manual formal_1v3 mode `{mode}` in {manifest_path}')
    manual_config['mode'] = mode
    manual_config['candidates'] = normalize_manual_shortlist_candidates(manual_config.get('candidates'))
    load_frozen_formal_config_snapshot(manual_config)
    protocol_arm = str(manual_config.get('protocol_arm') or '').strip()
    if not protocol_arm:
        raise RuntimeError(f'manual formal_1v3 manifest is missing protocol arm: {manifest_path}')
    manual_config['protocol_arm'] = protocol_arm
    manual_config['current_primary_protocol_arm'] = (
        str(manual_config.get('current_primary_protocol_arm') or '').strip() or protocol_arm
    )
    manual_config['publish_canonical_aliases'] = bool(manual_config.get('publish_canonical_aliases', False))
    if not manual_config['publish_canonical_aliases']:
        manual_config['pending_canonical_alias_targets'] = []
    shortlist_checkpoint_types = list(
        manual_config.get('shortlist_checkpoint_types')
        or manual_config['candidates'].keys()
    )
    manual_config['shortlist_checkpoint_types'] = shortlist_checkpoint_types
    manual_config['checkpoint_pack_types'] = list(
        manual_config.get('checkpoint_pack_types')
        or shortlist_checkpoint_types
    )

    state_path = run_dir / 'state.json'
    state = fidelity.load_json(state_path) if state_path.exists() else {}
    state[MANUAL_FORMAL_1V3_CONFIG_KEY] = fidelity.normalize_payload(manual_config)
    state['manual_formal_1v3_manifest'] = str(manifest_path.resolve())
    state.setdefault('final_conclusion', {})
    state.setdefault('status', 'pending_manual_formal_1v3')
    fidelity.atomic_write_json(state_path, state)
    return state_path


def resolve_frozen_config_dir(context: dict[str, Any]) -> Path | None:
    snapshot = context.get('config_snapshot')
    if not isinstance(snapshot, dict):
        return None
    config_dir = snapshot.get('config_dir')
    if not config_dir:
        return None
    return Path(str(config_dir))


def validate_published_canonical_checkpoints(
    *,
    expected_targets: list[str | Path],
    published: list[dict[str, str]] | None,
) -> None:
    normalized_expected_targets = {
        Path(str(target)).resolve()
        for target in expected_targets
        if str(target).strip()
    }
    if not normalized_expected_targets:
        return
    published_destinations = {
        Path(str(entry.get('destination') or '')).resolve()
        for entry in (published or [])
        if isinstance(entry, dict) and str(entry.get('destination') or '').strip()
    }
    if not published_destinations:
        raise RuntimeError(
            'formal_1v3 finished ranking but did not publish any expected supervised canonical aliases'
        )
    missing_targets = sorted(
        str(target)
        for target in normalized_expected_targets
        if target not in published_destinations
    )
    if missing_targets:
        raise RuntimeError(
            'formal_1v3 did not publish all expected supervised canonical aliases: '
            + ', '.join(missing_targets)
        )


def build_task_id(
    *,
    run_name: str | None = None,
    stage_name: str,
    round_label: str,
    arm_name: str,
    worker_label: str | None = None,
) -> str:
    parts = [stage_name, round_label, arm_name]
    if worker_label:
        parts.append(worker_label)
    if run_name:
        parts.append(run_name)
    return '__'.join(parts)


def filesystem_safe_token(value: str) -> str:
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(value))
    normalized = normalized.rstrip(' .')
    if not normalized:
        normalized = 'task'
    if normalized == value:
        return normalized
    suffix = hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:10]
    return f'{normalized}__{suffix}'


def task_artifact_stem(task_state: dict[str, Any]) -> str:
    cached = str(task_state.get('artifact_stem') or '').strip()
    if cached:
        return cached
    stem = filesystem_safe_token(str(task_state['task_id']))
    task_state['artifact_stem'] = stem
    return stem


def resolve_run_scoped_path(
    run_dir: Path,
    raw_path: str | os.PathLike[str] | None,
    *,
    fallback_name: str | None = None,
) -> Path | None:
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else run_dir / path
    if fallback_name:
        return run_dir / fallback_name
    return None


def load_completed_dispatch_final_payload(
    *,
    run_dir: Path,
    dispatch_state: dict[str, Any],
) -> dict[str, Any]:
    final_summary_path = resolve_run_scoped_path(
        run_dir,
        dispatch_state.get('final_round_summary_path'),
        fallback_name='formal_1v3_round.json',
    )
    if final_summary_path is not None and final_summary_path.exists():
        payload = fidelity.load_json(final_summary_path)
        if isinstance(payload, dict):
            return payload
    state_path = run_dir / 'state.json'
    if state_path.exists():
        state = fidelity.load_json(state_path)
        formal_1v3_state = state.get('formal_1v3')
        if isinstance(formal_1v3_state, dict):
            result = formal_1v3_state.get('result')
            if isinstance(result, dict):
                return result
    raise RuntimeError(
        'formal_1v3 dispatch is marked completed but final summary payload is missing; repair this run before resuming'
    )


def encode_frozen_cfg_payload(cfg: dict[str, Any]) -> str:
    encoded = json.dumps(
        fidelity.normalize_payload(cfg),
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return base64.urlsafe_b64encode(encoded).decode('ascii')


def decode_frozen_cfg_payload(payload_b64: str | None) -> dict[str, Any] | None:
    if not payload_b64:
        return None
    try:
        decoded = base64.urlsafe_b64decode(payload_b64.encode('ascii')).decode('utf-8')
        payload = json.loads(decoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError('invalid frozen 1v3 config payload') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('frozen 1v3 config payload must decode to an object')
    return payload


def format_completed_output(completed: subprocess.CompletedProcess[str]) -> str:
    parts = []
    for value in (completed.stdout, completed.stderr):
        text = str(value or '').strip()
        if text:
            parts.append(text)
    return '\n'.join(parts)


def build_worker_seed_schedule(
    worker_budgets: dict[str, dict[str, Any]],
    *,
    worker_labels: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    ordered_labels: list[str] = []
    seen_labels: set[str] = set()
    for label in worker_labels or []:
        normalized = str(label or '').strip()
        if not normalized or normalized in seen_labels:
            continue
        if normalized not in worker_budgets:
            raise RuntimeError(f'unknown worker label in seed schedule: {normalized}')
        ordered_labels.append(normalized)
        seen_labels.add(normalized)
    for label in worker_budgets:
        normalized = str(label or '').strip()
        if not normalized or normalized in seen_labels:
            continue
        ordered_labels.append(normalized)
        seen_labels.add(normalized)
    if not ordered_labels:
        return {}

    seed_counts: dict[str, int] = {}
    total_seed_count = 0
    for label in ordered_labels:
        worker_budget = worker_budgets.get(label)
        if not isinstance(worker_budget, dict):
            raise RuntimeError(f'missing frozen worker budget for `{label}`')
        seed_count = int(worker_budget.get('seed_count_per_iter') or 0)
        if seed_count <= 0:
            raise RuntimeError(f'invalid frozen seed_count_per_iter for `{label}`: {seed_count}')
        seed_counts[label] = seed_count
        total_seed_count += seed_count

    offset = 0
    schedule: dict[str, dict[str, int]] = {}
    for label in ordered_labels:
        schedule[label] = {
            'seed_start_offset': offset,
            'seed_stride_per_iter': total_seed_count,
        }
        offset += seed_counts[label]
    return schedule


def apply_stage_task_seed_schedule(
    stage_state: dict[str, Any],
    *,
    worker_budgets: dict[str, dict[str, Any]],
) -> None:
    worker_labels = [
        str(label)
        for label in (stage_state.get('worker_labels') or [])
        if str(label or '').strip()
    ]
    seed_schedule = build_worker_seed_schedule(
        worker_budgets,
        worker_labels=worker_labels if worker_labels else None,
    )
    if not worker_labels:
        stage_state['worker_labels'] = list(seed_schedule)
    stage_state['seed_stride_per_iter'] = (
        int(next(iter(seed_schedule.values()))['seed_stride_per_iter'])
        if seed_schedule
        else 0
    )
    for task in stage_state.get('tasks', {}).values():
        assigned_worker_label = str(task.get('assigned_worker_label') or '').strip()
        schedule = seed_schedule.get(assigned_worker_label)
        if schedule is None:
            raise RuntimeError(
                'formal_1v3 dispatch task is missing a valid worker seed schedule; restart this dispatch'
            )
        task['seed_start_offset'] = int(schedule['seed_start_offset'])
        task['seed_stride_per_iter'] = int(schedule['seed_stride_per_iter'])


def resolve_worker_budget_snapshot(
    *,
    machine_label: str,
    frozen_1v3_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = deepcopy(frozen_1v3_cfg)
    seed_count_per_iter, seed_count_source = one_vs_three.resolve_seed_count(cfg)
    shard_count, shard_count_source = one_vs_three.resolve_shard_count(cfg)
    shard_seed_counts = one_vs_three.plan_shards(seed_count_per_iter, shard_count)
    return {
        'machine_label': machine_label,
        'seed_count_per_iter': int(seed_count_per_iter),
        'games_per_iter': int(seed_count_per_iter * 4),
        'shard_count': int(shard_count),
        'shard_seed_counts': [int(value) for value in shard_seed_counts],
        'seed_count_source': seed_count_source,
        'shard_count_source': shard_count_source,
        'computer_name': str(os.environ.get('COMPUTERNAME', '') or ''),
        'gpu_name': one_vs_three.resolve_gpu_name(cfg),
    }


def build_remote_budget_command(worker: WorkerSpec, *, frozen_1v3_cfg: dict[str, Any]) -> list[str]:
    repo_root = Path(worker.repo or str(REPO_ROOT))
    remote_script = repo_root / 'mortal' / SCRIPT_PATH.name
    encoded_cfg = encode_frozen_cfg_payload(frozen_1v3_cfg)
    ps_command = (
        f"Set-Location {quote_ps(str(repo_root))}; "
        f"& {quote_ps(worker.python or sys.executable)} "
        f"{quote_ps(str(remote_script))} "
        f"resolve-budget --machine-label {quote_ps(worker.label)} "
        f"--cfg-json-b64 {quote_ps(encoded_cfg)}"
    )
    command = ['ssh']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.append(worker.host or common_dispatch.DEFAULT_REMOTE_HOST)
    command.append(ps_command)
    return command


def query_remote_worker_budget(worker: WorkerSpec, *, frozen_1v3_cfg: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        build_remote_budget_command(worker, frozen_1v3_cfg=frozen_1v3_cfg),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    completed_output = format_completed_output(completed)
    if completed.returncode != 0:
        raise RuntimeError(
            f'failed to resolve remote worker budget for `{worker.label}`: {completed_output}'
        )
    stdout_text = str(completed.stdout or '').strip()
    if not stdout_text:
        raise RuntimeError(
            f'invalid remote worker budget payload from `{worker.label}`: {completed_output or "<empty>"}'
        )
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'invalid remote worker budget payload from `{worker.label}`: {stdout_text}'
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f'remote worker budget payload for `{worker.label}` is not an object')
    return payload


def collect_worker_budgets(
    workers: list[WorkerSpec],
    *,
    frozen_1v3_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    budgets: dict[str, dict[str, Any]] = {}
    for worker in workers:
        if worker.kind == 'local':
            payload = resolve_worker_budget_snapshot(
                machine_label=worker.label,
                frozen_1v3_cfg=frozen_1v3_cfg,
            )
        elif worker.kind == 'remote':
            payload = query_remote_worker_budget(
                worker,
                frozen_1v3_cfg=frozen_1v3_cfg,
            )
        else:
            raise ValueError(f'unsupported worker kind `{worker.kind}`')
        normalized = dict(payload)
        normalized['machine_label'] = worker.label
        budgets[worker.label] = normalized
    return budgets


def build_remote_formal_state_payload(
    *,
    state: dict[str, Any],
    remote_checkpoint_paths: dict[str, Path],
    remote_champion_state_file: Path | None,
) -> dict[str, Any]:
    remote_state = deepcopy(state)
    manual_config = remote_state.get(MANUAL_FORMAL_1V3_CONFIG_KEY)
    if isinstance(manual_config, dict):
        candidates = manual_config.get('candidates')
        if not isinstance(candidates, dict):
            raise RuntimeError('run state manual formal_1v3 candidates are missing')
        snapshot = manual_config.get('config_snapshot')
    else:
        formal_state = remote_state.get('formal')
        if not isinstance(formal_state, dict):
            raise RuntimeError('run state is missing formal section')
        result = formal_state.get('result')
        if not isinstance(result, dict):
            raise RuntimeError('run state formal result is missing')
        candidates = result.get('candidates')
        if not isinstance(candidates, dict):
            raise RuntimeError('run state formal result candidates are missing')
        snapshot = result.get('config_snapshot')
    for checkpoint_type, remote_path in remote_checkpoint_paths.items():
        candidate_payload = candidates.get(checkpoint_type)
        if not isinstance(candidate_payload, dict):
            raise RuntimeError(f'formal candidate `{checkpoint_type}` is missing from state')
        candidate_payload['path'] = str(remote_path)
    if isinstance(snapshot, dict):
        base_1v3_cfg = snapshot.get('base_1v3_cfg')
        if isinstance(base_1v3_cfg, dict):
            champion_cfg = base_1v3_cfg.get('champion')
            if isinstance(champion_cfg, dict) and remote_champion_state_file is not None:
                champion_cfg['state_file'] = str(remote_champion_state_file)
    return remote_state


def publish_canonical_checkpoints_for_context(
    *,
    context: dict[str, Any],
    winner: str,
) -> list[dict[str, str]]:
    if not bool(context.get('publish_canonical_aliases', True)):
        return []
    return formal.publish_stage05_canonical_checkpoints(
        context['base_cfg'],
        {'winner': winner, 'candidates': dict(context['shortlist_candidates'])},
        config_dir=resolve_frozen_config_dir(context),
        protocol_arm=context['protocol_arm'],
        primary_protocol_arm=context.get('current_primary_protocol_arm'),
    )


def sync_remote_formal_shortlist_assets(
    *,
    worker: WorkerSpec,
    context: dict[str, Any],
) -> dict[str, Any]:
    remote_repo = Path(worker.repo or str(REPO_ROOT))
    remote_checkpoint_paths: dict[str, Path] = {}
    for checkpoint_type, candidate_payload in context['shortlist_candidates'].items():
        local_checkpoint_path = Path(str(candidate_payload.get('path', '') or '')).resolve()
        if not local_checkpoint_path.exists():
            raise FileNotFoundError(
                f'formal shortlist checkpoint for `{checkpoint_type}` does not exist: {local_checkpoint_path}'
            )
        remote_checkpoint_path = map_repo_path_to_remote(local_checkpoint_path, remote_repo=remote_repo)
        copy_local_file_to_remote(
            worker,
            local_path=local_checkpoint_path,
            remote_path=remote_checkpoint_path,
        )
        remote_checkpoint_paths[checkpoint_type] = remote_checkpoint_path
    remote_champion_state_file: Path | None = None
    champion_cfg = context['base_1v3_cfg'].get('champion', {})
    if isinstance(champion_cfg, dict):
        champion_state_file = str(champion_cfg.get('state_file', '') or '').strip()
        if champion_state_file:
            local_champion_state_file = Path(champion_state_file).resolve()
            if not local_champion_state_file.exists():
                raise FileNotFoundError(
                    'frozen 1v3 champion checkpoint does not exist: '
                    f'{local_champion_state_file}'
                )
            try:
                remote_champion_state_file = map_repo_path_to_remote(
                    local_champion_state_file,
                    remote_repo=remote_repo,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    'frozen 1v3 champion checkpoint is outside the repo root and '
                    f'cannot be synced to remote worker `{worker.label}`: {local_champion_state_file}'
                ) from exc
            copy_local_file_to_remote(
                worker,
                local_path=local_champion_state_file,
                remote_path=remote_champion_state_file,
            )
    remote_state = build_remote_formal_state_payload(
        state=context['state'],
        remote_checkpoint_paths=remote_checkpoint_paths,
        remote_champion_state_file=remote_champion_state_file,
    )
    remote_state_path = map_repo_path_to_remote(context['state_path'], remote_repo=remote_repo)
    with tempfile.TemporaryDirectory(prefix='mahjongai_formal_1v3_remote_state_') as tmp_dir:
        temp_state_path = Path(tmp_dir) / 'state.json'
        temp_state_path.write_text(
            json.dumps(fidelity.normalize_payload(remote_state), ensure_ascii=False, indent=2),
            encoding='utf-8',
            newline='\n',
        )
        copy_local_file_to_remote(
            worker,
            local_path=temp_state_path,
            remote_path=remote_state_path,
        )
    return {
        'state_path': str(remote_state_path),
        'checkpoints': {key: str(value) for key, value in remote_checkpoint_paths.items()},
        'config_assets': (
            {'champion_state_file': str(remote_champion_state_file)}
            if remote_champion_state_file is not None
            else {}
        ),
        'synced_at': fidelity.ts_now(),
    }


def build_round_spec(*, round_index: int, round_label: str, iters: int) -> dict[str, Any]:
    return {
        'round_index': int(round_index),
        'round_label': str(round_label),
        'iters': int(iters),
        'seed_key': int(secrets.randbits(64)),
    }


def build_stage_state(
    *,
    run_name: str | None = None,
    stage_name: str,
    candidates: list[fidelity.CandidateSpec],
    round_specs: list[dict[str, Any]],
    worker_budgets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    ordered_worker_budgets = [
        (worker_label, dict(worker_budget))
        for worker_label, worker_budget in worker_budgets.items()
    ]
    worker_labels = [worker_label for worker_label, _ in ordered_worker_budgets]
    worker_seed_schedule = build_worker_seed_schedule(
        worker_budgets,
        worker_labels=worker_labels,
    )
    for round_spec in round_specs:
        for candidate in candidates:
            for worker_label, worker_budget in ordered_worker_budgets:
                seed_schedule = worker_seed_schedule[worker_label]
                task_id = build_task_id(
                    run_name=run_name,
                    stage_name=stage_name,
                    round_label=str(round_spec['round_label']),
                    arm_name=candidate.arm_name,
                    worker_label=worker_label,
                )
                tasks[task_id] = {
                    'task_id': task_id,
                    'candidate_arm': candidate.arm_name,
                    'assigned_worker_label': worker_label,
                    'round_index': int(round_spec['round_index']),
                    'round_label': str(round_spec['round_label']),
                    'seed_key': int(round_spec['seed_key']),
                    'iters': int(round_spec['iters']),
                    'seed_count_per_iter': int(worker_budget['seed_count_per_iter']),
                    'shard_count': int(worker_budget['shard_count']),
                    'shard_seed_counts': [int(value) for value in worker_budget['shard_seed_counts']],
                    'seed_count_source': str(worker_budget['seed_count_source']),
                    'shard_count_source': str(worker_budget['shard_count_source']),
                    'seed_start_offset': int(seed_schedule['seed_start_offset']),
                    'seed_stride_per_iter': int(seed_schedule['seed_stride_per_iter']),
                    'status': 'pending',
                    'attempts': 0,
                }
    return {
        'stage_name': stage_name,
        'round_specs': [dict(spec) for spec in round_specs],
        'candidate_count': len(candidates),
        'worker_count': len(ordered_worker_budgets),
        'worker_labels': worker_labels,
        'seed_stride_per_iter': (
            int(next(iter(worker_seed_schedule.values()))['seed_stride_per_iter'])
            if worker_seed_schedule
            else 0
        ),
        'task_count': len(tasks),
        'tasks': tasks,
    }


def initialize_dispatch_state(
    *,
    context: dict[str, Any],
    local_label: str,
    remote_label: str | None,
    worker_budgets: dict[str, dict[str, Any]],
    coarse_iters: int,
    extra_iters: int,
    close_stderr_mult: float,
    close_extra_rounds: int,
    close_max_extra_rounds: int,
) -> dict[str, Any]:
    return {
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': ROUND_KIND_FORMAL_1V3,
        'run_name': context['run_name'],
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'status': 'running',
        'stage': 'seed1',
        'local_label': local_label,
        'remote_label': remote_label,
        'protocol_arm': context['protocol_arm'],
        'config_snapshot': deepcopy(context['config_snapshot']),
        'worker_budgets': {
            worker_label: dict(worker_budget)
            for worker_label, worker_budget in worker_budgets.items()
        },
        'shortlist_checkpoint_types': list(context['shortlist_checkpoint_types']),
        'unique_shortlist_checkpoint_types': list(
            context.get('unique_shortlist_checkpoint_types') or context['shortlist_checkpoint_types']
        ),
        'checkpoint_pack_types': list(context.get('checkpoint_pack_types') or context['shortlist_checkpoint_types']),
        'checkpoint_dedupe': deepcopy(context.get('checkpoint_dedupe') or {}),
        'close_stderr_mult': float(close_stderr_mult),
        'close_extra_rounds': int(close_extra_rounds),
        'close_max_extra_rounds': int(close_max_extra_rounds),
        'extra_round_iters': int(extra_iters),
        'candidate_payloads': [
            fidelity.candidate_cache_payload(candidate, include_meta=True)
            for candidate in context['candidates']
        ],
        'seed1': build_stage_state(
            run_name=context['run_name'],
            stage_name='seed1',
            candidates=context['candidates'],
            round_specs=[build_round_spec(round_index=0, round_label='coarse', iters=coarse_iters)],
            worker_budgets=worker_budgets,
        ),
        'seed2': None,
        'coarse_round_summary_path': None,
        'final_round_summary_path': None,
        'latest_close_call': None,
        'final_close_call': None,
        'final_winner': None,
        'published_canonical_checkpoints': [],
    }


def update_run_state_for_dispatch(
    *,
    run_dir: Path,
    dispatch_state_path: Path,
    dispatch_state: dict[str, Any],
    final_payload: dict[str, Any] | None = None,
    published: list[dict[str, str]] | None = None,
    status_override: str | None = None,
) -> None:
    state_path = run_dir / 'state.json'
    state = fidelity.load_json(state_path)
    final_conclusion = state.setdefault('final_conclusion', {})
    formal_1v3_state = state.setdefault('formal_1v3', {})
    formal_train_status = state.get('formal', {}).get('status', 'pending')
    formal_1v3_state.update(
        {
            'mode': 'desktop_dispatch_plus_optional_ssh_remote',
            'dispatch_state_path': str(dispatch_state_path),
            'stage': dispatch_state.get('stage'),
            'status': dispatch_state.get('status'),
            'coarse_round_summary_path': dispatch_state.get('coarse_round_summary_path'),
            'final_round_summary_path': dispatch_state.get('final_round_summary_path'),
            'shortlist_checkpoint_types': list(dispatch_state.get('shortlist_checkpoint_types') or []),
            'checkpoint_pack_types': list(
                dispatch_state.get('checkpoint_pack_types')
                or dispatch_state.get('shortlist_checkpoint_types')
                or []
            ),
            'close_call': dispatch_state.get('latest_close_call') or dispatch_state.get('final_close_call'),
            'winner': dispatch_state.get('final_winner'),
        }
    )
    final_conclusion['formal_train_status'] = formal_train_status
    if final_payload is not None:
        formal_1v3_state['result'] = final_payload
        formal_1v3_state['published_canonical_checkpoints'] = list(published or [])
        final_conclusion['formal_1v3_status'] = 'completed'
        final_conclusion['formal_status'] = 'completed'
        final_conclusion['formal_winner'] = dispatch_state.get('final_winner')
        state['status'] = status_override or 'completed'
    else:
        status = str(dispatch_state.get('status') or 'pending')
        if status == 'failed':
            final_conclusion['formal_1v3_status'] = 'failed'
            final_conclusion['formal_status'] = 'pending_1v3'
            state['status'] = status_override or 'failed'
        else:
            final_conclusion['formal_1v3_status'] = status
            final_conclusion['formal_status'] = 'pending_1v3'
            state['status'] = status_override or 'running_formal_1v3'
    fidelity.atomic_write_json(state_path, state)
    fidelity.update_results_doc(run_dir, state)


def reconcile_completed_dispatch_run_state(
    *,
    run_dir: Path,
    dispatch_state_path: Path,
    dispatch_state: dict[str, Any],
) -> None:
    state = fidelity.load_json(run_dir / 'state.json')
    validate_published_canonical_checkpoints(
        expected_targets=formal.resolve_pending_canonical_alias_targets(state),
        published=list(dispatch_state.get('published_canonical_checkpoints') or []),
    )
    final_payload = load_completed_dispatch_final_payload(
        run_dir=run_dir,
        dispatch_state=dispatch_state,
    )
    update_run_state_for_dispatch(
        run_dir=run_dir,
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        final_payload=final_payload,
        published=list(dispatch_state.get('published_canonical_checkpoints') or []),
        status_override='completed',
    )


@contextmanager
def temporary_environ(overrides: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def ranking_stats(rankings: np.ndarray) -> dict[str, Any]:
    games = int(rankings.sum())
    if games <= 0:
        raise RuntimeError('formal_1v3 evaluation produced zero games')
    avg_rank = float(rankings @ RANK_VALUES / games)
    avg_pt = float(rankings @ POINT_VALUES / games)
    if games <= 1:
        rank_stderr = 0.0
        pt_stderr = 0.0
    else:
        rank_var = float(rankings @ ((RANK_VALUES - avg_rank) ** 2) / (games - 1))
        pt_var = float(rankings @ ((POINT_VALUES - avg_pt) ** 2) / (games - 1))
        rank_stderr = math.sqrt(max(rank_var, 0.0) / games)
        pt_stderr = math.sqrt(max(pt_var, 0.0) / games)
    return {
        'games': games,
        'avg_rank': avg_rank,
        'avg_pt': avg_pt,
        'rank_stderr': rank_stderr,
        'pt_stderr': pt_stderr,
    }


def evaluate_formal_candidate(
    *,
    context: dict[str, Any],
    candidate: fidelity.CandidateSpec,
    round_index: int,
    round_label: str,
    seed_key: int,
    iters: int,
    machine_label: str,
    frozen_seed_count_per_iter: int | None = None,
    frozen_shard_count: int | None = None,
    seed_start_offset: int | None = None,
    seed_stride_per_iter: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(str(candidate.meta['checkpoint_path'])).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'formal checkpoint-pack checkpoint does not exist: {checkpoint_path}')
    cfg = deepcopy(context['base_1v3_cfg'])
    candidate_runtime_name = filesystem_safe_token(str(candidate.arm_name))
    cfg['challenger'] = deepcopy(cfg['challenger'])
    cfg['challenger']['state_file'] = str(checkpoint_path)
    cfg['challenger']['name'] = candidate_runtime_name
    cfg['execution_mode'] = 'process'
    cfg['enable_metadata'] = False
    cfg['log_dir'] = str(
        context['run_dir']
        / 'distributed'
        / 'formal_1v3_dispatch'
        / 'eval_logs'
        / f'{filesystem_safe_token(str(round_label))}__{candidate_runtime_name}__{filesystem_safe_token(str(machine_label))}'
    )
    with tempfile.TemporaryDirectory(prefix='mahjongai_formal_1v3_cfg_') as tmp_dir:
        cfg_path = Path(tmp_dir) / 'config.toml'
        write_toml_file(cfg_path, {'1v3': cfg})
        total_rankings = np.zeros(4, dtype=np.int64)
        with temporary_environ({'MORTAL_CFG': str(cfg_path)}):
            if frozen_seed_count_per_iter is None:
                seeds_per_iter, seed_count_source = one_vs_three.resolve_seed_count(cfg)
            else:
                seeds_per_iter = int(frozen_seed_count_per_iter)
                if seeds_per_iter <= 0:
                    raise ValueError(f'frozen seed_count_per_iter must be positive, got {seeds_per_iter}')
                seed_count_source = f'dispatch_frozen seed_count={seeds_per_iter}'
            if frozen_shard_count is None:
                shard_count, shard_count_source = one_vs_three.resolve_shard_count(cfg)
            else:
                shard_count = int(frozen_shard_count)
                if shard_count <= 0:
                    raise ValueError(f'frozen shard_count must be positive, got {shard_count}')
                shard_count_source = f'dispatch_frozen shard_count={shard_count}'
            shard_seed_counts = one_vs_three.plan_shards(seeds_per_iter, shard_count)
            effective_seed_start_offset = int(seed_start_offset or 0)
            if effective_seed_start_offset < 0:
                raise ValueError(
                    f'seed_start_offset must be non-negative, got {effective_seed_start_offset}'
                )
            effective_seed_stride_per_iter = (
                int(seed_stride_per_iter)
                if seed_stride_per_iter is not None
                else int(seeds_per_iter)
            )
            if effective_seed_stride_per_iter <= 0:
                raise ValueError(
                    'seed_stride_per_iter must be positive, '
                    f'got {effective_seed_stride_per_iter}'
                )
            if effective_seed_stride_per_iter < seeds_per_iter:
                raise ValueError(
                    'seed_stride_per_iter must be at least seed_count_per_iter, '
                    f'got stride={effective_seed_stride_per_iter} seed_count={seeds_per_iter}'
                )
            disable_progress_bar = bool(cfg.get('disable_progress_bar', False))
            log_dir = cfg.get('log_dir') or None
            single_shard_eval_context = None
            shard_worker_pool = None
            if len(shard_seed_counts) == 1:
                single_shard_eval_context = one_vs_three.load_eval_engines(cfg)
            else:
                shard_worker_pool = one_vs_three.start_persistent_shard_workers(
                    cfg=cfg,
                    shard_count=len(shard_seed_counts),
                    disable_progress_bar=disable_progress_bar,
                    execution_mode='process',
                )
            try:
                for iter_offset in range(int(iters)):
                    seed_start = (
                        FORMAL_1V3_SEED_START_BASE
                        + effective_seed_start_offset
                        + iter_offset * effective_seed_stride_per_iter
                    )
                    if single_shard_eval_context is not None:
                        rankings = one_vs_three.run_eval_once(
                            cfg=cfg,
                            seed_start=seed_start,
                            seed_key=seed_key,
                            seed_count=seeds_per_iter,
                            log_dir=log_dir,
                            disable_progress_bar=disable_progress_bar,
                            eval_context=single_shard_eval_context,
                        )
                        rankings = np.array(rankings, dtype=np.int64)
                    else:
                        rankings, _, _, _ = one_vs_three.run_sharded_iteration_with_workers(
                            iter_index=iter_offset,
                            seed_start=seed_start,
                            seed_key=seed_key,
                            shard_seed_counts=shard_seed_counts,
                            log_dir=log_dir,
                            shard_worker_pool=shard_worker_pool,
                        )
                        rankings = np.array(rankings, dtype=np.int64)
                    total_rankings += rankings
            finally:
                one_vs_three.stop_persistent_shard_workers(shard_worker_pool)
    stats = ranking_stats(total_rankings)
    return {
        'checkpoint_type': candidate.arm_name,
        'checkpoint_path': str(checkpoint_path),
        'round_index': int(round_index),
        'round_label': str(round_label),
        'seed_key': int(seed_key),
        'iters': int(iters),
        'machine_label': machine_label,
        'seed_count_per_iter': int(seeds_per_iter),
        'seed_start_offset': int(effective_seed_start_offset),
        'seed_stride_per_iter': int(effective_seed_stride_per_iter),
        'games_per_iter': int(seeds_per_iter * 4),
        'shard_count': len(shard_seed_counts),
        'seed_count_source': seed_count_source,
        'shard_count_source': shard_count_source,
        'rankings': total_rankings.tolist(),
        **stats,
    }


def execute_single_task(
    *,
    run_name: str,
    candidate_arm: str,
    round_index: int,
    round_label: str,
    seed_key: int,
    iters: int,
    result_json: Path,
    machine_label: str,
    seed_count_per_iter: int | None = None,
    shard_count: int | None = None,
    seed_start_offset: int | None = None,
    seed_stride_per_iter: int | None = None,
) -> dict[str, Any]:
    run_dir = fidelity.FIDELITY_ROOT / run_name
    context = load_formal_context(run_dir)
    candidate = context['candidate_index'].get(candidate_arm)
    if candidate is None:
        raise KeyError(f'unknown formal checkpoint-pack candidate `{candidate_arm}`')
    raw_payload = evaluate_formal_candidate(
        context=context,
        candidate=candidate,
        round_index=round_index,
        round_label=round_label,
        seed_key=seed_key,
        iters=iters,
        machine_label=machine_label,
        frozen_seed_count_per_iter=seed_count_per_iter,
        frozen_shard_count=shard_count,
        seed_start_offset=seed_start_offset,
        seed_stride_per_iter=seed_stride_per_iter,
    )
    summary = {
        'ok': True,
        'valid': True,
        'checkpoint_type': raw_payload['checkpoint_type'],
        'checkpoint_path': raw_payload['checkpoint_path'],
        'avg_pt': raw_payload['avg_pt'],
        'avg_rank': raw_payload['avg_rank'],
        'pt_stderr': raw_payload['pt_stderr'],
        'rank_stderr': raw_payload['rank_stderr'],
        'games': raw_payload['games'],
        'round_index': raw_payload['round_index'],
        'round_label': raw_payload['round_label'],
        'seed_key': raw_payload['seed_key'],
        'iters': raw_payload['iters'],
        'seed_start_offset': raw_payload['seed_start_offset'],
        'seed_stride_per_iter': raw_payload['seed_stride_per_iter'],
    }
    payload = {
        'schema_version': TASK_RESULT_SCHEMA_VERSION,
        'round_kind': ROUND_KIND_FORMAL_1V3,
        'run_name': run_name,
        'candidate_arm': candidate.arm_name,
        'round_index': int(round_index),
        'round_label': str(round_label),
        'seed_key': int(seed_key),
        'iters': int(iters),
        'seed_start_offset': int(raw_payload['seed_start_offset']),
        'seed_stride_per_iter': int(raw_payload['seed_stride_per_iter']),
        'machine_label': machine_label,
        'completed_at': fidelity.ts_now(),
        'payload': raw_payload,
        'summary': summary,
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    fidelity.atomic_write_json(result_json, payload)
    return payload


def build_run_task_cli_summary(payload: dict[str, Any], *, result_json: Path) -> dict[str, Any]:
    summary = payload.get('summary') or {}
    return {
        'status': 'ok',
        'round_kind': payload.get('round_kind'),
        'run_name': payload.get('run_name'),
        'candidate_arm': payload.get('candidate_arm'),
        'round_index': payload.get('round_index'),
        'round_label': payload.get('round_label'),
        'seed_key': payload.get('seed_key'),
        'iters': payload.get('iters'),
        'seed_start_offset': payload.get('seed_start_offset'),
        'seed_stride_per_iter': payload.get('seed_stride_per_iter'),
        'machine_label': payload.get('machine_label'),
        'completed_at': payload.get('completed_at'),
        'result_json': str(result_json),
        'valid': summary.get('valid'),
        'games': summary.get('games'),
        'avg_pt': summary.get('avg_pt'),
        'avg_rank': summary.get('avg_rank'),
    }


def load_task_result(path: Path) -> dict[str, Any]:
    payload = fidelity.load_json(path)
    summary = payload.get('summary')
    if not isinstance(summary, dict):
        raise RuntimeError(f'task result at {path} is missing summary')
    if not bool(summary.get('ok')):
        raise RuntimeError(
            f'task result at {path} reported 1v3 failure: {summary.get("error") or "unknown error"}'
        )
    if not bool(summary.get('valid')):
        raise RuntimeError(f'task result at {path} is not valid for formal_1v3 ranking')
    return payload


def stage_results_from_state(stage_state: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in stage_state.get('tasks', {}).values():
        if str(task.get('status')) != 'completed':
            continue
        result_path = task.get('local_result_path')
        if not result_path:
            continue
        results.append(load_task_result(Path(result_path)))
    return results


def aggregate_task_payloads(
    payloads: list[dict[str, Any]],
    *,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        candidate_arm = str(payload.get('candidate_arm', '') or '').strip()
        if not candidate_arm:
            continue
        raw = payload['payload']
        entry = by_candidate.setdefault(
            candidate_arm,
            {
                'checkpoint_type': candidate_arm,
                'checkpoint_path': str(raw.get('checkpoint_path', '') or ''),
                'rankings': np.zeros(4, dtype=np.int64),
                'machine_labels': set(),
                'round_labels': [],
                'seed_keys': [],
            },
        )
        entry['rankings'] += np.array(raw.get('rankings', [0, 0, 0, 0]), dtype=np.int64)
        entry['machine_labels'].add(str(payload.get('machine_label', '') or ''))
        entry['round_labels'].append(str(payload.get('round_label', '') or ''))
        entry['seed_keys'].append(int(payload.get('seed_key', 0) or 0))
    ranking: list[dict[str, Any]] = []
    for candidate in context['candidates']:
        aggregate = by_candidate.get(candidate.arm_name)
        if aggregate is None:
            continue
        stats = ranking_stats(aggregate['rankings'])
        ranking.append(
            {
                'checkpoint_type': candidate.arm_name,
                'checkpoint_path': aggregate['checkpoint_path'],
                'games': stats['games'],
                'rankings': aggregate['rankings'].tolist(),
                'avg_pt': stats['avg_pt'],
                'avg_rank': stats['avg_rank'],
                'pt_stderr': stats['pt_stderr'],
                'rank_stderr': stats['rank_stderr'],
                'machine_labels': sorted(label for label in aggregate['machine_labels'] if label),
                'round_labels': list(aggregate['round_labels']),
                'seed_keys': list(aggregate['seed_keys']),
                'valid': True,
            }
        )
    ranking.sort(
        key=lambda item: (
            -float(item['avg_pt']),
            float(item['avg_rank']),
            int(context['candidate_order'].get(item['checkpoint_type'], 999)),
        )
    )
    return ranking


def close_call_from_ranking(
    ranking: list[dict[str, Any]],
    *,
    stderr_mult: float,
) -> dict[str, Any]:
    if len(ranking) < 2:
        return {'triggered': False, 'reason': 'fewer_than_two_candidates'}
    leader = ranking[0]
    runner_up = ranking[1]
    avg_pt_gap = float(leader['avg_pt']) - float(runner_up['avg_pt'])
    avg_rank_gap = float(runner_up['avg_rank']) - float(leader['avg_rank'])
    combined_pt_stderr = math.sqrt(
        max(float(leader.get('pt_stderr', 0.0)), 0.0) ** 2
        + max(float(runner_up.get('pt_stderr', 0.0)), 0.0) ** 2
    )
    threshold = float(stderr_mult) * combined_pt_stderr
    return {
        'triggered': bool(avg_pt_gap <= threshold),
        'comparison_basis': 'avg_pt_primary_avg_rank_secondary',
        'leader': leader['checkpoint_type'],
        'runner_up': runner_up['checkpoint_type'],
        'leader_avg_pt': float(leader['avg_pt']),
        'runner_up_avg_pt': float(runner_up['avg_pt']),
        'avg_pt_gap': avg_pt_gap,
        'leader_avg_rank': float(leader['avg_rank']),
        'runner_up_avg_rank': float(runner_up['avg_rank']),
        'avg_rank_gap': avg_rank_gap,
        'combined_pt_stderr': combined_pt_stderr,
        'close_threshold': threshold,
        'stderr_mult': float(stderr_mult),
    }


def append_seed2_rounds(
    *,
    dispatch_state: dict[str, Any],
    candidates: list[fidelity.CandidateSpec],
    round_count: int,
) -> None:
    seed2_state = dispatch_state.get('seed2')
    if not isinstance(seed2_state, dict):
        raise RuntimeError('cannot append close rounds without a seed2 stage')
    existing_specs = list(seed2_state.get('round_specs') or [])
    start_index = (
        max(int(spec.get('round_index', 0)) for spec in existing_specs) + 1
        if existing_specs
        else 1
    )
    new_specs = [
        build_round_spec(
            round_index=start_index + offset,
            round_label=f'close_{start_index + offset:02d}',
            iters=int(dispatch_state['extra_round_iters']),
        )
        for offset in range(int(round_count))
    ]
    seed2_state['round_specs'] = existing_specs + new_specs
    worker_budgets = dict(dispatch_state.get('worker_budgets') or {})
    worker_seed_schedule = build_worker_seed_schedule(
        worker_budgets,
        worker_labels=[
            str(label)
            for label in (seed2_state.get('worker_labels') or [])
            if str(label or '').strip()
        ] or None,
    )
    for round_spec in new_specs:
        for candidate in candidates:
            for worker_label, worker_budget in worker_budgets.items():
                seed_schedule = worker_seed_schedule[worker_label]
                task_id = build_task_id(
                    run_name=str(dispatch_state.get('run_name') or ''),
                    stage_name='seed2',
                    round_label=str(round_spec['round_label']),
                    arm_name=candidate.arm_name,
                    worker_label=worker_label,
                )
                seed2_state['tasks'][task_id] = {
                    'task_id': task_id,
                    'candidate_arm': candidate.arm_name,
                    'assigned_worker_label': worker_label,
                    'round_index': int(round_spec['round_index']),
                    'round_label': str(round_spec['round_label']),
                    'seed_key': int(round_spec['seed_key']),
                    'iters': int(round_spec['iters']),
                    'seed_count_per_iter': int(worker_budget['seed_count_per_iter']),
                    'shard_count': int(worker_budget['shard_count']),
                    'shard_seed_counts': [int(value) for value in worker_budget['shard_seed_counts']],
                    'seed_count_source': str(worker_budget['seed_count_source']),
                    'shard_count_source': str(worker_budget['shard_count_source']),
                    'seed_start_offset': int(seed_schedule['seed_start_offset']),
                    'seed_stride_per_iter': int(seed_schedule['seed_stride_per_iter']),
                    'status': 'pending',
                    'attempts': 0,
                }
    seed2_state['task_count'] = len(seed2_state['tasks'])
    seed2_state['seed_stride_per_iter'] = (
        int(next(iter(worker_seed_schedule.values()))['seed_stride_per_iter'])
        if worker_seed_schedule
        else 0
    )


def finalize_dispatch(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> bool:
    coarse_payloads = stage_results_from_state(dispatch_state['seed1'])
    extra_payloads = stage_results_from_state(dispatch_state['seed2']) if isinstance(dispatch_state.get('seed2'), dict) else []
    final_ranking = aggregate_task_payloads(coarse_payloads + extra_payloads, context=context)
    if not final_ranking:
        raise RuntimeError('formal_1v3 produced no valid final candidates')
    final_close_call = close_call_from_ranking(final_ranking, stderr_mult=float(dispatch_state['close_stderr_mult']))
    final_payload = {
        'round_name': 'formal_1v3_round',
        'protocol_arm': context['protocol_arm'],
        'ranking_metric': 'avg_pt_primary_avg_rank_secondary',
        'coarse_round_summary_path': dispatch_state.get('coarse_round_summary_path'),
        'ranking': final_ranking,
        'coarse_task_count': len(coarse_payloads),
        'extra_task_count': len(extra_payloads),
        'close_call': final_close_call,
    }
    final_summary_path = context['run_dir'] / 'formal_1v3_round.json'
    fidelity.atomic_write_json(final_summary_path, final_payload)
    winner = str(final_ranking[0]['checkpoint_type'])
    published = publish_canonical_checkpoints_for_context(context=context, winner=winner)
    validate_published_canonical_checkpoints(
        expected_targets=list(context.get('pending_canonical_alias_targets') or []),
        published=published,
    )
    dispatch_state['status'] = 'completed'
    dispatch_state['stage'] = 'completed'
    dispatch_state['final_round_summary_path'] = str(final_summary_path)
    dispatch_state['final_close_call'] = final_close_call
    dispatch_state['final_winner'] = winner
    dispatch_state['published_canonical_checkpoints'] = list(published)
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        final_payload=final_payload,
        published=published,
    )
    return True


def finalize_single_unique_candidate_dispatch(
    *,
    context: dict[str, Any],
    dispatch_state_path: Path,
    local_label: str,
    remote_label: str | None,
    close_stderr_mult: float,
) -> dict[str, Any]:
    winner = str(context['candidates'][0].arm_name)
    dedupe_group = dict((context.get('checkpoint_dedupe', {}).get('groups') or {}).get(winner) or {})
    close_call = {
        'triggered': False,
        'reason': 'single_unique_candidate_after_weight_dedupe',
        'leader': winner,
        'stderr_mult': float(close_stderr_mult),
        'dedupe_group': dedupe_group,
    }
    final_payload = {
        'round_name': 'formal_1v3_round',
        'protocol_arm': context['protocol_arm'],
        'ranking_metric': 'avg_pt_primary_avg_rank_secondary',
        'coarse_round_summary_path': None,
        'ranking': [
            {
                'checkpoint_type': winner,
                'checkpoint_path': str(context['candidate_index'][winner].meta['checkpoint_path']),
                'games': 0,
                'rankings': [0, 0, 0, 0],
                'avg_pt': 0.0,
                'avg_rank': 0.0,
                'pt_stderr': 0.0,
                'rank_stderr': 0.0,
                'machine_labels': [],
                'round_labels': [],
                'seed_keys': [],
                'valid': True,
                'selection_source': 'weight_dedupe_unique',
                'dedupe_group': dedupe_group,
            }
        ],
        'coarse_task_count': 0,
        'extra_task_count': 0,
        'close_call': close_call,
        'dedupe': {
            'unique_checkpoint_types': list(context.get('unique_shortlist_checkpoint_types') or []),
            'checkpoint_pack_types': list(context.get('checkpoint_pack_types') or []),
            'groups': dict(context.get('checkpoint_dedupe', {}).get('groups') or {}),
        },
    }
    final_summary_path = context['run_dir'] / 'formal_1v3_round.json'
    fidelity.atomic_write_json(final_summary_path, final_payload)
    published = publish_canonical_checkpoints_for_context(context=context, winner=winner)
    validate_published_canonical_checkpoints(
        expected_targets=list(context.get('pending_canonical_alias_targets') or []),
        published=published,
    )
    dispatch_state = {
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': ROUND_KIND_FORMAL_1V3,
        'run_name': context['run_name'],
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'status': 'completed',
        'stage': 'completed',
        'local_label': local_label,
        'remote_label': remote_label,
        'protocol_arm': context['protocol_arm'],
        'config_snapshot': deepcopy(context['config_snapshot']),
        'worker_budgets': {},
        'shortlist_checkpoint_types': list(context.get('shortlist_checkpoint_types') or []),
        'unique_shortlist_checkpoint_types': list(context.get('unique_shortlist_checkpoint_types') or []),
        'checkpoint_pack_types': list(context.get('checkpoint_pack_types') or []),
        'checkpoint_dedupe': deepcopy(context.get('checkpoint_dedupe') or {}),
        'close_stderr_mult': float(close_stderr_mult),
        'close_extra_rounds': 0,
        'close_max_extra_rounds': 0,
        'extra_round_iters': 0,
        'candidate_payloads': [
            fidelity.candidate_cache_payload(candidate, include_meta=True)
            for candidate in context['candidates']
        ],
        'seed1': {
            'stage_name': 'seed1',
            'round_specs': [],
            'candidate_count': len(context['candidates']),
            'worker_count': 0,
            'worker_labels': [],
            'seed_stride_per_iter': 0,
            'task_count': 0,
            'tasks': {},
        },
        'seed2': None,
        'coarse_round_summary_path': None,
        'final_round_summary_path': str(final_summary_path),
        'latest_close_call': close_call,
        'final_close_call': close_call,
        'final_winner': winner,
        'published_canonical_checkpoints': list(published),
    }
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        final_payload=final_payload,
        published=published,
        status_override='completed',
    )
    return dispatch_state


def maybe_promote_seed1_to_seed2(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> bool:
    if dispatch_state.get('stage') != 'seed1':
        return False
    stage_state = dispatch_state['seed1']
    if not common_dispatch.stage_all_tasks_completed(stage_state):
        return False
    coarse_payloads = stage_results_from_state(stage_state)
    coarse_ranking = aggregate_task_payloads(coarse_payloads, context=context)
    close_call = close_call_from_ranking(coarse_ranking, stderr_mult=float(dispatch_state['close_stderr_mult']))
    coarse_summary_path = context['run_dir'] / 'formal_1v3_coarse_round.json'
    fidelity.atomic_write_json(
        coarse_summary_path,
        {
            'round_name': 'formal_1v3_coarse_round',
            'protocol_arm': context['protocol_arm'],
            'ranking_metric': 'avg_pt_primary_avg_rank_secondary',
            'ranking': coarse_ranking,
            'task_count': len(coarse_payloads),
            'close_call': close_call,
        },
    )
    dispatch_state['coarse_round_summary_path'] = str(coarse_summary_path)
    dispatch_state['latest_close_call'] = close_call
    if not close_call['triggered']:
        write_dispatch_state(dispatch_state_path, dispatch_state)
        return finalize_dispatch(context=context, dispatch_state=dispatch_state, dispatch_state_path=dispatch_state_path)
    finalists = [context['candidate_index'][entry['checkpoint_type']] for entry in coarse_ranking[:2]]
    dispatch_state['seed2'] = build_stage_state(
        run_name=str(dispatch_state.get('run_name') or context['run_name']),
        stage_name='seed2',
        candidates=finalists,
        round_specs=[
            build_round_spec(
                round_index=1 + offset,
                round_label=f'close_{1 + offset:02d}',
                iters=int(dispatch_state['extra_round_iters']),
            )
            for offset in range(int(dispatch_state['close_extra_rounds']))
        ],
        worker_budgets=dict(dispatch_state.get('worker_budgets') or {}),
    )
    dispatch_state['stage'] = 'seed2'
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
    )
    return False


def maybe_finalize_or_extend_seed2(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> bool:
    if dispatch_state.get('stage') != 'seed2':
        return False
    seed2_state = dispatch_state.get('seed2')
    if not isinstance(seed2_state, dict) or not common_dispatch.stage_all_tasks_completed(seed2_state):
        return False
    coarse_payloads = stage_results_from_state(dispatch_state['seed1'])
    extra_payloads = stage_results_from_state(seed2_state)
    combined_ranking = aggregate_task_payloads(coarse_payloads + extra_payloads, context=context)
    close_call = close_call_from_ranking(combined_ranking, stderr_mult=float(dispatch_state['close_stderr_mult']))
    dispatch_state['latest_close_call'] = close_call
    evaluated_extra_rounds = len(seed2_state.get('round_specs') or [])
    max_extra_rounds = int(dispatch_state['close_max_extra_rounds'])
    if close_call['triggered'] and evaluated_extra_rounds < max_extra_rounds:
        finalists = [context['candidate_index'][entry['checkpoint_type']] for entry in combined_ranking[:2]]
        append_seed2_rounds(
            dispatch_state=dispatch_state,
            candidates=finalists,
            round_count=min(2, max_extra_rounds - evaluated_extra_rounds),
        )
        write_dispatch_state(dispatch_state_path, dispatch_state)
        update_run_state_for_dispatch(
            run_dir=context['run_dir'],
            dispatch_state_path=dispatch_state_path,
            dispatch_state=dispatch_state,
        )
        return False
    write_dispatch_state(dispatch_state_path, dispatch_state)
    return finalize_dispatch(context=context, dispatch_state=dispatch_state, dispatch_state_path=dispatch_state_path)


def interrupt_active_task(active: ActiveTask) -> None:
    if active.process.poll() is not None:
        return
    subprocess.run(
        ['taskkill', '/PID', str(active.process.pid), '/T', '/F'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def build_task_command_args(*, run_name: str, task_state: dict[str, Any], machine_label: str) -> list[str]:
    return [
        'run-task',
        '--run-name',
        run_name,
        '--candidate-arm',
        str(task_state['candidate_arm']),
        '--round-index',
        str(task_state['round_index']),
        '--round-label',
        str(task_state['round_label']),
        '--seed-key',
        str(task_state['seed_key']),
        '--iters',
        str(task_state['iters']),
        '--machine-label',
        machine_label,
        '--seed-count-per-iter',
        str(task_state['seed_count_per_iter']),
        '--shard-count',
        str(task_state['shard_count']),
        '--seed-start-offset',
        str(int(task_state.get('seed_start_offset', 0) or 0)),
        '--seed-stride-per-iter',
        str(int(task_state.get('seed_stride_per_iter', task_state['seed_count_per_iter']) or task_state['seed_count_per_iter'])),
    ]


def build_remote_interactive_window_command(
    *,
    worker: WorkerSpec,
    run_name: str,
    task_state: dict[str, Any],
    remote_result_path: Path,
    remote_runtime_root: Path,
) -> list[str]:
    repo_root = Path(worker.repo or str(REPO_ROOT))
    remote_script = repo_root / 'mortal' / SCRIPT_PATH.name
    python_args = build_task_command_args(
        run_name=run_name,
        task_state=task_state,
        machine_label=worker.label,
    )
    python_args.extend(['--result-json', str(remote_result_path)])
    python_args_base64 = base64.b64encode(json.dumps(python_args, ensure_ascii=True).encode('utf-8')).decode('ascii')
    artifact_stem = task_artifact_stem(task_state)
    window_title = f'MahjongAI formal_1v3 {artifact_stem}'
    remote_helper = repo_root / 'scripts' / INTERACTIVE_REMOTE_PYTHON_HELPER.name
    ps_command = (
        f"if (Test-Path {quote_ps(str(remote_result_path))}) {{ Remove-Item -LiteralPath {quote_ps(str(remote_result_path))} -Force }}; "
        f"& {quote_ps(str(remote_helper))} "
        f"-RepoRoot {quote_ps(str(repo_root))} "
        f"-PythonExe {quote_ps(worker.python or sys.executable)} "
        f"-PythonScript {quote_ps(str(remote_script))} "
        f"-PythonArgsBase64 {quote_ps(python_args_base64)} "
        f"-TaskId {quote_ps(artifact_stem)} "
        f"-RuntimeRoot {quote_ps(str(remote_runtime_root))} "
        f"-WindowTitle {quote_ps(window_title)}"
    )
    command = ['ssh']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.append(worker.host or common_dispatch.DEFAULT_REMOTE_HOST)
    command.append(ps_command)
    return command


def launch_remote_task(
    *,
    worker: WorkerSpec,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    launch_mode: str,
) -> ActiveTask:
    remote_dispatch_root = map_repo_path_to_remote(dispatch_root, remote_repo=worker.repo or str(REPO_ROOT))
    artifact_stem = task_artifact_stem(task_state)
    remote_result_path = remote_dispatch_root / 'remote_results' / f'{artifact_stem}.json'
    remote_runtime_root = remote_dispatch_root / 'remote_runtime' / artifact_stem
    local_result_path = dispatch.ensure_dir(dispatch_root / 'results') / f'{artifact_stem}.json'
    log_path = dispatch.ensure_dir(dispatch_root / 'logs') / f'{artifact_stem}__{worker.label}.log'
    if local_result_path.exists():
        local_result_path.unlink()
    command_args = build_task_command_args(
        run_name=run_name,
        task_state=task_state,
        machine_label=worker.label,
    )
    if launch_mode == 'ssh_inline':
        command = dispatch.build_remote_python_command(
            worker=worker,
            script_path=SCRIPT_PATH,
            remote_result_path=remote_result_path,
            command_args=command_args,
        )
    elif launch_mode == 'interactive_window':
        command = build_remote_interactive_window_command(
            worker=worker,
            run_name=run_name,
            task_state=task_state,
            remote_result_path=remote_result_path,
            remote_runtime_root=remote_runtime_root,
        )
    else:
        raise ValueError(f'unsupported remote launch mode `{launch_mode}`')
    log_handle = log_path.open('w', encoding='utf-8', newline='\n')
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    task_state['status'] = 'running'
    task_state['attempts'] = int(task_state.get('attempts', 0)) + 1
    task_state['worker_label'] = worker.label
    task_state['local_result_path'] = str(local_result_path)
    task_state['log_path'] = str(log_path)
    task_state['remote_result_path'] = str(remote_result_path)
    task_state['remote_launch_mode'] = launch_mode
    task_state['remote_runtime_root'] = str(remote_runtime_root)
    task_state['remote_task_name'] = f'{REMOTE_INTERACTIVE_TASK_NAME_PREFIX}{artifact_stem}'
    task_state['started_at'] = fidelity.ts_now()
    return ActiveTask(
        worker=worker,
        stage_name='',
        task_id=str(task_state['task_id']),
        task_state=task_state,
        process=process,
        log_path=log_path,
        local_result_path=local_result_path,
        remote_result_path=remote_result_path,
    )


def handle_finished_task(
    *,
    active: ActiveTask,
    max_attempts: int,
) -> None:
    dispatch.handle_finished_json_task(
        active=active,
        max_attempts=max_attempts,
        finished_at=fidelity.ts_now(),
        validate_result=load_task_result,
    )


def try_collect_running_remote_result(active: ActiveTask) -> dict[str, Any] | None:
    if active.worker.kind != 'remote':
        return None
    if str(active.task_state.get('status') or '') != 'running':
        return None
    if str(active.task_state.get('remote_launch_mode') or '') != 'interactive_window':
        return None
    if active.remote_result_path is None:
        return None
    try:
        dispatch.fetch_remote_result(active.worker, str(active.remote_result_path), active.local_result_path)
        payload = load_task_result(active.local_result_path)
    except Exception:
        if active.local_result_path.exists():
            try:
                active.local_result_path.unlink()
            except OSError:
                pass
        return None
    common_dispatch.interrupt_remote_active_task(active)
    active.task_state['status'] = 'completed'
    active.task_state['finished_at'] = str(payload.get('completed_at') or fidelity.ts_now())
    active.task_state['completion_source'] = 'remote_result_json'
    active.task_state.pop('error', None)
    return payload


def find_next_pending_task_for_worker(
    stage_state: dict[str, Any],
    *,
    worker_label: str,
) -> tuple[str, dict[str, Any]] | None:
    pending = []
    for task_id, task in stage_state.get('tasks', {}).items():
        if str(task.get('status', 'pending')) != 'pending':
            continue
        assigned_worker_label = str(task.get('assigned_worker_label') or '')
        if assigned_worker_label and assigned_worker_label != worker_label:
            continue
        pending.append((task_id, task))
    if not pending:
        return None
    pending.sort(key=lambda item: item[0])
    return pending[0]


def validate_frozen_dispatch_state(dispatch_state: dict[str, Any]) -> None:
    config_snapshot = dispatch_state.get('config_snapshot')
    if not isinstance(config_snapshot, dict) or not isinstance(config_snapshot.get('base_1v3_cfg'), dict):
        raise RuntimeError(
            'formal_1v3 dispatch state is missing frozen config snapshot; restart this dispatch'
        )
    worker_budgets = dispatch_state.get('worker_budgets')
    if not isinstance(worker_budgets, dict) or not worker_budgets:
        raise RuntimeError('formal_1v3 dispatch state is missing frozen worker budgets; restart this dispatch')
    for stage_name in ('seed1', 'seed2'):
        stage_state = dispatch_state.get(stage_name)
        if not isinstance(stage_state, dict):
            continue
        apply_stage_task_seed_schedule(stage_state, worker_budgets=worker_budgets)
        for task in stage_state.get('tasks', {}).values():
            assigned_worker_label = str(task.get('assigned_worker_label') or '')
            if not assigned_worker_label or assigned_worker_label not in worker_budgets:
                raise RuntimeError(
                    'formal_1v3 dispatch state is missing assigned_worker_label for a task; restart this dispatch'
                )
            if task.get('seed_count_per_iter') is None or task.get('shard_count') is None:
                raise RuntimeError(
                    'formal_1v3 dispatch state is missing frozen seed/shard budgets for a task; restart this dispatch'
                )
            if task.get('seed_start_offset') is None or task.get('seed_stride_per_iter') is None:
                raise RuntimeError(
                    'formal_1v3 dispatch state is missing frozen seed schedule for a task; restart this dispatch'
                )


def validate_resume_worker_labels(
    *,
    dispatch_state: dict[str, Any],
    workers: list[WorkerSpec],
) -> None:
    current_worker_labels = sorted(
        {
            str(worker.label).strip()
            for worker in workers
            if str(worker.label or '').strip()
        }
    )
    if not current_worker_labels:
        raise RuntimeError('formal_1v3 resume has no available workers')
    worker_budgets = dispatch_state.get('worker_budgets')
    if not isinstance(worker_budgets, dict) or not worker_budgets:
        raise RuntimeError('formal_1v3 dispatch state is missing frozen worker budgets; restart this dispatch')
    frozen_worker_labels = sorted(
        str(label).strip()
        for label in worker_budgets
        if str(label or '').strip()
    )
    missing_worker_labels = [
        label for label in frozen_worker_labels if label not in current_worker_labels
    ]
    if missing_worker_labels:
        raise RuntimeError(
            'formal_1v3 dispatch worker labels do not match this resume invocation; '
            f'frozen dispatch requires {missing_worker_labels}, current workers are {current_worker_labels}. '
            'rerun with the original --local-label/--remote-label and remote enablement, or restart this dispatch'
        )


def launch_task_for_worker(
    *,
    worker: WorkerSpec,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    launch_mode: str | None = None,
) -> ActiveTask:
    results_dir = dispatch.ensure_dir(dispatch_root / 'results')
    logs_dir = dispatch.ensure_dir(dispatch_root / 'logs')
    artifact_stem = task_artifact_stem(task_state)
    result_path = results_dir / f'{artifact_stem}.json'
    log_path = logs_dir / f'{artifact_stem}__{worker.label}.log'
    if result_path.exists():
        result_path.unlink()
    if worker.kind == 'remote':
        return launch_remote_task(
            worker=worker,
            run_name=run_name,
            task_state=task_state,
            dispatch_root=dispatch_root,
            launch_mode=str(launch_mode or common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE),
        )
    spec = JsonTaskLaunchSpec(
        task_id=str(task_state['task_id']),
        stage_name='',
        local_result_path=result_path,
        log_path=log_path,
        remote_result_path=None,
        command_args=build_task_command_args(
            run_name=run_name,
            task_state=task_state,
            machine_label=worker.label,
        ),
        cwd=REPO_ROOT,
    )
    active = dispatch.launch_json_task(
        worker,
        task_state=task_state,
        script_path=SCRIPT_PATH,
        spec=spec,
    )
    task_state['started_at'] = fidelity.ts_now()
    return active


def run_dispatch(args: argparse.Namespace) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fidelity.acquire_run_lock(run_dir, args.run_name)
    atexit.register(fidelity.release_run_lock, lock_path)
    dispatch_root = dispatch.ensure_dir(dispatch_root_for_run(run_dir))
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    dispatch_control_path = dispatch_control_path_for_run(run_dir)
    try:
        if getattr(args, 'manual_shortlist_manifest', None):
            materialize_manual_formal_1v3_state(
                run_dir=run_dir,
                manifest_path=Path(args.manual_shortlist_manifest).resolve(),
            )
        dispatch_state: dict[str, Any] | None = None
        if dispatch_state_path.exists():
            dispatch_state = load_dispatch_state(dispatch_state_path)
            if dispatch_state_is_completed(dispatch_state):
                reconcile_completed_dispatch_run_state(
                    run_dir=run_dir,
                    dispatch_state_path=dispatch_state_path,
                    dispatch_state=dispatch_state,
                )
                return 0
            common_dispatch.reset_running_tasks_for_resume(dispatch_state)
            validate_frozen_dispatch_state(dispatch_state)
        context = load_formal_context(run_dir)
        if len(context['candidates']) == 1:
            finalize_single_unique_candidate_dispatch(
                context=context,
                dispatch_state_path=dispatch_state_path,
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                close_stderr_mult=args.close_stderr_mult,
            )
            return 0
        workers = common_dispatch.build_workers(
            enable_remote=not args.local_only,
            local_python=args.local_python,
            local_label=args.local_label,
            remote_host=args.remote_host,
            remote_repo=args.remote_repo,
            remote_python=args.remote_python,
            remote_label=args.remote_label,
            ssh_key=args.ssh_key,
        )
        if dispatch_state is not None:
            validate_resume_worker_labels(
                dispatch_state=dispatch_state,
                workers=workers,
            )
        if dispatch_state is None:
            worker_budgets = collect_worker_budgets(
                workers,
                frozen_1v3_cfg=context['base_1v3_cfg'],
            )
            dispatch_state = initialize_dispatch_state(
                context=context,
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                worker_budgets=worker_budgets,
                coarse_iters=args.coarse_iters,
                extra_iters=args.extra_iters,
                close_stderr_mult=args.close_stderr_mult,
                close_extra_rounds=args.close_extra_rounds,
                close_max_extra_rounds=args.close_max_extra_rounds,
            )
        if dispatch_control_path.exists():
            control_state = common_dispatch.load_dispatch_control(dispatch_control_path)
        else:
            control_state = common_dispatch.initialize_dispatch_control_state(
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                remote_launch_mode=args.remote_launch_mode,
            )
            common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
        if common_dispatch.ensure_control_state_workers(
            control_state=control_state,
            local_label=args.local_label,
            remote_label=args.remote_label if not args.local_only else None,
            remote_launch_mode=args.remote_launch_mode,
        ):
            common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
        remote_assets = dict(dispatch_state.get('remote_assets') or {})
        for worker in workers:
            if worker.kind != 'remote':
                continue
            remote_assets[worker.label] = sync_remote_formal_shortlist_assets(
                worker=worker,
                context=context,
            )
        if remote_assets:
            dispatch_state['remote_assets'] = remote_assets
        write_dispatch_state(dispatch_state_path, dispatch_state)
        update_run_state_for_dispatch(
            run_dir=run_dir,
            dispatch_state_path=dispatch_state_path,
            dispatch_state=dispatch_state,
        )
        active: dict[str, ActiveTask] = {}
        while True:
            control_state = common_dispatch.load_dispatch_control(dispatch_control_path)
            if common_dispatch.ensure_control_state_workers(
                control_state=control_state,
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                remote_launch_mode=args.remote_launch_mode,
            ):
                common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
            finished_labels: list[str] = []
            state_changed = False
            if common_dispatch.apply_worker_control_requests(control_state=control_state, active=active):
                state_changed = True
                common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
            for worker_label, active_task in list(active.items()):
                if active_task.process.poll() is None:
                    early_payload = try_collect_running_remote_result(active_task)
                    if early_payload is not None:
                        finished_labels.append(worker_label)
                        state_changed = True
                        continue
                if active_task.process.poll() is None:
                    continue
                handle_finished_task(active=active_task, max_attempts=args.max_attempts)
                finished_labels.append(worker_label)
            for worker_label in finished_labels:
                active.pop(worker_label, None)
            if finished_labels:
                state_changed = True
            if state_changed:
                write_dispatch_state(dispatch_state_path, dispatch_state)
            if dispatch_state.get('stage') == 'seed1' and not active:
                stage_state = dispatch_state['seed1']
                if common_dispatch.stage_any_task_failed(stage_state):
                    raise RuntimeError('formal_1v3 coarse stage exhausted retries')
                if maybe_promote_seed1_to_seed2(
                    context=context,
                    dispatch_state=dispatch_state,
                    dispatch_state_path=dispatch_state_path,
                ):
                    break
                if common_dispatch.stage_all_tasks_completed(stage_state):
                    continue
            if dispatch_state.get('stage') == 'seed2' and not active:
                stage_state = dispatch_state['seed2']
                if common_dispatch.stage_any_task_failed(stage_state):
                    raise RuntimeError('formal_1v3 close stage exhausted retries')
                if maybe_finalize_or_extend_seed2(
                    context=context,
                    dispatch_state=dispatch_state,
                    dispatch_state_path=dispatch_state_path,
                ):
                    break
                if common_dispatch.stage_all_tasks_completed(stage_state):
                    continue
            if dispatch_state.get('stage') == 'completed':
                break
            stage_state = common_dispatch.active_stage_state(dispatch_state)
            for worker in workers:
                worker_control = common_dispatch.worker_control_entry(control_state, worker.label)
                if bool(worker_control.get('paused')):
                    continue
                if worker.label in active:
                    continue
                next_task = find_next_pending_task_for_worker(stage_state, worker_label=worker.label)
                if next_task is None:
                    continue
                _, task_state = next_task
                active[worker.label] = launch_task_for_worker(
                    worker=worker,
                    run_name=args.run_name,
                    task_state=task_state,
                    dispatch_root=dispatch_root,
                    launch_mode=str(worker_control.get('launch_mode') or args.remote_launch_mode),
                )
                write_dispatch_state(dispatch_state_path, dispatch_state)
            if not active and common_dispatch.find_next_pending_task(common_dispatch.active_stage_state(dispatch_state)) is None:
                time.sleep(1.0)
                continue
            time.sleep(float(args.poll_seconds))
        return 0
    except Exception as exc:
        if dispatch_state_path.exists():
            dispatch_state = load_dispatch_state(dispatch_state_path)
            dispatch_state['status'] = 'failed'
            dispatch_state['error'] = str(exc)
            dispatch_state['traceback'] = traceback.format_exc()
            write_dispatch_state(dispatch_state_path, dispatch_state)
            update_run_state_for_dispatch(
                run_dir=run_dir,
                dispatch_state_path=dispatch_state_path,
                dispatch_state=dispatch_state,
                status_override='failed',
            )
        raise
    finally:
        fidelity.release_run_lock(lock_path)


def print_status(args: argparse.Namespace) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    dispatch_control_path = dispatch_control_path_for_run(run_dir)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state at {dispatch_state_path}')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    control_state = common_dispatch.load_dispatch_control(dispatch_control_path) if dispatch_control_path.exists() else {'workers': {}}
    payload = {
        'round_kind': ROUND_KIND_FORMAL_1V3,
        'run_name': args.run_name,
        'stage': dispatch_state.get('stage'),
        'status': dispatch_state.get('status'),
        'seed1': common_dispatch.summarize_dispatch_task_status(dispatch_state['seed1']),
        'seed2': (
            common_dispatch.summarize_dispatch_task_status(dispatch_state['seed2'])
            if isinstance(dispatch_state.get('seed2'), dict)
            else None
        ),
        'workers': control_state.get('workers', {}),
        'worker_budgets': dispatch_state.get('worker_budgets', {}),
        'close_call': dispatch_state.get('latest_close_call') or dispatch_state.get('final_close_call'),
        'final_winner': dispatch_state.get('final_winner'),
    }
    print(json.dumps(fidelity.normalize_payload(payload), ensure_ascii=False, indent=2))
    return 0


def update_worker_pause(args: argparse.Namespace, *, paused: bool) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    dispatch_control_path = dispatch_control_path_for_run(run_dir)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state at {dispatch_state_path}')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    local_label = str(dispatch_state.get('local_label') or common_dispatch.DEFAULT_LOCAL_LABEL)
    remote_label = dispatch_state.get('remote_label')
    if dispatch_control_path.exists():
        control_state = common_dispatch.load_dispatch_control(dispatch_control_path)
    else:
        control_state = common_dispatch.initialize_dispatch_control_state(
            local_label=local_label,
            remote_label=remote_label,
            remote_launch_mode=common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE,
        )
    common_dispatch.ensure_control_state_workers(
        control_state=control_state,
        local_label=local_label,
        remote_label=remote_label,
        remote_launch_mode=common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE,
    )
    entry = common_dispatch.set_worker_pause(
        control_state,
        worker_label=args.worker_label,
        paused=paused,
        stop_active=bool(getattr(args, 'stop_active', False)),
    )
    common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
    payload = {
        'round_kind': ROUND_KIND_FORMAL_1V3,
        'run_name': args.run_name,
        'worker_label': args.worker_label,
        'paused': bool(entry.get('paused')),
        'interrupt_requested': bool(entry.get('interrupt_requested')),
        'control_path': str(dispatch_control_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    dispatch_cmd = subparsers.add_parser('dispatch')
    dispatch_cmd.add_argument('--run-name', required=True)
    dispatch_cmd.add_argument('--manual-shortlist-manifest')
    dispatch_cmd.add_argument('--local-only', action='store_true')
    dispatch_cmd.add_argument('--local-python', default=sys.executable)
    dispatch_cmd.add_argument('--local-label', default=common_dispatch.hostname_fallback())
    dispatch_cmd.add_argument('--remote-host', default=common_dispatch.DEFAULT_REMOTE_HOST)
    dispatch_cmd.add_argument('--remote-repo', default=common_dispatch.DEFAULT_REMOTE_REPO)
    dispatch_cmd.add_argument('--remote-python', default=common_dispatch.DEFAULT_REMOTE_PYTHON)
    dispatch_cmd.add_argument('--remote-label', default=common_dispatch.DEFAULT_REMOTE_LABEL)
    dispatch_cmd.add_argument('--ssh-key', default=common_dispatch.DEFAULT_SSH_KEY)
    dispatch_cmd.add_argument(
        '--remote-launch-mode',
        default=common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE,
        choices=sorted(common_dispatch.REMOTE_LAUNCH_MODES),
    )
    dispatch_cmd.add_argument('--poll-seconds', type=float, default=common_dispatch.DEFAULT_POLL_SECONDS)
    dispatch_cmd.add_argument('--max-attempts', type=int, default=common_dispatch.DEFAULT_MAX_ATTEMPTS)
    dispatch_cmd.add_argument('--coarse-iters', type=int, default=DEFAULT_COARSE_ITERS)
    dispatch_cmd.add_argument('--extra-iters', type=int, default=DEFAULT_EXTRA_ITERS)
    dispatch_cmd.add_argument('--close-stderr-mult', type=float, default=DEFAULT_CLOSE_STDERR_MULT)
    dispatch_cmd.add_argument('--close-extra-rounds', type=int, default=DEFAULT_CLOSE_EXTRA_ROUNDS)
    dispatch_cmd.add_argument('--close-max-extra-rounds', type=int, default=DEFAULT_CLOSE_MAX_EXTRA_ROUNDS)
    run_task = subparsers.add_parser('run-task')
    run_task.add_argument('--run-name', required=True)
    run_task.add_argument('--candidate-arm', required=True)
    run_task.add_argument('--round-index', type=int, required=True)
    run_task.add_argument('--round-label', required=True)
    run_task.add_argument('--seed-key', type=int, required=True)
    run_task.add_argument('--iters', type=int, required=True)
    run_task.add_argument('--machine-label', required=True)
    run_task.add_argument('--seed-count-per-iter', type=int)
    run_task.add_argument('--shard-count', type=int)
    run_task.add_argument('--seed-start-offset', type=int)
    run_task.add_argument('--seed-stride-per-iter', type=int)
    run_task.add_argument('--result-json', required=True)
    resolve_budget = subparsers.add_parser('resolve-budget')
    resolve_budget.add_argument('--machine-label', required=True)
    resolve_budget.add_argument('--cfg-json-b64')
    status = subparsers.add_parser('status')
    status.add_argument('--run-name', required=True)
    pause_worker = subparsers.add_parser('pause-worker')
    pause_worker.add_argument('--run-name', required=True)
    pause_worker.add_argument('--worker-label', required=True)
    pause_worker.add_argument('--stop-active', action='store_true')
    resume_worker = subparsers.add_parser('resume-worker')
    resume_worker.add_argument('--run-name', required=True)
    resume_worker.add_argument('--worker-label', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'run-task':
        result_json = Path(args.result_json)
        payload = execute_single_task(
            run_name=args.run_name,
            candidate_arm=args.candidate_arm,
            round_index=args.round_index,
            round_label=args.round_label,
            seed_key=args.seed_key,
            iters=args.iters,
            result_json=result_json,
            machine_label=args.machine_label,
            seed_count_per_iter=args.seed_count_per_iter,
            shard_count=args.shard_count,
            seed_start_offset=getattr(args, 'seed_start_offset', None),
            seed_stride_per_iter=getattr(args, 'seed_stride_per_iter', None),
        )
        print(json.dumps(fidelity.normalize_payload(build_run_task_cli_summary(payload, result_json=result_json)), ensure_ascii=False, indent=2))
        return
    if args.command == 'resolve-budget':
        frozen_1v3_cfg = decode_frozen_cfg_payload(args.cfg_json_b64)
        if frozen_1v3_cfg is None:
            frozen_1v3_cfg = formal.build_formal_config_snapshot(ab.build_base_config())['base_1v3_cfg']
        print(
            json.dumps(
                fidelity.normalize_payload(
                    resolve_worker_budget_snapshot(
                        machine_label=args.machine_label,
                        frozen_1v3_cfg=frozen_1v3_cfg,
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == 'dispatch':
        raise SystemExit(run_dispatch(args))
    if args.command == 'status':
        raise SystemExit(print_status(args))
    if args.command == 'pause-worker':
        raise SystemExit(update_worker_pause(args, paused=True))
    if args.command == 'resume-worker':
        raise SystemExit(update_worker_pause(args, paused=False))
    raise ValueError(f'unknown command `{args.command}`')


if __name__ == '__main__':
    main()

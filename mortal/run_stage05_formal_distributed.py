from __future__ import annotations

import argparse
import atexit
import base64
import json
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import distributed_dispatch as dispatch
import run_stage05_ab as ab
import run_stage05_fidelity as fidelity
import run_stage05_formal as formal
import run_stage05_p1_only as p1_only
import run_stage05_winner_refine_distributed as common_dispatch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
ROUND_KIND_FORMAL = 'formal'
DISPATCH_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA_VERSION = 1
DEFAULT_FORMAL_SEED_OFFSET = 2000
DEFAULT_FORMAL_STEP_SCALE = 5.0
DEFAULT_REMOTE_FORMAL_NUM_WORKERS = 4
DEFAULT_REMOTE_FORMAL_FILE_BATCH_SIZE = 10
DEFAULT_REMOTE_FORMAL_PREFETCH_FACTOR = 4
DEFAULT_REMOTE_FORMAL_VAL_FILE_BATCH_SIZE = 7
DEFAULT_REMOTE_FORMAL_VAL_PREFETCH_FACTOR = 5
REMOTE_INTERACTIVE_TASK_NAME_PREFIX = 'MahjongAI-Formal-'

WorkerSpec = dispatch.WorkerSpec
ActiveTask = dispatch.ActiveTask
JsonTaskLaunchSpec = dispatch.JsonTaskLaunchSpec


def quote_ps(value: str) -> str:
    return dispatch.quote_ps(value)


def path_to_scp_remote(path: str | Path) -> str:
    return dispatch.path_to_scp_remote(path)


def ensure_dir(path: Path) -> Path:
    return dispatch.ensure_dir(path)


def dispatch_root_for_run(run_dir: Path) -> Path:
    return run_dir / 'distributed' / 'formal_dispatch'


def dispatch_state_path_for_run(run_dir: Path) -> Path:
    return dispatch_root_for_run(run_dir) / 'dispatch_state.json'


def dispatch_control_path_for_run(run_dir: Path) -> Path:
    return dispatch_root_for_run(run_dir) / 'dispatch_control.json'


def load_dispatch_state(path: Path) -> dict[str, Any]:
    return fidelity.load_json(path)


def write_dispatch_state(path: Path, payload: dict[str, Any]) -> None:
    payload['updated_at'] = fidelity.ts_now()
    fidelity.atomic_write_json(path, payload)


def reconstruct_candidate(payload: dict[str, Any]) -> fidelity.CandidateSpec:
    return fidelity.CandidateSpec(
        arm_name=str(payload['arm_name']),
        scheduler_profile=str(payload['scheduler_profile']),
        curriculum_profile=str(payload['curriculum_profile']),
        weight_profile=str(payload['weight_profile']),
        window_profile=str(payload['window_profile']),
        cfg_overrides=dict(payload.get('cfg_overrides') or {}),
        meta=dict(payload.get('meta') or {}),
    )


def remove_tree_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def build_child_run_name(coordinator_run_name: str, candidate_arm: str) -> str:
    suffix = candidate_arm.split('__', 1)[1] if '__' in candidate_arm else candidate_arm
    return f'{coordinator_run_name}__{suffix}'


def build_task_id(*, candidate_arm: str) -> str:
    return f'formal__{candidate_arm}'


def dedupe_candidate_arms(candidate_arms: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for arm_name in candidate_arms:
        normalized = str(arm_name or '').strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


WINNER_REFINE_BUDGET_KEYS = (
    'rank_budget_ratio',
    'opp_budget_ratio',
    'danger_budget_ratio',
)
WINNER_REFINE_ALIAS_SCALE_TARGETS = (0.85, 1.0, 1.15)


def winner_refine_center_alias(entry: dict[str, Any]) -> str:
    meta = entry.get('candidate_meta') or {}
    candidate_name = str(meta.get('candidate_name') or '').strip()
    if candidate_name:
        parts = candidate_name.split('_')
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        alias = '_'.join(parts)
        if alias:
            return alias
    mix_name = str(meta.get('mix_name') or '').strip()
    if mix_name:
        return mix_name
    arm_name = str(entry.get('arm_name') or '').strip()
    return arm_name.split('__', 1)[-1] if arm_name else 'candidate'


def winner_refine_nearest_scale(ratios: list[float]) -> float | None:
    if not ratios:
        return None
    avg_ratio = sum(ratios) / len(ratios)
    spread = max(abs(ratio - avg_ratio) for ratio in ratios)
    nearest = min(WINNER_REFINE_ALIAS_SCALE_TARGETS, key=lambda target: abs(target - avg_ratio))
    if spread <= 0.03 and abs(nearest - avg_ratio) <= 0.04:
        return nearest
    return None


def winner_refine_delta_marker(ratio: float) -> str:
    if ratio >= 1.20:
        return '++'
    if ratio >= 1.05:
        return '+'
    if ratio <= 0.80:
        return '--'
    if ratio <= 0.95:
        return '-'
    return ''


def winner_refine_scale_label(scale: float) -> str:
    if abs(scale - 1.0) <= 1e-9:
        return '*1.0'
    return f'*{scale:.2f}'


def winner_refine_candidate_alias(
    entry: dict[str, Any],
    *,
    ranking_entry_index: dict[str, dict[str, Any]],
    center_alias_index: dict[str, str] | None = None,
    center_entry_index: dict[str, dict[str, Any]] | None = None,
) -> str:
    if center_alias_index is None:
        center_alias_index = {}
    if center_entry_index is None:
        center_entry_index = {}
    meta = entry.get('candidate_meta') or {}
    source_arm = str(meta.get('source_arm') or '').strip()
    source_entry = ranking_entry_index.get(source_arm) or center_entry_index.get(source_arm)
    center_alias = center_alias_index.get(source_arm)
    if center_alias is None and source_entry is not None:
        center_alias = winner_refine_center_alias(source_entry)
    if center_alias is None:
        return winner_refine_center_alias(entry)
    if source_entry is None:
        return center_alias
    source_meta = source_entry.get('candidate_meta') or {}
    ratios: list[float] = []
    changes: list[str] = []
    for head_name, budget_key in (
        ('rank', 'rank_budget_ratio'),
        ('opp', 'opp_budget_ratio'),
        ('danger', 'danger_budget_ratio'),
    ):
        source_budget = float(source_meta.get(budget_key, 0.0) or 0.0)
        candidate_budget = float(meta.get(budget_key, 0.0) or 0.0)
        if source_budget <= 0.0:
            continue
        ratio = candidate_budget / source_budget
        ratios.append(ratio)
        marker = winner_refine_delta_marker(ratio)
        if marker:
            changes.append(f'{head_name}{marker}')
    matched_scale = winner_refine_nearest_scale(ratios)
    if matched_scale is not None:
        return f'{center_alias}{winner_refine_scale_label(matched_scale)}'
    if changes:
        return f'{center_alias}({"/".join(changes)})'
    return f'{center_alias}{winner_refine_scale_label(1.0)}'


def build_winner_refine_alias_index(
    ranking: list[dict[str, Any]],
    *,
    center_alias_index: dict[str, str] | None = None,
    center_entry_index: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    ranking_entry_index = {
        str(entry.get('arm_name') or ''): entry
        for entry in ranking
        if isinstance(entry, dict) and entry.get('arm_name')
    }
    alias_to_arm: dict[str, str] = {}
    arm_to_alias: dict[str, str] = {}
    collisions: set[str] = set()
    for arm_name, entry in ranking_entry_index.items():
        alias = winner_refine_candidate_alias(
            entry,
            ranking_entry_index=ranking_entry_index,
            center_alias_index=center_alias_index,
            center_entry_index=center_entry_index,
        )
        arm_to_alias[arm_name] = alias
        existing = alias_to_arm.get(alias)
        if existing is None:
            alias_to_arm[alias] = arm_name
        elif existing != arm_name:
            collisions.add(alias)
    for alias in collisions:
        alias_to_arm.pop(alias, None)
    return alias_to_arm, arm_to_alias


def resolve_candidate_arm_inputs(
    *,
    candidate_inputs: list[str],
    valid_candidate_index: dict[str, fidelity.CandidateSpec],
    alias_to_arm: dict[str, str],
) -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for item in candidate_inputs:
        normalized = str(item or '').strip()
        if not normalized:
            continue
        if normalized in valid_candidate_index:
            if normalized not in seen:
                resolved.append(normalized)
                seen.add(normalized)
            continue
        alias_match = alias_to_arm.get(normalized)
        if alias_match is not None:
            if alias_match not in seen:
                resolved.append(alias_match)
                seen.add(alias_match)
            continue
        missing.append(normalized)
    if missing:
        available_aliases = ', '.join(sorted(alias_to_arm))
        raise RuntimeError(
            'source winner_refine_round is missing valid candidate(s): '
            + ', '.join(sorted(missing))
            + (f'; available aliases: {available_aliases}' if available_aliases else '')
        )
    return resolved


def load_source_context(
    *,
    source_run_dir: Path,
    coordinator_run_name: str,
    candidate_arms: list[str],
    formal_seed_offset: int,
    formal_step_scale: float,
) -> dict[str, Any]:
    state_path = source_run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing source state.json under {source_run_dir}')
    state = fidelity.load_json(state_path)
    p1_state = state.get('p1')
    if not isinstance(p1_state, dict):
        raise RuntimeError('source run state has no p1 section')
    winner_refine_round = p1_state.get('winner_refine_round')
    if not isinstance(winner_refine_round, dict):
        raise RuntimeError('source run is missing completed p1 winner_refine_round')
    protocol_decide_round = p1_state.get('protocol_decide_round')
    ranking = winner_refine_round.get('ranking')
    if not isinstance(ranking, list):
        raise RuntimeError('source winner_refine_round ranking is missing')
    center_alias_index: dict[str, str] = {}
    center_entry_index: dict[str, dict[str, Any]] = {}
    if isinstance(protocol_decide_round, dict):
        protocol_ranking = protocol_decide_round.get('ranking')
        if isinstance(protocol_ranking, list):
            for entry in protocol_ranking:
                if not isinstance(entry, dict):
                    continue
                arm_name = str(entry.get('arm_name') or '').strip()
                if not arm_name:
                    continue
                center_alias_index[arm_name] = winner_refine_center_alias(entry)
                center_entry_index[arm_name] = entry
    candidate_index = {
        entry['arm_name']: fidelity.candidate_from_entry(entry)
        for entry in ranking
        if isinstance(entry, dict) and entry.get('valid')
    }
    alias_to_arm, arm_to_alias = build_winner_refine_alias_index(
        ranking,
        center_alias_index=center_alias_index,
        center_entry_index=center_entry_index,
    )
    resolved_candidate_arms = resolve_candidate_arm_inputs(
        candidate_inputs=candidate_arms,
        valid_candidate_index=candidate_index,
        alias_to_arm=alias_to_arm,
    )
    source_seed = p1_only.infer_resume_seed(state)
    if source_seed is None:
        raise RuntimeError('could not recover source base seed from state.json')
    selected_protocol_arm = str(
        p1_state.get('selected_protocol_arm')
        or state.get('final_conclusion', {}).get('p1_protocol_winner')
        or candidate_index[resolved_candidate_arms[0]].meta.get('protocol_arm', '')
        or ''
    ).strip()
    if not selected_protocol_arm:
        raise RuntimeError('source selected protocol arm is missing')
    selected_protocol_arms = p1_only.dedupe_protocol_arms(
        p1_state.get('protocol_arms') or state.get('selected_protocol_arms') or list(p1_only.FROZEN_TOP3)
    )
    source_refine_front_runner = str(
        p1_state.get('winner_refine_front_runner')
        or state.get('final_conclusion', {}).get('p1_refine_front_runner')
        or ''
    ).strip()
    ranking_index = {
        str(entry['arm_name']): int(entry.get('rank', 999))
        for entry in ranking
        if isinstance(entry, dict)
    }
    candidates = [candidate_index[arm_name] for arm_name in resolved_candidate_arms]
    return {
        'source_run_name': source_run_dir.name,
        'source_seed': int(source_seed),
        'selected_protocol_arm': selected_protocol_arm,
        'selected_protocol_arms': list(selected_protocol_arms),
        'source_refine_front_runner': source_refine_front_runner,
        'formal_seed': int(source_seed) + int(formal_seed_offset),
        'formal_step_scale': float(formal_step_scale),
        'candidate_source_ranks': dict(ranking_index),
        'candidate_alias_to_arm': dict(alias_to_arm),
        'candidate_arm_to_alias': {arm_name: arm_to_alias.get(arm_name, arm_name) for arm_name in candidate_index},
        'candidate_payloads': [
            {
                **fidelity.candidate_cache_payload(candidate, include_meta=True),
                'source_rank': int(ranking_index.get(candidate.arm_name, 999)),
                'candidate_alias': arm_to_alias.get(candidate.arm_name, candidate.arm_name),
                'child_run_name': build_child_run_name(coordinator_run_name, candidate.arm_name),
            }
            for candidate in candidates
        ],
    }


def load_dispatch_context(run_dir: Path) -> dict[str, Any]:
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state under {run_dir}')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    persisted_arm_to_alias = {
        str(key): str(value)
        for key, value in dict(dispatch_state.get('candidate_arm_to_alias') or {}).items()
        if str(key).strip() and str(value).strip()
    }
    candidate_payloads = list(dispatch_state.get('candidate_payloads') or [])
    candidate_index: dict[str, fidelity.CandidateSpec] = {}
    candidate_child_run_names: dict[str, str] = {}
    candidate_source_ranks: dict[str, int] = {}
    candidate_aliases: dict[str, str] = {}
    for payload in candidate_payloads:
        if not isinstance(payload, dict):
            continue
        candidate = reconstruct_candidate(payload)
        candidate_index[candidate.arm_name] = candidate
        candidate_child_run_names[candidate.arm_name] = str(
            payload.get('child_run_name') or build_child_run_name(run_dir.name, candidate.arm_name)
        )
        candidate_source_ranks[candidate.arm_name] = int(payload.get('source_rank', 999))
        candidate_aliases[candidate.arm_name] = str(
            payload.get('candidate_alias')
            or persisted_arm_to_alias.get(candidate.arm_name)
            or candidate.arm_name
        )
    tasks = dispatch_state.get('formal', {}).get('tasks', {})
    task_index = {
        str(task.get('candidate_arm')): dict(task)
        for task in tasks.values()
        if isinstance(task, dict) and task.get('candidate_arm')
    }
    return {
        'run_name': run_dir.name,
        'run_dir': run_dir,
        'dispatch_state_path': dispatch_state_path,
        'dispatch_state': dispatch_state,
        'source_run_name': str(dispatch_state.get('source_run_name') or ''),
        'source_seed': int(dispatch_state.get('source_seed', 0)),
        'selected_protocol_arm': str(dispatch_state.get('selected_protocol_arm') or ''),
        'selected_protocol_arms': list(dispatch_state.get('selected_protocol_arms') or []),
        'source_refine_front_runner': str(dispatch_state.get('source_refine_front_runner') or ''),
        'formal_seed': int(dispatch_state.get('formal_seed', 0)),
        'formal_step_scale': float(dispatch_state.get('formal_step_scale', DEFAULT_FORMAL_STEP_SCALE)),
        'candidate_index': candidate_index,
        'candidate_child_run_names': candidate_child_run_names,
        'candidate_source_ranks': candidate_source_ranks,
        'candidate_aliases': candidate_aliases,
        'task_index': task_index,
    }


def initialize_dispatch_state(
    *,
    run_name: str,
    source_context: dict[str, Any],
    local_label: str,
    remote_label: str | None,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for payload in source_context['candidate_payloads']:
        candidate_arm = str(payload['arm_name'])
        task_id = build_task_id(candidate_arm=candidate_arm)
        tasks[task_id] = {
            'task_id': task_id,
            'candidate_arm': candidate_arm,
            'candidate_alias': str(payload.get('candidate_alias') or candidate_arm),
            'child_run_name': str(payload['child_run_name']),
            'source_rank': int(payload.get('source_rank', 999)),
            'status': 'pending',
            'attempts': 0,
        }
    return {
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': ROUND_KIND_FORMAL,
        'run_name': run_name,
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'status': 'running',
        'stage': 'formal',
        'local_label': local_label,
        'remote_label': remote_label,
        'source_run_name': source_context['source_run_name'],
        'source_seed': int(source_context['source_seed']),
        'selected_protocol_arm': source_context['selected_protocol_arm'],
        'selected_protocol_arms': list(source_context['selected_protocol_arms']),
        'source_refine_front_runner': source_context['source_refine_front_runner'],
        'formal_seed': int(source_context['formal_seed']),
        'formal_step_scale': float(source_context['formal_step_scale']),
        'candidate_alias_to_arm': dict(source_context.get('candidate_alias_to_arm') or {}),
        'candidate_arm_to_alias': dict(source_context.get('candidate_arm_to_alias') or {}),
        'candidate_payloads': list(source_context['candidate_payloads']),
        'formal': {
            'stage_name': 'formal',
            'task_count': len(tasks),
            'tasks': tasks,
        },
        'child_run_names': [str(payload['child_run_name']) for payload in source_context['candidate_payloads']],
        'completed_children': [],
    }


def validate_completed_formal_child_run(*, child_run_dir: Path) -> dict[str, Any]:
    state_path = child_run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing child state.json at {state_path}')
    state = fidelity.load_json(state_path)
    formal_state = state.get('formal')
    if not isinstance(formal_state, dict):
        raise RuntimeError('child run is missing formal state')
    if formal_state.get('status') != 'completed':
        raise RuntimeError(f'child run formal status is `{formal_state.get("status")}`')
    result = formal_state.get('result')
    if not isinstance(result, dict):
        raise RuntimeError('child run formal result payload is missing')
    shortlist = formal.build_formal_shortlist_candidates(result)
    for checkpoint_type, payload in shortlist.items():
        checkpoint_path = Path(str(payload.get('path', '') or '')).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'child run shortlist checkpoint `{checkpoint_type}` does not exist: {checkpoint_path}'
            )
    return state


def rewrite_repo_paths(value: Any, *, remote_repo: Path, local_repo: Path) -> Any:
    if isinstance(value, dict):
        return {
            str(key): rewrite_repo_paths(item, remote_repo=remote_repo, local_repo=local_repo)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_repo_paths(item, remote_repo=remote_repo, local_repo=local_repo) for item in value]
    if isinstance(value, str):
        try:
            remote_path = Path(value)
            relative = remote_path.relative_to(remote_repo)
        except Exception:
            return value
        return str((local_repo / relative).resolve())
    return value


def rewrite_child_state_paths_to_local_repo(
    *,
    child_run_dir: Path,
    remote_repo: Path,
    local_repo: Path,
) -> None:
    state_path = child_run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing synced child state at {state_path}')
    state = fidelity.load_json(state_path)
    rewritten = rewrite_repo_paths(state, remote_repo=remote_repo, local_repo=local_repo)
    fidelity.atomic_write_json(state_path, rewritten)


def build_child_run_state(
    *,
    coordinator_run_name: str,
    source_run_name: str,
    source_seed: int,
    selected_protocol_arm: str,
    selected_protocol_arms: list[str],
    source_refine_front_runner: str,
    candidate: fidelity.CandidateSpec,
    candidate_alias: str,
    child_run_name: str,
    source_rank: int,
) -> dict[str, Any]:
    return {
        'seed': int(source_seed),
        'status': 'running_formal_train',
        'selected_protocol_arms': list(selected_protocol_arms),
        'manual_formal_playoff': {
            'coordinator_run_name': coordinator_run_name,
            'source_run_name': source_run_name,
            'child_run_name': child_run_name,
            'candidate_arm': candidate.arm_name,
            'candidate_alias': candidate_alias,
            'source_rank': int(source_rank),
        },
        'p1': {
            'selected_protocol_arm': selected_protocol_arm,
            'winner_refine_front_runner': source_refine_front_runner,
            'winner': candidate.arm_name,
            'winner_source': 'winner_refine_triplet_formal_playoff',
            'manual_formal_candidate': {
                'arm_name': candidate.arm_name,
                'alias': candidate_alias,
                'source_run_name': source_run_name,
                'source_rank': int(source_rank),
                'protocol_arm': str(candidate.meta.get('protocol_arm', selected_protocol_arm)),
            },
        },
        'formal': {'status': 'running'},
        'formal_1v3': {'status': 'pending'},
        'final_conclusion': {
            'p1_protocol_winner': selected_protocol_arm,
            'p1_refine_front_runner': source_refine_front_runner,
            'p1_winner': candidate.arm_name,
            'p1_winner_source': 'winner_refine_triplet_formal_playoff',
            'formal_train_status': 'running',
            'formal_1v3_status': 'pending',
            'formal_status': 'running',
        },
    }


@contextmanager
def patched_formal_defaults(overrides: dict[str, int | None]):
    original_screening = dict(ab.BASE_SCREENING)
    original_formal_defaults = deepcopy(formal.FORMAL_DEFAULTS)
    patched_defaults = deepcopy(formal.FORMAL_DEFAULTS)
    for key, value in overrides.items():
        if value is None:
            continue
        patched_defaults[key] = int(value)
    formal.FORMAL_DEFAULTS.clear()
    formal.FORMAL_DEFAULTS.update(patched_defaults)
    try:
        yield
    finally:
        formal.FORMAL_DEFAULTS.clear()
        formal.FORMAL_DEFAULTS.update(original_formal_defaults)
        ab.BASE_SCREENING.clear()
        ab.BASE_SCREENING.update(original_screening)


def execute_single_task(
    *,
    run_name: str,
    candidate_arm: str,
    result_json: Path,
    machine_label: str,
    num_workers: int | None = None,
    file_batch_size: int | None = None,
    prefetch_factor: int | None = None,
    val_file_batch_size: int | None = None,
    val_prefetch_factor: int | None = None,
) -> dict[str, Any]:
    run_dir = fidelity.FIDELITY_ROOT / run_name
    context = load_dispatch_context(run_dir)
    candidate = context['candidate_index'].get(candidate_arm)
    if candidate is None:
        raise KeyError(f'unknown formal candidate `{candidate_arm}`')
    candidate_alias = str(context.get('candidate_aliases', {}).get(candidate_arm) or candidate_arm)
    child_run_name = context['candidate_child_run_names'][candidate_arm]
    child_run_dir = fidelity.FIDELITY_ROOT / child_run_name
    remove_tree_if_exists(child_run_dir)
    child_run_dir.mkdir(parents=True, exist_ok=True)
    child_run_lock = fidelity.acquire_run_lock(child_run_dir, child_run_name)
    try:
        ab_dir = ab.AB_ROOT / f'{child_run_name}_formal'
        remove_tree_if_exists(ab_dir)
        source_rank = int(context['candidate_source_ranks'].get(candidate_arm, 999))
        state = build_child_run_state(
            coordinator_run_name=run_name,
            source_run_name=context['source_run_name'],
            source_seed=int(context['source_seed']),
            selected_protocol_arm=context['selected_protocol_arm'],
            selected_protocol_arms=list(context['selected_protocol_arms']),
            source_refine_front_runner=context['source_refine_front_runner'],
            candidate=candidate,
            candidate_alias=candidate_alias,
            child_run_name=child_run_name,
            source_rank=source_rank,
        )
        fidelity.atomic_write_json(child_run_dir / 'state.json', state)

        overrides = {
            'num_workers': num_workers,
            'file_batch_size': file_batch_size,
            'prefetch_factor': prefetch_factor,
            'val_file_batch_size': val_file_batch_size,
            'val_prefetch_factor': val_prefetch_factor,
        }
        with patched_formal_defaults(overrides):
            formal.apply_formal_defaults()
            base_cfg = ab.build_base_config()
            grouped = ab.group_files_by_month(ab.load_all_files())
            merged_cfg = ab.merge_dict(base_cfg, candidate.cfg_overrides)
            protocol_arm = str(candidate.meta.get('protocol_arm', candidate.arm_name))
            ab_name = f'{child_run_name}_formal'
            try:
                result = ab.run_ab6_checkpoint(
                    merged_cfg,
                    grouped,
                    seed=int(context['formal_seed']),
                    scheduler_profile=candidate.scheduler_profile,
                    curriculum_profile=candidate.curriculum_profile,
                    weight_profile=candidate.weight_profile,
                    window_profile=candidate.window_profile,
                    step_scale=float(context['formal_step_scale']),
                    ab_name=ab_name,
                )
                result = formal.finalize_formal_result(
                    merged_cfg,
                    result,
                    protocol_arm=protocol_arm,
                )
                shortlist_types = list(result.get('shortlist_checkpoint_types') or [])
                checkpoint_pack_types = list(result.get('checkpoint_pack_types') or shortlist_types)
                state['formal'] = {
                    'status': 'completed',
                    'ab_name': ab_name,
                    'offline_checkpoint_winner': result.get('offline_checkpoint_winner'),
                    'shortlist_checkpoint_types': shortlist_types,
                    'checkpoint_pack_types': checkpoint_pack_types,
                    'result': result,
                }
                state['formal_1v3'] = {
                    'status': 'pending',
                    'shortlist_checkpoint_types': shortlist_types,
                    'checkpoint_pack_types': checkpoint_pack_types,
                    'candidate_count': len(checkpoint_pack_types),
                }
                state.setdefault('final_conclusion', {})['formal_train_status'] = 'completed'
                state['final_conclusion']['formal_1v3_status'] = 'pending'
                state['final_conclusion']['formal_status'] = 'pending_1v3'
            except Exception as exc:
                state['formal'] = {
                    'status': 'failed',
                    'ab_name': ab_name,
                    'error': str(exc),
                    'traceback': traceback.format_exc(),
                }
                state['formal_1v3'] = {'status': 'blocked_by_formal_failure'}
                state.setdefault('final_conclusion', {})['formal_train_status'] = 'failed'
                state['final_conclusion']['formal_1v3_status'] = 'blocked_by_formal_failure'
                state['final_conclusion']['formal_status'] = 'failed'
            fidelity.atomic_write_json(child_run_dir / 'state.json', state)

        validated_state = validate_completed_formal_child_run(child_run_dir=child_run_dir)
        formal_state = validated_state['formal']
        result = formal_state['result']
        payload = {
            'schema_version': TASK_RESULT_SCHEMA_VERSION,
            'round_kind': ROUND_KIND_FORMAL,
            'run_name': run_name,
            'source_run_name': context['source_run_name'],
            'candidate_arm': candidate.arm_name,
            'candidate_alias': candidate_alias,
            'child_run_name': child_run_name,
            'child_run_dir': str(child_run_dir.resolve()),
            'ab_name': formal_state['ab_name'],
            'ab_dir': str((ab.AB_ROOT / formal_state['ab_name']).resolve()),
            'machine_label': machine_label,
            'source_rank': source_rank,
            'formal_seed': int(context['formal_seed']),
            'formal_step_scale': float(context['formal_step_scale']),
            'offline_checkpoint_winner': result.get('offline_checkpoint_winner'),
            'shortlist_checkpoint_types': list(formal_state.get('shortlist_checkpoint_types') or []),
            'completed_at': fidelity.ts_now(),
        }
        result_json.parent.mkdir(parents=True, exist_ok=True)
        fidelity.atomic_write_json(result_json, payload)
        return payload
    finally:
        fidelity.release_run_lock(child_run_lock)


def build_run_task_cli_summary(payload: dict[str, Any], *, result_json: Path) -> dict[str, Any]:
    return {
        'status': 'ok',
        'round_kind': payload.get('round_kind'),
        'run_name': payload.get('run_name'),
        'source_run_name': payload.get('source_run_name'),
        'candidate_arm': payload.get('candidate_arm'),
        'candidate_alias': payload.get('candidate_alias'),
        'child_run_name': payload.get('child_run_name'),
        'machine_label': payload.get('machine_label'),
        'offline_checkpoint_winner': payload.get('offline_checkpoint_winner'),
        'completed_at': payload.get('completed_at'),
        'result_json': str(result_json),
    }


def load_task_result(path: Path) -> dict[str, Any]:
    payload = fidelity.load_json(path)
    if str(payload.get('round_kind') or '') != ROUND_KIND_FORMAL:
        raise RuntimeError(f'task result at {path} is not valid for formal dispatch')
    for key in ('run_name', 'candidate_arm', 'child_run_name', 'child_run_dir', 'ab_dir'):
        if not str(payload.get(key) or '').strip():
            raise RuntimeError(f'task result at {path} is missing `{key}`')
    return payload


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


def map_repo_path_to_remote(local_path: str | Path, *, remote_repo: str | Path) -> Path:
    resolved = Path(local_path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f'path {resolved} is outside repo root {REPO_ROOT}') from exc
    return Path(remote_repo) / relative


def map_remote_repo_path_to_local(remote_path: str | Path, *, remote_repo: str | Path) -> Path:
    path = Path(str(remote_path))
    try:
        relative = path.relative_to(Path(remote_repo))
    except ValueError as exc:
        raise RuntimeError(f'remote path {path} is outside remote repo {remote_repo}') from exc
    return (REPO_ROOT / relative).resolve()


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


def fetch_remote_tree(worker: WorkerSpec, *, remote_path: Path, local_path: Path) -> None:
    remove_tree_if_exists(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = ['scp']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.extend(['-r', f'{worker.host}:{path_to_scp_remote(remote_path)}', str(local_path.parent)])
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
            f'failed to sync remote tree {remote_path} -> {local_path}: {completed.stdout.strip()}'
        )


def sync_remote_dispatch_state(*, worker: WorkerSpec, dispatch_state_path: Path) -> Path:
    remote_path = map_repo_path_to_remote(dispatch_state_path, remote_repo=worker.repo or str(REPO_ROOT))
    copy_local_file_to_remote(worker, local_path=dispatch_state_path, remote_path=remote_path)
    return remote_path


def sync_remote_task_outputs(*, worker: WorkerSpec, task_payload: dict[str, Any]) -> None:
    remote_repo = Path(worker.repo or str(REPO_ROOT))
    remote_child_run_dir = Path(str(task_payload['child_run_dir']))
    remote_ab_dir = Path(str(task_payload['ab_dir']))
    local_child_run_dir = map_remote_repo_path_to_local(remote_child_run_dir, remote_repo=remote_repo)
    local_ab_dir = map_remote_repo_path_to_local(remote_ab_dir, remote_repo=remote_repo)
    fetch_remote_tree(worker, remote_path=remote_child_run_dir, local_path=local_child_run_dir)
    fetch_remote_tree(worker, remote_path=remote_ab_dir, local_path=local_ab_dir)
    rewrite_child_state_paths_to_local_repo(
        child_run_dir=local_child_run_dir,
        remote_repo=remote_repo,
        local_repo=REPO_ROOT.resolve(),
    )
    validate_completed_formal_child_run(child_run_dir=local_child_run_dir)


def build_task_command_args(
    *,
    run_name: str,
    task_state: dict[str, Any],
    machine_label: str,
    formal_overrides: dict[str, int] | None = None,
) -> list[str]:
    command_args = [
        'run-task',
        '--run-name',
        run_name,
        '--candidate-arm',
        str(task_state['candidate_arm']),
        '--machine-label',
        machine_label,
    ]
    if formal_overrides:
        command_args.extend(
            [
                '--num-workers',
                str(formal_overrides['num_workers']),
                '--file-batch-size',
                str(formal_overrides['file_batch_size']),
                '--prefetch-factor',
                str(formal_overrides['prefetch_factor']),
                '--val-file-batch-size',
                str(formal_overrides['val_file_batch_size']),
                '--val-prefetch-factor',
                str(formal_overrides['val_prefetch_factor']),
            ]
        )
    return command_args


def build_remote_interactive_window_command(
    *,
    worker: WorkerSpec,
    run_name: str,
    task_state: dict[str, Any],
    remote_result_path: Path,
    remote_runtime_root: Path,
    formal_overrides: dict[str, int] | None,
) -> list[str]:
    python_args = build_task_command_args(
        run_name=run_name,
        task_state=task_state,
        machine_label=worker.label,
        formal_overrides=formal_overrides,
    )
    python_args.extend(
        [
            '--result-json',
            str(remote_result_path),
        ]
    )
    repo_root = Path(worker.repo or str(REPO_ROOT))
    window_title = f'MahjongAI formal {task_state["task_id"]}'
    python_args_base64 = base64.b64encode(json.dumps(python_args, ensure_ascii=True).encode('utf-8')).decode('ascii')
    ps_command = (
        f"if (Test-Path {quote_ps(str(remote_result_path))}) {{ Remove-Item -LiteralPath {quote_ps(str(remote_result_path))} -Force }}; "
        f"& {quote_ps(str(repo_root / 'scripts' / 'start_interactive_remote_python.ps1'))} "
        f"-RepoRoot {quote_ps(str(repo_root))} "
        f"-PythonExe {quote_ps(worker.python or sys.executable)} "
        f"-PythonScript {quote_ps(str(repo_root / 'mortal' / SCRIPT_PATH.name))} "
        f"-PythonArgsBase64 {quote_ps(python_args_base64)} "
        f"-TaskId {quote_ps(str(task_state['task_id']))} "
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
    worker: WorkerSpec,
    *,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    launch_mode: str,
    formal_overrides: dict[str, int] | None,
) -> ActiveTask:
    remote_results_dir = dispatch_root / 'remote_results'
    remote_runtime_dir = dispatch_root / 'remote_runtime' / str(task_state['task_id'])
    local_results_dir = ensure_dir(dispatch_root / 'results')
    logs_dir = ensure_dir(dispatch_root / 'logs')
    remote_result_path = remote_results_dir / f'{task_state["task_id"]}.json'
    local_result_path = local_results_dir / f'{task_state["task_id"]}.json'
    log_path = logs_dir / f'{task_state["task_id"]}__{worker.label}.log'
    if local_result_path.exists():
        local_result_path.unlink()
    if launch_mode == 'ssh_inline':
        command = dispatch.build_remote_python_command(
            worker=worker,
            script_path=SCRIPT_PATH,
            remote_result_path=remote_result_path,
            command_args=build_task_command_args(
                run_name=run_name,
                task_state=task_state,
                machine_label=worker.label,
                formal_overrides=formal_overrides,
            ),
        )
    elif launch_mode == 'interactive_window':
        command = build_remote_interactive_window_command(
            worker=worker,
            run_name=run_name,
            task_state=task_state,
            remote_result_path=remote_result_path,
            remote_runtime_root=remote_runtime_dir,
            formal_overrides=formal_overrides,
        )
    else:
        raise ValueError(f'unsupported remote launch mode `{launch_mode}`')
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    task_state['remote_runtime_root'] = str(remote_runtime_dir)
    task_state['remote_task_name'] = f'{REMOTE_INTERACTIVE_TASK_NAME_PREFIX}{task_state["task_id"]}'
    task_state['started_at'] = fidelity.ts_now()
    return ActiveTask(
        worker=worker,
        stage_name='formal',
        task_id=str(task_state['task_id']),
        task_state=task_state,
        process=process,
        log_path=log_path,
        local_result_path=local_result_path,
        remote_result_path=remote_result_path,
    )


def launch_local_task(
    worker: WorkerSpec,
    *,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
) -> ActiveTask:
    results_dir = ensure_dir(dispatch_root / 'results')
    logs_dir = ensure_dir(dispatch_root / 'logs')
    result_path = results_dir / f'{task_state["task_id"]}.json'
    log_path = logs_dir / f'{task_state["task_id"]}__{worker.label}.log'
    if result_path.exists():
        result_path.unlink()
    spec = JsonTaskLaunchSpec(
        task_id=str(task_state['task_id']),
        stage_name='formal',
        local_result_path=result_path,
        log_path=log_path,
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


def launch_task_for_worker(
    *,
    worker: WorkerSpec,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    control_state: dict[str, Any],
    dispatch_state_path: Path,
    remote_formal_overrides: dict[str, int],
) -> ActiveTask:
    if worker.kind == 'local':
        return launch_local_task(
            worker,
            run_name=run_name,
            task_state=task_state,
            dispatch_root=dispatch_root,
        )
    sync_remote_dispatch_state(worker=worker, dispatch_state_path=dispatch_state_path)
    worker_control = common_dispatch.worker_control_entry(control_state, worker.label)
    return launch_remote_task(
        worker,
        run_name=run_name,
        task_state=task_state,
        dispatch_root=dispatch_root,
        launch_mode=str(worker_control.get('launch_mode') or common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE),
        formal_overrides=remote_formal_overrides,
    )


def handle_finished_task(
    *,
    active: ActiveTask,
    max_attempts: int,
) -> None:
    task_state = active.task_state
    return_code = active.process.returncode
    if return_code != 0:
        dispatch.mark_task_failed(
            task_state,
            f'worker `{active.worker.label}` exited with code {return_code}; see {active.log_path}',
            max_attempts=max_attempts,
            finished_at=fidelity.ts_now(),
        )
        return
    local_result_path = active.local_result_path
    try:
        if active.worker.kind == 'remote':
            if active.remote_result_path is None:
                raise ValueError('remote active task requires remote_result_path')
            dispatch.fetch_remote_result(active.worker, active.remote_result_path, local_result_path)
        if not local_result_path.exists():
            raise FileNotFoundError(f'missing task result json at {local_result_path}')
        payload = load_task_result(local_result_path)
        if active.worker.kind == 'remote':
            sync_remote_task_outputs(worker=active.worker, task_payload=payload)
        else:
            validate_completed_formal_child_run(
                child_run_dir=Path(str(payload['child_run_dir'])).resolve()
            )
        task_state['child_run_name'] = str(payload['child_run_name'])
        task_state['offline_checkpoint_winner'] = str(payload.get('offline_checkpoint_winner') or '')
    except Exception as exc:
        if local_result_path.exists():
            try:
                local_result_path.unlink()
            except OSError:
                pass
        dispatch.mark_task_failed(
            task_state,
            f'worker `{active.worker.label}` result handling failed: {exc}',
            max_attempts=max_attempts,
            finished_at=fidelity.ts_now(),
        )
        return
    task_state['status'] = 'completed'
    task_state['finished_at'] = fidelity.ts_now()
    task_state.pop('error', None)


def reset_running_tasks_for_resume(dispatch_state: dict[str, Any]) -> None:
    stage_state = dispatch_state.get('formal')
    if not isinstance(stage_state, dict):
        return
    for task in stage_state.get('tasks', {}).values():
        if str(task.get('status')) == 'running':
            task['status'] = 'pending'
            task.pop('started_at', None)
            task.pop('worker_label', None)
            task.pop('local_result_path', None)
            task.pop('remote_result_path', None)
            task.pop('log_path', None)


def run_dispatch(args: argparse.Namespace) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fidelity.acquire_run_lock(run_dir, args.run_name)
    atexit.register(fidelity.release_run_lock, lock_path)
    dispatch_root = ensure_dir(dispatch_root_for_run(run_dir))
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    dispatch_control_path = dispatch_control_path_for_run(run_dir)
    try:
        if dispatch_state_path.exists():
            dispatch_state = load_dispatch_state(dispatch_state_path)
            reset_running_tasks_for_resume(dispatch_state)
        else:
            candidate_arms = dedupe_candidate_arms(list(args.candidate_arm or []))
            if not candidate_arms:
                raise RuntimeError('formal dispatch requires at least one --candidate-arm')
            source_run_dir = fidelity.FIDELITY_ROOT / args.source_run_name
            source_context = load_source_context(
                source_run_dir=source_run_dir,
                coordinator_run_name=args.run_name,
                candidate_arms=candidate_arms,
                formal_seed_offset=args.seed_offset,
                formal_step_scale=args.formal_step_scale,
            )
            dispatch_state = initialize_dispatch_state(
                run_name=args.run_name,
                source_context=source_context,
                local_label=args.local_label,
                remote_label=None if args.local_only else args.remote_label,
            )
            write_dispatch_state(dispatch_state_path, dispatch_state)
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
        if dispatch_control_path.exists():
            control_state = common_dispatch.load_dispatch_control(dispatch_control_path)
        else:
            control_state = common_dispatch.initialize_dispatch_control_state(
                local_label=args.local_label,
                remote_label=None if args.local_only else args.remote_label,
                remote_launch_mode=args.remote_launch_mode,
            )
            common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
        if common_dispatch.ensure_control_state_workers(
            control_state=control_state,
            local_label=args.local_label,
            remote_label=None if args.local_only else args.remote_label,
            remote_launch_mode=args.remote_launch_mode,
        ):
            common_dispatch.write_dispatch_control(dispatch_control_path, control_state)

        remote_formal_overrides = {
            'num_workers': int(args.remote_num_workers),
            'file_batch_size': int(args.remote_file_batch_size),
            'prefetch_factor': int(args.remote_prefetch_factor),
            'val_file_batch_size': int(args.remote_val_file_batch_size),
            'val_prefetch_factor': int(args.remote_val_prefetch_factor),
        }
        active: dict[str, ActiveTask] = {}
        while True:
            control_state = common_dispatch.load_dispatch_control(dispatch_control_path)
            if common_dispatch.ensure_control_state_workers(
                control_state=control_state,
                local_label=args.local_label,
                remote_label=None if args.local_only else args.remote_label,
                remote_launch_mode=args.remote_launch_mode,
            ):
                common_dispatch.write_dispatch_control(dispatch_control_path, control_state)
            if common_dispatch.apply_worker_control_requests(control_state=control_state, active=active):
                write_dispatch_state(dispatch_state_path, dispatch_state)
                common_dispatch.write_dispatch_control(dispatch_control_path, control_state)

            for worker_label, active_task in list(active.items()):
                if active_task.process.poll() is None:
                    continue
                handle_finished_task(active=active_task, max_attempts=args.max_attempts)
                active.pop(worker_label, None)
                write_dispatch_state(dispatch_state_path, dispatch_state)

            stage_state = dispatch_state['formal']
            if common_dispatch.stage_any_task_failed(stage_state):
                dispatch_state['status'] = 'failed'
                dispatch_state['stage'] = 'failed'
                write_dispatch_state(dispatch_state_path, dispatch_state)
                raise RuntimeError('formal dispatch exhausted retries')
            if common_dispatch.stage_all_tasks_completed(stage_state):
                dispatch_state['status'] = 'completed'
                dispatch_state['stage'] = 'completed'
                dispatch_state['completed_children'] = [
                    {
                        'candidate_arm': str(task.get('candidate_arm') or ''),
                        'candidate_alias': str(task.get('candidate_alias') or task.get('candidate_arm') or ''),
                        'child_run_name': str(task.get('child_run_name') or ''),
                        'offline_checkpoint_winner': str(task.get('offline_checkpoint_winner') or ''),
                    }
                    for task in stage_state.get('tasks', {}).values()
                ]
                write_dispatch_state(dispatch_state_path, dispatch_state)
                break

            for worker in workers:
                if worker.label in active:
                    continue
                worker_control = common_dispatch.worker_control_entry(control_state, worker.label)
                if bool(worker_control.get('paused')):
                    continue
                pending = common_dispatch.find_next_pending_task(stage_state)
                if pending is None:
                    break
                _, task_state = pending
                active[worker.label] = launch_task_for_worker(
                    worker=worker,
                    run_name=args.run_name,
                    task_state=task_state,
                    dispatch_root=dispatch_root,
                    control_state=control_state,
                    dispatch_state_path=dispatch_state_path,
                    remote_formal_overrides=remote_formal_overrides,
                )
                write_dispatch_state(dispatch_state_path, dispatch_state)

            if not active and common_dispatch.find_next_pending_task(stage_state) is None:
                dispatch_state['status'] = 'completed'
                dispatch_state['stage'] = 'completed'
                write_dispatch_state(dispatch_state_path, dispatch_state)
                break
            time.sleep(args.poll_seconds)
    finally:
        fidelity.release_run_lock(lock_path)
    return 0


def print_status(args: argparse.Namespace) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state for run `{args.run_name}`')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    dispatch_control = dispatch_control_path_for_run(run_dir)
    control_state = common_dispatch.load_dispatch_control(dispatch_control) if dispatch_control.exists() else {'workers': {}}
    tasks_payload = []
    for task in dispatch_state.get('formal', {}).get('tasks', {}).values():
        if not isinstance(task, dict):
            continue
        child_run_name = str(task.get('child_run_name') or '')
        child_state_path = fidelity.FIDELITY_ROOT / child_run_name / 'state.json'
        formal_status = None
        offline_checkpoint_winner = None
        if child_state_path.exists():
            try:
                child_state = fidelity.load_json(child_state_path)
                formal_state = child_state.get('formal')
                if isinstance(formal_state, dict):
                    formal_status = formal_state.get('status')
                    offline_checkpoint_winner = formal_state.get('offline_checkpoint_winner')
            except Exception:
                formal_status = 'unreadable'
        tasks_payload.append(
            {
                'candidate_arm': task.get('candidate_arm'),
                'candidate_alias': task.get('candidate_alias'),
                'child_run_name': child_run_name,
                'status': task.get('status'),
                'attempts': task.get('attempts'),
                'worker_label': task.get('worker_label'),
                'formal_status': formal_status,
                'offline_checkpoint_winner': offline_checkpoint_winner,
            }
        )
    payload = {
        'run_name': args.run_name,
        'source_run_name': dispatch_state.get('source_run_name'),
        'selected_protocol_arm': dispatch_state.get('selected_protocol_arm'),
        'formal_seed': dispatch_state.get('formal_seed'),
        'formal_step_scale': dispatch_state.get('formal_step_scale'),
        'status': dispatch_state.get('status'),
        'stage': dispatch_state.get('stage'),
        'candidate_alias_to_arm': dispatch_state.get('candidate_alias_to_arm', {}),
        'tasks': common_dispatch.summarize_dispatch_task_status(dispatch_state.get('formal', {})),
        'workers': control_state.get('workers', {}),
        'task_details': tasks_payload,
    }
    print(json.dumps(fidelity.normalize_payload(payload), ensure_ascii=False, indent=2))
    return 0


def update_worker_pause(args: argparse.Namespace, *, paused: bool) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_control = dispatch_control_path_for_run(run_dir)
    if dispatch_control.exists():
        control_state = common_dispatch.load_dispatch_control(dispatch_control)
    else:
        control_state = common_dispatch.initialize_dispatch_control_state(
            local_label=common_dispatch.DEFAULT_LOCAL_LABEL,
            remote_label=common_dispatch.DEFAULT_REMOTE_LABEL,
            remote_launch_mode=common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE,
        )
    common_dispatch.ensure_control_state_workers(
        control_state=control_state,
        local_label=common_dispatch.DEFAULT_LOCAL_LABEL,
        remote_label=common_dispatch.DEFAULT_REMOTE_LABEL,
        remote_launch_mode=common_dispatch.DEFAULT_REMOTE_LAUNCH_MODE,
    )
    entry = common_dispatch.set_worker_pause(
        control_state,
        worker_label=args.worker_label,
        paused=paused,
        stop_active=bool(getattr(args, 'stop_active', False)),
    )
    common_dispatch.write_dispatch_control(dispatch_control, control_state)
    print(json.dumps({'worker_label': args.worker_label, 'paused': entry['paused']}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    dispatch_cmd = subparsers.add_parser('dispatch')
    dispatch_cmd.add_argument('--run-name', required=True)
    dispatch_cmd.add_argument('--source-run-name', required=True)
    dispatch_cmd.add_argument('--candidate-arm', action='append', required=True)
    dispatch_cmd.add_argument('--seed-offset', type=int, default=DEFAULT_FORMAL_SEED_OFFSET)
    dispatch_cmd.add_argument('--formal-step-scale', type=float, default=DEFAULT_FORMAL_STEP_SCALE)
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
    dispatch_cmd.add_argument('--remote-num-workers', type=int, default=DEFAULT_REMOTE_FORMAL_NUM_WORKERS)
    dispatch_cmd.add_argument('--remote-file-batch-size', type=int, default=DEFAULT_REMOTE_FORMAL_FILE_BATCH_SIZE)
    dispatch_cmd.add_argument('--remote-prefetch-factor', type=int, default=DEFAULT_REMOTE_FORMAL_PREFETCH_FACTOR)
    dispatch_cmd.add_argument('--remote-val-file-batch-size', type=int, default=DEFAULT_REMOTE_FORMAL_VAL_FILE_BATCH_SIZE)
    dispatch_cmd.add_argument('--remote-val-prefetch-factor', type=int, default=DEFAULT_REMOTE_FORMAL_VAL_PREFETCH_FACTOR)

    run_task = subparsers.add_parser('run-task')
    run_task.add_argument('--run-name', required=True)
    run_task.add_argument('--candidate-arm', required=True)
    run_task.add_argument('--machine-label', required=True)
    run_task.add_argument('--num-workers', type=int)
    run_task.add_argument('--file-batch-size', type=int)
    run_task.add_argument('--prefetch-factor', type=int)
    run_task.add_argument('--val-file-batch-size', type=int)
    run_task.add_argument('--val-prefetch-factor', type=int)
    run_task.add_argument('--result-json', required=True)

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
        result_json = Path(args.result_json).resolve()
        payload = execute_single_task(
            run_name=args.run_name,
            candidate_arm=args.candidate_arm,
            result_json=result_json,
            machine_label=args.machine_label,
            num_workers=args.num_workers,
            file_batch_size=args.file_batch_size,
            prefetch_factor=args.prefetch_factor,
            val_file_batch_size=args.val_file_batch_size,
            val_prefetch_factor=args.val_prefetch_factor,
        )
        print(
            json.dumps(
                fidelity.normalize_payload(build_run_task_cli_summary(payload, result_json=result_json)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == 'dispatch':
        sys.exit(run_dispatch(args))
    if args.command == 'status':
        sys.exit(print_status(args))
    if args.command == 'pause-worker':
        sys.exit(update_worker_pause(args, paused=True))
    if args.command == 'resume-worker':
        sys.exit(update_worker_pause(args, paused=False))
    raise RuntimeError(f'unsupported command `{args.command}`')


if __name__ == '__main__':
    main()

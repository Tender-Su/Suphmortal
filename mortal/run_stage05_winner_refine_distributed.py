from __future__ import annotations

import argparse
import atexit
import base64
import json
import math
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import distributed_dispatch as dispatch
import run_stage05_ab as ab
import run_stage05_fidelity as fidelity
import run_stage05_p1_only as p1_only


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
INTERACTIVE_REMOTE_PYTHON_HELPER = REPO_ROOT / 'scripts' / 'start_interactive_remote_python.ps1'
DEFAULT_REMOTE_HOST = 'mahjong-laptop'
DEFAULT_REMOTE_REPO = str(REPO_ROOT)
DEFAULT_REMOTE_PYTHON = r'C:\Users\numbe\miniconda3\envs\mortal\python.exe'
DEFAULT_SSH_KEY = str(Path.home() / '.ssh' / 'mahjong_laptop_ed25519')
DEFAULT_LOCAL_LABEL = 'desktop'
DEFAULT_REMOTE_LABEL = 'laptop'
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_MAX_ATTEMPTS = 2
INTERRUPT_COMMAND_TIMEOUT_SECONDS = 10.0
DEFAULT_SEED2_MIN_KEEP = 4
DEFAULT_SEED2_SELECTION_GAP = 0.001
DEFAULT_SEED2_MAX_KEEP = 12
DEFAULT_REMOTE_LAUNCH_MODE = 'interactive_window'
DEFAULT_REMOTE_SCREENING_NUM_WORKERS = 4
DEFAULT_REMOTE_SCREENING_FILE_BATCH_SIZE = 10
DEFAULT_REMOTE_SCREENING_PREFETCH_FACTOR = 4
DEFAULT_REMOTE_SCREENING_VAL_FILE_BATCH_SIZE = 7
DEFAULT_REMOTE_SCREENING_VAL_PREFETCH_FACTOR = 5
DISPATCH_SCHEMA_VERSION = 1
CONTROL_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA_VERSION = 1
REMOTE_LAUNCH_MODES = frozenset({'ssh_inline', 'interactive_window'})
ROUND_KIND_WINNER_REFINE = 'winner_refine'
ROUND_KIND_PROTOCOL_DECIDE = 'protocol_decide'
ROUND_KIND_ABLATION = 'ablation'
ROUND_KIND_CHOICES = (
    ROUND_KIND_WINNER_REFINE,
    ROUND_KIND_PROTOCOL_DECIDE,
    ROUND_KIND_ABLATION,
)


WorkerSpec = dispatch.WorkerSpec
ActiveTask = dispatch.ActiveTask
JsonTaskLaunchSpec = dispatch.JsonTaskLaunchSpec


REMOTE_INTERACTIVE_TASK_NAME_PREFIX = 'MahjongAI-WinnerRefine-'


def quote_ps(value: str) -> str:
    return dispatch.quote_ps(value)


def path_to_scp_remote(path: str | Path) -> str:
    return dispatch.path_to_scp_remote(path)


def normalize_round_kind(round_kind: str) -> str:
    value = str(round_kind or ROUND_KIND_WINNER_REFINE).strip().lower()
    if value not in ROUND_KIND_CHOICES:
        raise ValueError(f'unsupported round kind `{round_kind}`')
    return value


def dispatch_dir_name(round_kind: str) -> str:
    normalized = normalize_round_kind(round_kind)
    if normalized == ROUND_KIND_PROTOCOL_DECIDE:
        return 'protocol_decide_dispatch'
    if normalized == ROUND_KIND_ABLATION:
        return 'ablation_dispatch'
    return 'winner_refine_dispatch'


def dispatch_root_for_run(run_dir: Path, round_kind: str = ROUND_KIND_WINNER_REFINE) -> Path:
    return run_dir / 'distributed' / dispatch_dir_name(round_kind)


def dispatch_state_path_for_run(run_dir: Path, round_kind: str = ROUND_KIND_WINNER_REFINE) -> Path:
    return dispatch_root_for_run(run_dir, round_kind) / 'dispatch_state.json'


def dispatch_control_path_for_run(run_dir: Path, round_kind: str = ROUND_KIND_WINNER_REFINE) -> Path:
    return dispatch_root_for_run(run_dir, round_kind) / 'dispatch_control.json'


def ensure_dir(path: Path) -> Path:
    return dispatch.ensure_dir(path)


def hostname_fallback() -> str:
    return dispatch.hostname_fallback(DEFAULT_LOCAL_LABEL)


def build_task_id(*, stage_name: str, seed_label: str, arm_name: str) -> str:
    return f'{stage_name}__{seed_label}__{arm_name}'


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


def summarize_dispatch_task_status(stage_state: dict[str, Any]) -> dict[str, int]:
    return dispatch.summarize_task_status(stage_state.get('tasks', {}))


def build_seed_round_name(seed: int, round_kind: str = ROUND_KIND_WINNER_REFINE) -> str:
    normalized = normalize_round_kind(round_kind)
    if normalized == ROUND_KIND_PROTOCOL_DECIDE:
        return f'p1_protocol_decide_round__s{seed}'
    if normalized == ROUND_KIND_ABLATION:
        return f'p1_ablation_round__s{seed}'
    return f'p1_winner_refine_round__s{seed}'


def build_seed_ab_name(run_name: str, seed: int, round_kind: str = ROUND_KIND_WINNER_REFINE) -> str:
    normalized = normalize_round_kind(round_kind)
    if normalized == ROUND_KIND_PROTOCOL_DECIDE:
        return f'{run_name}_p1_protocol_decide_s{seed}'
    if normalized == ROUND_KIND_ABLATION:
        return f'{run_name}_p1_ablation_s{seed}'
    return f'{run_name}_p1_winner_refine_s{seed}'


def final_round_name_for_round_kind(round_kind: str) -> str:
    normalized = normalize_round_kind(round_kind)
    if normalized == ROUND_KIND_PROTOCOL_DECIDE:
        return 'p1_protocol_decide_round'
    if normalized == ROUND_KIND_ABLATION:
        return 'p1_ablation_round'
    return 'p1_winner_refine_round'


def load_dispatch_state(path: Path) -> dict[str, Any]:
    return fidelity.load_json(path)


def write_dispatch_state(path: Path, payload: dict[str, Any]) -> None:
    payload['updated_at'] = fidelity.ts_now()
    fidelity.atomic_write_json(path, payload)


def initialize_dispatch_control_state(
    *,
    local_label: str,
    remote_label: str | None,
    remote_launch_mode: str,
) -> dict[str, Any]:
    workers = {
        local_label: {
            'kind': 'local',
            'paused': False,
            'interrupt_requested': False,
        }
    }
    if remote_label:
        workers[remote_label] = {
            'kind': 'remote',
            'paused': False,
            'interrupt_requested': False,
            'launch_mode': remote_launch_mode,
        }
    return {
        'schema_version': CONTROL_SCHEMA_VERSION,
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'workers': workers,
    }


def load_dispatch_control(path: Path) -> dict[str, Any]:
    return fidelity.load_json(path)


def write_dispatch_control(path: Path, payload: dict[str, Any]) -> None:
    payload['updated_at'] = fidelity.ts_now()
    fidelity.atomic_write_json(path, payload)


def ensure_control_state_workers(
    *,
    control_state: dict[str, Any],
    local_label: str,
    remote_label: str | None,
    remote_launch_mode: str,
) -> bool:
    workers = control_state.setdefault('workers', {})
    changed = False
    local = workers.get(local_label)
    if not isinstance(local, dict):
        workers[local_label] = {
            'kind': 'local',
            'paused': False,
            'interrupt_requested': False,
        }
        changed = True
    else:
        local.setdefault('kind', 'local')
        local.setdefault('paused', False)
        local.setdefault('interrupt_requested', False)
    if remote_label:
        remote = workers.get(remote_label)
        if not isinstance(remote, dict):
            workers[remote_label] = {
                'kind': 'remote',
                'paused': False,
                'interrupt_requested': False,
                'launch_mode': remote_launch_mode,
            }
            changed = True
        else:
            remote.setdefault('kind', 'remote')
            remote.setdefault('paused', False)
            remote.setdefault('interrupt_requested', False)
            remote.setdefault('launch_mode', remote_launch_mode)
    return changed


def worker_control_entry(control_state: dict[str, Any], worker_label: str) -> dict[str, Any]:
    workers = control_state.setdefault('workers', {})
    entry = workers.get(worker_label)
    if not isinstance(entry, dict):
        entry = {
            'paused': False,
            'interrupt_requested': False,
        }
        workers[worker_label] = entry
    entry.setdefault('paused', False)
    entry.setdefault('interrupt_requested', False)
    return entry


def set_worker_pause(
    control_state: dict[str, Any],
    *,
    worker_label: str,
    paused: bool,
    stop_active: bool = False,
) -> dict[str, Any]:
    entry = worker_control_entry(control_state, worker_label)
    entry['paused'] = bool(paused)
    if paused and stop_active:
        entry['interrupt_requested'] = True
    elif not paused:
        entry['interrupt_requested'] = False
    return entry


def update_run_state_for_dispatch(
    *,
    run_dir: Path,
    dispatch_state_path: Path,
    dispatch_state: dict[str, Any],
    round_kind: str = ROUND_KIND_WINNER_REFINE,
    front_runner: str | None = None,
    final_round: dict[str, Any] | None = None,
    status_override: str | None = None,
) -> None:
    normalized_round_kind = normalize_round_kind(round_kind)
    state_path = run_dir / 'state.json'
    state = fidelity.load_json(state_path)
    p1_state = state.setdefault('p1', {})
    final_conclusion = state.setdefault('final_conclusion', {})
    seed1_state = dispatch_state.get('seed1')
    if not isinstance(seed1_state, dict):
        seed1_state = {}
    seed2_state = dispatch_state.get('seed2')
    if not isinstance(seed2_state, dict):
        seed2_state = {}
    seed2_selector = dispatch_state.get('seed2_selector')
    if not isinstance(seed2_selector, dict):
        seed2_selector = {}
    dispatch_summary = {
        'mode': 'desktop_dispatch_plus_optional_ssh_remote',
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': normalized_round_kind,
        'dispatch_state_path': str(dispatch_state_path),
        'stage': dispatch_state.get('stage'),
        'status': dispatch_state.get('status'),
        'seed1_candidate_count': seed1_state.get('candidate_count'),
        'seed2_candidate_count': seed2_state.get('candidate_count'),
        'local_label': dispatch_state.get('local_label'),
        'remote_label': dispatch_state.get('remote_label'),
    }
    if normalized_round_kind == ROUND_KIND_PROTOCOL_DECIDE:
        seed2_plan = dispatch_state.get('seed2_plan')
        if not isinstance(seed2_plan, dict):
            seed2_plan = {}
        dispatch_summary['seed2_probe_arm_names'] = list(seed2_plan.get('probe_candidate_names', []))
        dispatch_summary['seed2_decision_arm_names'] = list(seed2_plan.get('decision_candidate_names', []))
        dispatch_summary['expanded_groups'] = list(seed2_plan.get('expanded_groups', []))
        p1_state['protocol_decide_dispatch'] = dispatch_summary
        if final_round is not None:
            p1_state['protocol_decide_round'] = final_round
        protocol_compare = dispatch_state.get('final_protocol_compare')
        if isinstance(protocol_compare, dict):
            p1_state['protocol_compare'] = protocol_compare
        selected_protocol_arm = str(dispatch_state.get('final_protocol_winner') or front_runner or '').strip()
        if selected_protocol_arm:
            p1_state['selected_protocol_arm'] = selected_protocol_arm
            final_conclusion['p1_protocol_winner'] = selected_protocol_arm
        if status_override:
            state['status'] = status_override
        elif final_round is not None:
            if not p1_only.has_completed_p1_results(state):
                state['status'] = 'stopped_after_p1_protocol_decide'
        else:
            state['status'] = 'running_p1_protocol_decide'
    elif normalized_round_kind == ROUND_KIND_WINNER_REFINE:
        dispatch_summary['seed2_selected_arm_names'] = list(seed2_selector.get('selected_arm_names', []))
        p1_state['winner_refine_dispatch'] = dispatch_summary
        if dispatch_state.get('winner_refine_centers'):
            p1_state['winner_refine_centers'] = list(dispatch_state['winner_refine_centers'])
        if final_round is not None:
            p1_state['winner_refine_round'] = final_round
            p1_state['final_compare'] = {
                'round_name': 'p1_final_compare',
                'ranking': list(final_round.get('ranking', [])),
            }
        if front_runner:
            p1_state['winner_refine_front_runner'] = front_runner
            p1_state['winner'] = front_runner
            p1_state['winner_source'] = 'winner_refine_mainline'
            p1_state['ablation_policy'] = fidelity.P1_ABLATION_POLICY
            final_conclusion['p1_refine_front_runner'] = front_runner
            final_conclusion['p1_winner'] = front_runner
            final_conclusion['p1_winner_source'] = 'winner_refine_mainline'
            final_conclusion['p1_ablation_policy'] = fidelity.P1_ABLATION_POLICY
        if status_override:
            state['status'] = status_override
        elif final_round is not None:
            if state.get('status') != 'completed':
                state['status'] = 'stopped_after_p1_winner_refine'
        else:
            state['status'] = 'running_p1_winner_refine'
    else:
        p1_state['ablation_dispatch'] = dispatch_summary
        if final_round is not None:
            p1_state['ablation_round'] = final_round
            p1_state['final_compare'] = {
                'round_name': 'p1_final_compare',
                'ranking': list(final_round.get('ranking', [])),
            }
        if front_runner:
            p1_state['winner'] = front_runner
            p1_state['winner_source'] = 'ablation_backlog'
            p1_state['ablation_policy'] = fidelity.P1_ABLATION_POLICY
            final_conclusion['p1_winner'] = front_runner
            final_conclusion['p1_winner_source'] = 'ablation_backlog'
            final_conclusion['p1_ablation_policy'] = fidelity.P1_ABLATION_POLICY
        if status_override:
            state['status'] = status_override
        elif final_round is not None:
            state['status'] = 'completed'
        else:
            state['status'] = 'running_p1_ablation'
    fidelity.atomic_write_json(state_path, state)
    fidelity.update_results_doc(run_dir, state)


def load_refine_context(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing state.json under {run_dir}')
    state = fidelity.load_json(state_path)
    p1_state = state.get('p1')
    if not isinstance(p1_state, dict):
        raise RuntimeError('run state has no p1 section')
    calibration = p1_state.get('calibration')
    if not isinstance(calibration, dict):
        raise RuntimeError('p1 calibration is missing; distributed winner_refine requires a completed calibration')
    protocol_decide_round = p1_state.get('protocol_decide_round')
    if not isinstance(protocol_decide_round, dict):
        raise RuntimeError(
            'p1 protocol_decide_round is missing; distributed winner_refine requires a completed protocol_decide'
        )
    seed = p1_only.infer_resume_seed(state)
    if seed is None:
        raise RuntimeError(
            'could not recover the original p1 base seed from state.json; '
            'distributed winner_refine requires a run state with recoverable round seeds'
        )
    protocol_arms = p1_only.dedupe_protocol_arms(
        p1_state.get('protocol_arms') or state.get('selected_protocol_arms') or list(p1_only.FROZEN_TOP3)
    )
    selected_protocol_arm = str(
        p1_state.get('selected_protocol_arm')
        or state.get('final_conclusion', {}).get('p1_protocol_winner')
        or ''
    ).strip()
    if not selected_protocol_arm:
        raise RuntimeError('selected protocol arm is missing from state')
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(grouped, seed + 55, ab.BASE_SCREENING['eval_files'])
    protocols = p1_only.build_protocol_candidates(protocol_arms)
    search_space = p1_only.hydrate_post_protocol_decide_search_space(
        p1_state=p1_state,
        final_conclusion=state.get('final_conclusion') if isinstance(state.get('final_conclusion'), dict) else {},
        require_winner_refine_selection=True,
        require_effective_precision=True,
    )
    winner_center_selection = fidelity.winner_refine_center_selection_from_search_space(
        search_space,
        selected_protocol_arm,
    )
    winner_centers = p1_only.select_protocol_centers(
        protocol_decide_round['ranking'],
        protocol_arm=selected_protocol_arm,
        keep=winner_center_selection['keep'],
        explicit_arm_names=winner_center_selection['explicit_arm_names'],
    )
    persisted_candidates = load_persisted_dispatch_candidates(run_dir, ROUND_KIND_WINNER_REFINE)
    candidates = (
        persisted_candidates
        if persisted_candidates is not None
        else fidelity.build_p1_winner_refine_candidates(
            protocols,
            calibration,
            winner_centers,
            search_space=search_space,
        )
    )
    candidate_index = {candidate.arm_name: candidate for candidate in candidates}
    return {
        'round_kind': ROUND_KIND_WINNER_REFINE,
        'run_dir': run_dir,
        'run_name': run_dir.name,
        'state': state,
        'seed': seed,
        'base_cfg': base_cfg,
        'grouped': grouped,
        'eval_splits': eval_splits,
        'calibration': calibration,
        'protocols': protocols,
        'selected_protocol_arm': selected_protocol_arm,
        'winner_centers': winner_centers,
        'candidates': candidates,
        'candidate_index': candidate_index,
        'seed_base': seed + 606,
        'seed_offsets': list(fidelity.P1_WINNER_REFINE_SEED_OFFSETS),
        'step_scale': fidelity.P1_WINNER_REFINE_STEP_SCALE,
        'ab_name': f'{run_dir.name}_p1_winner_refine',
    }


def load_persisted_dispatch_candidates(
    run_dir: Path,
    round_kind: str,
) -> list[fidelity.CandidateSpec] | None:
    dispatch_path = dispatch_state_path_for_run(run_dir, round_kind)
    if not dispatch_path.exists():
        return None
    dispatch_state = load_dispatch_state(dispatch_path)
    payloads = dispatch_state.get('candidate_payloads')
    if not isinstance(payloads, list):
        return None
    candidates: list[fidelity.CandidateSpec] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        candidates.append(reconstruct_candidate(payload))
    return candidates or None


def load_protocol_decide_context(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing state.json under {run_dir}')
    state = fidelity.load_json(state_path)
    p1_state = state.get('p1')
    if not isinstance(p1_state, dict):
        raise RuntimeError('run state has no p1 section')
    calibration = p1_state.get('calibration')
    if not isinstance(calibration, dict):
        raise RuntimeError('p1 calibration is missing; distributed protocol_decide requires a completed calibration')
    seed = p1_only.infer_resume_seed(state)
    if seed is None:
        raise RuntimeError(
            'could not recover the original p1 base seed from state.json; '
            'distributed protocol_decide requires a run state with recoverable round seeds'
        )
    protocol_arms = p1_only.dedupe_protocol_arms(
        p1_state.get('protocol_arms') or state.get('selected_protocol_arms') or list(p1_only.FROZEN_TOP3)
    )
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(grouped, seed + 55, ab.BASE_SCREENING['eval_files'])
    protocols = p1_only.build_protocol_candidates(protocol_arms)
    search_space = p1_state.get('search_space')
    if not isinstance(search_space, dict):
        search_space = p1_only.build_p1_search_space(calibration)
    else:
        search_space = dict(search_space)
        search_space.pop('winner_refine_centers', None)
        explicit_center_arm_names = [
            str(item).strip()
            for item in search_space.get('winner_refine_center_arm_names', [])
            if str(item).strip()
        ]
        if any(fidelity.is_budget_triplet_arm_name(item) for item in explicit_center_arm_names):
            search_space.pop('winner_refine_center_mode', None)
            search_space.pop('winner_refine_center_protocol_arm', None)
            search_space.pop('winner_refine_center_arm_names', None)
        if (
            str(search_space.get('protocol_decide_progressive_ambiguity_mode', '') or '').strip()
            == fidelity.P1_PROGRESSIVE_AMBIGUITY_MODE_LEGACY
        ):
            search_space.pop('protocol_decide_progressive_ambiguity_mode', None)
            search_space.pop('protocol_decide_progressive_gap_threshold', None)
        search_space.setdefault('protocol_decide_coordinate_mode', fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE)
        search_space.setdefault('budget_ratio_digits', fidelity.P1_BUDGET_RATIO_DIGITS)
        search_space.setdefault('aux_weight_digits', fidelity.P1_AUX_WEIGHT_DIGITS)
    search_space = fidelity.apply_protocol_decide_progressive_settings(search_space)
    persisted_candidates = load_persisted_dispatch_candidates(run_dir, ROUND_KIND_PROTOCOL_DECIDE)
    candidates = (
        persisted_candidates
        if persisted_candidates is not None
        else fidelity.build_p1_protocol_decide_candidates(protocols, calibration, search_space=search_space)
    )
    candidate_index = {candidate.arm_name: candidate for candidate in candidates}
    return {
        'round_kind': ROUND_KIND_PROTOCOL_DECIDE,
        'run_dir': run_dir,
        'run_name': run_dir.name,
        'state': state,
        'seed': seed,
        'base_cfg': base_cfg,
        'grouped': grouped,
        'eval_splits': eval_splits,
        'calibration': calibration,
        'protocols': protocols,
        'search_space': search_space,
        'candidates': candidates,
        'candidate_index': candidate_index,
        'seed_base': seed + 505,
        'seed_offsets': list(fidelity.P1_PROTOCOL_DECIDE_SEED_OFFSETS),
        'step_scale': fidelity.P1_PROTOCOL_DECIDE_STEP_SCALE,
        'ab_name': f'{run_dir.name}_p1_protocol_decide',
    }


def load_ablation_context(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / 'state.json'
    if not state_path.exists():
        raise FileNotFoundError(f'missing state.json under {run_dir}')
    state = fidelity.load_json(state_path)
    p1_state = state.get('p1')
    if not isinstance(p1_state, dict):
        raise RuntimeError('run state has no p1 section')
    calibration = p1_state.get('calibration')
    if not isinstance(calibration, dict):
        raise RuntimeError('p1 calibration is missing; distributed ablation requires a completed calibration')
    winner_refine_round = p1_state.get('winner_refine_round')
    if not isinstance(winner_refine_round, dict):
        raise RuntimeError(
            'p1 winner_refine_round is missing; distributed ablation requires a completed winner_refine'
        )
    seed = p1_only.infer_resume_seed(state)
    if seed is None:
        raise RuntimeError(
            'could not recover the original p1 base seed from state.json; '
            'distributed ablation requires a run state with recoverable round seeds'
        )
    protocol_arms = p1_only.dedupe_protocol_arms(
        p1_state.get('protocol_arms') or state.get('selected_protocol_arms') or list(p1_only.FROZEN_TOP3)
    )
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(grouped, seed + 55, ab.BASE_SCREENING['eval_files'])
    protocols = p1_only.build_protocol_candidates(protocol_arms)
    search_space = p1_only.hydrate_post_protocol_decide_search_space(
        p1_state=p1_state,
        final_conclusion=state.get('final_conclusion') if isinstance(state.get('final_conclusion'), dict) else {},
        require_winner_refine_selection=False,
        require_effective_precision=True,
    )
    valid_refine_entries = [entry for entry in winner_refine_round.get('ranking', []) if entry.get('valid')]
    if not valid_refine_entries:
        raise RuntimeError(
            'p1_winner_refine_round produced no valid candidates; distributed ablation requires a completed winner_refine'
        )
    requested_front_runner = str(
        p1_state.get('winner_refine_front_runner')
        or state.get('final_conclusion', {}).get('p1_refine_front_runner')
        or ''
    ).strip()
    if requested_front_runner:
        refine_entry = next(
            (entry for entry in valid_refine_entries if str(entry.get('arm_name')) == requested_front_runner),
            None,
        )
        if refine_entry is None:
            raise RuntimeError(
                'winner_refine front runner from state.json does not exist in winner_refine_round ranking: '
                f'{requested_front_runner}'
            )
    else:
        refine_entry = valid_refine_entries[0]
    refine_winner = fidelity.candidate_from_entry(refine_entry)
    persisted_candidates = load_persisted_dispatch_candidates(run_dir, ROUND_KIND_ABLATION)
    candidates = (
        persisted_candidates
        if persisted_candidates is not None
        else fidelity.build_p1_ablation_candidates(
            protocols,
            calibration,
            refine_winner,
            search_space=search_space,
        )
    )
    candidate_index = {candidate.arm_name: candidate for candidate in candidates}
    return {
        'round_kind': ROUND_KIND_ABLATION,
        'run_dir': run_dir,
        'run_name': run_dir.name,
        'state': state,
        'seed': seed,
        'base_cfg': base_cfg,
        'grouped': grouped,
        'eval_splits': eval_splits,
        'calibration': calibration,
        'protocols': protocols,
        'refine_winner': refine_winner,
        'candidates': candidates,
        'candidate_index': candidate_index,
        'seed_base': seed + 707,
        'seed_offsets': list(fidelity.P1_ABLATION_SEED_OFFSETS),
        'step_scale': fidelity.P1_ABLATION_STEP_SCALE,
        'ab_name': f'{run_dir.name}_p1_ablation',
    }


def load_round_context(run_dir: Path, round_kind: str) -> dict[str, Any]:
    normalized = normalize_round_kind(round_kind)
    if normalized == ROUND_KIND_PROTOCOL_DECIDE:
        return load_protocol_decide_context(run_dir)
    if normalized == ROUND_KIND_ABLATION:
        return load_ablation_context(run_dir)
    return load_refine_context(run_dir)


@contextmanager
def patched_base_screening(overrides: dict[str, int | None]):
    original = dict(ab.BASE_SCREENING)
    for key, value in overrides.items():
        if value is None:
            continue
        ab.BASE_SCREENING[key] = int(value)
    try:
        yield
    finally:
        ab.BASE_SCREENING.clear()
        ab.BASE_SCREENING.update(original)


def execute_single_task(
    *,
    round_kind: str = ROUND_KIND_WINNER_REFINE,
    run_name: str,
    candidate_arm: str,
    seed: int,
    result_json: Path,
    machine_label: str,
    screening_num_workers: int | None = None,
    screening_file_batch_size: int | None = None,
    screening_prefetch_factor: int | None = None,
    screening_val_file_batch_size: int | None = None,
    screening_val_prefetch_factor: int | None = None,
) -> dict[str, Any]:
    normalized_round_kind = normalize_round_kind(round_kind)
    run_dir = fidelity.FIDELITY_ROOT / run_name
    context = load_round_context(run_dir, normalized_round_kind)
    candidate = context['candidate_index'].get(candidate_arm)
    if candidate is None:
        raise KeyError(f'unknown {normalized_round_kind} candidate `{candidate_arm}`')
    with patched_base_screening(
        {
            'num_workers': screening_num_workers,
            'file_batch_size': screening_file_batch_size,
            'prefetch_factor': screening_prefetch_factor,
            'val_file_batch_size': screening_val_file_batch_size,
            'val_prefetch_factor': screening_val_prefetch_factor,
        }
    ):
        raw_payload = fidelity.run_arm_cached(
            base_cfg=context['base_cfg'],
            grouped=context['grouped'],
            eval_splits=context['eval_splits'],
            candidate=candidate,
            seed=seed,
            step_scale=context['step_scale'],
            ab_name=build_seed_ab_name(run_name, seed, normalized_round_kind),
        )
    raw_payload['machine_label'] = machine_label
    summary = fidelity.summarize_entry(
        candidate.arm_name,
        candidate,
        raw_payload,
        ranking_mode=fidelity.P1_RANKING_MODE,
    )
    summary['machine_label'] = machine_label
    payload = {
        'schema_version': TASK_RESULT_SCHEMA_VERSION,
        'round_kind': normalized_round_kind,
        'run_name': run_name,
        'candidate_arm': candidate.arm_name,
        'seed': seed,
        'seed_label': f's{seed}',
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
        'seed': payload.get('seed'),
        'seed_label': payload.get('seed_label'),
        'machine_label': payload.get('machine_label'),
        'completed_at': payload.get('completed_at'),
        'result_json': str(result_json),
        'valid': summary.get('valid'),
        'selection_quality_score': summary.get('selection_quality_score'),
        'recent_policy_loss': summary.get('recent_policy_loss'),
    }


def seed2_selection_within_gap(
    entry: dict[str, Any],
    *,
    leader_selection_score: float,
    selection_gap: float,
    leader_recent_loss: float,
) -> bool:
    selection_score = float(entry.get('selection_quality_score', float('-inf')))
    if math.isfinite(leader_selection_score) and math.isfinite(selection_score):
        return selection_score >= leader_selection_score - selection_gap
    recent_loss = float(entry.get('comparison_recent_loss', entry.get('recent_policy_loss', math.inf)))
    if not math.isfinite(leader_recent_loss) or not math.isfinite(recent_loss):
        return False
    return recent_loss <= leader_recent_loss + fidelity.P1_POLICY_LOSS_EPSILON


def winner_refine_seed2_source_arm(
    entry: dict[str, Any],
    *,
    candidate_index: dict[str, fidelity.CandidateSpec],
) -> str:
    entry_meta = entry.get('candidate_meta')
    if isinstance(entry_meta, dict):
        source_arm = str(entry_meta.get('source_arm', '') or '').strip()
        if source_arm:
            return source_arm
    arm_name = str(entry.get('arm_name', '') or '').strip()
    candidate = candidate_index.get(arm_name)
    if candidate is not None:
        source_arm = str(candidate.meta.get('source_arm', '') or '').strip()
        if source_arm:
            return source_arm
    return arm_name


def select_winner_refine_seed2_candidates(
    ranking: list[dict[str, Any]],
    candidates: list[fidelity.CandidateSpec],
    *,
    min_keep: int,
    selection_gap: float,
    max_keep: int | None,
) -> tuple[list[fidelity.CandidateSpec], dict[str, Any]]:
    candidate_index = {candidate.arm_name: candidate for candidate in candidates}
    valid_entries = [entry for entry in ranking if entry.get('valid')]
    if not valid_entries:
        raise RuntimeError('winner_refine seed1 produced no valid candidates')
    preferred_pool = [entry for entry in valid_entries if entry.get('eligible')]
    pool = preferred_pool or valid_entries
    leader = pool[0]
    leader_score = float(leader.get('selection_quality_score', float('-inf')))
    leader_recent_loss = float(
        leader.get('comparison_recent_loss', leader.get('recent_policy_loss', math.inf))
    )
    selected_name_set: set[str] = set()
    center_floor_entries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for entry in valid_entries:
        source_arm = winner_refine_seed2_source_arm(entry, candidate_index=candidate_index)
        if source_arm in seen_sources:
            continue
        seen_sources.add(source_arm)
        center_floor_entries.append(entry)
    floor_keep = max(int(min_keep), 1)
    center_floor_limit = len(center_floor_entries)
    if max_keep is not None:
        center_floor_limit = min(center_floor_limit, max_keep)
    for entry in center_floor_entries[:center_floor_limit]:
        selected_name_set.add(str(entry['arm_name']))
    floor_target = max(floor_keep, len(selected_name_set))
    if max_keep is not None:
        floor_target = min(floor_target, max_keep)
    for entry in pool:
        if max_keep is not None and len(selected_name_set) >= max_keep:
            break
        arm_name = str(entry['arm_name'])
        if arm_name in selected_name_set:
            continue
        keep_due_to_floor = len(selected_name_set) < floor_target
        keep_due_to_gap = seed2_selection_within_gap(
            entry,
            leader_selection_score=leader_score,
            selection_gap=selection_gap,
            leader_recent_loss=leader_recent_loss,
        )
        if keep_due_to_floor or keep_due_to_gap:
            selected_name_set.add(arm_name)
            continue
        if len(selected_name_set) >= floor_target:
            break
    if len(selected_name_set) < floor_target:
        for entry in pool:
            arm_name = str(entry['arm_name'])
            if arm_name in selected_name_set:
                continue
            if max_keep is not None and len(selected_name_set) >= max_keep:
                break
            selected_name_set.add(arm_name)
            if len(selected_name_set) >= floor_target:
                break
    selected_names = [
        str(entry['arm_name'])
        for entry in valid_entries
        if str(entry.get('arm_name', '')) in selected_name_set
    ]
    selected_candidates = [candidate_index[name] for name in selected_names if name in candidate_index]
    if not selected_candidates:
        raise RuntimeError('winner_refine seed2 selector returned no candidates')
    details = {
        'mode': 'per_source_floor_then_selection_gap',
        'pool_mode': 'eligible' if preferred_pool else 'valid',
        'min_keep': floor_keep,
        'floor_target': floor_target,
        'per_source_floor': 1,
        'source_count': len(center_floor_entries),
        'source_floor_arm_names': [str(entry['arm_name']) for entry in center_floor_entries[:center_floor_limit]],
        'selection_gap': float(selection_gap),
        'max_keep': None if max_keep is None else int(max_keep),
        'leader_arm_name': leader['arm_name'],
        'leader_selection_quality_score': None if not math.isfinite(leader_score) else leader_score,
        'leader_recent_policy_loss': None if not math.isfinite(leader_recent_loss) else leader_recent_loss,
        'selected_arm_names': [candidate.arm_name for candidate in selected_candidates],
        'candidate_count': len(selected_candidates),
    }
    return selected_candidates, details


def build_seed_stage_state(
    *,
    stage_name: str,
    candidates: list[fidelity.CandidateSpec],
    actual_seed: int,
) -> dict[str, Any]:
    seed_label = f's{actual_seed}'
    tasks: dict[str, Any] = {}
    for candidate in candidates:
        task_id = build_task_id(stage_name=stage_name, seed_label=seed_label, arm_name=candidate.arm_name)
        tasks[task_id] = {
            'task_id': task_id,
            'candidate_arm': candidate.arm_name,
            'seed': actual_seed,
            'seed_label': seed_label,
            'status': 'pending',
            'attempts': 0,
        }
    return {
        'stage_name': stage_name,
        'seed': actual_seed,
        'seed_label': seed_label,
        'candidate_count': len(candidates),
        'tasks': tasks,
    }


def initialize_dispatch_state(
    *,
    context: dict[str, Any],
    local_label: str,
    remote_label: str | None,
    seed2_min_keep: int,
    seed2_selection_gap: float,
    seed2_max_keep: int | None,
) -> dict[str, Any]:
    first_seed = context['seed_base'] + context['seed_offsets'][0]
    return {
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': ROUND_KIND_WINNER_REFINE,
        'run_name': context['run_name'],
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'status': 'running',
        'stage': 'seed1',
        'selected_protocol_arm': context['selected_protocol_arm'],
        'winner_refine_centers': [candidate.arm_name for candidate in context['winner_centers']],
        'local_label': local_label,
        'remote_label': remote_label,
        'seed_base': context['seed_base'],
        'seed_offsets': list(context['seed_offsets']),
        'step_scale': context['step_scale'],
        'candidate_payloads': [
            fidelity.candidate_cache_payload(candidate, include_meta=True)
            for candidate in context['candidates']
        ],
        'seed2_selector_config': {
            'min_keep': int(seed2_min_keep),
            'selection_gap': float(seed2_selection_gap),
            'max_keep': None if seed2_max_keep is None else int(seed2_max_keep),
        },
        'seed1': build_seed_stage_state(
            stage_name='seed1',
            candidates=context['candidates'],
            actual_seed=first_seed,
        ),
        'seed2': None,
        'seed2_selector': None,
        'seed1_round_summary_path': None,
        'seed2_round_summary_path': None,
        'final_round_summary_path': None,
        'final_front_runner': None,
    }


def initialize_fixed_pool_multiseed_dispatch_state(
    *,
    context: dict[str, Any],
    local_label: str,
    remote_label: str | None,
) -> dict[str, Any]:
    first_seed = context['seed_base'] + context['seed_offsets'][0]
    return {
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': normalize_round_kind(context['round_kind']),
        'run_name': context['run_name'],
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'status': 'running',
        'stage': 'seed1',
        'local_label': local_label,
        'remote_label': remote_label,
        'seed_base': context['seed_base'],
        'seed_offsets': list(context['seed_offsets']),
        'step_scale': context['step_scale'],
        'candidate_payloads': [
            fidelity.candidate_cache_payload(candidate, include_meta=True)
            for candidate in context['candidates']
        ],
        'seed1': build_seed_stage_state(
            stage_name='seed1',
            candidates=context['candidates'],
            actual_seed=first_seed,
        ),
        'seed2': None,
        'seed1_round_summary_path': None,
        'seed2_round_summary_path': None,
        'final_round_summary_path': None,
        'final_front_runner': None,
        'final_p1_winner': None,
    }


def initialize_protocol_decide_dispatch_state(
    *,
    context: dict[str, Any],
    local_label: str,
    remote_label: str | None,
) -> dict[str, Any]:
    first_seed = context['seed_base'] + context['seed_offsets'][0]
    search_space = context['search_space']
    return {
        'schema_version': DISPATCH_SCHEMA_VERSION,
        'round_kind': ROUND_KIND_PROTOCOL_DECIDE,
        'run_name': context['run_name'],
        'created_at': fidelity.ts_now(),
        'updated_at': fidelity.ts_now(),
        'status': 'running',
        'stage': 'seed1',
        'local_label': local_label,
        'remote_label': remote_label,
        'seed_base': context['seed_base'],
        'seed_offsets': list(context['seed_offsets']),
        'step_scale': context['step_scale'],
        'candidate_payloads': [
            fidelity.candidate_cache_payload(candidate, include_meta=True)
            for candidate in context['candidates']
        ],
        'seed2_plan': {
            'probe_selector_name': 'protocol_all_three_top4',
            'group_key': 'protocol_arm',
            'probe_keep_per_protocol': int(
                search_space.get(
                    'protocol_decide_probe_keep_per_protocol',
                    fidelity.P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
                )
            ),
            'ambiguity_mode': str(search_space['protocol_decide_progressive_ambiguity_mode']),
            'gap_threshold': search_space['protocol_decide_progressive_gap_threshold'],
            'noise_margin_mult': float(search_space['protocol_decide_progressive_noise_margin_mult']),
            'probe_candidate_names': [],
            'decision_candidate_names': [],
            'expanded_groups': [],
            'ambiguity_details': {},
        },
        'seed1': build_seed_stage_state(
            stage_name='seed1',
            candidates=context['candidates'],
            actual_seed=first_seed,
        ),
        'seed2': None,
        'seed1_round_summary_path': None,
        'seed2_round_summary_path': None,
        'final_round_summary_path': None,
        'final_protocol_winner': None,
        'final_protocol_compare': None,
    }


def reset_running_tasks_for_resume(dispatch_state: dict[str, Any]) -> None:
    for stage_key in ('seed1', 'seed2'):
        stage_state = dispatch_state.get(stage_key)
        if not isinstance(stage_state, dict):
            continue
        for task in stage_state.get('tasks', {}).values():
            if str(task.get('status')) == 'running':
                task['status'] = 'pending'
                task.pop('started_at', None)
                task.pop('worker_label', None)
                task.pop('local_result_path', None)
                task.pop('remote_result_path', None)
                task.pop('log_path', None)


def find_next_pending_task(stage_state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    pending = [
        (task_id, task)
        for task_id, task in stage_state.get('tasks', {}).items()
        if str(task.get('status', 'pending')) == 'pending'
    ]
    if not pending:
        return None
    pending.sort(key=lambda item: item[0])
    return pending[0]


def stage_all_tasks_completed(stage_state: dict[str, Any]) -> bool:
    tasks = list(stage_state.get('tasks', {}).values())
    return bool(tasks) and all(str(task.get('status')) == 'completed' for task in tasks)


def stage_any_task_failed(stage_state: dict[str, Any]) -> bool:
    return any(str(task.get('status')) == 'failed' for task in stage_state.get('tasks', {}).values())


def reset_task_after_operator_interrupt(task_state: dict[str, Any], *, note: str) -> None:
    attempts = int(task_state.get('attempts', 0))
    task_state['status'] = 'pending'
    task_state['attempts'] = max(0, attempts - 1)
    task_state['error'] = note
    task_state['interrupted_at'] = fidelity.ts_now()
    task_state.pop('finished_at', None)
    task_state.pop('started_at', None)
    task_state.pop('worker_label', None)
    task_state.pop('local_result_path', None)
    task_state.pop('remote_result_path', None)
    task_state.pop('log_path', None)
    task_state.pop('pid', None)


def launch_local_task(
    worker: WorkerSpec,
    *,
    round_kind: str = ROUND_KIND_WINNER_REFINE,
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
        stage_name='',
        local_result_path=result_path,
        log_path=log_path,
        command_args=[
            'run-task',
            '--round-kind',
            normalize_round_kind(round_kind),
            '--run-name',
            run_name,
            '--candidate-arm',
            str(task_state['candidate_arm']),
            '--seed',
            str(task_state['seed']),
            '--machine-label',
            worker.label,
        ],
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


def build_remote_command(
    *,
    worker: WorkerSpec,
    round_kind: str = ROUND_KIND_WINNER_REFINE,
    run_name: str,
    task_state: dict[str, Any],
    remote_result_path: Path,
    screening_overrides: dict[str, int],
) -> list[str]:
    command_args = [
        'run-task',
        '--round-kind',
        normalize_round_kind(round_kind),
        '--run-name',
        run_name,
        '--candidate-arm',
        str(task_state['candidate_arm']),
        '--seed',
        str(task_state['seed']),
        '--machine-label',
        worker.label,
    ]
    if screening_overrides:
        command_args.extend(
            [
                '--screening-num-workers',
                str(screening_overrides['num_workers']),
                '--screening-file-batch-size',
                str(screening_overrides['file_batch_size']),
                '--screening-prefetch-factor',
                str(screening_overrides['prefetch_factor']),
                '--screening-val-file-batch-size',
                str(screening_overrides['val_file_batch_size']),
                '--screening-val-prefetch-factor',
                str(screening_overrides['val_prefetch_factor']),
            ]
        )
    return dispatch.build_remote_python_command(
        worker=worker,
        script_path=SCRIPT_PATH,
        remote_result_path=remote_result_path,
        command_args=command_args,
    )


def build_remote_interactive_window_command(
    *,
    worker: WorkerSpec,
    round_kind: str = ROUND_KIND_WINNER_REFINE,
    run_name: str,
    task_state: dict[str, Any],
    remote_result_path: Path,
    remote_runtime_root: Path,
    screening_overrides: dict[str, int],
) -> list[str]:
    python_args = [
        'run-task',
        '--round-kind',
        normalize_round_kind(round_kind),
        '--run-name',
        run_name,
        '--candidate-arm',
        str(task_state['candidate_arm']),
        '--seed',
        str(task_state['seed']),
        '--machine-label',
        worker.label,
        '--result-json',
        str(remote_result_path),
    ]
    if screening_overrides:
        python_args.extend(
            [
                '--screening-num-workers',
                str(screening_overrides['num_workers']),
                '--screening-file-batch-size',
                str(screening_overrides['file_batch_size']),
                '--screening-prefetch-factor',
                str(screening_overrides['prefetch_factor']),
                '--screening-val-file-batch-size',
                str(screening_overrides['val_file_batch_size']),
                '--screening-val-prefetch-factor',
                str(screening_overrides['val_prefetch_factor']),
            ]
        )
    repo_root = Path(worker.repo or str(REPO_ROOT))
    window_title = f"MahjongAI {normalize_round_kind(round_kind)} {task_state['task_id']}"
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
    command.append(worker.host or DEFAULT_REMOTE_HOST)
    command.append(ps_command)
    return command


def launch_remote_task(
    worker: WorkerSpec,
    *,
    round_kind: str = ROUND_KIND_WINNER_REFINE,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    launch_mode: str,
    screening_overrides: dict[str, int],
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
        command = build_remote_command(
            worker=worker,
            round_kind=round_kind,
            run_name=run_name,
            task_state=task_state,
            remote_result_path=remote_result_path,
            screening_overrides=screening_overrides,
        )
    elif launch_mode == 'interactive_window':
        command = build_remote_interactive_window_command(
            worker=worker,
            round_kind=round_kind,
            run_name=run_name,
            task_state=task_state,
            remote_result_path=remote_result_path,
            remote_runtime_root=remote_runtime_dir,
            screening_overrides=screening_overrides,
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
        stage_name='',
        task_id=str(task_state['task_id']),
        task_state=task_state,
        process=process,
        log_path=log_path,
        local_result_path=local_result_path,
        remote_result_path=remote_result_path,
    )


def fetch_remote_result(worker: WorkerSpec, remote_result_path: str, local_result_path: Path) -> None:
    dispatch.fetch_remote_result(worker, remote_result_path, local_result_path)


def load_task_result(path: Path) -> dict[str, Any]:
    payload = fidelity.load_json(path)
    summary = payload.get('summary')
    if not isinstance(summary, dict):
        raise RuntimeError(f'task result at {path} is missing summary')
    if not bool(summary.get('ok')):
        raise RuntimeError(
            f'task result at {path} reported training failure: {summary.get("error") or "unknown error"}'
        )
    if not bool(summary.get('valid')):
        raise RuntimeError(f'task result at {path} is not valid for ranking')
    return payload


def interrupt_local_active_task(active: ActiveTask) -> None:
    if active.process.poll() is not None:
        return
    try:
        subprocess.run(
            ['taskkill', '/PID', str(active.process.pid), '/T', '/F'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=INTERRUPT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pass


def interrupt_remote_active_task(active: ActiveTask) -> None:
    remote_result_path = active.remote_result_path
    if remote_result_path is None:
        return
    match = str(remote_result_path)
    task_name = str(active.task_state.get('remote_task_name') or f'{REMOTE_INTERACTIVE_TASK_NAME_PREFIX}{active.task_id}')
    runtime_root = str(
        active.task_state.get('remote_runtime_root')
        or (Path(remote_result_path).parent.parent / 'remote_runtime' / str(active.task_id))
    )
    kill_command = (
        "$match = "
        + quote_ps(match)
        + "; $taskName = "
        + quote_ps(task_name)
        + "; $runtimeRoot = "
        + quote_ps(runtime_root)
        + "; "
        + "try { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Out-Null } catch {} "
        + "$targets = Get-CimInstance Win32_Process "
        + "| Where-Object { $_.Name -in @('python.exe','powershell.exe','cmd.exe') } "
        + "| Where-Object { "
        + "($_.ProcessId -ne $PID) -and ("
        + "($_.CommandLine -like ('*' + $match + '*')) "
        + "-or ($_.CommandLine -like ('*' + $runtimeRoot + '*'))"
        + ") "
        + "}; "
        + "foreach ($target in $targets) { taskkill /PID $target.ProcessId /T /F | Out-Null } "
        + "try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null } catch {}"
    )
    command = ['ssh']
    if active.worker.ssh_key:
        command.extend(['-i', active.worker.ssh_key])
    command.append(active.worker.host or DEFAULT_REMOTE_HOST)
    command.append(kill_command)
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=INTERRUPT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pass
    if active.process.poll() is None:
        try:
            subprocess.run(
                ['taskkill', '/PID', str(active.process.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=INTERRUPT_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            pass


def interrupt_active_task(active: ActiveTask) -> None:
    if active.worker.kind == 'local':
        interrupt_local_active_task(active)
        return
    if active.worker.kind == 'remote':
        interrupt_remote_active_task(active)
        return
    raise ValueError(f'unsupported worker kind `{active.worker.kind}`')


def try_collect_running_remote_result(active: ActiveTask) -> dict[str, Any] | None:
    if active.worker.kind != 'remote':
        return None
    if str(active.task_state.get('status')) != 'running':
        return None
    if str(active.task_state.get('remote_launch_mode') or '') != 'interactive_window':
        return None
    if active.remote_result_path is None:
        return None
    local_result_path = active.local_result_path
    try:
        fetch_remote_result(active.worker, str(active.remote_result_path), local_result_path)
        payload = load_task_result(local_result_path)
    except Exception:
        if local_result_path.exists():
            try:
                local_result_path.unlink()
            except OSError:
                pass
        return None
    interrupt_remote_active_task(active)
    active.task_state['status'] = 'completed'
    active.task_state['finished_at'] = str(payload.get('completed_at') or fidelity.ts_now())
    active.task_state['completion_source'] = 'remote_result_json'
    active.task_state.pop('error', None)
    return payload


def build_seed_round_payload(
    *,
    context: dict[str, Any],
    candidates: list[fidelity.CandidateSpec],
    actual_seed: int,
    task_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    round_kind = str(context.get('round_kind') or ROUND_KIND_WINNER_REFINE)
    round_name = build_seed_round_name(actual_seed, round_kind)
    ab_name = build_seed_ab_name(context['run_name'], actual_seed, round_kind)
    ranking = fidelity.rank_round_entries(
        [task_results[candidate.arm_name]['summary'] for candidate in candidates if candidate.arm_name in task_results],
        ranking_mode=fidelity.P1_RANKING_MODE,
        eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    raw_results = {
        candidate.arm_name: task_results[candidate.arm_name]['payload']
        for candidate in candidates
        if candidate.arm_name in task_results
    }
    signature = fidelity.round_cache_signature(
        round_name=round_name,
        ab_name=ab_name,
        base_cfg=context['base_cfg'],
        grouped=context['grouped'],
        eval_splits=context['eval_splits'],
        candidates=candidates,
        seed=actual_seed,
        step_scale=context['step_scale'],
        selector_weights=None,
        ranking_mode=fidelity.P1_RANKING_MODE,
        eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    payload = {
        'scenario_score_version': fidelity.SCENARIO_SCORE_VERSION,
        'round_name': round_name,
        'ab_name': ab_name,
        'seed': actual_seed,
        'step_scale': context['step_scale'],
        'ranking_mode': fidelity.P1_RANKING_MODE,
        'eligibility_group_key': fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        'round_signature': signature,
        'evaluated_arms': len(candidates),
        'eval_split_counts': {
            'monitor_recent_files': len(context['eval_splits']['monitor_recent_files']),
            'full_recent_files': len(context['eval_splits']['full_recent_files']),
            'old_regression_files': len(context['eval_splits']['old_regression_files']),
        },
        'best_loss': min((entry['full_recent_loss'] for entry in ranking if entry.get('valid')), default=None),
        'best_recent_policy_loss': min(
            (entry.get('recent_policy_loss', math.inf) for entry in ranking if entry.get('valid')),
            default=None,
        ),
        'ranking': ranking,
        'raw_results': raw_results,
    }
    summary_path = context['run_dir'] / f'{round_name}.json'
    fidelity.atomic_write_json(summary_path, payload)
    return payload


def build_final_round_payload(
    *,
    context: dict[str, Any],
    seed1_payload: dict[str, Any],
    seed2_payload: dict[str, Any],
    decision_candidates: list[fidelity.CandidateSpec],
    selector_details: dict[str, Any],
) -> dict[str, Any]:
    first_seed = int(seed1_payload['seed'])
    second_seed = int(seed2_payload['seed'])
    first_label = f's{first_seed}'
    second_label = f's{second_seed}'
    current_round_signature = fidelity.stable_payload_digest(
        fidelity.apply_round_signature_ranking_fields(
            {
                'schema_version': fidelity.ROUND_CACHE_SCHEMA_VERSION,
                'scenario_score_version': fidelity.SCENARIO_SCORE_VERSION,
                'round_name': 'p1_winner_refine_round',
                'ab_name': context['ab_name'],
                'base_cfg': context['base_cfg'],
                'grouped': context['grouped'],
                'eval_splits': context['eval_splits'],
                'candidates': [
                    fidelity.candidate_cache_payload(candidate, include_meta=True)
                    for candidate in context['candidates']
                ],
                'seed': context['seed_base'],
                'seed_offsets': list(context['seed_offsets']),
                'step_scale': context['step_scale'],
                'selector_weights': None,
                'seed_strategy': 'distributed_seed1_then_dynamic_seed2',
                'seed2_selector': selector_details,
            },
            ranking_mode=fidelity.P1_RANKING_MODE,
            eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        )
    )
    seed_rounds = {
        first_label: {
            'seed': first_seed,
            'round_name': seed1_payload['round_name'],
            'ab_name': seed1_payload['ab_name'],
            'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            **seed1_payload,
        },
        second_label: {
            'seed': second_seed,
            'round_name': seed2_payload['round_name'],
            'ab_name': seed2_payload['ab_name'],
            'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
            **seed2_payload,
        },
    }
    final_ranking = fidelity.summarize_multiseed_candidates(
        decision_candidates,
        seed_rounds,
        ranking_mode=fidelity.P1_RANKING_MODE,
        eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    pruned_arm_names = sorted(
        candidate.arm_name
        for candidate in context['candidates']
        if candidate.arm_name not in {item.arm_name for item in decision_candidates}
    )
    payload = fidelity.build_multiseed_payload(
        round_name='p1_winner_refine_round',
        ab_name=context['ab_name'],
        seed=context['seed_base'],
        seed_offsets=list(context['seed_offsets']),
        step_scale=context['step_scale'],
        round_signature=current_round_signature,
        eval_splits=context['eval_splits'],
        seed_rounds=seed_rounds,
        ranked=final_ranking,
        evaluated_arms=len(context['candidates']),
        extra={
            'seed_strategy': 'distributed_seed1_then_dynamic_seed2',
            'ranking_mode': fidelity.P1_RANKING_MODE,
            'eligibility_group_key': fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
            'seed1_candidate_count': len(context['candidates']),
            'seed2_candidate_count': len(decision_candidates),
            'pruned_candidate_count': len(pruned_arm_names),
            'pruned_arm_names': pruned_arm_names,
            'seed2_selector': selector_details,
            'seed1_round': {
                'actual_seed': first_seed,
                'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            },
            'seed2_round': {
                'actual_seed': second_seed,
                'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
            },
        },
    )
    fidelity.atomic_write_json(context['run_dir'] / 'p1_winner_refine_round.json', payload)
    return payload


def build_fixed_pool_final_round_payload(
    *,
    context: dict[str, Any],
    seed1_payload: dict[str, Any],
    seed2_payload: dict[str, Any],
) -> dict[str, Any]:
    round_name = final_round_name_for_round_kind(context['round_kind'])
    first_seed = int(seed1_payload['seed'])
    second_seed = int(seed2_payload['seed'])
    seed_rounds = {
        f's{first_seed}': {
            'seed': first_seed,
            'round_name': seed1_payload['round_name'],
            'ab_name': seed1_payload['ab_name'],
            'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            **seed1_payload,
        },
        f's{second_seed}': {
            'seed': second_seed,
            'round_name': seed2_payload['round_name'],
            'ab_name': seed2_payload['ab_name'],
            'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
            **seed2_payload,
        },
    }
    current_round_signature = fidelity.stable_payload_digest(
        fidelity.apply_round_signature_ranking_fields(
            {
                'schema_version': fidelity.ROUND_CACHE_SCHEMA_VERSION,
                'scenario_score_version': fidelity.SCENARIO_SCORE_VERSION,
                'round_name': round_name,
                'ab_name': context['ab_name'],
                'base_cfg': context['base_cfg'],
                'grouped': context['grouped'],
                'eval_splits': context['eval_splits'],
                'candidates': [
                    fidelity.candidate_cache_payload(candidate, include_meta=True)
                    for candidate in context['candidates']
                ],
                'seed': context['seed_base'],
                'seed_offsets': list(context['seed_offsets']),
                'step_scale': context['step_scale'],
                'selector_weights': None,
                'seed_strategy': 'distributed_full_multiseed',
            },
            ranking_mode=fidelity.P1_RANKING_MODE,
            eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        )
    )
    final_ranking = fidelity.summarize_multiseed_candidates(
        context['candidates'],
        seed_rounds,
        ranking_mode=fidelity.P1_RANKING_MODE,
        eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    payload = fidelity.build_multiseed_payload(
        round_name=round_name,
        ab_name=context['ab_name'],
        seed=context['seed_base'],
        seed_offsets=list(context['seed_offsets']),
        step_scale=context['step_scale'],
        round_signature=current_round_signature,
        eval_splits=context['eval_splits'],
        seed_rounds=seed_rounds,
        ranked=final_ranking,
        evaluated_arms=len(context['candidates']),
        extra={
            'seed_strategy': 'distributed_full_multiseed',
            'ranking_mode': fidelity.P1_RANKING_MODE,
            'eligibility_group_key': fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
            'seed1_candidate_count': len(context['candidates']),
            'seed2_candidate_count': len(context['candidates']),
            'seed1_round': {
                'actual_seed': first_seed,
                'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            },
            'seed2_round': {
                'actual_seed': second_seed,
                'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
            },
        },
    )
    fidelity.atomic_write_json(context['run_dir'] / f'{round_name}.json', payload)
    return payload


def stage_results_from_state(stage_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in stage_state.get('tasks', {}).values():
        if str(task.get('status')) != 'completed':
            continue
        result_path = task.get('local_result_path')
        if not result_path:
            continue
        payload = load_task_result(Path(result_path))
        results[str(task['candidate_arm'])] = payload
    return results


def maybe_promote_seed1_to_seed2(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> None:
    if dispatch_state.get('stage') != 'seed1':
        return
    stage_state = dispatch_state['seed1']
    if not stage_all_tasks_completed(stage_state):
        return
    seed1_results = stage_results_from_state(stage_state)
    seed1_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(stage_state['seed']),
        task_results=seed1_results,
    )
    selector_candidates, selector_details = select_winner_refine_seed2_candidates(
        seed1_payload['ranking'],
        context['candidates'],
        min_keep=int(dispatch_state['seed2_selector_config']['min_keep']),
        selection_gap=float(dispatch_state['seed2_selector_config']['selection_gap']),
        max_keep=dispatch_state['seed2_selector_config'].get('max_keep'),
    )
    second_seed = context['seed_base'] + context['seed_offsets'][1]
    dispatch_state['seed1_round_summary_path'] = str(
        context['run_dir'] / f'{seed1_payload["round_name"]}.json'
    )
    dispatch_state['seed2_selector'] = selector_details
    dispatch_state['seed2'] = build_seed_stage_state(
        stage_name='seed2',
        candidates=selector_candidates,
        actual_seed=second_seed,
    )
    dispatch_state['stage'] = 'seed2'
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
    )


def maybe_promote_fixed_pool_seed1_to_seed2(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> None:
    if dispatch_state.get('stage') != 'seed1':
        return
    stage_state = dispatch_state['seed1']
    if not stage_all_tasks_completed(stage_state):
        return
    seed1_results = stage_results_from_state(stage_state)
    seed1_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(stage_state['seed']),
        task_results=seed1_results,
    )
    second_seed = context['seed_base'] + context['seed_offsets'][1]
    dispatch_state['seed1_round_summary_path'] = str(
        context['run_dir'] / f'{seed1_payload["round_name"]}.json'
    )
    dispatch_state['seed2'] = build_seed_stage_state(
        stage_name='seed2',
        candidates=context['candidates'],
        actual_seed=second_seed,
    )
    dispatch_state['stage'] = 'seed2'
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        round_kind=context['round_kind'],
    )


def maybe_finalize_dispatch(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> bool:
    if dispatch_state.get('stage') != 'seed2':
        return False
    seed2_state = dispatch_state.get('seed2')
    if not isinstance(seed2_state, dict) or not stage_all_tasks_completed(seed2_state):
        return False
    seed1_results = stage_results_from_state(dispatch_state['seed1'])
    seed2_results = stage_results_from_state(seed2_state)
    seed1_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(dispatch_state['seed1']['seed']),
        task_results=seed1_results,
    )
    selector_names = set(dispatch_state.get('seed2_selector', {}).get('selected_arm_names', []))
    decision_candidates = [
        candidate for candidate in context['candidates'] if candidate.arm_name in selector_names
    ]
    if not decision_candidates:
        raise RuntimeError('distributed winner_refine selected no seed2 candidates')
    seed2_payload = build_seed_round_payload(
        context=context,
        candidates=decision_candidates,
        actual_seed=int(seed2_state['seed']),
        task_results=seed2_results,
    )
    final_round = build_final_round_payload(
        context=context,
        seed1_payload=seed1_payload,
        seed2_payload=seed2_payload,
        decision_candidates=decision_candidates,
        selector_details=dict(dispatch_state.get('seed2_selector') or {}),
    )
    final_valid = [entry for entry in final_round['ranking'] if entry.get('valid')]
    if not final_valid:
        raise RuntimeError('distributed winner_refine produced no valid final candidates')
    front_runner = str(final_valid[0]['arm_name'])
    dispatch_state['status'] = 'completed'
    dispatch_state['stage'] = 'completed'
    dispatch_state['seed2_round_summary_path'] = str(
        context['run_dir'] / f'{seed2_payload["round_name"]}.json'
    )
    dispatch_state['final_round_summary_path'] = str(context['run_dir'] / 'p1_winner_refine_round.json')
    dispatch_state['final_front_runner'] = front_runner
    dispatch_state['final_p1_winner'] = front_runner
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        front_runner=front_runner,
        final_round=final_round,
    )
    return True


def maybe_finalize_fixed_pool_dispatch(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> bool:
    if dispatch_state.get('stage') != 'seed2':
        return False
    seed2_state = dispatch_state.get('seed2')
    if not isinstance(seed2_state, dict) or not stage_all_tasks_completed(seed2_state):
        return False
    seed1_results = stage_results_from_state(dispatch_state['seed1'])
    seed2_results = stage_results_from_state(seed2_state)
    seed1_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(dispatch_state['seed1']['seed']),
        task_results=seed1_results,
    )
    seed2_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(seed2_state['seed']),
        task_results=seed2_results,
    )
    final_round = build_fixed_pool_final_round_payload(
        context=context,
        seed1_payload=seed1_payload,
        seed2_payload=seed2_payload,
    )
    final_valid = [entry for entry in final_round['ranking'] if entry.get('valid')]
    if not final_valid:
        raise RuntimeError(f'distributed {context["round_kind"]} produced no valid final candidates')
    front_runner = str(final_valid[0]['arm_name'])
    dispatch_state['status'] = 'completed'
    dispatch_state['stage'] = 'completed'
    dispatch_state['seed2_round_summary_path'] = str(
        context['run_dir'] / f'{seed2_payload["round_name"]}.json'
    )
    dispatch_state['final_round_summary_path'] = str(
        context['run_dir'] / f'{final_round_name_for_round_kind(context["round_kind"])}.json'
    )
    dispatch_state['final_front_runner'] = front_runner
    dispatch_state['final_p1_winner'] = front_runner
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        round_kind=context['round_kind'],
        front_runner=front_runner,
        final_round=final_round,
    )
    return True


def build_protocol_decide_final_round_payload(
    *,
    context: dict[str, Any],
    seed1_payload: dict[str, Any],
    seed2_payload: dict[str, Any],
    probe_candidates: list[fidelity.CandidateSpec],
    decision_candidates: list[fidelity.CandidateSpec],
    ambiguity_details: dict[str, dict[str, Any]],
    expanded_groups: list[str],
    screening_diagnostic: dict[str, Any] | None,
) -> dict[str, Any]:
    first_seed = int(seed1_payload['seed'])
    second_seed = int(seed2_payload['seed'])
    first_label = f's{first_seed}'
    second_label = f's{second_seed}'
    settings = context['search_space']
    current_round_signature = fidelity.stable_payload_digest(
        fidelity.apply_round_signature_ranking_fields(
            {
                'schema_version': fidelity.ROUND_CACHE_SCHEMA_VERSION,
                'scenario_score_version': fidelity.SCENARIO_SCORE_VERSION,
                'round_name': 'p1_protocol_decide_round',
                'ab_name': context['ab_name'],
                'base_cfg': context['base_cfg'],
                'grouped': context['grouped'],
                'eval_splits': context['eval_splits'],
                'candidates': [
                    fidelity.candidate_cache_payload(candidate, include_meta=True)
                    for candidate in context['candidates']
                ],
                'seed': context['seed_base'],
                'seed_offsets': list(context['seed_offsets']),
                'step_scale': context['step_scale'],
                'selector_weights': None,
                'seed_strategy': 'distributed_progressive_probe_then_expand',
                'probe_selector_name': 'protocol_all_three_top4',
                'probe_signature_data': {},
                'group_key': 'protocol_arm',
                'ambiguity_mode': settings['protocol_decide_progressive_ambiguity_mode'],
                'gap_threshold': settings['protocol_decide_progressive_gap_threshold'],
                'noise_margin_mult': settings['protocol_decide_progressive_noise_margin_mult'],
                'seed1_probe_compare_scope': 'probe_candidates_only',
                'screening_diagnostic_name': 'ce_only_anchor',
            },
            ranking_mode=fidelity.P1_RANKING_MODE,
            eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        )
    )
    seed_rounds = {
        first_label: {
            'seed': first_seed,
            'round_name': seed1_payload['round_name'],
            'ab_name': seed1_payload['ab_name'],
            'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            **seed1_payload,
        },
        second_label: {
            'seed': second_seed,
            'round_name': seed2_payload['round_name'],
            'ab_name': seed2_payload['ab_name'],
            'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
            **seed2_payload,
        },
    }
    final_ranking = fidelity.summarize_multiseed_candidates(
        decision_candidates,
        seed_rounds,
        ranking_mode=fidelity.P1_RANKING_MODE,
        eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    pruned_arm_names = sorted(
        candidate.arm_name
        for candidate in context['candidates']
        if candidate.arm_name not in {item.arm_name for item in decision_candidates}
    )
    payload = fidelity.build_multiseed_payload(
        round_name='p1_protocol_decide_round',
        ab_name=context['ab_name'],
        seed=context['seed_base'],
        seed_offsets=list(context['seed_offsets']),
        step_scale=context['step_scale'],
        round_signature=current_round_signature,
        eval_splits=context['eval_splits'],
        seed_rounds=seed_rounds,
        ranked=final_ranking,
        evaluated_arms=len(context['candidates']),
        extra={
            'seed_strategy': 'distributed_progressive_probe_then_expand',
            'probe_selector_name': 'protocol_all_three_top4',
            'probe_signature_data': {},
            'group_key': 'protocol_arm',
            'ranking_mode': fidelity.P1_RANKING_MODE,
            'eligibility_group_key': fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
            'ambiguity_mode': settings['protocol_decide_progressive_ambiguity_mode'],
            'gap_threshold': settings['protocol_decide_progressive_gap_threshold'],
            'noise_margin_mult': settings['protocol_decide_progressive_noise_margin_mult'],
            'seed1_probe_compare_scope': 'probe_candidates_only',
            'probe_candidate_count': len(probe_candidates),
            'decision_candidate_count': len(decision_candidates),
            'pruned_candidate_count': len(pruned_arm_names),
            'pruned_arm_names': pruned_arm_names,
            'expanded_groups': list(expanded_groups),
            'ambiguity_details': ambiguity_details,
            'screening_diagnostic': screening_diagnostic,
            'screening_round': {
                'actual_seed': first_seed,
                'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            },
            'probe_round': {
                'actual_seed': second_seed,
                'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
            },
        },
    )
    fidelity.atomic_write_json(context['run_dir'] / 'p1_protocol_decide_round.json', payload)
    return payload


def maybe_promote_protocol_decide_seed1_to_seed2(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> None:
    if dispatch_state.get('stage') != 'seed1':
        return
    stage_state = dispatch_state['seed1']
    if not stage_all_tasks_completed(stage_state):
        return
    seed1_results = stage_results_from_state(stage_state)
    seed1_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(stage_state['seed']),
        task_results=seed1_results,
    )
    probe_candidates = fidelity.unique_candidates(
        fidelity.select_p1_protocol_decide_probe_candidates(
            seed1_payload['ranking'],
            context['candidates'],
            keep=int(dispatch_state['seed2_plan']['probe_keep_per_protocol']),
        )
    )
    second_seed = context['seed_base'] + context['seed_offsets'][1]
    dispatch_state['seed1_round_summary_path'] = str(
        context['run_dir'] / f'{seed1_payload["round_name"]}.json'
    )
    dispatch_state['seed2_plan']['probe_candidate_names'] = [candidate.arm_name for candidate in probe_candidates]
    dispatch_state['seed2_plan']['decision_candidate_names'] = [candidate.arm_name for candidate in probe_candidates]
    dispatch_state['seed2'] = build_seed_stage_state(
        stage_name='seed2',
        candidates=probe_candidates,
        actual_seed=second_seed,
    )
    dispatch_state['stage'] = 'seed2'
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        round_kind=ROUND_KIND_PROTOCOL_DECIDE,
    )


def maybe_finalize_or_expand_protocol_decide_dispatch(
    *,
    context: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_state_path: Path,
) -> bool:
    if dispatch_state.get('stage') != 'seed2':
        return False
    seed2_state = dispatch_state.get('seed2')
    if not isinstance(seed2_state, dict) or not stage_all_tasks_completed(seed2_state):
        return False
    seed1_results = stage_results_from_state(dispatch_state['seed1'])
    seed2_results = stage_results_from_state(seed2_state)
    seed1_payload = build_seed_round_payload(
        context=context,
        candidates=context['candidates'],
        actual_seed=int(dispatch_state['seed1']['seed']),
        task_results=seed1_results,
    )
    plan = dispatch_state.get('seed2_plan')
    if not isinstance(plan, dict):
        raise RuntimeError('protocol_decide dispatch is missing seed2_plan')
    probe_names = set(plan.get('probe_candidate_names', []))
    probe_candidates = [candidate for candidate in context['candidates'] if candidate.arm_name in probe_names]
    if not probe_candidates:
        raise RuntimeError('protocol_decide seed2 probe pool is empty')
    current_seed2_names = {
        str(task.get('candidate_arm'))
        for task in seed2_state.get('tasks', {}).values()
    }
    seed2_current_candidates = [
        candidate for candidate in context['candidates'] if candidate.arm_name in current_seed2_names
    ]
    seed2_payload = build_seed_round_payload(
        context=context,
        candidates=seed2_current_candidates,
        actual_seed=int(seed2_state['seed']),
        task_results=seed2_results,
    )
    decision_names = set(plan.get('decision_candidate_names', []))
    ambiguity_details = dict(plan.get('ambiguity_details') or {})
    expanded_groups = list(plan.get('expanded_groups') or [])
    screening_diagnostic = plan.get('screening_diagnostic')
    if not decision_names or decision_names == probe_names:
        seed_rounds = {
            f's{int(dispatch_state["seed1"]["seed"])}': {
                'seed': int(dispatch_state['seed1']['seed']),
                'round_name': seed1_payload['round_name'],
                'ab_name': seed1_payload['ab_name'],
                'summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
                **seed1_payload,
            },
            f's{int(seed2_state["seed"])}': {
                'seed': int(seed2_state['seed']),
                'round_name': seed2_payload['round_name'],
                'ab_name': seed2_payload['ab_name'],
                'summary_path': str(context['run_dir'] / f'{seed2_payload["round_name"]}.json'),
                **seed2_payload,
            },
        }
        probe_ranking = fidelity.summarize_multiseed_candidates(
            probe_candidates,
            seed_rounds,
            ranking_mode=fidelity.P1_RANKING_MODE,
            eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        )
        seed1_probe_ranking = fidelity.rerank_filtered_entries(
            seed1_payload['ranking'],
            entry_selector=lambda entry: entry['arm_name'] in probe_names,
            ranking_mode=fidelity.P1_RANKING_MODE,
            eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        )
        ambiguity_groups, ambiguity_details = fidelity.detect_progressive_ambiguous_groups(
            seed1_ranking=seed1_probe_ranking,
            probe_ranking=probe_ranking,
            group_key='protocol_arm',
            ranking_mode=fidelity.P1_RANKING_MODE,
            ambiguity_mode=str(plan['ambiguity_mode']),
            gap_threshold=plan.get('gap_threshold'),
            noise_margin_mult=float(plan['noise_margin_mult']),
        )
        decision_candidates = fidelity.build_progressive_active_candidates(
            all_candidates=context['candidates'],
            probe_candidates=probe_candidates,
            ambiguous_groups=ambiguity_groups,
            group_key='protocol_arm',
        )
        decision_names = {candidate.arm_name for candidate in decision_candidates}
        expanded_groups = sorted(ambiguity_groups)
        screening_diagnostic = {
            'name': 'ce_only_anchor',
            'source_seed': int(dispatch_state['seed1']['seed']),
            'source_round_name': seed1_payload['round_name'],
            'source_summary_path': str(context['run_dir'] / f'{seed1_payload["round_name"]}.json'),
            'ranking': fidelity.rerank_filtered_entries(
                seed1_payload['ranking'],
                entry_selector=lambda entry: fidelity.is_p1_protocol_decide_diagnostic_meta(
                    entry.get('candidate_meta', {})
                ),
                ranking_mode=fidelity.P1_RANKING_MODE,
                eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
            ),
        }
        plan['decision_candidate_names'] = sorted(decision_names)
        plan['expanded_groups'] = list(expanded_groups)
        plan['ambiguity_details'] = ambiguity_details
        plan['screening_diagnostic'] = screening_diagnostic
        missing_candidates = [
            candidate for candidate in decision_candidates if candidate.arm_name not in current_seed2_names
        ]
        if missing_candidates:
            for candidate in missing_candidates:
                task_id = build_task_id(
                    stage_name='seed2',
                    seed_label=str(seed2_state['seed_label']),
                    arm_name=candidate.arm_name,
                )
                if task_id in seed2_state['tasks']:
                    continue
                seed2_state['tasks'][task_id] = {
                    'task_id': task_id,
                    'candidate_arm': candidate.arm_name,
                    'seed': int(seed2_state['seed']),
                    'seed_label': str(seed2_state['seed_label']),
                    'status': 'pending',
                    'attempts': 0,
                }
            seed2_state['candidate_count'] = len(seed2_state['tasks'])
            write_dispatch_state(dispatch_state_path, dispatch_state)
            update_run_state_for_dispatch(
                run_dir=context['run_dir'],
                dispatch_state_path=dispatch_state_path,
                dispatch_state=dispatch_state,
                round_kind=ROUND_KIND_PROTOCOL_DECIDE,
            )
            return False
    decision_candidates = [
        candidate for candidate in context['candidates'] if candidate.arm_name in decision_names
    ]
    final_round = build_protocol_decide_final_round_payload(
        context=context,
        seed1_payload=seed1_payload,
        seed2_payload=seed2_payload,
        probe_candidates=probe_candidates,
        decision_candidates=decision_candidates,
        ambiguity_details=ambiguity_details,
        expanded_groups=expanded_groups,
        screening_diagnostic=screening_diagnostic,
    )
    protocol_compare = fidelity.build_p1_protocol_compare(final_round['ranking'])
    selected_protocol_arm = str(
        protocol_compare[0].get('candidate_meta', {}).get('protocol_arm', protocol_compare[0]['arm_name'])
    )
    dispatch_state['status'] = 'completed'
    dispatch_state['stage'] = 'completed'
    dispatch_state['seed2_round_summary_path'] = str(
        context['run_dir'] / f'{seed2_payload["round_name"]}.json'
    )
    dispatch_state['final_round_summary_path'] = str(context['run_dir'] / 'p1_protocol_decide_round.json')
    dispatch_state['final_protocol_winner'] = selected_protocol_arm
    dispatch_state['final_protocol_compare'] = {
        'round_name': 'p1_protocol_compare',
        'ranking': protocol_compare,
    }
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        round_kind=ROUND_KIND_PROTOCOL_DECIDE,
        front_runner=selected_protocol_arm,
        final_round=final_round,
    )
    return True


def launch_task_for_worker(
    *,
    worker: WorkerSpec,
    round_kind: str,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    stage_name: str,
    control_state: dict[str, Any],
    remote_screening_overrides: dict[str, int],
) -> ActiveTask:
    if worker.kind == 'local':
        active = launch_local_task(
            worker,
            round_kind=round_kind,
            run_name=run_name,
            task_state=task_state,
            dispatch_root=dispatch_root,
        )
    elif worker.kind == 'remote':
        worker_control = worker_control_entry(control_state, worker.label)
        active = launch_remote_task(
            worker,
            round_kind=round_kind,
            run_name=run_name,
            task_state=task_state,
            dispatch_root=dispatch_root,
            launch_mode=str(worker_control.get('launch_mode') or DEFAULT_REMOTE_LAUNCH_MODE),
            screening_overrides=remote_screening_overrides,
        )
    else:
        raise ValueError(f'unknown worker kind `{worker.kind}`')
    active.stage_name = stage_name
    return active


def mark_task_failed(task_state: dict[str, Any], message: str, *, max_attempts: int) -> None:
    dispatch.mark_task_failed(
        task_state,
        message,
        max_attempts=max_attempts,
        finished_at=fidelity.ts_now(),
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


def build_workers(
    *,
    enable_remote: bool,
    local_python: str,
    local_label: str,
    remote_host: str,
    remote_repo: str,
    remote_python: str,
    remote_label: str,
    ssh_key: str | None,
) -> list[WorkerSpec]:
    return dispatch.build_workers(
        enable_remote=enable_remote,
        local_python=local_python,
        local_label=local_label,
        remote_host=remote_host,
        remote_repo=remote_repo,
        remote_python=remote_python,
        remote_label=remote_label,
        ssh_key=ssh_key,
    )


def active_stage_state(dispatch_state: dict[str, Any]) -> dict[str, Any]:
    stage = str(dispatch_state.get('stage'))
    if stage == 'seed1':
        return dispatch_state['seed1']
    if stage == 'seed2':
        stage_state = dispatch_state.get('seed2')
        if not isinstance(stage_state, dict):
            raise RuntimeError('seed2 stage is active but seed2 state is missing')
        return stage_state
    raise RuntimeError(f'unsupported active stage `{stage}`')


def apply_worker_control_requests(
    *,
    control_state: dict[str, Any],
    active: dict[str, ActiveTask],
) -> bool:
    changed = False
    for worker_label, active_task in list(active.items()):
        control = worker_control_entry(control_state, worker_label)
        if not bool(control.get('interrupt_requested')):
            continue
        interrupt_active_task(active_task)
        reset_task_after_operator_interrupt(
            active_task.task_state,
            note=f'worker `{worker_label}` paused by operator',
        )
        active.pop(worker_label, None)
        control['interrupt_requested'] = False
        changed = True
    return changed


def run_dispatch(args: argparse.Namespace) -> int:
    round_kind = normalize_round_kind(getattr(args, 'round_kind', ROUND_KIND_WINNER_REFINE))
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fidelity.acquire_run_lock(run_dir, args.run_name)
    atexit.register(fidelity.release_run_lock, lock_path)
    dispatch_root = ensure_dir(dispatch_root_for_run(run_dir, round_kind))
    dispatch_state_path = dispatch_state_path_for_run(run_dir, round_kind)
    dispatch_control_path = dispatch_control_path_for_run(run_dir, round_kind)
    context = load_round_context(run_dir, round_kind)
    try:
        if dispatch_state_path.exists():
            dispatch_state = load_dispatch_state(dispatch_state_path)
            reset_running_tasks_for_resume(dispatch_state)
        else:
            if round_kind == ROUND_KIND_PROTOCOL_DECIDE:
                dispatch_state = initialize_protocol_decide_dispatch_state(
                    context=context,
                    local_label=args.local_label,
                    remote_label=args.remote_label if not args.local_only else None,
                )
            elif round_kind == ROUND_KIND_ABLATION:
                dispatch_state = initialize_fixed_pool_multiseed_dispatch_state(
                    context=context,
                    local_label=args.local_label,
                    remote_label=args.remote_label if not args.local_only else None,
                )
            else:
                dispatch_state = initialize_dispatch_state(
                    context=context,
                    local_label=args.local_label,
                    remote_label=args.remote_label if not args.local_only else None,
                    seed2_min_keep=args.seed2_min_keep,
                    seed2_selection_gap=args.seed2_selection_gap,
                    seed2_max_keep=args.seed2_max_keep,
                )
        if dispatch_control_path.exists():
            control_state = load_dispatch_control(dispatch_control_path)
        else:
            control_state = initialize_dispatch_control_state(
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                remote_launch_mode=args.remote_launch_mode,
            )
            write_dispatch_control(dispatch_control_path, control_state)
        if ensure_control_state_workers(
            control_state=control_state,
            local_label=args.local_label,
            remote_label=args.remote_label if not args.local_only else None,
            remote_launch_mode=args.remote_launch_mode,
        ):
            write_dispatch_control(dispatch_control_path, control_state)
        write_dispatch_state(dispatch_state_path, dispatch_state)
        update_run_state_for_dispatch(
            run_dir=run_dir,
            dispatch_state_path=dispatch_state_path,
            dispatch_state=dispatch_state,
            round_kind=round_kind,
        )
        workers = build_workers(
            enable_remote=not args.local_only,
            local_python=args.local_python,
            local_label=args.local_label,
            remote_host=args.remote_host,
            remote_repo=args.remote_repo,
            remote_python=args.remote_python,
            remote_label=args.remote_label,
            ssh_key=args.ssh_key,
        )
        remote_screening_overrides = {
            'num_workers': int(args.remote_screening_num_workers),
            'file_batch_size': int(args.remote_screening_file_batch_size),
            'prefetch_factor': int(args.remote_screening_prefetch_factor),
            'val_file_batch_size': int(args.remote_screening_val_file_batch_size),
            'val_prefetch_factor': int(args.remote_screening_val_prefetch_factor),
        }
        active: dict[str, ActiveTask] = {}
        while True:
            control_state = load_dispatch_control(dispatch_control_path)
            if ensure_control_state_workers(
                control_state=control_state,
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                remote_launch_mode=args.remote_launch_mode,
            ):
                write_dispatch_control(dispatch_control_path, control_state)
            finished_labels: list[str] = []
            state_changed = False
            if apply_worker_control_requests(control_state=control_state, active=active):
                state_changed = True
                write_dispatch_control(dispatch_control_path, control_state)
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
                if stage_any_task_failed(stage_state):
                    raise RuntimeError(f'distributed {round_kind} seed1 exhausted retries')
                if stage_all_tasks_completed(stage_state):
                    if round_kind == ROUND_KIND_PROTOCOL_DECIDE:
                        maybe_promote_protocol_decide_seed1_to_seed2(
                            context=context,
                            dispatch_state=dispatch_state,
                            dispatch_state_path=dispatch_state_path,
                        )
                    elif round_kind == ROUND_KIND_ABLATION:
                        maybe_promote_fixed_pool_seed1_to_seed2(
                            context=context,
                            dispatch_state=dispatch_state,
                            dispatch_state_path=dispatch_state_path,
                        )
                    else:
                        maybe_promote_seed1_to_seed2(
                            context=context,
                            dispatch_state=dispatch_state,
                            dispatch_state_path=dispatch_state_path,
                        )
                    continue
            if dispatch_state.get('stage') == 'seed2' and not active:
                stage_state = dispatch_state['seed2']
                if stage_any_task_failed(stage_state):
                    raise RuntimeError(f'distributed {round_kind} seed2 exhausted retries')
                finalized = False
                if round_kind == ROUND_KIND_PROTOCOL_DECIDE:
                    finalized = maybe_finalize_or_expand_protocol_decide_dispatch(
                        context=context,
                        dispatch_state=dispatch_state,
                        dispatch_state_path=dispatch_state_path,
                    )
                elif round_kind == ROUND_KIND_ABLATION:
                    finalized = maybe_finalize_fixed_pool_dispatch(
                        context=context,
                        dispatch_state=dispatch_state,
                        dispatch_state_path=dispatch_state_path,
                    )
                else:
                    finalized = maybe_finalize_dispatch(
                        context=context,
                        dispatch_state=dispatch_state,
                        dispatch_state_path=dispatch_state_path,
                    )
                if finalized:
                    break
            if dispatch_state.get('stage') == 'completed':
                break
            stage_state = active_stage_state(dispatch_state)
            for worker in workers:
                worker_control = worker_control_entry(control_state, worker.label)
                if bool(worker_control.get('paused')):
                    continue
                if worker.label in active:
                    continue
                next_task = find_next_pending_task(stage_state)
                if next_task is None:
                    break
                _, task_state = next_task
                active[worker.label] = launch_task_for_worker(
                    worker=worker,
                    round_kind=round_kind,
                    run_name=args.run_name,
                    task_state=task_state,
                    dispatch_root=dispatch_root,
                    stage_name=str(dispatch_state['stage']),
                    control_state=control_state,
                    remote_screening_overrides=remote_screening_overrides,
                )
                write_dispatch_state(dispatch_state_path, dispatch_state)
            if not active and find_next_pending_task(active_stage_state(dispatch_state)) is None:
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
                round_kind=round_kind,
                status_override='failed',
            )
        raise
    finally:
        fidelity.release_run_lock(lock_path)


def print_status(args: argparse.Namespace) -> int:
    round_kind = normalize_round_kind(getattr(args, 'round_kind', ROUND_KIND_WINNER_REFINE))
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_state_path = dispatch_state_path_for_run(run_dir, round_kind)
    dispatch_control_path = dispatch_control_path_for_run(run_dir, round_kind)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state at {dispatch_state_path}')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    control_state = load_dispatch_control(dispatch_control_path) if dispatch_control_path.exists() else {'workers': {}}
    payload = {
        'round_kind': round_kind,
        'run_name': args.run_name,
        'stage': dispatch_state.get('stage'),
        'status': dispatch_state.get('status'),
        'seed1': summarize_dispatch_task_status(dispatch_state['seed1']),
        'seed2': (
            summarize_dispatch_task_status(dispatch_state['seed2'])
            if isinstance(dispatch_state.get('seed2'), dict)
            else None
        ),
        'workers': control_state.get('workers', {}),
    }
    if round_kind == ROUND_KIND_PROTOCOL_DECIDE:
        payload['seed2_plan'] = dispatch_state.get('seed2_plan')
        payload['final_protocol_winner'] = dispatch_state.get('final_protocol_winner')
    elif round_kind == ROUND_KIND_WINNER_REFINE:
        payload['seed2_selector'] = dispatch_state.get('seed2_selector')
        payload['final_front_runner'] = dispatch_state.get('final_front_runner')
        payload['final_p1_winner'] = dispatch_state.get('final_p1_winner') or dispatch_state.get('final_front_runner')
    else:
        payload['final_p1_winner'] = dispatch_state.get('final_p1_winner') or dispatch_state.get('final_front_runner')
    print(json.dumps(fidelity.normalize_payload(payload), ensure_ascii=False, indent=2))
    return 0


def update_worker_pause(args: argparse.Namespace, *, paused: bool) -> int:
    round_kind = normalize_round_kind(getattr(args, 'round_kind', ROUND_KIND_WINNER_REFINE))
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_state_path = dispatch_state_path_for_run(run_dir, round_kind)
    dispatch_control_path = dispatch_control_path_for_run(run_dir, round_kind)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state at {dispatch_state_path}')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    local_label = str(dispatch_state.get('local_label') or DEFAULT_LOCAL_LABEL)
    remote_label = dispatch_state.get('remote_label')
    if dispatch_control_path.exists():
        control_state = load_dispatch_control(dispatch_control_path)
    else:
        control_state = initialize_dispatch_control_state(
            local_label=local_label,
            remote_label=remote_label,
            remote_launch_mode=DEFAULT_REMOTE_LAUNCH_MODE,
        )
    ensure_control_state_workers(
        control_state=control_state,
        local_label=local_label,
        remote_label=remote_label,
        remote_launch_mode=DEFAULT_REMOTE_LAUNCH_MODE,
    )
    entry = set_worker_pause(
        control_state,
        worker_label=args.worker_label,
        paused=paused,
        stop_active=bool(getattr(args, 'stop_active', False)),
    )
    write_dispatch_control(dispatch_control_path, control_state)
    payload = {
        'round_kind': round_kind,
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

    dispatch = subparsers.add_parser(
        'dispatch',
        help='Run distributed P1 protocol_decide or winner_refine on the mainline, or manual backlog ablation, with a local worker and an optional SSH remote worker.',
    )
    dispatch.add_argument('--round-kind', choices=ROUND_KIND_CHOICES, default=ROUND_KIND_WINNER_REFINE)
    dispatch.add_argument('--run-name', required=True)
    dispatch.add_argument('--local-only', action='store_true')
    dispatch.add_argument('--local-python', default=sys.executable)
    dispatch.add_argument('--local-label', default=hostname_fallback())
    dispatch.add_argument('--remote-host', default=DEFAULT_REMOTE_HOST)
    dispatch.add_argument('--remote-repo', default=DEFAULT_REMOTE_REPO)
    dispatch.add_argument('--remote-python', default=DEFAULT_REMOTE_PYTHON)
    dispatch.add_argument('--remote-label', default=DEFAULT_REMOTE_LABEL)
    dispatch.add_argument('--ssh-key', default=DEFAULT_SSH_KEY)
    dispatch.add_argument('--poll-seconds', type=float, default=DEFAULT_POLL_SECONDS)
    dispatch.add_argument('--max-attempts', type=int, default=DEFAULT_MAX_ATTEMPTS)
    dispatch.add_argument('--seed2-min-keep', type=int, default=DEFAULT_SEED2_MIN_KEEP)
    dispatch.add_argument('--seed2-selection-gap', type=float, default=DEFAULT_SEED2_SELECTION_GAP)
    dispatch.add_argument('--seed2-max-keep', type=int, default=DEFAULT_SEED2_MAX_KEEP)
    dispatch.add_argument('--remote-launch-mode', choices=sorted(REMOTE_LAUNCH_MODES), default=DEFAULT_REMOTE_LAUNCH_MODE)
    dispatch.add_argument('--remote-screening-num-workers', type=int, default=DEFAULT_REMOTE_SCREENING_NUM_WORKERS)
    dispatch.add_argument('--remote-screening-file-batch-size', type=int, default=DEFAULT_REMOTE_SCREENING_FILE_BATCH_SIZE)
    dispatch.add_argument('--remote-screening-prefetch-factor', type=int, default=DEFAULT_REMOTE_SCREENING_PREFETCH_FACTOR)
    dispatch.add_argument('--remote-screening-val-file-batch-size', type=int, default=DEFAULT_REMOTE_SCREENING_VAL_FILE_BATCH_SIZE)
    dispatch.add_argument('--remote-screening-val-prefetch-factor', type=int, default=DEFAULT_REMOTE_SCREENING_VAL_PREFETCH_FACTOR)

    run_task = subparsers.add_parser(
        'run-task',
        help='Execute a single distributed P1 candidate+seed task and write a result json.',
    )
    run_task.add_argument('--round-kind', choices=ROUND_KIND_CHOICES, default=ROUND_KIND_WINNER_REFINE)
    run_task.add_argument('--run-name', required=True)
    run_task.add_argument('--candidate-arm', required=True)
    run_task.add_argument('--seed', type=int, required=True)
    run_task.add_argument('--machine-label', required=True)
    run_task.add_argument('--result-json', required=True)
    run_task.add_argument('--screening-num-workers', type=int)
    run_task.add_argument('--screening-file-batch-size', type=int)
    run_task.add_argument('--screening-prefetch-factor', type=int)
    run_task.add_argument('--screening-val-file-batch-size', type=int)
    run_task.add_argument('--screening-val-prefetch-factor', type=int)

    status = subparsers.add_parser(
        'status',
        help='Print distributed P1 dispatch status.',
    )
    status.add_argument('--round-kind', choices=ROUND_KIND_CHOICES, default=ROUND_KIND_WINNER_REFINE)
    status.add_argument('--run-name', required=True)

    pause_worker = subparsers.add_parser(
        'pause-worker',
        help='Pause a worker; optionally interrupt its active task and requeue it.',
    )
    pause_worker.add_argument('--round-kind', choices=ROUND_KIND_CHOICES, default=ROUND_KIND_WINNER_REFINE)
    pause_worker.add_argument('--run-name', required=True)
    pause_worker.add_argument('--worker-label', required=True)
    pause_worker.add_argument('--stop-active', action='store_true')

    resume_worker = subparsers.add_parser(
        'resume-worker',
        help='Resume scheduling on a paused worker.',
    )
    resume_worker.add_argument('--round-kind', choices=ROUND_KIND_CHOICES, default=ROUND_KIND_WINNER_REFINE)
    resume_worker.add_argument('--run-name', required=True)
    resume_worker.add_argument('--worker-label', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'run-task':
        result_json = Path(args.result_json)
        payload = execute_single_task(
            round_kind=getattr(args, 'round_kind', ROUND_KIND_WINNER_REFINE),
            run_name=args.run_name,
            candidate_arm=args.candidate_arm,
            seed=args.seed,
            result_json=result_json,
            machine_label=args.machine_label,
            screening_num_workers=args.screening_num_workers,
            screening_file_batch_size=args.screening_file_batch_size,
            screening_prefetch_factor=args.screening_prefetch_factor,
            screening_val_file_batch_size=args.screening_val_file_batch_size,
            screening_val_prefetch_factor=args.screening_val_prefetch_factor,
        )
        print(
            json.dumps(
                fidelity.normalize_payload(
                    build_run_task_cli_summary(payload, result_json=result_json)
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

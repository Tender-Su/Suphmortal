from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import distributed_dispatch as dispatch
import run_stage05_ab as ab
import run_stage05_fidelity as fidelity
import run_stage05_p1_only as p1_only


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_REMOTE_HOST = 'mahjong-laptop'
DEFAULT_REMOTE_REPO = str(REPO_ROOT)
DEFAULT_REMOTE_PYTHON = r'C:\Users\numbe\miniconda3\envs\mortal\python.exe'
DEFAULT_SSH_KEY = str(Path.home() / '.ssh' / 'mahjong_laptop_ed25519')
DEFAULT_LOCAL_LABEL = 'desktop'
DEFAULT_REMOTE_LABEL = 'laptop'
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_SEED2_MIN_KEEP = 3
DEFAULT_SEED2_SELECTION_GAP = 0.001
DEFAULT_SEED2_MAX_KEEP = 9
DISPATCH_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA_VERSION = 1


WorkerSpec = dispatch.WorkerSpec
ActiveTask = dispatch.ActiveTask
JsonTaskLaunchSpec = dispatch.JsonTaskLaunchSpec


def quote_ps(value: str) -> str:
    return dispatch.quote_ps(value)


def path_to_scp_remote(path: str | Path) -> str:
    return dispatch.path_to_scp_remote(path)


def dispatch_root_for_run(run_dir: Path) -> Path:
    return run_dir / 'distributed' / 'winner_refine_dispatch'


def dispatch_state_path_for_run(run_dir: Path) -> Path:
    return dispatch_root_for_run(run_dir) / 'dispatch_state.json'


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


def build_seed_round_name(seed: int) -> str:
    return f'p1_winner_refine_round__s{seed}'


def build_seed_ab_name(run_name: str, seed: int) -> str:
    return f'{run_name}_p1_winner_refine_s{seed}'


def load_dispatch_state(path: Path) -> dict[str, Any]:
    return fidelity.load_json(path)


def write_dispatch_state(path: Path, payload: dict[str, Any]) -> None:
    payload['updated_at'] = fidelity.ts_now()
    fidelity.atomic_write_json(path, payload)


def update_run_state_for_dispatch(
    *,
    run_dir: Path,
    dispatch_state_path: Path,
    dispatch_state: dict[str, Any],
    front_runner: str | None = None,
    final_round: dict[str, Any] | None = None,
    status_override: str | None = None,
) -> None:
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
        'dispatch_state_path': str(dispatch_state_path),
        'stage': dispatch_state.get('stage'),
        'status': dispatch_state.get('status'),
        'seed1_candidate_count': seed1_state.get('candidate_count'),
        'seed2_candidate_count': seed2_state.get('candidate_count'),
        'seed2_selected_arm_names': list(
            seed2_selector.get('selected_arm_names', [])
        ),
        'local_label': dispatch_state.get('local_label'),
        'remote_label': dispatch_state.get('remote_label'),
    }
    p1_state['winner_refine_dispatch'] = dispatch_summary
    if dispatch_state.get('winner_refine_centers'):
        p1_state['winner_refine_centers'] = list(dispatch_state['winner_refine_centers'])
    if final_round is not None:
        p1_state['winner_refine_round'] = final_round
    if front_runner:
        p1_state['winner_refine_front_runner'] = front_runner
        final_conclusion['p1_refine_front_runner'] = front_runner
    if status_override:
        state['status'] = status_override
    elif final_round is not None:
        if not state.get('final_conclusion', {}).get('p1_winner'):
            state['status'] = 'stopped_after_p1_winner_refine'
    else:
        state['status'] = 'running_p1_winner_refine'
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
    explicit_centers = fidelity.current_p1_winner_refine_explicit_center_arm_names(selected_protocol_arm)
    winner_centers = p1_only.select_protocol_centers(
        protocol_decide_round['ranking'],
        protocol_arm=selected_protocol_arm,
        explicit_arm_names=explicit_centers,
    )
    candidates = fidelity.build_p1_winner_refine_candidates(protocols, calibration, winner_centers)
    candidate_index = {candidate.arm_name: candidate for candidate in candidates}
    return {
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


def execute_single_task(
    *,
    run_name: str,
    candidate_arm: str,
    seed: int,
    result_json: Path,
    machine_label: str,
) -> dict[str, Any]:
    run_dir = fidelity.FIDELITY_ROOT / run_name
    context = load_refine_context(run_dir)
    candidate = context['candidate_index'].get(candidate_arm)
    if candidate is None:
        raise KeyError(f'unknown winner_refine candidate `{candidate_arm}`')
    raw_payload = fidelity.run_arm_cached(
        base_cfg=context['base_cfg'],
        grouped=context['grouped'],
        eval_splits=context['eval_splits'],
        candidate=candidate,
        seed=seed,
        step_scale=context['step_scale'],
        ab_name=build_seed_ab_name(run_name, seed),
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
    selected_names: list[str] = []
    floor_keep = max(int(min_keep), 1)
    for entry in pool:
        if max_keep is not None and len(selected_names) >= max_keep:
            break
        keep_due_to_floor = len(selected_names) < floor_keep
        keep_due_to_gap = seed2_selection_within_gap(
            entry,
            leader_selection_score=leader_score,
            selection_gap=selection_gap,
            leader_recent_loss=leader_recent_loss,
        )
        if keep_due_to_floor or keep_due_to_gap:
            selected_names.append(entry['arm_name'])
            continue
        if len(selected_names) >= floor_keep:
            break
    if len(selected_names) < floor_keep:
        for entry in pool:
            if entry['arm_name'] in selected_names:
                continue
            if max_keep is not None and len(selected_names) >= max_keep:
                break
            selected_names.append(entry['arm_name'])
            if len(selected_names) >= floor_keep:
                break
    selected_candidates = [candidate_index[name] for name in selected_names if name in candidate_index]
    if not selected_candidates:
        raise RuntimeError('winner_refine seed2 selector returned no candidates')
    details = {
        'mode': 'eligible_then_selection_gap',
        'pool_mode': 'eligible' if preferred_pool else 'valid',
        'min_keep': floor_keep,
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
    spec = JsonTaskLaunchSpec(
        task_id=str(task_state['task_id']),
        stage_name='',
        local_result_path=result_path,
        log_path=log_path,
        command_args=[
            'run-task',
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
    run_name: str,
    task_state: dict[str, Any],
    remote_result_path: Path,
) -> list[str]:
    return dispatch.build_remote_python_command(
        worker=worker,
        script_path=SCRIPT_PATH,
        remote_result_path=remote_result_path,
        command_args=[
            'run-task',
            '--run-name',
            run_name,
            '--candidate-arm',
            str(task_state['candidate_arm']),
            '--seed',
            str(task_state['seed']),
            '--machine-label',
            worker.label,
        ],
    )


def launch_remote_task(
    worker: WorkerSpec,
    *,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
) -> ActiveTask:
    remote_results_dir = dispatch_root / 'remote_results'
    local_results_dir = ensure_dir(dispatch_root / 'results')
    logs_dir = ensure_dir(dispatch_root / 'logs')
    remote_result_path = remote_results_dir / f'{task_state["task_id"]}.json'
    local_result_path = local_results_dir / f'{task_state["task_id"]}.json'
    log_path = logs_dir / f'{task_state["task_id"]}__{worker.label}.log'
    spec = JsonTaskLaunchSpec(
        task_id=str(task_state['task_id']),
        stage_name='',
        local_result_path=local_result_path,
        log_path=log_path,
        remote_result_path=remote_result_path,
        command_args=[
            'run-task',
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


def fetch_remote_result(worker: WorkerSpec, remote_result_path: str, local_result_path: Path) -> None:
    dispatch.fetch_remote_result(worker, remote_result_path, local_result_path)


def load_task_result(path: Path) -> dict[str, Any]:
    return fidelity.load_json(path)


def build_seed_round_payload(
    *,
    context: dict[str, Any],
    candidates: list[fidelity.CandidateSpec],
    actual_seed: int,
    task_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    round_name = build_seed_round_name(actual_seed)
    ab_name = build_seed_ab_name(context['run_name'], actual_seed)
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
    write_dispatch_state(dispatch_state_path, dispatch_state)
    update_run_state_for_dispatch(
        run_dir=context['run_dir'],
        dispatch_state_path=dispatch_state_path,
        dispatch_state=dispatch_state,
        front_runner=front_runner,
        final_round=final_round,
    )
    return True


def launch_task_for_worker(
    *,
    worker: WorkerSpec,
    run_name: str,
    task_state: dict[str, Any],
    dispatch_root: Path,
    stage_name: str,
) -> ActiveTask:
    if worker.kind == 'local':
        active = launch_local_task(
            worker,
            run_name=run_name,
            task_state=task_state,
            dispatch_root=dispatch_root,
        )
    elif worker.kind == 'remote':
        active = launch_remote_task(
            worker,
            run_name=run_name,
            task_state=task_state,
            dispatch_root=dispatch_root,
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


def run_dispatch(args: argparse.Namespace) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fidelity.acquire_run_lock(run_dir, args.run_name)
    atexit.register(fidelity.release_run_lock, lock_path)
    dispatch_root = ensure_dir(dispatch_root_for_run(run_dir))
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    context = load_refine_context(run_dir)
    try:
        if dispatch_state_path.exists():
            dispatch_state = load_dispatch_state(dispatch_state_path)
            reset_running_tasks_for_resume(dispatch_state)
        else:
            dispatch_state = initialize_dispatch_state(
                context=context,
                local_label=args.local_label,
                remote_label=args.remote_label if not args.local_only else None,
                seed2_min_keep=args.seed2_min_keep,
                seed2_selection_gap=args.seed2_selection_gap,
                seed2_max_keep=args.seed2_max_keep,
            )
        write_dispatch_state(dispatch_state_path, dispatch_state)
        update_run_state_for_dispatch(
            run_dir=run_dir,
            dispatch_state_path=dispatch_state_path,
            dispatch_state=dispatch_state,
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
        active: dict[str, ActiveTask] = {}
        while True:
            finished_labels: list[str] = []
            for worker_label, active_task in list(active.items()):
                if active_task.process.poll() is None:
                    continue
                handle_finished_task(active=active_task, max_attempts=args.max_attempts)
                finished_labels.append(worker_label)
            for worker_label in finished_labels:
                active.pop(worker_label, None)
            if finished_labels:
                write_dispatch_state(dispatch_state_path, dispatch_state)
            if dispatch_state.get('stage') == 'seed1' and not active:
                stage_state = dispatch_state['seed1']
                if stage_any_task_failed(stage_state):
                    raise RuntimeError('distributed winner_refine seed1 exhausted retries')
                if stage_all_tasks_completed(stage_state):
                    maybe_promote_seed1_to_seed2(
                        context=context,
                        dispatch_state=dispatch_state,
                        dispatch_state_path=dispatch_state_path,
                    )
                    continue
            if dispatch_state.get('stage') == 'seed2' and not active:
                stage_state = dispatch_state['seed2']
                if stage_any_task_failed(stage_state):
                    raise RuntimeError('distributed winner_refine seed2 exhausted retries')
                if maybe_finalize_dispatch(
                    context=context,
                    dispatch_state=dispatch_state,
                    dispatch_state_path=dispatch_state_path,
                ):
                    break
            if dispatch_state.get('stage') == 'completed':
                break
            stage_state = active_stage_state(dispatch_state)
            for worker in workers:
                if worker.label in active:
                    continue
                next_task = find_next_pending_task(stage_state)
                if next_task is None:
                    break
                _, task_state = next_task
                active[worker.label] = launch_task_for_worker(
                    worker=worker,
                    run_name=args.run_name,
                    task_state=task_state,
                    dispatch_root=dispatch_root,
                    stage_name=str(dispatch_state['stage']),
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
                status_override='failed',
            )
        raise
    finally:
        fidelity.release_run_lock(lock_path)


def print_status(args: argparse.Namespace) -> int:
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    dispatch_state_path = dispatch_state_path_for_run(run_dir)
    if not dispatch_state_path.exists():
        raise FileNotFoundError(f'missing dispatch state at {dispatch_state_path}')
    dispatch_state = load_dispatch_state(dispatch_state_path)
    payload = {
        'run_name': args.run_name,
        'stage': dispatch_state.get('stage'),
        'status': dispatch_state.get('status'),
        'seed1': summarize_dispatch_task_status(dispatch_state['seed1']),
        'seed2': (
            summarize_dispatch_task_status(dispatch_state['seed2'])
            if isinstance(dispatch_state.get('seed2'), dict)
            else None
        ),
        'seed2_selector': dispatch_state.get('seed2_selector'),
        'final_front_runner': dispatch_state.get('final_front_runner'),
    }
    print(json.dumps(fidelity.normalize_payload(payload), ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    dispatch = subparsers.add_parser(
        'dispatch',
        help='Run distributed P1 winner_refine with a local worker and an optional SSH remote worker.',
    )
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

    run_task = subparsers.add_parser(
        'run-task',
        help='Execute a single winner_refine candidate+seed task and write a result json.',
    )
    run_task.add_argument('--run-name', required=True)
    run_task.add_argument('--candidate-arm', required=True)
    run_task.add_argument('--seed', type=int, required=True)
    run_task.add_argument('--machine-label', required=True)
    run_task.add_argument('--result-json', required=True)

    status = subparsers.add_parser(
        'status',
        help='Print distributed winner_refine dispatch status.',
    )
    status.add_argument('--run-name', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'run-task':
        payload = execute_single_task(
            run_name=args.run_name,
            candidate_arm=args.candidate_arm,
            seed=args.seed,
            result_json=Path(args.result_json),
            machine_label=args.machine_label,
        )
        print(json.dumps(fidelity.normalize_payload(payload), ensure_ascii=False, indent=2))
        return
    if args.command == 'dispatch':
        raise SystemExit(run_dispatch(args))
    if args.command == 'status':
        raise SystemExit(print_status(args))
    raise ValueError(f'unknown command `{args.command}`')


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import atexit
import json
import traceback
from pathlib import Path

import run_stage05_ab as ab
import run_stage05_fidelity as fidelity
import stage05_current_defaults as stage05_defaults


DEFAULT_RUN_NAME = 's05_fidelity_p1_top3'
DEFAULT_P1_SEED = ab.BASE_SCREENING['seed'] + 1000
FROZEN_TOP3 = stage05_defaults.CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS


def build_protocol_arm_map() -> dict[str, tuple[str, str, str, str]]:
    arms: dict[str, tuple[str, str, str, str]] = {}
    for scheduler_prefix, scheduler_profile in ab.SCHEDULER_PREFIXES:
        for curriculum_prefix, curriculum_profile in ab.CURRICULUM_PREFIXES:
            for weight_prefix, weight_profile in ab.WEIGHT_PREFIXES:
                for window_prefix, window_profile in ab.WINDOW_PREFIXES:
                    arm_name = (
                        f'{scheduler_prefix}_{curriculum_prefix}{weight_prefix}{window_prefix}_'
                        f'{scheduler_profile}_{curriculum_profile}_{weight_profile}_{window_profile}'
                    )
                    arms[arm_name] = (
                        scheduler_profile,
                        curriculum_profile,
                        weight_profile,
                        window_profile,
                    )
    return arms


PROTOCOL_ARM_MAP = build_protocol_arm_map()


def dedupe_protocol_arms(protocol_arms: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for protocol_arm in protocol_arms:
        if protocol_arm in seen:
            continue
        seen.add(protocol_arm)
        unique.append(protocol_arm)
    return unique


def build_protocol_candidate(protocol_arm: str) -> fidelity.CandidateSpec:
    try:
        scheduler_profile, curriculum_profile, weight_profile, window_profile = PROTOCOL_ARM_MAP[protocol_arm]
    except KeyError as exc:
        raise ValueError(f'unknown protocol arm: {protocol_arm}') from exc
    return fidelity.CandidateSpec(
        arm_name=protocol_arm,
        scheduler_profile=scheduler_profile,
        curriculum_profile=curriculum_profile,
        weight_profile=weight_profile,
        window_profile=window_profile,
        cfg_overrides={},
        meta={
            'stage': 'P0',
            'protocol_arm': protocol_arm,
            'selection_source': 'frozen_top3' if protocol_arm in FROZEN_TOP3 else 'cli',
        },
    )


def build_protocol_candidates(protocol_arms: list[str]) -> list[fidelity.CandidateSpec]:
    resolved_arms = dedupe_protocol_arms(protocol_arms)
    if not resolved_arms:
        raise ValueError('at least one protocol arm is required')
    return [build_protocol_candidate(protocol_arm) for protocol_arm in resolved_arms]


def build_p1_search_space(calibration: dict[str, object]) -> dict[str, object]:
    return fidelity.apply_protocol_decide_progressive_settings({
        'budget_ratios': calibration['budget_ratios'],
        'calibration_mode': calibration.get('calibration_mode', fidelity.P1_CALIBRATION_DEFAULT_MODE),
        'calibration_mode_note': calibration.get(
            'calibration_mode_note',
            fidelity.p1_calibration_mode_note(
                str(calibration.get('calibration_mode', fidelity.P1_CALIBRATION_DEFAULT_MODE))
            ),
        ),
        'calibration_protocol_arms': list(calibration.get('calibration_protocol_arms', [])),
        'inherited_single_head_source': calibration.get('inherited_single_head_source'),
        'protocol_decide_coordinate_mode': fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE,
        'protocol_decide_total_budget_ratios': list(fidelity.P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS),
        'protocol_decide_mixes': fidelity.current_protocol_decide_mix_payload(),
        'ranking_mode': fidelity.P1_RANKING_MODE,
        'eligibility_group_key': fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        'policy_loss_epsilon': fidelity.P1_POLICY_LOSS_EPSILON,
        'old_regression_policy_loss_epsilon': fidelity.P1_OLD_REGRESSION_POLICY_EPSILON,
        'budget_ratio_digits': fidelity.P1_BUDGET_RATIO_DIGITS,
        'aux_weight_digits': fidelity.P1_AUX_WEIGHT_DIGITS,
        'mapping_mode': calibration.get('mapping_mode', fidelity.P1_CALIBRATION_MAPPING_MODE),
        'combo_scheme': calibration.get('combo_scheme', fidelity.P1_CALIBRATION_SCHEME),
        'opp_weight_per_budget_unit': calibration['opp_weight_per_budget_unit'],
        'danger_weight_per_budget_unit': calibration['danger_weight_per_budget_unit'],
        'calibration_seed_offsets': list(fidelity.P1_CALIBRATION_SEED_OFFSETS),
        'grad_probe_max_batches': fidelity.P1_GRAD_CALIBRATION_MAX_BATCHES,
        'protocol_decide_seed_offsets': list(fidelity.P1_PROTOCOL_DECIDE_SEED_OFFSETS),
        'protocol_decide_seed_strategy': 'progressive_probe_then_expand',
        'protocol_decide_probe_keep_per_protocol': fidelity.P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
        **fidelity.current_p1_winner_refine_search_space_payload(),
        'ablation_seed_offsets': list(fidelity.P1_ABLATION_SEED_OFFSETS),
        'rank_opp_combo_factor': calibration.get('rank_opp_combo_factor', 1.0),
        'rank_danger_combo_factor': calibration.get('rank_danger_combo_factor', 1.0),
        'opp_danger_combo_factor': calibration.get(
            'opp_danger_combo_factor',
            calibration.get('joint_combo_factor', 1.0),
        ),
        'triple_combo_factor': calibration.get('triple_combo_factor', 1.0),
        'protocol_rank_opp_combo_factors': calibration.get('protocol_rank_opp_combo_factors', {}),
        'protocol_rank_danger_combo_factors': calibration.get('protocol_rank_danger_combo_factors', {}),
        'protocol_opp_danger_combo_factors': calibration.get(
            'protocol_opp_danger_combo_factors',
            calibration.get('protocol_joint_combo_factors', {}),
        ),
        'protocol_triple_combo_factors': calibration.get('protocol_triple_combo_factors', {}),
        'joint_combo_factor': calibration.get('joint_combo_factor', 1.0),
        'protocol_joint_combo_factors': calibration.get('protocol_joint_combo_factors', {}),
        'probe_weight': fidelity.P1_CALIBRATION_PROBE_WEIGHT,
        'selection_policy': fidelity.p1_selection_policy_metadata(),
    })


def normalize_protocol_arm_list(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if not normalized:
        return None
    return dedupe_protocol_arms(normalized)


def normalize_candidate_name_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def infer_protocol_arm_from_candidate_name(candidate_arm: str) -> str:
    return str(candidate_arm).split('__', 1)[0].strip()


def hydrate_post_protocol_decide_search_space(
    *,
    p1_state: dict,
    final_conclusion: dict,
    require_winner_refine_selection: bool,
    require_effective_precision: bool,
) -> dict[str, object]:
    raw_search_space = p1_state.get('search_space')
    search_space: dict[str, object] = dict(raw_search_space) if isinstance(raw_search_space, dict) else {}
    search_space.pop('winner_refine_centers', None)
    if (
        str(search_space.get('protocol_decide_progressive_ambiguity_mode', '') or '').strip()
        == fidelity.P1_PROGRESSIVE_AMBIGUITY_MODE_LEGACY
    ):
        search_space.pop('protocol_decide_progressive_ambiguity_mode', None)
        search_space.pop('protocol_decide_progressive_gap_threshold', None)
    selected_protocol_arm = str(
        p1_state.get('selected_protocol_arm')
        or final_conclusion.get('p1_protocol_winner')
        or ''
    ).strip()
    explicit_center_arm_names = normalize_candidate_name_list(search_space.get('winner_refine_center_arm_names'))
    persisted_centers = explicit_center_arm_names or normalize_candidate_name_list(p1_state.get('winner_refine_centers'))
    if persisted_centers and any(fidelity.is_budget_triplet_arm_name(item) for item in persisted_centers):
        persisted_centers = []
        search_space.pop('winner_refine_center_mode', None)
        search_space.pop('winner_refine_center_protocol_arm', None)
        search_space.pop('winner_refine_center_arm_names', None)
    if persisted_centers:
        search_space['winner_refine_center_mode'] = 'explicit_arm_names'
        search_space['winner_refine_center_arm_names'] = list(persisted_centers)
        if not str(search_space.get('winner_refine_center_protocol_arm', '') or '').strip():
            inferred_protocol_arm = selected_protocol_arm or infer_protocol_arm_from_candidate_name(persisted_centers[0])
            if inferred_protocol_arm:
                search_space['winner_refine_center_protocol_arm'] = inferred_protocol_arm
    if not str(search_space.get('winner_refine_center_mode', '') or '').strip():
        search_space['winner_refine_center_mode'] = fidelity.P1_WINNER_REFINE_CENTER_MODE
    if str(search_space.get('winner_refine_center_mode', '') or '').strip() == 'top_ranked_keep':
        search_space.setdefault('winner_refine_center_keep', fidelity.P1_WINNER_REFINE_CENTER_KEEP)

    mode = str(search_space.get('winner_refine_center_mode', '') or '').strip()
    if require_winner_refine_selection:
        if mode == 'explicit_arm_names':
            explicit_center_arm_names = normalize_candidate_name_list(search_space.get('winner_refine_center_arm_names'))
            if not explicit_center_arm_names:
                raise RuntimeError(
                    'existing p1-only run already completed protocol_decide, but persisted winner_refine centers are missing; '
                    'cannot safely continue winner_refine with current defaults'
                )
        elif mode == 'top_ranked_keep':
            if search_space.get('winner_refine_center_keep') is None:
                raise RuntimeError(
                    'existing p1-only run already completed protocol_decide, but winner_refine_center_keep is missing; '
                    'cannot safely continue winner_refine with current defaults'
                )
        else:
            raise RuntimeError(
                'existing p1-only run already completed protocol_decide, but persisted search_space does not preserve '
                'a recoverable winner_refine center selection; restore the original search_space or winner_refine_centers'
            )

    reference_arm_names = list(persisted_centers)
    winner_refine_round = p1_state.get('winner_refine_round')
    if isinstance(winner_refine_round, dict):
        reference_arm_names.extend(
            str(entry.get('arm_name')).strip()
            for entry in winner_refine_round.get('ranking', [])
            if isinstance(entry, dict) and str(entry.get('arm_name', '')).strip()
        )
    if (
        search_space.get('budget_ratio_digits') is None
        or search_space.get('aux_weight_digits') is None
    ):
        inferred_precision = fidelity.infer_effective_precision_from_arm_names(reference_arm_names)
        if inferred_precision is not None:
            budget_digits, aux_digits = inferred_precision
            search_space.setdefault('budget_ratio_digits', budget_digits)
            search_space.setdefault('aux_weight_digits', aux_digits)
    search_space.setdefault('protocol_decide_coordinate_mode', fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE)
    search_space.setdefault('budget_ratio_digits', fidelity.P1_BUDGET_RATIO_DIGITS)
    search_space.setdefault('aux_weight_digits', fidelity.P1_AUX_WEIGHT_DIGITS)
    coordinate_mode = fidelity.protocol_decide_coordinate_mode_from_search_space(search_space)
    if (
        require_effective_precision
        and coordinate_mode == fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE
        and (
            search_space.get('budget_ratio_digits') is None
            or search_space.get('aux_weight_digits') is None
        )
    ):
        raise RuntimeError(
            'existing p1-only run already completed protocol_decide under effective coordinates, '
            'but persisted search_space precision is missing; cannot safely continue with current defaults'
        )
    if (
        coordinate_mode == fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE
        and not require_effective_precision
        and (
            search_space.get('budget_ratio_digits') is None
            or search_space.get('aux_weight_digits') is None
        )
    ):
        return fidelity.apply_protocol_decide_progressive_settings(search_space)
    return fidelity.apply_protocol_decide_progressive_settings(search_space)


def infer_resume_seed(state: dict) -> int | None:
    try:
        stored_seed = int(state.get('seed'))
    except (TypeError, ValueError):
        stored_seed = None
    if stored_seed is not None:
        return stored_seed

    p1_state = state.get('p1')
    if not isinstance(p1_state, dict):
        return None

    for round_key, round_seed_offset in (
        ('ablation_round', 707),
        ('winner_refine_round', 606),
        ('protocol_decide_round', 505),
        ('calibration_round', 404),
    ):
        round_payload = p1_state.get(round_key)
        if not isinstance(round_payload, dict):
            continue
        try:
            return int(round_payload.get('seed')) - round_seed_offset
        except (TypeError, ValueError):
            continue
    return None


def resolve_continue_run_config(
    state_path: Path,
    *,
    seed: int | None,
    protocol_arms: list[str] | None,
    calibration_protocol_arms: list[str] | None,
    calibration_mode: str | None,
) -> tuple[dict, int, list[str], list[str], str]:
    if not state_path.exists():
        raise ValueError(
            'continue mode requires an existing state.json; use the original --run-name or start a new run.'
        )

    state = fidelity.load_json(state_path)
    p1_state = state.get('p1') if isinstance(state.get('p1'), dict) else {}
    final_conclusion = (
        state.get('final_conclusion')
        if isinstance(state.get('final_conclusion'), dict)
        else {}
    )
    search_space = p1_state.get('search_space') if isinstance(p1_state.get('search_space'), dict) else {}
    calibration = p1_state.get('calibration') if isinstance(p1_state.get('calibration'), dict) else {}

    existing_protocol_arms = (
        normalize_protocol_arm_list(p1_state.get('protocol_arms'))
        or normalize_protocol_arm_list(state.get('selected_protocol_arms'))
        or normalize_protocol_arm_list(final_conclusion.get('p1_entry_protocols'))
    )
    if existing_protocol_arms is None:
        if protocol_arms is None:
            raise ValueError(
                'existing p1-only run is missing protocol_arms; rerun with the original --protocol-arm values '
                'or use a new run-name.'
            )
        existing_protocol_arms = dedupe_protocol_arms(protocol_arms)

    existing_calibration_protocol_arms = (
        normalize_protocol_arm_list(p1_state.get('calibration_protocol_arms'))
        or normalize_protocol_arm_list(search_space.get('calibration_protocol_arms'))
        or normalize_protocol_arm_list(calibration.get('calibration_protocol_arms'))
    )
    if existing_calibration_protocol_arms is None:
        if calibration_protocol_arms is None:
            raise ValueError(
                'existing p1-only run is missing calibration_protocol_arms; rerun with the original '
                '--calibration-protocol-arm values or use a new run-name.'
            )
        existing_calibration_protocol_arms = dedupe_protocol_arms(calibration_protocol_arms)

    existing_calibration_mode = str(
        p1_state.get('calibration_mode')
        or search_space.get('calibration_mode')
        or calibration.get('calibration_mode')
        or ''
    ).strip()
    if not existing_calibration_mode:
        if calibration_mode is None:
            raise ValueError(
                'existing p1-only run is missing calibration_mode; rerun with the original '
                '--calibration-mode value or use a new run-name.'
            )
        existing_calibration_mode = calibration_mode

    existing_seed = infer_resume_seed(state)
    if existing_seed is None:
        if seed is None:
            raise ValueError(
                'existing p1-only run does not record a recoverable base seed; rerun with the original '
                '--seed value or use a new run-name.'
            )
        existing_seed = seed

    if seed is not None and seed != existing_seed:
        raise ValueError(
            f'continue mode seed mismatch: requested {seed}, existing run uses {existing_seed}. '
            'Use the original --seed or start a new run-name.'
        )

    if protocol_arms is not None:
        requested_protocol_arms = dedupe_protocol_arms(protocol_arms)
        if requested_protocol_arms != existing_protocol_arms:
            raise ValueError(
                'continue mode protocol_arms mismatch; this run already uses '
                f'{existing_protocol_arms}, not {requested_protocol_arms}. Use a new run-name for a new search space.'
            )

    if calibration_protocol_arms is not None:
        requested_calibration_protocol_arms = dedupe_protocol_arms(calibration_protocol_arms)
        if requested_calibration_protocol_arms != existing_calibration_protocol_arms:
            raise ValueError(
                'continue mode calibration_protocol_arms mismatch; this run already uses '
                f'{existing_calibration_protocol_arms}, not {requested_calibration_protocol_arms}. '
                'Use a new run-name for a new search space.'
            )

    if calibration_mode is not None and calibration_mode != existing_calibration_mode:
        raise ValueError(
            'continue mode calibration_mode mismatch; this run already uses '
            f'{existing_calibration_mode}, not {calibration_mode}. Use a new run-name for a new search space.'
        )

    return (
        state,
        existing_seed,
        existing_protocol_arms,
        existing_calibration_protocol_arms,
        existing_calibration_mode,
    )


def initialize_state(
    state: dict,
    *,
    seed: int,
    protocol_arms: list[str],
    started_at: str | None = None,
) -> dict:
    state.clear()
    state['started_at'] = started_at or fidelity.ts_now()
    state['seed'] = seed
    state['mode'] = 'p1_only'
    state['selected_protocol_arms'] = list(protocol_arms)
    state['status'] = 'running_p1_calibration'
    state['p1'] = {
        'protocol_arms': list(protocol_arms),
        'selection_policy': fidelity.p1_selection_policy_metadata(),
    }
    state['final_conclusion'] = {
        'p1_entry_protocols': list(protocol_arms),
    }
    return state


def ensure_state_metadata(
    state: dict,
    *,
    seed: int,
    protocol_arms: list[str],
    started_at: str | None = None,
) -> dict:
    if 'started_at' not in state:
        state['started_at'] = started_at or fidelity.ts_now()
    state['seed'] = seed
    state['mode'] = 'p1_only'
    state['selected_protocol_arms'] = list(protocol_arms)
    p1_state = state.setdefault('p1', {})
    p1_state['protocol_arms'] = list(protocol_arms)
    p1_state['selection_policy'] = fidelity.p1_selection_policy_metadata()
    final_conclusion = state.setdefault('final_conclusion', {})
    final_conclusion['p1_entry_protocols'] = list(protocol_arms)
    return state


def has_completed_p1_results(state: dict) -> bool:
    if state.get('status') != 'completed':
        return False
    p1_state = state.get('p1')
    if not isinstance(p1_state, dict):
        return False
    final_conclusion = state.get('final_conclusion')
    if not isinstance(final_conclusion, dict):
        final_conclusion = {}
    return (
        isinstance(p1_state.get('ablation_round'), dict)
        or isinstance(p1_state.get('winner_refine_round'), dict)
        or bool(final_conclusion.get('p1_winner'))
    )


def build_protocol_compare(ranked: list[dict]) -> list[dict]:
    eligible_ranked = [
        entry
        for entry in ranked
        if fidelity.is_p1_protocol_compare_meta(entry.get('candidate_meta', {}))
    ]
    winner_names = set(fidelity.select_group_top_k(eligible_ranked, 'protocol_arm', 1, valid_only=True))
    winners = [entry for entry in eligible_ranked if entry['arm_name'] in winner_names]
    if not winners:
        raise RuntimeError('p1_protocol_decide_round produced no valid all_three protocol winners')
    return fidelity.rank_round_entries(
        winners,
        ranking_mode=fidelity.P1_RANKING_MODE,
    )


def select_protocol_centers(
    ranked: list[dict],
    *,
    protocol_arm: str,
    keep: int | None = None,
    explicit_arm_names: list[str] | tuple[str, ...] | None = None,
) -> list[fidelity.CandidateSpec]:
    return fidelity.select_p1_protocol_centers(
        ranked,
        protocol_arm=protocol_arm,
        keep=keep,
        explicit_arm_names=explicit_arm_names,
    )


def run_p1_only(
    *,
    run_dir: Path,
    seed: int | None = None,
    protocol_arms: list[str] | None = None,
    calibration_protocol_arms: list[str] | None = None,
    calibration_mode: str | None = None,
    stop_after_calibration: bool = False,
    continue_to_winner_refine: bool = False,
    continue_to_ablation: bool = False,
) -> dict:
    continue_to_winner_refine = continue_to_winner_refine or continue_to_ablation
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name
    lock_path = fidelity.acquire_run_lock(run_dir, run_name)
    atexit.register(fidelity.release_run_lock, lock_path)
    state: dict | None = None
    try:
        state_path = run_dir / 'state.json'
        resume_existing_run = state_path.exists()
        if resume_existing_run:
            existing_state, seed, protocol_arms, resolved_calibration_protocol_arms, calibration_mode = (
                resolve_continue_run_config(
                    state_path,
                    seed=seed,
                    protocol_arms=protocol_arms,
                    calibration_protocol_arms=calibration_protocol_arms,
                    calibration_mode=calibration_mode,
                )
            )
            started_at = str(existing_state.get('started_at') or fidelity.ts_now())
            state = existing_state
        else:
            seed = DEFAULT_P1_SEED if seed is None else seed
            protocol_arms = dedupe_protocol_arms(protocol_arms or list(FROZEN_TOP3))
            resolved_calibration_protocol_arms = dedupe_protocol_arms(
                calibration_protocol_arms or list(fidelity.P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS)
            )
            calibration_mode = calibration_mode or fidelity.P1_CALIBRATION_DEFAULT_MODE
            started_at = fidelity.ts_now()
            state = {}
            initialize_state(
                state,
                seed=seed,
                protocol_arms=protocol_arms,
                started_at=started_at,
            )
            fidelity.atomic_write_json(state_path, state)
            fidelity.update_results_doc(run_dir, state)

        ensure_state_metadata(
            state,
            seed=seed,
            protocol_arms=protocol_arms,
            started_at=started_at,
        )
        state.pop('fatal_error', None)
        state.pop('fatal_traceback', None)

        base_cfg = ab.build_base_config()
        grouped = ab.group_files_by_month(ab.load_all_files())
        protocols = build_protocol_candidates(protocol_arms)
        calibration_protocols = build_protocol_candidates(resolved_calibration_protocol_arms)
        p1_state = state['p1']
        p1_state['protocols'] = [
            fidelity.candidate_cache_payload(candidate, include_meta=True)
            for candidate in protocols
        ]
        p1_state['calibration_mode'] = calibration_mode
        p1_state['calibration_mode_note'] = fidelity.p1_calibration_mode_note(calibration_mode)
        p1_state['calibration_protocol_arms'] = list(resolved_calibration_protocol_arms)

        eval_splits = ab.build_eval_splits(grouped, seed + 55, ab.BASE_SCREENING['eval_files'])
        calibration_eval_splits = {
            **eval_splits,
            'old_regression_files': [],
        }
        rank_template_mean = fidelity.rank_weight_mean_for_files(
            eval_splits['full_recent_files'],
            version=int(base_cfg['control']['version']),
            file_batch_size=int(ab.BASE_SCREENING['file_batch_size']),
        )
        p1_state['shared_rank_template_mean'] = rank_template_mean
        calibration_round = p1_state.get('calibration_round') if resume_existing_run else None
        calibration = p1_state.get('calibration') if isinstance(p1_state.get('calibration'), dict) else None
        if not isinstance(calibration_round, dict) or calibration is None:
            state['status'] = 'running_p1_calibration'
            fidelity.atomic_write_json(state_path, state)
            fidelity.update_results_doc(run_dir, state)

            calibration_round = fidelity.execute_round_multiseed(
                run_dir=run_dir,
                round_name='p1_calibration',
                ab_name=f'{run_name}_p1_calibration',
                base_cfg=base_cfg,
                grouped=grouped,
                eval_splits=calibration_eval_splits,
                candidates=fidelity.build_p1_calibration_candidates(
                    calibration_protocols,
                    rank_template_mean,
                    calibration_mode=calibration_mode,
                ),
                seed=seed + 404,
                seed_offsets=fidelity.P1_CALIBRATION_SEED_OFFSETS,
                step_scale=fidelity.P1_CALIBRATION_STEP_SCALE,
                ranking_mode=fidelity.P1_RANKING_MODE,
                eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
            )
            p1_state['calibration_round'] = calibration_round
            calibration = fidelity.derive_p1_budget_calibration(
                calibration_round,
                requested_calibration_mode=calibration_mode,
                inherited_single_head=(
                    fidelity.P1_SINGLE_HEAD_CALIBRATION_BASELINE
                    if calibration_mode == fidelity.P1_CALIBRATION_MODE_COMBO_ONLY
                    else None
                ),
                inherited_single_head_source=(
                    fidelity.P1_SINGLE_HEAD_CALIBRATION_SOURCE
                    if calibration_mode == fidelity.P1_CALIBRATION_MODE_COMBO_ONLY
                    else None
                ),
            )
            calibration['shared_rank_template_mean'] = rank_template_mean
            calibration['calibration_protocol_arms'] = list(resolved_calibration_protocol_arms)
            p1_state['calibration'] = calibration
            p1_state['search_space'] = build_p1_search_space(calibration)
            fidelity.atomic_write_json(state_path, state)
            fidelity.update_results_doc(run_dir, state)
        else:
            calibration['shared_rank_template_mean'] = rank_template_mean
            calibration['calibration_protocol_arms'] = list(resolved_calibration_protocol_arms)
            p1_state['calibration'] = calibration
            protocol_decide_round = p1_state.get('protocol_decide_round')
            winner_refine_round = p1_state.get('winner_refine_round')
            if isinstance(protocol_decide_round, dict):
                need_winner_refine_rebuild = continue_to_winner_refine and not isinstance(winner_refine_round, dict)
                need_effective_precision = need_winner_refine_rebuild or (
                    continue_to_ablation and isinstance(winner_refine_round, dict)
                )
                p1_state['search_space'] = hydrate_post_protocol_decide_search_space(
                    p1_state=p1_state,
                    final_conclusion=state['final_conclusion'],
                    require_winner_refine_selection=need_winner_refine_rebuild,
                    require_effective_precision=need_effective_precision,
                )
            elif not isinstance(p1_state.get('search_space'), dict):
                p1_state['search_space'] = build_p1_search_space(calibration)
            else:
                p1_state['search_space'] = fidelity.apply_protocol_decide_progressive_settings(
                    p1_state.get('search_space'),
                )

        if stop_after_calibration:
            state['status'] = 'stopped_after_p1_calibration'
        else:
            search_space = fidelity.apply_protocol_decide_progressive_settings(
                p1_state.get('search_space'),
            )
            p1_state['search_space'] = search_space
            protocol_decide_round = (
                p1_state.get('protocol_decide_round') if isinstance(p1_state.get('protocol_decide_round'), dict) else None
            )
            protocol_compare_payload = (
                p1_state.get('protocol_compare') if isinstance(p1_state.get('protocol_compare'), dict) else {}
            )
            protocol_compare = (
                list(protocol_compare_payload.get('ranking', []))
                if isinstance(protocol_compare_payload.get('ranking'), list)
                else []
            )
            selected_protocol_arm = str(
                p1_state.get('selected_protocol_arm')
                or state['final_conclusion'].get('p1_protocol_winner')
                or ''
            ).strip()
            if not isinstance(protocol_decide_round, dict):
                state['status'] = 'running_p1_protocol_decide'
                fidelity.atomic_write_json(state_path, state)
                fidelity.update_results_doc(run_dir, state)

                protocol_decide_round = fidelity.execute_round_progressive_multiseed(
                    run_dir=run_dir,
                    round_name='p1_protocol_decide_round',
                    ab_name=f'{run_name}_p1_protocol_decide',
                    base_cfg=base_cfg,
                    grouped=grouped,
                    eval_splits=eval_splits,
                    candidates=fidelity.build_p1_protocol_decide_candidates(
                        protocols,
                        calibration,
                        search_space=search_space,
                    ),
                    seed=seed + 505,
                    seed_offsets=fidelity.P1_PROTOCOL_DECIDE_SEED_OFFSETS,
                    step_scale=fidelity.P1_PROTOCOL_DECIDE_STEP_SCALE,
                    probe_selector_name='protocol_all_three_top4',
                    probe_selector=lambda ranking, candidates: fidelity.select_p1_protocol_decide_probe_candidates(
                        ranking,
                        candidates,
                        keep=fidelity.P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
                    ),
                    group_key='protocol_arm',
                    probe_signature_data={
                        'keep_per_protocol': fidelity.P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
                        'eligible_aux_family': 'all_three',
                    },
                    ranking_mode=fidelity.P1_RANKING_MODE,
                    eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
                    ambiguity_mode=search_space['protocol_decide_progressive_ambiguity_mode'],
                    gap_threshold=search_space['protocol_decide_progressive_gap_threshold'],
                    noise_margin_mult=search_space['protocol_decide_progressive_noise_margin_mult'],
                )
                p1_state['protocol_decide_round'] = protocol_decide_round
                fidelity.update_results_doc(run_dir, state)
            if not protocol_compare:
                protocol_compare = build_protocol_compare(protocol_decide_round['ranking'])
            p1_state['protocol_compare'] = {
                'round_name': 'p1_protocol_compare',
                'ranking': protocol_compare,
            }
            if not selected_protocol_arm:
                selected_protocol_arm = str(
                    protocol_compare[0].get('candidate_meta', {}).get('protocol_arm', protocol_compare[0]['arm_name'])
                )
            p1_state['selected_protocol_arm'] = selected_protocol_arm
            state['final_conclusion']['p1_protocol_winner'] = selected_protocol_arm
            fidelity.atomic_write_json(state_path, state)
            fidelity.update_results_doc(run_dir, state)
            if not continue_to_winner_refine:
                if not has_completed_p1_results(state):
                    state['status'] = 'stopped_after_p1_protocol_decide'
            else:
                winner_refine_round = (
                    p1_state.get('winner_refine_round')
                    if isinstance(p1_state.get('winner_refine_round'), dict)
                    else None
                )
                if not isinstance(winner_refine_round, dict):
                    state['status'] = 'running_p1_winner_refine'
                    fidelity.atomic_write_json(state_path, state)
                    fidelity.update_results_doc(run_dir, state)

                    winner_center_selection = fidelity.winner_refine_center_selection_from_search_space(
                        search_space,
                        selected_protocol_arm,
                    )
                    winner_centers = select_protocol_centers(
                        protocol_decide_round['ranking'],
                        protocol_arm=selected_protocol_arm,
                        keep=winner_center_selection['keep'],
                        explicit_arm_names=winner_center_selection['explicit_arm_names'],
                    )
                    p1_state['winner_refine_centers'] = [candidate.arm_name for candidate in winner_centers]
                    winner_refine_round = fidelity.execute_round_multiseed(
                        run_dir=run_dir,
                        round_name='p1_winner_refine_round',
                        ab_name=f'{run_name}_p1_winner_refine',
                        base_cfg=base_cfg,
                        grouped=grouped,
                        eval_splits=eval_splits,
                        candidates=fidelity.build_p1_winner_refine_candidates(
                            protocols,
                            calibration,
                            winner_centers,
                            search_space=search_space,
                        ),
                        seed=seed + 606,
                        seed_offsets=fidelity.P1_WINNER_REFINE_SEED_OFFSETS,
                        step_scale=fidelity.P1_WINNER_REFINE_STEP_SCALE,
                        ranking_mode=fidelity.P1_RANKING_MODE,
                        eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
                    )
                    p1_state['winner_refine_round'] = winner_refine_round
                    fidelity.update_results_doc(run_dir, state)
                winner_refine_valid = [
                    entry for entry in winner_refine_round['ranking'] if entry.get('valid')
                ]
                if not winner_refine_valid:
                    raise RuntimeError('p1_winner_refine_round produced no valid candidates')
                refine_winner = fidelity.candidate_from_entry(winner_refine_valid[0])
                p1_state['winner_refine_front_runner'] = refine_winner.arm_name
                p1_state['final_compare'] = {
                    'round_name': 'p1_final_compare',
                    'ranking': list(winner_refine_round['ranking']),
                }
                p1_state['winner'] = refine_winner.arm_name
                p1_state['winner_source'] = 'winner_refine_mainline'
                p1_state['ablation_policy'] = fidelity.P1_ABLATION_POLICY
                state['final_conclusion']['p1_refine_front_runner'] = refine_winner.arm_name
                state['final_conclusion']['p1_winner'] = refine_winner.arm_name
                state['final_conclusion']['p1_winner_source'] = 'winner_refine_mainline'
                state['final_conclusion']['p1_ablation_policy'] = fidelity.P1_ABLATION_POLICY
                fidelity.atomic_write_json(state_path, state)
                fidelity.update_results_doc(run_dir, state)
                if not continue_to_ablation:
                    if not has_completed_p1_results(state):
                        state['status'] = 'stopped_after_p1_winner_refine'
                else:
                    ablation_round = (
                        p1_state.get('ablation_round') if isinstance(p1_state.get('ablation_round'), dict) else None
                    )
                    final_compare_payload = (
                        p1_state.get('final_compare') if isinstance(p1_state.get('final_compare'), dict) else {}
                    )
                    final_ranked = (
                        list(final_compare_payload.get('ranking', []))
                        if isinstance(final_compare_payload.get('ranking'), list)
                        else []
                    )
                    if not isinstance(ablation_round, dict):
                        state['status'] = 'running_p1_ablation'
                        fidelity.atomic_write_json(state_path, state)
                        fidelity.update_results_doc(run_dir, state)

                        ablation_round = fidelity.execute_round_multiseed(
                            run_dir=run_dir,
                            round_name='p1_ablation_round',
                            ab_name=f'{run_name}_p1_ablation',
                            base_cfg=base_cfg,
                            grouped=grouped,
                            eval_splits=eval_splits,
                            candidates=fidelity.build_p1_ablation_candidates(
                                protocols,
                                calibration,
                                refine_winner,
                                search_space=search_space,
                            ),
                            seed=seed + 707,
                            seed_offsets=fidelity.P1_ABLATION_SEED_OFFSETS,
                            step_scale=fidelity.P1_ABLATION_STEP_SCALE,
                            ranking_mode=fidelity.P1_RANKING_MODE,
                            eligibility_group_key=fidelity.P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
                        )
                        p1_state['ablation_round'] = ablation_round
                        final_ranked = list(ablation_round['ranking'])
                    if not final_ranked:
                        final_ranked = list(ablation_round['ranking'])
                    p1_state['final_compare'] = {
                        'round_name': 'p1_final_compare',
                        'ranking': final_ranked,
                    }
                    final_valid = [entry for entry in final_ranked if entry.get('valid')]
                    if not final_valid:
                        raise RuntimeError('p1_ablation_round produced no valid candidates')
                    final_winner = fidelity.candidate_from_entry(final_valid[0])
                    p1_state['winner'] = final_winner.arm_name
                    p1_state['winner_source'] = 'ablation_backlog'
                    p1_state['ablation_policy'] = fidelity.P1_ABLATION_POLICY
                    state['final_conclusion']['p1_winner'] = final_winner.arm_name
                    state['final_conclusion']['p1_winner_source'] = 'ablation_backlog'
                    state['final_conclusion']['p1_ablation_policy'] = fidelity.P1_ABLATION_POLICY
                    state['status'] = 'completed'
    except Exception as exc:
        if state is None:
            raise
        state['status'] = 'failed'
        state['fatal_error'] = str(exc)
        state['fatal_traceback'] = traceback.format_exc()
        fidelity.atomic_write_json(state_path, state)
        fidelity.update_results_doc(run_dir, state)
        print(json.dumps(fidelity.normalize_payload(state), ensure_ascii=False, indent=2))
        return state
    finally:
        fidelity.release_run_lock(lock_path)

    fidelity.atomic_write_json(state_path, state)
    fidelity.update_results_doc(run_dir, state)
    print(json.dumps(fidelity.normalize_payload(state), ensure_ascii=False, indent=2))
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=DEFAULT_RUN_NAME)
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='For new runs defaults to the frozen P1 seed; continue mode inherits the existing run seed when omitted.',
    )
    parser.add_argument(
        '--protocol-arm',
        action='append',
        dest='protocol_arms',
        choices=sorted(PROTOCOL_ARM_MAP),
        help='Repeat to override the default frozen top3 protocol seeds.',
    )
    parser.add_argument(
        '--calibration-protocol-arm',
        action='append',
        dest='calibration_protocol_arms',
        choices=sorted(PROTOCOL_ARM_MAP),
        help='Repeat to override the default representative protocol arms used by p1_calibration.',
    )
    parser.add_argument(
        '--calibration-mode',
        choices=fidelity.P1_CALIBRATION_MODE_CHOICES,
        default=None,
        help=(
            'Default is combo_only: inherit the frozen 2026-03-25 post-shape '
            'single-head calibration numbers and rerun only pairwise/triple '
            'combo probes.'
        ),
    )
    parser.add_argument('--stop-after-calibration', action='store_true')
    parser.add_argument(
        '--continue-to-winner-refine',
        action='store_true',
        help='After protocol_decide, continue into winner-only all-three local refine.',
    )
    parser.add_argument(
        '--continue-to-ablation',
        action='store_true',
        help='After winner_refine, continue into manual backlog leave-one-head ablation compare.',
    )
    parser.add_argument('--list-default-protocols', action='store_true')
    args = parser.parse_args()

    if args.list_default_protocols:
        print(
            json.dumps(
                {
                    'default_run_name': DEFAULT_RUN_NAME,
                    'default_seed': DEFAULT_P1_SEED,
                    'frozen_top3': list(FROZEN_TOP3),
                    'default_calibration_protocol_arms': list(fidelity.P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS),
                    'default_calibration_mode': fidelity.P1_CALIBRATION_DEFAULT_MODE,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    continue_to_winner_refine = args.continue_to_winner_refine or args.continue_to_ablation
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    run_p1_only(
        run_dir=run_dir,
        seed=args.seed,
        protocol_arms=args.protocol_arms,
        calibration_protocol_arms=args.calibration_protocol_arms,
        calibration_mode=args.calibration_mode,
        stop_after_calibration=args.stop_after_calibration,
        continue_to_winner_refine=continue_to_winner_refine,
        continue_to_ablation=args.continue_to_ablation,
    )


if __name__ == '__main__':
    main()

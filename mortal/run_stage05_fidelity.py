from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import json
import math
import os
import statistics
import shutil
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import run_stage05_ab as ab
import run_stage05_formal as formal
import stage05_current_defaults as stage05_defaults
from dataloader import SupervisedFileDatasetsIter
from stage05_selection import (
    ACTION_SCORE_WEIGHTS,
    LOSS_EPSILON,
    SCENARIO_SCORE_VERSION,
    SCENARIO_SCORE_VERSION_FIELD,
    SELECTION_SCENARIO_FACTOR,
    action_quality_score,
    refresh_scenario_quality_score,
    refresh_selection_quality_score,
    selection_tiebreak_key,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DOC_PATH = REPO_ROOT / 'docs' / 'status' / 'stage05-fidelity-results.md'
FIDELITY_ROOT = REPO_ROOT / 'logs' / 'stage05_fidelity'
FIDELITY_ROOT.mkdir(parents=True, exist_ok=True)

# Frozen by the 2026-03-25 A2y internal-shape micro AB.
# The freeze was validated under the historical P1 solo winner metric and is
# now reused as the default internal shape in the redesigned mainline.
RANK_TEMPLATE = {
    'base_weight': 0.03,
    'south_factor': 1.59,
    'all_last_factor': 1.617,
    'gap_focus_points': 4000.0,
    'gap_close_bonus': 0.0,
    'max_weight': 0.10,
}

P1_EFFECTIVE_BUDGET_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
# Current solo probes show that rank/opp inflate diagnostic full_recent_loss
# more sharply than danger at the old smallest budgets. The selector itself now
# uses policy_quality, but the next round still narrows rank/opp search bands
# and keeps a slightly wider one for danger.
P1_SOLO_RANK_BUDGET_RATIOS = [0.03, 0.06, 0.10, 0.15]
P1_SOLO_OPP_BUDGET_RATIOS = [0.03, 0.06, 0.10, 0.15]
P1_SOLO_DANGER_BUDGET_RATIOS = [0.05, 0.10, 0.20, 0.30]
P1_DEFAULT_OPP_WEIGHT_PER_BUDGET = 0.01
P1_DEFAULT_DANGER_WEIGHT_PER_BUDGET = 0.01
P1_CALIBRATION_PROBE_WEIGHT = 0.06
P1_GRAD_CALIBRATION_MAX_BATCHES = 8
P1_CALIBRATION_MAPPING_MODE = 'hybrid_loss_grad_geomean'
P1_CALIBRATION_SCHEME = 'single_head_mapping_plus_pairwise_triple_combo'
P1_CALIBRATION_MODE_COMBO_ONLY = 'combo_only'
P1_CALIBRATION_MODE_FULL = 'full'
P1_CALIBRATION_MODE_CHOICES = (
    P1_CALIBRATION_MODE_COMBO_ONLY,
    P1_CALIBRATION_MODE_FULL,
)
P1_CALIBRATION_DEFAULT_MODE = stage05_defaults.CURRENT_P1_CALIBRATION_MODE
P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS = stage05_defaults.CURRENT_P1_CALIBRATION_PROTOCOL_ARMS
P1_SINGLE_HEAD_CALIBRATION_BASELINE = dict(
    stage05_defaults.CURRENT_P1_SINGLE_HEAD_CALIBRATION_BASELINE
)
P1_SINGLE_HEAD_CALIBRATION_SOURCE = stage05_defaults.CURRENT_P1_SINGLE_HEAD_CALIBRATION_SOURCE
# combo_only inherits the frozen single-head numbers above and only reruns
# pairwise/triple probes to refresh combo factors. It is intentionally not a
# full single-head recalibration inside the current run.
P1_CALIBRATION_STEP_SCALE = 0.5
P1_CALIBRATION_SEED_OFFSETS = [0]
P1_SOLO_ROUND_SEED_OFFSETS = [0, 1009]
P1_PAIRWISE_ROUND_SEED_OFFSETS = [0, 1009]
P1_JOINT_REFINE_ROUND_SEED_OFFSETS = [0, 1009, 2027]
P1_PROTOCOL_DECIDE_STEP_SCALE = 1.0
P1_PROTOCOL_DECIDE_SEED_OFFSETS = [0, 1009]
P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL = 4
P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS = [0.09, 0.12]
# protocol_decide only needs enough spread to pick the winning protocol.
# Keep the grid centered near the solo sweet spot, but leave the old upper edge
# for winner_refine where we can search more locally around a single protocol.
P1_PROTOCOL_DECIDE_MIXES = [
    ('anchor', 0.43, 0.21, 0.36),
    ('rank_lean', 0.53, 0.16, 0.31),
    ('opp_lean', 0.38, 0.31, 0.31),
    ('danger_lean', 0.38, 0.16, 0.46),
]
P1_WINNER_REFINE_STEP_SCALE = 1.5
P1_WINNER_REFINE_SEED_OFFSETS = [0, 1009]
P1_WINNER_REFINE_CENTERS = 2
P1_WINNER_REFINE_TOTAL_SCALE_FACTORS = [0.85, 1.0, 1.15]
P1_WINNER_REFINE_TRANSFER_DELTA = 0.01
P1_ABLATION_STEP_SCALE = 1.5
P1_ABLATION_SEED_OFFSETS = [0, 1009]
P1_SOLO_PROBE_KEEP_PER_FAMILY = 2
P1_PAIRWISE_PROBE_KEEP_PER_PROTOCOL = 4
P1_PROGRESSIVE_NOISE_MARGIN_MULT = 2.0
P1_PAIRWISE_KEEP_PER_PROTOCOL = 4
P1_JOINT_REFINE_CENTERS_PER_PROTOCOL = 2
P1_JOINT_REFINE_BUDGET_DELTA = 0.25
P1_POLICY_LOSS_EPSILON = LOSS_EPSILON
# Stat-backed by 2026-03-26 multiseed selector audit rather than tied to policy epsilon.
P1_OLD_REGRESSION_POLICY_EPSILON = 0.0035
P1_RANKING_MODE = 'policy_quality'
P0_ROUND1_MIN_KEEP = 8
P1_PROTOCOL_ELIGIBILITY_GROUP_KEY = 'protocol_arm'
P1_BUDGET_RATIO_MIN = 0.00
P1_BUDGET_RATIO_MAX = 2.00
P1_AUX_WEIGHT_MIN = 0.00
P1_AUX_WEIGHT_MAX = 0.18
P1_COMBO_FACTOR_MIN = 0.75
P1_COMBO_FACTOR_MAX = 1.25
ARM_CACHE_SCHEMA_VERSION = 2
ROUND_CACHE_SCHEMA_VERSION = 3
P2_MAX_FINALISTS = 4

SELECTOR_PROFILES = {
    'S0_default': dict(ACTION_SCORE_WEIGHTS),
    'S1_riichi_heavy': {
        'discard_nll': 0.40,
        'riichi_decision_balanced_bce': 0.24,
        'agari_decision_balanced_bce': 0.18,
        'chi_decision_balanced_bce': 0.04,
        'chi_exact_nll': 0.02,
        'pon_decision_balanced_bce': 0.07,
        'kan_decision_balanced_bce': 0.05,
    },
    'S2_call_heavy': {
        'discard_nll': 0.40,
        'riichi_decision_balanced_bce': 0.18,
        'agari_decision_balanced_bce': 0.16,
        'chi_decision_balanced_bce': 0.06,
        'chi_exact_nll': 0.04,
        'pon_decision_balanced_bce': 0.10,
        'kan_decision_balanced_bce': 0.06,
    },
    'S3_discard_heavy': {
        'discard_nll': 0.52,
        'riichi_decision_balanced_bce': 0.16,
        'agari_decision_balanced_bce': 0.16,
        'chi_decision_balanced_bce': 0.04,
        'chi_exact_nll': 0.02,
        'pon_decision_balanced_bce': 0.06,
        'kan_decision_balanced_bce': 0.04,
    },
}

P1_SELECTION_TIEBREAK_FIELDS = [
    'selection_quality_score',
    '-recent_policy_loss',
    '-old_regression_policy_loss',
]

P1_CALIBRATION_ROLE_ALIASES = {
    'rank_only': 'rank_only',
    'opp_probe': 'rank_opp_probe',
    'danger_probe': 'rank_danger_probe',
    'both_probe': 'triple_probe',
    'rank_opp_probe': 'rank_opp_probe',
    'rank_danger_probe': 'rank_danger_probe',
    'opp_danger_probe': 'opp_danger_probe',
    'triple_probe': 'triple_probe',
}


def p1_calibration_mode_note(calibration_mode: str) -> str:
    if calibration_mode == P1_CALIBRATION_MODE_COMBO_ONLY:
        return (
            'combo_only inherits the frozen 2026-03-25 post-shape single-head '
            'calibration baseline (rank/opp/danger mapping and derived '
            'weight-per-budget values) and reruns only pairwise/triple probes '
            'to refresh combo factors; it does not recompute pure opponent-only '
            'or danger-only single-head probes in the current run'
        )
    if calibration_mode == P1_CALIBRATION_MODE_FULL:
        return (
            'full does not inherit the frozen combo_only baseline; it '
            're-estimates calibration quantities from probes generated inside '
            'the current run'
        )
    return f'unknown calibration mode: {calibration_mode}'


def p1_selection_policy_metadata() -> dict[str, Any]:
    return {
        'canonical_selector': P1_RANKING_MODE,
        'comparison_metric': 'recent_policy_loss',
        'comparison_alias': 'comparison_recent_loss',
        'diagnostic_metric': 'full_recent_loss',
        'eligibility_group_key': P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        'policy_loss_epsilon': P1_POLICY_LOSS_EPSILON,
        'old_regression_policy_loss_epsilon': P1_OLD_REGRESSION_POLICY_EPSILON,
        'selection_scenario_factor': SELECTION_SCENARIO_FACTOR,
        'tiebreak_order': list(P1_SELECTION_TIEBREAK_FIELDS),
        'ce_only_guardrail': (
            'historical family-survivor guardrail: family survivor must not lose clearly to ce_only under the same policy_quality gate'
        ),
        'applies_to': [
            'A2y internal-shape micro AB',
            'p1_protocol_decide_round',
            'p1_winner_refine_round',
            'p1_ablation_round',
        ],
        'calibration_note': (
            'p1_calibration is a mapping step for single-head units plus pairwise/triple combo factors; it must not be used to declare family winners'
        ),
        'ce_only_anchor_note': (
            'ce_only is kept as a diagnostic anchor; mainline P1 no longer uses family survivors as the protocol gate'
        ),
    }


class FILETIME(ctypes.Structure):
    _fields_ = [
        ('dwLowDateTime', ctypes.c_ulong),
        ('dwHighDateTime', ctypes.c_ulong),
    ]


@dataclass(frozen=True)
class CandidateSpec:
    arm_name: str
    scheduler_profile: str
    curriculum_profile: str
    weight_profile: str
    window_profile: str
    cfg_overrides: dict[str, Any]
    meta: dict[str, Any]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(text, encoding='utf-8', newline='\n')
    for attempt in range(1, 11):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in {5, 32} or attempt >= 10:
                raise
            time.sleep(min(1.0, 0.1 * attempt))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(normalize_payload(payload), ensure_ascii=False, indent=2))


def remove_tree_with_retries(path: Path) -> None:
    if not path.exists():
        return
    for attempt in range(1, 16):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in {5, 32} or attempt >= 15:
                raise
            time.sleep(min(1.5, 0.1 * attempt))


def apply_round_signature_ranking_fields(
    payload: dict[str, Any],
    *,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> dict[str, Any]:
    if ranking_mode != 'full_recent':
        payload['ranking_mode'] = ranking_mode
    if eligibility_group_key is not None:
        payload['eligibility_group_key'] = eligibility_group_key
    return payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == 'nt':
        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information | synchronize,
            False,
            pid,
        )
        if not handle:
            error_code = ctypes.get_last_error()
            return error_code == 5
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def normalize_payload(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_payload(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def stable_payload_digest(value: Any) -> str:
    normalized = normalize_payload(value)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def ts_now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def dedupe_string_items(items: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def parse_ts_to_unix_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def process_start_unix_ms(pid: int) -> int | None:
    if pid <= 0:
        return None
    if os.name != 'nt':
        return None
    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        creation_time = FILETIME()
        exit_time = FILETIME()
        kernel_time = FILETIME()
        user_time = FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation_time),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        raw = (creation_time.dwHighDateTime << 32) | creation_time.dwLowDateTime
        if raw <= 0:
            return None
        return raw // 10_000 - 11_644_473_600_000
    finally:
        kernel32.CloseHandle(handle)


def build_run_lock_payload(run_name: str) -> dict[str, Any]:
    return {
        'pid': os.getpid(),
        'run_name': run_name,
        'created_at': ts_now(),
        'process_start_unix_ms': process_start_unix_ms(os.getpid()),
    }


def lock_belongs_to_running_process(payload: dict[str, Any]) -> bool:
    pid = int(payload.get('pid', 0) or 0)
    if not process_is_alive(pid):
        return False
    live_start_ms = process_start_unix_ms(pid)
    if live_start_ms is None:
        return False
    expected_start_ms = payload.get('process_start_unix_ms')
    if isinstance(expected_start_ms, (int, float)) and int(expected_start_ms) > 0:
        return live_start_ms == int(expected_start_ms)
    created_at_ms = parse_ts_to_unix_ms(payload.get('created_at'))
    if created_at_ms is None:
        return False
    delta_ms = created_at_ms - live_start_ms
    return 0 <= delta_ms <= 60_000


def acquire_run_lock(run_dir: Path, run_name: str) -> Path:
    lock_path = run_dir / 'run.lock.json'
    payload = build_run_lock_payload(run_name)
    for _ in range(2):
        try:
            with lock_path.open('x', encoding='utf-8', newline='\n') as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write('\n')
            return lock_path
        except FileExistsError:
            try:
                existing = load_json(lock_path)
            except Exception:
                existing = {}
            existing_pid = int(existing.get('pid', 0) or 0)
            if lock_belongs_to_running_process(existing):
                raise RuntimeError(
                    f'run `{run_name}` is already active under pid={existing_pid}; '
                    f'use `scripts\\stop_stage05_fidelity.bat {run_name}` first'
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    raise RuntimeError(f'failed to acquire run lock for `{run_name}` at {lock_path}')


def release_run_lock(lock_path: Path | None) -> None:
    if lock_path is None or not lock_path.exists():
        return
    try:
        payload = load_json(lock_path)
    except Exception:
        payload = {}
    if int(payload.get('pid', -1) or -1) != os.getpid():
        return
    current_start_ms = process_start_unix_ms(os.getpid())
    expected_start_ms = payload.get('process_start_unix_ms')
    if isinstance(expected_start_ms, (int, float)) and current_start_ms is not None:
        if int(expected_start_ms) != current_start_ms:
            return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def reset_state_for_stop_flags(
    state: dict[str, Any],
    *,
    stop_after_p0: bool,
    stop_after_p1_calibration: bool,
    stop_after_p1_protocol_decide: bool,
    stop_after_p1_winner_refine: bool,
) -> None:
    if not (
        stop_after_p0
        or stop_after_p1_calibration
        or stop_after_p1_protocol_decide
        or stop_after_p1_winner_refine
    ):
        return
    for key in ('p0', 'p1', 'p2', 'formal', 'final_conclusion'):
        state.pop(key, None)


def build_p0_candidates(*, scheduler_profiles: set[str] | None = None) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for scheduler_prefix, scheduler_profile in ab.SCHEDULER_PREFIXES:
        if scheduler_profiles is not None and scheduler_profile not in scheduler_profiles:
            continue
        for curriculum_prefix, curriculum_profile in ab.CURRICULUM_PREFIXES:
            for weight_prefix, weight_profile in ab.WEIGHT_PREFIXES:
                for window_prefix, window_profile in ab.WINDOW_PREFIXES:
                    arm_name = (
                        f'{scheduler_prefix}_{curriculum_prefix}{weight_prefix}{window_prefix}_'
                        f'{scheduler_profile}_{curriculum_profile}_{weight_profile}_{window_profile}'
                    )
                    candidates.append(
                        CandidateSpec(
                            arm_name=arm_name,
                            scheduler_profile=scheduler_profile,
                            curriculum_profile=curriculum_profile,
                            weight_profile=weight_profile,
                            window_profile=window_profile,
                            cfg_overrides={},
                            meta={'stage': 'P0'},
                        )
                    )
    return candidates


def build_protocol_candidate(protocol_arm: str) -> CandidateSpec:
    for candidate in build_p0_candidates():
        if candidate.arm_name != protocol_arm:
            continue
        return CandidateSpec(
            arm_name=candidate.arm_name,
            scheduler_profile=candidate.scheduler_profile,
            curriculum_profile=candidate.curriculum_profile,
            weight_profile=candidate.weight_profile,
            window_profile=candidate.window_profile,
            cfg_overrides=dict(candidate.cfg_overrides),
            meta={
                **candidate.meta,
                'protocol_arm': protocol_arm,
            },
        )
    raise ValueError(f'unknown protocol arm: {protocol_arm}')


def build_p0_round0_survivors(
    ranking: list[dict[str, Any]],
    all_index: dict[str, CandidateSpec],
    *,
    candidate_subset: str,
) -> list[CandidateSpec]:
    if candidate_subset == 'cosine_only':
        ranked_entries = [
            entry
            for entry in ranking
            if entry['arm_name'] in all_index
        ]
        source_entries = [entry for entry in ranked_entries if entry.get('valid')]
        if not source_entries:
            source_entries = ranked_entries
        survivor_names = [
            entry['arm_name']
            for entry in source_entries
        ]
    else:
        survivor_names = [
            entry['arm_name']
            for entry in ranking
            if entry['arm_name'] in set(select_group_top_k(ranking, 'scheduler_profile', 6))
        ]
    return [all_index[name] for name in survivor_names]


def build_p0_round1_survivors(
    ranking: list[dict[str, Any]],
    *,
    loss_epsilon: float = LOSS_EPSILON,
    min_keep: int = P0_ROUND1_MIN_KEEP,
    max_keep: int | None = None,
) -> list[CandidateSpec]:
    valid_entries = [entry for entry in ranking if entry.get('valid')]
    ranked_entries = valid_entries if valid_entries else list(ranking)
    if not ranked_entries:
        return []
    best_loss = min(
        (
            float(entry.get('full_recent_loss', math.inf))
            for entry in ranked_entries
            if math.isfinite(float(entry.get('full_recent_loss', math.inf)))
        ),
        default=math.inf,
    )
    survivor_entries: list[dict[str, Any]] = []
    survivor_names: set[str] = set()

    def add_survivor(entry: dict[str, Any]) -> None:
        arm_name = str(entry.get('arm_name') or '')
        if not arm_name or arm_name in survivor_names:
            return
        survivor_entries.append(entry)
        survivor_names.add(arm_name)

    for idx, entry in enumerate(ranked_entries):
        loss = float(entry.get('full_recent_loss', math.inf))
        if idx < min_keep or (math.isfinite(loss) and loss <= best_loss + loss_epsilon):
            add_survivor(entry)
    min_survivors = min(2, len(ranking))
    target_survivors = min(len(ranking), max(min_keep, min_survivors))
    if len(survivor_entries) < target_survivors:
        for entry in ranking:
            add_survivor(entry)
            if len(survivor_entries) >= target_survivors:
                break
    if max_keep is not None and max_keep > 0:
        keep_cap = max_keep if len(ranking) < 2 else max(2, max_keep)
        survivor_entries = survivor_entries[:keep_cap]
    return [candidate_from_entry(entry) for entry in survivor_entries]


def skipped_round_payload(round_name: str, *, reason: str) -> dict[str, Any]:
    return {
        'round_name': round_name,
        'status': 'skipped',
        'reason': reason,
        'ranking': [],
    }


def select_p0_stage1_top4(round2_ranking: list[dict[str, Any]]) -> list[CandidateSpec]:
    valid_entries = [entry for entry in round2_ranking if entry.get('valid')]
    if not valid_entries:
        valid_entries = list(round2_ranking)
    return [candidate_from_entry(entry) for entry in valid_entries[:4]]


def full_recent_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get('last_full_recent_metrics') or {}


def old_regression_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get('last_old_regression_metrics') or {}


def as_float(metrics: dict[str, Any], key: str, default: float = float('-inf')) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def action_quality_with_weights(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    return action_quality_score(metrics, weights)


def selection_key_for_summary(summary: dict[str, Any], weights: dict[str, float] | None = None) -> tuple[float, ...]:
    metrics = full_recent_metrics(summary) or summary
    recent_loss = recent_ranking_loss_for_summary(summary, ranking_mode='full_recent')
    old_loss = old_regression_ranking_loss_for_summary(summary, ranking_mode='full_recent')
    return selection_tiebreak_key(
        metrics,
        recent_loss=recent_loss,
        old_regression_loss=math.inf if old_loss is None or not math.isfinite(old_loss) else old_loss,
        action_weights=ACTION_SCORE_WEIGHTS if weights is None else weights,
    )


def recent_policy_loss_for_summary(summary: dict[str, Any]) -> float:
    metrics = full_recent_metrics(summary) or summary
    return as_float(metrics, 'policy_loss', ab.full_recent_loss(summary))


def old_regression_policy_loss_for_summary(summary: dict[str, Any]) -> float | None:
    metrics = old_regression_metrics(summary)
    if not metrics:
        return None
    return as_float(metrics, 'policy_loss', None)


def recent_policy_loss_for_entry(entry: dict[str, Any]) -> float:
    metrics = entry.get('full_recent_metrics', {})
    return as_float(metrics, 'policy_loss', entry['full_recent_loss'])


def old_regression_policy_loss_for_entry(entry: dict[str, Any]) -> float | None:
    metrics = entry.get('old_regression_metrics', {})
    return as_float(metrics, 'policy_loss', None)


def recent_ranking_loss_for_summary(summary: dict[str, Any], *, ranking_mode: str) -> float:
    if ranking_mode == 'policy_quality':
        return recent_policy_loss_for_summary(summary)
    return ab.full_recent_loss(summary)


def old_regression_ranking_loss_for_summary(
    summary: dict[str, Any],
    *,
    ranking_mode: str,
) -> float | None:
    if ranking_mode == 'policy_quality':
        return old_regression_policy_loss_for_summary(summary)
    return as_float(old_regression_metrics(summary), 'loss', None)


def recent_ranking_loss_for_entry(entry: dict[str, Any], *, ranking_mode: str) -> float:
    if ranking_mode == 'policy_quality':
        return recent_policy_loss_for_entry(entry)
    return entry['full_recent_loss']


def old_regression_ranking_loss_for_entry(
    entry: dict[str, Any],
    *,
    ranking_mode: str,
) -> float | None:
    if ranking_mode == 'policy_quality':
        return old_regression_policy_loss_for_entry(entry)
    return as_float(entry.get('old_regression_metrics', {}), 'loss', None)


def ranking_loss_baselines(
    loss_pairs: list[tuple[float, float | None]],
    *,
    ranking_mode: str,
) -> tuple[float, float | None]:
    finite_recent_losses = [recent_loss for recent_loss, _ in loss_pairs if math.isfinite(recent_loss)]
    best_recent = min(finite_recent_losses) if finite_recent_losses else math.inf
    finite_old_losses = [
        old_loss
        for recent_loss, old_loss in loss_pairs
        if old_loss is not None
        and math.isfinite(old_loss)
        and (
            ranking_mode != 'policy_quality'
            or (
                math.isfinite(recent_loss)
                and recent_loss <= best_recent + P1_POLICY_LOSS_EPSILON
            )
        )
    ]
    best_old = min(finite_old_losses) if finite_old_losses else None
    return best_recent, best_old


def checkpoint_sort_key(
    summary: dict[str, Any],
    weights: dict[str, float] | None = None,
    *,
    eligible: bool,
    ranking_mode: str = 'full_recent',
) -> tuple[float, ...]:
    metrics = full_recent_metrics(summary) or summary
    recent_loss = recent_ranking_loss_for_summary(summary, ranking_mode=ranking_mode)
    old_loss = old_regression_ranking_loss_for_summary(summary, ranking_mode=ranking_mode)
    return (
        1 if eligible else 0,
        *selection_tiebreak_key(
            metrics,
            recent_loss=recent_loss,
            old_regression_loss=math.inf if old_loss is None or not math.isfinite(old_loss) else old_loss,
            action_weights=ACTION_SCORE_WEIGHTS if weights is None else weights,
        ),
        -recent_loss if math.isfinite(recent_loss) else float('-inf'),
    )


def choose_checkpoint_summary(
    summaries: dict[str, dict[str, Any]],
    *,
    ranking_mode: str = 'full_recent',
    weights: dict[str, float] | None = None,
) -> tuple[str, dict[str, Any], list[str]]:
    available = {
        checkpoint_type: summary
        for checkpoint_type, summary in summaries.items()
        if isinstance(summary, dict)
    }
    if not available:
        raise RuntimeError('no checkpoint summaries available')
    if ranking_mode == 'full_recent' and 'best_loss' in available:
        return 'best_loss', available['best_loss'], ['best_loss']

    losses_by_type = {
        checkpoint_type: (
            recent_ranking_loss_for_summary(summary, ranking_mode=ranking_mode),
            old_regression_ranking_loss_for_summary(summary, ranking_mode=ranking_mode),
        )
        for checkpoint_type, summary in available.items()
    }
    best_recent, best_old = ranking_loss_baselines(
        list(losses_by_type.values()),
        ranking_mode=ranking_mode,
    )
    epsilon = P1_POLICY_LOSS_EPSILON if ranking_mode == 'policy_quality' else LOSS_EPSILON

    eligible_types: list[str] = []
    for checkpoint_type, summary in available.items():
        recent_loss, old_loss = losses_by_type[checkpoint_type]
        eligible = math.isfinite(recent_loss) and recent_loss <= best_recent + epsilon
        if eligible and ranking_mode == 'policy_quality' and best_old is not None:
            eligible = (
                old_loss is not None
                and math.isfinite(old_loss)
                and old_loss <= best_old + P1_OLD_REGRESSION_POLICY_EPSILON
            )
        if eligible:
            eligible_types.append(checkpoint_type)

    winner_type = max(
        available.items(),
        key=lambda item: checkpoint_sort_key(
            item[1],
            weights,
            eligible=item[0] in eligible_types,
            ranking_mode=ranking_mode,
        ),
    )[0]
    return winner_type, available[winner_type], sorted(eligible_types)


def summarize_entry(
    name: str,
    candidate: CandidateSpec,
    payload: dict[str, Any],
    *,
    ranking_mode: str = 'full_recent',
) -> dict[str, Any]:
    ok = bool(payload.get('ok'))
    if ok:
        checkpoint_type, summary, eligible_checkpoint_types = choose_checkpoint_summary(
            payload['run']['final'],
            ranking_mode=ranking_mode,
        )
        loss = ab.full_recent_loss(summary)
        metrics = dict(full_recent_metrics(summary))
        old_metrics = old_regression_metrics(summary)
        valid = math.isfinite(loss)
        action_score = action_quality_with_weights(metrics, ACTION_SCORE_WEIGHTS)
        scenario_score = refresh_scenario_quality_score(metrics)
        selection_score = refresh_selection_quality_score(metrics)
        rank_acc = as_float(metrics, 'rank_acc', -1.0)
        ckpt_path = summary.get('path')
        recent_policy_loss = recent_policy_loss_for_summary(summary)
        old_regression_policy_loss = old_regression_policy_loss_for_summary(summary)
    else:
        loss = math.inf
        metrics = {}
        old_metrics = {}
        valid = False
        action_score = float('-inf')
        scenario_score = float('-inf')
        selection_score = float('-inf')
        rank_acc = -1.0
        ckpt_path = None
        recent_policy_loss = math.inf
        old_regression_policy_loss = None
        checkpoint_type = None
        eligible_checkpoint_types = []
    return {
        'arm_name': name,
        'candidate_meta': candidate.meta,
        'scheduler_profile': candidate.scheduler_profile,
        'curriculum_profile': candidate.curriculum_profile,
        'weight_profile': candidate.weight_profile,
        'window_profile': candidate.window_profile,
        'cfg_overrides': candidate.cfg_overrides,
        'ok': ok,
        'valid': valid,
        'full_recent_loss': loss,
        'recent_policy_loss': recent_policy_loss,
        'full_recent_metrics': metrics,
        'action_quality_score': action_score,
        'scenario_quality_score': scenario_score,
        'scenario_quality_score_version': metrics.get(SCENARIO_SCORE_VERSION_FIELD),
        'selection_quality_score': selection_score,
        'rank_acc': rank_acc,
        'discard_top1_acc': as_float(metrics, 'discard_top1_acc', -1.0),
        'riichi_bal_acc': as_float(metrics, 'riichi_decision_balanced_acc', -1.0),
        'agari_bal_acc': as_float(metrics, 'agari_decision_balanced_acc', -1.0),
        'chi_bal_acc': as_float(metrics, 'chi_decision_balanced_acc', -1.0),
        'pon_bal_acc': as_float(metrics, 'pon_decision_balanced_acc', -1.0),
        'kan_bal_acc': as_float(metrics, 'kan_decision_balanced_acc', -1.0),
        'old_regression_policy_loss': old_regression_policy_loss,
        'old_regression_loss': as_float(old_metrics, 'loss', math.inf),
        'old_regression_metrics': old_metrics,
        'checkpoint_type': checkpoint_type,
        'eligible_checkpoint_types': eligible_checkpoint_types,
        'checkpoint_path': ckpt_path,
        'cache_path': payload.get('cache_path'),
        'error': payload.get('error'),
    }


def aggregate_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    aggregated: dict[str, float] = {}
    for key in keys:
        values: list[float] = []
        for metrics in metric_dicts:
            value = metrics.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        if values:
            aggregated[key] = statistics.median(values)
    return aggregated


def choose_representative_seed(
    seed_entries: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    target_loss: float,
    ranking_mode: str = 'full_recent',
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    valid_seed_entries = [item for item in seed_entries if item[2]['valid']]
    pool = valid_seed_entries or seed_entries
    return min(
        pool,
        key=lambda item: (
            abs(recent_ranking_loss_for_entry(item[2], ranking_mode=ranking_mode) - target_loss),
            -item[2]['action_quality_score'],
            -item[2]['scenario_quality_score'],
            item[0],
        ),
    )


def summarize_multiseed_entry(
    name: str,
    candidate: CandidateSpec,
    seed_payloads: dict[str, dict[str, Any]],
    *,
    ranking_mode: str = 'full_recent',
) -> dict[str, Any]:
    seed_entries = [
        (
            seed_label,
            payload,
            summarize_entry(
                name,
                candidate,
                payload,
                ranking_mode=ranking_mode,
            ),
        )
        for seed_label, payload in sorted(seed_payloads.items())
    ]
    seed_count = len(seed_entries)
    valid_seed_entries = [item for item in seed_entries if item[2]['valid']]
    valid_seed_count = len(valid_seed_entries)
    multiseed_valid = valid_seed_count == max(1, seed_count)

    if valid_seed_entries:
        median_loss = statistics.median(item[2]['full_recent_loss'] for item in valid_seed_entries)
        median_recent_ranking_loss = statistics.median(
            recent_ranking_loss_for_entry(item[2], ranking_mode=ranking_mode)
            for item in valid_seed_entries
        )
        aggregated_metrics = aggregate_metric_dicts([item[2]['full_recent_metrics'] for item in valid_seed_entries])
        aggregated_old_metrics = aggregate_metric_dicts([item[2]['old_regression_metrics'] for item in valid_seed_entries])
        representative_seed, representative_payload, representative_summary = choose_representative_seed(
            valid_seed_entries,
            target_loss=median_recent_ranking_loss,
            ranking_mode=ranking_mode,
        )
        action_score = action_quality_with_weights(aggregated_metrics, ACTION_SCORE_WEIGHTS)
        scenario_score = refresh_scenario_quality_score(aggregated_metrics)
        selection_score = refresh_selection_quality_score(aggregated_metrics)
        rank_acc = as_float(aggregated_metrics, 'rank_acc', -1.0)
        ckpt_path = representative_summary.get('checkpoint_path')
        recent_policy_loss = as_float(aggregated_metrics, 'policy_loss', median_loss)
        old_regression_policy_loss = as_float(aggregated_old_metrics, 'policy_loss', None)
        checkpoint_type = representative_summary.get('checkpoint_type')
        eligible_checkpoint_types = representative_summary.get('eligible_checkpoint_types', [])
    else:
        median_loss = math.inf
        aggregated_metrics = {}
        aggregated_old_metrics = {}
        representative_seed, representative_payload, representative_summary = seed_entries[0]
        action_score = float('-inf')
        scenario_score = float('-inf')
        selection_score = float('-inf')
        rank_acc = -1.0
        ckpt_path = None
        recent_policy_loss = math.inf
        old_regression_policy_loss = None
        checkpoint_type = None
        eligible_checkpoint_types = []

    candidate_meta = {
        **candidate.meta,
        'seed_count': seed_count,
        'valid_seed_count': valid_seed_count,
        'representative_seed': representative_seed,
        'representative_result_path': representative_payload.get('cache_path'),
    }
    return {
        'arm_name': name,
        'candidate_meta': candidate_meta,
        'scheduler_profile': candidate.scheduler_profile,
        'curriculum_profile': candidate.curriculum_profile,
        'weight_profile': candidate.weight_profile,
        'window_profile': candidate.window_profile,
        'cfg_overrides': candidate.cfg_overrides,
        'ok': bool(valid_seed_entries),
        'valid': multiseed_valid,
        'full_recent_loss': median_loss,
        'recent_policy_loss': recent_policy_loss,
        'full_recent_metrics': aggregated_metrics,
        'action_quality_score': action_score,
        'scenario_quality_score': scenario_score,
        'scenario_quality_score_version': aggregated_metrics.get(SCENARIO_SCORE_VERSION_FIELD),
        'selection_quality_score': selection_score,
        'rank_acc': rank_acc,
        'discard_top1_acc': as_float(aggregated_metrics, 'discard_top1_acc', -1.0),
        'riichi_bal_acc': as_float(aggregated_metrics, 'riichi_decision_balanced_acc', -1.0),
        'agari_bal_acc': as_float(aggregated_metrics, 'agari_decision_balanced_acc', -1.0),
        'chi_bal_acc': as_float(aggregated_metrics, 'chi_decision_balanced_acc', -1.0),
        'pon_bal_acc': as_float(aggregated_metrics, 'pon_decision_balanced_acc', -1.0),
        'kan_bal_acc': as_float(aggregated_metrics, 'kan_decision_balanced_acc', -1.0),
        'old_regression_policy_loss': old_regression_policy_loss,
        'old_regression_loss': as_float(aggregated_old_metrics, 'loss', math.inf),
        'old_regression_metrics': aggregated_old_metrics,
        'checkpoint_type': checkpoint_type,
        'eligible_checkpoint_types': eligible_checkpoint_types,
        'checkpoint_path': ckpt_path,
        'cache_path': representative_payload.get('cache_path'),
        'error': None if valid_seed_entries else representative_payload.get('error'),
        'seed_count': seed_count,
        'valid_seed_count': valid_seed_count,
        'seed_summaries': [
            {
                'seed_label': seed_label,
                'valid': summary['valid'],
                'full_recent_loss': summary['full_recent_loss'],
                'recent_policy_loss': summary.get('recent_policy_loss'),
                'old_regression_policy_loss': summary.get('old_regression_policy_loss'),
                'checkpoint_type': summary.get('checkpoint_type'),
                'action_quality_score': summary['action_quality_score'],
                'scenario_quality_score': summary['scenario_quality_score'],
                'scenario_quality_score_version': summary.get('scenario_quality_score_version'),
                'cache_path': payload.get('cache_path'),
                'error': payload.get('error'),
            }
            for seed_label, payload, summary in seed_entries
        ],
    }


def entry_sort_key(
    entry: dict[str, Any],
    weights: dict[str, float] | None = None,
    *,
    eligible: bool,
    ranking_mode: str = 'full_recent',
) -> tuple[float, ...]:
    recent_loss = recent_ranking_loss_for_entry(entry, ranking_mode=ranking_mode)
    old_loss = old_regression_ranking_loss_for_entry(entry, ranking_mode=ranking_mode)
    return (
        1 if entry['valid'] else 0,
        1 if eligible else 0,
        *selection_tiebreak_key(
            entry.get('full_recent_metrics', {}),
            recent_loss=recent_loss,
            old_regression_loss=math.inf if old_loss is None or not math.isfinite(old_loss) else old_loss,
            action_weights=ACTION_SCORE_WEIGHTS if weights is None else weights,
        ),
        -recent_loss if math.isfinite(recent_loss) else float('-inf'),
    )


def rank_round_entries(
    entries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    *,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> list[dict[str, Any]]:
    def group_value(entry: dict[str, Any]) -> str:
        if eligibility_group_key is None:
            return ''
        return str(entry.get(eligibility_group_key, entry.get('candidate_meta', {}).get(eligibility_group_key, '')))

    entry_losses: list[tuple[dict[str, Any], str, float, float | None]] = []
    group_loss_pairs: dict[str, list[tuple[float, float | None]]] = {}
    for entry in entries:
        group = group_value(entry)
        recent_loss = recent_ranking_loss_for_entry(entry, ranking_mode=ranking_mode)
        old_loss = old_regression_ranking_loss_for_entry(entry, ranking_mode=ranking_mode)
        entry_losses.append((entry, group, recent_loss, old_loss))
        if entry['valid']:
            group_loss_pairs.setdefault(group, []).append((recent_loss, old_loss))

    group_baselines = {
        group: ranking_loss_baselines(loss_pairs, ranking_mode=ranking_mode)
        for group, loss_pairs in group_loss_pairs.items()
    }

    ranked = []
    for entry, group, recent_loss, old_loss in entry_losses:
        best_recent, best_old = group_baselines.get(group, (math.inf, None))
        epsilon = P1_POLICY_LOSS_EPSILON if ranking_mode == 'policy_quality' else LOSS_EPSILON
        eligible = entry['valid'] and math.isfinite(recent_loss) and recent_loss <= best_recent + epsilon
        if eligible and ranking_mode == 'policy_quality' and best_old is not None:
            eligible = (
                old_loss is not None
                and math.isfinite(old_loss)
                and old_loss <= best_old + P1_OLD_REGRESSION_POLICY_EPSILON
            )
        sort_key = entry_sort_key(
            entry,
            weights,
            eligible=eligible,
            ranking_mode=ranking_mode,
        )
        ranked.append(
            {
                **entry,
                'eligible': eligible,
                'sort_key': list(sort_key),
                'comparison_recent_loss': recent_loss,
                'comparison_old_regression_loss': old_loss,
                'eligibility_group': group if eligibility_group_key is not None else None,
                'eligibility_recent_loss_baseline': None if not math.isfinite(best_recent) else best_recent,
                'eligibility_old_regression_loss_baseline': best_old,
            }
        )
    ranked.sort(key=lambda item: tuple(item['sort_key']), reverse=True)
    for rank, entry in enumerate(ranked, start=1):
        entry['rank'] = rank
    return ranked


def best_family_entry(
    entries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    *,
    ranking_mode: str = 'full_recent',
) -> dict[str, Any] | None:
    valid_entries = [entry for entry in entries if entry.get('valid')]
    if not valid_entries:
        return None
    ranked = rank_round_entries(
        valid_entries,
        weights,
        ranking_mode=ranking_mode,
    )
    return ranked[0]


def select_group_top_k(
    ranked: list[dict[str, Any]],
    group_key: str,
    keep: int,
    *,
    valid_only: bool = False,
) -> list[str]:
    group_counts: dict[str, int] = {}
    winners: list[str] = []
    for entry in ranked:
        if valid_only and not entry.get('valid'):
            continue
        group = str(entry.get(group_key, entry['candidate_meta'].get(group_key, '')))
        group_counts.setdefault(group, 0)
        if group_counts[group] >= keep:
            continue
        winners.append(entry['arm_name'])
        group_counts[group] += 1
    return winners


def select_p1_finalists(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winner_names = set(select_group_top_k(ranked, 'protocol_arm', 1, valid_only=True))
    finalists = [entry for entry in ranked if entry['arm_name'] in winner_names]
    if finalists:
        return finalists
    raise RuntimeError('legacy p1_joint_refine_round produced no valid finalists after multiseed validation')


def is_p1_protocol_compare_meta(meta: dict[str, Any]) -> bool:
    return str(meta.get('aux_family', '')) == 'all_three'


def build_p1_protocol_compare(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible_ranked = [
        entry for entry in ranked if is_p1_protocol_compare_meta(entry.get('candidate_meta', {}))
    ]
    winner_names = set(select_group_top_k(eligible_ranked, 'protocol_arm', 1, valid_only=True))
    winners = [entry for entry in eligible_ranked if entry['arm_name'] in winner_names]
    if winners:
        return rank_round_entries(
            winners,
            ranking_mode=P1_RANKING_MODE,
        )
    raise RuntimeError('p1_protocol_decide_round produced no valid all_three protocol winners')


def is_p1_winner_refine_center_meta(meta: dict[str, Any]) -> bool:
    return str(meta.get('aux_family', '')) == 'all_three'


def select_p1_protocol_centers(
    ranked: list[dict[str, Any]],
    *,
    protocol_arm: str,
    keep: int,
) -> list[CandidateSpec]:
    centers: list[CandidateSpec] = []
    for entry in ranked:
        entry_protocol = str(entry.get('candidate_meta', {}).get('protocol_arm', ''))
        entry_meta = entry.get('candidate_meta', {})
        if (
            entry_protocol != protocol_arm
            or not entry.get('valid')
            or not is_p1_winner_refine_center_meta(entry_meta)
        ):
            continue
        centers.append(candidate_from_entry(entry))
        if len(centers) >= keep:
            break
    if centers:
        return centers
    raise RuntimeError(f'no valid all_three winner_refine centers for protocol {protocol_arm}')


def candidate_cache_payload(candidate: CandidateSpec, *, include_meta: bool) -> dict[str, Any]:
    payload = {
        'arm_name': candidate.arm_name,
        'scheduler_profile': candidate.scheduler_profile,
        'curriculum_profile': candidate.curriculum_profile,
        'weight_profile': candidate.weight_profile,
        'window_profile': candidate.window_profile,
        'cfg_overrides': candidate.cfg_overrides,
    }
    if include_meta:
        payload['meta'] = candidate.meta
    return payload


def arm_cache_signature(
    *,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidate: CandidateSpec,
    seed: int,
    step_scale: float,
    ab_name: str,
) -> str:
    return stable_payload_digest(
        {
            'schema_version': ARM_CACHE_SCHEMA_VERSION,
            'ab_name': ab_name,
            'base_cfg': base_cfg,
            'grouped': grouped,
            'eval_splits': eval_splits,
            'candidate': candidate_cache_payload(candidate, include_meta=False),
            'seed': seed,
            'step_scale': step_scale,
        }
    )


def round_cache_signature(
    *,
    round_name: str,
    ab_name: str,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidates: list[CandidateSpec],
    seed: int,
    step_scale: float,
    selector_weights: dict[str, float] | None,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> str:
    return stable_payload_digest(
        apply_round_signature_ranking_fields(
            {
                'schema_version': ROUND_CACHE_SCHEMA_VERSION,
                'scenario_score_version': SCENARIO_SCORE_VERSION,
                'round_name': round_name,
                'ab_name': ab_name,
                'base_cfg': base_cfg,
                'grouped': grouped,
                'eval_splits': eval_splits,
                'candidates': [candidate_cache_payload(candidate, include_meta=True) for candidate in candidates],
                'seed': seed,
                'step_scale': step_scale,
                'selector_weights': selector_weights,
            },
            ranking_mode=ranking_mode,
            eligibility_group_key=eligibility_group_key,
        )
    )


def legacy_round_payload_matches(
    payload: dict[str, Any],
    *,
    round_name: str,
    ab_name: str,
    expected_candidates: list[CandidateSpec],
    seed: int,
    step_scale: float,
    eval_splits: dict[str, list[str]],
) -> bool:
    if payload.get('round_name') != round_name:
        return False
    if payload.get('ab_name') != ab_name:
        return False
    try:
        payload_seed = int(payload.get('seed', -1) or -1)
    except (TypeError, ValueError):
        return False
    if payload_seed != seed:
        return False
    try:
        payload_step_scale = float(payload.get('step_scale'))
    except (TypeError, ValueError):
        return False
    if not math.isclose(payload_step_scale, step_scale, rel_tol=0.0, abs_tol=1e-9):
        return False
    expected_eval_split_counts = {
        key: len(files)
        for key, files in eval_splits.items()
    }
    if payload.get('eval_split_counts') != expected_eval_split_counts:
        return False
    ranking = payload.get('ranking') or []
    expected_names = {candidate.arm_name for candidate in expected_candidates}
    actual_names = {entry.get('arm_name') for entry in ranking}
    if None in actual_names:
        return False
    if len(ranking) != len(expected_candidates):
        return False
    if actual_names != expected_names:
        return False
    evaluated_arms = payload.get('evaluated_arms')
    if evaluated_arms is not None:
        try:
            if int(evaluated_arms) != len(expected_candidates):
                return False
        except (TypeError, ValueError):
            return False
    return True


def load_cached_round_if_valid(
    path: Path,
    expected_signature: str,
    *,
    legacy_matcher=None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if payload.get('round_signature') != expected_signature:
        if payload.get('schema_version') is None and legacy_matcher is not None:
            if legacy_matcher(payload):
                payload['legacy_cache_accepted'] = True
                return payload
        return None
    return payload


def revalidated_round_payload_matches(
    payload: dict[str, Any],
    *,
    source_dir_name: str,
    expected_candidates: list[CandidateSpec],
) -> bool:
    if payload.get('checkpoint_kind') != 'best_loss':
        return False
    if str(payload.get('phase_name', 'final')) != 'final':
        return False
    source_dir = payload.get('source_dir')
    if not source_dir or Path(str(source_dir)).name != source_dir_name:
        return False
    ranking = payload.get('ranking') or []
    expected_names = {candidate.arm_name for candidate in expected_candidates}
    actual_names = {entry.get('arm_name') for entry in ranking}
    if None in actual_names:
        return False
    if len(ranking) != len(expected_candidates):
        return False
    if actual_names != expected_names:
        return False
    evaluated_arms = payload.get('evaluated_arms')
    if evaluated_arms is not None:
        try:
            if int(evaluated_arms) != len(expected_candidates):
                return False
        except (TypeError, ValueError):
            return False
    return True


def adopt_revalidated_round_payload(
    payload: dict[str, Any],
    *,
    round_name: str,
    ab_name: str,
    expected_signature: str,
    seed: int,
    step_scale: float,
    eval_splits: dict[str, list[str]],
) -> dict[str, Any]:
    adopted = dict(payload)
    adopted['round_name'] = round_name
    adopted['ab_name'] = ab_name
    adopted['seed'] = seed
    adopted['step_scale'] = step_scale
    adopted['round_signature'] = expected_signature
    adopted['eval_split_counts'] = {
        key: len(files)
        for key, files in eval_splits.items()
    }
    adopted['best_loss'] = min(
        (
            entry['full_recent_loss']
            for entry in adopted.get('ranking', [])
            if entry.get('valid')
        ),
        default=None,
    )
    adopted['revalidated_cache_accepted'] = True
    return adopted


def load_revalidated_round_if_valid(
    *,
    run_dir: Path,
    round_name: str,
    ab_name: str,
    ab_root_name: str,
    expected_signature: str,
    expected_candidates: list[CandidateSpec],
    seed: int,
    step_scale: float,
    eval_splits: dict[str, list[str]],
) -> dict[str, Any] | None:
    revalidated_path = ab.AB_ROOT / ab_root_name / 'revalidated_best_loss_final_round.json'
    if not revalidated_path.exists():
        return None
    payload = load_json(revalidated_path)
    if not revalidated_round_payload_matches(
        payload,
        source_dir_name=ab_root_name,
        expected_candidates=expected_candidates,
    ):
        return None
    adopted = adopt_revalidated_round_payload(
        payload,
        round_name=round_name,
        ab_name=ab_name,
        expected_signature=expected_signature,
        seed=seed,
        step_scale=step_scale,
        eval_splits=eval_splits,
    )
    atomic_write_json(run_dir / f'{round_name}.json', adopted)
    return adopted


def load_authoritative_revalidated_p0_rounds(
    run_dir: Path,
    *,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    seed: int,
    all_candidates: list[CandidateSpec],
    all_index: dict[str, CandidateSpec],
    candidate_subset: str,
    round2_max_candidates: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    eval_file_count = ab.BASE_SCREENING['eval_files']

    def adopt_authoritative_round(
        *,
        round_name: str,
        stage_name: str,
        expected_signature: str,
        expected_candidates: list[CandidateSpec],
        round_seed: int,
        step_scale: float,
        eval_splits: dict[str, list[str]],
    ) -> dict[str, Any] | None:
        path = ab.AB_ROOT / f'{run_dir.name}_{stage_name}' / 'revalidated_best_loss_final_round.json'
        if not path.exists():
            return None
        payload = load_json(path)
        if not revalidated_round_payload_matches(
            payload,
            source_dir_name=f'{run_dir.name}_{stage_name}',
            expected_candidates=expected_candidates,
        ):
            return None
        adopted = adopt_revalidated_round_payload(
            payload,
            round_name=round_name,
            ab_name=f'{run_dir.name}_{stage_name}',
            expected_signature=expected_signature,
            seed=round_seed,
            step_scale=step_scale,
            eval_splits=eval_splits,
        )
        if not adopted.get('ranking'):
            return None
        atomic_write_json(run_dir / f'{round_name}.json', adopted)
        return adopted

    round0_eval_splits = ab.build_eval_splits(grouped, seed + 11, eval_file_count)
    round0_signature = round_cache_signature(
        round_name='p0_round0',
        ab_name=f'{run_dir.name}_p0_r0',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=round0_eval_splits,
        candidates=all_candidates,
        seed=seed + 101,
        step_scale=0.5,
        selector_weights=None,
    )
    round0 = adopt_authoritative_round(
        round_name='p0_round0',
        stage_name='p0_r0',
        expected_signature=round0_signature,
        expected_candidates=all_candidates,
        round_seed=seed + 101,
        step_scale=0.5,
        eval_splits=round0_eval_splits,
    )
    if round0 is None:
        return None
    top18 = build_p0_round0_survivors(
        round0['ranking'],
        all_index,
        candidate_subset=candidate_subset,
    )
    if not top18:
        return None

    round1_eval_splits = ab.build_eval_splits(grouped, seed + 22, eval_file_count)
    round1_signature = round_cache_signature(
        round_name='p0_round1',
        ab_name=f'{run_dir.name}_p0_r1',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=round1_eval_splits,
        candidates=top18,
        seed=seed + 202,
        step_scale=1.0,
        selector_weights=None,
    )
    round1 = adopt_authoritative_round(
        round_name='p0_round1',
        stage_name='p0_r1',
        expected_signature=round1_signature,
        expected_candidates=top18,
        round_seed=seed + 202,
        step_scale=1.0,
        eval_splits=round1_eval_splits,
    )
    if round1 is None:
        return None
    round2_candidates = build_p0_round1_survivors(
        round1['ranking'],
        max_keep=round2_max_candidates,
    )
    if not round2_candidates:
        return None

    round2_eval_splits = ab.build_eval_splits(grouped, seed + 33, eval_file_count)
    round2_signature = round_cache_signature(
        round_name='p0_round2',
        ab_name=f'{run_dir.name}_p0_r2',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=round2_eval_splits,
        candidates=round2_candidates,
        seed=seed + 303,
        step_scale=2.0,
        selector_weights=None,
    )
    round2 = adopt_authoritative_round(
        round_name='p0_round2',
        stage_name='p0_r2',
        expected_signature=round2_signature,
        expected_candidates=round2_candidates,
        round_seed=seed + 303,
        step_scale=2.0,
        eval_splits=round2_eval_splits,
    )
    if round2 is None:
        return None
    return round0, round1, round2


def run_arm_cached(
    *,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidate: CandidateSpec,
    seed: int,
    step_scale: float,
    ab_name: str,
) -> dict[str, Any]:
    arm_root = ab.AB_ROOT / ab_name / candidate.arm_name
    cache_path = arm_root / 'arm_result.json'
    cache_signature = arm_cache_signature(
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidate=candidate,
        seed=seed,
        step_scale=step_scale,
        ab_name=ab_name,
    )
    if cache_path.exists():
        payload = load_json(cache_path)
        if payload.get('cache_signature') == cache_signature:
            payload['cache_path'] = str(cache_path)
            return payload
        if arm_root.exists():
            remove_tree_with_retries(arm_root)
        else:
            cache_path.unlink()
    if arm_root.exists():
        remove_tree_with_retries(arm_root)

    merged_cfg = ab.merge_dict(base_cfg, candidate.cfg_overrides)
    try:
        result = ab.run_arm(
            merged_cfg,
            grouped,
            ab_name=ab_name,
            arm_name=candidate.arm_name,
            scheduler_profile=candidate.scheduler_profile,
            curriculum_profile=candidate.curriculum_profile,
            weight_profile=candidate.weight_profile,
            window_profile=candidate.window_profile,
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
        payload = {
            'ok': True,
            'arm_name': candidate.arm_name,
            'run': result,
            'candidate_meta': candidate.meta,
            'cache_signature': cache_signature,
        }
    except Exception as exc:
        payload = {
            'ok': False,
            'arm_name': candidate.arm_name,
            'candidate_meta': candidate.meta,
            'error': str(exc),
            'traceback': traceback.format_exc(),
            'cache_signature': cache_signature,
        }
    atomic_write_json(cache_path, payload)
    payload['cache_path'] = str(cache_path)
    return payload


def execute_round(
    *,
    run_dir: Path,
    round_name: str,
    ab_name: str,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidates: list[CandidateSpec],
    seed: int,
    step_scale: float,
    selector_weights: dict[str, float] | None = None,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> dict[str, Any]:
    summary_path = run_dir / f'{round_name}.json'
    current_round_signature = round_cache_signature(
        round_name=round_name,
        ab_name=ab_name,
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=candidates,
        seed=seed,
        step_scale=step_scale,
        selector_weights=selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    if summary_path.exists():
        payload = load_json(summary_path)
        if payload.get('round_signature') == current_round_signature:
            return payload

    raw_results: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = run_arm_cached(
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=eval_splits,
            candidate=candidate,
            seed=seed,
            step_scale=step_scale,
            ab_name=ab_name,
        )
        raw_results[candidate.arm_name] = payload
        entries.append(
            summarize_entry(
                candidate.arm_name,
                candidate,
                payload,
                ranking_mode=ranking_mode,
            )
        )

    ranked = rank_round_entries(
        entries,
        selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    payload = {
        'scenario_score_version': SCENARIO_SCORE_VERSION,
        'round_name': round_name,
        'ab_name': ab_name,
        'seed': seed,
        'step_scale': step_scale,
        'ranking_mode': ranking_mode,
        'eligibility_group_key': eligibility_group_key,
        'round_signature': current_round_signature,
        'evaluated_arms': len(candidates),
        'eval_split_counts': {
            'monitor_recent_files': len(eval_splits['monitor_recent_files']),
            'full_recent_files': len(eval_splits['full_recent_files']),
            'old_regression_files': len(eval_splits['old_regression_files']),
        },
        'best_loss': min((entry['full_recent_loss'] for entry in ranked if entry['valid']), default=None),
        'best_recent_policy_loss': min(
            (entry.get('recent_policy_loss', math.inf) for entry in ranked if entry['valid']),
            default=None,
        ),
        'ranking': ranked,
        'raw_results': raw_results,
    }
    atomic_write_json(summary_path, payload)
    return payload


def execute_round_multiseed(
    *,
    run_dir: Path,
    round_name: str,
    ab_name: str,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidates: list[CandidateSpec],
    seed: int,
    seed_offsets: list[int],
    step_scale: float,
    selector_weights: dict[str, float] | None = None,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> dict[str, Any]:
    summary_path = run_dir / f'{round_name}.json'
    current_round_signature = stable_payload_digest(
        apply_round_signature_ranking_fields(
            {
                'schema_version': ROUND_CACHE_SCHEMA_VERSION,
                'scenario_score_version': SCENARIO_SCORE_VERSION,
                'round_name': round_name,
                'ab_name': ab_name,
                'base_cfg': base_cfg,
                'grouped': grouped,
                'eval_splits': eval_splits,
                'candidates': [candidate_cache_payload(candidate, include_meta=True) for candidate in candidates],
                'seed': seed,
                'seed_offsets': seed_offsets,
                'step_scale': step_scale,
                'selector_weights': selector_weights,
            },
            ranking_mode=ranking_mode,
            eligibility_group_key=eligibility_group_key,
        )
    )
    if summary_path.exists():
        payload = load_json(summary_path)
        if payload.get('round_signature') == current_round_signature:
            return payload

    seed_rounds: dict[str, dict[str, Any]] = {}
    for seed_offset in seed_offsets:
        actual_seed = seed + seed_offset
        seed_label = f's{actual_seed}'
        seed_round_name = f'{round_name}__{seed_label}'
        seed_ab_name = f'{ab_name}_{seed_label}'
        seed_payload = execute_round(
            run_dir=run_dir,
            round_name=seed_round_name,
            ab_name=seed_ab_name,
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=eval_splits,
            candidates=candidates,
            seed=actual_seed,
            step_scale=step_scale,
            selector_weights=selector_weights,
            ranking_mode=ranking_mode,
            eligibility_group_key=eligibility_group_key,
        )
        seed_rounds[seed_label] = {
            'seed': actual_seed,
            'round_name': seed_round_name,
            'ab_name': seed_ab_name,
            'summary_path': str(run_dir / f'{seed_round_name}.json'),
        }
        seed_rounds[seed_label].update(seed_payload)

    ranked = rank_round_entries(
        [
            summarize_multiseed_entry(
                candidate.arm_name,
                candidate,
                {
                    seed_label: seed_rounds[seed_label]['raw_results'][candidate.arm_name]
                    for seed_label in seed_rounds
                },
                ranking_mode=ranking_mode,
            )
            for candidate in candidates
        ],
        selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    payload = {
        'scenario_score_version': SCENARIO_SCORE_VERSION,
        'round_name': round_name,
        'ab_name': ab_name,
        'seed': seed,
        'seed_offsets': list(seed_offsets),
        'actual_seeds': [seed + offset for offset in seed_offsets],
        'step_scale': step_scale,
        'ranking_mode': ranking_mode,
        'eligibility_group_key': eligibility_group_key,
        'round_signature': current_round_signature,
        'evaluated_arms': len(candidates),
        'eval_split_counts': {
            'monitor_recent_files': len(eval_splits['monitor_recent_files']),
            'full_recent_files': len(eval_splits['full_recent_files']),
            'old_regression_files': len(eval_splits['old_regression_files']),
        },
        'best_loss': min((entry['full_recent_loss'] for entry in ranked if entry['valid']), default=None),
        'best_recent_policy_loss': min(
            (entry.get('recent_policy_loss', math.inf) for entry in ranked if entry['valid']),
            default=None,
        ),
        'ranking': ranked,
        'seed_rounds': {
            seed_label: {
                'seed': seed_round['seed'],
                'round_name': seed_round['round_name'],
                'ab_name': seed_round['ab_name'],
                'summary_path': seed_round['summary_path'],
            }
            for seed_label, seed_round in seed_rounds.items()
        },
    }
    atomic_write_json(summary_path, payload)
    return payload


def summarize_multiseed_candidates(
    candidates: list[CandidateSpec],
    seed_rounds: dict[str, dict[str, Any]],
    selector_weights: dict[str, float] | None = None,
    *,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_seed_payloads = {
            seed_label: seed_round['raw_results'][candidate.arm_name]
            for seed_label, seed_round in seed_rounds.items()
            if candidate.arm_name in seed_round.get('raw_results', {})
        }
        if not candidate_seed_payloads:
            continue
        entries.append(
            summarize_multiseed_entry(
                candidate.arm_name,
                candidate,
                candidate_seed_payloads,
                ranking_mode=ranking_mode,
            )
        )
    return rank_round_entries(
        entries,
        selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )


def build_multiseed_payload(
    *,
    round_name: str,
    ab_name: str,
    seed: int,
    seed_offsets: list[int],
    step_scale: float,
    round_signature: str,
    eval_splits: dict[str, list[str]],
    seed_rounds: dict[str, dict[str, Any]],
    ranked: list[dict[str, Any]],
    evaluated_arms: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        'scenario_score_version': SCENARIO_SCORE_VERSION,
        'round_name': round_name,
        'ab_name': ab_name,
        'seed': seed,
        'seed_offsets': list(seed_offsets),
        'actual_seeds': [seed + offset for offset in seed_offsets],
        'step_scale': step_scale,
        'ranking_mode': extra.get('ranking_mode') if extra else None,
        'eligibility_group_key': extra.get('eligibility_group_key') if extra else None,
        'round_signature': round_signature,
        'evaluated_arms': evaluated_arms,
        'eval_split_counts': {
            'monitor_recent_files': len(eval_splits['monitor_recent_files']),
            'full_recent_files': len(eval_splits['full_recent_files']),
            'old_regression_files': len(eval_splits['old_regression_files']),
        },
        'best_loss': min((entry['full_recent_loss'] for entry in ranked if entry['valid']), default=None),
        'best_recent_policy_loss': min(
            (entry.get('recent_policy_loss', math.inf) for entry in ranked if entry['valid']),
            default=None,
        ),
        'ranking': ranked,
        'seed_rounds': {
            seed_label: {
                'seed': seed_round['seed'],
                'round_name': seed_round['round_name'],
                'ab_name': seed_round['ab_name'],
                'summary_path': seed_round['summary_path'],
            }
            for seed_label, seed_round in seed_rounds.items()
        },
    }
    if extra:
        payload.update(extra)
    return payload


def entry_group_value(entry: dict[str, Any], group_key: str) -> str:
    return str(entry.get(group_key, entry.get('candidate_meta', {}).get(group_key, '')))


def candidate_group_value(candidate: CandidateSpec, group_key: str) -> str:
    return str(candidate.meta.get(group_key, candidate.arm_name))


def group_ranked_entries(ranking: list[dict[str, Any]], group_key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in ranking:
        grouped.setdefault(entry_group_value(entry, group_key), []).append(entry)
    return grouped


def select_p1_solo_probe_candidates(
    ranking: list[dict[str, Any]],
    candidates: list[CandidateSpec],
) -> list[CandidateSpec]:
    selected_names: set[str] = set()
    grouped_entries = group_ranked_entries(ranking, 'protocol_arm')
    for protocol_entries in grouped_entries.values():
        family_entries: dict[str, list[dict[str, Any]]] = {}
        for entry in protocol_entries:
            family = str(entry.get('candidate_meta', {}).get('aux_family', ''))
            family_entries.setdefault(family, []).append(entry)
        for family in ('ce_only', 'rank', 'opp', 'danger'):
            entries = family_entries.get(family, [])
            limit = 1 if family == 'ce_only' else P1_SOLO_PROBE_KEEP_PER_FAMILY
            for entry in entries[:limit]:
                selected_names.add(entry['arm_name'])
    return [candidate for candidate in candidates if candidate.arm_name in selected_names]


def is_p1_protocol_decide_probe_meta(meta: dict[str, Any]) -> bool:
    return str(meta.get('aux_family', '')) == 'all_three'


def select_p1_protocol_decide_probe_candidates(
    ranking: list[dict[str, Any]],
    candidates: list[CandidateSpec],
    *,
    keep: int,
) -> list[CandidateSpec]:
    eligible_ranked = [
        entry for entry in ranking if is_p1_protocol_decide_probe_meta(entry.get('candidate_meta', {}))
    ]
    selected_names = set(select_group_top_k(eligible_ranked, 'protocol_arm', keep))
    return [candidate for candidate in candidates if candidate.arm_name in selected_names]


def is_p1_protocol_decide_diagnostic_meta(meta: dict[str, Any]) -> bool:
    return str(meta.get('aux_family', '')) == 'ce_only'


def rerank_filtered_entries(
    ranking: list[dict[str, Any]],
    *,
    entry_selector: Any,
    selector_weights: dict[str, float] | None = None,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
) -> list[dict[str, Any]]:
    filtered_entries = [entry for entry in ranking if entry_selector(entry)]
    if not filtered_entries:
        return []
    return rank_round_entries(
        filtered_entries,
        selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )


def select_group_probe_candidates(
    ranking: list[dict[str, Any]],
    candidates: list[CandidateSpec],
    *,
    group_key: str,
    keep: int,
) -> list[CandidateSpec]:
    selected_names = set(select_group_top_k(ranking, group_key, keep))
    return [candidate for candidate in candidates if candidate.arm_name in selected_names]


def entry_seed_loss_range(entry: dict[str, Any]) -> float | None:
    losses = [
        float(seed_summary.get('recent_policy_loss', seed_summary['full_recent_loss']))
        for seed_summary in entry.get('seed_summaries', [])
        if seed_summary.get('valid')
        and math.isfinite(float(seed_summary.get('recent_policy_loss', seed_summary.get('full_recent_loss', math.inf))))
    ]
    if len(losses) < 2:
        return None
    return max(losses) - min(losses)


def detect_progressive_ambiguous_groups(
    *,
    seed1_ranking: list[dict[str, Any]],
    probe_ranking: list[dict[str, Any]],
    group_key: str,
    ranking_mode: str = 'full_recent',
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    seed1_groups = group_ranked_entries(seed1_ranking, group_key)
    probe_groups = group_ranked_entries(probe_ranking, group_key)
    ambiguous_groups: set[str] = set()
    details: dict[str, dict[str, Any]] = {}
    for group, entries in probe_groups.items():
        if not entries:
            continue
        probe_top1 = entries[0]
        probe_top2 = entries[1] if len(entries) > 1 else None
        seed1_entries = seed1_groups.get(group, [])
        seed1_top1 = seed1_entries[0] if seed1_entries else None
        winner_flipped = (
            seed1_top1 is None
            or not seed1_top1.get('valid')
            or not probe_top1.get('valid')
            or seed1_top1['arm_name'] != probe_top1['arm_name']
        )
        gap = math.inf
        if (
            probe_top2 is not None
            and probe_top1.get('valid')
            and probe_top2.get('valid')
            and math.isfinite(float(probe_top1.get('comparison_recent_loss', probe_top1['full_recent_loss'])))
            and math.isfinite(float(probe_top2.get('comparison_recent_loss', probe_top2['full_recent_loss'])))
        ):
            gap = float(probe_top2.get('comparison_recent_loss', probe_top2['full_recent_loss'])) - float(
                probe_top1.get('comparison_recent_loss', probe_top1['full_recent_loss'])
            )
        noise_values = [
            noise
            for noise in (entry_seed_loss_range(entry) for entry in entries)
            if noise is not None and math.isfinite(noise)
        ]
        noise = statistics.median(noise_values) if noise_values else 0.0
        epsilon = P1_POLICY_LOSS_EPSILON if ranking_mode == 'policy_quality' else LOSS_EPSILON
        ambiguity_margin = max(epsilon, P1_PROGRESSIVE_NOISE_MARGIN_MULT * noise)
        ambiguous = winner_flipped or gap <= ambiguity_margin
        if ambiguous:
            ambiguous_groups.add(group)
        details[group] = {
            'seed1_top1': None if seed1_top1 is None else seed1_top1['arm_name'],
            'probe_top1': probe_top1['arm_name'],
            'probe_top2': None if probe_top2 is None else probe_top2['arm_name'],
            'winner_flipped': winner_flipped,
            'top12_gap': None if not math.isfinite(gap) else gap,
            'seed_noise_median': noise,
            'ambiguity_margin': ambiguity_margin,
            'ambiguous': ambiguous,
        }
    return ambiguous_groups, details


def build_progressive_active_candidates(
    *,
    all_candidates: list[CandidateSpec],
    probe_candidates: list[CandidateSpec],
    ambiguous_groups: set[str],
    group_key: str,
) -> list[CandidateSpec]:
    probe_names = {candidate.arm_name for candidate in probe_candidates}
    active: list[CandidateSpec] = []
    for candidate in all_candidates:
        group = candidate_group_value(candidate, group_key)
        if candidate.arm_name in probe_names or group in ambiguous_groups:
            active.append(candidate)
    return unique_candidates(active)


def execute_round_progressive_multiseed(
    *,
    run_dir: Path,
    round_name: str,
    ab_name: str,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidates: list[CandidateSpec],
    seed: int,
    seed_offsets: list[int],
    step_scale: float,
    probe_selector_name: str,
    probe_selector: Any,
    group_key: str,
    probe_signature_data: dict[str, Any] | None = None,
    selector_weights: dict[str, float] | None = None,
    ranking_mode: str = 'full_recent',
    eligibility_group_key: str | None = None,
    screening_diagnostic_name: str | None = None,
    screening_diagnostic_selector: Any | None = None,
) -> dict[str, Any]:
    if len(seed_offsets) <= 1:
        return execute_round_multiseed(
            run_dir=run_dir,
            round_name=round_name,
            ab_name=ab_name,
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=eval_splits,
            candidates=candidates,
            seed=seed,
            seed_offsets=seed_offsets,
            step_scale=step_scale,
            selector_weights=selector_weights,
            ranking_mode=ranking_mode,
            eligibility_group_key=eligibility_group_key,
        )

    summary_path = run_dir / f'{round_name}.json'
    current_round_signature = stable_payload_digest(
        apply_round_signature_ranking_fields(
            {
                'schema_version': ROUND_CACHE_SCHEMA_VERSION,
                'scenario_score_version': SCENARIO_SCORE_VERSION,
                'round_name': round_name,
                'ab_name': ab_name,
                'base_cfg': base_cfg,
                'grouped': grouped,
                'eval_splits': eval_splits,
                'candidates': [candidate_cache_payload(candidate, include_meta=True) for candidate in candidates],
                'seed': seed,
                'seed_offsets': seed_offsets,
                'step_scale': step_scale,
                'selector_weights': selector_weights,
                'seed_strategy': 'progressive_probe_then_expand',
                'probe_selector_name': probe_selector_name,
                'probe_signature_data': probe_signature_data,
                'group_key': group_key,
                'noise_margin_mult': P1_PROGRESSIVE_NOISE_MARGIN_MULT,
                'seed1_probe_compare_scope': 'probe_candidates_only',
                'screening_diagnostic_name': screening_diagnostic_name,
            },
            ranking_mode=ranking_mode,
            eligibility_group_key=eligibility_group_key,
        )
    )
    if summary_path.exists():
        payload = load_json(summary_path)
        if payload.get('round_signature') == current_round_signature:
            return payload

    first_offset, second_offset = seed_offsets[:2]
    first_seed = seed + first_offset
    second_seed = seed + second_offset
    first_label = f's{first_seed}'
    second_label = f's{second_seed}'
    seed_round_name = f'{round_name}__{first_label}'
    seed_ab_name = f'{ab_name}_{first_label}'
    seed1_payload = execute_round(
        run_dir=run_dir,
        round_name=seed_round_name,
        ab_name=seed_ab_name,
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=candidates,
        seed=first_seed,
        step_scale=step_scale,
        selector_weights=selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    seed_rounds: dict[str, dict[str, Any]] = {
        first_label: {
            'seed': first_seed,
            'round_name': seed_round_name,
            'ab_name': seed_ab_name,
            'summary_path': str(run_dir / f'{seed_round_name}.json'),
            **seed1_payload,
        }
    }

    probe_candidates = unique_candidates(probe_selector(seed1_payload['ranking'], candidates))
    probe_candidate_names = {candidate.arm_name for candidate in probe_candidates}
    seed1_probe_ranking = rerank_filtered_entries(
        seed1_payload['ranking'],
        entry_selector=lambda entry: entry['arm_name'] in probe_candidate_names,
        selector_weights=selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    screening_diagnostic = None
    if screening_diagnostic_name is not None and screening_diagnostic_selector is not None:
        screening_diagnostic = {
            'name': screening_diagnostic_name,
            'source_seed': first_seed,
            'source_round_name': seed_round_name,
            'source_summary_path': str(run_dir / f'{seed_round_name}.json'),
            'ranking': rerank_filtered_entries(
                seed1_payload['ranking'],
                entry_selector=screening_diagnostic_selector,
                selector_weights=selector_weights,
                ranking_mode=ranking_mode,
                eligibility_group_key=eligibility_group_key,
            ),
        }
    second_round_name = f'{round_name}__{second_label}'
    second_ab_name = f'{ab_name}_{second_label}'
    second_payload = execute_round(
        run_dir=run_dir,
        round_name=second_round_name,
        ab_name=second_ab_name,
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=probe_candidates,
        seed=second_seed,
        step_scale=step_scale,
        selector_weights=selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    probe_seed_rounds = {
        **seed_rounds,
        second_label: {
            'seed': second_seed,
            'round_name': second_round_name,
            'ab_name': second_ab_name,
            'summary_path': str(run_dir / f'{second_round_name}.json'),
            **second_payload,
        },
    }
    probe_ranking = summarize_multiseed_candidates(
        probe_candidates,
        probe_seed_rounds,
        selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    ambiguous_groups, ambiguity_details = detect_progressive_ambiguous_groups(
        seed1_ranking=seed1_probe_ranking,
        probe_ranking=probe_ranking,
        group_key=group_key,
        ranking_mode=ranking_mode,
    )
    decision_candidates = build_progressive_active_candidates(
        all_candidates=candidates,
        probe_candidates=probe_candidates,
        ambiguous_groups=ambiguous_groups,
        group_key=group_key,
    )
    if {candidate.arm_name for candidate in decision_candidates} != {candidate.arm_name for candidate in probe_candidates}:
        second_payload = execute_round(
            run_dir=run_dir,
            round_name=second_round_name,
            ab_name=second_ab_name,
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=eval_splits,
            candidates=decision_candidates,
            seed=second_seed,
            step_scale=step_scale,
            selector_weights=selector_weights,
            ranking_mode=ranking_mode,
            eligibility_group_key=eligibility_group_key,
        )

    seed_rounds[second_label] = {
        'seed': second_seed,
        'round_name': second_round_name,
        'ab_name': second_ab_name,
        'summary_path': str(run_dir / f'{second_round_name}.json'),
        **second_payload,
    }
    final_ranking = summarize_multiseed_candidates(
        decision_candidates,
        seed_rounds,
        selector_weights,
        ranking_mode=ranking_mode,
        eligibility_group_key=eligibility_group_key,
    )
    pruned_names = sorted(
        candidate.arm_name
        for candidate in candidates
        if candidate.arm_name not in {active.arm_name for active in decision_candidates}
    )
    payload = build_multiseed_payload(
        round_name=round_name,
        ab_name=ab_name,
        seed=seed,
        seed_offsets=[first_offset, second_offset],
        step_scale=step_scale,
        round_signature=current_round_signature,
        eval_splits=eval_splits,
        seed_rounds=seed_rounds,
        ranked=final_ranking,
        evaluated_arms=len(candidates),
        extra={
            'seed_strategy': 'progressive_probe_then_expand',
            'probe_selector_name': probe_selector_name,
            'probe_signature_data': probe_signature_data or {},
            'group_key': group_key,
            'ranking_mode': ranking_mode,
            'eligibility_group_key': eligibility_group_key,
            'seed1_probe_compare_scope': 'probe_candidates_only',
            'probe_candidate_count': len(probe_candidates),
            'decision_candidate_count': len(decision_candidates),
            'pruned_candidate_count': len(pruned_names),
            'pruned_arm_names': pruned_names,
            'expanded_groups': sorted(ambiguous_groups),
            'ambiguity_details': ambiguity_details,
            'screening_round': {
                'actual_seed': first_seed,
                'summary_path': str(run_dir / f'{seed_round_name}.json'),
            },
            'probe_round': {
                'actual_seed': second_seed,
                'summary_path': str(run_dir / f'{second_round_name}.json'),
            },
            'screening_diagnostic': screening_diagnostic,
        },
    )
    atomic_write_json(summary_path, payload)
    return payload


def candidate_from_entry(entry: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(
        arm_name=entry['arm_name'],
        scheduler_profile=entry['scheduler_profile'],
        curriculum_profile=entry['curriculum_profile'],
        weight_profile=entry['weight_profile'],
        window_profile=entry['window_profile'],
        cfg_overrides=entry['cfg_overrides'],
        meta=entry['candidate_meta'],
    )


def decode_protocol_meta(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        'scheduler_profile': candidate.scheduler_profile,
        'curriculum_profile': candidate.curriculum_profile,
        'weight_profile': candidate.weight_profile,
        'window_profile': candidate.window_profile,
        **candidate.meta,
    }


def build_rank_override(scale: float) -> dict[str, Any]:
    return {
        'supervised': {
            'rank_aux': {
                'base_weight': RANK_TEMPLATE['base_weight'] * scale,
                'south_factor': RANK_TEMPLATE['south_factor'],
                'all_last_factor': RANK_TEMPLATE['all_last_factor'],
                'gap_focus_points': RANK_TEMPLATE['gap_focus_points'],
                'gap_close_bonus': RANK_TEMPLATE['gap_close_bonus'],
                'max_weight': RANK_TEMPLATE['max_weight'] * scale,
            }
        }
    }


def build_aux_override(*, opp_weight: float, danger_weight: float, danger_enabled: bool = True) -> dict[str, Any]:
    return {
        'aux': {
            'opponent_state_weight': opp_weight,
            # HYBRID_GRAD winner, scaled to preserve the old sum-2.0 opp shape semantics.
            'opponent_shanten_weight': 0.8506568408072642,
            'opponent_tenpai_weight': 1.1493431591927359,
            'danger_enabled': danger_enabled and danger_weight > 0,
            'danger_weight': danger_weight,
            # 18K_STAT winner. Danger internally renormalizes these three mix weights.
            'danger_any_weight': 0.09042179466099699,
            'danger_value_weight': 0.8180402859274302,
            'danger_player_weight': 0.09153791941157279,
            'danger_focal_gamma': 0.0,
            'danger_ramp_steps': 1000,
            'danger_value_cap': 96000.0,
        }
    }


def rank_weight_mean_for_files(files: list[str], *, version: int, file_batch_size: int) -> float:
    dataset = SupervisedFileDatasetsIter(
        version=version,
        file_list=list(files),
        oracle=False,
        file_batch_size=file_batch_size,
        reserve_ratio=0.0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        shuffle_files=False,
        worker_torch_num_threads=1,
        worker_torch_num_interop_threads=1,
        rayon_num_threads=0,
        emit_opponent_state_labels=False,
        track_danger_labels=False,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=4096,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    total_weight = 0.0
    total_samples = 0
    for batch in loader:
        context_meta = torch.as_tensor(batch[4], dtype=torch.int64)
        weights = torch.full(
            (context_meta.shape[0],),
            float(RANK_TEMPLATE['base_weight']),
            dtype=torch.float32,
        )
        south_mask = context_meta[:, 1] == 1
        weights = weights * torch.where(
            south_mask,
            torch.full_like(weights, float(RANK_TEMPLATE['south_factor'])),
            torch.ones_like(weights),
        )
        all_last_mask = context_meta[:, 3].to(torch.bool)
        weights = weights * torch.where(
            all_last_mask,
            torch.full_like(weights, float(RANK_TEMPLATE['all_last_factor'])),
            torch.ones_like(weights),
        )
        nearest_gap = torch.minimum(context_meta[:, 6], context_meta[:, 7]).to(torch.float32) * 100.0
        closeness = (1.0 - nearest_gap / float(RANK_TEMPLATE['gap_focus_points'])).clamp(min=0.0, max=1.0)
        weights = weights * (1.0 + float(RANK_TEMPLATE['gap_close_bonus']) * closeness)
        weights = weights.clamp(max=float(RANK_TEMPLATE['max_weight']))
        total_weight += float(weights.sum().item())
        total_samples += int(context_meta.shape[0])
    if total_samples == 0:
        raise RuntimeError('rank weight calibration found zero samples')
    return total_weight / total_samples


def clamp_search_value(value: float, *, lower: float, upper: float, digits: int = 3) -> float:
    return round(min(max(float(value), lower), upper), digits)


def encode_budget_ratio(ratio: float) -> str:
    # Budget ratios are clamped to 3 decimal places, so arm names must preserve
    # thousandths to keep cache keys and output directories collision-free.
    return f'{int(round(ratio * 1000)):04d}'


def unique_candidates(candidates: list[CandidateSpec]) -> list[CandidateSpec]:
    unique_by_signature: dict[str, CandidateSpec] = {}
    for candidate in candidates:
        protocol_arm = str(candidate.meta.get('protocol_arm', candidate.arm_name))
        signature = stable_payload_digest(
            {
                'protocol_arm': protocol_arm,
                'scheduler_profile': candidate.scheduler_profile,
                'curriculum_profile': candidate.curriculum_profile,
                'weight_profile': candidate.weight_profile,
                'window_profile': candidate.window_profile,
                'cfg_overrides': candidate.cfg_overrides,
            }
        )
        unique_by_signature.setdefault(signature, candidate)
    return list(unique_by_signature.values())


def metric_or_default(metrics: dict[str, Any], key: str) -> float:
    return as_float(metrics or {}, key, 0.0)


def rank_effective_contribution(metrics: dict[str, Any]) -> float:
    return metric_or_default(metrics, 'rank_aux_weight_mean') * metric_or_default(metrics, 'rank_aux_raw_loss')


def clamp_budget_ratio(value: float) -> float:
    return clamp_search_value(
        value,
        lower=P1_BUDGET_RATIO_MIN,
        upper=P1_BUDGET_RATIO_MAX,
        digits=3,
    )


def infer_aux_family(*, rank_budget_ratio: float, opp_budget_ratio: float, danger_budget_ratio: float) -> str:
    active = [
        name
        for name, ratio in (
            ('rank', rank_budget_ratio),
            ('opp', opp_budget_ratio),
            ('danger', danger_budget_ratio),
        )
        if ratio > 0
    ]
    if not active:
        return 'ce_only'
    return '+'.join(active)


def normalize_triplet_shares(
    rank_share: float,
    opp_share: float,
    danger_share: float,
) -> tuple[float, float, float]:
    total = rank_share + opp_share + danger_share
    if total <= 0:
        raise ValueError('triplet shares must sum to a positive value')
    return (
        rank_share / total,
        opp_share / total,
        danger_share / total,
    )


def derive_weight_per_budget_unit(
    *,
    rank_effective_base: float,
    per_unit_effective: float,
    fallback_weight_per_budget_unit: float,
) -> float:
    if rank_effective_base <= 0 or per_unit_effective <= 0:
        return clamp_search_value(
            fallback_weight_per_budget_unit,
            lower=P1_AUX_WEIGHT_MIN,
            upper=P1_AUX_WEIGHT_MAX,
            digits=3,
        )
    return clamp_search_value(
        rank_effective_base / per_unit_effective,
        lower=P1_AUX_WEIGHT_MIN,
        upper=P1_AUX_WEIGHT_MAX,
        digits=3,
    )


def blend_positive_calibration_values(
    *,
    loss_value: float,
    grad_value: float,
    fallback: float,
) -> float:
    if loss_value > 0 and grad_value > 0:
        return clamp_search_value(
            math.sqrt(loss_value * grad_value),
            lower=P1_AUX_WEIGHT_MIN,
            upper=P1_AUX_WEIGHT_MAX,
            digits=3,
        )
    if loss_value > 0:
        return clamp_search_value(
            loss_value,
            lower=P1_AUX_WEIGHT_MIN,
            upper=P1_AUX_WEIGHT_MAX,
            digits=3,
        )
    if grad_value > 0:
        return clamp_search_value(
            grad_value,
            lower=P1_AUX_WEIGHT_MIN,
            upper=P1_AUX_WEIGHT_MAX,
            digits=3,
        )
    return clamp_search_value(
        fallback,
        lower=P1_AUX_WEIGHT_MIN,
        upper=P1_AUX_WEIGHT_MAX,
        digits=3,
    )


def inherited_single_head_probe_reference(
    inherited_single_head: dict[str, Any],
) -> dict[str, float]:
    probe_weight = float(inherited_single_head.get('probe_weight', P1_CALIBRATION_PROBE_WEIGHT) or 0.0)
    if probe_weight <= 0:
        probe_weight = P1_CALIBRATION_PROBE_WEIGHT
    return {
        'probe_weight': probe_weight,
        'rank_effective': float(inherited_single_head.get('rank_effective_base', 0.0) or 0.0),
        'opp_effective': float(inherited_single_head.get('opp_effective_per_unit', 0.0) or 0.0) * probe_weight,
        'danger_effective': float(inherited_single_head.get('danger_effective_per_unit', 0.0) or 0.0)
        * probe_weight,
        'rank_phi_grad_rms': float(inherited_single_head.get('rank_grad_effective_base', 0.0) or 0.0),
        'opponent_phi_grad_rms': float(
            inherited_single_head.get('opp_grad_effective_per_unit', 0.0) or 0.0
        )
        * probe_weight,
        'danger_phi_grad_rms': float(
            inherited_single_head.get('danger_grad_effective_per_unit', 0.0) or 0.0
        )
        * probe_weight,
    }


def resolve_single_head_expected_value(
    observed_value: Any,
    inherited_value: float,
    *,
    prefer_inherited: bool,
) -> float:
    if prefer_inherited and inherited_value > 0:
        return inherited_value
    resolved = float(observed_value or 0.0)
    if resolved > 0:
        return resolved
    return inherited_value if inherited_value > 0 else 0.0


def normalize_p1_calibration_role(role: Any) -> str:
    return P1_CALIBRATION_ROLE_ALIASES.get(str(role or ''), str(role or ''))


def loss_combo_factor(*, observed: float, expected: float) -> float:
    if observed <= 0 or expected <= 0:
        return 1.0
    return clamp_combo_factor(observed / expected)


def gradient_pair_combo_factor_from_components(
    lhs_rms: float,
    rhs_rms: float,
    cosine: float,
) -> float:
    lhs_rms = max(float(lhs_rms), 0.0)
    rhs_rms = max(float(rhs_rms), 0.0)
    denom = lhs_rms + rhs_rms
    if denom <= 1e-12:
        return 1.0
    cosine = max(-1.0, min(1.0, float(cosine)))
    combo_sq = max(lhs_rms * lhs_rms + rhs_rms * rhs_rms + 2.0 * lhs_rms * rhs_rms * cosine, 0.0)
    return clamp_combo_factor(math.sqrt(combo_sq) / denom)


def gradient_triple_combo_factor_from_components(
    first_rms: float,
    second_rms: float,
    third_rms: float,
    *,
    cosine_12: float,
    cosine_13: float,
    cosine_23: float,
) -> float:
    first_rms = max(float(first_rms), 0.0)
    second_rms = max(float(second_rms), 0.0)
    third_rms = max(float(third_rms), 0.0)
    denom = first_rms + second_rms + third_rms
    if denom <= 1e-12:
        return 1.0
    cosine_12 = max(-1.0, min(1.0, float(cosine_12)))
    cosine_13 = max(-1.0, min(1.0, float(cosine_13)))
    cosine_23 = max(-1.0, min(1.0, float(cosine_23)))
    combo_sq = max(
        first_rms * first_rms
        + second_rms * second_rms
        + third_rms * third_rms
        + 2.0 * first_rms * second_rms * cosine_12
        + 2.0 * first_rms * third_rms * cosine_13
        + 2.0 * second_rms * third_rms * cosine_23,
        0.0,
    )
    return clamp_combo_factor(math.sqrt(combo_sq) / denom)


def blend_combo_factors(
    *,
    loss_value: float,
    grad_value: float,
    fallback: float,
) -> float:
    if loss_value > 0 and grad_value > 0:
        return clamp_combo_factor(math.sqrt(loss_value * grad_value))
    if loss_value > 0:
        return clamp_combo_factor(loss_value)
    if grad_value > 0:
        return clamp_combo_factor(grad_value)
    return clamp_combo_factor(fallback)


def clamp_combo_factor(value: float) -> float:
    return clamp_search_value(
        value,
        lower=P1_COMBO_FACTOR_MIN,
        upper=P1_COMBO_FACTOR_MAX,
        digits=3,
    )


def budget_ratio_to_aux_weight(budget_ratio: float, *, weight_per_budget_unit: float) -> float:
    return clamp_search_value(
        budget_ratio * weight_per_budget_unit,
        lower=P1_AUX_WEIGHT_MIN,
        upper=P1_AUX_WEIGHT_MAX,
        digits=3,
    )


def build_gradient_calibration_override() -> dict[str, Any]:
    return {
        'supervised': {
            'gradient_calibration': {
                'enabled': True,
                'split': 'full_recent',
                'max_batches': P1_GRAD_CALIBRATION_MAX_BATCHES,
            }
        }
    }


def p1_calibration_probe_specs(calibration_mode: str) -> list[tuple[str, float, float, float]]:
    if calibration_mode == P1_CALIBRATION_MODE_COMBO_ONLY:
        # Slim mode intentionally inherits the frozen single-head baseline
        # instead of regenerating pure single-head probes in the current run.
        return [
            ('rank_opp_probe', 1.0, P1_CALIBRATION_PROBE_WEIGHT, 0.0),
            ('rank_danger_probe', 1.0, 0.0, P1_CALIBRATION_PROBE_WEIGHT),
            ('opp_danger_probe', 0.0, P1_CALIBRATION_PROBE_WEIGHT, P1_CALIBRATION_PROBE_WEIGHT),
            ('triple_probe', 1.0, P1_CALIBRATION_PROBE_WEIGHT, P1_CALIBRATION_PROBE_WEIGHT),
        ]
    if calibration_mode == P1_CALIBRATION_MODE_FULL:
        return [
            ('rank_only', 1.0, 0.0, 0.0),
            ('rank_opp_probe', 1.0, P1_CALIBRATION_PROBE_WEIGHT, 0.0),
            ('rank_danger_probe', 1.0, 0.0, P1_CALIBRATION_PROBE_WEIGHT),
            ('opp_danger_probe', 0.0, P1_CALIBRATION_PROBE_WEIGHT, P1_CALIBRATION_PROBE_WEIGHT),
            ('triple_probe', 1.0, P1_CALIBRATION_PROBE_WEIGHT, P1_CALIBRATION_PROBE_WEIGHT),
        ]
    raise ValueError(f'unknown p1 calibration mode: {calibration_mode}')


def resolve_p1_calibration_protocol_arms(
    protocols: list[CandidateSpec],
    calibration_protocol_arms: list[str] | None,
) -> tuple[list[str], list[CandidateSpec]]:
    if calibration_protocol_arms is None:
        requested_calibration_protocol_arms = list(P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS)
    else:
        requested_calibration_protocol_arms = list(calibration_protocol_arms)
    resolved_calibration_protocol_arms = dedupe_string_items(requested_calibration_protocol_arms)
    protocol_index = {
        str(protocol.meta.get('protocol_arm', protocol.arm_name)): protocol
        for protocol in protocols
    }
    calibration_protocols: list[CandidateSpec] = []
    missing: list[str] = []
    for arm in resolved_calibration_protocol_arms:
        protocol = protocol_index.get(arm)
        if protocol is None:
            try:
                protocol = build_protocol_candidate(arm)
            except ValueError:
                missing.append(arm)
                continue
        calibration_protocols.append(protocol)
    if missing:
        raise ValueError(f'unknown calibration protocol arms for p1 run: {sorted(missing)}')
    return resolved_calibration_protocol_arms, calibration_protocols


def make_p1_budget_candidate(
    protocol: CandidateSpec,
    *,
    calibration: dict[str, Any],
    rank_budget_ratio: float,
    opp_budget_ratio: float,
    danger_budget_ratio: float,
    stage: str,
    source_arm: str | None = None,
    family: str | None = None,
) -> CandidateSpec:
    protocol_arm = str(protocol.meta.get('protocol_arm', protocol.arm_name))
    rank_budget_ratio = clamp_budget_ratio(rank_budget_ratio)
    opp_budget_ratio = clamp_budget_ratio(opp_budget_ratio)
    danger_budget_ratio = clamp_budget_ratio(danger_budget_ratio)
    rank_scale = rank_budget_ratio
    effective_budget_base = float(calibration.get('rank_effective_base', 0.0) or 0.0)
    opp_weight_per_budget_unit = float(
        calibration.get('opp_weight_per_budget_unit', P1_DEFAULT_OPP_WEIGHT_PER_BUDGET) or 0.0
    )
    danger_weight_per_budget_unit = float(
        calibration.get('danger_weight_per_budget_unit', P1_DEFAULT_DANGER_WEIGHT_PER_BUDGET) or 0.0
    )
    protocol_rank_opp_combo_factor = float(
        calibration.get('protocol_rank_opp_combo_factors', {}).get(
            protocol_arm,
            calibration.get('rank_opp_combo_factor', 1.0),
        )
        or 1.0
    )
    protocol_rank_danger_combo_factor = float(
        calibration.get('protocol_rank_danger_combo_factors', {}).get(
            protocol_arm,
            calibration.get('rank_danger_combo_factor', 1.0),
        )
        or 1.0
    )
    protocol_opp_danger_combo_factor = float(
        calibration.get('protocol_opp_danger_combo_factors', {}).get(
            protocol_arm,
            calibration.get(
                'opp_danger_combo_factor',
                calibration.get('joint_combo_factor', 1.0),
            ),
        )
        or 1.0
    )
    protocol_triple_combo_factor = float(
        calibration.get('protocol_triple_combo_factors', {}).get(
            protocol_arm,
            calibration.get('triple_combo_factor', 1.0),
        )
        or 1.0
    )
    opp_weight = budget_ratio_to_aux_weight(
        opp_budget_ratio,
        weight_per_budget_unit=opp_weight_per_budget_unit,
    )
    danger_weight = budget_ratio_to_aux_weight(
        danger_budget_ratio,
        weight_per_budget_unit=danger_weight_per_budget_unit,
    )
    active_heads = tuple(
        head
        for head, ratio in (
            ('rank', rank_budget_ratio),
            ('opp', opp_budget_ratio),
            ('danger', danger_budget_ratio),
        )
        if ratio > 0
    )
    applied_combo_mode = 'none'
    applied_combo_factor = 1.0
    if active_heads == ('rank', 'opp'):
        applied_combo_mode = 'rank_opp'
        applied_combo_factor = protocol_rank_opp_combo_factor
    elif active_heads == ('rank', 'danger'):
        applied_combo_mode = 'rank_danger'
        applied_combo_factor = protocol_rank_danger_combo_factor
    elif active_heads == ('opp', 'danger'):
        applied_combo_mode = 'opp_danger'
        applied_combo_factor = protocol_opp_danger_combo_factor
    elif active_heads == ('rank', 'opp', 'danger'):
        applied_combo_mode = 'triple'
        applied_combo_factor = protocol_triple_combo_factor
    if applied_combo_factor > 0 and applied_combo_mode != 'none':
        compensation = 1.0 / applied_combo_factor
        if opp_weight > 0:
            opp_weight = clamp_search_value(
                opp_weight * compensation,
                lower=P1_AUX_WEIGHT_MIN,
                upper=P1_AUX_WEIGHT_MAX,
                digits=3,
            )
        if danger_weight > 0:
            danger_weight = clamp_search_value(
                danger_weight * compensation,
                lower=P1_AUX_WEIGHT_MIN,
                upper=P1_AUX_WEIGHT_MAX,
                digits=3,
            )
    rank_target_budget = effective_budget_base * rank_budget_ratio
    opp_target_budget = effective_budget_base * opp_budget_ratio
    danger_target_budget = effective_budget_base * danger_budget_ratio
    total_budget_est = rank_target_budget + opp_target_budget + danger_target_budget
    name = (
        f'B_r{encode_budget_ratio(rank_budget_ratio)}'
        f'_o{encode_budget_ratio(opp_budget_ratio)}'
        f'_d{encode_budget_ratio(danger_budget_ratio)}'
    )
    cfg_overrides = ab.merge_dict(deepcopy(protocol.cfg_overrides), build_rank_override(rank_scale))
    cfg_overrides = ab.merge_dict(
        cfg_overrides,
        build_aux_override(
            opp_weight=opp_weight,
            danger_weight=danger_weight,
            danger_enabled=danger_weight > 0,
        ),
    )
    return CandidateSpec(
        arm_name=f'{protocol_arm}__{name}',
        scheduler_profile=protocol.scheduler_profile,
        curriculum_profile=protocol.curriculum_profile,
        weight_profile=protocol.weight_profile,
        window_profile=protocol.window_profile,
        cfg_overrides=cfg_overrides,
        meta={
            'stage': stage,
            'protocol_arm': protocol_arm,
            'candidate_name': name,
            'shared_rank_template_mean': calibration.get('shared_rank_template_mean', 0.0),
            'effective_budget_base': effective_budget_base,
            'rank_budget_ratio': rank_budget_ratio,
            'opp_budget_ratio': opp_budget_ratio,
            'danger_budget_ratio': danger_budget_ratio,
            'rank_scale': rank_scale,
            'rank_target_budget': rank_target_budget,
            'opp_target_budget': opp_target_budget,
            'danger_target_budget': danger_target_budget,
            'opp_weight_per_budget_unit': opp_weight_per_budget_unit,
            'danger_weight_per_budget_unit': danger_weight_per_budget_unit,
            'opp_weight': opp_weight,
            'danger_weight': danger_weight,
            'total_budget_est': total_budget_est,
            'protocol_rank_opp_combo_factor': protocol_rank_opp_combo_factor,
            'protocol_rank_danger_combo_factor': protocol_rank_danger_combo_factor,
            'protocol_opp_danger_combo_factor': protocol_opp_danger_combo_factor,
            'protocol_triple_combo_factor': protocol_triple_combo_factor,
            'protocol_joint_combo_factor': protocol_opp_danger_combo_factor,
            'active_heads': list(active_heads),
            'applied_combo_mode': applied_combo_mode,
            'applied_combo_factor': applied_combo_factor,
            'aux_family': family or infer_aux_family(
                rank_budget_ratio=rank_budget_ratio,
                opp_budget_ratio=opp_budget_ratio,
                danger_budget_ratio=danger_budget_ratio,
            ),
            'scheduler_profile': protocol.scheduler_profile,
            'curriculum_profile': protocol.curriculum_profile,
            'weight_profile': protocol.weight_profile,
            'window_profile': protocol.window_profile,
            'source_arm': source_arm,
        },
    )


def make_p1_triplet_candidate(
    protocol: CandidateSpec,
    *,
    calibration: dict[str, Any],
    total_budget_ratio: float,
    rank_share: float,
    opp_share: float,
    danger_share: float,
    stage: str,
    source_arm: str | None = None,
    family: str = 'all_three',
    mix_name: str | None = None,
) -> CandidateSpec:
    rank_share, opp_share, danger_share = normalize_triplet_shares(
        rank_share,
        opp_share,
        danger_share,
    )
    total_budget_ratio = clamp_budget_ratio(total_budget_ratio)
    base_candidate = make_p1_budget_candidate(
        protocol,
        calibration=calibration,
        rank_budget_ratio=total_budget_ratio * rank_share,
        opp_budget_ratio=total_budget_ratio * opp_share,
        danger_budget_ratio=total_budget_ratio * danger_share,
        stage=stage,
        source_arm=source_arm,
        family=family,
    )
    return CandidateSpec(
        arm_name=base_candidate.arm_name,
        scheduler_profile=base_candidate.scheduler_profile,
        curriculum_profile=base_candidate.curriculum_profile,
        weight_profile=base_candidate.weight_profile,
        window_profile=base_candidate.window_profile,
        cfg_overrides=base_candidate.cfg_overrides,
        meta={
            **base_candidate.meta,
            'total_budget_ratio': total_budget_ratio,
            'mix_name': mix_name,
            'mix_rank_share': rank_share,
            'mix_opp_share': opp_share,
            'mix_danger_share': danger_share,
        },
    )


def build_p1_calibration_candidates(
    protocols: list[CandidateSpec],
    shared_rank_template_mean: float,
    *,
    calibration_mode: str = P1_CALIBRATION_DEFAULT_MODE,
) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for protocol in protocols:
        protocol_arm = str(protocol.meta.get('protocol_arm', protocol.arm_name))
        probes = p1_calibration_probe_specs(calibration_mode)
        for role, rank_scale, opp_weight, danger_weight in probes:
            cfg_overrides = ab.merge_dict(deepcopy(protocol.cfg_overrides), build_rank_override(rank_scale))
            cfg_overrides = ab.merge_dict(
                cfg_overrides,
                build_aux_override(
                    opp_weight=opp_weight,
                    danger_weight=danger_weight,
                    danger_enabled=danger_weight > 0,
                ),
            )
            cfg_overrides = ab.merge_dict(
                cfg_overrides,
                build_gradient_calibration_override(),
            )
            candidates.append(
                CandidateSpec(
                    arm_name=f'{protocol.arm_name}__CAL_{role}',
                    scheduler_profile=protocol.scheduler_profile,
                    curriculum_profile=protocol.curriculum_profile,
                    weight_profile=protocol.weight_profile,
                    window_profile=protocol.window_profile,
                    cfg_overrides=cfg_overrides,
                    meta={
                        'stage': 'P1_calibration',
                        'protocol_arm': protocol_arm,
                        'candidate_name': role,
                        'shared_rank_template_mean': shared_rank_template_mean,
                        'rank_scale': rank_scale,
                        'opp_weight': opp_weight,
                        'danger_weight': danger_weight,
                        'calibration_role': role,
                        'calibration_mode': calibration_mode,
                        'active_heads': [
                            head
                            for head, enabled in (
                                ('rank', rank_scale > 0),
                                ('opp', opp_weight > 0),
                                ('danger', danger_weight > 0),
                            )
                            if enabled
                        ],
                        'probe_weight': P1_CALIBRATION_PROBE_WEIGHT,
                        'scheduler_profile': protocol.scheduler_profile,
                        'curriculum_profile': protocol.curriculum_profile,
                        'weight_profile': protocol.weight_profile,
                        'window_profile': protocol.window_profile,
                    },
                )
            )
    return candidates


def derive_p1_budget_calibration(
    calibration_round: dict[str, Any],
    *,
    requested_calibration_mode: str | None = None,
    inherited_single_head: dict[str, Any] | None = None,
    inherited_single_head_source: str | None = None,
) -> dict[str, Any]:
    rank_effectives: list[float] = []
    opp_per_unit: list[float] = []
    danger_per_unit: list[float] = []
    rank_grad_effectives: list[float] = []
    opp_grad_per_unit: list[float] = []
    danger_grad_per_unit: list[float] = []
    details: list[dict[str, Any]] = []
    protocol_role_details: dict[str, dict[str, dict[str, float]]] = {}
    observed_calibration_modes: set[str] = set()
    for entry in calibration_round.get('ranking', []):
        if not entry.get('valid'):
            continue
        metrics = entry.get('full_recent_metrics') or {}
        role = normalize_p1_calibration_role(entry.get('candidate_meta', {}).get('calibration_role'))
        protocol_arm = str(entry.get('candidate_meta', {}).get('protocol_arm'))
        observed_mode = str(entry.get('candidate_meta', {}).get('calibration_mode') or '').strip()
        if observed_mode:
            observed_calibration_modes.add(observed_mode)
        rank_eff = rank_effective_contribution(metrics)
        opp_eff = metric_or_default(metrics, 'opponent_aux_loss')
        danger_eff = metric_or_default(metrics, 'danger_aux_loss')
        rank_grad_eff = metric_or_default(metrics, 'rank_phi_grad_rms')
        opp_grad_eff = metric_or_default(metrics, 'opponent_phi_grad_rms')
        danger_grad_eff = metric_or_default(metrics, 'danger_phi_grad_rms')
        rank_opp_cos = metric_or_default(metrics, 'rank_opponent_phi_grad_cos')
        rank_danger_cos = metric_or_default(metrics, 'rank_danger_phi_grad_cos')
        opp_danger_cos = metric_or_default(metrics, 'opp_danger_phi_grad_cos')
        opp_danger_combo_eff = metric_or_default(metrics, 'opp_danger_phi_combo_factor')
        protocol_role_details.setdefault(protocol_arm, {})
        protocol_role_details[protocol_arm][role] = {
            'rank_effective': rank_eff,
            'opp_effective': opp_eff,
            'danger_effective': danger_eff,
            'rank_phi_grad_rms': rank_grad_eff,
            'opponent_phi_grad_rms': opp_grad_eff,
            'danger_phi_grad_rms': danger_grad_eff,
            'rank_opponent_phi_grad_cos': rank_opp_cos,
            'rank_danger_phi_grad_cos': rank_danger_cos,
            'opp_danger_phi_grad_cos': opp_danger_cos,
            'opp_danger_phi_combo_factor': opp_danger_combo_eff,
        }
        details.append(
            {
                'arm_name': entry.get('arm_name'),
                'role': role,
                'protocol_arm': protocol_arm,
                'rank_effective': rank_eff,
                'opp_effective': opp_eff,
                'danger_effective': danger_eff,
                'rank_phi_grad_rms': rank_grad_eff,
                'opponent_phi_grad_rms': opp_grad_eff,
                'danger_phi_grad_rms': danger_grad_eff,
                'rank_policy_phi_grad_cos': metric_or_default(metrics, 'rank_policy_phi_grad_cos'),
                'opponent_policy_phi_grad_cos': metric_or_default(metrics, 'opponent_policy_phi_grad_cos'),
                'danger_policy_phi_grad_cos': metric_or_default(metrics, 'danger_policy_phi_grad_cos'),
                'rank_opponent_phi_grad_cos': rank_opp_cos,
                'rank_danger_phi_grad_cos': rank_danger_cos,
                'opp_danger_phi_grad_cos': opp_danger_cos,
                'opp_danger_phi_combo_factor': opp_danger_combo_eff,
                'grad_probe_batches': metric_or_default(metrics, 'grad_probe_batches'),
            }
        )
        if role == 'rank_only' and rank_eff > 0:
            rank_effectives.append(rank_eff)
        if role == 'rank_only' and rank_grad_eff > 0:
            rank_grad_effectives.append(rank_grad_eff)
        if role == 'rank_opp_probe' and opp_eff > 0:
            opp_per_unit.append(opp_eff / max(P1_CALIBRATION_PROBE_WEIGHT, 1e-8))
        if role == 'rank_opp_probe' and opp_grad_eff > 0:
            opp_grad_per_unit.append(opp_grad_eff / max(P1_CALIBRATION_PROBE_WEIGHT, 1e-8))
        if role == 'rank_danger_probe' and danger_eff > 0:
            danger_per_unit.append(danger_eff / max(P1_CALIBRATION_PROBE_WEIGHT, 1e-8))
        if role == 'rank_danger_probe' and danger_grad_eff > 0:
            danger_grad_per_unit.append(danger_grad_eff / max(P1_CALIBRATION_PROBE_WEIGHT, 1e-8))
    requested_calibration_mode = str(requested_calibration_mode or '').strip()
    if len(observed_calibration_modes) == 1:
        calibration_mode = sorted(observed_calibration_modes)[0]
    elif requested_calibration_mode in P1_CALIBRATION_MODE_CHOICES:
        # Keep the user-selected mode when the round produced no valid probes.
        calibration_mode = requested_calibration_mode
    else:
        calibration_mode = P1_CALIBRATION_MODE_FULL
    inherited_single_head = {} if inherited_single_head is None else dict(inherited_single_head)
    use_inherited_single_head = (
        calibration_mode == P1_CALIBRATION_MODE_COMBO_ONLY and bool(inherited_single_head)
    )
    effective_inherited_single_head = inherited_single_head if use_inherited_single_head else {}
    effective_inherited_single_head_source = (
        inherited_single_head_source if use_inherited_single_head else None
    )
    inherited_probe_reference = inherited_single_head_probe_reference(effective_inherited_single_head)
    if use_inherited_single_head:
        rank_effective_base = float(
            effective_inherited_single_head.get('rank_effective_base', 0.0) or 0.0
        )
        opp_effective_per_unit = float(
            effective_inherited_single_head.get('opp_effective_per_unit', 0.0) or 0.0
        )
        danger_effective_per_unit = float(
            effective_inherited_single_head.get('danger_effective_per_unit', 0.0) or 0.0
        )
        rank_grad_effective_base = float(
            effective_inherited_single_head.get('rank_grad_effective_base', 0.0) or 0.0
        )
        opp_grad_effective_per_unit = float(
            effective_inherited_single_head.get('opp_grad_effective_per_unit', 0.0) or 0.0
        )
        danger_grad_effective_per_unit = float(
            effective_inherited_single_head.get('danger_grad_effective_per_unit', 0.0) or 0.0
        )
        opp_per_unit = []
        danger_per_unit = []
        opp_grad_per_unit = []
        danger_grad_per_unit = []
    else:
        rank_effective_base = statistics.median(rank_effectives) if rank_effectives else float(
            effective_inherited_single_head.get('rank_effective_base', 0.0) or 0.0
        )
        opp_effective_per_unit = statistics.median(opp_per_unit) if opp_per_unit else float(
            effective_inherited_single_head.get('opp_effective_per_unit', 0.0) or 0.0
        )
        danger_effective_per_unit = statistics.median(danger_per_unit) if danger_per_unit else float(
            effective_inherited_single_head.get('danger_effective_per_unit', 0.0) or 0.0
        )
        rank_grad_effective_base = statistics.median(rank_grad_effectives) if rank_grad_effectives else float(
            effective_inherited_single_head.get('rank_grad_effective_base', 0.0) or 0.0
        )
        opp_grad_effective_per_unit = statistics.median(opp_grad_per_unit) if opp_grad_per_unit else float(
            effective_inherited_single_head.get('opp_grad_effective_per_unit', 0.0) or 0.0
        )
        danger_grad_effective_per_unit = statistics.median(danger_grad_per_unit) if danger_grad_per_unit else float(
            effective_inherited_single_head.get('danger_grad_effective_per_unit', 0.0) or 0.0
        )
    combo_names = ('rank_opp', 'rank_danger', 'opp_danger', 'triple')
    combo_loss_values: dict[str, list[float]] = {name: [] for name in combo_names}
    combo_grad_values: dict[str, list[float]] = {name: [] for name in combo_names}
    protocol_combo_loss: dict[str, dict[str, float]] = {name: {} for name in combo_names}
    protocol_combo_grad: dict[str, dict[str, float]] = {name: {} for name in combo_names}
    protocol_combo: dict[str, dict[str, float]] = {name: {} for name in combo_names}
    for protocol_arm, protocol_roles in protocol_role_details.items():
        rank_only = protocol_roles.get('rank_only', {})
        rank_opp = protocol_roles.get('rank_opp_probe', {})
        rank_danger = protocol_roles.get('rank_danger_probe', {})
        opp_danger = protocol_roles.get('opp_danger_probe', {})
        triple = protocol_roles.get('triple_probe', {})
        expected_rank_effective = resolve_single_head_expected_value(
            rank_only.get('rank_effective', 0.0),
            inherited_probe_reference.get('rank_effective', 0.0),
            prefer_inherited=use_inherited_single_head,
        )
        expected_opp_effective = resolve_single_head_expected_value(
            rank_opp.get('opp_effective', 0.0),
            inherited_probe_reference.get('opp_effective', 0.0),
            prefer_inherited=use_inherited_single_head,
        )
        expected_danger_effective = resolve_single_head_expected_value(
            rank_danger.get('danger_effective', 0.0),
            inherited_probe_reference.get('danger_effective', 0.0),
            prefer_inherited=use_inherited_single_head,
        )

        rank_opp_loss = loss_combo_factor(
            observed=float(rank_opp.get('rank_effective', 0.0)) + float(rank_opp.get('opp_effective', 0.0)),
            expected=expected_rank_effective + expected_opp_effective,
        )
        rank_opp_grad = gradient_pair_combo_factor_from_components(
            float(rank_opp.get('rank_phi_grad_rms', 0.0)),
            float(rank_opp.get('opponent_phi_grad_rms', 0.0)),
            float(rank_opp.get('rank_opponent_phi_grad_cos', 0.0)),
        )

        rank_danger_loss = loss_combo_factor(
            observed=float(rank_danger.get('rank_effective', 0.0)) + float(rank_danger.get('danger_effective', 0.0)),
            expected=expected_rank_effective + expected_danger_effective,
        )
        rank_danger_grad = gradient_pair_combo_factor_from_components(
            float(rank_danger.get('rank_phi_grad_rms', 0.0)),
            float(rank_danger.get('danger_phi_grad_rms', 0.0)),
            float(rank_danger.get('rank_danger_phi_grad_cos', 0.0)),
        )

        opp_danger_loss = loss_combo_factor(
            observed=float(opp_danger.get('opp_effective', 0.0)) + float(opp_danger.get('danger_effective', 0.0)),
            expected=expected_opp_effective + expected_danger_effective,
        )
        opp_danger_grad_metric = float(opp_danger.get('opp_danger_phi_combo_factor', 0.0))
        opp_danger_grad = (
            clamp_combo_factor(opp_danger_grad_metric)
            if opp_danger_grad_metric > 0
            else gradient_pair_combo_factor_from_components(
                float(opp_danger.get('opponent_phi_grad_rms', 0.0)),
                float(opp_danger.get('danger_phi_grad_rms', 0.0)),
                float(opp_danger.get('opp_danger_phi_grad_cos', 0.0)),
            )
        )

        triple_loss = loss_combo_factor(
            observed=(
                float(triple.get('rank_effective', 0.0))
                + float(triple.get('opp_effective', 0.0))
                + float(triple.get('danger_effective', 0.0))
            ),
            expected=expected_rank_effective + expected_opp_effective + expected_danger_effective,
        )
        triple_grad = gradient_triple_combo_factor_from_components(
            float(triple.get('rank_phi_grad_rms', 0.0)),
            float(triple.get('opponent_phi_grad_rms', 0.0)),
            float(triple.get('danger_phi_grad_rms', 0.0)),
            cosine_12=float(triple.get('rank_opponent_phi_grad_cos', 0.0)),
            cosine_13=float(triple.get('rank_danger_phi_grad_cos', 0.0)),
            cosine_23=float(triple.get('opp_danger_phi_grad_cos', 0.0)),
        )

        per_protocol_pairs = {
            'rank_opp': (rank_opp_loss, rank_opp_grad),
            'rank_danger': (rank_danger_loss, rank_danger_grad),
            'opp_danger': (opp_danger_loss, opp_danger_grad),
            'triple': (triple_loss, triple_grad),
        }
        for combo_name, (loss_value, grad_value) in per_protocol_pairs.items():
            protocol_combo_loss[combo_name][protocol_arm] = loss_value
            protocol_combo_grad[combo_name][protocol_arm] = grad_value
            protocol_combo[combo_name][protocol_arm] = blend_combo_factors(
                loss_value=loss_value,
                grad_value=grad_value,
                fallback=1.0,
            )
            combo_loss_values[combo_name].append(loss_value)
            combo_grad_values[combo_name].append(grad_value)

    combo_factor_loss = {
        combo_name: (statistics.median(values) if values else 1.0)
        for combo_name, values in combo_loss_values.items()
    }
    combo_factor_grad = {
        combo_name: (statistics.median(values) if values else 1.0)
        for combo_name, values in combo_grad_values.items()
    }
    combo_factor = {
        combo_name: blend_combo_factors(
            loss_value=combo_factor_loss[combo_name],
            grad_value=combo_factor_grad[combo_name],
            fallback=1.0,
        )
        for combo_name in combo_names
    }
    opp_weight_per_budget_unit_loss = derive_weight_per_budget_unit(
        rank_effective_base=rank_effective_base,
        per_unit_effective=opp_effective_per_unit,
        fallback_weight_per_budget_unit=P1_DEFAULT_OPP_WEIGHT_PER_BUDGET,
    )
    danger_weight_per_budget_unit_loss = derive_weight_per_budget_unit(
        rank_effective_base=rank_effective_base,
        per_unit_effective=danger_effective_per_unit,
        fallback_weight_per_budget_unit=P1_DEFAULT_DANGER_WEIGHT_PER_BUDGET,
    )
    opp_weight_per_budget_unit_grad = derive_weight_per_budget_unit(
        rank_effective_base=rank_grad_effective_base,
        per_unit_effective=opp_grad_effective_per_unit,
        fallback_weight_per_budget_unit=P1_DEFAULT_OPP_WEIGHT_PER_BUDGET,
    )
    danger_weight_per_budget_unit_grad = derive_weight_per_budget_unit(
        rank_effective_base=rank_grad_effective_base,
        per_unit_effective=danger_grad_effective_per_unit,
        fallback_weight_per_budget_unit=P1_DEFAULT_DANGER_WEIGHT_PER_BUDGET,
    )
    opp_weight_per_budget_unit = blend_positive_calibration_values(
        loss_value=opp_weight_per_budget_unit_loss,
        grad_value=opp_weight_per_budget_unit_grad,
        fallback=P1_DEFAULT_OPP_WEIGHT_PER_BUDGET,
    )
    danger_weight_per_budget_unit = blend_positive_calibration_values(
        loss_value=danger_weight_per_budget_unit_loss,
        grad_value=danger_weight_per_budget_unit_grad,
        fallback=P1_DEFAULT_DANGER_WEIGHT_PER_BUDGET,
    )
    if (
        not opp_per_unit
        and 'opp_weight_per_budget_unit_loss' in effective_inherited_single_head
    ):
        opp_weight_per_budget_unit_loss = float(
            effective_inherited_single_head['opp_weight_per_budget_unit_loss']
        )
    if (
        not danger_per_unit
        and 'danger_weight_per_budget_unit_loss' in effective_inherited_single_head
    ):
        danger_weight_per_budget_unit_loss = float(
            effective_inherited_single_head['danger_weight_per_budget_unit_loss']
        )
    if (
        not opp_grad_per_unit
        and 'opp_weight_per_budget_unit_grad' in effective_inherited_single_head
    ):
        opp_weight_per_budget_unit_grad = float(
            effective_inherited_single_head['opp_weight_per_budget_unit_grad']
        )
    if (
        not danger_grad_per_unit
        and 'danger_weight_per_budget_unit_grad' in effective_inherited_single_head
    ):
        danger_weight_per_budget_unit_grad = float(
            effective_inherited_single_head['danger_weight_per_budget_unit_grad']
        )
    if (
        not opp_per_unit
        and not opp_grad_per_unit
        and 'opp_weight_per_budget_unit' in effective_inherited_single_head
    ):
        opp_weight_per_budget_unit = float(
            effective_inherited_single_head['opp_weight_per_budget_unit']
        )
    if (
        not danger_per_unit
        and not danger_grad_per_unit
        and 'danger_weight_per_budget_unit' in effective_inherited_single_head
    ):
        danger_weight_per_budget_unit = float(
            effective_inherited_single_head['danger_weight_per_budget_unit']
        )
    return {
        'budget_ratios': list(P1_EFFECTIVE_BUDGET_RATIOS),
        'mapping_mode': P1_CALIBRATION_MAPPING_MODE,
        'calibration_mode': calibration_mode,
        'calibration_mode_note': p1_calibration_mode_note(calibration_mode),
        'rank_effective_base': rank_effective_base,
        'opp_effective_per_unit': opp_effective_per_unit,
        'danger_effective_per_unit': danger_effective_per_unit,
        'rank_grad_effective_base': rank_grad_effective_base,
        'opp_grad_effective_per_unit': opp_grad_effective_per_unit,
        'danger_grad_effective_per_unit': danger_grad_effective_per_unit,
        'opp_weight_per_budget_unit_loss': opp_weight_per_budget_unit_loss,
        'danger_weight_per_budget_unit_loss': danger_weight_per_budget_unit_loss,
        'opp_weight_per_budget_unit_grad': opp_weight_per_budget_unit_grad,
        'danger_weight_per_budget_unit_grad': danger_weight_per_budget_unit_grad,
        'opp_weight_per_budget_unit': opp_weight_per_budget_unit,
        'danger_weight_per_budget_unit': danger_weight_per_budget_unit,
        'combo_scheme': P1_CALIBRATION_SCHEME,
        'rank_opp_combo_factor': combo_factor['rank_opp'],
        'rank_opp_combo_factor_loss': combo_factor_loss['rank_opp'],
        'rank_opp_combo_factor_grad': combo_factor_grad['rank_opp'],
        'rank_danger_combo_factor': combo_factor['rank_danger'],
        'rank_danger_combo_factor_loss': combo_factor_loss['rank_danger'],
        'rank_danger_combo_factor_grad': combo_factor_grad['rank_danger'],
        'opp_danger_combo_factor': combo_factor['opp_danger'],
        'opp_danger_combo_factor_loss': combo_factor_loss['opp_danger'],
        'opp_danger_combo_factor_grad': combo_factor_grad['opp_danger'],
        'triple_combo_factor': combo_factor['triple'],
        'triple_combo_factor_loss': combo_factor_loss['triple'],
        'triple_combo_factor_grad': combo_factor_grad['triple'],
        'protocol_rank_opp_combo_factors': protocol_combo['rank_opp'],
        'protocol_rank_opp_combo_factors_loss': protocol_combo_loss['rank_opp'],
        'protocol_rank_opp_combo_factors_grad': protocol_combo_grad['rank_opp'],
        'protocol_rank_danger_combo_factors': protocol_combo['rank_danger'],
        'protocol_rank_danger_combo_factors_loss': protocol_combo_loss['rank_danger'],
        'protocol_rank_danger_combo_factors_grad': protocol_combo_grad['rank_danger'],
        'protocol_opp_danger_combo_factors': protocol_combo['opp_danger'],
        'protocol_opp_danger_combo_factors_loss': protocol_combo_loss['opp_danger'],
        'protocol_opp_danger_combo_factors_grad': protocol_combo_grad['opp_danger'],
        'protocol_triple_combo_factors': protocol_combo['triple'],
        'protocol_triple_combo_factors_loss': protocol_combo_loss['triple'],
        'protocol_triple_combo_factors_grad': protocol_combo_grad['triple'],
        # Legacy alias retained for historical helpers; current mainline should
        # read `opp_danger_combo_factor` or `triple_combo_factor` explicitly.
        'joint_combo_factor': combo_factor['opp_danger'],
        'joint_combo_factor_loss': combo_factor_loss['opp_danger'],
        'joint_combo_factor_grad': combo_factor_grad['opp_danger'],
        'protocol_joint_combo_factors': protocol_combo['opp_danger'],
        'protocol_joint_combo_factors_loss': protocol_combo_loss['opp_danger'],
        'protocol_joint_combo_factors_grad': protocol_combo_grad['opp_danger'],
        'probe_weight': P1_CALIBRATION_PROBE_WEIGHT,
        'grad_probe_batches': P1_GRAD_CALIBRATION_MAX_BATCHES,
        'details': details,
        'single_head_probe_reference': inherited_probe_reference,
        'inherited_single_head': use_inherited_single_head,
        'inherited_single_head_source': effective_inherited_single_head_source,
        'fallback_used': (
            rank_effective_base <= 0
            or opp_effective_per_unit <= 0
            or danger_effective_per_unit <= 0
            or rank_grad_effective_base <= 0
            or opp_grad_effective_per_unit <= 0
            or danger_grad_effective_per_unit <= 0
        ),
    }


def build_p1_solo_candidates(
    protocols: list[CandidateSpec],
    calibration: dict[str, Any],
) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for protocol in protocols:
        candidates.append(
            make_p1_budget_candidate(
                protocol,
                calibration=calibration,
                rank_budget_ratio=0.0,
                opp_budget_ratio=0.0,
                danger_budget_ratio=0.0,
                stage='P1_solo_round',
                family='ce_only',
            )
        )
        for budget_ratio in P1_SOLO_RANK_BUDGET_RATIOS:
            candidates.append(
                make_p1_budget_candidate(
                    protocol,
                    calibration=calibration,
                    rank_budget_ratio=budget_ratio,
                    opp_budget_ratio=0.0,
                    danger_budget_ratio=0.0,
                    stage='P1_solo_round',
                    family='rank',
                )
            )
        for budget_ratio in P1_SOLO_OPP_BUDGET_RATIOS:
            candidates.append(
                make_p1_budget_candidate(
                    protocol,
                    calibration=calibration,
                    rank_budget_ratio=0.0,
                    opp_budget_ratio=budget_ratio,
                    danger_budget_ratio=0.0,
                    stage='P1_solo_round',
                    family='opp',
                )
            )
        for budget_ratio in P1_SOLO_DANGER_BUDGET_RATIOS:
            candidates.append(
                make_p1_budget_candidate(
                    protocol,
                    calibration=calibration,
                    rank_budget_ratio=0.0,
                    opp_budget_ratio=0.0,
                    danger_budget_ratio=budget_ratio,
                    stage='P1_solo_round',
                    family='danger',
                )
            )
    return unique_candidates(candidates)


def build_p1_protocol_decide_candidates(
    protocols: list[CandidateSpec],
    calibration: dict[str, Any],
) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for protocol in protocols:
        candidates.append(
            make_p1_budget_candidate(
                protocol,
                calibration=calibration,
                rank_budget_ratio=0.0,
                opp_budget_ratio=0.0,
                danger_budget_ratio=0.0,
                stage='P1_protocol_decide_round',
                family='ce_only',
            )
        )
        for total_budget_ratio in P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS:
            for mix_name, rank_share, opp_share, danger_share in P1_PROTOCOL_DECIDE_MIXES:
                candidates.append(
                    make_p1_triplet_candidate(
                        protocol,
                        calibration=calibration,
                        total_budget_ratio=total_budget_ratio,
                        rank_share=rank_share,
                        opp_share=opp_share,
                        danger_share=danger_share,
                        stage='P1_protocol_decide_round',
                        family='all_three',
                        mix_name=mix_name,
                    )
                )
    return unique_candidates(candidates)


def build_p1_winner_refine_points(
    *,
    rank_budget_ratio: float,
    opp_budget_ratio: float,
    danger_budget_ratio: float,
) -> list[tuple[float, float, float]]:
    center = (
        clamp_budget_ratio(rank_budget_ratio),
        clamp_budget_ratio(opp_budget_ratio),
        clamp_budget_ratio(danger_budget_ratio),
    )
    points: set[tuple[float, float, float]] = {center}
    for scale in P1_WINNER_REFINE_TOTAL_SCALE_FACTORS:
        scaled = tuple(clamp_budget_ratio(value * scale) for value in center)
        if min(scaled) > 0:
            points.add(scaled)
    transfer = P1_WINNER_REFINE_TRANSFER_DELTA
    transfers = (
        (transfer, -transfer, 0.0),
        (-transfer, transfer, 0.0),
        (transfer, 0.0, -transfer),
        (-transfer, 0.0, transfer),
        (0.0, transfer, -transfer),
        (0.0, -transfer, transfer),
    )
    for delta_rank, delta_opp, delta_danger in transfers:
        candidate = (
            clamp_budget_ratio(center[0] + delta_rank),
            clamp_budget_ratio(center[1] + delta_opp),
            clamp_budget_ratio(center[2] + delta_danger),
        )
        if min(candidate) <= 0:
            continue
        points.add(candidate)
    return sorted(points)


def build_p1_winner_refine_candidates(
    protocols: list[CandidateSpec],
    calibration: dict[str, Any],
    centers: list[CandidateSpec],
) -> list[CandidateSpec]:
    protocol_index = {
        str(protocol.meta.get('protocol_arm', protocol.arm_name)): protocol
        for protocol in protocols
    }
    candidates: list[CandidateSpec] = []
    usable_centers = [
        center for center in centers if is_p1_winner_refine_center_meta(center.meta)
    ]
    for center in usable_centers:
        protocol_arm = str(center.meta.get('protocol_arm', center.arm_name))
        protocol = protocol_index[protocol_arm]
        center_rank = float(center.meta.get('rank_budget_ratio', 0.0))
        center_opp = float(center.meta.get('opp_budget_ratio', 0.0))
        center_danger = float(center.meta.get('danger_budget_ratio', 0.0))
        center_name = center.arm_name
        for rank_point, opp_point, danger_point in build_p1_winner_refine_points(
            rank_budget_ratio=center_rank,
            opp_budget_ratio=center_opp,
            danger_budget_ratio=center_danger,
        ):
            candidates.append(
                make_p1_budget_candidate(
                    protocol,
                    calibration=calibration,
                    rank_budget_ratio=rank_point,
                    opp_budget_ratio=opp_point,
                    danger_budget_ratio=danger_point,
                    stage='P1_winner_refine_round',
                    family='all_three',
                    source_arm=center_name,
                )
            )
    if not candidates:
        raise RuntimeError('p1_winner_refine_round requires at least one all_three center')
    return unique_candidates(candidates)


def build_p1_ablation_candidates(
    protocols: list[CandidateSpec],
    calibration: dict[str, Any],
    winner: CandidateSpec,
) -> list[CandidateSpec]:
    protocol_index = {
        str(protocol.meta.get('protocol_arm', protocol.arm_name)): protocol
        for protocol in protocols
    }
    protocol_arm = str(winner.meta.get('protocol_arm', winner.arm_name))
    protocol = protocol_index[protocol_arm]
    rank_budget_ratio = float(winner.meta.get('rank_budget_ratio', 0.0))
    opp_budget_ratio = float(winner.meta.get('opp_budget_ratio', 0.0))
    danger_budget_ratio = float(winner.meta.get('danger_budget_ratio', 0.0))
    winner_name = winner.arm_name
    candidates = [
        make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=0.0,
            opp_budget_ratio=0.0,
            danger_budget_ratio=0.0,
            stage='P1_ablation_round',
            family='ce_only',
            source_arm=winner_name,
        ),
        make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=rank_budget_ratio,
            opp_budget_ratio=opp_budget_ratio,
            danger_budget_ratio=danger_budget_ratio,
            stage='P1_ablation_round',
            family='all_three',
            source_arm=winner_name,
        ),
        make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=0.0,
            opp_budget_ratio=opp_budget_ratio,
            danger_budget_ratio=danger_budget_ratio,
            stage='P1_ablation_round',
            family='drop_rank',
            source_arm=winner_name,
        ),
        make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=rank_budget_ratio,
            opp_budget_ratio=0.0,
            danger_budget_ratio=danger_budget_ratio,
            stage='P1_ablation_round',
            family='drop_opp',
            source_arm=winner_name,
        ),
        make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=rank_budget_ratio,
            opp_budget_ratio=opp_budget_ratio,
            danger_budget_ratio=0.0,
            stage='P1_ablation_round',
            family='drop_danger',
            source_arm=winner_name,
        ),
    ]
    return unique_candidates(candidates)


def select_p1_family_survivors(
    ranking: list[dict[str, Any]],
    protocols: list[CandidateSpec],
    *,
    ranking_mode: str = 'full_recent',
) -> dict[str, dict[str, CandidateSpec]]:
    # Legacy pre-2026-03-28 helper kept only for historical result replay.
    # Mainline P1 no longer uses family survivors as the protocol gate.
    survivors: dict[str, dict[str, CandidateSpec]] = {}
    for protocol in protocols:
        protocol_arm = str(protocol.meta.get('protocol_arm', protocol.arm_name))
        protocol_entries = [
            entry
            for entry in ranking
            if str(entry.get('candidate_meta', {}).get('protocol_arm', '')) == protocol_arm
        ]
        protocol_survivors: dict[str, CandidateSpec] = {}
        for family in ('ce_only', 'rank', 'opp', 'danger'):
            family_entries = [
                entry
                for entry in protocol_entries
                if str(entry.get('candidate_meta', {}).get('aux_family', '')) == family
            ]
            if not family_entries:
                continue
            if family == 'ce_only':
                protocol_survivors[family] = candidate_from_entry(family_entries[0])
                continue
            family_winner = best_family_entry(
                family_entries,
                ranking_mode=ranking_mode,
            )
            if family_winner is None:
                continue
            ce_only_entry = next(
                (
                    entry
                    for entry in protocol_entries
                    if str(entry.get('candidate_meta', {}).get('aux_family', '')) == 'ce_only'
                    and entry.get('valid')
                ),
                None,
            )
            epsilon = P1_POLICY_LOSS_EPSILON if ranking_mode == 'policy_quality' else LOSS_EPSILON
            family_recent_loss = recent_ranking_loss_for_entry(
                family_winner,
                ranking_mode=ranking_mode,
            )
            ce_recent_loss = None if ce_only_entry is None else recent_ranking_loss_for_entry(
                ce_only_entry,
                ranking_mode=ranking_mode,
            )
            if (
                ce_only_entry is not None
                and family_winner.get('valid')
                and (
                    not math.isfinite(family_recent_loss)
                    or ce_recent_loss is None
                    or not math.isfinite(ce_recent_loss)
                    or family_recent_loss > ce_recent_loss + epsilon
                )
            ):
                continue
            if ce_only_entry is not None and family_winner.get('valid') and ranking_mode == 'policy_quality':
                ce_old_loss = old_regression_ranking_loss_for_entry(
                    ce_only_entry,
                    ranking_mode=ranking_mode,
                )
                family_old_loss = old_regression_ranking_loss_for_entry(
                    family_winner,
                    ranking_mode=ranking_mode,
                )
                if ce_old_loss is not None and math.isfinite(ce_old_loss):
                    if (
                        family_old_loss is None
                        or not math.isfinite(family_old_loss)
                        or family_old_loss > ce_old_loss + P1_OLD_REGRESSION_POLICY_EPSILON
                    ):
                        continue
            protocol_survivors[family] = candidate_from_entry(family_winner)
        survivors[protocol_arm] = protocol_survivors
    return survivors


def build_p1_pairwise_candidates(
    protocols: list[CandidateSpec],
    calibration: dict[str, Any],
    solo_survivors: dict[str, dict[str, CandidateSpec]],
) -> list[CandidateSpec]:
    # Legacy pre-2026-03-28 helper kept only for historical result replay.
    # Mainline P1 no longer builds protocol candidates by directly summing single-head winners.
    protocol_index = {
        str(protocol.meta.get('protocol_arm', protocol.arm_name)): protocol
        for protocol in protocols
    }
    candidates: list[CandidateSpec] = []
    for protocol_arm, family_winners in solo_survivors.items():
        protocol = protocol_index[protocol_arm]
        ce_only = family_winners.get('ce_only')
        if ce_only is not None:
            candidates.append(ce_only)
        for family in ('rank', 'opp', 'danger'):
            candidate = family_winners.get(family)
            if candidate is not None:
                candidates.append(candidate)
        for left_family, right_family in (('rank', 'opp'), ('rank', 'danger'), ('opp', 'danger')):
            left = family_winners.get(left_family)
            right = family_winners.get(right_family)
            if left is None or right is None:
                continue
            candidates.append(
                make_p1_budget_candidate(
                    protocol,
                    calibration=calibration,
                    rank_budget_ratio=float(left.meta.get('rank_budget_ratio', 0.0) or right.meta.get('rank_budget_ratio', 0.0)),
                    opp_budget_ratio=float(left.meta.get('opp_budget_ratio', 0.0) or right.meta.get('opp_budget_ratio', 0.0)),
                    danger_budget_ratio=float(left.meta.get('danger_budget_ratio', 0.0) or right.meta.get('danger_budget_ratio', 0.0)),
                    stage='P1_pairwise_round',
                    family=f'{left_family}+{right_family}',
                    source_arm=f'{left.arm_name}|{right.arm_name}',
                )
            )
    return unique_candidates(candidates)


def build_p1_joint_refine_candidates(
    protocols: list[CandidateSpec],
    calibration: dict[str, Any],
    center_candidates: list[CandidateSpec],
) -> list[CandidateSpec]:
    # Legacy pre-2026-03-28 helper kept only for historical result replay.
    protocol_index = {
        str(protocol.meta.get('protocol_arm', protocol.arm_name)): protocol
        for protocol in protocols
    }
    candidates: list[CandidateSpec] = []
    for center in center_candidates:
        protocol_arm = str(center.meta.get('protocol_arm', center.arm_name))
        protocol = protocol_index[protocol_arm]
        rank_budget_ratio = float(center.meta.get('rank_budget_ratio', 0.0))
        opp_budget_ratio = float(center.meta.get('opp_budget_ratio', 0.0))
        danger_budget_ratio = float(center.meta.get('danger_budget_ratio', 0.0))
        refine_points = {
            (rank_budget_ratio, opp_budget_ratio, danger_budget_ratio),
            (rank_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA, opp_budget_ratio, danger_budget_ratio),
            (rank_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA, opp_budget_ratio, danger_budget_ratio),
            (rank_budget_ratio, opp_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA, danger_budget_ratio),
            (rank_budget_ratio, opp_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA, danger_budget_ratio),
            (rank_budget_ratio, opp_budget_ratio, danger_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA),
            (rank_budget_ratio, opp_budget_ratio, danger_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA),
            (
                rank_budget_ratio,
                opp_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA,
                danger_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA,
            ),
            (
                rank_budget_ratio,
                opp_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA,
                danger_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA,
            ),
            (
                rank_budget_ratio,
                opp_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA,
                danger_budget_ratio - P1_JOINT_REFINE_BUDGET_DELTA,
            ),
            (
                rank_budget_ratio,
                opp_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA,
                danger_budget_ratio + P1_JOINT_REFINE_BUDGET_DELTA,
            ),
        }
        for rank_point, opp_point, danger_point in refine_points:
            candidates.append(
                make_p1_budget_candidate(
                    protocol,
                    calibration=calibration,
                    rank_budget_ratio=rank_point,
                    opp_budget_ratio=opp_point,
                    danger_budget_ratio=danger_point,
                    stage='P1_joint_refine_round',
                    source_arm=center.arm_name,
                )
            )
    return unique_candidates(candidates)


def entry_table(
    entries: list[dict[str, Any]],
    limit: int = 8,
    *,
    ranking_mode: str = 'full_recent',
) -> list[str]:
    comparison_label = 'cmp_policy' if ranking_mode == 'policy_quality' else 'cmp_loss'
    full_loss_label = 'full_loss(diag)' if ranking_mode == 'policy_quality' else 'full_loss'
    lines = [
        f'| rank | arm | {comparison_label} | {full_loss_label} | selection | action | scenario | rank_acc | eligible |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for entry in entries[:limit]:
        comparison_loss_value = float(entry.get('comparison_recent_loss', entry.get('recent_policy_loss', entry['full_recent_loss'])))
        comparison_loss = 'NA' if not math.isfinite(comparison_loss_value) else f'{comparison_loss_value:.4f}'
        loss = 'NA' if not math.isfinite(entry['full_recent_loss']) else f'{entry["full_recent_loss"]:.4f}'
        selection_score = 'NA' if not math.isfinite(entry.get('selection_quality_score', float("-inf"))) else f'{entry["selection_quality_score"]:.4f}'
        action_score = 'NA' if not math.isfinite(entry['action_quality_score']) else f'{entry["action_quality_score"]:.4f}'
        scenario_score = 'NA' if not math.isfinite(entry['scenario_quality_score']) else f'{entry["scenario_quality_score"]:.4f}'
        rank_acc = 'NA' if entry['rank_acc'] < 0 else f'{entry["rank_acc"]:.4f}'
        lines.append(
            f'| {entry["rank"]} | `{entry["arm_name"]}` | {comparison_loss} | {loss} | {selection_score} | {action_score} | {scenario_score} | {rank_acc} | {entry["eligible"]} |'
        )
    return lines


def format_small_threshold(value: float) -> str:
    return f'{float(value):g}'


def current_protocol_decide_mix_payload() -> list[dict[str, float | str]]:
    return [
        {
            'name': name,
            'rank_share': rank_share,
            'opp_share': opp_share,
            'danger_share': danger_share,
        }
        for name, rank_share, opp_share, danger_share in P1_PROTOCOL_DECIDE_MIXES
    ]


def p1_snapshot_uses_current_defaults(p1: dict[str, Any]) -> bool:
    search_space = p1.get('search_space') or {}
    if not search_space:
        return True
    if list(search_space.get('calibration_protocol_arms', list(P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS))) != list(
        P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS
    ):
        return False
    if list(search_space.get('protocol_decide_total_budget_ratios', [])) != list(
        P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS
    ):
        return False
    if list(search_space.get('protocol_decide_mixes', [])) != current_protocol_decide_mix_payload():
        return False
    if str(search_space.get('calibration_mode', P1_CALIBRATION_DEFAULT_MODE)) != P1_CALIBRATION_DEFAULT_MODE:
        return False
    if str(
        search_space.get('inherited_single_head_source', P1_SINGLE_HEAD_CALIBRATION_SOURCE)
    ) != str(P1_SINGLE_HEAD_CALIBRATION_SOURCE):
        return False
    selection_policy = search_space.get('selection_policy') or {}
    if float(selection_policy.get('policy_loss_epsilon', P1_POLICY_LOSS_EPSILON)) != float(
        P1_POLICY_LOSS_EPSILON
    ):
        return False
    if float(
        selection_policy.get(
            'old_regression_policy_loss_epsilon',
            P1_OLD_REGRESSION_POLICY_EPSILON,
        )
    ) != float(P1_OLD_REGRESSION_POLICY_EPSILON):
        return False
    return True


def update_results_doc(run_dir: Path, state: dict[str, Any]) -> None:
    p1 = state.get('p1') or {}
    historical_snapshot = not p1_snapshot_uses_current_defaults(p1)
    title = '# Stage 0.5 保真版 A/B 实时结果'
    if historical_snapshot:
        title += ' (historical snapshot)'
    lines = [
        '# Stage 0.5 保真版 A/B 实时结果',
        '',
        f'- 运行目录：`{run_dir}`',
        f'- 更新时间：`{ts_now()}`',
        f'- 当前状态：`{state.get("status", "unknown")}`',
        '- 自动串联范围：`P0 + P1 + P2 + Stage 0.5 formal`',
        '- 说明：`P3(Stage 1 transfer)` 不作为本轮 `Stage 0.5 formal` 启动前置条件，避免把 Stage 1 下游转移实验混入 0.5 阶段主协议定型。',
        '',
    ]

    lines[0] = title
    if historical_snapshot:
        lines.extend([
            '> historical snapshot: this file records one run output and may not match the current default search space.',
            '> For current defaults, prefer `docs/agent/current-plan.md`, `docs/agent/mainline.md`, `docs/status/stage05-verified-status.md`, and `docs/status/p1-selection-canonical.md`.',
            '',
        ])
    else:
        lines.extend([
            '> run-scoped snapshot: if this file conflicts with the verified/canonical status docs, prefer those docs for current defaults.',
            '',
        ])

    final = state.get('final_conclusion') or {}
    if final:
        lines.extend([
            '## 当前结论',
            '',
            f'- P0 下游种子 top4：`{", ".join(final.get("p0_stage1_top4", []))}`',
            f'- P0 round3 winner：`{final.get("p0_winner", "TBD")}`',
            f'- P1 总胜者：`{final.get("p1_winner", "TBD")}`',
            f'- P2 默认 checkpoint：`{final.get("p2_default_checkpoint", "TBD")}`',
            f'- 正式训练：`{final.get("formal_status", "pending")}`',
            '',
        ])

    p0 = state.get('p0')
    if p0:
        lines.extend(['## P0', ''])
        for key in ('round0', 'round1', 'round2', 'round3'):
            payload = p0.get(key)
            if payload:
                lines.append(f'### {key}')
                lines.append('')
                lines.extend(entry_table(payload['ranking'], ranking_mode=str(payload.get('ranking_mode') or 'full_recent')))
                lines.append('')

    p1 = state.get('p1')
    if p1:
        lines.extend(['## P1', ''])
        policy = p1.get('selection_policy') or p1_selection_policy_metadata()
        lines.extend([
            '### canonical_selection',
            '',
            f"- `selector = {policy.get('canonical_selector', P1_RANKING_MODE)}`",
            f"- `适用范围 = {' / '.join(policy.get('applies_to', []))}`",
            (
                f"- `比较字段 = {policy.get('comparison_alias', 'comparison_recent_loss')} = "
                f"{policy.get('comparison_metric', 'recent_policy_loss')}`"
            ),
            (
                f"- `eligible 分组 = {policy.get('eligibility_group_key', P1_PROTOCOL_ELIGIBILITY_GROUP_KEY)}`；"
                f"`recent_policy_loss <= group_best + {format_small_threshold(policy.get('policy_loss_epsilon', P1_POLICY_LOSS_EPSILON))}`"
            ),
            (
                '- `old_regression` 可用时，再要求 '
                f"`old_regression_policy_loss <= group_best_old + "
                f"{format_small_threshold(policy.get('old_regression_policy_loss_epsilon', P1_OLD_REGRESSION_POLICY_EPSILON))}`"
            ),
            (
                f"- `selection_quality_score = action_quality_score + "
                f"{policy.get('selection_scenario_factor', SELECTION_SCENARIO_FACTOR):.3f} * "
                'scenario_quality_score`'
            ),
            '- `主排序 = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`',
            (
                f"- `family guardrail = {policy.get('ce_only_guardrail', 'family survivor must not lose clearly to ce_only')}`"
            ),
            (
                f"- `ce_only anchor = {policy.get('ce_only_anchor_note', 'ce_only is a diagnostic anchor only')}`"
            ),
            (
                f"- `full_recent_loss = {policy.get('diagnostic_metric', 'full_recent_loss')}` 仅作 aux tax / 总 loss 诊断，"
                '不是 `P1` family winner 的主判胜字段'
            ),
            (
                f"- `calibration note = {policy.get('calibration_note', 'p1_calibration is mapping-only')}`"
            ),
            '',
        ])
        calibration = p1.get('calibration')
        if calibration:
            lines.append('### calibration')
            lines.append('')
            lines.append(f"- `mapping_mode = {calibration.get('mapping_mode', P1_CALIBRATION_MAPPING_MODE)}`")
            lines.append(f"- `combo_scheme = {calibration.get('combo_scheme', P1_CALIBRATION_SCHEME)}`")
            lines.append(
                f"- `calibration_mode_note = {calibration.get('calibration_mode_note', p1_calibration_mode_note(str(calibration.get('calibration_mode', P1_CALIBRATION_DEFAULT_MODE))))}`"
            )
            lines.append(f"- `inherited_single_head = {calibration.get('inherited_single_head', False)}`")
            lines.append(
                f"- `inherited_single_head_source = {calibration.get('inherited_single_head_source', 'NA')}`"
            )
            lines.append(f"- `rank_effective_base = {calibration.get('rank_effective_base', 0.0):.6f}`")
            lines.append(f"- `opp_effective_per_unit = {calibration.get('opp_effective_per_unit', 0.0):.6f}`")
            lines.append(f"- `danger_effective_per_unit = {calibration.get('danger_effective_per_unit', 0.0):.6f}`")
            lines.append(f"- `rank_grad_effective_base = {calibration.get('rank_grad_effective_base', 0.0):.6f}`")
            lines.append(f"- `opp_grad_effective_per_unit = {calibration.get('opp_grad_effective_per_unit', 0.0):.6f}`")
            lines.append(f"- `danger_grad_effective_per_unit = {calibration.get('danger_grad_effective_per_unit', 0.0):.6f}`")
            single_head_probe_reference = calibration.get('single_head_probe_reference') or {}
            if single_head_probe_reference:
                lines.append(
                    f"- `single_head_probe_reference.probe_weight = {single_head_probe_reference.get('probe_weight', 0.0):.6f}`"
                )
                lines.append(
                    f"- `single_head_probe_reference.rank_effective = {single_head_probe_reference.get('rank_effective', 0.0):.6f}`"
                )
                lines.append(
                    f"- `single_head_probe_reference.opp_effective = {single_head_probe_reference.get('opp_effective', 0.0):.6f}`"
                )
                lines.append(
                    f"- `single_head_probe_reference.danger_effective = {single_head_probe_reference.get('danger_effective', 0.0):.6f}`"
                )
                lines.append(
                    f"- `single_head_probe_reference.rank_phi_grad_rms = {single_head_probe_reference.get('rank_phi_grad_rms', 0.0):.6f}`"
                )
                lines.append(
                    f"- `single_head_probe_reference.opponent_phi_grad_rms = {single_head_probe_reference.get('opponent_phi_grad_rms', 0.0):.6f}`"
                )
                lines.append(
                    f"- `single_head_probe_reference.danger_phi_grad_rms = {single_head_probe_reference.get('danger_phi_grad_rms', 0.0):.6f}`"
                )
            lines.append(f"- `budget_ratios = {calibration.get('budget_ratios', [])}`")
            lines.append(f"- `opp_weight_per_budget_unit_loss = {calibration.get('opp_weight_per_budget_unit_loss', 0.0):.6f}`")
            lines.append(f"- `danger_weight_per_budget_unit_loss = {calibration.get('danger_weight_per_budget_unit_loss', 0.0):.6f}`")
            lines.append(f"- `opp_weight_per_budget_unit_grad = {calibration.get('opp_weight_per_budget_unit_grad', 0.0):.6f}`")
            lines.append(f"- `danger_weight_per_budget_unit_grad = {calibration.get('danger_weight_per_budget_unit_grad', 0.0):.6f}`")
            lines.append(f"- `opp_weight_per_budget_unit = {calibration.get('opp_weight_per_budget_unit', 0.0):.6f}`")
            lines.append(f"- `danger_weight_per_budget_unit = {calibration.get('danger_weight_per_budget_unit', 0.0):.6f}`")
            lines.append(f"- `rank_opp_combo_factor = {calibration.get('rank_opp_combo_factor', 1.0):.6f}`")
            lines.append(f"- `rank_opp_combo_factor_loss = {calibration.get('rank_opp_combo_factor_loss', 1.0):.6f}`")
            lines.append(f"- `rank_opp_combo_factor_grad = {calibration.get('rank_opp_combo_factor_grad', 1.0):.6f}`")
            lines.append(f"- `rank_danger_combo_factor = {calibration.get('rank_danger_combo_factor', 1.0):.6f}`")
            lines.append(f"- `rank_danger_combo_factor_loss = {calibration.get('rank_danger_combo_factor_loss', 1.0):.6f}`")
            lines.append(f"- `rank_danger_combo_factor_grad = {calibration.get('rank_danger_combo_factor_grad', 1.0):.6f}`")
            lines.append(f"- `opp_danger_combo_factor = {calibration.get('opp_danger_combo_factor', calibration.get('joint_combo_factor', 1.0)):.6f}`")
            lines.append(f"- `opp_danger_combo_factor_loss = {calibration.get('opp_danger_combo_factor_loss', calibration.get('joint_combo_factor_loss', 1.0)):.6f}`")
            lines.append(f"- `opp_danger_combo_factor_grad = {calibration.get('opp_danger_combo_factor_grad', calibration.get('joint_combo_factor_grad', 1.0)):.6f}`")
            lines.append(f"- `triple_combo_factor = {calibration.get('triple_combo_factor', 1.0):.6f}`")
            lines.append(f"- `triple_combo_factor_loss = {calibration.get('triple_combo_factor_loss', 1.0):.6f}`")
            lines.append(f"- `triple_combo_factor_grad = {calibration.get('triple_combo_factor_grad', 1.0):.6f}`")
            lines.append(f"- `joint_combo_factor(legacy_opp_danger_alias) = {calibration.get('joint_combo_factor', 1.0):.6f}`")
            lines.append(f"- `grad_probe_batches = {calibration.get('grad_probe_batches', 0)}`")
            lines.append(f"- `fallback_used = {calibration.get('fallback_used', False)}`")
            lines.append('')
        search_space = p1.get('search_space')
        if search_space:
            lines.append('### search_space')
            lines.append('')
            for key in (
                'calibration_mode',
                'calibration_mode_note',
                'combo_scheme',
                'inherited_single_head_source',
                'protocol_decide_total_budget_ratios',
                'protocol_decide_mixes',
                'rank_opp_combo_factor',
                'rank_danger_combo_factor',
                'opp_danger_combo_factor',
                'triple_combo_factor',
                'protocol_decide_probe_keep_per_protocol',
                'winner_refine_total_scale_factors',
                'winner_refine_transfer_delta',
            ):
                if key in search_space:
                    lines.append(f"- `{key} = {search_space[key]}`")
            lines.append('')
        for subkey in (
            'calibration_round',
            'solo_round',
            'pairwise_round',
            'joint_refine_round',
            'protocol_decide_round',
            'winner_refine_round',
            'ablation_round',
        ):
            payload = p1.get(subkey)
            if payload:
                ranking_mode = str(payload.get('ranking_mode') or 'full_recent')
                lines.append(f'### {subkey}')
                lines.append('')
                if payload.get('seed_offsets') is not None:
                    lines.append(f"- `actual_seeds = {payload.get('actual_seeds', [])}`")
                    lines.append(f"- `ranking_mode = {ranking_mode}`")
                    if payload.get('seed_strategy'):
                        lines.append(f"- `seed_strategy = {payload['seed_strategy']}`")
                    if payload.get('probe_selector_name'):
                        lines.append(f"- `probe_selector = {payload['probe_selector_name']}`")
                    if payload.get('probe_candidate_count') is not None:
                        lines.append(f"- `probe_candidate_count = {payload['probe_candidate_count']}`")
                    if payload.get('decision_candidate_count') is not None:
                        lines.append(f"- `decision_candidate_count = {payload['decision_candidate_count']}`")
                    if payload.get('expanded_groups'):
                        lines.append(f"- `expanded_groups = {payload.get('expanded_groups', [])}`")
                    lines.append('')
                if ranking_mode == 'policy_quality':
                    lines.append('- `cmp_policy = comparison_recent_loss = recent_policy_loss`')
                    lines.append('- `full_loss(diag) = full_recent_loss`')
                    lines.append('')
                elif subkey == 'calibration_round':
                    lines.append('- `说明 = calibration 只负责 budget mapping；family winner / survivor 不从这张表直接宣布`')
                    lines.append('')
                lines.extend(entry_table(payload['ranking'], ranking_mode=ranking_mode))
                lines.append('')
                screening_diagnostic = payload.get('screening_diagnostic')
                if screening_diagnostic and screening_diagnostic.get('ranking'):
                    lines.append(
                        f"#### screening_diagnostic: {screening_diagnostic.get('name', 'diagnostic')}"
                    )
                    lines.append('')
                    lines.append(f"- `source_seed = {screening_diagnostic.get('source_seed', 'NA')}`")
                    lines.append(
                        f"- `source_round = {screening_diagnostic.get('source_round_name', 'NA')}`"
                    )
                    lines.append('')
                    lines.extend(
                        entry_table(
                            screening_diagnostic['ranking'],
                            ranking_mode=ranking_mode,
                        )
                    )
                    lines.append('')
        for compare_key in ('protocol_compare', 'final_compare'):
            final_round = p1.get(compare_key)
            if not final_round:
                continue
            lines.append(f'### {compare_key}')
            lines.append('')
            lines.extend(entry_table(final_round['ranking'], ranking_mode=P1_RANKING_MODE))
            lines.append('')

    p2 = state.get('p2')
    if p2:
        lines.extend(['## P2', ''])
        for curve in p2.get('curves', []):
            lines.append(f"- `{curve['run_name']}`")
            lines.append(f"  - 默认 selector：`{curve['default_selector_winner']}`")
            lines.append(f"  - 去重后候选：`{', '.join(curve['unique_checkpoint_types'])}`")
        if p2.get('selected_checkpoints'):
            lines.append(f"- P2 汇总候选：`{', '.join(item['label'] for item in p2['selected_checkpoints'])}`")
        lines.append('')

    formal_state = state.get('formal')
    if formal_state:
        lines.extend(['## Formal', '', f"- 状态：`{formal_state.get('status', 'pending')}`"])
        if formal_state.get('ab_name'):
            lines.append(f"- 日志目录：`logs/stage05_ab/{formal_state['ab_name']}`")
        if formal_state.get('winner'):
            lines.append(f"- checkpoint winner：`{formal_state['winner']}`")
        lines.append('')

    lines.extend([
        '## 路径',
        '',
        f'- 状态文件：`{run_dir / "state.json"}`',
        f'- 文档文件：`{RESULTS_DOC_PATH}`',
        '',
    ])
    atomic_write_text(RESULTS_DOC_PATH, '\n'.join(lines))


def find_arm_result_path(run_dir: Path, candidate: CandidateSpec) -> Path:
    representative_result_path = candidate.meta.get('representative_result_path')
    if representative_result_path:
        path = Path(representative_result_path)
        if path.exists():
            return path
    for stage_name in (
        'p1_ablation',
        'p1_winner_refine',
        'p1_protocol_decide',
        'p1_joint_refine',
        'p1_pairwise',
        'p1_solo',
        'p1_joint_r2',
        'p1_joint_r1',
        'p1_joint_r0',
        'p0_r3',
        'p0_r2',
        'p0_r1',
        'p0_r0',
    ):
        path = ab.AB_ROOT / f'{run_dir.name}_{stage_name}' / candidate.arm_name / 'arm_result.json'
        if path.exists():
            return path
    raise FileNotFoundError(f'arm result not found for {candidate.arm_name}')


def load_cached_p0_rounds_upto_round2(
    run_dir: Path,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    seed: int,
    *,
    candidate_subset: str = 'all',
    skip_round1: bool = False,
    round2_max_candidates: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    scheduler_profiles = {'cosine'} if candidate_subset == 'cosine_only' else None
    all_candidates = build_p0_candidates(scheduler_profiles=scheduler_profiles)
    all_index = {candidate.arm_name: candidate for candidate in all_candidates}
    authoritative_revalidated = None if skip_round1 else load_authoritative_revalidated_p0_rounds(
        run_dir,
        base_cfg=base_cfg,
        grouped=grouped,
        seed=seed,
        all_candidates=all_candidates,
        all_index=all_index,
        candidate_subset=candidate_subset,
        round2_max_candidates=round2_max_candidates,
    )
    if authoritative_revalidated is not None:
        return authoritative_revalidated
    eval_file_count = ab.BASE_SCREENING['eval_files']
    round0_eval_splits = ab.build_eval_splits(grouped, seed + 11, eval_file_count)
    round0_signature = round_cache_signature(
        round_name='p0_round0',
        ab_name=f'{run_dir.name}_p0_r0',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=round0_eval_splits,
        candidates=all_candidates,
        seed=seed + 101,
        step_scale=0.5,
        selector_weights=None,
    )

    round0 = load_cached_round_if_valid(
        run_dir / 'p0_round0.json',
        round0_signature,
        legacy_matcher=lambda payload: legacy_round_payload_matches(
            payload,
            round_name='p0_round0',
            ab_name=f'{run_dir.name}_p0_r0',
            expected_candidates=all_candidates,
            seed=seed + 101,
            step_scale=0.5,
            eval_splits=round0_eval_splits,
        ),
    )
    if round0 is None:
        round0 = load_revalidated_round_if_valid(
            run_dir=run_dir,
            round_name='p0_round0',
            ab_name=f'{run_dir.name}_p0_r0',
            ab_root_name=f'{run_dir.name}_p0_r0',
            expected_signature=round0_signature,
            expected_candidates=all_candidates,
            seed=seed + 101,
            step_scale=0.5,
            eval_splits=round0_eval_splits,
        )
    if round0 is None:
        return None
    round0_ranking = round0.get('ranking') or []
    if not round0_ranking:
        return None
    top18 = build_p0_round0_survivors(
        round0_ranking,
        all_index,
        candidate_subset=candidate_subset,
    )
    if not top18:
        return None
    if skip_round1:
        round1 = skipped_round_payload('p0_round1', reason='direct_round2_top8')
        round2_candidates = build_p0_round1_survivors(
            round0_ranking,
            max_keep=round2_max_candidates,
        )
    else:
        round1_eval_splits = ab.build_eval_splits(grouped, seed + 22, eval_file_count)
        round1_signature = round_cache_signature(
            round_name='p0_round1',
            ab_name=f'{run_dir.name}_p0_r1',
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=round1_eval_splits,
            candidates=top18,
            seed=seed + 202,
            step_scale=1.0,
            selector_weights=None,
        )

        round1 = load_cached_round_if_valid(
            run_dir / 'p0_round1.json',
            round1_signature,
            legacy_matcher=lambda payload: legacy_round_payload_matches(
                payload,
                round_name='p0_round1',
                ab_name=f'{run_dir.name}_p0_r1',
                expected_candidates=top18,
                seed=seed + 202,
                step_scale=1.0,
                eval_splits=round1_eval_splits,
            ),
        )
        if round1 is None:
            round1 = load_revalidated_round_if_valid(
                run_dir=run_dir,
                round_name='p0_round1',
                ab_name=f'{run_dir.name}_p0_r1',
                ab_root_name=f'{run_dir.name}_p0_r1',
                expected_signature=round1_signature,
                expected_candidates=top18,
                seed=seed + 202,
                step_scale=1.0,
                eval_splits=round1_eval_splits,
            )
        if round1 is None:
            return None
        round1_ranking = round1.get('ranking') or []
        if not round1_ranking:
            return None
        round2_candidates = build_p0_round1_survivors(
            round1_ranking,
            max_keep=round2_max_candidates,
        )
    if not round2_candidates:
        return None
    round2_eval_splits = ab.build_eval_splits(grouped, seed + 33, eval_file_count)
    round2_signature = round_cache_signature(
        round_name='p0_round2',
        ab_name=f'{run_dir.name}_p0_r2',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=round2_eval_splits,
        candidates=round2_candidates,
        seed=seed + 303,
        step_scale=2.0,
        selector_weights=None,
    )

    round2 = load_cached_round_if_valid(
        run_dir / 'p0_round2.json',
        round2_signature,
        legacy_matcher=lambda payload: legacy_round_payload_matches(
            payload,
            round_name='p0_round2',
            ab_name=f'{run_dir.name}_p0_r2',
            expected_candidates=round2_candidates,
            seed=seed + 303,
            step_scale=2.0,
            eval_splits=round2_eval_splits,
        ),
    )
    if round2 is None:
        round2 = load_revalidated_round_if_valid(
            run_dir=run_dir,
            round_name='p0_round2',
            ab_name=f'{run_dir.name}_p0_r2',
            ab_root_name=f'{run_dir.name}_p0_r2',
            expected_signature=round2_signature,
            expected_candidates=round2_candidates,
            seed=seed + 303,
            step_scale=2.0,
            eval_splits=round2_eval_splits,
        )
    if round2 is None:
        return None
    if not (round2.get('ranking') or []):
        return None
    return round0, round1, round2


def load_cached_p0_stage1_top4(
    run_dir: Path,
    state: dict[str, Any],
    *,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    seed: int,
    candidate_subset: str = 'all',
    skip_round1: bool = False,
    round2_max_candidates: int | None = None,
) -> list[CandidateSpec] | None:
    cached_rounds = load_cached_p0_rounds_upto_round2(
        run_dir,
        base_cfg,
        grouped,
        seed,
        candidate_subset=candidate_subset,
        skip_round1=skip_round1,
        round2_max_candidates=round2_max_candidates,
    )
    if cached_rounds is None:
        return None
    round0, round1, round2 = cached_rounds
    ranking = round2['ranking']
    stage1_top4 = select_p0_stage1_top4(ranking)
    p0_state = state.setdefault('p0', {})
    p0_state['round0'] = round0
    p0_state['round1'] = round1
    p0_state['round2'] = round2
    p0_state['stage1_top4'] = [candidate.arm_name for candidate in stage1_top4]
    state.setdefault('final_conclusion', {})['p0_stage1_top4'] = p0_state['stage1_top4']
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)
    return stage1_top4


def run_p0(
    run_dir: Path,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    seed: int,
    state: dict[str, Any],
    *,
    candidate_subset: str = 'all',
    skip_round1: bool = False,
    round2_max_candidates: int | None = None,
    skip_round3: bool = False,
) -> list[CandidateSpec]:
    p0_state = state.setdefault('p0', {})
    scheduler_profiles = {'cosine'} if candidate_subset == 'cosine_only' else None
    all_candidates = build_p0_candidates(scheduler_profiles=scheduler_profiles)
    all_index = {candidate.arm_name: candidate for candidate in all_candidates}

    round0 = execute_round(
        run_dir=run_dir,
        round_name='p0_round0',
        ab_name=f'{run_dir.name}_p0_r0',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=ab.build_eval_splits(grouped, seed + 11, ab.BASE_SCREENING['eval_files']),
        candidates=all_candidates,
        seed=seed + 101,
        step_scale=0.5,
    )
    p0_state['round0'] = round0
    update_results_doc(run_dir, state)
    top18 = build_p0_round0_survivors(
        round0['ranking'],
        all_index,
        candidate_subset=candidate_subset,
    )
    p0_state['round0_survivors'] = [candidate.arm_name for candidate in top18]
    if skip_round1:
        round1 = skipped_round_payload('p0_round1', reason='direct_round2_top8')
        p0_state['round1'] = round1
        round2_candidates = build_p0_round1_survivors(
            round0['ranking'],
            max_keep=round2_max_candidates,
        )
        update_results_doc(run_dir, state)
    else:
        round1 = execute_round(
            run_dir=run_dir,
            round_name='p0_round1',
            ab_name=f'{run_dir.name}_p0_r1',
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=ab.build_eval_splits(grouped, seed + 22, ab.BASE_SCREENING['eval_files']),
            candidates=top18,
            seed=seed + 202,
            step_scale=1.0,
        )
        p0_state['round1'] = round1
        update_results_doc(run_dir, state)
        round2_candidates = build_p0_round1_survivors(
            round1['ranking'],
            max_keep=round2_max_candidates,
        )
    p0_state['round1_survivors'] = [candidate.arm_name for candidate in round2_candidates]

    round2 = execute_round(
        run_dir=run_dir,
        round_name='p0_round2',
        ab_name=f'{run_dir.name}_p0_r2',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=ab.build_eval_splits(grouped, seed + 33, ab.BASE_SCREENING['eval_files']),
        candidates=round2_candidates,
        seed=seed + 303,
        step_scale=2.0,
    )
    p0_state['round2'] = round2
    update_results_doc(run_dir, state)
    round2_ranked = [candidate_from_entry(entry) for entry in round2['ranking']]
    stage1_top4 = select_p0_stage1_top4(round2['ranking'])
    p0_state['stage1_top4'] = [candidate.arm_name for candidate in stage1_top4]
    state.setdefault('final_conclusion', {})['p0_stage1_top4'] = p0_state['stage1_top4']
    if skip_round3:
        atomic_write_json(run_dir / 'state.json', state)
        update_results_doc(run_dir, state)
        return stage1_top4
    top2 = round2_ranked[:2]
    if not top2:
        raise RuntimeError('P0 round2 produced no finalists')
    if len(top2) < 2:
        p0_state['round3'] = skipped_round_payload('p0_round3', reason='insufficient_finalists')
        p0_state['winner'] = top2[0].arm_name
        p0_state['runner_up'] = None
        state.setdefault('final_conclusion', {})['p0_winner'] = p0_state['winner']
        atomic_write_json(run_dir / 'state.json', state)
        update_results_doc(run_dir, state)
        return stage1_top4

    round3 = execute_round(
        run_dir=run_dir,
        round_name='p0_round3',
        ab_name=f'{run_dir.name}_p0_r3',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=ab.build_eval_splits(grouped, seed + 44, ab.BASE_SCREENING['eval_files']),
        candidates=top2,
        seed=seed + 404,
        step_scale=4.0,
    )
    p0_state['round3'] = round3
    top2_final = [candidate_from_entry(entry) for entry in round3['ranking']]
    p0_state['winner'] = top2_final[0].arm_name
    p0_state['runner_up'] = top2_final[1].arm_name
    state.setdefault('final_conclusion', {})['p0_winner'] = p0_state['winner']
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)
    return stage1_top4


def run_p1(
    run_dir: Path,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    seed: int,
    protocols: list[CandidateSpec],
    state: dict[str, Any],
    *,
    calibration_protocol_arms: list[str] | None = None,
    calibration_mode: str = P1_CALIBRATION_DEFAULT_MODE,
    stop_after_calibration: bool = False,
    stop_after_protocol_decide: bool = False,
    stop_after_winner_refine: bool = False,
) -> tuple[CandidateSpec | None, list[CandidateSpec]]:
    p1_state = state.setdefault('p1', {})
    p1_state['selection_policy'] = p1_selection_policy_metadata()
    eval_splits = ab.build_eval_splits(grouped, seed + 55, ab.BASE_SCREENING['eval_files'])
    calibration_eval_splits = {
        **eval_splits,
        'old_regression_files': [],
    }
    rank_template_mean = rank_weight_mean_for_files(
        eval_splits['full_recent_files'],
        version=int(base_cfg['control']['version']),
        file_batch_size=int(ab.BASE_SCREENING['file_batch_size']),
    )
    p1_state['shared_rank_template_mean'] = rank_template_mean
    calibration_protocol_arms, calibration_protocols = resolve_p1_calibration_protocol_arms(
        protocols,
        calibration_protocol_arms,
    )
    p1_state['calibration_mode'] = calibration_mode
    p1_state['calibration_mode_note'] = p1_calibration_mode_note(calibration_mode)
    p1_state['calibration_protocol_arms'] = list(calibration_protocol_arms)
    calibration_round = execute_round_multiseed(
        run_dir=run_dir,
        round_name='p1_calibration',
        ab_name=f'{run_dir.name}_p1_calibration',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=calibration_eval_splits,
        candidates=build_p1_calibration_candidates(
            calibration_protocols,
            rank_template_mean,
            calibration_mode=calibration_mode,
        ),
        seed=seed + 404,
        seed_offsets=P1_CALIBRATION_SEED_OFFSETS,
        step_scale=P1_CALIBRATION_STEP_SCALE,
        ranking_mode=P1_RANKING_MODE,
        eligibility_group_key=P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    p1_state['calibration_round'] = calibration_round
    calibration = derive_p1_budget_calibration(
        calibration_round,
        requested_calibration_mode=calibration_mode,
        inherited_single_head=(
            P1_SINGLE_HEAD_CALIBRATION_BASELINE
            if calibration_mode == P1_CALIBRATION_MODE_COMBO_ONLY
            else None
        ),
        inherited_single_head_source=(
            P1_SINGLE_HEAD_CALIBRATION_SOURCE
            if calibration_mode == P1_CALIBRATION_MODE_COMBO_ONLY
            else None
        ),
    )
    calibration['shared_rank_template_mean'] = rank_template_mean
    calibration['calibration_protocol_arms'] = list(calibration_protocol_arms)
    p1_state['calibration'] = calibration
    p1_state['search_space'] = {
        'budget_ratios': calibration['budget_ratios'],
        'calibration_mode': calibration.get('calibration_mode', P1_CALIBRATION_DEFAULT_MODE),
        'calibration_mode_note': calibration.get(
            'calibration_mode_note',
            p1_calibration_mode_note(
                str(calibration.get('calibration_mode', P1_CALIBRATION_DEFAULT_MODE))
            ),
        ),
        'calibration_protocol_arms': list(calibration.get('calibration_protocol_arms', [])),
        'inherited_single_head_source': calibration.get('inherited_single_head_source'),
        'protocol_decide_total_budget_ratios': list(P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS),
        'protocol_decide_mixes': [
            {
                'name': name,
                'rank_share': rank_share,
                'opp_share': opp_share,
                'danger_share': danger_share,
            }
            for name, rank_share, opp_share, danger_share in P1_PROTOCOL_DECIDE_MIXES
        ],
        'ranking_mode': P1_RANKING_MODE,
        'eligibility_group_key': P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        'policy_loss_epsilon': P1_POLICY_LOSS_EPSILON,
        'old_regression_policy_loss_epsilon': P1_OLD_REGRESSION_POLICY_EPSILON,
        'mapping_mode': calibration.get('mapping_mode', P1_CALIBRATION_MAPPING_MODE),
        'combo_scheme': calibration.get('combo_scheme', P1_CALIBRATION_SCHEME),
        'opp_weight_per_budget_unit': calibration['opp_weight_per_budget_unit'],
        'danger_weight_per_budget_unit': calibration['danger_weight_per_budget_unit'],
        'calibration_seed_offsets': list(P1_CALIBRATION_SEED_OFFSETS),
        'grad_probe_max_batches': P1_GRAD_CALIBRATION_MAX_BATCHES,
        'protocol_decide_seed_offsets': list(P1_PROTOCOL_DECIDE_SEED_OFFSETS),
        'protocol_decide_seed_strategy': 'progressive_probe_then_expand',
        'protocol_decide_probe_keep_per_protocol': P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
        'winner_refine_seed_offsets': list(P1_WINNER_REFINE_SEED_OFFSETS),
        'winner_refine_centers': P1_WINNER_REFINE_CENTERS,
        'winner_refine_total_scale_factors': list(P1_WINNER_REFINE_TOTAL_SCALE_FACTORS),
        'winner_refine_transfer_delta': P1_WINNER_REFINE_TRANSFER_DELTA,
        'ablation_seed_offsets': list(P1_ABLATION_SEED_OFFSETS),
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
        'probe_weight': P1_CALIBRATION_PROBE_WEIGHT,
        'selection_policy': p1_selection_policy_metadata(),
    }
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)
    if stop_after_calibration:
        return None, []

    protocol_decide_round = execute_round_progressive_multiseed(
        run_dir=run_dir,
        round_name='p1_protocol_decide_round',
        ab_name=f'{run_dir.name}_p1_protocol_decide',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=build_p1_protocol_decide_candidates(protocols, calibration),
        seed=seed + 505,
        seed_offsets=P1_PROTOCOL_DECIDE_SEED_OFFSETS,
        step_scale=P1_PROTOCOL_DECIDE_STEP_SCALE,
        probe_selector_name='protocol_all_three_top4',
        probe_selector=lambda ranking, candidates: select_p1_protocol_decide_probe_candidates(
            ranking,
            candidates,
            keep=P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
        ),
        group_key='protocol_arm',
        probe_signature_data={
            'keep_per_protocol': P1_PROTOCOL_DECIDE_PROBE_KEEP_PER_PROTOCOL,
            'eligible_aux_family': 'all_three',
        },
        ranking_mode=P1_RANKING_MODE,
        eligibility_group_key=P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
        screening_diagnostic_name='ce_only_anchor',
        screening_diagnostic_selector=lambda entry: is_p1_protocol_decide_diagnostic_meta(
            entry.get('candidate_meta', {})
        ),
    )
    p1_state['protocol_decide_round'] = protocol_decide_round
    update_results_doc(run_dir, state)
    protocol_compare = build_p1_protocol_compare(protocol_decide_round['ranking'])
    p1_state['protocol_compare'] = {'round_name': 'p1_protocol_compare', 'ranking': protocol_compare}
    selected_protocol_arm = str(
        protocol_compare[0].get('candidate_meta', {}).get('protocol_arm', protocol_compare[0]['arm_name'])
    )
    p1_state['selected_protocol_arm'] = selected_protocol_arm
    state.setdefault('final_conclusion', {})['p1_protocol_winner'] = selected_protocol_arm
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)
    if stop_after_protocol_decide:
        return None, []

    winner_centers = select_p1_protocol_centers(
        protocol_decide_round['ranking'],
        protocol_arm=selected_protocol_arm,
        keep=P1_WINNER_REFINE_CENTERS,
    )
    p1_state['winner_refine_centers'] = [candidate.arm_name for candidate in winner_centers]
    winner_refine_round = execute_round_multiseed(
        run_dir=run_dir,
        round_name='p1_winner_refine_round',
        ab_name=f'{run_dir.name}_p1_winner_refine',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=build_p1_winner_refine_candidates(protocols, calibration, winner_centers),
        seed=seed + 606,
        seed_offsets=P1_WINNER_REFINE_SEED_OFFSETS,
        step_scale=P1_WINNER_REFINE_STEP_SCALE,
        ranking_mode=P1_RANKING_MODE,
        eligibility_group_key=P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    p1_state['winner_refine_round'] = winner_refine_round
    update_results_doc(run_dir, state)
    winner_refine_valid = [entry for entry in winner_refine_round['ranking'] if entry.get('valid')]
    if not winner_refine_valid:
        raise RuntimeError('p1_winner_refine_round produced no valid candidates')
    refine_winner = candidate_from_entry(winner_refine_valid[0])
    p1_state['winner_refine_front_runner'] = refine_winner.arm_name
    state.setdefault('final_conclusion', {})['p1_refine_front_runner'] = refine_winner.arm_name
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)
    if stop_after_winner_refine:
        return refine_winner, [candidate_from_entry(entry) for entry in winner_refine_valid]

    ablation_round = execute_round_multiseed(
        run_dir=run_dir,
        round_name='p1_ablation_round',
        ab_name=f'{run_dir.name}_p1_ablation',
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=build_p1_ablation_candidates(protocols, calibration, refine_winner),
        seed=seed + 707,
        seed_offsets=P1_ABLATION_SEED_OFFSETS,
        step_scale=P1_ABLATION_STEP_SCALE,
        ranking_mode=P1_RANKING_MODE,
        eligibility_group_key=P1_PROTOCOL_ELIGIBILITY_GROUP_KEY,
    )
    p1_state['ablation_round'] = ablation_round
    update_results_doc(run_dir, state)
    final_ranked = list(ablation_round['ranking'])
    p1_state['final_compare'] = {'round_name': 'p1_final_compare', 'ranking': final_ranked}
    final_valid = [entry for entry in final_ranked if entry.get('valid')]
    if not final_valid:
        raise RuntimeError('p1_ablation_round produced no valid candidates')
    winner = candidate_from_entry(final_valid[0])
    p1_state['winner'] = winner.arm_name
    state.setdefault('final_conclusion', {})['p1_winner'] = winner.arm_name
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)
    return winner, [candidate_from_entry(entry) for entry in final_valid]


def checkpoint_selector_winner(candidates: dict[str, dict[str, Any]], action_weights: dict[str, float]) -> tuple[str, list[str]]:
    best_loss = min(ab.full_recent_loss(candidate) for candidate in candidates.values())
    eligible = {
        name: candidate
        for name, candidate in candidates.items()
        if ab.full_recent_loss(candidate) <= best_loss + LOSS_EPSILON
    }
    winner = max(eligible.items(), key=lambda item: selection_key_for_summary(item[1], action_weights))[0]
    return winner, sorted(eligible)


def run_p2(run_dir: Path, finalists: list[CandidateSpec], state: dict[str, Any]) -> None:
    p2_state = {'curves': [], 'selected_checkpoints': []}
    selected: dict[str, dict[str, Any]] = {}
    for candidate in finalists[:P2_MAX_FINALISTS]:
        arm_result = load_json(find_arm_result_path(run_dir, candidate))
        final = arm_result['run']['final']
        ckpt_candidates = {
            'K1_best_loss': final['best_loss'],
            'K2_best_acc': final['best_acc'],
            'K3_best_rank': final['best_rank'],
            'K4_latest': final['latest'],
        }
        selector_winners = {}
        unique_types: set[str] = set()
        for selector_name, weights in SELECTOR_PROFILES.items():
            winner, eligible = checkpoint_selector_winner(ckpt_candidates, weights)
            selector_winners[selector_name] = {'winner': winner, 'eligible': eligible}
            unique_types.add(winner)
            label = f'{candidate.arm_name}:{winner}'
            selected.setdefault(
                label,
                {
                    'label': label,
                    'run_name': candidate.arm_name,
                    'checkpoint_type': winner,
                    'checkpoint_path': ckpt_candidates[winner]['path'],
                    'selectors': [],
                },
            )
            selected[label]['selectors'].append(selector_name)
        p2_state['curves'].append(
            {
                'run_name': candidate.arm_name,
                'default_selector_winner': selector_winners['S0_default']['winner'],
                'selector_winners': selector_winners,
                'unique_checkpoint_types': sorted(unique_types),
            }
        )
    p2_state['selected_checkpoints'] = sorted(
        selected.values(),
        key=lambda item: (
            len(item['selectors']),
            1 if 'S0_default' in item['selectors'] else 0,
            item['run_name'],
            item['checkpoint_type'],
        ),
        reverse=True,
    )[:3]
    state['p2'] = p2_state
    state.setdefault('final_conclusion', {})['p2_default_checkpoint'] = (
        p2_state['curves'][0]['default_selector_winner'] if p2_state['curves'] else None
    )
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)


def run_formal(
    run_dir: Path,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    winner: CandidateSpec,
    seed: int,
    formal_step_scale: float,
    state: dict[str, Any],
) -> None:
    formal.apply_formal_defaults()
    merged_cfg = ab.merge_dict(base_cfg, winner.cfg_overrides)
    protocol_arm = str(winner.meta.get('protocol_arm', winner.arm_name))
    ab_name = f'{run_dir.name}_formal'
    try:
        result = ab.run_ab6_checkpoint(
            merged_cfg,
            grouped,
            seed=seed,
            scheduler_profile=winner.scheduler_profile,
            curriculum_profile=winner.curriculum_profile,
            weight_profile=winner.weight_profile,
            window_profile=winner.window_profile,
            step_scale=formal_step_scale,
            ab_name=ab_name,
        )
        result = formal.finalize_formal_result(
            merged_cfg,
            result,
            protocol_arm=protocol_arm,
        )
        state['formal'] = {'status': 'completed', 'ab_name': ab_name, 'winner': result['winner'], 'result': result}
    except Exception as exc:
        state['formal'] = {'status': 'failed', 'ab_name': ab_name, 'error': str(exc), 'traceback': traceback.format_exc()}
    state.setdefault('final_conclusion', {})['formal_status'] = state['formal']['status']
    atomic_write_json(run_dir / 'state.json', state)
    update_results_doc(run_dir, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='s05_fidelity_main')
    parser.add_argument('--seed', type=int, default=ab.BASE_SCREENING['seed'])
    parser.add_argument('--formal-step-scale', type=float, default=5.0)
    parser.add_argument('--skip-formal', action='store_true')
    parser.add_argument('--stop-after-p0', action='store_true')
    parser.add_argument('--stop-after-p1-calibration', action='store_true')
    parser.add_argument('--stop-after-p1-protocol-decide', action='store_true')
    parser.add_argument('--stop-after-p1-winner-refine', action='store_true')
    parser.add_argument(
        '--p1-calibration-protocol-arm',
        action='append',
        dest='p1_calibration_protocol_arms',
        choices=sorted(
            {
                f'{scheduler_prefix}_{curriculum_prefix}{weight_prefix}{window_prefix}_'
                f'{scheduler_profile}_{curriculum_profile}_{weight_profile}_{window_profile}'
                for scheduler_prefix, scheduler_profile in ab.SCHEDULER_PREFIXES
                for curriculum_prefix, curriculum_profile in ab.CURRICULUM_PREFIXES
                for weight_prefix, weight_profile in ab.WEIGHT_PREFIXES
                for window_prefix, window_profile in ab.WINDOW_PREFIXES
            }
        ),
        help='Override the default representative protocol arms used by p1_calibration.',
    )
    parser.add_argument(
        '--p1-calibration-mode',
        choices=P1_CALIBRATION_MODE_CHOICES,
        default=P1_CALIBRATION_DEFAULT_MODE,
        help=(
            'Default is combo_only: inherit the frozen 2026-03-25 post-shape '
            'single-head calibration numbers and rerun only pairwise/triple '
            'combo probes.'
        ),
    )
    parser.add_argument(
        '--p0-candidate-subset',
        choices=('all', 'cosine_only'),
        default='all',
    )
    parser.add_argument('--p0-skip-round1', action='store_true')
    parser.add_argument('--p0-round2-max-candidates', type=int, default=0)
    args = parser.parse_args()

    run_dir = FIDELITY_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock_path = acquire_run_lock(run_dir, args.run_name)
    atexit.register(release_run_lock, run_lock_path)
    state_path = run_dir / 'state.json'
    state = load_json(state_path) if state_path.exists() else {'started_at': ts_now()}
    state.setdefault('started_at', ts_now())
    reset_state_for_stop_flags(
        state,
        stop_after_p0=args.stop_after_p0,
        stop_after_p1_calibration=args.stop_after_p1_calibration,
        stop_after_p1_protocol_decide=args.stop_after_p1_protocol_decide,
        stop_after_p1_winner_refine=args.stop_after_p1_winner_refine,
    )
    state['status'] = 'running'
    state.pop('fatal_error', None)
    state.pop('fatal_traceback', None)
    atomic_write_json(state_path, state)
    update_results_doc(run_dir, state)

    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())

    try:
        if (
            args.stop_after_p1_calibration
            or args.stop_after_p1_protocol_decide
            or args.stop_after_p1_winner_refine
        ):
            p0_top4 = load_cached_p0_stage1_top4(
                run_dir,
                state,
                base_cfg=base_cfg,
                grouped=grouped,
                seed=args.seed,
                candidate_subset=args.p0_candidate_subset,
                skip_round1=args.p0_skip_round1,
                round2_max_candidates=args.p0_round2_max_candidates or None,
            )
            if p0_top4 is None:
                p0_top4 = run_p0(
                    run_dir,
                    base_cfg,
                    grouped,
                    args.seed,
                    state,
                    candidate_subset=args.p0_candidate_subset,
                    skip_round1=args.p0_skip_round1,
                    round2_max_candidates=args.p0_round2_max_candidates or None,
                    skip_round3=True,
                )
        else:
            p0_top4 = run_p0(
                run_dir,
                base_cfg,
                grouped,
                args.seed,
                state,
                candidate_subset=args.p0_candidate_subset,
                skip_round1=args.p0_skip_round1,
                round2_max_candidates=args.p0_round2_max_candidates or None,
                skip_round3=args.stop_after_p0,
            )
        if args.stop_after_p0:
            state['status'] = 'stopped_after_p0'
            atomic_write_json(state_path, state)
            update_results_doc(run_dir, state)
            print(json.dumps(normalize_payload(state), ensure_ascii=False, indent=2))
            return
        p1_winner, p1_final_ranked = run_p1(
            run_dir,
            base_cfg,
            grouped,
            args.seed + 1000,
            p0_top4,
            state,
            calibration_protocol_arms=args.p1_calibration_protocol_arms,
            calibration_mode=args.p1_calibration_mode,
            stop_after_calibration=args.stop_after_p1_calibration,
            stop_after_protocol_decide=(
                args.stop_after_p1_protocol_decide and not args.stop_after_p1_calibration
            ),
            stop_after_winner_refine=(
                args.stop_after_p1_winner_refine
                and not args.stop_after_p1_calibration
                and not args.stop_after_p1_protocol_decide
            ),
        )
        if args.stop_after_p1_calibration:
            state['status'] = 'stopped_after_p1_calibration'
        elif args.stop_after_p1_protocol_decide:
            state['status'] = 'stopped_after_p1_protocol_decide'
        elif args.stop_after_p1_winner_refine:
            state['status'] = 'stopped_after_p1_winner_refine'
        else:
            run_p2(run_dir, p1_final_ranked, state)
            if args.skip_formal:
                state['formal'] = {'status': 'skipped'}
                state.setdefault('final_conclusion', {})['formal_status'] = 'skipped'
                atomic_write_json(state_path, state)
                update_results_doc(run_dir, state)
            else:
                assert p1_winner is not None
                run_formal(run_dir, base_cfg, grouped, p1_winner, args.seed + 2000, args.formal_step_scale, state)
            state['status'] = 'completed'
    except Exception as exc:
        state['status'] = 'failed'
        state['fatal_error'] = str(exc)
        state['fatal_traceback'] = traceback.format_exc()
    atomic_write_json(state_path, state)
    update_results_doc(run_dir, state)
    release_run_lock(run_lock_path)
    print(json.dumps(normalize_payload(state), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

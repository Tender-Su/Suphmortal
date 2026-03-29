from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, MutableMapping


LOSS_EPSILON = 0.003
# Retuned by the 2026-03-26 dense multiseed selector search after removing the
# duplicate scenario/action lexicographic tail. Scenario remains secondary to
# action quality without being drowned out by it.
SELECTION_SCENARIO_FACTOR = 0.20

# Weighted by expected downstream importance rather than raw frequency.
# Lower is better for every component below, so the final score negates them.
ACTION_SCORE_WEIGHTS = {
    'discard_nll': 0.45,
    'riichi_decision_balanced_bce': 0.18,
    'agari_decision_balanced_bce': 0.18,
    'chi_decision_balanced_bce': 0.04,
    'chi_exact_nll': 0.03,
    'pon_decision_balanced_bce': 0.07,
    'kan_decision_balanced_bce': 0.05,
}

SCENARIO_SCORE_WEIGHTS = {
    # P0: immediate scenario-score upgrades on existing hard slices.
    'riichi_decision_turn_late_balanced_bce': 0.08,
    'riichi_decision_round_southplus_balanced_bce': 0.05,
    'riichi_decision_all_last_yes_balanced_bce': 0.06,
    'riichi_decision_pressure_threat_balanced_bce': 0.05,
    'riichi_decision_pressure_calm_balanced_bce': 0.02,
    'riichi_decision_gap_close_2k_balanced_bce': 0.03,
    'riichi_decision_gap_close_4k_balanced_bce': 0.03,
    'riichi_decision_all_last_gap_close_4k_balanced_bce': 0.04,
    'riichi_decision_role_dealer_balanced_bce': 0.02,
    'riichi_decision_role_nondealer_balanced_bce': 0.02,
    'riichi_decision_rank_1_balanced_bce': 0.015,
    'riichi_decision_rank_2_balanced_bce': 0.015,
    'riichi_decision_rank_3_balanced_bce': 0.015,
    'riichi_decision_rank_4_balanced_bce': 0.015,
    'agari_decision_turn_late_balanced_bce': 0.03,
    'agari_decision_round_southplus_balanced_bce': 0.02,
    'agari_decision_all_last_yes_balanced_bce': 0.02,
    'agari_decision_pressure_threat_balanced_bce': 0.02,
    'agari_decision_gap_close_4k_balanced_bce': 0.02,
    'kan_decision_turn_late_balanced_bce': 0.02,
    'kan_decision_round_southplus_balanced_bce': 0.015,
    'kan_decision_all_last_yes_balanced_bce': 0.015,
    'kan_decision_pressure_threat_balanced_bce': 0.015,
    'kan_decision_gap_close_4k_balanced_bce': 0.015,
    'chi_decision_turn_late_balanced_bce': 0.015,
    'pon_decision_turn_late_balanced_bce': 0.015,
    # P1: directional gap, all-last target, threat intensity, opponent-state conditionals.
    'riichi_decision_pressure_single_threat_balanced_bce': 0.03,
    'riichi_decision_pressure_multi_threat_balanced_bce': 0.04,
    'riichi_decision_gap_up_close_2k_balanced_bce': 0.03,
    'riichi_decision_gap_down_close_2k_balanced_bce': 0.03,
    'riichi_decision_all_last_target_keep_first_balanced_bce': 0.03,
    'riichi_decision_all_last_target_chase_first_balanced_bce': 0.03,
    'riichi_decision_all_last_target_keep_above_fourth_balanced_bce': 0.03,
    'riichi_decision_all_last_target_escape_fourth_balanced_bce': 0.04,
    'riichi_decision_opp_any_tenpai_balanced_bce': 0.03,
    'riichi_decision_opp_multi_tenpai_balanced_bce': 0.03,
    'riichi_decision_opp_any_near_tenpai_balanced_bce': 0.02,
    'agari_decision_opp_any_tenpai_balanced_bce': 0.02,
    # P2: discard / push-fold score in truly decisive spots.
    'discard_all_last_yes_nll': 0.03,
    'discard_pressure_threat_nll': 0.03,
    'discard_pressure_single_threat_nll': 0.02,
    'discard_pressure_multi_threat_nll': 0.03,
    'discard_gap_close_2k_nll': 0.02,
    'discard_gap_close_4k_nll': 0.02,
    'discard_gap_up_close_2k_nll': 0.02,
    'discard_gap_down_close_2k_nll': 0.02,
    'discard_all_last_gap_close_4k_nll': 0.02,
    'discard_all_last_target_keep_first_nll': 0.02,
    'discard_all_last_target_chase_first_nll': 0.02,
    'discard_all_last_target_keep_above_fourth_nll': 0.02,
    'discard_all_last_target_escape_fourth_nll': 0.03,
    'discard_opp_any_tenpai_nll': 0.02,
    'discard_opp_multi_tenpai_nll': 0.02,
    'discard_opp_any_near_tenpai_nll': 0.02,
    'discard_push_fold_core_nll': 0.04,
    'discard_push_fold_extreme_nll': 0.05,
}

SCENARIO_SCORE_VERSION_FIELD = 'scenario_quality_score_version'

GLOBAL_DECISION_PREFIXES = frozenset({
    'riichi_decision',
    'agari_decision',
    'chi_decision',
    'pon_decision',
    'kan_decision',
})


def _weights_version(weights: Mapping[str, float]) -> str:
    payload = json.dumps(
        {key: float(value) for key, value in sorted(weights.items())},
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


SCENARIO_SCORE_VERSION = _weights_version(SCENARIO_SCORE_WEIGHTS)


def _as_float(metrics: Mapping[str, object], key: str, default: float | None = None) -> float | None:
    value = metrics.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_loss(value: float) -> float:
    clipped = max(float(value), 0.0)
    return clipped / (1.0 + clipped)


def _confidence_from_count(count: float | None, *, prior: float) -> float:
    if count is None:
        return 1.0
    clipped = max(float(count), 0.0)
    if clipped <= 0:
        return 0.0
    return clipped / (clipped + prior)


def _binary_effective_count(metrics: Mapping[str, object], prefix: str) -> float | None:
    pos_count = _as_float(metrics, f'{prefix}_pos_count')
    neg_count = _as_float(metrics, f'{prefix}_neg_count')
    if pos_count is not None and neg_count is not None:
        return min(pos_count, neg_count)
    return _as_float(metrics, f'{prefix}_count')


def _metric_effective_count(metrics: Mapping[str, object], key: str) -> float | None:
    if key == 'discard_nll':
        return _as_float(metrics, 'discard_count')
    if key == 'chi_exact_nll':
        return _as_float(metrics, 'chi_exact_count')
    if key.endswith('_balanced_bce'):
        prefix = key[:-len('_balanced_bce')]
        return _binary_effective_count(metrics, prefix)
    if key.endswith('_nll'):
        prefix = key[:-len('_nll')]
        return _as_float(metrics, f'{prefix}_count')
    return None


def _metric_confidence_prior(key: str) -> float:
    if key == 'discard_nll':
        return 8192.0
    if key == 'chi_exact_nll':
        return 384.0
    if key.endswith('_balanced_bce'):
        prefix = key[:-len('_balanced_bce')]
        if prefix in GLOBAL_DECISION_PREFIXES:
            return 1024.0
        return 256.0
    if key.startswith('discard_') and key.endswith('_nll'):
        return 384.0
    return 0.0


def _metric_confidence(metrics: Mapping[str, object], key: str) -> float:
    prior = _metric_confidence_prior(key)
    if prior <= 0:
        return 1.0
    return _confidence_from_count(_metric_effective_count(metrics, key), prior=prior)


def _weighted_quality_score(
    metrics: Mapping[str, object],
    weights: Mapping[str, float],
) -> tuple[float, bool]:
    score = 0.0
    used = False
    for key, weight in weights.items():
        value = _as_float(metrics, key)
        if value is None:
            continue
        score -= float(weight) * _metric_confidence(metrics, key) * _normalize_loss(value)
        used = True
    return score, used


def action_quality_score(
    metrics: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
) -> float:
    score, used = _weighted_quality_score(metrics, ACTION_SCORE_WEIGHTS if weights is None else weights)
    if used:
        return score

    explicit = _as_float(metrics, 'action_quality_score')
    if explicit is not None:
        return explicit

    legacy_macro = _as_float(metrics, 'macro_action_acc')
    if legacy_macro is not None:
        return legacy_macro

    legacy_action = _as_float(metrics, 'action_acc')
    if legacy_action is not None:
        return legacy_action

    return -math.inf


def action_quality_breakdown(metrics: Mapping[str, object]) -> dict[str, float | None]:
    return {key: _as_float(metrics, key) for key in ACTION_SCORE_WEIGHTS}


def scenario_quality_score(metrics: Mapping[str, object]) -> float:
    score, used = _weighted_quality_score(metrics, SCENARIO_SCORE_WEIGHTS)
    if used:
        return score

    explicit = _as_float(metrics, 'scenario_quality_score')
    if explicit is not None and scenario_score_version_matches(metrics):
        return explicit
    return -math.inf


def scenario_score_version_matches(metrics: Mapping[str, object]) -> bool:
    version = metrics.get(SCENARIO_SCORE_VERSION_FIELD)
    if version is None:
        return False
    return str(version) == SCENARIO_SCORE_VERSION


def refresh_scenario_quality_score(metrics: MutableMapping[str, object]) -> float:
    score = scenario_quality_score(metrics)
    if math.isfinite(score):
        metrics['scenario_quality_score'] = score
        metrics[SCENARIO_SCORE_VERSION_FIELD] = SCENARIO_SCORE_VERSION
        return score
    metrics.pop('scenario_quality_score', None)
    metrics.pop(SCENARIO_SCORE_VERSION_FIELD, None)
    return score


def scenario_quality_breakdown(metrics: Mapping[str, object]) -> dict[str, float | None]:
    return {key: _as_float(metrics, key) for key in SCENARIO_SCORE_WEIGHTS}


def selection_quality_score(
    metrics: Mapping[str, object],
    *,
    action_weights: Mapping[str, float] | None = None,
) -> float:
    action_score = action_quality_score(metrics, action_weights)
    scenario_score = scenario_quality_score(metrics)
    if math.isfinite(action_score) and math.isfinite(scenario_score):
        return action_score + SELECTION_SCENARIO_FACTOR * scenario_score
    if math.isfinite(action_score):
        return action_score
    if math.isfinite(scenario_score):
        return SELECTION_SCENARIO_FACTOR * scenario_score
    return -math.inf


def refresh_selection_quality_score(
    metrics: MutableMapping[str, object],
    *,
    action_weights: Mapping[str, float] | None = None,
) -> float:
    score = selection_quality_score(metrics, action_weights=action_weights)
    if math.isfinite(score):
        metrics['selection_quality_score'] = score
        return score
    metrics.pop('selection_quality_score', None)
    return score


def selection_tiebreak_key(
    metrics: Mapping[str, object],
    *,
    recent_loss: float,
    old_regression_loss: float,
    action_weights: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    return (
        selection_quality_score(metrics, action_weights=action_weights),
        -recent_loss,
        -old_regression_loss,
    )

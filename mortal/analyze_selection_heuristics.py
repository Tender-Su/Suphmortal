from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from libriichi.dataset import GameplayLoader
from sl_selection import SELECTION_SCENARIO_FACTOR


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(r"D:/mahjong_data/dataset_json")
DEFAULT_FIDELITY_ROOT = REPO_ROOT / "logs" / "sl_fidelity"
DEFAULT_REPORT_JSON = REPO_ROOT / "logs" / "selection_heuristic_audit.json"
DEFAULT_REPORT_MD = REPO_ROOT / "logs" / "selection_heuristic_audit.md"

ROUND_GLOBS = (
    "p1_protocol_decide_round.json",
    "p1_winner_refine_round.json",
    "p1_ablation_round.json",
    # Historical fallback only.
    "p1_solo_round.json",
)
CONTEXT_META_SPECS = {
    "at_turn": 0,
    "round_stage": 1,
    "is_dealer": 2,
    "is_all_last": 3,
    "self_rank": 4,
    "opp_riichi_count": 5,
    "up_gap_100": 6,
    "down_gap_100": 7,
}
ENCODED_MISSING_GAP = 65535 * 100
DECISION_SLICE_NAMES = (
    "turn_early",
    "turn_mid",
    "turn_late",
    "round_east",
    "round_southplus",
    "role_dealer",
    "role_nondealer",
    "all_last_yes",
    "all_last_no",
    "pressure_threat",
    "pressure_calm",
    "pressure_single_threat",
    "pressure_multi_threat",
    "gap_close_2k",
    "gap_close_4k",
    "gap_up_close_2k",
    "gap_up_close_4k",
    "gap_down_close_2k",
    "gap_down_close_4k",
    "all_last_gap_close_4k",
    "rank_1",
    "rank_2",
    "rank_3",
    "rank_4",
    "all_last_target_keep_first",
    "all_last_target_chase_first",
    "all_last_target_keep_above_fourth",
    "all_last_target_escape_fourth",
    "opp_any_tenpai",
    "opp_multi_tenpai",
    "opp_any_near_tenpai",
    "opp_multi_near_tenpai",
)
DISCARD_SLICE_NAMES = (
    "all_last_yes",
    "pressure_threat",
    "pressure_single_threat",
    "pressure_multi_threat",
    "gap_close_2k",
    "gap_close_4k",
    "gap_up_close_2k",
    "gap_down_close_2k",
    "all_last_gap_close_4k",
    "all_last_target_keep_first",
    "all_last_target_chase_first",
    "all_last_target_keep_above_fourth",
    "all_last_target_escape_fourth",
    "opp_any_tenpai",
    "opp_multi_tenpai",
    "opp_any_near_tenpai",
    "opp_multi_near_tenpai",
    "push_fold_core",
    "push_fold_extreme",
)
ROUNDING_STEP = 0.0005
SCENARIO_FACTOR_RECOMMENDATION_STEP = 0.01
SCENARIO_FACTOR_PAIRWISE_TOLERANCE = 0.005
DEFAULT_SCENARIO_FACTOR_MIN = 0.0
DEFAULT_SCENARIO_FACTOR_MAX = 0.25
DEFAULT_SCENARIO_FACTOR_STEP = 0.001
LEGACY_SCENARIO_FACTOR = 0.5


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def round_up_step(value: float, step: float = ROUNDING_STEP) -> float:
    if value <= 0:
        return 0.0
    return math.ceil(value / step) * step


def round_nearest_step(value: float, step: float = SCENARIO_FACTOR_RECOMMENDATION_STEP) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def build_scenario_factor_grid(
    start: float = DEFAULT_SCENARIO_FACTOR_MIN,
    stop: float = DEFAULT_SCENARIO_FACTOR_MAX,
    step: float = DEFAULT_SCENARIO_FACTOR_STEP,
    *,
    extra_points: list[float] | None = None,
) -> list[float]:
    if step <= 0:
        raise ValueError("scenario-factor step must be > 0")
    if stop < start:
        raise ValueError("scenario-factor stop must be >= start")
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 12))
        current += step
    if extra_points:
        values.extend(round(point, 12) for point in extra_points)
    return sorted(set(values))


def collect_sample_paths(
    dataset_root: Path,
    *,
    seed: int,
    per_year: int,
    max_files: int,
) -> list[str]:
    year_to_paths: dict[str, list[str]] = defaultdict(list)
    for path in dataset_root.rglob("*.json"):
        year = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        year_to_paths[year].append(str(path))

    rng = random.Random(seed)
    selected: list[str] = []
    for year, year_paths in sorted(year_to_paths.items()):
        rng.shuffle(year_paths)
        selected.extend(year_paths[:per_year])
    if max_files > 0:
        return selected[:max_files]
    return selected


def iter_games(paths: list[str], file_batch_size: int) -> Any:
    for start in range(0, len(paths), file_batch_size):
        batch_paths = paths[start:start + file_batch_size]
        print(
            f"[selection-audit] loading files {start + 1}-{start + len(batch_paths)} / {len(paths)}",
            flush=True,
        )
        loader = GameplayLoader(
            version=4,
            oracle=False,
            track_opponent_states=True,
            track_danger_labels=False,
        )
        try:
            nested = loader.load_log_files(batch_paths)
        except Exception as exc:
            print(
                f"[selection-audit] batch failed, falling back to per-file scan: {exc}",
                flush=True,
            )
            nested = []
            for path in batch_paths:
                try:
                    nested.extend(loader.load_log_files([path]))
                except Exception as inner_exc:
                    print(
                        f"[selection-audit] skip bad file: {path} :: {inner_exc}",
                        flush=True,
                    )
        for games in nested:
            for game in games:
                yield game


def decode_context_meta(context_meta: np.ndarray) -> dict[str, np.ndarray]:
    up_gap_points = context_meta[:, CONTEXT_META_SPECS["up_gap_100"]].astype(np.int64) * 100
    down_gap_points = context_meta[:, CONTEXT_META_SPECS["down_gap_100"]].astype(np.int64) * 100
    sentinel = np.iinfo(np.int64).max
    up_gap_points = np.where(up_gap_points == ENCODED_MISSING_GAP, sentinel, up_gap_points)
    down_gap_points = np.where(down_gap_points == ENCODED_MISSING_GAP, sentinel, down_gap_points)
    nearest_gap_points = np.minimum(up_gap_points, down_gap_points)
    return {
        "at_turn": context_meta[:, CONTEXT_META_SPECS["at_turn"]].astype(np.int64),
        "round_stage": context_meta[:, CONTEXT_META_SPECS["round_stage"]].astype(np.int64),
        "is_dealer": context_meta[:, CONTEXT_META_SPECS["is_dealer"]].astype(bool),
        "is_all_last": context_meta[:, CONTEXT_META_SPECS["is_all_last"]].astype(bool),
        "self_rank": context_meta[:, CONTEXT_META_SPECS["self_rank"]].astype(np.int64),
        "opp_riichi_count": context_meta[:, CONTEXT_META_SPECS["opp_riichi_count"]].astype(np.int64),
        "up_gap_points": up_gap_points,
        "down_gap_points": down_gap_points,
        "nearest_gap_points": nearest_gap_points,
    }


def build_decision_slice_masks(
    context: dict[str, np.ndarray],
    *,
    opponent_shanten: np.ndarray | None,
    opponent_tenpai: np.ndarray | None,
) -> dict[str, np.ndarray]:
    up_gap_close_2k = context["up_gap_points"] <= 2000
    up_gap_close_4k = context["up_gap_points"] <= 4000
    down_gap_close_2k = context["down_gap_points"] <= 2000
    down_gap_close_4k = context["down_gap_points"] <= 4000
    nearest_gap_close_2k = context["nearest_gap_points"] <= 2000
    nearest_gap_close_4k = context["nearest_gap_points"] <= 4000
    pressure_single_threat = context["opp_riichi_count"] == 1
    pressure_multi_threat = context["opp_riichi_count"] >= 2

    if opponent_tenpai is None:
        opp_any_tenpai = np.zeros_like(context["is_all_last"], dtype=bool)
        opp_multi_tenpai = np.zeros_like(context["is_all_last"], dtype=bool)
    else:
        opponent_tenpai = opponent_tenpai.astype(bool)
        opp_any_tenpai = opponent_tenpai.any(axis=1)
        opp_multi_tenpai = opponent_tenpai.sum(axis=1) >= 2

    if opponent_shanten is None:
        opp_any_near_tenpai = np.zeros_like(context["is_all_last"], dtype=bool)
        opp_multi_near_tenpai = np.zeros_like(context["is_all_last"], dtype=bool)
    else:
        near_tenpai = opponent_shanten.astype(np.int64) <= 1
        opp_any_near_tenpai = near_tenpai.any(axis=1)
        opp_multi_near_tenpai = near_tenpai.sum(axis=1) >= 2

    return {
        "turn_early": context["at_turn"] <= 4,
        "turn_mid": (context["at_turn"] >= 5) & (context["at_turn"] <= 11),
        "turn_late": context["at_turn"] >= 12,
        "round_east": context["round_stage"] == 0,
        "round_southplus": context["round_stage"] == 1,
        "role_dealer": context["is_dealer"],
        "role_nondealer": ~context["is_dealer"],
        "all_last_yes": context["is_all_last"],
        "all_last_no": ~context["is_all_last"],
        "pressure_threat": context["opp_riichi_count"] >= 1,
        "pressure_calm": context["opp_riichi_count"] == 0,
        "pressure_single_threat": pressure_single_threat,
        "pressure_multi_threat": pressure_multi_threat,
        "gap_close_2k": nearest_gap_close_2k,
        "gap_close_4k": nearest_gap_close_4k,
        "gap_up_close_2k": up_gap_close_2k,
        "gap_up_close_4k": up_gap_close_4k,
        "gap_down_close_2k": down_gap_close_2k,
        "gap_down_close_4k": down_gap_close_4k,
        "all_last_gap_close_4k": context["is_all_last"] & nearest_gap_close_4k,
        "rank_1": context["self_rank"] == 0,
        "rank_2": context["self_rank"] == 1,
        "rank_3": context["self_rank"] == 2,
        "rank_4": context["self_rank"] == 3,
        "all_last_target_keep_first": context["is_all_last"] & (context["self_rank"] == 0) & down_gap_close_4k,
        "all_last_target_chase_first": context["is_all_last"] & (context["self_rank"] == 1) & up_gap_close_4k,
        "all_last_target_keep_above_fourth": context["is_all_last"] & (context["self_rank"] == 2) & down_gap_close_4k,
        "all_last_target_escape_fourth": context["is_all_last"] & (context["self_rank"] == 3) & up_gap_close_4k,
        "opp_any_tenpai": opp_any_tenpai,
        "opp_multi_tenpai": opp_multi_tenpai,
        "opp_any_near_tenpai": opp_any_near_tenpai,
        "opp_multi_near_tenpai": opp_multi_near_tenpai,
    }


def build_discard_slice_masks(
    context: dict[str, np.ndarray],
    *,
    opponent_shanten: np.ndarray | None,
    opponent_tenpai: np.ndarray | None,
) -> dict[str, np.ndarray]:
    slice_masks = build_decision_slice_masks(
        context,
        opponent_shanten=opponent_shanten,
        opponent_tenpai=opponent_tenpai,
    )
    threat_mask = slice_masks["pressure_threat"] | slice_masks["opp_any_tenpai"]
    extreme_threat_mask = slice_masks["pressure_multi_threat"] | slice_masks["opp_multi_tenpai"]
    micro_gap_mask = (
        slice_masks["gap_close_2k"]
        | slice_masks["gap_up_close_2k"]
        | slice_masks["gap_down_close_2k"]
    )
    slice_masks.update({
        "push_fold_core": threat_mask & (slice_masks["all_last_yes"] | slice_masks["gap_close_4k"]),
        "push_fold_extreme": extreme_threat_mask & (slice_masks["all_last_gap_close_4k"] | micro_gap_mask),
    })
    return {name: slice_masks[name] for name in DISCARD_SLICE_NAMES}


class SelectorCoverageAudit:
    def __init__(self) -> None:
        self.sample_files = 0
        self.sample_states = 0
        self.decision_counts = Counter[str]()
        self.discard_counts = Counter[str]()
        self.decision_active_hist = Counter[int]()
        self.discard_active_hist = Counter[int]()

    def observe_game(self, game: Any) -> None:
        context_meta = np.asarray(game.take_context_meta_batch())
        opponent_shanten = np.asarray(game.take_opponent_shanten_batch(), dtype=np.int64)
        opponent_tenpai = np.asarray(game.take_opponent_tenpai_batch(), dtype=np.int64)
        context = decode_context_meta(context_meta)
        decision_masks = build_decision_slice_masks(
            context,
            opponent_shanten=opponent_shanten,
            opponent_tenpai=opponent_tenpai,
        )
        discard_masks = build_discard_slice_masks(
            context,
            opponent_shanten=opponent_shanten,
            opponent_tenpai=opponent_tenpai,
        )

        self.sample_states += int(context_meta.shape[0])
        for name, mask in decision_masks.items():
            self.decision_counts[name] += int(mask.sum())
        for name, mask in discard_masks.items():
            self.discard_counts[name] += int(mask.sum())

        decision_active = np.stack([decision_masks[name] for name in DECISION_SLICE_NAMES], axis=1).sum(axis=1)
        discard_active = np.stack([discard_masks[name] for name in DISCARD_SLICE_NAMES], axis=1).sum(axis=1)
        self.decision_active_hist.update(int(v) for v in decision_active.tolist())
        self.discard_active_hist.update(int(v) for v in discard_active.tolist())

    @staticmethod
    def _hist_summary(hist: Counter[int], total: int) -> dict[str, Any]:
        if total <= 0:
            return {"mean": 0.0, "median": 0, "p90": 0, "max": 0}
        running = 0
        items = sorted(hist.items())
        targets = {
            "median": max(1, math.ceil(total * 0.50)),
            "p90": max(1, math.ceil(total * 0.90)),
        }
        result: dict[str, Any] = {}
        for value, count in items:
            running += count
            for name, threshold in list(targets.items()):
                if running >= threshold and name not in result:
                    result[name] = value
        weighted_sum = sum(value * count for value, count in hist.items())
        result["mean"] = weighted_sum / total
        result["median"] = result.get("median", 0)
        result["p90"] = result.get("p90", 0)
        result["max"] = max(hist) if hist else 0
        return result

    def summary(self) -> dict[str, Any]:
        total = self.sample_states
        decision_rows = [
            {
                "slice": name,
                "count": self.decision_counts[name],
                "rate": (self.decision_counts[name] / total) if total > 0 else 0.0,
            }
            for name in DECISION_SLICE_NAMES
        ]
        discard_rows = [
            {
                "slice": name,
                "count": self.discard_counts[name],
                "rate": (self.discard_counts[name] / total) if total > 0 else 0.0,
            }
            for name in DISCARD_SLICE_NAMES
        ]
        decision_rows.sort(key=lambda item: item["count"], reverse=True)
        discard_rows.sort(key=lambda item: item["count"], reverse=True)
        return {
            "sample_files": self.sample_files,
            "sample_states": self.sample_states,
            "decision_slice_rates": decision_rows,
            "discard_slice_rates": discard_rows,
            "decision_active_slices_per_state": self._hist_summary(self.decision_active_hist, total),
            "discard_active_slices_per_state": self._hist_summary(self.discard_active_hist, total),
        }


def collect_selector_coverage(paths: list[str], *, file_batch_size: int) -> SelectorCoverageAudit:
    audit = SelectorCoverageAudit()
    audit.sample_files = len(paths)
    for game in iter_games(paths, file_batch_size):
        audit.observe_game(game)
    return audit


def load_multiseed_rounds(fidelity_root: Path) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for round_glob in ROUND_GLOBS:
        for path in sorted(fidelity_root.rglob(round_glob)):
            if path in seen:
                continue
            seen.add(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            actual_seeds = payload.get("actual_seeds", [])
            if payload.get("ranking_mode") != "policy_quality":
                continue
            if len(actual_seeds) < 2:
                continue
            rounds.append({"path": str(path), "payload": payload})
    return rounds


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_noise_stats(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    policy_diffs: list[float] = []
    old_diffs: list[float] = []
    full_diffs: list[float] = []

    for round_info in rounds:
        for entry in round_info["payload"].get("ranking", []):
            seeds = [seed for seed in entry.get("seed_summaries", []) if seed.get("valid")]
            if len(seeds) < 2:
                continue
            for key, bucket in (
                ("recent_policy_loss", policy_diffs),
                ("old_regression_policy_loss", old_diffs),
                ("full_recent_loss", full_diffs),
            ):
                values = [safe_float(seed.get(key)) for seed in seeds if seed.get(key) is not None]
                if len(values) >= 2 and values[0] is not None and values[1] is not None:
                    bucket.append(abs(values[0] - values[1]))

    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "mean_abs_diff": None,
                "median_abs_diff": None,
                "p90_abs_diff": None,
                "p95_abs_diff": None,
                "max_abs_diff": None,
                "suggested_epsilon_roundup_p90": None,
            }
        return {
            "count": len(values),
            "mean_abs_diff": sum(values) / len(values),
            "median_abs_diff": quantile(values, 0.50),
            "p90_abs_diff": quantile(values, 0.90),
            "p95_abs_diff": quantile(values, 0.95),
            "max_abs_diff": max(values),
            "suggested_epsilon_roundup_p90": round_up_step(quantile(values, 0.90) or 0.0),
        }

    return {
        "policy": summarize(policy_diffs),
        "old_regression": summarize(old_diffs),
        "full_loss": summarize(full_diffs),
    }


def selection_sort_key(seed_summary: dict[str, Any], scenario_factor: float) -> tuple[float, ...]:
    action_score = safe_float(seed_summary.get("action_quality_score"), float("-inf"))
    scenario_score = safe_float(seed_summary.get("scenario_quality_score"), float("-inf"))
    if math.isfinite(action_score) and math.isfinite(scenario_score):
        selection_score = action_score + scenario_factor * scenario_score
    elif math.isfinite(action_score):
        selection_score = action_score
    elif math.isfinite(scenario_score):
        selection_score = scenario_factor * scenario_score
    else:
        selection_score = float("-inf")
    return (
        selection_score,
        -safe_float(seed_summary.get("recent_policy_loss"), float("inf")),
        -safe_float(seed_summary.get("old_regression_policy_loss"), float("inf")),
    )


def recommend_scenario_factor(
    results: list[dict[str, Any]],
    *,
    current_factor: float,
    search_min: float,
    search_max: float,
) -> dict[str, Any]:
    all_rows = [
        row
        for row in results
        if row["aggregate_winner_match_rate"] is not None and row["pairwise_agreement"] is not None
    ]
    valid_rows = [
        row
        for row in all_rows
        if search_min - 1e-12 <= float(row["scenario_factor"]) <= search_max + 1e-12
    ]
    if not valid_rows:
        return {
            "replaceable_by_statistics": False,
            "reason": "当前没有足够的多 seed 结果支持 scenario_factor 搜索。",
            "recommended_value": current_factor,
            "best_dense_band": [None, None],
            "best_aggregate_winner_match_rate": None,
            "current_aggregate_winner_match_rate": None,
            "best_pairwise_agreement": None,
            "current_pairwise_agreement": None,
            "legacy_pairwise_agreement_at_0_5": None,
            "legacy_aggregate_winner_match_rate_at_0_5": None,
        }

    best_pairwise = max(float(row["pairwise_agreement"]) for row in valid_rows)
    winner_rows = [
        row
        for row in valid_rows
        if (best_pairwise - float(row["pairwise_agreement"])) <= SCENARIO_FACTOR_PAIRWISE_TOLERANCE + 1e-12
    ]
    best_winner_match = max(float(row["aggregate_winner_match_rate"]) for row in winner_rows)
    best_rows = [
        row
        for row in winner_rows
        if abs(float(row["aggregate_winner_match_rate"]) - best_winner_match) <= 1e-12
    ]
    plateau = sorted(float(row["scenario_factor"]) for row in best_rows)
    recommended_raw = float(statistics.median(plateau))
    recommended_value = round_nearest_step(recommended_raw)
    current = next((row for row in all_rows if abs(float(row["scenario_factor"]) - current_factor) <= 1e-12), None)
    legacy = next((row for row in all_rows if abs(float(row["scenario_factor"]) - LEGACY_SCENARIO_FACTOR) <= 1e-12), None)
    return {
        "replaceable_by_statistics": False,
        "reason": (
            "scenario_factor 仍然不是可由独立真实牌力标签唯一拟合出的统计常数；"
            "但在 selection_tiebreak_key 收敛到 `selection_quality_score -> recent -> old_regression` 后，"
            "当前密集搜索带内更适合先用 `pairwise_agreement` 看排序稳定性，"
            "再用 `aggregate_winner_match_rate` 做次级筛选；"
            f"按这条规则，最优平台收敛到 `{plateau[0]:.3f}-{plateau[-1]:.3f}`，因此把运行时默认值收敛到平台中部。"
        ),
        "recommended_value": recommended_value,
        "best_dense_band": [plateau[0], plateau[-1]],
        "best_aggregate_winner_match_rate": best_winner_match,
        "current_aggregate_winner_match_rate": None if current is None else current["aggregate_winner_match_rate"],
        "best_pairwise_agreement": best_pairwise,
        "current_pairwise_agreement": None if current is None else current["pairwise_agreement"],
        "legacy_pairwise_agreement_at_0_5": None if legacy is None else legacy["pairwise_agreement"],
        "legacy_aggregate_winner_match_rate_at_0_5": (
            None if legacy is None else legacy["aggregate_winner_match_rate"]
        ),
    }


def scenario_factor_scan(
    rounds: list[dict[str, Any]],
    *,
    policy_epsilon: float,
    old_epsilon: float,
    scenario_factor_grid: list[float],
    search_min: float,
    search_max: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for scenario_factor in scenario_factor_grid:
        aggregate_winner_match = 0
        aggregate_winner_total = 0
        seed_winner_cross_agree = 0
        seed_winner_cross_total = 0
        pairwise_agree = 0
        pairwise_total = 0

        for round_info in rounds:
            payload = round_info["payload"]
            by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entry in payload.get("ranking", []):
                group_key = str(entry.get("eligibility_group") or entry.get("candidate_meta", {}).get("protocol_arm") or "")
                if group_key:
                    by_group[group_key].append(entry)

            for group_entries in by_group.values():
                winners: list[str] = []
                valid_entries = [
                    entry for entry in group_entries
                    if sum(1 for seed in entry.get("seed_summaries", []) if seed.get("valid")) >= 2
                ]
                if not valid_entries:
                    continue

                for seed_idx in range(2):
                    per_seed_valid = []
                    for entry in valid_entries:
                        seed = entry["seed_summaries"][seed_idx]
                        if seed.get("valid") and seed.get("recent_policy_loss") is not None:
                            per_seed_valid.append((entry, seed))
                    if not per_seed_valid:
                        continue
                    best_policy = min(safe_float(seed["recent_policy_loss"], float("inf")) for _, seed in per_seed_valid)
                    eligible = [
                        (entry, seed)
                        for entry, seed in per_seed_valid
                        if safe_float(seed["recent_policy_loss"], float("inf")) <= best_policy + policy_epsilon
                    ]
                    old_values = [
                        safe_float(seed.get("old_regression_policy_loss"))
                        for _, seed in eligible
                        if seed.get("old_regression_policy_loss") is not None
                    ]
                    if old_values:
                        best_old = min(old_values)
                        eligible = [
                            (entry, seed)
                            for entry, seed in eligible
                            if seed.get("old_regression_policy_loss") is not None
                            and safe_float(seed["old_regression_policy_loss"], float("inf")) <= best_old + old_epsilon
                        ]
                    eligible.sort(key=lambda item: selection_sort_key(item[1], scenario_factor), reverse=True)
                    winners.append(eligible[0][0]["arm_name"])

                mean_policy_rows = []
                for entry in valid_entries:
                    seed_values = [
                        safe_float(seed.get("recent_policy_loss"))
                        for seed in entry.get("seed_summaries", [])
                        if seed.get("valid") and seed.get("recent_policy_loss") is not None
                    ]
                    if not seed_values:
                        continue
                    mean_policy_rows.append((entry, sum(seed_values) / len(seed_values)))
                if not mean_policy_rows:
                    continue
                best_mean_policy = min(mean_policy for _, mean_policy in mean_policy_rows)
                pool = [
                    entry
                    for entry, mean_policy in mean_policy_rows
                    if mean_policy <= best_mean_policy + policy_epsilon
                ]
                old_mean_rows = [
                    safe_float(entry.get("old_regression_policy_loss"))
                    for entry in pool
                    if entry.get("old_regression_policy_loss") is not None
                ]
                if old_mean_rows:
                    best_mean_old = min(value for value in old_mean_rows if value is not None)
                    pool = [
                        entry
                        for entry in pool
                        if entry.get("old_regression_policy_loss") is not None
                        and safe_float(entry.get("old_regression_policy_loss"), float("inf")) <= best_mean_old + old_epsilon
                    ]
                if not pool:
                    continue
                pool.sort(key=lambda entry: selection_sort_key(entry, scenario_factor), reverse=True)
                aggregate_winner = pool[0]["arm_name"]
                aggregate_winner_total += len(winners)
                aggregate_winner_match += sum(int(winner == aggregate_winner) for winner in winners)
                if len(winners) >= 2:
                    seed_winner_cross_total += 1
                    seed_winner_cross_agree += int(winners[0] == winners[1])
                for idx, left in enumerate(pool):
                    for right in pool[idx + 1:]:
                        left_seed0 = left["seed_summaries"][0]
                        left_seed1 = left["seed_summaries"][1]
                        right_seed0 = right["seed_summaries"][0]
                        right_seed1 = right["seed_summaries"][1]
                        order0 = (
                            selection_sort_key(left_seed0, scenario_factor)
                            > selection_sort_key(right_seed0, scenario_factor)
                        ) - (
                            selection_sort_key(left_seed0, scenario_factor)
                            < selection_sort_key(right_seed0, scenario_factor)
                        )
                        order1 = (
                            selection_sort_key(left_seed1, scenario_factor)
                            > selection_sort_key(right_seed1, scenario_factor)
                        ) - (
                            selection_sort_key(left_seed1, scenario_factor)
                            < selection_sort_key(right_seed1, scenario_factor)
                        )
                        if order0 == 0 or order1 == 0:
                            continue
                        pairwise_total += 1
                        pairwise_agree += int(order0 == order1)

        pairwise_agreement = (pairwise_agree / pairwise_total) if pairwise_total > 0 else None
        aggregate_winner_match_rate = (
            aggregate_winner_match / aggregate_winner_total if aggregate_winner_total > 0 else None
        )
        seed_winner_cross_agreement = (
            seed_winner_cross_agree / seed_winner_cross_total if seed_winner_cross_total > 0 else None
        )
        results.append({
            "scenario_factor": scenario_factor,
            "aggregate_winner_match_rate": aggregate_winner_match_rate,
            "aggregate_winner_total": aggregate_winner_total,
            "seed_winner_cross_agreement": seed_winner_cross_agreement,
            "seed_winner_cross_total": seed_winner_cross_total,
            "pairwise_agreement": pairwise_agreement,
            "pairwise_total": pairwise_total,
        })
    recommendation = recommend_scenario_factor(
        results,
        current_factor=SELECTION_SCENARIO_FACTOR,
        search_min=search_min,
        search_max=search_max,
    )
    return {"grid": results, "recommendation": recommendation}


def build_report(
    *,
    dataset_root: Path,
    fidelity_root: Path,
    seed: int,
    per_year: int,
    max_files: int,
    file_batch_size: int,
    scenario_factor_min: float,
    scenario_factor_max: float,
    scenario_factor_step: float,
) -> dict[str, Any]:
    sample_paths = collect_sample_paths(
        dataset_root,
        seed=seed,
        per_year=per_year,
        max_files=max_files,
    )
    coverage = collect_selector_coverage(sample_paths, file_batch_size=file_batch_size).summary()
    rounds = load_multiseed_rounds(fidelity_root)
    noise = collect_noise_stats(rounds)
    policy_epsilon = safe_float(noise["policy"]["suggested_epsilon_roundup_p90"], 0.003) or 0.003
    old_epsilon = safe_float(noise["old_regression"]["suggested_epsilon_roundup_p90"], 0.004) or 0.004
    factor_scan = scenario_factor_scan(
        rounds,
        policy_epsilon=policy_epsilon,
        old_epsilon=old_epsilon,
        scenario_factor_grid=build_scenario_factor_grid(
            scenario_factor_min,
            scenario_factor_max,
            scenario_factor_step,
            extra_points=[LEGACY_SCENARIO_FACTOR, SELECTION_SCENARIO_FACTOR],
        ),
        search_min=scenario_factor_min,
        search_max=scenario_factor_max,
    )
    return {
        "dataset_root": str(dataset_root),
        "fidelity_root": str(fidelity_root),
        "sample_seed": seed,
        "sample_per_year": per_year,
        "sample_max_files": max_files,
        "sample_files": len(sample_paths),
        "coverage": coverage,
        "multiseed_rounds": [round_info["path"] for round_info in rounds],
        "noise": noise,
        "recommended_runtime_constants": {
            "policy_loss_epsilon": policy_epsilon,
            "old_regression_policy_loss_epsilon": old_epsilon,
            "selection_scenario_factor": factor_scan["recommendation"]["recommended_value"],
        },
        "scenario_factor_scan": factor_scan,
        "selector_components": [
            {
                "component": "comparison_recent_loss = recent_policy_loss",
                "replaceable_by_statistics": False,
                "status": "keep_principled_rule",
                "reason": "这是实验目标定义，不是数据分布自己能推出的常数。",
            },
            {
                "component": "eligibility_group_key = protocol_arm",
                "replaceable_by_statistics": False,
                "status": "keep_principled_rule",
                "reason": "这是公平对照约束，不是经验频率问题。",
            },
            {
                "component": "policy_loss_epsilon",
                "replaceable_by_statistics": True,
                "status": "replace_with_noise_estimate",
                "reason": "可直接用多 seed 同 arm 的 recent_policy_loss 抖动分布估计。",
                "recommended_value": policy_epsilon,
            },
            {
                "component": "old_regression_policy_loss_epsilon",
                "replaceable_by_statistics": True,
                "status": "replace_with_noise_estimate",
                "reason": "可直接用多 seed 同 arm 的 old_regression_policy_loss 抖动分布估计。",
                "recommended_value": old_epsilon,
            },
            {
                "component": "selection_quality_score scenario factor",
                "replaceable_by_statistics": False,
                "status": "retune_by_dense_multiseed_search",
                "reason": factor_scan["recommendation"]["reason"],
                "recommended_value": factor_scan["recommendation"]["recommended_value"],
            },
            {
                "component": "ACTION_SCORE_WEIGHTS / SCENARIO_SCORE_WEIGHTS",
                "replaceable_by_statistics": False,
                "status": "keep_heuristic_for_now",
                "reason": (
                    "切片高度重叠，长期样本只能给覆盖率和重叠结构，不能唯一识别“真实牌力重要度”权重；"
                    "若强行数据拟合，容易把重叠切片的共线性误当成权重结论。"
                ),
            },
            {
                "component": "selection_tiebreak_key lexicographic tail",
                "replaceable_by_statistics": False,
                "status": "simplify_tail",
                "reason": (
                    "当前 selector 已改为只用 selection_quality_score 做唯一综合质量分，"
                    "排序键只保留 `-recent_policy_loss / -old_regression_policy_loss` 回退；"
                    "各类 accuracy 字段仅作诊断输出，不再参与排序。"
                ),
            },
            {
                "component": "count-shrinkage priors",
                "replaceable_by_statistics": False,
                "status": "keep_heuristic_for_now",
                "reason": "当前仓库没有针对 prior 的受控多 seed 重复实验，无法把 shrinkage prior 精确拟合成统计常数。",
            },
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    policy_noise = report["noise"]["policy"]
    old_noise = report["noise"]["old_regression"]
    factor = report["scenario_factor_scan"]["recommendation"]
    decision_top = coverage["decision_slice_rates"][:8]
    decision_rare = coverage["decision_slice_rates"][-8:]
    discard_top = coverage["discard_slice_rates"][:8]
    discard_rare = coverage["discard_slice_rates"][-8:]

    lines = [
        "# Selector Heuristic Audit",
        "",
        "## Data",
        "",
        f"- dataset_root: `{report['dataset_root']}`",
        f"- fidelity_root: `{report['fidelity_root']}`",
        f"- sampled files: `{report['sample_files']}`",
        f"- sampled states: `{coverage['sample_states']}`",
        f"- multiseed rounds: `{len(report['multiseed_rounds'])}`",
        "",
        "## Slice Coverage",
        "",
        f"- decision active slices per state: `{coverage['decision_active_slices_per_state']}`",
        f"- discard active slices per state: `{coverage['discard_active_slices_per_state']}`",
        "- highest-coverage decision slices:",
    ]
    for row in decision_top:
        lines.append(f"  - `{row['slice']}`: `{row['rate']:.4%}`")
    lines.append("- rarest decision slices:")
    for row in decision_rare:
        lines.append(f"  - `{row['slice']}`: `{row['rate']:.4%}`")
    lines.append("- highest-coverage discard slices:")
    for row in discard_top:
        lines.append(f"  - `{row['slice']}`: `{row['rate']:.4%}`")
    lines.append("- rarest discard slices:")
    for row in discard_rare:
        lines.append(f"  - `{row['slice']}`: `{row['rate']:.4%}`")

    lines.extend([
        "",
        "## Noise",
        "",
        f"- policy abs diff: `{policy_noise}`",
        f"- old_regression abs diff: `{old_noise}`",
        f"- recommended policy epsilon: `{report['recommended_runtime_constants']['policy_loss_epsilon']:.4f}`",
        f"- recommended old_regression epsilon: `{report['recommended_runtime_constants']['old_regression_policy_loss_epsilon']:.4f}`",
        "",
        "## Scenario Factor",
        "",
        f"- replaceable_by_statistics: `{factor['replaceable_by_statistics']}`",
        f"- recommended_value: `{factor['recommended_value']}`",
        f"- runtime_selection_scenario_factor: `{SELECTION_SCENARIO_FACTOR}`",
        f"- best_dense_band: `{factor['best_dense_band']}`",
        f"- best_aggregate_winner_match_rate: `{factor['best_aggregate_winner_match_rate']}`",
        f"- current_aggregate_winner_match_rate: `{factor['current_aggregate_winner_match_rate']}`",
        f"- best_pairwise_agreement: `{factor['best_pairwise_agreement']}`",
        f"- current_pairwise_agreement: `{factor['current_pairwise_agreement']}`",
        f"- legacy_pairwise_agreement_at_0_5: `{factor['legacy_pairwise_agreement_at_0_5']}`",
        f"- legacy_aggregate_winner_match_rate_at_0_5: `{factor['legacy_aggregate_winner_match_rate_at_0_5']}`",
        f"- note: {factor['reason']}",
        "",
        "## Component Decisions",
        "",
    ])
    for item in report["selector_components"]:
        lines.append(
            f"- `{item['component']}`: `{item['status']}`; replaceable=`{item['replaceable_by_statistics']}`; {item['reason']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--fidelity-root", type=Path, default=DEFAULT_FIDELITY_ROOT)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--per-year", type=int, default=180)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--file-batch-size", type=int, default=4)
    parser.add_argument("--scenario-factor-min", type=float, default=DEFAULT_SCENARIO_FACTOR_MIN)
    parser.add_argument("--scenario-factor-max", type=float, default=DEFAULT_SCENARIO_FACTOR_MAX)
    parser.add_argument("--scenario-factor-step", type=float, default=DEFAULT_SCENARIO_FACTOR_STEP)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    report = build_report(
        dataset_root=args.dataset_root,
        fidelity_root=args.fidelity_root,
        seed=args.seed,
        per_year=args.per_year,
        max_files=args.max_files,
        file_batch_size=args.file_batch_size,
        scenario_factor_min=args.scenario_factor_min,
        scenario_factor_max=args.scenario_factor_max,
        scenario_factor_step=args.scenario_factor_step,
    )
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")

    print(args.report_json)
    print(args.report_md)


if __name__ == "__main__":
    main()

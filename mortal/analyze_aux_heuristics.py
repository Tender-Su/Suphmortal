from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from libriichi.dataset import GameplayLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(r"D:/mahjong_data/dataset_json")
DEFAULT_SAMPLE_PATHS = REPO_ROOT / "logs" / "turn_weight_sample_paths.txt"
DEFAULT_FIDELITY_DIR = REPO_ROOT / "logs" / "stage05_fidelity" / "s05_fidelity_main"
DEFAULT_REPORT_JSON = REPO_ROOT / "logs" / "aux_heuristic_audit.json"
DEFAULT_REPORT_MD = REPO_ROOT / "logs" / "aux_heuristic_audit.md"

DANGER_VALUE_CAP = 96000.0
DANGER_VALUE_CAP_LOG = math.log1p(DANGER_VALUE_CAP)

TURN_BUCKETS = (
    ("early", 0, 4),
    ("mid", 5, 11),
    ("late", 12, 99),
)
GAP_BUCKETS = (
    ("gap_le_1k", 0, 1000),
    ("gap_1k_2k", 1000, 2000),
    ("gap_2k_4k", 2000, 4000),
    ("gap_4k_8k", 4000, 8000),
    ("gap_8k_12k", 8000, 12000),
    ("gap_gt_12k", 12000, 10**9),
)
GAP_FOCUS_CANDIDATES = (2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000)


def positive_or_none(values: list[float]) -> float | None:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return median(positives)


def entropy_from_counter(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob)
    return entropy


def binary_entropy(pos_count: int, total_count: int) -> float:
    if total_count <= 0 or pos_count <= 0 or pos_count >= total_count:
        return 0.0
    prob = pos_count / total_count
    return -(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob))


def smooth_l1_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    if target.numel() <= 0:
        return 0.0
    prediction = torch.full_like(target, float(pred))
    return float(F.smooth_l1_loss(prediction, target, reduction="mean").item())


@dataclass
class RoundMetrics:
    opponent_runs: list[dict[str, Any]]
    danger_runs: list[dict[str, Any]]


class DatasetAudit:
    def __init__(self) -> None:
        self.sample_files = 0
        self.sample_states = 0

        self.rank_total = 0
        self.rank_final_match = 0
        self.rank_round_counts = Counter[str]()
        self.rank_round_match = Counter[str]()
        self.rank_all_last_counts = Counter[str]()
        self.rank_all_last_match = Counter[str]()
        self.rank_turn_counts = Counter[str]()
        self.rank_turn_match = Counter[str]()
        self.rank_gap_counts = Counter[str]()
        self.rank_gap_match = Counter[str]()
        self.rank_gap_by_bucket_sum = Counter[str]()
        self.rank_gap_by_bucket_match = Counter[str]()
        self.rank_gap_by_bucket_count = Counter[str]()

        self.opponent_shanten_counter = Counter[int]()
        self.opponent_tenpai_pos = 0
        self.opponent_tenpai_total = 0

        self.danger_any_pos = 0
        self.danger_any_total = 0
        self.danger_player_pos = 0
        self.danger_player_total = 0
        self.danger_positive_values: list[np.ndarray] = []

    def observe_game(self, game: Any) -> None:
        ctx = np.asarray(game.take_context_meta_batch())
        masks = np.asarray(game.take_masks_batch(), dtype=bool)
        opponent_shanten = np.asarray(game.take_opponent_shanten_batch(), dtype=np.int64)
        opponent_tenpai = np.asarray(game.take_opponent_tenpai_batch(), dtype=np.int64)
        danger_valid = np.asarray(game.take_danger_valid_batch(), dtype=bool)
        danger_any = np.asarray(game.take_danger_any_batch(), dtype=bool)
        danger_value = np.asarray(game.take_danger_value_batch(), dtype=np.float32)
        danger_player_mask = np.asarray(game.take_danger_player_mask_batch(), dtype=bool)

        grp = game.take_grp()
        player_id = int(game.take_player_id())
        final_rank = int(grp.take_rank_by_player()[player_id])
        self_rank = ctx[:, 4].astype(np.int64)
        match = self_rank == final_rank

        self.sample_states += int(ctx.shape[0])
        self.rank_total += int(ctx.shape[0])
        self.rank_final_match += int(match.sum())

        round_stage = ctx[:, 1].astype(np.int64)
        is_all_last = ctx[:, 3].astype(bool)
        at_turn = ctx[:, 0].astype(np.int64)
        nearest_gap = np.minimum(ctx[:, 6], ctx[:, 7]).astype(np.int64) * 100

        east_mask = round_stage == 0
        south_mask = round_stage == 1
        self._update_mask_counter(self.rank_round_counts, self.rank_round_match, "east", east_mask, match)
        self._update_mask_counter(self.rank_round_counts, self.rank_round_match, "southplus", south_mask, match)
        self._update_mask_counter(self.rank_all_last_counts, self.rank_all_last_match, "all_last_yes", is_all_last, match)
        self._update_mask_counter(self.rank_all_last_counts, self.rank_all_last_match, "all_last_no", ~is_all_last, match)

        for bucket_name, lo, hi in TURN_BUCKETS:
            bucket_mask = (at_turn >= lo) & (at_turn <= hi)
            self._update_mask_counter(self.rank_turn_counts, self.rank_turn_match, bucket_name, bucket_mask, match)

        for gap_value, matched in zip(nearest_gap.tolist(), match.tolist(), strict=False):
            bucket = self._gap_bucket_name(int(gap_value))
            self.rank_gap_by_bucket_sum[bucket] += int(gap_value)
            self.rank_gap_by_bucket_match[bucket] += int(matched)
            self.rank_gap_by_bucket_count[bucket] += 1
        for bucket_name, lo, hi in GAP_BUCKETS:
            bucket_mask = (nearest_gap >= lo) & (nearest_gap < hi)
            self._update_mask_counter(self.rank_gap_counts, self.rank_gap_match, bucket_name, bucket_mask, match)

        self.opponent_shanten_counter.update(int(v) for v in opponent_shanten.reshape(-1))
        self.opponent_tenpai_pos += int(opponent_tenpai.sum())
        self.opponent_tenpai_total += int(opponent_tenpai.size)

        eligible = danger_valid[:, None] & masks[:, :37]
        eligible_player = eligible[:, :, None]
        any_pos = danger_any & eligible
        self.danger_any_pos += int(any_pos.sum())
        self.danger_any_total += int(eligible.sum())
        self.danger_player_pos += int((danger_player_mask & eligible_player).sum())
        self.danger_player_total += int(eligible_player.sum())

        positive_value = danger_value[any_pos]
        if positive_value.size > 0:
            normalized = np.log1p(np.clip(positive_value, 0.0, DANGER_VALUE_CAP)) / DANGER_VALUE_CAP_LOG
            self.danger_positive_values.append(normalized)

    @staticmethod
    def _update_mask_counter(
        count_counter: Counter[str],
        match_counter: Counter[str],
        name: str,
        mask: np.ndarray,
        match: np.ndarray,
    ) -> None:
        count_counter[name] += int(mask.sum())
        match_counter[name] += int(match[mask].sum())

    @staticmethod
    def _gap_bucket_name(value: int) -> str:
        for name, lo, hi in GAP_BUCKETS:
            if lo <= value < hi:
                return name
        return GAP_BUCKETS[-1][0]

    def rank_summary(self) -> dict[str, Any]:
        overall_match = self.rank_final_match / self.rank_total
        overall_change = 1.0 - overall_match

        def rate(counter: Counter[str], match_counter: Counter[str], key: str) -> dict[str, float]:
            count = counter[key]
            matched = match_counter[key]
            match_rate = matched / count if count else 0.0
            return {
                "count": count,
                "match_rate": match_rate,
                "change_rate": 1.0 - match_rate,
            }

        east = rate(self.rank_round_counts, self.rank_round_match, "east")
        south = rate(self.rank_round_counts, self.rank_round_match, "southplus")
        all_last_yes = rate(self.rank_all_last_counts, self.rank_all_last_match, "all_last_yes")
        all_last_no = rate(self.rank_all_last_counts, self.rank_all_last_match, "all_last_no")

        gap_rows = []
        gap_match_points: list[tuple[float, float, int]] = []
        gap_change_points: list[tuple[float, float, int]] = []
        for bucket_name, _, _ in GAP_BUCKETS:
            bucket_count = self.rank_gap_by_bucket_count[bucket_name]
            if bucket_count <= 0:
                continue
            avg_gap = self.rank_gap_by_bucket_sum[bucket_name] / bucket_count
            match_rate = self.rank_gap_by_bucket_match[bucket_name] / bucket_count
            change_rate = 1.0 - match_rate
            gap_rows.append(
                {
                    "bucket": bucket_name,
                    "count": bucket_count,
                    "avg_nearest_gap": avg_gap,
                    "match_rate": match_rate,
                    "change_rate": change_rate,
                }
            )
            gap_match_points.append((avg_gap, match_rate, bucket_count))
            gap_change_points.append((avg_gap, change_rate, bucket_count))

        gap_fit_match = fit_rank_gap_shape(gap_match_points, overall_match, rate_mode="match_rate")
        gap_fit_change = fit_rank_gap_shape(gap_change_points, overall_change, rate_mode="change_rate")

        turn_rows = {}
        for bucket_name, _, _ in TURN_BUCKETS:
            turn_rows[bucket_name] = rate(self.rank_turn_counts, self.rank_turn_match, bucket_name)

        south_factor = (south["match_rate"] / east["match_rate"]) if east["match_rate"] > 0 else 1.0
        all_last_factor = (
            all_last_yes["match_rate"] / all_last_no["match_rate"]
            if all_last_no["match_rate"] > 0 else 1.0
        )

        return {
            "sample_states": self.sample_states,
            "overall_match_rate": overall_match,
            "overall_change_rate": overall_change,
            "round_stage": {
                "east": east,
                "southplus": south,
                "suggested_south_factor_from_match_ratio": south_factor,
            },
            "all_last": {
                "yes": all_last_yes,
                "no": all_last_no,
                "suggested_all_last_factor_from_match_ratio": all_last_factor,
            },
            "turn_buckets": turn_rows,
            "nearest_gap_bins": gap_rows,
            "gap_shape_fit_match_rate": gap_fit_match,
            "gap_shape_fit_change_rate": gap_fit_change,
            "notes": [
                "south/all-last suggestions below use final-rank match-rate proxies; gap fitting is reported for both match-rate and change-rate proxies",
                "because the current repo does not contain an isolated sweep over rank-shape templates, these values should be treated as evidence-backed proxies rather than settled winners",
            ],
        }

    def opponent_summary(self, opponent_runs: list[dict[str, Any]]) -> dict[str, Any]:
        shanten_entropy = entropy_from_counter(self.opponent_shanten_counter)
        tenpai_entropy = binary_entropy(self.opponent_tenpai_pos, self.opponent_tenpai_total)

        shanten_losses = [row["opponent_shanten_loss"] for row in opponent_runs if row["opponent_shanten_loss"] > 0]
        tenpai_losses = [row["opponent_tenpai_loss"] for row in opponent_runs if row["opponent_tenpai_loss"] > 0]
        shanten_loss_median = positive_or_none(shanten_losses) or 0.0
        tenpai_loss_median = positive_or_none(tenpai_losses) or 0.0

        shanten_norm = (shanten_loss_median / shanten_entropy) if shanten_entropy > 0 else 0.0
        tenpai_norm = (tenpai_loss_median / tenpai_entropy) if tenpai_entropy > 0 else 0.0

        suggested = suggest_internal_mix(
            {
                "shanten": shanten_norm,
                "tenpai": tenpai_norm,
            }
        )

        return {
            "label_stats": {
                "shanten_counts": dict(sorted(self.opponent_shanten_counter.items())),
                "shanten_entropy_nats": shanten_entropy,
                "tenpai_positive_rate": (
                    self.opponent_tenpai_pos / self.opponent_tenpai_total
                    if self.opponent_tenpai_total > 0 else 0.0
                ),
                "tenpai_entropy_nats": tenpai_entropy,
            },
            "experiment_medians": {
                "run_count": len(opponent_runs),
                "opponent_aux_loss": positive_or_none([row["opponent_aux_loss"] for row in opponent_runs]),
                "opponent_shanten_loss": shanten_loss_median,
                "opponent_tenpai_loss": tenpai_loss_median,
                "opponent_shanten_macro_acc": positive_or_none([row["opponent_shanten_macro_acc"] for row in opponent_runs]),
                "opponent_tenpai_macro_acc": positive_or_none([row["opponent_tenpai_macro_acc"] for row in opponent_runs]),
            },
            "normalized_scale": {
                "shanten_loss_over_entropy": shanten_norm,
                "tenpai_loss_over_entropy": tenpai_norm,
            },
            "suggested_internal_mix_from_entropy_normalized_loss": suggested,
            "notes": [
                "suggested mix equalizes entropy-normalized subtask loss scales across observed opponent-enabled runs",
                "if both normalized scales are already close, keeping 1:1 is evidence-compatible",
            ],
        }

    def danger_summary(self, danger_runs: list[dict[str, Any]]) -> dict[str, Any]:
        any_entropy = binary_entropy(self.danger_any_pos, self.danger_any_total)
        player_entropy = binary_entropy(self.danger_player_pos, self.danger_player_total)

        value_targets = (
            np.concatenate(self.danger_positive_values, axis=0)
            if self.danger_positive_values else np.empty((0,), dtype=np.float32)
        )
        if value_targets.size > 0:
            value_target_tensor = torch.from_numpy(value_targets.astype(np.float32))
            value_baseline = smooth_l1_mean(float(value_target_tensor.mean().item()), value_target_tensor)
            value_mean = float(value_target_tensor.mean().item())
            value_std = float(value_target_tensor.std(unbiased=False).item())
        else:
            value_baseline = 0.0
            value_mean = 0.0
            value_std = 0.0

        any_loss_median = positive_or_none([row["danger_any_loss"] for row in danger_runs]) or 0.0
        value_loss_median = positive_or_none([row["danger_value_loss"] for row in danger_runs]) or 0.0
        player_loss_median = positive_or_none([row["danger_player_loss"] for row in danger_runs]) or 0.0

        any_norm = (any_loss_median / any_entropy) if any_entropy > 0 else 0.0
        value_norm = (value_loss_median / value_baseline) if value_baseline > 0 else 0.0
        player_norm = (player_loss_median / player_entropy) if player_entropy > 0 else 0.0

        suggested = suggest_internal_mix(
            {
                "any": any_norm,
                "value": value_norm,
                "player": player_norm,
            }
        )

        return {
            "label_stats": {
                "danger_any_positive_rate": (self.danger_any_pos / self.danger_any_total) if self.danger_any_total > 0 else 0.0,
                "danger_any_entropy_nats": any_entropy,
                "danger_player_positive_rate": (self.danger_player_pos / self.danger_player_total) if self.danger_player_total > 0 else 0.0,
                "danger_player_entropy_nats": player_entropy,
                "danger_value_positive_count": int(value_targets.size),
                "danger_value_mean_log_target": value_mean,
                "danger_value_std_log_target": value_std,
                "danger_value_smooth_l1_constant_baseline": value_baseline,
            },
            "experiment_medians": {
                "run_count": len(danger_runs),
                "danger_aux_loss": positive_or_none([row["danger_aux_loss"] for row in danger_runs]),
                "danger_any_loss": any_loss_median,
                "danger_value_loss": value_loss_median,
                "danger_player_loss": player_loss_median,
            },
            "normalized_scale": {
                "danger_any_loss_over_entropy": any_norm,
                "danger_value_loss_over_constant_baseline": value_norm,
                "danger_player_loss_over_entropy": player_norm,
            },
            "suggested_internal_mix_from_normalized_loss": suggested,
            "notes": [
                "the value branch uses smooth-l1 on the log-capped target, so its normalization uses a constant-predictor baseline rather than entropy",
                "suggested mix equalizes normalized subtask scales; it does not prove this is the downstream-optimal playing-strength mix",
            ],
        }


def fit_rank_gap_shape(
    gap_points: list[tuple[float, float, int]],
    overall_rate: float,
    *,
    rate_mode: str,
) -> dict[str, Any]:
    if not gap_points or overall_rate <= 0:
        return {
            "rate_mode": rate_mode,
            "focus_points_candidates": list(GAP_FOCUS_CANDIDATES),
            "best_focus_points": None,
            "best_gap_close_bonus": None,
            "weighted_mse": None,
        }

    far_baseline_points = [point for point in gap_points if point[0] >= 8000]
    if far_baseline_points:
        far_baseline = sum(rate * count for _, rate, count in far_baseline_points) / sum(count for _, _, count in far_baseline_points)
    else:
        far_baseline = overall_rate
    if far_baseline <= 0:
        far_baseline = overall_rate

    best: dict[str, Any] | None = None
    for focus in GAP_FOCUS_CANDIDATES:
        numer = 0.0
        denom = 0.0
        rows = []
        for avg_gap, match_rate, count in gap_points:
            closeness = max(1.0 - avg_gap / focus, 0.0)
            target_ratio = match_rate / far_baseline if far_baseline > 0 else 1.0
            rows.append((avg_gap, closeness, target_ratio, count))
            numer += count * closeness * (target_ratio - 1.0)
            denom += count * closeness * closeness
        bonus = max(numer / denom, 0.0) if denom > 0 else 0.0

        weighted_sq = 0.0
        total_count = 0
        for avg_gap, closeness, target_ratio, count in rows:
            pred = 1.0 + bonus * closeness
            weighted_sq += count * (pred - target_ratio) ** 2
            total_count += count
        mse = weighted_sq / total_count if total_count > 0 else 0.0
        candidate = {
            "best_focus_points": focus,
            "best_gap_close_bonus": bonus,
            "far_gap_match_rate_baseline": far_baseline,
            "weighted_mse": mse,
        }
        if best is None or mse < best["weighted_mse"]:
            best = candidate

    assert best is not None
    return {
        "rate_mode": rate_mode,
        "focus_points_candidates": list(GAP_FOCUS_CANDIDATES),
        **best,
    }


def suggest_internal_mix(scale_by_task: dict[str, float]) -> dict[str, Any]:
    inverse = {}
    for name, value in scale_by_task.items():
        inverse[name] = 0.0 if value <= 0 else 1.0 / value
    total = sum(inverse.values())
    if total <= 0:
        return {
            "weights": {name: 0.0 for name in scale_by_task},
            "criterion": "inverse_normalized_scale",
        }
    return {
        "weights": {name: value / total for name, value in inverse.items()},
        "criterion": "inverse_normalized_scale",
    }


def collect_sample_paths(
    dataset_root: Path,
    sample_paths_file: Path | None,
    seed: int,
    per_year: int,
    max_files: int,
) -> list[str]:
    if sample_paths_file and sample_paths_file.exists():
        paths = [line.strip() for line in sample_paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if paths:
            return paths[:max_files] if max_files > 0 else paths

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
            f"[audit] loading files {start + 1}-{start + len(batch_paths)} / {len(paths)}",
            flush=True,
        )
        loader = GameplayLoader(
            version=4,
            oracle=False,
            track_opponent_states=True,
            track_danger_labels=True,
        )
        try:
            nested = loader.load_log_files(batch_paths)
        except Exception as exc:
            print(
                f"[audit] batch failed, falling back to per-file scan: {exc}",
                flush=True,
            )
            nested = []
            for path in batch_paths:
                try:
                    nested.extend(loader.load_log_files([path]))
                except Exception as inner_exc:
                    print(
                        f"[audit] skip bad file: {path} :: {inner_exc}",
                        flush=True,
                    )
        for games in nested:
            for game in games:
                yield game


def collect_dataset_audit(paths: list[str], *, file_batch_size: int) -> DatasetAudit:
    audit = DatasetAudit()
    audit.sample_files = len(paths)
    for game in iter_games(paths, file_batch_size):
        audit.observe_game(game)
    return audit


def parse_round_metrics(fidelity_dir: Path) -> RoundMetrics:
    opponent_runs: list[dict[str, Any]] = []
    danger_runs: list[dict[str, Any]] = []
    round_files = [
        fidelity_dir / "p1_calibration.json",
        fidelity_dir / "p1_protocol_decide_round.json",
        fidelity_dir / "p1_winner_refine_round.json",
        fidelity_dir / "p1_ablation_round.json",
        # Historical fallback only.
        fidelity_dir / "p1_solo_round.json",
    ]
    seen_round_arms: set[tuple[str, str]] = set()
    for round_file in round_files:
        if not round_file.exists():
            continue
        data = json.loads(round_file.read_text(encoding="utf-8"))
        round_name = str(data.get("round_name", round_file.stem))
        ranking = data.get("ranking", [])
        for entry in ranking:
            arm_name = entry.get("arm_name")
            metrics = entry.get("full_recent_metrics", {})
            round_arm_key = (round_name, str(arm_name))
            if not arm_name or round_arm_key in seen_round_arms:
                continue
            seen_round_arms.add(round_arm_key)
            if metrics.get("opponent_aux_loss", 0.0) > 0:
                opponent_runs.append(
                    {
                        "round_name": round_name,
                        "arm_name": arm_name,
                        "opponent_aux_loss": float(metrics.get("opponent_aux_loss", 0.0)),
                        "opponent_shanten_loss": float(metrics.get("opponent_shanten_loss", 0.0)),
                        "opponent_tenpai_loss": float(metrics.get("opponent_tenpai_loss", 0.0)),
                        "opponent_shanten_macro_acc": float(metrics.get("opponent_shanten_macro_acc", 0.0)),
                        "opponent_tenpai_macro_acc": float(metrics.get("opponent_tenpai_macro_acc", 0.0)),
                    }
                )
            if metrics.get("danger_aux_loss", 0.0) > 0:
                danger_runs.append(
                    {
                        "round_name": round_name,
                        "arm_name": arm_name,
                        "danger_aux_loss": float(metrics.get("danger_aux_loss", 0.0)),
                        "danger_any_loss": float(metrics.get("danger_any_loss", 0.0)),
                        "danger_value_loss": float(metrics.get("danger_value_loss", 0.0)),
                        "danger_player_loss": float(metrics.get("danger_player_loss", 0.0)),
                    }
                )
    return RoundMetrics(opponent_runs=opponent_runs, danger_runs=danger_runs)


def build_report(
    dataset_root: Path,
    sample_paths_file: Path | None,
    fidelity_dir: Path,
    *,
    seed: int,
    per_year: int,
    max_files: int,
    file_batch_size: int,
) -> dict[str, Any]:
    sample_paths = collect_sample_paths(dataset_root, sample_paths_file, seed, per_year, max_files)
    dataset_audit = collect_dataset_audit(sample_paths, file_batch_size=file_batch_size)
    round_metrics = parse_round_metrics(fidelity_dir)

    report = {
        "dataset_root": str(dataset_root),
        "sample_paths_file": str(sample_paths_file) if sample_paths_file else None,
        "sample_files": dataset_audit.sample_files,
        "sample_states": dataset_audit.sample_states,
        "fidelity_dir": str(fidelity_dir),
        "rank": dataset_audit.rank_summary(),
        "opponent_state": dataset_audit.opponent_summary(round_metrics.opponent_runs),
        "danger": dataset_audit.danger_summary(round_metrics.danger_runs),
        "unresolved": [
            {
                "parameter": "rank_aux.base_weight / max_weight final operating point",
                "reason": "data can support proxy shape fitting, but the final budget target still needs downstream A/B or a policy-budget criterion",
            },
            {
                "parameter": "danger_ramp_steps",
                "reason": "current repo contains danger-enabled runs with ramp=1000, but no controlled sweep over ramp values; the exact numeric choice remains an engineering heuristic",
            },
            {
                "parameter": "rank south/all-last/gap exact multipliers as final winners",
                "reason": "current repo has proxy statistics but no isolated rank-shape A/B sweep to promote one exact template from evidence-backed proxy to settled winner",
            },
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rank = report["rank"]
    opp = report["opponent_state"]
    danger = report["danger"]

    lines = [
        "# Auxiliary Heuristic Audit",
        "",
        f"- dataset_root: `{report['dataset_root']}`",
        f"- sample_files: `{report['sample_files']}`",
        f"- sample_states: `{report['sample_states']}`",
        f"- fidelity_dir: `{report['fidelity_dir']}`",
        "",
        "## Rank",
        "",
        f"- overall final-rank match rate: `{rank['overall_match_rate']:.4f}`",
        f"- suggested south factor from match-ratio proxy: `{rank['round_stage']['suggested_south_factor_from_match_ratio']:.3f}`",
        f"- suggested all-last factor from match-ratio proxy: `{rank['all_last']['suggested_all_last_factor_from_match_ratio']:.3f}`",
        f"- fitted gap focus points (match proxy): `{rank['gap_shape_fit_match_rate']['best_focus_points']}`",
        f"- fitted gap close bonus (match proxy): `{rank['gap_shape_fit_match_rate']['best_gap_close_bonus']:.3f}`",
        f"- fitted gap focus points (change proxy): `{rank['gap_shape_fit_change_rate']['best_focus_points']}`",
        f"- fitted gap close bonus (change proxy): `{rank['gap_shape_fit_change_rate']['best_gap_close_bonus']:.3f}`",
        "",
        "## Opponent State",
        "",
        f"- shanten entropy: `{opp['label_stats']['shanten_entropy_nats']:.4f}`",
        f"- tenpai entropy: `{opp['label_stats']['tenpai_entropy_nats']:.4f}`",
        f"- shanten median loss: `{opp['experiment_medians']['opponent_shanten_loss']:.4f}`",
        f"- tenpai median loss: `{opp['experiment_medians']['opponent_tenpai_loss']:.4f}`",
        f"- suggested internal mix: `{json.dumps(opp['suggested_internal_mix_from_entropy_normalized_loss']['weights'], ensure_ascii=False)}`",
        "",
        "## Danger",
        "",
        f"- any entropy: `{danger['label_stats']['danger_any_entropy_nats']:.6f}`",
        f"- player entropy: `{danger['label_stats']['danger_player_entropy_nats']:.6f}`",
        f"- value constant-baseline smooth-l1: `{danger['label_stats']['danger_value_smooth_l1_constant_baseline']:.6f}`",
        f"- any median loss: `{danger['experiment_medians']['danger_any_loss']:.6f}`",
        f"- value median loss: `{danger['experiment_medians']['danger_value_loss']:.6f}`",
        f"- player median loss: `{danger['experiment_medians']['danger_player_loss']:.6f}`",
        f"- suggested internal mix: `{json.dumps(danger['suggested_internal_mix_from_normalized_loss']['weights'], ensure_ascii=False)}`",
        "",
        "## Unresolved",
        "",
    ]
    for item in report["unresolved"]:
        lines.append(f"- `{item['parameter']}`: {item['reason']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sample-paths-file", type=Path, default=DEFAULT_SAMPLE_PATHS)
    parser.add_argument("--fidelity-dir", type=Path, default=DEFAULT_FIDELITY_DIR)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--per-year", type=int, default=60)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--file-batch-size", type=int, default=16)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    report = build_report(
        args.dataset_root,
        args.sample_paths_file,
        args.fidelity_dir,
        seed=args.seed,
        per_year=args.per_year,
        max_files=args.max_files,
        file_batch_size=args.file_batch_size,
    )

    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")

    print(args.report_json)
    print(args.report_md)


if __name__ == "__main__":
    main()

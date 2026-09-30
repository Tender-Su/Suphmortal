from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


SCALAR_METRICS = (
    'loss',
    'policy_loss',
    'aux_loss',
    'rank_aux_raw_loss',
    'rank_aux_weight_mean',
    'opponent_turn_weight_mean',
    'danger_turn_weight_mean',
    'search_distill_loss',
    'search_teacher_gap',
    'search_active_fraction',
    'search_hard_fraction',
    'action_quality_score',
    'scenario_quality_score',
    'selection_quality_score',
    'action_acc',
    'macro_action_acc',
    'rank_acc',
    'discard_nll',
    'chi_exact_nll',
    'discard_top3_acc',
    'opponent_aux_loss',
    'danger_aux_loss',
    'danger_any_loss',
    'danger_value_loss',
    'danger_player_loss',
    'opponent_shanten_macro_acc',
    'opponent_tenpai_macro_acc',
)

DECISION_METRIC_SUFFIXES = (
    'balanced_acc',
    'balanced_bce',
    'pred_rate',
    'target_rate',
)


def write_metric_scalars(
    writer: Any,
    prefix: str,
    metrics: Mapping[str, Any],
    step: int,
    *,
    decision_metric_names: Sequence[str] = (),
) -> None:
    for key in SCALAR_METRICS:
        if key in metrics:
            writer.add_scalar(f'{prefix}/{key}', metrics[key], step)
    for name in decision_metric_names:
        for suffix in DECISION_METRIC_SUFFIXES:
            key = f'{name}_{suffix}'
            if key in metrics:
                writer.add_scalar(f'{prefix}/{key}', metrics[key], step)


class ClusterMetricAccumulator:
    """Aggregate sample-level metrics into deterministic per-game records."""

    __slots__ = ('_totals',)

    def __init__(self) -> None:
        self._totals: dict[str, dict[int, tuple[float, int]]] = {}

    def merge(
        self,
        source: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        if source is None:
            return
        for metric_name, (game_ids, values) in source.items():
            ids = game_ids.reshape(-1)
            metric_values = values.reshape(-1)
            if ids.numel() != metric_values.numel():
                raise ValueError(
                    f'cluster metric {metric_name!r} has mismatched ids and values'
                )
            unique_ids, inverse, counts = torch.unique(
                ids,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            sums = torch.zeros(
                unique_ids.numel(),
                dtype=torch.float64,
                device=metric_values.device,
            )
            sums.scatter_add_(0, inverse, metric_values.to(dtype=torch.float64))

            metric_totals = self._totals.setdefault(metric_name, {})
            for game_id, value_sum, count in zip(
                unique_ids.tolist(),
                sums.tolist(),
                counts.tolist(),
            ):
                previous_sum, previous_count = metric_totals.get(
                    int(game_id),
                    (0.0, 0),
                )
                metric_totals[int(game_id)] = (
                    previous_sum + float(value_sum),
                    previous_count + int(count),
                )

    def records(self) -> dict[str, list[list[float | int]]]:
        return {
            metric_name: [
                [game_id, value_sum, count]
                for game_id, (value_sum, count) in sorted(records.items())
            ]
            for metric_name, records in self._totals.items()
        }

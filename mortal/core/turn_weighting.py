from __future__ import annotations

from typing import Any

import torch


def resolve_turn_weighting_cfg(
    raw_cfg: Any,
    *,
    default_early_factor: float,
    default_mid_factor: float,
    default_late_factor: float,
    default_early_max_turn: int = 6,
    default_late_min_turn: int = 13,
) -> dict[str, float | int]:
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    early_max_turn = raw_cfg.get('early_max_turn', default_early_max_turn)
    late_min_turn = raw_cfg.get('late_min_turn', default_late_min_turn)
    early_factor = raw_cfg.get('early_factor', default_early_factor)
    mid_factor = raw_cfg.get('mid_factor', default_mid_factor)
    late_factor = raw_cfg.get('late_factor', default_late_factor)
    if early_max_turn is None:
        early_max_turn = default_early_max_turn
    if late_min_turn is None:
        late_min_turn = default_late_min_turn
    if early_factor is None:
        early_factor = default_early_factor
    if mid_factor is None:
        mid_factor = default_mid_factor
    if late_factor is None:
        late_factor = default_late_factor
    early_max_turn = max(int(early_max_turn), 0)
    late_min_turn = max(int(late_min_turn), early_max_turn + 1)
    return {
        'early_factor': max(float(early_factor), 0.0),
        'mid_factor': max(float(mid_factor), 0.0),
        'late_factor': max(float(late_factor), 0.0),
        'early_max_turn': early_max_turn,
        'late_min_turn': late_min_turn,
    }


def compute_turn_bucket_weights(
    at_turn: Any,
    *,
    early_factor: float,
    mid_factor: float,
    late_factor: float,
    early_max_turn: int = 6,
    late_min_turn: int = 13,
) -> torch.Tensor:
    if not torch.is_tensor(at_turn):
        at_turn = torch.as_tensor(at_turn, dtype=torch.int64)
    else:
        at_turn = at_turn.to(dtype=torch.int64)
    weights = torch.full(
        at_turn.shape,
        float(mid_factor),
        dtype=torch.float32,
        device=at_turn.device,
    )
    weights = torch.where(
        at_turn <= int(early_max_turn),
        torch.full_like(weights, float(early_factor)),
        weights,
    )
    weights = torch.where(
        at_turn >= int(late_min_turn),
        torch.full_like(weights, float(late_factor)),
        weights,
    )
    return weights

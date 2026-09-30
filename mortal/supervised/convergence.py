from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONVERGENCE_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConvergenceConfig:
    core_optimizer_steps: int
    tail_lr_levels: tuple[float, ...]
    smoothing_checks: int = 5
    improvement_delta: float = 2e-4
    reduce_patience_steps: int = 160_000
    stop_patience_steps: int = 240_000
    min_level_steps: int = 80_000
    metric: str = 'policy_loss'

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ConvergenceConfig:
        levels = tuple(float(value) for value in raw.get('tail_lr_levels', ()))
        config = cls(
            core_optimizer_steps=int(raw.get('core_optimizer_steps') or 0),
            tail_lr_levels=levels,
            smoothing_checks=int(raw.get('smoothing_checks', 5)),
            improvement_delta=float(raw.get('improvement_delta', 2e-4)),
            reduce_patience_steps=int(raw.get('reduce_patience_steps', 160_000)),
            stop_patience_steps=int(raw.get('stop_patience_steps', 240_000)),
            min_level_steps=int(raw.get('min_level_steps', 80_000)),
            metric=str(raw.get('metric', 'policy_loss')),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.core_optimizer_steps <= 0:
            raise ValueError('convergence.core_optimizer_steps must be positive')
        if not self.tail_lr_levels:
            raise ValueError('convergence.tail_lr_levels must not be empty')
        if any(not math.isfinite(value) or value <= 0 for value in self.tail_lr_levels):
            raise ValueError('convergence.tail_lr_levels must contain positive finite values')
        if any(left <= right for left, right in zip(self.tail_lr_levels, self.tail_lr_levels[1:])):
            raise ValueError('convergence.tail_lr_levels must be strictly descending')
        if self.smoothing_checks <= 0:
            raise ValueError('convergence.smoothing_checks must be positive')
        if self.improvement_delta < 0:
            raise ValueError('convergence.improvement_delta must be non-negative')
        if self.reduce_patience_steps <= 0:
            raise ValueError('convergence.reduce_patience_steps must be positive')
        if self.stop_patience_steps < self.reduce_patience_steps:
            raise ValueError(
                'convergence.stop_patience_steps must be at least reduce_patience_steps'
            )
        if self.min_level_steps <= 0:
            raise ValueError('convergence.min_level_steps must be positive')
        if not self.metric:
            raise ValueError('convergence.metric must not be empty')


@dataclass(frozen=True)
class ConvergenceDecision:
    action: str
    state: dict[str, Any]
    smoothed_metric: float | None
    target_lr: float | None = None
    reason: str = ''


def initial_convergence_state(config: ConvergenceConfig) -> dict[str, Any]:
    return {
        'schema_version': CONVERGENCE_STATE_SCHEMA_VERSION,
        'metric': config.metric,
        'tail_started': False,
        'level_index': 0,
        'level_started_optimizer_step': None,
        'last_meaningful_improvement_optimizer_step': None,
        'best_smoothed_metric_at_level': math.inf,
        'best_raw_metric': math.inf,
        'best_raw_metric_optimizer_step': None,
        'recent_metrics': [],
        'converged': False,
        'converged_optimizer_step': None,
        'last_action': 'core',
        'last_reason': '',
    }


def normalize_convergence_state(
    raw: Mapping[str, Any] | None,
    config: ConvergenceConfig,
) -> dict[str, Any]:
    state = initial_convergence_state(config)
    if raw:
        state.update(dict(raw))
    if int(state.get('schema_version') or 0) != CONVERGENCE_STATE_SCHEMA_VERSION:
        raise ValueError(
            'unsupported convergence state schema: '
            f'{state.get("schema_version")!r}'
        )
    if state.get('metric') != config.metric:
        raise ValueError(
            'convergence state metric mismatch: '
            f'{state.get("metric")!r} != {config.metric!r}'
        )
    level_index = int(state.get('level_index') or 0)
    if not 0 <= level_index < len(config.tail_lr_levels):
        raise ValueError(f'invalid convergence level_index: {level_index}')
    state['level_index'] = level_index
    state['recent_metrics'] = [
        [int(step), float(value)]
        for step, value in list(state.get('recent_metrics') or [])[-config.smoothing_checks :]
    ]
    return state


def _append_metric(
    state: dict[str, Any],
    *,
    optimizer_steps: int,
    metric_value: float,
    smoothing_checks: int,
) -> float | None:
    recent = list(state.get('recent_metrics') or [])
    recent.append([optimizer_steps, metric_value])
    recent = recent[-smoothing_checks:]
    state['recent_metrics'] = recent
    if len(recent) < smoothing_checks:
        return None
    return float(statistics.median(value for _, value in recent))


def observe_convergence(
    raw_state: Mapping[str, Any] | None,
    config: ConvergenceConfig,
    *,
    optimizer_steps: int,
    metric_value: float,
) -> ConvergenceDecision:
    if optimizer_steps < 0:
        raise ValueError('optimizer_steps must be non-negative')
    if not math.isfinite(metric_value):
        raise ValueError(f'convergence metric must be finite, got {metric_value!r}')

    state = normalize_convergence_state(raw_state, config)
    if state['converged']:
        return ConvergenceDecision(
            action='stop',
            state=state,
            smoothed_metric=None,
            target_lr=config.tail_lr_levels[state['level_index']],
            reason=str(state.get('last_reason') or 'already converged'),
        )

    if metric_value < float(state.get('best_raw_metric', math.inf)):
        state['best_raw_metric'] = metric_value
        state['best_raw_metric_optimizer_step'] = optimizer_steps

    if optimizer_steps < config.core_optimizer_steps:
        state['recent_metrics'] = []
        state['last_action'] = 'core'
        state['last_reason'] = (
            f'core schedule active: {optimizer_steps:,}/'
            f'{config.core_optimizer_steps:,} optimizer steps'
        )
        return ConvergenceDecision(
            action='continue',
            state=state,
            smoothed_metric=None,
            reason=state['last_reason'],
        )

    if not state['tail_started']:
        state['tail_started'] = True
        state['level_index'] = 0
        state['level_started_optimizer_step'] = optimizer_steps
        state['last_meaningful_improvement_optimizer_step'] = optimizer_steps
        state['best_smoothed_metric_at_level'] = math.inf
        state['recent_metrics'] = []

    smoothed = _append_metric(
        state,
        optimizer_steps=optimizer_steps,
        metric_value=metric_value,
        smoothing_checks=config.smoothing_checks,
    )
    level_index = state['level_index']
    target_lr = config.tail_lr_levels[level_index]

    if smoothed is None:
        state['last_action'] = 'warm_tail_window'
        state['last_reason'] = (
            f'collecting tail smoothing window '
            f'{len(state["recent_metrics"])}/{config.smoothing_checks}'
        )
        return ConvergenceDecision(
            action='continue',
            state=state,
            smoothed_metric=None,
            target_lr=target_lr,
            reason=state['last_reason'],
        )

    best_smoothed = float(state.get('best_smoothed_metric_at_level', math.inf))
    if smoothed < best_smoothed - config.improvement_delta:
        state['best_smoothed_metric_at_level'] = smoothed
        state['last_meaningful_improvement_optimizer_step'] = optimizer_steps
        state['last_action'] = 'improved'
        state['last_reason'] = (
            f'{config.metric} improved to {smoothed:.6f} '
            f'at tail level {level_index}'
        )
        return ConvergenceDecision(
            action='continue',
            state=state,
            smoothed_metric=smoothed,
            target_lr=target_lr,
            reason=state['last_reason'],
        )

    level_started = int(state['level_started_optimizer_step'])
    last_improvement = int(state['last_meaningful_improvement_optimizer_step'])
    level_steps = optimizer_steps - level_started
    stale_steps = optimizer_steps - last_improvement
    at_final_level = level_index == len(config.tail_lr_levels) - 1
    patience_steps = (
        config.stop_patience_steps if at_final_level else config.reduce_patience_steps
    )

    if level_steps < config.min_level_steps or stale_steps < patience_steps:
        state['last_action'] = 'observe_tail'
        state['last_reason'] = (
            f'tail level {level_index}: level_steps={level_steps:,}, '
            f'stale_steps={stale_steps:,}, patience={patience_steps:,}'
        )
        return ConvergenceDecision(
            action='continue',
            state=state,
            smoothed_metric=smoothed,
            target_lr=target_lr,
            reason=state['last_reason'],
        )

    if not at_final_level:
        next_level = level_index + 1
        next_lr = config.tail_lr_levels[next_level]
        state['level_index'] = next_level
        state['level_started_optimizer_step'] = optimizer_steps
        state['last_meaningful_improvement_optimizer_step'] = optimizer_steps
        state['best_smoothed_metric_at_level'] = math.inf
        state['recent_metrics'] = []
        state['last_action'] = 'reduce_lr'
        state['last_reason'] = (
            f'{config.metric} plateaued for {stale_steps:,} optimizer steps; '
            f'advance tail lr {target_lr:.3e} -> {next_lr:.3e}'
        )
        return ConvergenceDecision(
            action='reduce_lr',
            state=state,
            smoothed_metric=smoothed,
            target_lr=next_lr,
            reason=state['last_reason'],
        )

    state['converged'] = True
    state['converged_optimizer_step'] = optimizer_steps
    state['last_action'] = 'stop'
    state['last_reason'] = (
        f'{config.metric} plateaued for {stale_steps:,} optimizer steps '
        f'at final tail lr {target_lr:.3e}'
    )
    return ConvergenceDecision(
        action='stop',
        state=state,
        smoothed_metric=smoothed,
        target_lr=target_lr,
        reason=state['last_reason'],
    )


def candidate_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_policy = float(left['policy_loss'])
    right_policy = float(right['policy_loss'])
    left_action = float(left['action_quality_score'])
    right_action = float(right['action_quality_score'])
    left_old_raw = left.get('old_regression_policy_loss')
    right_old_raw = right.get('old_regression_policy_loss')
    left_old = math.inf if left_old_raw is None else float(left_old_raw)
    right_old = math.inf if right_old_raw is None else float(right_old_raw)
    no_worse = (
        left_policy <= right_policy
        and left_action >= right_action
        and left_old <= right_old
    )
    strictly_better = (
        left_policy < right_policy
        or left_action > right_action
        or left_old < right_old
    )
    return no_worse and strictly_better


def select_bounded_pareto_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if limit < 3:
        raise ValueError('candidate portfolio limit must be at least three')

    ordered = [dict(candidate) for candidate in candidates]
    frontier = [
        candidate
        for index, candidate in enumerate(ordered)
        if not any(
            other_index != index and candidate_dominates(other, candidate)
            for other_index, other in enumerate(ordered)
        )
    ]
    frontier.sort(key=lambda item: (float(item['policy_loss']), -float(item['action_quality_score']), int(item['step'])))
    if len(frontier) <= limit:
        removed = [candidate for candidate in ordered if candidate not in frontier]
        return frontier, removed

    protected_indices = {
        min(range(len(frontier)), key=lambda idx: float(frontier[idx]['policy_loss'])),
        max(range(len(frontier)), key=lambda idx: float(frontier[idx]['action_quality_score'])),
        min(
            range(len(frontier)),
            key=lambda idx: (
                math.inf
                if frontier[idx].get('old_regression_policy_loss') is None
                else float(frontier[idx]['old_regression_policy_loss'])
            ),
        ),
    }
    selected_indices = set(protected_indices)
    remaining_slots = limit - len(selected_indices)
    unprotected = [idx for idx in range(len(frontier)) if idx not in selected_indices]
    if remaining_slots > 0 and unprotected:
        if remaining_slots >= len(unprotected):
            selected_indices.update(unprotected)
        elif remaining_slots == 1:
            selected_indices.add(unprotected[len(unprotected) // 2])
        else:
            for slot in range(remaining_slots):
                position = round(slot * (len(unprotected) - 1) / (remaining_slots - 1))
                selected_indices.add(unprotected[position])

    selected = [candidate for idx, candidate in enumerate(frontier) if idx in selected_indices]
    selected.sort(key=lambda item: (float(item['policy_loss']), -float(item['action_quality_score']), int(item['step'])))
    removed = [candidate for candidate in ordered if candidate not in selected]
    return selected, removed

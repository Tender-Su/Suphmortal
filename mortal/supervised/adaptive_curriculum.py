from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from mortal.core.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    AdaptiveCurriculumDecision,
    MetricSpec,
    inherit_adaptive_curriculum_baseline,
    initial_adaptive_curriculum_state,
    normalize_adaptive_curriculum_state,
    observe_adaptive_curriculum as observe_core_adaptive_curriculum,
    paired_cluster_summary,
)


def observe_adaptive_curriculum(
    raw_state: Mapping[str, Any] | None,
    config: AdaptiveCurriculumConfig,
    *,
    optimizer_steps: int,
    metrics: Mapping[str, Any],
    cluster_records: Mapping[str, Sequence[Any]],
) -> AdaptiveCurriculumDecision:
    """Apply the shared gate while preventing unproductive LR cascades."""
    previous = normalize_adaptive_curriculum_state(raw_state, config)
    previous_level = int(previous['lr_level_index'])
    previous_level_improved = bool(
        previous.get('lr_level_has_improved', previous_level == 0)
    )
    decision = observe_core_adaptive_curriculum(
        previous,
        config,
        optimizer_steps=optimizer_steps,
        metrics=metrics,
        cluster_records=cluster_records,
    )
    state = deepcopy(decision.state)

    if decision.action in {'continue', 'update_best'}:
        if state.get('best_step') is not None and (
            decision.action == 'update_best'
            or state.get('last_action') == 'set_baseline'
        ):
            state['lr_level_has_improved'] = True
        else:
            state.setdefault('lr_level_has_improved', previous_level_improved)
        return AdaptiveCurriculumDecision(
            action=decision.action,
            state=state,
            comparisons=decision.comparisons,
            target_lr=decision.target_lr,
            reason=decision.reason,
        )

    if decision.action != 'reduce_lr':
        state.setdefault('lr_level_has_improved', previous_level_improved)
        return AdaptiveCurriculumDecision(
            action=decision.action,
            state=state,
            comparisons=decision.comparisons,
            target_lr=decision.target_lr,
            reason=decision.reason,
        )

    if previous_level > 0 and not previous_level_improved:
        reason = (
            f'lr level {previous_level} failed to improve the paired phase-best; '
            'stop instead of cascading to a smaller learning rate'
        )
        state.update({
            'lr_level_index': previous_level,
            'lr_level_has_improved': False,
            'completed': True,
            'completed_step': optimizer_steps,
            'last_action': 'stop',
            'last_reason': reason,
        })
        history = list(state.get('history') or [])
        if history:
            history[-1] = {
                **history[-1],
                'action': 'stop',
                'reason': reason,
            }
            state['history'] = history
        return AdaptiveCurriculumDecision(
            action='stop',
            state=state,
            comparisons=decision.comparisons,
            target_lr=config.lr_levels[previous_level],
            reason=reason,
        )

    state['lr_level_has_improved'] = False
    state['lr_level_started_step'] = optimizer_steps
    return AdaptiveCurriculumDecision(
        action='reduce_lr',
        state=state,
        comparisons=decision.comparisons,
        target_lr=decision.target_lr,
        reason=decision.reason,
    )


__all__ = [
    'AdaptiveCurriculumConfig',
    'AdaptiveCurriculumDecision',
    'MetricSpec',
    'inherit_adaptive_curriculum_baseline',
    'initial_adaptive_curriculum_state',
    'normalize_adaptive_curriculum_state',
    'observe_adaptive_curriculum',
    'paired_cluster_summary',
]

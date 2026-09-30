from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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

    made_level_improvement = (
        decision.action == 'update_best'
        or state.get('last_action') == 'set_baseline'
    )
    if decision.action != 'reduce_lr':
        if made_level_improvement:
            state['lr_level_has_improved'] = True
        else:
            state.setdefault('lr_level_has_improved', previous_level_improved)
        return replace(decision, state=state)

    if previous_level == 0 or previous_level_improved:
        state['lr_level_has_improved'] = False
        state['lr_level_started_step'] = optimizer_steps
        return replace(decision, state=state)

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
    return replace(
        decision,
        action='stop',
        state=state,
        target_lr=config.lr_levels[previous_level],
        reason=reason,
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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ADAPTIVE_CURRICULUM_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str = 'lower'
    meaningful_delta: float = 0.0
    noninferiority_margin: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MetricSpec:
        spec = cls(
            name=str(raw.get('name') or '').strip(),
            direction=str(raw.get('direction', 'lower')).strip().lower(),
            meaningful_delta=float(raw.get('meaningful_delta', 0.0)),
            noninferiority_margin=float(raw.get('noninferiority_margin', 0.0)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not math.isfinite(self.noninferiority_margin) or self.noninferiority_margin < 0:
            raise ValueError('guardrail noninferiority_margin must be finite and non-negative')
        if not self.name:
            raise ValueError('adaptive metric name must not be empty')
        if self.direction not in {'lower', 'higher'}:
            raise ValueError(
                f'adaptive metric direction must be lower or higher: {self.name}'
            )
        if not math.isfinite(self.meaningful_delta) or self.meaningful_delta < 0:
            raise ValueError(
                f'adaptive metric meaningful_delta must be finite and non-negative: '
                f'{self.name}'
            )


@dataclass(frozen=True)
class AdaptiveCurriculumConfig:
    phase_name: str
    primary: MetricSpec
    final_phase: bool = False
    gate_every_steps: int = 50_000
    required_futile_gates: int = 2
    confidence_z: float = 1.96
    primary_noninferiority_margin: float = 0.0
    guardrails: tuple[MetricSpec, ...] = ()
    lr_levels: tuple[float, ...] = ()
    max_unresolved_gates: int = 4
    min_paired_games: int = 2
    selection_protocol: str = 'legacy_guardrails'

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> AdaptiveCurriculumConfig:
        primary_raw = raw.get('primary') or {
            'name': raw.get('primary_metric', 'primary_loss'),
            'direction': raw.get('primary_direction', 'lower'),
            'meaningful_delta': raw.get('meaningful_primary_delta', 2e-4),
        }
        config = cls(
            phase_name=str(raw.get('phase_name') or '').strip(),
            selection_protocol=str(raw.get('selection_protocol', 'legacy_guardrails')),
            primary=MetricSpec.from_mapping(primary_raw),
            final_phase=bool(raw.get('final_phase', False)),
            gate_every_steps=int(raw.get('gate_every_steps', 50_000)),
            required_futile_gates=int(raw.get('required_futile_gates', 2)),
            confidence_z=float(raw.get('confidence_z', 1.96)),
            primary_noninferiority_margin=float(
                raw.get('primary_noninferiority_margin', 0.0)
            ),
            guardrails=tuple(
                MetricSpec.from_mapping(item)
                for item in raw.get('guardrails', ())
            ),
            lr_levels=tuple(float(value) for value in raw.get('lr_levels', ())),
            max_unresolved_gates=int(raw.get('max_unresolved_gates', 4)),
            min_paired_games=int(raw.get('min_paired_games', 2)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.selection_protocol not in {'legacy_guardrails', 'primary_with_diagnostics'}:
            raise ValueError('unsupported adaptive selection_protocol')
        if self.max_unresolved_gates < 1 or self.min_paired_games < 2:
            raise ValueError('adaptive evidence budget must be positive and min_paired_games >= 2')
        if not self.phase_name:
            raise ValueError('adaptive curriculum phase_name must not be empty')
        self.primary.validate()
        if self.gate_every_steps <= 0:
            raise ValueError('adaptive curriculum gate_every_steps must be positive')
        if self.required_futile_gates <= 0:
            raise ValueError(
                'adaptive curriculum required_futile_gates must be positive'
            )
        if not math.isfinite(self.confidence_z) or self.confidence_z <= 0:
            raise ValueError('adaptive curriculum confidence_z must be positive')
        if (
            not math.isfinite(self.primary_noninferiority_margin)
            or self.primary_noninferiority_margin < 0
        ):
            raise ValueError(
                'adaptive curriculum primary_noninferiority_margin must be '
                'finite and non-negative'
            )
        names = {self.primary.name}
        for spec in self.guardrails:
            spec.validate()
            if spec.name in names:
                raise ValueError(f'duplicate adaptive metric: {spec.name}')
            names.add(spec.name)
        if self.final_phase:
            if not self.lr_levels:
                raise ValueError('final adaptive phase requires lr_levels')
            if any(
                not math.isfinite(value) or value <= 0
                for value in self.lr_levels
            ):
                raise ValueError(
                    'adaptive curriculum lr_levels must be positive and finite'
                )
            if any(
                left <= right
                for left, right in zip(self.lr_levels, self.lr_levels[1:])
            ):
                raise ValueError(
                    'adaptive curriculum lr_levels must be strictly descending'
                )
        elif self.lr_levels:
            raise ValueError('non-final adaptive phase must not define lr_levels')


@dataclass(frozen=True)
class AdaptiveCurriculumDecision:
    action: str
    state: dict[str, Any]
    comparisons: dict[str, dict[str, float | int | None]]
    target_lr: float | None = None
    reason: str = ''


def _normalize_record(record: Any) -> list[float | int]:
    if isinstance(record, Mapping):
        raw_game_id = record['game_id']
        game_id = int(raw_game_id)
        loss_sum = float(record.get('sum', record.get('value_sum')))
        raw_count = record['count']
        count = int(raw_count)
    else:
        raw_game_id, loss_sum, raw_count = record
        game_id = int(raw_game_id)
        loss_sum = float(loss_sum)
        count = int(raw_count)
    if game_id != raw_game_id:
        raise ValueError('cluster game id must be integral')
    if not math.isfinite(loss_sum):
        raise ValueError(f'cluster sum must be finite for game {game_id}')
    if count != raw_count:
        raise ValueError(f'cluster count must be integral for game {game_id}')
    if count <= 0:
        raise ValueError(f'cluster count must be positive for game {game_id}')
    return [game_id, loss_sum, count]


def normalize_cluster_records(records: Sequence[Any]) -> list[list[float | int]]:
    normalized = sorted((_normalize_record(record) for record in records), key=lambda x: x[0])
    if not normalized:
        raise ValueError('adaptive paired comparison cannot be empty')
    game_ids = [record[0] for record in normalized]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError('adaptive cluster records contain duplicate game ids')
    return normalized


def paired_cluster_summary(
    candidate_records: Sequence[Any],
    reference_records: Sequence[Any],
    *,
    confidence_z: float = 1.96,
) -> dict[str, float | int]:
    candidate = normalize_cluster_records(candidate_records)
    reference = normalize_cluster_records(reference_records)
    if len(candidate) != len(reference):
        raise ValueError('paired cluster records use different game sets')

    residual_terms: list[tuple[float, int]] = []
    total_difference = 0.0
    total_count = 0
    game_differences: list[float] = []
    for left, right in zip(candidate, reference):
        if left[0] != right[0] or left[2] != right[2]:
            raise ValueError(
                'paired cluster records use different game ids or sample counts'
            )
        difference_sum = float(left[1]) - float(right[1])
        count = int(left[2])
        total_difference += difference_sum
        total_count += count
        residual_terms.append((difference_sum, count))
        game_differences.append(difference_sum / count)

    mean = total_difference / total_count
    num_games = len(candidate)
    if num_games > 1:
        residual_square_sum = sum(
            (difference_sum - mean * count) ** 2
            for difference_sum, count in residual_terms
        )
        cluster_se = math.sqrt(
            num_games / (num_games - 1) * residual_square_sum / total_count**2
        )
        game_balanced_mean = sum(game_differences) / num_games
        game_residual_square_sum = sum(
            (value - game_balanced_mean) ** 2 for value in game_differences
        )
        game_balanced_se = math.sqrt(
            game_residual_square_sum / (num_games - 1) / num_games
        )
    else:
        cluster_se = 0.0
        game_balanced_mean = game_differences[0]
        game_balanced_se = 0.0
    if not all(math.isfinite(value) for value in
               (mean, cluster_se, game_balanced_mean, game_balanced_se)):
        raise ValueError('paired cluster summary must be finite')
    return {
        'mean': mean,
        'cluster_se': cluster_se,
        'ci_low': mean - confidence_z * cluster_se,
        'ci_high': mean + confidence_z * cluster_se,
        'game_balanced_mean': game_balanced_mean,
        'game_balanced_se': game_balanced_se,
        'num_samples': total_count,
        'num_games': num_games,
    }


def initial_adaptive_curriculum_state(
    config: AdaptiveCurriculumConfig,
) -> dict[str, Any]:
    return {
        **({'selection_protocol': config.selection_protocol}
           if config.selection_protocol != 'legacy_guardrails' else {}),
        'schema_version': ADAPTIVE_CURRICULUM_STATE_SCHEMA_VERSION,
        'phase_name': config.phase_name,
        'gate_index': 0,
        'last_gate_step': None,
        'consecutive_futile_gates': 0,
        'consecutive_unresolved_gates': 0,
        'best_step': None,
        'best_metrics': None,
        'best_cluster_records': None,
        'lr_level_index': 0,
        'completed': False,
        'completed_step': None,
        'last_action': 'uninitialized',
        'last_reason': '',
        'history': [],
    }


def normalize_adaptive_curriculum_state(
    raw: Mapping[str, Any] | None,
    config: AdaptiveCurriculumConfig,
) -> dict[str, Any]:
    state = initial_adaptive_curriculum_state(config)
    if raw:
        if raw.get('selection_protocol', 'legacy_guardrails') != config.selection_protocol:
            raise ValueError('adaptive selection protocol changed; use an explicit new branch')
        state.update(dict(raw))
    if int(state.get('schema_version') or 0) != ADAPTIVE_CURRICULUM_STATE_SCHEMA_VERSION:
        raise ValueError(
            'unsupported adaptive curriculum state schema: '
            f'{state.get("schema_version")!r}'
        )
    if state.get('phase_name') != config.phase_name:
        raise ValueError(
            'adaptive curriculum phase mismatch: '
            f'{state.get("phase_name")!r} != {config.phase_name!r}'
        )
    level_index = int(state.get('lr_level_index') or 0)
    if config.final_phase and not 0 <= level_index < len(config.lr_levels):
        raise ValueError(f'invalid adaptive lr_level_index: {level_index}')
    if not config.final_phase and level_index != 0:
        raise ValueError('non-final adaptive phase has a nonzero lr level')
    state['lr_level_index'] = level_index
    state['gate_index'] = int(state.get('gate_index') or 0)
    state['consecutive_futile_gates'] = int(
        state.get('consecutive_futile_gates') or 0
    )
    state['history'] = list(state.get('history') or [])[-32:]
    return state


def _normalize_observation(
    config: AdaptiveCurriculumConfig,
    metrics: Mapping[str, Any],
    cluster_records: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, float | None], dict[str, list[list[float | int]]]]:
    specs = (config.primary, *config.guardrails)
    normalized_metrics: dict[str, float | None] = {}
    normalized_records: dict[str, list[list[float | int]]] = {}
    for spec in specs:
        if (config.selection_protocol == 'primary_with_diagnostics'
                and spec != config.primary and spec.name in cluster_records
                and len(cluster_records[spec.name]) == 0):
            if metrics.get(spec.name) is not None:
                raise ValueError(f'empty diagnostic must have no metric value: {spec.name}')
            normalized_metrics[spec.name] = None
            normalized_records[spec.name] = []
            continue
        if spec.name not in metrics:
            raise ValueError(f'adaptive observation missing metric: {spec.name}')
        value = float(metrics[spec.name])
        if not math.isfinite(value):
            raise ValueError(f'adaptive metric must be finite: {spec.name}')
        if spec.name not in cluster_records:
            raise ValueError(
                f'adaptive observation missing cluster records: {spec.name}'
            )
        normalized_metrics[spec.name] = value
        normalized_records[spec.name] = normalize_cluster_records(
            cluster_records[spec.name]
        )
    return normalized_metrics, normalized_records


def validate_adaptive_observation(config, metrics, cluster_records, reference_state=None):
    """Validate observation integrity without changing gate or promotion state."""
    _, normalized = _normalize_observation(config, metrics, cluster_records)
    if reference_state and reference_state.get('best_step') is not None:
        _, reference = _normalize_observation(
            config, reference_state['best_metrics'], reference_state['best_cluster_records']
        )
        for name, records in normalized.items():
            if records or reference[name]:
                paired_cluster_summary(records, reference[name], confidence_z=config.confidence_z)


def inherit_adaptive_curriculum_baseline(
    config: AdaptiveCurriculumConfig,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    if source_state.get('selection_protocol', 'legacy_guardrails') != config.selection_protocol:
        raise ValueError('cannot inherit baseline across selection protocols; establish a new branch baseline')
    source_metrics = source_state.get('best_metrics')
    source_records = source_state.get('best_cluster_records')
    source_step = source_state.get('best_step')
    if (
        source_step is None
        or not isinstance(source_metrics, Mapping)
        or not isinstance(source_records, Mapping)
    ):
        raise ValueError('source adaptive phase has no paired best baseline')
    normalized_metrics, normalized_records = _normalize_observation(
        config,
        source_metrics,
        source_records,
    )
    state = initial_adaptive_curriculum_state(config)
    state.update({
        'best_step': int(source_step),
        'best_metrics': normalized_metrics,
        'best_cluster_records': normalized_records,
        'last_action': 'inherit_baseline',
        'last_reason': (
            f'inherited paired phase-best baseline from '
            f'{source_state.get("phase_name", "previous phase")}'
        ),
    })
    return state


def _is_clear_improvement(
    spec: MetricSpec,
    summary: Mapping[str, float | int],
) -> bool:
    if spec.direction == 'lower':
        return float(summary['ci_high']) <= -spec.meaningful_delta and float(summary['mean']) < 0
    return float(summary['ci_low']) >= spec.meaningful_delta and float(summary['mean']) > 0


def _can_still_improve(
    spec: MetricSpec,
    summary: Mapping[str, float | int],
) -> bool:
    if spec.direction == 'lower':
        return float(summary['ci_low']) <= -spec.meaningful_delta
    return float(summary['ci_high']) >= spec.meaningful_delta


def _is_primary_noninferior(
    config: AdaptiveCurriculumConfig,
    summary: Mapping[str, float | int],
) -> bool:
    margin = config.primary_noninferiority_margin
    if config.primary.direction == 'lower':
        return float(summary['ci_high']) <= margin
    return float(summary['ci_low']) >= -margin


def _guardrails_noninferior(config, comparisons):
    return all(
        int(comparisons[spec.name]['num_games']) >= config.min_paired_games
        and (
            float(comparisons[spec.name]['ci_high']) <= spec.noninferiority_margin
            if spec.direction == 'lower' else
            float(comparisons[spec.name]['ci_low']) >= -spec.noninferiority_margin
        )
        for spec in config.guardrails
    )


def _record_history(
    state: dict[str, Any],
    *,
    step: int,
    action: str,
    futile: bool,
    reason: str,
    comparisons: Mapping[str, Any],
) -> None:
    history = list(state.get('history') or [])
    history.append({
        'step': step,
        'action': action,
        'futile': futile,
        'consecutive_futile_gates': state['consecutive_futile_gates'],
        'reason': reason,
        'comparisons': {name: dict(value) for name, value in comparisons.items()},
    })
    state['history'] = history[-32:]


def observe_adaptive_curriculum(
    raw_state: Mapping[str, Any] | None,
    config: AdaptiveCurriculumConfig,
    *,
    optimizer_steps: int,
    metrics: Mapping[str, Any],
    cluster_records: Mapping[str, Sequence[Any]],
) -> AdaptiveCurriculumDecision:
    if optimizer_steps < 0:
        raise ValueError('optimizer_steps must be non-negative')
    state = normalize_adaptive_curriculum_state(raw_state, config)
    normalized_metrics, normalized_records = _normalize_observation(
        config,
        metrics,
        cluster_records,
    )
    if state['completed']:
        return AdaptiveCurriculumDecision(
            action=(
                'inconclusive' if state.get('last_action') == 'inconclusive'
                else 'stop' if config.final_phase else 'transition'
            ),
            state=state,
            comparisons={},
            target_lr=(
                config.lr_levels[state['lr_level_index']]
                if config.final_phase
                else None
            ),
            reason=str(state.get('last_reason') or 'phase already completed'),
        )

    last_gate_step = state.get('last_gate_step')
    if last_gate_step is not None and optimizer_steps <= int(last_gate_step):
        raise ValueError(
            'adaptive curriculum gate steps must be strictly increasing'
        )
    state['last_gate_step'] = optimizer_steps
    state['gate_index'] += 1

    if state.get('best_step') is None:
        state['best_step'] = optimizer_steps
        state['best_metrics'] = normalized_metrics
        state['best_cluster_records'] = normalized_records
        state['consecutive_futile_gates'] = 0
        state['last_action'] = 'set_baseline'
        state['last_reason'] = f'set phase baseline at step {optimizer_steps:,}'
        _record_history(
            state,
            step=optimizer_steps,
            action='set_baseline',
            futile=False,
            reason=state['last_reason'],
            comparisons={},
        )
        return AdaptiveCurriculumDecision(
            action='continue',
            state=state,
            comparisons={},
            target_lr=(config.lr_levels[0] if config.final_phase else None),
            reason=state['last_reason'],
        )

    _, reference_records = _normalize_observation(
        config, state.get('best_metrics') or {}, state.get('best_cluster_records') or {}
    )
    diagnostic_only = config.selection_protocol == 'primary_with_diagnostics'
    comparisons = {
        spec.name: paired_cluster_summary(
            normalized_records[spec.name],
            reference_records[spec.name],
            confidence_z=config.confidence_z,
        ) if normalized_records[spec.name] or reference_records[spec.name] else {
            'mean': None, 'cluster_se': None, 'ci_low': None, 'ci_high': None,
            'game_balanced_mean': None, 'game_balanced_se': None,
            'num_samples': 0, 'num_games': 0,
        }
        for spec in (config.primary, *config.guardrails)
    }
    primary_summary = comparisons[config.primary.name]
    enough_games = (int(primary_summary['num_games']) >= config.min_paired_games
                    if diagnostic_only else all(
                        int(item['num_games']) >= config.min_paired_games
                        for item in comparisons.values()))
    guards_pass = diagnostic_only or _guardrails_noninferior(config, comparisons)
    if enough_games and guards_pass and _is_clear_improvement(config.primary, primary_summary):
        state['best_step'] = optimizer_steps
        state['best_metrics'] = normalized_metrics
        state['best_cluster_records'] = normalized_records
        state['consecutive_futile_gates'] = 0
        state['consecutive_unresolved_gates'] = 0
        state['last_action'] = 'update_best'
        state['last_reason'] = (
            f'{config.primary.name} made a paired meaningful improvement'
        )
        _record_history(
            state,
            step=optimizer_steps,
            action='update_best',
            futile=False,
            reason=state['last_reason'],
            comparisons=comparisons,
        )
        return AdaptiveCurriculumDecision(
            action='update_best',
            state=state,
            comparisons=comparisons,
            target_lr=(
                config.lr_levels[state['lr_level_index']]
                if config.final_phase
                else None
            ),
            reason=state['last_reason'],
        )

    can_still_improve = _can_still_improve(config.primary, primary_summary)
    guardrail_compensation = False
    if not diagnostic_only and enough_games and guards_pass and _is_primary_noninferior(config, primary_summary):
        guardrail_compensation = any(
            _is_clear_improvement(spec, comparisons[spec.name])
            for spec in config.guardrails
        )
    futile = enough_games and not can_still_improve and not guardrail_compensation
    state['consecutive_unresolved_gates'] = (
        0 if futile else int(state.get('consecutive_unresolved_gates', 0)) + 1
    )
    if state['consecutive_unresolved_gates'] >= config.max_unresolved_gates:
        state.update({
            'completed': True,
            'completed_step': optimizer_steps,
            'last_action': 'inconclusive',
            'last_reason': (
                'predeclared unresolved-gate budget exhausted; retain the baseline '
                'and expand independent validation before continuing or advancing phase'
            ),
        })
        _record_history(state, step=optimizer_steps, action='inconclusive', futile=False,
                        reason=state['last_reason'], comparisons=comparisons)
        return AdaptiveCurriculumDecision(
            action='inconclusive', state=state, comparisons=comparisons,
            reason=state['last_reason'],
        )
    if futile:
        state['consecutive_futile_gates'] += 1
    else:
        state['consecutive_futile_gates'] = 0

    if state['consecutive_futile_gates'] < config.required_futile_gates:
        state['last_action'] = 'observe'
        if guardrail_compensation:
            state['last_reason'] = 'guardrail compensation keeps the phase open'
        elif not enough_games and diagnostic_only:
            state['last_reason'] = 'paired primary evidence has insufficient games'
        elif not enough_games or not guards_pass:
            state['last_reason'] = 'paired evidence does not establish every guardrail as noninferior'
        elif can_still_improve:
            state['last_reason'] = (
                'paired interval still permits a meaningful primary improvement'
            )
        else:
            state['last_reason'] = (
                f'futile gate {state["consecutive_futile_gates"]}/'
                f'{config.required_futile_gates}'
            )
        _record_history(
            state,
            step=optimizer_steps,
            action='observe',
            futile=futile,
            reason=state['last_reason'],
            comparisons=comparisons,
        )
        return AdaptiveCurriculumDecision(
            action='continue',
            state=state,
            comparisons=comparisons,
            target_lr=(
                config.lr_levels[state['lr_level_index']]
                if config.final_phase
                else None
            ),
            reason=state['last_reason'],
        )

    if not config.final_phase:
        state['completed'] = True
        state['completed_step'] = optimizer_steps
        state['last_action'] = 'transition'
        state['last_reason'] = (
            f'{config.required_futile_gates} consecutive paired gates cannot '
            + ('show a meaningful primary gain' if diagnostic_only
               else 'show a meaningful gain or guardrail compensation')
        )
        _record_history(
            state,
            step=optimizer_steps,
            action='transition',
            futile=True,
            reason=state['last_reason'],
            comparisons=comparisons,
        )
        return AdaptiveCurriculumDecision(
            action='transition',
            state=state,
            comparisons=comparisons,
            reason=state['last_reason'],
        )

    level_index = state['lr_level_index']
    if level_index < len(config.lr_levels) - 1:
        state['lr_level_index'] = level_index + 1
        state['consecutive_futile_gates'] = 0
        state['last_action'] = 'reduce_lr'
        target_lr = config.lr_levels[state['lr_level_index']]
        state['last_reason'] = (
            f'paired evidence exhausted lr level {level_index}; '
            f'reduce to {target_lr:.3e}'
        )
        _record_history(
            state,
            step=optimizer_steps,
            action='reduce_lr',
            futile=True,
            reason=state['last_reason'],
            comparisons=comparisons,
        )
        return AdaptiveCurriculumDecision(
            action='reduce_lr',
            state=state,
            comparisons=comparisons,
            target_lr=target_lr,
            reason=state['last_reason'],
        )

    state['completed'] = True
    state['completed_step'] = optimizer_steps
    state['last_action'] = 'stop'
    state['last_reason'] = (
        f'{config.required_futile_gates} consecutive futile paired gates at '
        'the final lr level'
    )
    _record_history(
        state,
        step=optimizer_steps,
        action='stop',
        futile=True,
        reason=state['last_reason'],
        comparisons=comparisons,
    )
    return AdaptiveCurriculumDecision(
        action='stop',
        state=state,
        comparisons=comparisons,
        target_lr=config.lr_levels[level_index],
        reason=state['last_reason'],
    )

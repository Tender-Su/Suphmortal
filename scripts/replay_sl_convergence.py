from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mortal.supervised.convergence import ConvergenceConfig, observe_convergence


PAIR_RE = re.compile(r'(\w+)=([^\s]+)')


def parse_number(raw: str) -> float:
    return float(raw.rstrip(',').replace(',', ''))


def parse_full_recent(log_path: Path) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if '[FULL RECENT]' not in line:
            continue
        fields = {key: value for key, value in PAIR_RE.findall(line)}
        if 'step' not in fields or 'policy' not in fields:
            continue
        step = int(parse_number(fields['step']))
        by_step[step] = {
            'step': step,
            'policy_loss': parse_number(fields['policy']),
            'lr': parse_number(fields['lr']) if 'lr' in fields else float('nan'),
        }
    return [by_step[step] for step in sorted(by_step)]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def adjacent_changes(values: list[float]) -> dict[str, float | None]:
    changes = [abs(right - left) for left, right in zip(values, values[1:])]
    return {
        'count': len(changes),
        'median': statistics.median(changes) if changes else None,
        'p90': percentile(changes, 0.9),
        'max': max(changes) if changes else None,
    }


def rolling_medians(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError('window must be positive')
    return [
        float(statistics.median(values[index - window + 1 : index + 1]))
        for index in range(window - 1, len(values))
    ]


def replay(log_path: Path, config: ConvergenceConfig) -> dict:
    rows = parse_full_recent(log_path)
    state = None
    decisions = []
    for row in rows:
        decision = observe_convergence(
            state,
            config,
            optimizer_steps=int(row['step']),
            metric_value=float(row['policy_loss']),
        )
        state = decision.state
        if decision.action != 'continue' or state['last_action'] != 'core':
            decisions.append({
                'step': int(row['step']),
                'policy_loss': float(row['policy_loss']),
                'action': decision.action,
                'state_action': state['last_action'],
                'target_lr': decision.target_lr,
                'reason': decision.reason,
            })
    values = [float(row['policy_loss']) for row in rows]
    smoothed = rolling_medians(values, config.smoothing_checks)
    return {
        'log_path': str(log_path.resolve()),
        'points': len(rows),
        'first_step': int(rows[0]['step']) if rows else None,
        'last_step': int(rows[-1]['step']) if rows else None,
        'first_policy_loss': values[0] if values else None,
        'last_policy_loss': values[-1] if values else None,
        'best_policy_loss': min(values) if values else None,
        'raw_adjacent_abs_change': adjacent_changes(values),
        'smoothed_adjacent_abs_change': adjacent_changes(smoothed),
        'core_optimizer_steps': config.core_optimizer_steps,
        'tail_started': bool((state or {}).get('tail_started', False)),
        'converged': bool((state or {}).get('converged', False)),
        'non_core_decisions': decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, action='append', required=True)
    parser.add_argument('--core-steps', type=int, action='append', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(args.log) != len(args.core_steps):
        parser.error('--log and --core-steps must have the same number of values')
    reports = []
    for log_path, core_steps in zip(args.log, args.core_steps):
        config = ConvergenceConfig(
            core_optimizer_steps=core_steps,
            tail_lr_levels=(1e-5, 5e-6, 2.5e-6, 1e-6),
            smoothing_checks=5,
            improvement_delta=2e-4,
            reduce_patience_steps=160_000,
            stop_patience_steps=240_000,
            min_level_steps=80_000,
        )
        reports.append(replay(log_path, config))
    payload = {'schema_version': 1, 'reports': reports}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8', newline='\n')
    print(rendered)


if __name__ == '__main__':
    main()

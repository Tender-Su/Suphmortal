"""Paired duplicate-1v3 inference, clustered by the full (seed, seed_key).

Intervals describe these fixed checkpoints on the supplied evaluation seeds.
They do not correct for checkpoint selection or training-seed variability.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mortal.core.artifacts import atomic_write_json
from mortal.eval.summarize_duplicate_logs import load_challenger_game


RANK_POINTS = np.asarray([90.0, 45.0, 0.0, -135.0])


def duplicate_sets(games):
    sets = defaultdict(dict)
    for game in games:
        if game.get('seed_key') is None:
            raise ValueError('native duplicate logs must retain the full seed_key')
        key = (int(game['seed']), int(game['seed_key']))
        seat = int(game['challenger_seat'])
        rank = int(game['challenger_rank'])
        if seat not in range(4) or rank not in range(1, 5):
            raise ValueError(f'invalid seat or rank in duplicate set {key}')
        if seat in sets[key]:
            raise ValueError(f'duplicate challenger seat {seat} for seed set {key}')
        sets[key][seat] = rank
    if len(sets) < 2:
        raise ValueError('at least two independent seed sets are required for inference')
    for key, seats in sets.items():
        if set(seats) != {0, 1, 2, 3}:
            raise ValueError(f'incomplete four-seat rotation for seed set {key}')
    return {key: np.asarray([seats[i] for i in range(4)]) for key, seats in sorted(sets.items())}


def bootstrap_interval(values, *, replicates=20_000, seed=20260905):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or replicates < 1000:
        raise ValueError('need at least two clusters and 1000 bootstrap replicates')
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[sample].mean(axis=1)
    return {
        'mean': float(values.mean()),
        'cluster_se': float(values.std(ddof=1) / np.sqrt(len(values))),
        'ci95': np.quantile(means, [0.025, 0.975]).tolist(),
    }


def compare_games(candidate_games, reference_games, *, replicates=20_000, seed=20260905):
    candidate = duplicate_sets(candidate_games)
    reference = duplicate_sets(reference_games)
    if candidate.keys() != reference.keys():
        raise ValueError('candidate and reference must have exactly the same seed sets')
    left = np.stack(list(candidate.values()))
    right = np.stack(list(reference.values()))
    delta_pt = (RANK_POINTS[left - 1] - RANK_POINTS[right - 1]).mean(axis=1)
    delta_rank = (left - right).mean(axis=1)
    return {
        'games_per_arm': int(left.size),
        'independent_seed_sets': len(candidate),
        'seed_keys': sorted({key[1] for key in candidate}),
        'candidate_rankings': np.bincount(left.reshape(-1), minlength=5)[1:5].tolist(),
        'reference_rankings': np.bincount(right.reshape(-1), minlength=5)[1:5].tolist(),
        'candidate_avg_pt': float(RANK_POINTS[left - 1].mean()),
        'reference_avg_pt': float(RANK_POINTS[right - 1].mean()),
        'candidate_avg_rank': float(left.mean()),
        'reference_avg_rank': float(right.mean()),
        'delta_pt_candidate_minus_reference': bootstrap_interval(delta_pt, replicates=replicates, seed=seed),
        'delta_rank_candidate_minus_reference': bootstrap_interval(delta_rank, replicates=replicates, seed=seed),
        'changed_game_outcomes': int((left != right).sum()),
        'ci_unit': 'paired_full_seed_key_four_seat_set',
        'bootstrap_replicates': replicates,
        'bootstrap_seed': seed,
        'selection_adjusted': False,
        'training_seed_uncertainty_included': False,
        'requires_matching_opponent_rules_and_inference_settings': True,
    }


def load_games(log_dir, challenger_name):
    paths = sorted(Path(log_dir).rglob('*.json.gz'))
    if not paths:
        raise FileNotFoundError(f'no native .json.gz game logs under {log_dir}')
    games = [load_challenger_game(path, challenger_name) for path in paths]
    duplicate_sets(games)
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-log-dir', required=True)
    parser.add_argument('--reference-log-dir', required=True)
    parser.add_argument('--candidate-name', default='mortal')
    parser.add_argument('--reference-name', default='mortal')
    parser.add_argument('--bootstrap-replicates', type=int, default=20_000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    candidate = load_games(args.candidate_log_dir, args.candidate_name)
    reference = load_games(args.reference_log_dir, args.reference_name)
    result = {
        'summary': compare_games(candidate, reference, replicates=args.bootstrap_replicates),
        'candidate_games': candidate,
        'reference_games': reference,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

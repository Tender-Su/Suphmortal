"""Raw frozen-probe cache, complete four-seat groups and descriptive statistics."""
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import time

from mortal.core.artifacts import atomic_output_path, atomic_write_json, file_sha256


class BudgetExpired(RuntimeError):
    pass


class ProbeBudget:
    def __init__(self, deadline_unix, *, reserve_seconds=60, stop_file=None):
        if reserve_seconds != 60 or not 0 < deadline_unix - time.time() <= 600:
            raise ValueError('single-reference probe needs a <=600 second lease and 60 second cleanup reserve')
        self.deadline = deadline_unix
        self.stop_unix = deadline_unix - reserve_seconds
        self.stop_monotonic = time.monotonic() + self.stop_unix - time.time()
        self.stop_file = Path(stop_file) if stop_file else None

    @classmethod
    def from_environment(cls):
        value = os.environ.get('MORTAL_RUN_DEADLINE_UTC')
        if not value or not os.environ.get('MORTAL_RUN_ID'):
            raise ValueError('single-reference probe requires the existing deadline supervisor')
        deadline = datetime.fromisoformat(value)
        if deadline.tzinfo is None:
            raise ValueError('deadline must carry its UTC offset')
        return cls(deadline.timestamp(), stop_file=os.environ.get('MORTAL_STOP_FILE'))

    def stopping(self):
        return (time.time() >= self.stop_unix or time.monotonic() >= self.stop_monotonic
                or bool(self.stop_file and self.stop_file.exists()))

    def check(self):
        if self.stopping():
            raise BudgetExpired('no new work: cleanup reserve or supervisor stop reached')

    def manifest(self):
        return {'hard_deadline_utc': datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
                'no_new_work_utc': datetime.fromtimestamp(self.stop_unix, timezone.utc).isoformat(),
                'cleanup_reserve_seconds': 60, 'total_cap_seconds': 600,
                'stop_file': str(self.stop_file) if self.stop_file else None}


def inference_slices(length, batch_size, budget=None):
    for offset in range(0, length, batch_size):
        if budget:
            budget.check()
        yield offset


class GroupCacheWriter:
    """Publish each complete game first; admit statistics only after all four seats."""
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'cache').mkdir(exist_ok=False)
        self.buffer = io.StringIO()
        self.groups = []
        self.pending = []
        self.completed_games = 0
        self.publish()

    def write(self, line):
        self.buffer.write(line)

    def publish(self):
        atomic_write_json(self.root / 'cache_index.json',
                          {'schema': 1, 'complete_groups': self.groups,
                           'complete_games_in_incomplete_group': self.pending,
                           'completed_games': self.completed_games,
                           'partial_groups_included_in_statistics': False})

    def finish_game(self, game):
        key = (game['seed'], game['seed_key'])
        seat = game['challenger_seat']
        if (not 0 <= seat < 4 or any(item['seat'] == seat for item in self.pending)
                or (self.pending and tuple(self.pending[0]['key']) != key)
                or any(tuple(group['key']) == key for group in self.groups)):
            raise ValueError('cache requires unique contiguous four-seat seed groups')
        raw = self.buffer.getvalue()
        if not raw:
            raise ValueError('cannot publish an empty trajectory')
        rows = [json.loads(line) for line in raw.splitlines()]
        if (any((row['seed'], row['seed_key'], row['trainee_seat']) != (*key, seat) for row in rows)
                or [row['decision_index'] for row in rows] != list(range(len(rows)))
                or any(row['truncated'] for row in rows)
                or [row['done'] for row in rows] != [False] * (len(rows) - 1) + [True]):
            raise ValueError('cache trajectory identity, terminal or truncation contract differs')
        filename = f'cache/{key[0]}_{key[1]}_seat{seat}.jsonl.gz'
        with atomic_output_path(self.root / filename) as temporary:
            with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as stream:
                stream.write(raw)
        self.pending.append({'key': list(key), 'seat': seat, 'states': len(rows),
                             'path': filename, 'sha256': file_sha256(self.root / filename)})
        self.buffer.close()
        self.buffer = io.StringIO()
        if len(self.pending) == 4:
            self.groups.append({'key': list(key), 'games': self.pending,
                                'states': sum(item['states'] for item in self.pending)})
            self.completed_games += 4
            self.pending = []
        self.publish()

    def close(self):
        self.buffer.close()


@contextmanager
def prediction_stream(root, *, reference):
    if reference:
        stream = GroupCacheWriter(root)
        try:
            yield stream
        finally:
            stream.close()
    else:
        with atomic_output_path(Path(root) / 'predictions.jsonl.gz') as temporary:
            with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as stream:
                yield stream


def temporal_fields(trajectory, game, rewards, target, pred, gae):
    import numpy as np
    n = len(target)
    seat = int(trajectory['player_id'])
    indices = np.asarray(trajectory['decision_indices'])
    if seat != game['challenger_seat'] or not np.array_equal(indices, np.arange(n)):
        raise ValueError('controlled actor seat or full native decision clock differs')
    if n == 0 or np.asarray(trajectory['actions']).shape != (n,):
        raise ValueError('missing complete ordered actions')
    next_value = np.concatenate((pred[1:], np.zeros((1, 4), dtype=pred.dtype)))
    adv1 = np.stack([gae(rewards[:, h], pred[:, h], 1., 1.) for h in range(4)], axis=1)
    adv95 = np.stack([gae(rewards[:, h], pred[:, h], 1., .95) for h in range(4)], axis=1)
    np.testing.assert_allclose(adv1 + pred, target, atol=2e-5, rtol=2e-5)
    if not all(np.isfinite(x).all() for x in (rewards, target, pred, next_value, adv1, adv95)):
        raise ValueError('non-finite raw value/return/advantage')
    return {'decision_index': indices.tolist(), 'action': trajectory['actions'].tolist(),
            'at_kyoku': trajectory['at_kyoku'].tolist(),
            'context_meta': trajectory['context_meta'].tolist(),
            'head_absolute_seats': [[(seat + h) % 4 for h in range(4)]] * n,
            'r_t': rewards.tolist(), 'V_t': pred.tolist(), 'next_V': next_value.tolist(),
            'G_t': target.tolist(), 'gae_lambda1': adv1.tolist(), 'gae_lambda095': adv95.tolist(),
            'done': [False] * (n - 1) + [True], 'truncated': [False] * n}


def raw_metrics(target, pred):
    import numpy as np
    from mortal.research.frozen_actor_critic_probe import distribution
    target, pred = np.asarray(target, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if target.shape != pred.shape or target.ndim != 2 or target.shape[1] != 4:
        raise ValueError('expected matching four-head raw arrays')
    if len(target) == 0:
        return {'states': 0, 'p0': None, 'all_players': None}
    result = {'states': len(target)}
    for name, actual, value in [('p0', target[:, 0], pred[:, 0]), ('all_players', target.ravel(), pred.ravel())]:
        error = value - actual
        variance = float(actual.var())
        correlation = float(np.corrcoef(actual, value)[0, 1]) if variance > 0 and value.var() > 0 else None
        result[name] = {'bias_V_minus_G': float(error.mean()), 'mae': float(np.abs(error).mean()),
                        'mse': float(np.square(error).mean()), 'rmse': float(np.sqrt(np.square(error).mean())),
                        'target': distribution(actual), 'prediction': distribution(value),
                        'correlation': correlation,
                        'explained_variance': 1 - float(error.var()) / variance if variance > 0 else None}
    return result


def cluster_ratio_intervals(cluster_sums, counts, *, replicates, seed):
    """Resample whole four-seat groups, then divide total sums by total decisions."""
    import numpy as np
    sums = np.asarray(cluster_sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if sums.ndim != 2 or counts.shape != (len(sums),) or len(sums) == 0 or (counts <= 0).any():
        raise ValueError('invalid complete-group sufficient statistics')
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(counts), size=(replicates, len(counts)))
    draws = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None]
    return {'estimate': (sums.sum(axis=0) / counts.sum()).tolist(),
            'ci95_low': np.quantile(draws, .025, axis=0).tolist(),
            'ci95_high': np.quantile(draws, .975, axis=0).tolist()}


def reference_summary(args, games, targets, predictions, contexts, advantages, groups):
    import numpy as np
    from mortal.research.frozen_actor_critic_probe import calibration, distribution
    expected = [(seed, args.seed_key) for seed in range(args.seed_start, args.seed_start + args.games // 4)]
    complete = [tuple(group['key']) for group in groups]
    if len(complete) != len(set(complete)) or any(key not in expected for key in complete):
        raise ValueError('unregistered or duplicate seed group')
    if not len(games) == len(targets) == len(predictions) == len(contexts) == len(advantages) == len(groups) * 4:
        raise ValueError('statistics require full cached four-seat groups')
    for index, key in enumerate(complete):
        block = games[4 * index:4 * index + 4]
        if {game['challenger_seat'] for game in block} != set(range(4)) or any(
                (game['seed'], game['seed_key']) != key for game in block):
            raise ValueError('statistics group seat or seed mismatch')
    full = complete == expected
    result = {'status': 'complete' if full else 'incomplete', 'planned_games': args.games,
              'games': len(games), 'states': sum(map(len, targets)),
              'complete_groups': [list(key) for key in complete],
              'missing_groups': [list(key) for key in expected if key not in complete],
              'planned_statistics_available': full,
              'duration_selection_bias': not full,
              'interpretation': ('descriptive fixed-actor diagnostic only; no readiness or strength threshold'
                                 if full else 'incomplete duration-selected subset; contract/feasibility only, no planned sample success'),
              'estimand': 'ratio of sums over decisions; bootstrap unit is complete four-seat seed group',
              'group_artifacts': groups}
    # Incomplete runs deliberately publish counts and contracts without selected-subset inferential statistics.
    if not full:
        return result
    target, pred, context = np.concatenate(targets), np.concatenate(predictions), np.concatenate(contexts)
    result['metrics'] = raw_metrics(target, pred)
    result['constant_zero_baseline'] = raw_metrics(target, np.zeros_like(target))
    result['p0_prediction_bins'] = calibration(target[:, 0], pred[:, 0])
    result['p0_gae_lambda095'] = distribution(np.concatenate(advantages))
    result['input_strata'] = {
        f'current_rank={rank},all_last={last}': raw_metrics(target[keep], pred[keep])
        for rank in range(4) for last in (0, 1)
        for keep in [((context[:, 4] == rank) & (context[:, 3] == last))]}
    sums, counts = [], []
    for index in range(len(groups)):
        y = np.concatenate(targets[4 * index:4 * index + 4]).astype(np.float64)
        p = np.concatenate(predictions[4 * index:4 * index + 4]).astype(np.float64)
        e = p - y
        sums.append([e[:, 0].sum(), np.abs(e[:, 0]).sum(), np.square(e[:, 0]).sum(),
                     np.square(e).sum() / 4,
                     (np.square(e[:, 0]) - np.square(y[:, 0])).sum()])
        counts.append(len(y))
    result['cluster_intervals'] = {
        'columns': ['p0_bias', 'p0_mae', 'p0_mse', 'all_players_mse', 'p0_mse_minus_zero'],
        **cluster_ratio_intervals(sums, counts, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed),
        'replicates': args.bootstrap_replicates, 'seed': args.bootstrap_seed,
        'training_seed_uncertainty_included': False, 'selection_adjusted': False}
    return result

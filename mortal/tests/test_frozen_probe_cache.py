"""CPU regressions for the optional single-reference cache; no arena or model forward."""
import ast
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mortal.research.frozen_actor_critic_probe import parse_args, validate_critic_contract
from mortal.research.frozen_probe_cache import (
    BudgetExpired, ProbeBudget, GroupCacheWriter, temporal_fields,
    cluster_ratio_intervals, reference_summary, inference_slices, raw_metrics,
)


def production_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ReferenceTests(unittest.TestCase):
    def test_explicit_reference_identity_and_old_roles_remain_distinct(self):
        base = ['--config', 'cfg', '--actor', 'actor', '--opponent', 'opp', '--output-dir', 'new',
                '--seed-start', '10', '--seed-key', '2', '--sampling-seed', '3']
        ref = ['--reference', 'selected', '--reference-sha256', 'a' * 64, '--reference-steps', '250000']
        self.assertEqual(parse_args(base + ref + ['--games', '128']).reference_steps, 250000)
        for extra in (['--warm0', 'old'], ['--shuffle-hidden'], ['--reference-sha256', 'wrong'],
                      ['--reference-steps', '-1'], ['--reuse-rollout-dir', 'old']):
            with self.assertRaises(SystemExit):
                parse_args(base + ref + extra)
        with self.assertRaises(SystemExit):
            parse_args(base + ['--reference', 'selected'])
        old = sum((['--' + key, key] for key in ('warm0', 'warm40k', 'clean40k')), [])
        self.assertIsNone(parse_args(base + old).reference)

    def test_selected_step_is_not_relabeled_legacy_40k(self):
        state = {'steps': 250000, 'config': {'control': {'version': 4}, 'env': {'pts': [2, 1, 0, -3]}},
                 'oracle_critic_pretrain': {'target_mode': 'all_players', 'return_mode': 'score_rank_mc',
                                           'discount_gamma': 1, 'critic_arch': 'dual_tower', 'value_loss_mode': 'mse'}}
        validate_critic_contract(state, version=4, pts=[2, 1, 0, -3])
        with self.assertRaises(ValueError):
            validate_critic_contract(state, version=4, pts=[2, 1, 0, -3], role='warm40k')

    def test_terminal_reward_skipped_kyoku_and_all_seat_rotations(self):
        root = Path(__file__).resolve().parents[1]
        namespace = {'np': np}
        expand = production_function(root / 'data/oracle_value.py', 'expand_kyoku_rewards_to_steps', namespace)
        returns = production_function(root / 'data/oracle_value.py', 'discounted_returns_from_step_rewards', namespace)
        gae = production_function(root / 'online/train_online.py', 'compute_gae_advantages_from_step_rewards', {'_np': np})
        rotate = production_function(root / 'data/dataloader.py', 'rotate_values_to_relative_order', {'np': np})
        absolute_rewards = np.array([[1, -1, 2, -2], [2, 1, -1, -2], [-1, 2, -2, 1]], np.float32)
        for seat in range(4):
            # A kyoku without this actor's decision is assigned to its preceding transition.
            rewards = expand(rotate(absolute_rewards, seat), [0, 0, 2])
            target = returns(rewards, 1.)
            pred = np.array([[.4, -.3, .2, -.3], [.2, -.2, .1, -.1], [1, 2, -1, -2]], np.float32)
            trajectory = {'player_id': seat, 'decision_indices': np.arange(3),
                          'actions': np.array([3, 4, 5]), 'at_kyoku': np.array([0, 0, 2]),
                          'context_meta': np.zeros((3, 8))}
            fields = temporal_fields(trajectory, {'challenger_seat': seat}, rewards, target, pred, gae)
            np.testing.assert_array_equal(rewards[0], [0, 0, 0, 0])
            np.testing.assert_allclose(rewards[1], np.roll(absolute_rewards[:2].sum(0), -seat))
            np.testing.assert_allclose(target[0], np.roll(absolute_rewards.sum(0), -seat))
            self.assertEqual(fields['head_absolute_seats'][0], [(seat + h) % 4 for h in range(4)])
            self.assertEqual(fields['done'], [False, False, True])
            self.assertEqual(fields['truncated'], [False] * 3)
            self.assertEqual(fields['next_V'][-1], [0] * 4)
            np.testing.assert_allclose(np.array(fields['gae_lambda1']) + pred, target, atol=2e-5, rtol=2e-5)
            self.assertFalse(np.allclose(np.array(fields['gae_lambda095']) + pred, target))
            with self.assertRaises(ValueError):
                temporal_fields({**trajectory, 'player_id': (seat + 1) % 4},
                                {'challenger_seat': seat}, rewards, target, pred, gae)
            with self.assertRaises(ValueError):
                temporal_fields({**trajectory, 'decision_indices': np.array([0, 2, 3])},
                                {'challenger_seat': seat}, rewards, target, pred, gae)

    def write_game(self, writer, seed, seat, **changes):
        row = {'seed': seed, 'seed_key': 2, 'trainee_seat': seat, 'decision_index': 0,
               'done': True, 'truncated': False, **changes}
        writer.write(json.dumps(row) + '\n')
        writer.finish_game({'seed': seed, 'seed_key': 2, 'challenger_seat': seat})

    def test_partial_groups_survive_but_never_count_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = GroupCacheWriter(temp)
            for seat in range(3):
                self.write_game(writer, 10, seat)
            self.assertEqual(writer.completed_games, 0)
            self.assertEqual(len(writer.pending), 3)
            self.write_game(writer, 10, 3)
            self.assertEqual(writer.completed_games, 4)
            self.write_game(writer, 11, 0)
            writer.close()
            index = json.loads((Path(temp) / 'cache_index.json').read_text())
            self.assertEqual(index['completed_games'], 4)
            self.assertEqual(len(index['complete_games_in_incomplete_group']), 1)
            for game in index['complete_groups'][0]['games']:
                with gzip.open(Path(temp) / game['path'], 'rt') as stream:
                    self.assertTrue(json.loads(stream.readline())['done'])

    def test_truncated_or_nonterminal_cache_is_never_published(self):
        for changes in ({'truncated': True}, {'done': False}, {'decision_index': 1}):
            with tempfile.TemporaryDirectory() as temp:
                writer = GroupCacheWriter(temp)
                with self.assertRaises(ValueError):
                    self.write_game(writer, 10, 0, **changes)
                writer.close()
                self.assertFalse(list((Path(temp) / 'cache').glob('*.gz')))

    def test_repeated_seat_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = GroupCacheWriter(temp)
            self.write_game(writer, 10, 0)
            with self.assertRaises(ValueError):
                self.write_game(writer, 10, 0)
            writer.close()

    def test_cluster_bootstrap_uses_ratio_of_sums_with_unequal_lengths(self):
        report = cluster_ratio_intervals([[0], [100]], [1, 10], replicates=1000, seed=2)
        self.assertAlmostEqual(report['estimate'][0], 100 / 11)
        self.assertNotAlmostEqual(report['estimate'][0], 5)
        self.assertGreaterEqual(report['ci95_low'][0], 0)
        self.assertLessEqual(report['ci95_high'][0], 10)

    def test_incomplete_fixed_plan_reports_missing_not_selected_subset_statistics(self):
        args = SimpleNamespace(seed_start=10, seed_key=2, games=128, bootstrap_replicates=1000, bootstrap_seed=3)
        games = [{'seed': 10, 'seed_key': 2, 'challenger_seat': seat} for seat in range(4)]
        values = [np.zeros((1, 4))] * 4
        result = reference_summary(args, games, values, values, [np.zeros((1, 8))] * 4,
                                   [np.zeros(1)] * 4, [{'key': [10, 2]}])
        self.assertFalse(result['planned_statistics_available'])
        self.assertTrue(result['duration_selection_bias'])
        self.assertEqual(len(result['missing_groups']), 31)
        self.assertNotIn('metrics', result)
        self.assertNotIn('cluster_intervals', result)

    def test_zero_variance_is_not_reported_as_correlation_or_skill(self):
        report = raw_metrics(np.ones((3, 4)), np.zeros((3, 4)))['p0']
        self.assertIsNone(report['correlation'])
        self.assertIsNone(report['explained_variance'])
        self.assertEqual(report['mae'], 1)
        self.assertEqual(report['rmse'], 1)

    def test_budget_stops_new_batches_and_does_not_extend_for_clock_changes(self):
        with patch('mortal.research.frozen_probe_cache.time.time', return_value=100), \
                patch('mortal.research.frozen_probe_cache.time.monotonic', return_value=10):
            budget = ProbeBudget(700)
            iterator = inference_slices(3, 1, budget)
            self.assertEqual(next(iterator), 0)
        with patch('mortal.research.frozen_probe_cache.time.time', return_value=50), \
                patch('mortal.research.frozen_probe_cache.time.monotonic', return_value=550):
            with self.assertRaises(BudgetExpired):
                next(iterator)
        with patch('mortal.research.frozen_probe_cache.time.time', return_value=100):
            with self.assertRaises(ValueError):
                ProbeBudget(701)
            with self.assertRaises(ValueError):
                ProbeBudget(700, reserve_seconds=30)

    def test_explicit_stop_file_is_respected(self):
        with tempfile.TemporaryDirectory() as temp:
            stop = Path(temp) / 'STOP'
            with patch('mortal.research.frozen_probe_cache.time.time', return_value=100):
                budget = ProbeBudget(700, stop_file=stop)
                self.assertFalse(budget.stopping())
                stop.touch()
                with self.assertRaises(BudgetExpired):
                    budget.check()


if __name__ == '__main__':
    unittest.main()

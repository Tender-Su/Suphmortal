"""Lightweight contract tests; do not imply torch/native integration coverage."""
import ast
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from mortal.research.frozen_actor_critic_probe import (
    calibration, parse_args, reserve_output, summarize_predictions,
    validate_critic_contract, verify_complete_log,
)

ROOT = Path(__file__).resolve().parents[1]


class ProbeContractTests(unittest.TestCase):
    def test_output_is_new_and_never_deletes_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'new'
            self.assertEqual(reserve_output(path), path)
            marker = path / 'keep'
            marker.write_text('preserved')
            with self.assertRaises(FileExistsError):
                reserve_output(path)
            self.assertEqual(marker.read_text(), 'preserved')

    def test_cli_defaults_and_seed_group_limits(self):
        base = []
        for name in ('config', 'actor', 'opponent', 'warm0', 'warm40k', 'clean40k', 'output-dir'):
            base.extend(['--' + name, 'placeholder'])
        base += ['--seed-start', '1000000', '--seed-key', '2026093001', '--sampling-seed', '123']
        args = parse_args(base)
        self.assertEqual(args.games, 256)
        self.assertEqual(args.imputation_seed, 20260905)
        for extra in (['--games', '4'], ['--games', '10'], ['--seed-key', '-1'], ['--batch-size', '0']):
            with self.assertRaises(SystemExit):
                parse_args(base + extra)

    def test_metrics_and_bias_are_signed_prediction_minus_target(self):
        target = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
        result = summarize_predictions(target, target + [1, 2, 3, 4])
        self.assertEqual(result['states'], 2)
        self.assertEqual(result['p0_mse'], 1)
        self.assertEqual(result['all_players_mse'], 7.5)
        self.assertEqual(result['all_players_bias'], 2.5)
        self.assertEqual(summarize_predictions(target, np.zeros_like(target))['p0_mse'], 2.5)
        for bad in (np.full((2, 4), np.nan), np.empty((0, 4)), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                summarize_predictions(np.zeros_like(bad), bad)

    def test_prediction_bins_cover_boundaries_once(self):
        pred = np.array([-10., -4, -3, -2, -1, 0, 1, 2, 3, 4, 10])
        bins = calibration(pred + 1, pred)
        self.assertEqual(sum(b['count'] for b in bins), len(pred))
        for row in bins:
            if row['count']:
                self.assertAlmostEqual(row['target_mean'] - row['prediction_mean'], 1)

    def test_critic_contract_rejects_wrong_label_semantics(self):
        state = {'config': {'control': {'version': 4}, 'env': {'pts': [2, 1, 0, -3]}},
                 'oracle_critic_pretrain': {'target_mode': 'all_players', 'return_mode': 'score_rank_mc',
                                           'discount_gamma': 1, 'critic_arch': 'dual_tower'}}
        self.assertEqual(validate_critic_contract(state, version=4, pts=[2, 1, 0, -3])['discount_gamma'], 1)
        for field, value in (('discount_gamma', .999), ('target_mode', 'current_player'),
                             ('return_mode', 'terminal_rank'), ('critic_arch', 'single_tower')):
            bad = json.loads(json.dumps(state))
            bad['oracle_critic_pretrain'][field] = value
            with self.assertRaises(ValueError):
                validate_critic_contract(bad, version=4, pts=[2, 1, 0, -3])
        with self.assertRaises(ValueError):
            validate_critic_contract(state, version=4, pts=[3, 1, -1, -3])
        state['steps'] = 0
        validate_critic_contract(state, version=4, pts=[2, 1, 0, -3], role='warm0')
        with self.assertRaisesRegex(ValueError, 'internal steps'):
            validate_critic_contract(state, version=4, pts=[2, 1, 0, -3], role='warm40k')
        state['steps'] = 40000
        state['optimizer'] = {'param_groups': [{'train_mode': True}]}
        with self.assertRaisesRegex(ValueError, 'train mode'):
            validate_critic_contract(state, version=4, pts=[2, 1, 0, -3], role='clean40k')
        state['optimizer']['param_groups'][0]['train_mode'] = False
        validate_critic_contract(state, version=4, pts=[2, 1, 0, -3], role='clean40k')

    def test_complete_native_log_required(self):
        events = [{'type': 'start_game', 'names': ['trainee', 'canonical', 'canonical', 'canonical']},
                  {'type': 'start_kyoku'}, {'type': 'end_kyoku'}, {'type': 'end_game'}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'game.json.gz'
            def write(rows):
                with gzip.open(path, 'wt', encoding='utf-8') as handle:
                    handle.write('\n'.join(json.dumps(e) for e in rows))
            write(events)
            self.assertEqual(verify_complete_log(path), 1)
            for rows in (events[:-1], [events[0], events[1], events[-1]], [],
                         [{'type': 'start_game', 'names': ['trainee'] * 4}, *events[1:]]):
                write(rows)
                with self.assertRaises(ValueError):
                    verify_complete_log(path)


class TrajectoryImputationOptionTests(unittest.TestCase):
    """Compile only the production method AST; no fake torch installations."""
    def setUp(self):
        path = ROOT / 'data/dataloader.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FileDatasetsIter')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'iter_game_trajectories')
        self.native = Mock()
        self.namespace = {'GameplayLoader': self.native,
                          'iter_loaded_gameplay_batches': lambda loader, files: iter(())}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), self.namespace)
        self.method = self.namespace['iter_game_trajectories']
        self.dataset = SimpleNamespace(value_reward_source='score_rank', version=4, oracle=True,
                                       player_names=['trainee'], excludes=None, track_opponent_states=False,
                                       track_danger_labels=False, track_regret_labels=False)

    def test_default_does_not_call_imputation_setter_or_change_native_protocol(self):
        self.assertEqual(list(self.method(self.dataset, [])), [])
        self.native.return_value.set_oracle_imputation_seed.assert_not_called()
        self.assertNotIn('trust_seed', self.native.call_args.kwargs)
        native_source = (ROOT.parent / 'libriichi/src/dataset/gameplay.rs').read_text()
        self.assertIn('trust_seed = false', native_source)

    def test_requested_seed_is_forwarded(self):
        list(self.method(self.dataset, [], oracle_imputation_seed=20260905))
        self.native.return_value.set_oracle_imputation_seed.assert_called_once_with(20260905)

    def test_invalid_seeds_fail_before_native_loader_created(self):
        for seed in (-1, 2**64, 1.5, True, '2'):
            with self.assertRaises(ValueError):
                list(self.method(self.dataset, [], oracle_imputation_seed=seed))
        self.native.assert_not_called()

    def test_missing_native_support_fails_closed(self):
        self.native.return_value = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, 'lacks fixed Oracle'):
            list(self.method(self.dataset, [], oracle_imputation_seed=1))


if __name__ == '__main__':
    unittest.main()

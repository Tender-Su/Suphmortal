"""Lightweight contract tests; do not imply torch/native integration coverage."""
import ast
import gzip
import hashlib
import subprocess
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mortal.core.artifacts import file_sha256
from mortal.research.frozen_actor_critic_probe import (
    calibration, parse_args, reserve_output, summarize_predictions,
    validate_critic_contract, verify_complete_log, shuffle_hidden_indices,
    validate_reuse_provenance, validate_reuse_model_identity, load_verified_rollout,
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
        self.assertFalse(args.shuffle_hidden)
        self.assertIsNone(args.reuse_rollout_dir)
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


class HiddenShuffleTests(unittest.TestCase):
    def test_shared_full_game_mapping_is_deterministic_and_has_no_fixed_points(self):
        kwargs = dict(seed=1000000, seed_key=2026093001, seat=2, shuffle_seed=20260930)
        for length in (2, 3, 31, 128, 301):
            indices = shuffle_hidden_indices(length, **kwargs)
            np.testing.assert_array_equal(np.sort(indices), np.arange(length))
            self.assertTrue(np.all(indices != np.arange(length)))
            for batch_size in (1, 7, 32, 256):
                other = shuffle_hidden_indices(length, **kwargs)
                np.testing.assert_array_equal(np.concatenate([other[i:i + batch_size] for i in range(0, length, batch_size)]), indices)
        self.assertFalse(np.array_equal(shuffle_hidden_indices(301, **kwargs),
                                       shuffle_hidden_indices(301, **{**kwargs, 'shuffle_seed': 9})))

    def test_impossible_derangement_fails_instead_of_identity(self):
        with self.assertRaises(ValueError):
            shuffle_hidden_indices(1, seed=1, seed_key=2, seat=3, shuffle_seed=4)


class RolloutReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'original'
        (self.source / 'games').mkdir(parents=True)
        self.provenance = {
            'schema': 1, 'status': 'running', 'weights_and_config': {
                'actor': {'sha256': 'actor'}, 'opponent': {'sha256': 'opponent'}},
            'native_sha256': 'native-init', 'native_extension_sha256': {'native.pyd': 'native-binary'},
            'torch_version': 'test', 'numpy_version': 'test', 'actor_explore_rate': 1.0,
            'opponent_explore_rate': 0.0, 'actor_agari_guard': False, 'opponent_agari_guard': True,
            'search_enabled': False, 'actor_oracle_guiding': False, 'probe_pts': [2, 1, 0, -3],
            'source_hashes': {'mortal/eval/engine.py': 'engine', 'mortal/eval/player.py': 'player',
                              'mortal/core/model.py': 'model', 'mortal/core/checkpoint_utils.py': 'checkpoint-utils'},
            'arguments': {'games': 8, 'seed_start': 1000, 'seed_key': 42, 'sampling_seed': 3,
                          'device': 'cuda', 'torch_threads': 1, 'rayon_threads': 4}}
        self.games = []
        for seed in (1000, 1001):
            for seat in range(4):
                path = self.source / 'games' / f'{seed}-{seat}.json.gz'
                names = ['canonical'] * 4
                names[seat] = 'trainee'
                with gzip.open(path, 'wt', encoding='utf-8') as handle:
                    handle.write('\n'.join(json.dumps(e) for e in [
                        {'type': 'start_game', 'names': names, 'seed': [seed, 42]},
                        {'type': 'start_kyoku'}, {'type': 'end_kyoku'}, {'type': 'end_game'}]))
                self.games.append({'log_path': str(path), 'seed': seed, 'seed_key': 42,
                                   'challenger_seat': seat, 'challenger_rank': 1,
                                   'kyoku_count': 1, 'sha256': file_sha256(path)})
        self.save()

    def save(self):
        (self.source / 'provenance.json').write_text(json.dumps(self.provenance))
        (self.source / 'outcomes.json').write_text(json.dumps(self.games))

    def load(self, *, fresh=None):
        return load_verified_rollout(self.source, self.provenance,
                                    load_games=lambda root, name: fresh if fresh is not None else self.games,
                                    duplicate_sets=lambda games: {(g['seed'], g['seed_key']) for g in games})

    def test_complete_running_rollout_can_be_reused_read_only(self):
        before = {p: file_sha256(p) for p in self.source.rglob('*') if p.is_file()}
        games, evidence = self.load()
        self.assertEqual(len(games), 8)
        self.assertEqual(evidence['source_status_at_read'], 'running')
        self.assertEqual(before, {p: file_sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_optional_timing_report_is_not_required_for_verified_rollout_reuse(self):
        self.provenance['timing_report'] = 'timing.json'
        self.save()
        self.assertFalse((self.source / 'timing.json').exists())
        games, _ = self.load()
        self.assertEqual(len(games), 8)

    def test_completed_rollout_requires_matching_outcomes_artifact_hash(self):
        self.provenance['status'] = 'complete'
        self.provenance['artifact_sha256'] = {'outcomes.json': 'wrong'}
        self.save()
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            self.load()

    def test_identity_sampling_native_and_source_mismatches_fail(self):
        mutations = [lambda p: p['weights_and_config']['actor'].update(sha256='wrong'),
                     lambda p: p['native_extension_sha256'].update({'native.pyd': 'wrong'}),
                     lambda p: p['arguments'].update(sampling_seed=4),
                     lambda p: p['arguments'].update(games=256),
                     lambda p: p.update(actor_agari_guard=True),
                     lambda p: p['source_hashes'].update({'mortal/eval/engine.py': 'wrong'})]
        for mutate in mutations:
            previous = json.loads(json.dumps(self.provenance))
            mutate(previous)
            with self.assertRaises(ValueError):
                validate_reuse_provenance(previous, self.provenance)

    def test_reuse_chains_are_rejected(self):
        previous = json.loads(json.dumps(self.provenance))
        previous['rollout_reuse'] = {'source_dir': 'older'}
        with self.assertRaisesRegex(ValueError, 'chain'):
            validate_reuse_provenance(previous, self.provenance)

    def test_modified_log_is_rejected(self):
        Path(self.games[0]['log_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'log hash'):
            self.load()

    def test_outside_or_duplicate_registered_logs_are_rejected(self):
        self.games[0]['log_path'] = str(self.source / 'outside.json.gz')
        self.save()
        with self.assertRaisesRegex(ValueError, 'original logs'):
            self.load()

    def test_missing_unregistered_and_outcome_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            self.load(fresh=self.games[:-1])
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            self.load(fresh=self.games + [self.games[0]])
        fresh = json.loads(json.dumps(self.games))
        fresh[0]['challenger_rank'] = 2
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.load(fresh=fresh)

    def test_wrong_seed_range_is_rejected(self):
        for game in self.games:
            game['seed'] += 10
        self.save()
        with self.assertRaisesRegex(ValueError, 'four-seat seed'):
            self.load()


class ModelSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.blob = b'unchanged model runtime'
        self.hashes = {path: hashlib.sha256(self.blob).hexdigest() for path in
                       ('mortal/core/model.py', 'mortal/core/checkpoint_utils.py')}
        self.current = {'source_hashes': self.hashes}
        self.previous = {'source_hashes': {}, 'source_commit': 'a' * 40, 'source_status': ''}

    def test_missing_legacy_hashes_require_matching_local_git_blobs(self):
        with patch('mortal.research.frozen_actor_critic_probe.subprocess.check_output', return_value=self.blob) as git:
            evidence = validate_reuse_model_identity(self.previous, self.current)
        self.assertEqual(git.call_count, 4)
        self.assertTrue(all(call.kwargs['env']['GIT_NO_LAZY_FETCH'] == '1' for call in git.call_args_list))
        self.assertTrue(all(call.args[0][0:3] == ['git', 'cat-file', 'blob'] for call in git.call_args_list))
        self.assertTrue(all(item['mode'] == 'local_git_blob_match' for item in evidence.values()))

    def test_missing_commit_unresolvable_commit_and_dirty_source_rejected(self):
        for change in ({'source_commit': ''}, {'source_commit': 'HEAD'}, {'source_status': ' M mortal/core/model.py'}):
            with self.assertRaises(ValueError), patch('mortal.research.frozen_actor_critic_probe.subprocess.check_output') as git:
                validate_reuse_model_identity({**self.previous, **change}, self.current)
            git.assert_not_called()
        with patch('mortal.research.frozen_actor_critic_probe.subprocess.check_output',
                   side_effect=subprocess.CalledProcessError(128, ['git'])):
            with self.assertRaisesRegex(ValueError, 'local Git'):
                validate_reuse_model_identity(self.previous, self.current)

    def test_git_blob_or_actual_current_file_mismatch_rejected(self):
        with patch('mortal.research.frozen_actor_critic_probe.subprocess.check_output', side_effect=[self.blob, b'changed']):
            with self.assertRaisesRegex(ValueError, 'blob mismatch'):
                validate_reuse_model_identity(self.previous, self.current)
        with patch('mortal.research.frozen_actor_critic_probe.subprocess.check_output', return_value=self.blob):
            with self.assertRaisesRegex(ValueError, 'blob mismatch'):
                validate_reuse_model_identity(self.previous, {'source_hashes': {**self.hashes, 'mortal/core/model.py': 'modified'}})

    def test_recorded_hashes_match_without_git_and_mismatch_fails(self):
        previous = {**self.previous, 'source_hashes': self.hashes}
        with patch('mortal.research.frozen_actor_critic_probe.subprocess.check_output') as git:
            validate_reuse_model_identity(previous, self.current)
            git.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            validate_reuse_model_identity({**previous, 'source_hashes': {**self.hashes, 'mortal/core/model.py': 'wrong'}}, self.current)
        with self.assertRaisesRegex(ValueError, 'hash missing'):
            validate_reuse_model_identity(previous, {'source_hashes': {}})


if __name__ == '__main__':
    unittest.main()

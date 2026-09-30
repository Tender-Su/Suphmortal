"""Real Torch integration tests, separately skipped when Torch/native are absent.

The optional final test decodes a real registered rollout on CPU. Set
MORTAL_CURRENT_POLICY_TEST_MANIFEST to the formal manifest; it never updates a
model, creates rollout data, or touches a GPU.
"""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
    import numpy as np
    from libriichi.consts import obs_shape
except ImportError as exc:
    RUNTIME_ERROR = str(exc)
else:
    RUNTIME_ERROR = ''


@unittest.skipIf(bool(RUNTIME_ERROR), 'Torch/native integration unavailable: ' + RUNTIME_ERROR)
class CurrentPolicyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mortal.online import pretrain_oracle_critic
        from mortal.data.oracle_value import OracleTerminalValueDataset
        cls.pretrain = pretrain_oracle_critic
        cls.dataset_type = OracleTerminalValueDataset

    def test_legacy_dataset_and_contract_are_unchanged(self):
        cfg = {'file_batch_size': 1, 'num_workers': 0}
        dataset = self.pretrain.make_dataset([], cfg, train=False)
        self.assertIsNone(dataset.player_names)
        self.assertIsNone(dataset.game_id_by_source)
        self.assertEqual(dataset.cluster_unit, 'source_game')
        self.assertNotIn('current_policy', self.pretrain.data_stream_signature(cfg))
        self.assertNotIn('current_policy', self.pretrain.training_contract(cfg, {'type': 'constant'}))

    def test_dataset_filter_and_stream_contract_are_bound(self):
        filename = str(Path('registered.json.gz').resolve())
        data = {'group_ids': {filename: 123}, 'contract': {'manifest_fingerprint': 'fixed'}}
        cfg = {'file_batch_size': 1, 'num_workers': 0, 'target_mode': 'all_players'}
        with patch.object(self.pretrain, 'current_policy_data', return_value=data):
            dataset = self.pretrain.make_dataset([filename], cfg, train=False)
            stream = self.pretrain.data_stream_signature(cfg)
            contract = self.pretrain.training_contract(cfg, {'type': 'constant'})
        self.assertEqual(dataset.player_names, ['trainee'])
        self.assertEqual(dataset.value_target_mode, 'all_players')
        self.assertEqual(dataset.cluster_unit, 'full_seed_key_four_seat_group')
        self.assertEqual(stream['current_policy'], data['contract'])
        self.assertEqual(contract['current_policy'], data['contract'])
        with self.assertRaises(ValueError):
            self.pretrain.validate_resume_data_stream({'data_progress': {'signature': stream}}, {})

    def test_missing_mapping_and_native_omission_fail_closed(self):
        filename = str(Path('missing.json.gz').resolve())
        kwargs = dict(version=4, file_list=[filename], pts=[2, 1, 0, -3],
                      player_names=['trainee'], value_target_mode='all_players', emit_game_id=True)
        with self.assertRaisesRegex(ValueError, 'missing manifest'):
            self.dataset_type(**kwargs, game_id_by_source={})
        dataset = self.dataset_type(**kwargs, game_id_by_source={filename: 123})
        with patch('mortal.data.oracle_value.iter_loaded_gameplay_batches', return_value=iter([])):
            with self.assertRaisesRegex(ValueError, 'omitted registered'):
                dataset.populate_buffer(None, [filename], [])
        with patch('mortal.data.oracle_value.iter_loaded_gameplay_batches', return_value=iter([(filename, [])])):
            with self.assertRaisesRegex(ValueError, 'one complete trainee'):
                dataset.populate_buffer(None, [filename], [])

    def test_four_games_share_each_cluster_and_ci_uses_two_groups(self):
        errors = torch.tensor([1.] * 4 + [3.] * 4)
        pred = errors[:, None].expand(-1, 4)
        target = torch.zeros_like(pred)
        parts = [self.pretrain.batch_metrics(pred, target, game_id=torch.tensor([11] * 4 + [22] * 4))]
        metrics = self.pretrain.finalize_metrics(parts, include_cluster_records=True)
        self.assertEqual(metrics['num_games'], 2)
        self.assertAlmostEqual(metrics['loss'], 5.0)
        self.assertAlmostEqual(metrics['loss_cluster_se'], 4.0)
        self.assertEqual(metrics['_adaptive_cluster_records']['primary_loss'], [[11, 4.0, 4], [22, 36.0, 4]])

    def test_evaluator_names_seed_group_unit_explicitly(self):
        errors = torch.tensor([1.] * 4 + [3.] * 4)
        obs = errors[:, None]
        dataset = torch.utils.data.TensorDataset(obs, obs, torch.zeros((8, 4)),
                                                torch.zeros(8), torch.tensor([11] * 4 + [22] * 4))
        dataset.cluster_unit = 'full_seed_key_four_seat_group'
        loader = torch.utils.data.DataLoader(dataset, batch_size=3)
        def predict(brain, head, visible, invisible, **kwargs):
            return visible.expand(-1, 4)
        with patch.object(self.pretrain, 'model_forward', side_effect=predict):
            result = self.pretrain.evaluate_modes(torch.nn.Identity(), torch.nn.Identity(), loader,
                                                 torch.device('cpu'), enable_amp=False, max_batches=0,
                                                 input_modes=['true'], include_cluster_records=True)['true']
        self.assertEqual(result['num_seed_groups'], 2)
        self.assertEqual(result['num_games_key_unit'], 'seed_groups')
        self.assertEqual(result['ci_cluster_unit'], 'full_seed_key_four_seat_group')
        self.assertAlmostEqual(result['seed_group_balanced_loss'], 5.0)

    @unittest.skipUnless(os.environ.get('MORTAL_CURRENT_POLICY_TEST_MANIFEST'),
                         'real registered rollout not supplied; set MORTAL_CURRENT_POLICY_TEST_MANIFEST')
    def test_real_registered_group_has_only_trainee_and_four_head_mc(self):
        from mortal.data.current_policy_manifest import load_current_policy_manifest
        filename = os.environ['MORTAL_CURRENT_POLICY_TEST_MANIFEST']
        data = load_current_policy_manifest(filename)
        payload = json.loads(Path(filename).read_text(encoding='utf-8'))
        games = [game for game in payload['games'] if game['split'] == 'dev'][:4]
        self.assertEqual(len({game['cluster_id'] for game in games}), 1)
        for game in games:
            dataset = self.dataset_type(
                version=4, file_list=[game['path']], pts=[2, 1, 0, -3],
                player_names=['trainee'], value_target_mode='all_players',
                return_mode='score_rank_mc', discount_gamma=1.0,
                shuffle_files=False, file_batch_size=1, num_epochs=1,
                oracle_imputation_seed=20260905, emit_game_id=True,
                game_id_by_source=data['group_ids'], rayon_num_threads=1,
            )
            count = 0
            for obs, hidden, target, player_id, group_id in dataset:
                count += 1
                self.assertEqual(obs.shape, (1012, 34))
                self.assertEqual(hidden.shape, (217, 34))
                self.assertEqual(target.shape, (4,))
                self.assertTrue(np.isfinite(target).all())
                self.assertAlmostEqual(float(target.sum()), 0.0)
                self.assertEqual(player_id, game['trainee_seat'])
                self.assertEqual(group_id, game['cluster_id'])
            self.assertGreater(count, 0)


if __name__ == '__main__':
    unittest.main()

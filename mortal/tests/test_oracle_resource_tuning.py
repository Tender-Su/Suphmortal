import logging
import unittest
from unittest.mock import patch

import torch

from mortal.data.oracle_value import deterministic_game_id
from mortal.online.pretrain_oracle_critic import evaluation_files, evaluation_thread_scope, oracle_worker_init_fn
from mortal.core.process_resources import configure_windows_high_qos
from mortal.online.train_online import online_gae_inference_batch_size


class OracleResourceTuningTests(unittest.TestCase):
    def test_qos_is_opt_in_without_loading_windows_api_by_default(self):
        with patch('mortal.core.process_resources.ctypes.WinDLL', create=True) as api:
            self.assertFalse(configure_windows_high_qos())
            api.assert_not_called()

    def test_qos_before_role_startup_does_not_configure_root_logging(self):
        root = logging.getLogger()
        handlers = root.handlers[:]
        try:
            root.handlers.clear()
            with patch('mortal.core.process_resources.os.name', 'nt'), patch(
                    'mortal.core.process_resources.ctypes.WinDLL', create=True), patch(
                    'mortal.core.process_resources.logging.basicConfig') as basic_config:
                self.assertTrue(configure_windows_high_qos(True))
                basic_config.assert_not_called()
                self.assertEqual(root.handlers, [])
        finally:
            root.handlers[:] = handlers

    def test_oracle_resource_worker_keeps_original_worker_initialization(self):
        calls = []
        with patch('mortal.online.pretrain_oracle_critic.configure_windows_high_qos',
                   side_effect=lambda enabled: calls.append(('qos', enabled))), patch(
                   'mortal.online.pretrain_oracle_critic.worker_init_fn',
                   side_effect=lambda worker_id: calls.append(('worker', worker_id))):
            oracle_worker_init_fn(3, windows_high_qos=True)
        self.assertEqual(calls, [('qos', True), ('worker', 3)])

    def test_gae_physical_inference_block_rejects_nonpositive_values(self):
        self.assertEqual(online_gae_inference_batch_size({}), 2048)
        self.assertEqual(online_gae_inference_batch_size({'online': {'gae_inference_batch_size': 512}}), 512)
        for invalid in (0, -1):
            with self.assertRaisesRegex(ValueError, 'positive'):
                online_gae_inference_batch_size({'online': {'gae_inference_batch_size': invalid}})

    def test_subset_matches_late_game_id_filter_without_reordering_duplicates(self):
        files = [f'C:/train/game_{i}.json' for i in range(40)]
        files += files[10:20]
        chosen = evaluation_files(files, {'eval_prefilter_games': True},
                                  game_id_modulus=5, game_id_remainders=(0, 2))
        expected = [f for f in files if deterministic_game_id(f) % 5 in (0, 2)]
        self.assertEqual(chosen, expected)
        self.assertTrue(chosen)
        self.assertLess(len(chosen), len(files))

    def test_full_evaluation_is_not_replaced_by_monitor_subset(self):
        files = ['C:/train/game_0.json', 'C:/train/game_1.events.zst']
        self.assertIs(evaluation_files(files, {'eval_prefilter_games': True}), files)
        self.assertIs(evaluation_files(files, {}, game_id_modulus=5, game_id_remainders=(0,)), files)

    def test_bounded_batches_and_shuffle_keep_original_batch_composition(self):
        files = [f'C:/train/game_{i}.json' for i in range(20)]
        cfg = {'eval_prefilter_games': True}
        self.assertIs(evaluation_files(files, cfg, game_id_modulus=5,
                                       game_id_remainders=(0,), max_batches=1), files)
        self.assertIs(evaluation_files(files, cfg, game_id_modulus=5,
                                       game_id_remainders=(0,), input_modes=('true', 'shuffled')), files)

    def test_eval_threads_restored_after_pause_or_error(self):
        before = torch.get_num_threads()
        @evaluation_thread_scope
        def failing_evaluation():
            self.assertEqual(torch.get_num_threads(), 1)
            raise InterruptedError('pause at batch boundary')
        with patch('mortal.online.pretrain_oracle_critic.config',
                   {'oracle_critic_pretrain': {'eval_torch_num_threads': 1}}):
            with self.assertRaises(InterruptedError):
                failing_evaluation()
        self.assertEqual(torch.get_num_threads(), before)


if __name__ == '__main__':
    unittest.main()

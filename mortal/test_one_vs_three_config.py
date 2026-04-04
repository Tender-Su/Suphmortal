import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import one_vs_three


class ResolveSeedCountTest(unittest.TestCase):
    def test_env_override_wins(self):
        cfg = {'games_per_iter': 2000}
        with mock.patch.dict(os.environ, {'MORTAL_1V3_SEED_COUNT': '123'}, clear=False):
            seed_count, source = one_vs_three.resolve_seed_count(cfg)
        self.assertEqual(123, seed_count)
        self.assertIn('MORTAL_1V3_SEED_COUNT', source)

    def test_machine_override_uses_seed_count(self):
        cfg = {
            'games_per_iter': 2000,
            'machine_overrides': {
                'TESTHOST': {
                    'seed_count': 321,
                }
            },
        }
        with (
            mock.patch.dict(os.environ, {'COMPUTERNAME': 'TESTHOST'}, clear=False),
            mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=False),
        ):
            seed_count, source = one_vs_three.resolve_seed_count(cfg)
        self.assertEqual(321, seed_count)
        self.assertIn('machine override', source)

    def test_builtin_gpu_default_applies(self):
        cfg = {
            'games_per_iter': 2000,
            'challenger': {
                'device': 'cuda:0',
            },
        }
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(
                one_vs_three.torch.cuda,
                'get_device_name',
                return_value='NVIDIA GeForce RTX 4060 Laptop GPU',
            ),
        ):
            seed_count, source = one_vs_three.resolve_seed_count(cfg)
        self.assertEqual(640, seed_count)
        self.assertIn('built-in gpu default', source)

    def test_falls_back_to_games_per_iter(self):
        cfg = {'games_per_iter': 2000}
        with mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=False):
            seed_count, source = one_vs_three.resolve_seed_count(cfg)
        self.assertEqual(500, seed_count)
        self.assertIn('[1v3]', source)


class ResolveShardCountTest(unittest.TestCase):
    def test_env_override_wins(self):
        cfg = {}
        with mock.patch.dict(os.environ, {'MORTAL_1V3_SHARD_COUNT': '3'}, clear=False):
            shard_count, source = one_vs_three.resolve_shard_count(cfg)
        self.assertEqual(3, shard_count)
        self.assertIn('MORTAL_1V3_SHARD_COUNT', source)

    def test_builtin_gpu_default_applies(self):
        cfg = {
            'challenger': {
                'device': 'cuda:0',
            },
        }
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(
                one_vs_three.torch.cuda,
                'get_device_name',
                return_value='NVIDIA GeForce RTX 5070 Ti',
            ),
        ):
            shard_count, source = one_vs_three.resolve_shard_count(cfg)
        self.assertEqual(4, shard_count)
        self.assertIn('built-in gpu default', source)

    def test_falls_back_to_implicit_single_shard(self):
        with mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=False):
            shard_count, source = one_vs_three.resolve_shard_count({})
        self.assertEqual(1, shard_count)
        self.assertIn('implicit default', source)


class PlanShardsTest(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual([384, 384], one_vs_three.plan_shards(768, 2))

    def test_remainder_split(self):
        self.assertEqual([3, 2], one_vs_three.plan_shards(5, 2))

    def test_clamps_to_total_seed_count(self):
        self.assertEqual([1, 1], one_vs_three.plan_shards(2, 4))


class ResolveEnableMetadataTest(unittest.TestCase):
    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {'MORTAL_1V3_ENABLE_METADATA': '0'}, clear=False):
            enabled = one_vs_three.resolve_enable_metadata({'enable_metadata': True, 'log_dir': './logs'})
        self.assertFalse(enabled)

    def test_prefers_explicit_config(self):
        self.assertTrue(one_vs_three.resolve_enable_metadata({'enable_metadata': True, 'log_dir': ''}))
        self.assertFalse(one_vs_three.resolve_enable_metadata({'enable_metadata': False, 'log_dir': './logs'}))

    def test_falls_back_to_log_dir(self):
        self.assertTrue(one_vs_three.resolve_enable_metadata({'log_dir': './logs'}))
        self.assertFalse(one_vs_three.resolve_enable_metadata({'log_dir': ''}))


class RunMainTest(unittest.TestCase):
    def test_single_shard_reuses_loaded_engines(self):
        cfg = {
            'iters': 3,
            'disable_progress_bar': True,
        }
        fake_eval_context = object()
        with (
            mock.patch.dict(one_vs_three.config, {'1v3': cfg}, clear=True),
            mock.patch.object(one_vs_three, 'resolve_seed_count', return_value=(2, 'test seed source')),
            mock.patch.object(one_vs_three, 'resolve_shard_count', return_value=(1, 'test shard source')),
            mock.patch.object(one_vs_three, 'load_eval_engines', return_value=fake_eval_context) as mock_load_eval_engines,
            mock.patch.object(one_vs_three, 'run_eval_once', return_value=[1, 1, 1, 1]) as mock_run_eval_once,
            mock.patch('builtins.print'),
        ):
            one_vs_three.run_main(args=None)

        mock_load_eval_engines.assert_called_once_with(cfg)
        self.assertEqual(3, mock_run_eval_once.call_count)
        for call in mock_run_eval_once.call_args_list:
            self.assertIs(fake_eval_context, call.kwargs['eval_context'])

    def test_multi_shard_reuses_worker_pool_across_iterations(self):
        cfg = {
            'iters': 3,
            'disable_progress_bar': True,
        }
        fake_pool = {'workers': []}
        with (
            mock.patch.dict(one_vs_three.config, {'1v3': cfg}, clear=True),
            mock.patch.object(one_vs_three, 'resolve_seed_count', return_value=(8, 'test seed source')),
            mock.patch.object(one_vs_three, 'resolve_shard_count', return_value=(2, 'test shard source')),
            mock.patch.object(one_vs_three, 'start_persistent_shard_workers', return_value=fake_pool) as mock_start_workers,
            mock.patch.object(
                one_vs_three,
                'run_sharded_iteration_with_workers',
                return_value=([2, 2, 2, 2], 2.5, 0.0, Path(r'C:\tmp\runtime')),
            ) as mock_run_iteration,
            mock.patch.object(one_vs_three, 'stop_persistent_shard_workers') as mock_stop_workers,
            mock.patch('builtins.print'),
        ):
            one_vs_three.run_main(args=None)

        mock_start_workers.assert_called_once_with(
            cfg=cfg,
            shard_count=2,
            disable_progress_bar=True,
        )
        self.assertEqual(3, mock_run_iteration.call_count)
        for call in mock_run_iteration.call_args_list:
            self.assertIs(fake_pool, call.kwargs['shard_worker_pool'])
        mock_stop_workers.assert_called_once_with(fake_pool)


class RunShardedIterationTest(unittest.TestCase):
    def test_keyboard_interrupt_terminates_child_processes(self):
        class FakeProcess:
            def __init__(self, *, wait_raises=False):
                self.wait_raises = wait_raises
                self.terminated = False
                self.killed = False

            def wait(self, timeout=None):
                if timeout is None and self.wait_raises:
                    raise KeyboardInterrupt
                self.terminated = True
                return 0

            def poll(self):
                return None if not (self.terminated or self.killed) else 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True
                self.terminated = True

        first_process = FakeProcess(wait_raises=True)
        second_process = FakeProcess()

        with (
            mock.patch.object(one_vs_three, 'normalize_child_env', return_value={}),
            mock.patch.object(one_vs_three, 'build_worker_command', return_value=['python', 'worker']),
            mock.patch.object(one_vs_three.subprocess, 'Popen', side_effect=[first_process, second_process]),
        ):
            with self.assertRaises(KeyboardInterrupt):
                one_vs_three.run_sharded_iteration(
                    cfg={},
                    iter_index=0,
                    seed_start=10000,
                    seed_key=123,
                    seed_count=2,
                    shard_seed_counts=[1, 1],
                    log_dir=None,
                )

        self.assertTrue(first_process.terminated)
        self.assertTrue(second_process.terminated)


if __name__ == '__main__':
    unittest.main()

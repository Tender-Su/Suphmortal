import os
import sys
import threading
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine
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

    def test_desktop_builtin_gpu_default_applies(self):
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
                return_value='NVIDIA GeForce RTX 5070 Ti',
            ),
        ):
            seed_count, source = one_vs_three.resolve_seed_count(cfg)
        self.assertEqual(1024, seed_count)
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

    def test_laptop_builtin_gpu_default_applies(self):
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
                return_value='NVIDIA GeForce RTX 4060 Laptop GPU',
            ),
        ):
            shard_count, source = one_vs_three.resolve_shard_count(cfg)
        self.assertEqual(3, shard_count)
        self.assertIn('built-in gpu default', source)

    def test_falls_back_to_implicit_single_shard(self):
        with mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=False):
            shard_count, source = one_vs_three.resolve_shard_count({})
        self.assertEqual(1, shard_count)
        self.assertIn('implicit default', source)


class ResolveBrokerShardCountTest(unittest.TestCase):
    def test_inherits_base_shard_count_by_default(self):
        shard_count, source = one_vs_three.resolve_broker_shard_count({}, 4)
        self.assertEqual(4, shard_count)
        self.assertIn('inherit', source)

    def test_desktop_builtin_broker_shard_default_applies(self):
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
            shard_count, source = one_vs_three.resolve_broker_shard_count(cfg, 4)
        self.assertEqual(3, shard_count)
        self.assertIn('built-in broker gpu default', source)


class PlanShardsTest(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual([384, 384], one_vs_three.plan_shards(768, 2))

    def test_remainder_split(self):
        self.assertEqual([3, 2], one_vs_three.plan_shards(5, 2))

    def test_clamps_to_total_seed_count(self):
        self.assertEqual([1, 1], one_vs_three.plan_shards(2, 4))


class ResolveExecutionModeTest(unittest.TestCase):
    def test_auto_prefers_broker_on_shared_cuda_multishard(self):
        cfg = {
            'execution_mode': 'auto',
            'challenger': {'device': 'cuda:0'},
            'champion': {'device': 'cuda:0'},
        }
        with mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=True):
            mode, source = one_vs_three.resolve_execution_mode(cfg, effective_shard_count=3)
        self.assertEqual('broker', mode)
        self.assertIn('auto -> broker', source)

    def test_auto_falls_back_to_process_without_cuda(self):
        cfg = {
            'execution_mode': 'auto',
            'challenger': {'device': 'cpu'},
            'champion': {'device': 'cpu'},
        }
        with mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=False):
            mode, source = one_vs_three.resolve_execution_mode(cfg, effective_shard_count=3)
        self.assertEqual('process', mode)
        self.assertIn('auto -> process', source)

    def test_explicit_broker_single_shard_falls_back_to_direct_path(self):
        cfg = {
            'execution_mode': 'broker',
            'challenger': {'device': 'cuda:0'},
            'champion': {'device': 'cuda:0'},
        }
        with mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=True):
            mode, source = one_vs_three.resolve_execution_mode(cfg, effective_shard_count=1)
        self.assertEqual('process', mode)
        self.assertIn('single-shard', source)

    def test_explicit_broker_uses_broker_shard_count_before_mode_selection(self):
        cfg = {
            'execution_mode': 'broker',
            'challenger': {'device': 'cuda:0'},
            'champion': {'device': 'cuda:0'},
        }
        with mock.patch.object(one_vs_three, 'can_use_local_broker', return_value=(True, 'single-device cuda:0')):
            mode, source = one_vs_three.resolve_execution_mode(
                cfg,
                effective_shard_count=1,
                broker_effective_shard_count=3,
            )
        self.assertEqual('broker', mode)
        self.assertIn('config/env broker', source)

    def test_auto_uses_broker_shard_count_before_mode_selection(self):
        cfg = {
            'execution_mode': 'auto',
            'challenger': {'device': 'cuda:0'},
            'champion': {'device': 'cuda:0'},
        }
        with mock.patch.object(one_vs_three, 'can_use_local_broker', return_value=(True, 'single-device cuda:0')):
            mode, source = one_vs_three.resolve_execution_mode(
                cfg,
                effective_shard_count=1,
                broker_effective_shard_count=3,
            )
        self.assertEqual('broker', mode)
        self.assertIn('auto -> broker', source)

    def test_auto_treats_cuda_and_cuda_zero_as_same_device(self):
        cfg = {
            'execution_mode': 'auto',
            'challenger': {'device': 'cuda'},
            'champion': {'device': 'cuda:0'},
        }
        with (
            mock.patch.object(one_vs_three.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(one_vs_three.torch.cuda, 'current_device', return_value=0),
        ):
            mode, source = one_vs_three.resolve_execution_mode(cfg, effective_shard_count=3)
        self.assertEqual('broker', mode)
        self.assertIn('auto -> broker', source)


class BrokerTest(unittest.TestCase):
    def test_broker_client_matches_direct_engine_and_batches_requests(self):
        class FakeBatchEngine:
            def __init__(self):
                self.batch_sizes = []
                self.lock = threading.Lock()

            def react_batch(self, obs, masks, invisible_obs):
                obs, masks, invisible_obs = engine.coerce_batch_inputs(obs, masks, invisible_obs)
                with self.lock:
                    self.batch_sizes.append(int(obs.shape[0]))
                actions = [int(value) for value in obs[:, 0, 0]]
                q_out = (obs[:, :1, 0].astype(np.float32) / 10.0).repeat(46, axis=1).tolist()
                masks_out = masks.astype(bool).tolist()
                is_greedy = [True] * obs.shape[0]
                return actions, q_out, masks_out, is_greedy

            def react_batch_action_only(self, obs, masks, invisible_obs):
                obs, _, _ = engine.coerce_batch_inputs(obs, masks, invisible_obs)
                actions = [int(value) for value in obs[:, 0, 0]]
                return actions, actions

        fake_engine = FakeBatchEngine()
        broker = one_vs_three.InferenceBroker(
            engine_map={
                'challenger': fake_engine,
                'champion': fake_engine,
            },
            engine_metadata={
                'challenger': {
                    'name': 'challenger',
                    'is_oracle': False,
                    'version': 4,
                    'enable_quick_eval': True,
                    'enable_rule_based_agari_guard': True,
                    'enable_metadata': True,
                },
                'champion': {
                    'name': 'champion',
                    'is_oracle': False,
                    'version': 4,
                    'enable_quick_eval': True,
                    'enable_rule_based_agari_guard': True,
                    'enable_metadata': True,
                },
            },
            batch_window_ms=20.0,
        )
        client = broker.make_eval_context()['engine_chal']
        barrier = threading.Barrier(3)
        results = []

        def run_request(base_value, batch_size):
            obs = np.full((batch_size, 1012, 34), base_value, dtype=np.float32)
            masks = np.ones((batch_size, 46), dtype=np.bool_)
            barrier.wait()
            results.append(client.react_batch(obs, masks, None))

        thread_a = threading.Thread(target=run_request, args=(3.0, 2))
        thread_b = threading.Thread(target=run_request, args=(7.0, 3))
        thread_a.start()
        thread_b.start()
        barrier.wait()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)
        broker.close()

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(thread_b.is_alive())
        self.assertEqual(2, len(results))
        flat_actions = sorted(action for payload in results for action in payload[0])
        self.assertEqual([3, 3, 7, 7, 7], flat_actions)
        self.assertGreaterEqual(max(fake_engine.batch_sizes), 2)

    def test_closed_broker_rejects_new_requests(self):
        fake_engine = mock.Mock()
        broker = one_vs_three.InferenceBroker(
            engine_map={
                'challenger': fake_engine,
                'champion': fake_engine,
            },
            engine_metadata={
                'challenger': {
                    'name': 'challenger',
                    'is_oracle': False,
                    'version': 4,
                    'enable_quick_eval': True,
                    'enable_rule_based_agari_guard': True,
                    'enable_metadata': True,
                },
                'champion': {
                    'name': 'champion',
                    'is_oracle': False,
                    'version': 4,
                    'enable_quick_eval': True,
                    'enable_rule_based_agari_guard': True,
                    'enable_metadata': True,
                },
            },
            batch_window_ms=0.0,
        )
        broker.close()
        client = broker.make_eval_context()['engine_chal']
        obs = np.zeros((1, 1012, 34), dtype=np.float32)
        masks = np.ones((1, 46), dtype=np.bool_)

        with self.assertRaisesRegex(RuntimeError, 'closed'):
            client.react_batch(obs, masks, None)


class StopPersistentShardWorkersTest(unittest.TestCase):
    def test_broker_close_is_deferred_while_worker_thread_is_still_alive(self):
        class FakeWorkerThread:
            def __init__(self):
                self.join_calls = []

            def join(self, timeout=None):
                self.join_calls.append(timeout)

            def is_alive(self):
                return True

        class FakeReaperThread:
            def __init__(self, *, target, args, name, daemon):
                self.target = target
                self.args = args
                self.name = name
                self.daemon = daemon
                self.started = False

            def start(self):
                self.started = True

        created_reapers = []

        def build_reaper(*args, **kwargs):
            reaper = FakeReaperThread(**kwargs)
            created_reapers.append(reaper)
            return reaper

        broker = mock.Mock()
        task_queue = mock.Mock()
        worker_thread = FakeWorkerThread()
        shard_worker_pool = {
            'mode': 'broker',
            'workers': [
                {
                    'thread': worker_thread,
                    'task_queue': task_queue,
                }
            ],
            'broker': broker,
        }

        with mock.patch.object(one_vs_three.threading, 'Thread', side_effect=build_reaper):
            one_vs_three.stop_persistent_shard_workers(shard_worker_pool)

        task_queue.put.assert_called_once_with({'kind': 'stop'})
        broker.close.assert_not_called()
        self.assertEqual([10], worker_thread.join_calls)
        self.assertEqual(1, len(created_reapers))
        self.assertTrue(created_reapers[0].started)
        self.assertIs(created_reapers[0], shard_worker_pool['broker_close_reaper'])


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
            mock.patch.object(one_vs_three, 'resolve_execution_mode', return_value=('process', 'test mode source')),
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
        fake_pool = {'mode': 'process'}
        with (
            mock.patch.dict(one_vs_three.config, {'1v3': cfg}, clear=True),
            mock.patch.object(one_vs_three, 'resolve_seed_count', return_value=(8, 'test seed source')),
            mock.patch.object(one_vs_three, 'resolve_shard_count', return_value=(2, 'test shard source')),
            mock.patch.object(one_vs_three, 'resolve_execution_mode', return_value=('process', 'test mode source')),
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
            execution_mode='process',
            broker=None,
        )
        self.assertEqual(3, mock_run_iteration.call_count)
        for call in mock_run_iteration.call_args_list:
            self.assertIs(fake_pool, call.kwargs['shard_worker_pool'])
        mock_stop_workers.assert_called_once_with(fake_pool)

    def test_multi_shard_broker_mode_uses_shared_broker(self):
        cfg = {
            'iters': 2,
            'disable_progress_bar': True,
            'execution_mode': 'broker',
        }
        fake_pool = {'mode': 'broker'}
        fake_broker = object()
        with (
            mock.patch.dict(one_vs_three.config, {'1v3': cfg}, clear=True),
            mock.patch.object(one_vs_three, 'resolve_seed_count', return_value=(8, 'test seed source')),
            mock.patch.object(one_vs_three, 'resolve_shard_count', return_value=(2, 'test shard source')),
            mock.patch.object(one_vs_three, 'resolve_execution_mode', return_value=('broker', 'test mode source')),
            mock.patch.object(one_vs_three, 'resolve_broker_shard_count', return_value=(3, 'test broker shard source')),
            mock.patch.object(one_vs_three.InferenceBroker, 'from_cfg', return_value=fake_broker) as mock_build_broker,
            mock.patch.object(one_vs_three, 'start_persistent_shard_workers', return_value=fake_pool) as mock_start_workers,
            mock.patch.object(
                one_vs_three,
                'run_sharded_iteration_with_workers',
                return_value=([2, 2, 2, 2], 2.5, 0.0, Path(r'C:\tmp\runtime')),
            ),
            mock.patch.object(one_vs_three, 'stop_persistent_shard_workers') as mock_stop_workers,
            mock.patch('builtins.print'),
        ):
            one_vs_three.run_main(args=None)

        mock_build_broker.assert_called_once_with(cfg)
        mock_start_workers.assert_called_once_with(
            cfg=cfg,
            shard_count=3,
            disable_progress_bar=True,
            execution_mode='broker',
            broker=fake_broker,
        )
        mock_stop_workers.assert_called_once_with(fake_pool)

    def test_broker_shard_count_can_promote_single_base_shard_to_broker_mode(self):
        cfg = {
            'iters': 2,
            'disable_progress_bar': True,
            'execution_mode': 'broker',
        }
        fake_pool = {'mode': 'broker'}
        fake_broker = object()
        with (
            mock.patch.dict(one_vs_three.config, {'1v3': cfg}, clear=True),
            mock.patch.object(one_vs_three, 'resolve_seed_count', return_value=(8, 'test seed source')),
            mock.patch.object(one_vs_three, 'resolve_shard_count', return_value=(1, 'base shard source')),
            mock.patch.object(one_vs_three, 'resolve_broker_shard_count', return_value=(3, 'broker shard source')),
            mock.patch.object(one_vs_three, 'can_use_local_broker', return_value=(True, 'single-device cuda:0')),
            mock.patch.object(one_vs_three.InferenceBroker, 'from_cfg', return_value=fake_broker) as mock_build_broker,
            mock.patch.object(one_vs_three, 'start_persistent_shard_workers', return_value=fake_pool) as mock_start_workers,
            mock.patch.object(
                one_vs_three,
                'run_sharded_iteration_with_workers',
                return_value=([2, 2, 2, 2], 2.5, 0.0, Path(r'C:\tmp\runtime')),
            ),
            mock.patch.object(one_vs_three, 'stop_persistent_shard_workers') as mock_stop_workers,
            mock.patch('builtins.print'),
        ):
            one_vs_three.run_main(args=None)

        mock_build_broker.assert_called_once_with(cfg)
        mock_start_workers.assert_called_once_with(
            cfg=cfg,
            shard_count=3,
            disable_progress_bar=True,
            execution_mode='broker',
            broker=fake_broker,
        )
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

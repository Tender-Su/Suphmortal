from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import torch

from mortal.research.runtime_profile import PhaseTimes, ResourceSampler, batch_fingerprints, closure_patch


class RuntimeProfileTests(unittest.TestCase):
    def test_scheduling_probe_ticks_and_returns_operation_result(self):
        from scripts.profile_sl_1v3_runtime import thread_schedule_during
        marker = object()

        def operation():
            time.sleep(0.15)
            return marker

        result, report = thread_schedule_during(operation, interval=0.01)
        self.assertIs(result, marker)
        self.assertGreater(report['ticks_during_call'], 0)
        self.assertTrue(report['sentinel_stopped'])
        self.assertLessEqual(report['max_gap_s'], report['elapsed_s'])

    def test_scheduling_probe_cleans_up_after_failure(self):
        from scripts.profile_sl_1v3_runtime import thread_schedule_during
        with self.assertRaisesRegex(RuntimeError, 'operation failed'):
            thread_schedule_during(Mock(side_effect=RuntimeError('operation failed')))
        self.assertFalse(any(t.name == 'native-scheduling-sentinel' for t in threading.enumerate()))
        with self.assertRaises(ValueError):
            thread_schedule_during(lambda: None, interval=0)

    def test_loader_scheduling_checks_capacity_before_output_or_native_work(self):
        from scripts.profile_sl_1v3_runtime import profile_loader_scheduling
        with patch('scripts.profile_sl_1v3_runtime.new_output') as output, \
             patch('scripts.profile_sl_1v3_runtime.windows_memory_snapshot',
                   return_value={'available_ram_bytes': 11 * 2**30}):
            with self.assertRaisesRegex(ValueError, 'four files'):
                profile_loader_scheduling(Mock(files=5))
            with self.assertRaisesRegex(ValueError, 'RAM headroom'):
                profile_loader_scheduling(Mock(files=1))
            output.assert_not_called()

    def test_eval_capacity_requires_measured_adjacent_tier_and_headroom(self):
        from scripts.profile_sl_1v3_runtime import validate_eval_capacity
        proof = {'mode': 'formal_inference', 'timing_mode': 'throughput',
                 'protocol_fingerprint': 'frozen', 'ordered_game_events_equal': True,
                 'chunk_seeds': 64, 'games': 256, 'cuda_peak_reserved_bytes': 220 * 2**20,
                 'resources': {'process_lifetime_peak_commit_bytes': 3 * 2**30}}
        args = dict(chunk_seeds=128, protocol_fingerprint='frozen', gpu_free_bytes=5 * 2**30,
                    gpu_total_bytes=8 * 2**30, ram_free_bytes=24 * 2**30, gpu_fraction=0.25)
        self.assertGreater(validate_eval_capacity(proof, **args)['projected_gpu_bytes'], 0)
        for change in ({'chunk_seeds': 256}, {'ram_free_bytes': 18 * 2**30},
                       {'gpu_free_bytes': 2 * 2**30}, {'protocol_fingerprint': 'other'}):
            with self.assertRaises(ValueError):
                validate_eval_capacity(proof, **dict(args, **change))
        proof['ordered_game_events_equal'] = False
        with self.assertRaises(ValueError):
            validate_eval_capacity(proof, **args)

    def test_throughput_excludes_hash_pass_and_requires_exact_metrics(self):
        from scripts.profile_sl_1v3_runtime import ValidationOnlyProbe

        @contextmanager
        def hashing(hashes):
            hashes.extend(['one', 'two'])
            yield

        with tempfile.TemporaryDirectory() as output, \
             patch('scripts.profile_sl_1v3_runtime.validation_input_hashes', side_effect=hashing), \
             patch('torch.cuda.synchronize'):
            record = {}
            evaluate = Mock(return_value=({'policy_loss': 0.5}, 1))
            probe = ValidationOnlyProbe(['one-file'], Path(output), record,
                                        timing_mode='throughput', repeats=2)
            probe.observe(0, evaluate, None, None, 0)
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(len(record['evaluation_times_s']), 2)
            self.assertEqual(record['samples'], 2)
            self.assertEqual(record['phases'], {})
            evaluate.side_effect = [({'policy_loss': 0.5}, 1), ({'policy_loss': 0.6}, 1)]
            with self.assertRaisesRegex(RuntimeError, 'changed metrics'):
                probe.observe(0, evaluate, None, None, 0)

    @unittest.skipUnless(os.name == 'nt', 'Windows process counters')
    def test_resource_sampler_has_no_optional_dependency_and_stops(self):
        with ResourceSampler() as sampler:
            self.assertGreater(sampler.samples[0]['available_ram_bytes'], 0)
        self.assertFalse(sampler.thread.is_alive())
        self.assertGreater(sampler.report()['max_rss_bytes'], 0)

    def test_exclusive_timers_do_not_double_count(self):
        values = iter([0.0, 1.0, 3.0, 5.0])
        phases = PhaseTimes(clock=lambda: next(values))
        with phases.measure('outer'):
            with phases.measure('inner'):
                pass
        self.assertEqual(phases.report()['outer']['exclusive_s'], 3.0)
        self.assertEqual(phases.report()['outer']['inclusive_s'], 5.0)
        self.assertEqual(phases.report()['inner']['exclusive_s'], 2.0)

    def test_closure_patch_restores_on_error(self):
        def create():
            value = lambda: 2
            return lambda: value()
        function = create()
        with self.assertRaises(RuntimeError):
            with ExitStack() as stack:
                closure_patch(stack, function, 'value', lambda: 7)
                self.assertEqual(function(), 7)
                raise RuntimeError('test')
        self.assertEqual(function(), 2)

    def test_every_field_and_order_are_bound(self):
        batch = [torch.arange(12).reshape(3, 4), torch.tensor([1, 0, 1], dtype=torch.bool)]
        hashes = batch_fingerprints(batch)
        self.assertEqual(hashes, batch_fingerprints([value.clone() for value in batch]))
        changed = [value.clone() for value in batch]
        changed[1][1] = True
        other = batch_fingerprints(changed)
        self.assertEqual(hashes[0], other[0])
        self.assertNotEqual(hashes[1], other[1])
        self.assertEqual(hashes[::-1], batch_fingerprints([value.flip(0) for value in batch]))


if __name__ == '__main__':
    unittest.main()

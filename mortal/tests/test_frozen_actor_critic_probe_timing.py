"""CPU timing/orchestration tests, not torch/native or GPU integration tests."""
import ast
from contextlib import nullcontext, redirect_stdout
import io
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mortal.research import frozen_actor_critic_probe as probe


class ProbePhaseTimingTests(unittest.TestCase):
    def test_monotonic_exclusive_repeated_phases_and_frozen_total(self):
        with patch.object(probe.time, 'perf_counter', side_effect=[10, 12, 15, 19, 24]), \
                patch.object(probe.time, 'time', side_effect=AssertionError('not a duration clock')):
            timing = probe.ProbePhaseTimings()
            timing.switch('critic_inference')
            timing.switch('aggregation_statistics')
            timing.switch('critic_inference')
            timing.finish()
            result = timing.report()
            self.assertEqual(timing.report(), result)
        self.assertEqual(result['total_seconds'], 14)
        self.assertEqual(result['phase_seconds'], {
            'preflight_input_fingerprint_model_load': 2,
            'critic_inference': 8, 'aggregation_statistics': 4})
        self.assertEqual(sum(result['phase_seconds'].values()), result['total_seconds'])
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['incomplete_phase'])
        with self.assertRaisesRegex(RuntimeError, 'already finished'):
            timing.switch('too_late')

    def test_failure_retains_only_completed_intervals_and_labels_active_work(self):
        with patch.object(probe.time, 'perf_counter', side_effect=[10, 12, 17]):
            timing = probe.ProbePhaseTimings()
            timing.switch('rollout_reuse_validation')
            result = timing.report()
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['phase_seconds'], {'preflight_input_fingerprint_model_load': 2})
        self.assertEqual(result['incomplete_phase'], {'name': 'rollout_reuse_validation', 'seconds': 5})
        self.assertEqual(result['total_seconds'], 7)
        self.assertNotIn('arena_generation_validation', result['phase_seconds'])


class HostTensor:
    """Minimal host result used only by the dependency-free orchestration harness."""
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class HostModel:
    def eval(self):
        return self

    def requires_grad_(self, enabled):
        return self

    def load_state_dict(self, state, *, strict):
        pass

    def cpu(self):
        return self

    def to(self, device):
        return self

    def __call__(self, obs, **kwargs):
        return obs


class ProbeTimingOrchestrationTests(unittest.TestCase):
    """Run the production control flow with explicit local host dependency stubs.

    Only run()'s imports are removed from its AST, just like the existing pure
    function tests. No fake torch/native packages are installed or imported.
    """
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.argv = []
        for name in ('config', 'actor', 'opponent', *probe.CRITICS):
            path = self.root / name
            path.write_text(name)
            self.argv.extend(['--' + name, str(path)])
        self.argv += ['--games', '8', '--seed-start', '1000', '--seed-key', '42',
                      '--sampling-seed', '3', '--batch-size', '2', '--device', 'cpu']
        self.games = []
        for seed in (1000, 1001):
            for seat in range(4):
                path = self.root / f'{seed}-{seat}.json.gz'
                path.write_bytes(b'host fixture, not a native game')
                self.games.append({'log_path': str(path), 'seed': seed, 'seed_key': 42,
                                   'challenger_seat': seat, 'challenger_rank': 1, 'kyoku_count': 1,
                                   'sha256': probe.file_sha256(path)})
        (self.root / 'outcomes.json').write_text(json.dumps(self.games))
        cfg = {'control': {'version': 4}, 'resnet': {}, 'env': {'pts': [2, 1, 0, -3]}}
        self.arena = Mock(return_value=(np.array([8, 0, 0, 0]), [g['log_path'] for g in self.games]))
        self.reuse = Mock(return_value=(self.games, {
            'source_dir': str(self.root), 'source_outcomes_sha256': probe.file_sha256(self.root / 'outcomes.json')}))
        self.cuda = SimpleNamespace(is_available=lambda: False, synchronize=Mock(), empty_cache=Mock())
        def checkpoint(path, **kwargs):
            return {'config': cfg, 'steps': {'warm0': 0, 'warm40k': 40000, 'clean40k': 40000}[Path(path).name],
                    'oracle_brain': {}, 'value_net': {},
                    'oracle_critic_pretrain': {'target_mode': 'all_players', 'return_mode': 'score_rank_mc',
                                              'discount_gamma': 1, 'critic_arch': 'dual_tower'}}
        torch = SimpleNamespace(
            __version__='host-test', set_num_threads=Mock(), set_num_interop_threads=Mock(),
            device=lambda name: SimpleNamespace(type=name), load=checkpoint, cuda=self.cuda,
            manual_seed=Mock(), float32=np.float32, inference_mode=nullcontext,
            as_tensor=lambda data, **kwargs: HostTensor(data),
            cat=lambda tensors: HostTensor(np.concatenate([tensor.numpy() for tensor in tensors])))
        trajectory = {'obs': np.zeros((4, 4)), 'invisible_obs': np.zeros((4, 4)),
                      'at_kyoku': np.zeros(4, dtype=int), 'kyoku_value_target': np.zeros((1, 4)),
                      'context_meta': np.zeros((4, 8), dtype=int)}
        namespace = {**vars(probe), 'np': np, 'torch': torch, 'config': {'control': {}, 'env': {}},
                     'libriichi': SimpleNamespace(__file__=__file__),
                     'GameplayLoader': lambda **kwargs: SimpleNamespace(set_oracle_imputation_seed=Mock()),
                     'OracleDualTowerBrain': lambda **kwargs: HostModel(), 'ValueHead': lambda **kwargs: HostModel(),
                     'MortalEngine': Mock(), 'TrainPlayer': type('HostPlayer', (), {'train_play': self.arena}),
                     'FileDatasetsIter': lambda **kwargs: SimpleNamespace(
                         iter_game_trajectories=lambda *args, **kwargs: iter([trajectory])),
                     'load_policy': lambda *args: (HostModel(), HostModel(), cfg, 0),
                     'load_games': lambda *args: self.games, 'load_verified_rollout': self.reuse,
                     'duplicate_sets': lambda games: {(g['seed'], g['seed_key']) for g in games},
                     'verify_complete_log': lambda path: 1,
                     'expand_kyoku_rewards_to_steps': lambda values, at: values[at],
                     'discounted_returns_from_step_rewards': lambda rewards, gamma: rewards,
                     'compute_gae_advantages_from_step_rewards': lambda rewards, values, gamma, lam: rewards - values,
                     'bootstrap_interval': lambda values, **kwargs: {'mean': float(values.mean())}}
        source = Path(probe.__file__)
        tree = ast.parse(source.read_text(encoding='utf-8'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        run.body = [node for node in run.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=[run], type_ignores=[]), str(source), 'exec'), namespace)
        self.run = namespace['run']
        random_state, numpy_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, random_state)
        self.addCleanup(np.random.set_state, numpy_state)

    def invoke(self, *, reuse=False, cuda=False, shuffle=False):
        output = self.root / 'result'
        argv = self.argv + ['--output-dir', str(output)]
        if reuse:
            # Sibling output, never a descendant of the read-only original.
            source = self.root / 'original'
            argv += ['--reuse-rollout-dir', str(source)]
        if cuda:
            argv += ['--device', 'cuda']
        if shuffle:
            argv += ['--shuffle-hidden']
        with patch.object(probe, 'run', side_effect=self.run), \
                patch.object(probe.subprocess, 'check_output', return_value='test-source'), \
                patch.object(probe.time, 'perf_counter', side_effect=itertools.count()), \
                patch.dict(os.environ), redirect_stdout(io.StringIO()):
            probe.main(argv)
        return output

    def assert_complete_report(self, root, arena_phase):
        report = json.loads((root / 'timing.json').read_text())
        phases = report['phase_seconds']
        self.assertEqual(set(phases), {
            'preflight_input_fingerprint_model_load', arena_phase, 'scoring_setup',
            'replay_decode_validation', 'critic_inference', 'aggregation_statistics',
            'prediction_write', 'final_integrity_write'})
        self.assertEqual(sum(phases.values()), report['total_seconds'])
        self.assertTrue(all(seconds > 0 for seconds in phases.values()))
        self.assertEqual(report['status'], 'complete')
        self.assertIsNone(report['incomplete_phase'])
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(provenance['schema'], 1)
        self.assertEqual(provenance['status'], 'complete')
        self.assertEqual(provenance['timing_report'], 'timing.json')
        self.assertNotIn('timing.json', provenance['artifact_sha256'])
        for name, digest in provenance['artifact_sha256'].items():
            self.assertEqual(probe.file_sha256(root / name), digest)

    def test_eight_game_host_smoke_records_exclusive_timings_and_valid_core_hashes(self):
        self.assert_complete_report(self.invoke(), 'arena_generation_validation')
        self.arena.assert_called_once()
        self.reuse.assert_not_called()
        self.cuda.synchronize.assert_not_called()

    def test_reuse_never_calls_arena_or_records_a_fake_zero_arena_phase(self):
        self.assert_complete_report(self.invoke(reuse=True), 'rollout_reuse_validation')
        self.arena.assert_not_called()
        self.reuse.assert_called_once()
        self.cuda.synchronize.assert_not_called()

    def test_cuda_boundary_calls_are_constant_not_per_game_or_batch(self):
        self.invoke(cuda=True)
        self.assertEqual(self.cuda.synchronize.call_count, 2)

    def test_reuse_only_synchronizes_critic_placement(self):
        self.invoke(reuse=True, cuda=True)
        self.cuda.synchronize.assert_called_once()

    def test_optional_shuffled_forwards_share_inference_phase_without_extra_syncs(self):
        root = self.invoke(reuse=True, cuda=True, shuffle=True)
        self.assert_complete_report(root, 'rollout_reuse_validation')
        self.cuda.synchronize.assert_called_once()
        self.assertIn('hidden_shuffle', json.loads((root / 'metrics.json').read_text()))

    def test_missing_timing_sidecar_does_not_invalidate_published_core_artifacts(self):
        root = self.invoke()
        (root / 'timing.json').unlink()
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(provenance['status'], 'complete')
        for name, digest in provenance['artifact_sha256'].items():
            self.assertEqual(probe.file_sha256(root / name), digest)

    def test_failure_json_preserves_partial_timing_without_completing_phases(self):
        self.arena.side_effect = RuntimeError('host arena failure')
        with self.assertRaisesRegex(RuntimeError, 'host arena failure'):
            self.invoke()
        root = self.root / 'result'
        failure = json.loads((root / 'failure.json').read_text())
        self.assertEqual(failure['timing']['status'], 'incomplete')
        self.assertEqual(failure['timing']['incomplete_phase']['name'], 'arena_generation_validation')
        self.assertEqual(set(failure['timing']['phase_seconds']), {'preflight_input_fingerprint_model_load'})
        self.assertFalse((root / 'timing.json').exists())
        self.assertFalse((root / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main()

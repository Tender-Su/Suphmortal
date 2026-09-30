import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mortal.core.adaptive_curriculum import AdaptiveCurriculumConfig, MetricSpec, observe_adaptive_curriculum
from mortal.core.evidence_contract import validation_input_contract
from mortal.data.oracle_value import full_decision_clock, oracle_step_value_targets
from mortal.online.policy_objective import (
    actor_surrogate, behavior_version_is_usable, policy_drift, validate_actor_objective,
)
from mortal.online.train_online import compute_vtrace_targets_from_step_rewards
from mortal.online.pretrain_oracle_critic import OracleEvaluationPaused, evaluate_modes
from mortal.data.oracle_value import OracleTerminalValueDataset


class HardGuardrailTests(unittest.TestCase):
    def observe(self, state, step, primary, tail, cfg=None):
        cfg = cfg or AdaptiveCurriculumConfig(
            'phase_a', MetricSpec('p0', meaningful_delta=0.01),
            guardrails=(MetricSpec('tail', noninferiority_margin=0.02),),
            max_unresolved_gates=2,
        )
        return observe_adaptive_curriculum(
            state, cfg, optimizer_steps=step,
            metrics={'p0': np.mean(primary), 'tail': np.mean(tail)},
            cluster_records={name: [[i, v, 1] for i, v in enumerate(values)]
                             for name, values in [('p0', primary), ('tail', tail)]},
        )

    def test_primary_improves_but_tail_vetoes_then_budget_is_inconclusive(self):
        baseline = self.observe(None, 0, [1, 1, 1], [1, 1, 1]).state
        veto = self.observe(baseline, 1, [0.9] * 3, [1.2] * 3)
        self.assertEqual(0, veto.state['best_step'])
        stop = self.observe(veto.state, 2, [0.8] * 3, [1.2] * 3)
        self.assertEqual('inconclusive', stop.action)
        self.assertEqual(0, stop.state['best_step'])
        self.assertEqual('inconclusive', self.observe(stop.state, 3, [0.8] * 3, [1.2] * 3).action)

    def test_noninferiority_margin_is_respected(self):
        baseline = self.observe(None, 0, [1] * 3, [1] * 3).state
        self.assertEqual('update_best', self.observe(baseline, 1, [0.9] * 3, [1.01] * 3).action)

    def test_single_game_cannot_establish_improvement(self):
        baseline = self.observe(None, 0, [1], [1]).state
        self.assertEqual(0, self.observe(baseline, 1, [0.1], [0.1]).state['best_step'])

    def test_missing_guard_is_rejected(self):
        cfg = AdaptiveCurriculumConfig('a', MetricSpec('p'), guardrails=(MetricSpec('g'),))
        with self.assertRaisesRegex(ValueError, 'missing metric'):
            observe_adaptive_curriculum(None, cfg, optimizer_steps=0,
                                        metrics={'p': 1}, cluster_records={'p': [[0, 1, 1]]})


class ActorContractTests(unittest.TestCase):
    def test_vtrace_gradient_has_exactly_one_importance_factor(self):
        for behavior_ratio in (0.25, 0.5, 1.0, 2.0):
            _, pg = compute_vtrace_targets_from_step_rewards(
                [2.0], [0.0], [np.log(behavior_ratio)], 1.0, rho_clip=1.0, c_clip=1.0,
            )
            logp = torch.tensor([-1.0], requires_grad=True)
            ratio = torch.tensor([behavior_ratio])
            loss = -actor_surrogate(logp, ratio, torch.from_numpy(pg), objective='vtrace', clip_ratio=0.2).mean()
            loss.backward()
            self.assertAlmostEqual(-2 * min(behavior_ratio, 1.0), logp.grad.item())

    def test_ppo_clip_suppresses_outside_positive_gradient(self):
        log_ratio = torch.tensor([np.log(1.4)], requires_grad=True)
        gain = actor_surrogate(log_ratio, log_ratio.exp(), torch.ones(1), objective='ppo', clip_ratio=0.2)
        gain.sum().backward()
        self.assertEqual(0, log_ratio.grad.item())

    def test_hybrid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot consume'):
            validate_actor_objective({}, vtrace_enabled=True, gae_enabled=True, replay_is=True)

    def test_aa_drift_zero_and_invalid_ratio_fails_closed(self):
        metrics = policy_drift(torch.ones(4), 0.2)
        self.assertEqual(0, metrics['approx_kl'].item())
        self.assertEqual(0, metrics['clip_fraction'].item())
        with self.assertRaises(ValueError):
            policy_drift(torch.tensor([float('nan')]), 0.2)

    def test_unknown_stale_and_future_versions_are_excluded(self):
        history = {1: {}, 2: {}, 3: {}}
        for version in (None, -1, 1, 4):
            self.assertFalse(behavior_version_is_usable(version, history, published_version=3, max_gap=1))
        self.assertTrue(behavior_version_is_usable(2, history, published_version=3, max_gap=1))


class InputContractTests(unittest.TestCase):
    def test_pause_aborts_validation_without_returning_partial_metrics(self):
        brain, head = MagicMock(), MagicMock()
        with patch('mortal.online.pretrain_oracle_critic.external_pause_requested', return_value=True):
            with self.assertRaises(OracleEvaluationPaused):
                evaluate_modes(brain, head, [None], 'cpu', enable_amp=False, max_batches=0)
        brain.assert_not_called()
        head.assert_not_called()

    def test_training_completion_changes_per_pass_and_validation_stays_fixed(self):
        for shuffle in (False, True):
            with patch('mortal.data.oracle_value.GameplayLoader') as native:
                dataset = OracleTerminalValueDataset(version=4, file_list=[], pts=[2, 1, 0, -3],
                                             shuffle_files=shuffle, oracle_imputation_seed=17)
                list(dataset.load_files(False, stream_pass=0))
                list(dataset.load_files(False, stream_pass=2))
                list(dataset.load_files(False, stream_pass=2))
                expected = 2000023 if shuffle else 17
                actual = [call.args[0] for call in native.return_value.set_oracle_imputation_seed.call_args_list]
                self.assertEqual([17, expected, expected], actual)

    def test_legacy_native_fold_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'full decision clock'):
            full_decision_clock(object(), np.array([0]), require_metadata=True)

    def test_validation_hash_binds_file_contents_and_completion_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'game'
            file.write_bytes(b'old')
            native = Path(directory) / 'native'
            native.write_bytes(b'engine')
            def contract(seed):
                return validation_input_contract([file], settings={'seed': seed}, native_file=native)['fingerprint']
            before = contract(1)
            self.assertEqual(before, contract(1))
            self.assertNotEqual(before, contract(2))
            file.write_bytes(b'new')
            self.assertNotEqual(before, contract(1))


if __name__ == '__main__':
    unittest.main()

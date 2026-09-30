"""CPU-only selection protocol regressions (no torch/native imports)."""
import ast
import copy
import math
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from mortal.core.adaptive_curriculum import (
    AdaptiveCurriculumConfig, MetricSpec, observe_adaptive_curriculum,
    normalize_adaptive_curriculum_state, validate_adaptive_observation,
    inherit_adaptive_curriculum_baseline,
)
from mortal.core.oracle_checkpoint_selection import checkpoint_selection, baseline_checkpoint_roles


class OraclePrimarySelectionTests(unittest.TestCase):
    def config(self, protocol='primary_with_diagnostics'):
        return AdaptiveCurriculumConfig(
            phase_name='a', primary=MetricSpec('mse', meaningful_delta=0.01),
            guardrails=(MetricSpec('mae'), MetricSpec('zero')),
            selection_protocol=protocol,
        )

    def observation(self, prediction):
        target = [0, 0, 0, 0, 5]
        square = sum((prediction - x) ** 2 for x in target)
        absolute = sum(abs(prediction - x) for x in target)
        # Two independent game clusters, each with the same target distribution.
        records = {'mse': [[g, square, 5] for g in range(2)],
                   'mae': [[g, absolute, 5] for g in range(2)],
                   'zero': [[g, 4 * prediction**2, 4] for g in range(2)]}
        return {'mse': square / 5, 'mae': absolute / 5, 'zero': prediction**2}, records

    def baseline(self, config):
        metrics, records = self.observation(0)
        return observe_adaptive_curriculum(None, config, optimizer_steps=0,
                                           metrics=metrics, cluster_records=records).state

    def test_mean_predictor_promotes_despite_mae_and_zero_worsening(self):
        config = self.config()
        metrics, records = self.observation(1)
        self.assertEqual({'mse': 4, 'mae': 1.6, 'zero': 1}, metrics)
        result = observe_adaptive_curriculum(self.baseline(config), config,
                    optimizer_steps=50, metrics=metrics, cluster_records=records)
        self.assertEqual('update_best', result.action)
        self.assertAlmostEqual(0.6, result.comparisons['mae']['mean'])
        self.assertEqual(-1, result.comparisons['mse']['mean'])

    def test_legacy_default_keeps_veto(self):
        config = self.config('legacy_guardrails')
        self.assertEqual('legacy_guardrails', AdaptiveCurriculumConfig('a', MetricSpec('mse')).selection_protocol)
        metrics, records = self.observation(1)
        result = observe_adaptive_curriculum(self.baseline(config), config,
                    optimizer_steps=50, metrics=metrics, cluster_records=records)
        self.assertEqual(0, result.state['best_step'])

    def test_empty_and_sparse_diagnostics_do_not_veto(self):
        config = self.config()
        for empty in (False, True):
            before_metrics, before = self.observation(0)
            after_metrics, after = self.observation(1)
            before['zero'] = [] if empty else before['zero'][:1]
            after['zero'] = [] if empty else after['zero'][:1]
            if empty:
                before_metrics.pop('zero'); after_metrics.pop('zero')
            baseline = observe_adaptive_curriculum(None, config, optimizer_steps=0,
                         metrics=before_metrics, cluster_records=before).state
            result = observe_adaptive_curriculum(baseline, config, optimizer_steps=50,
                         metrics=after_metrics, cluster_records=after)
            self.assertEqual('update_best', result.action)
            self.assertEqual(0 if empty else 1, result.comparisons['zero']['num_games'])
            if empty:
                self.assertIsNone(result.comparisons['zero']['mean'])
                self.assertIsNone(result.comparisons['zero']['ci_high'])

    def test_malformed_nonfinite_and_changed_samples_fail(self):
        config = self.config()
        for bad in ([[0, math.nan, 4]], [[0, 1, 0]], [[0, 1, 0.5]],
                    [[0, 1, 4], [0, 2, 4]], [[0, 1, 3]]):
            metrics, records = self.observation(1)
            records['zero'] = bad
            with self.assertRaises(ValueError):
                validate_adaptive_observation(config, metrics, records, self.baseline(config))
        metrics, records = self.observation(1)
        records['zero'] = []
        with self.assertRaises(ValueError):
            validate_adaptive_observation(config, metrics, records)
        metrics, records = self.observation(1)
        metrics['mae'] = math.inf
        with self.assertRaises(ValueError):
            validate_adaptive_observation(config, metrics, records)

    def test_diagnostics_cannot_compensate_for_futile_primary(self):
        config = self.config()
        baseline = self.baseline(config)
        metrics, records = self.observation(0)
        # Diagnostic improvement cannot keep flat primary open.
        metrics['mae'] = 0
        records['mae'] = [[g, 0, 5] for g in range(2)]
        first = observe_adaptive_curriculum(baseline, config, optimizer_steps=50,
                    metrics=metrics, cluster_records=records)
        second = observe_adaptive_curriculum(first.state, config, optimizer_steps=100,
                    metrics=metrics, cluster_records=records)
        self.assertEqual('transition', second.action)

    def test_non_gate_candidate_is_not_accepted_best(self):
        flags = checkpoint_selection(primary_loss=4, all_players_loss=4,
                    best_primary_loss=5, best_val_loss=5, best_observed_primary_loss=5,
                    primary_with_diagnostics=True, adaptive_active=True,
                    gate_observed=False, accepted=False)
        self.assertEqual({'best': False, 'best_primary': False,
                          'best_observed_primary': True}, flags)
        self.assertIn('best_observed_primary', baseline_checkpoint_roles(True))
        self.assertIn('best_primary', baseline_checkpoint_roles(True))
        self.assertEqual(('latest', 'adaptive_best'), baseline_checkpoint_roles(False))

    def test_checkpoint_compatibility_ties_and_unaccepted_tiny_gain(self):
        base = dict(primary_loss=4.999999, all_players_loss=4.999999,
                    best_primary_loss=5, best_val_loss=5, best_observed_primary_loss=5,
                    primary_with_diagnostics=True, adaptive_active=True,
                    gate_observed=True, accepted=False)
        self.assertTrue(checkpoint_selection(**base)['best_observed_primary'])
        self.assertFalse(checkpoint_selection(**base)['best_primary'])
        self.assertFalse(checkpoint_selection(**{**base, 'primary_loss': 5})['best_observed_primary'])
        legacy = {**base, 'primary_with_diagnostics': False,
                  'best_observed_primary_loss': None}
        self.assertFalse(checkpoint_selection(**legacy)['best_primary'])
        self.assertTrue(checkpoint_selection(**{**legacy, 'gate_observed': False})['best_primary'])
        self.assertTrue(checkpoint_selection(**{**legacy, 'adaptive_active': False})['best_primary'])

    def test_protocol_boundary_rejects_legacy_resume(self):
        for source, destination in (('legacy_guardrails', 'primary_with_diagnostics'),
                                    ('primary_with_diagnostics', 'legacy_guardrails')):
            baseline = self.baseline(self.config(source))
            with self.assertRaisesRegex(ValueError, 'explicit new branch'):
                normalize_adaptive_curriculum_state(baseline, self.config(destination))
            with self.assertRaisesRegex(ValueError, 'across selection protocols'):
                inherit_adaptive_curriculum_baseline(self.config(destination), baseline)
        legacy = self.baseline(self.config('legacy_guardrails'))
        self.assertNotIn('selection_protocol', legacy)
        self.assertEqual(legacy, normalize_adaptive_curriculum_state(legacy, self.config('legacy_guardrails')))

    def test_training_contract_includes_only_opt_in_protocol(self):
        # Exercise actual lightweight contract function without importing torch/libriichi.
        source = Path('mortal/online/pretrain_oracle_critic.py').read_text()
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'adaptive_curriculum_contract', 'validate_resume_training_contract'}]
        namespace = {'copy': copy}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<contract>', 'exec'), namespace)
        contract = namespace['adaptive_curriculum_contract']
        self.assertNotIn('selection_protocol', contract(self.config('legacy_guardrails')))
        self.assertEqual('primary_with_diagnostics', contract(self.config())['selection_protocol'])
        legacy = {'adaptive_curriculum': contract(self.config('legacy_guardrails'))}
        opt_in = {'adaptive_curriculum': contract(self.config())}
        validate = namespace['validate_resume_training_contract']
        self.assertFalse(validate({'training_contract': legacy}, legacy))
        self.assertFalse(validate({'training_contract': opt_in}, opt_in))
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            validate({'training_contract': legacy}, opt_in)


class OracleSelectionResumeTests(unittest.TestCase):
    """Execute production validation helpers; only the torch.load boundary is mocked."""

    def setUp(self):
        source_file = Path(__file__).resolve().parents[1] / 'online' / 'pretrain_oracle_critic.py'
        tree = ast.parse(source_file.read_text(encoding='utf-8'))
        names = {'maybe_load_protocol_best_loss', 'validate_resume_training_contract',
                 'validate_resume_file_splits'}
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        self.contract = {'adaptive_curriculum': {'selection_protocol': 'primary_with_diagnostics'},
                         'validation_input_fingerprint': 'fixed-dev-input'}
        self.splits = {'dev': {'sha256': 'fixed-dev'}, 'test': {'sha256': 'sealed'}}
        self.saved = {'training_contract': copy.deepcopy(self.contract),
                      'file_splits': copy.deepcopy(self.splits), 'steps': 100,
                      'adaptive_curriculum_state': {'best_step': 100},
                      'best_primary_loss': 4.0, 'best_observed_primary_loss': 3.9}
        self.loader = Mock(side_effect=lambda *args, **kwargs: self.saved)
        namespace = {'math': math, 'copy': copy,
                     'path': SimpleNamespace(exists=lambda _: True),
                     'torch': SimpleNamespace(load=self.loader)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_file), 'exec'), namespace)
        self.load_best = namespace['maybe_load_protocol_best_loss']

    def load(self, key='best_primary_loss'):
        return self.load_best('companion.pth', 5.0, key, self.contract, self.splits)

    def test_valid_companion_preserves_minimum(self):
        self.assertEqual(4.0, self.load())
        self.loader.assert_called_once_with('companion.pth', weights_only=False, map_location='cpu')
        self.assertEqual(3.9, self.load('best_observed_primary_loss'))

    def test_foreign_contract_or_validation_fingerprint_rejected(self):
        for key, value in (('adaptive_curriculum', {'selection_protocol': 'legacy_guardrails'}),
                           ('validation_input_fingerprint', 'foreign-input')):
            with self.subTest(key=key):
                self.saved['training_contract'] = {**self.contract, key: value}
                with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                    self.load()

    def test_unaccepted_step_cannot_be_imported_as_accepted_best(self):
        self.saved['adaptive_curriculum_state']['best_step'] = 50
        with self.assertRaisesRegex(ValueError, 'accepted step'):
            self.load()
        # A raw observation is allowed to be unaccepted.
        self.assertEqual(3.9, self.load('best_observed_primary_loss'))

    def test_missing_or_changed_splits_rejected(self):
        self.saved.pop('file_splits')
        with self.assertRaisesRegex(ValueError, 'missing file split provenance'):
            self.load()
        self.saved['file_splits'] = {'dev': {'sha256': 'foreign-dev'}}
        with self.assertRaisesRegex(ValueError, 'file split mismatch'):
            self.load()

    def test_nonfinite_companion_loss_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.saved['best_primary_loss'] = value
                with self.assertRaisesRegex(ValueError, 'must be finite'):
                    self.load()


if __name__ == '__main__':
    unittest.main()

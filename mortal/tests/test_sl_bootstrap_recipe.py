"""Regressions for unintended objective changes at SL curriculum bootstrap."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from mortal.supervised.run_sl_ab import (
    migrate_adaptive_phase_handoff,
    resolve_bootstrap_auxiliary_recipe,
)
from mortal.supervised.auxiliary_config import (
    auxiliary_step_offset_from_state,
    effective_auxiliary_recipe,
    resolve_effective_aux_cfg,
    validate_exact_resume_auxiliary_recipe,
)


class BootstrapAuxiliaryRecipeTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            'aux': {'opponent_state_weight': 0.00135, 'danger_enabled': True,
                    'danger_weight': 0.00804},
            'supervised': {'rank_aux': {'base_weight': 0.001548, 'max_weight': 0.00516}},
        }
        self.current = {
            'aux': {'opponent_state_weight': 0.0, 'danger_enabled': False, 'danger_weight': 0.0},
            'supervised': {'rank_aux': {'base_weight': 0.03, 'max_weight': 0.1},
                           'lr': 1e-5, 'file_index': 'new-stage-index.pth'},
            'optim': {'weight_decay': 0.1},
        }

    def test_inheritance_prevents_observed_nineteen_fold_rank_weight_jump(self):
        original = deepcopy(self.current)
        resolved, record = resolve_bootstrap_auxiliary_recipe(self.current, self.source)
        self.assertEqual(self.source['aux'], resolved['aux'])
        self.assertEqual(self.source['supervised']['rank_aux'], resolved['supervised']['rank_aux'])
        self.assertEqual(1e-5, resolved['supervised']['lr'])
        self.assertEqual('new-stage-index.pth', resolved['supervised']['file_index'])
        self.assertEqual(self.current['optim'], resolved['optim'])
        self.assertFalse(record['changed_from_source'])
        resolved['aux']['danger_weight'] = 7
        self.assertEqual(0.00804, self.source['aux']['danger_weight'])
        self.assertEqual(0.00804, record['effective']['aux']['danger_weight'])
        self.assertEqual(original, self.current)

    def test_objective_ablation_requires_explicit_current_policy_and_is_recorded(self):
        resolved, record = resolve_bootstrap_auxiliary_recipe(
            self.current, self.source, policy='current',
        )
        self.assertEqual(self.current, resolved)
        self.assertTrue(record['changed_from_source'])
        self.assertEqual(0.001548, record['source']['rank_aux']['base_weight'])
        self.assertEqual(0.03, record['effective']['rank_aux']['base_weight'])

    def test_unknown_source_recipe_cannot_be_claimed_inherited(self):
        with self.assertRaisesRegex(ValueError, 'no config'):
            resolve_bootstrap_auxiliary_recipe(self.current, None)
        resolved, record = resolve_bootstrap_auxiliary_recipe(self.current, None, policy='current')
        self.assertEqual(self.current, resolved)
        self.assertIsNone(record['changed_from_source'])

    def test_source_defaults_remove_unrelated_destination_overrides(self):
        resolved, record = resolve_bootstrap_auxiliary_recipe(self.current, {})
        self.assertEqual({}, resolved['aux'])
        self.assertEqual({}, resolved['supervised']['rank_aux'])
        self.assertFalse(record['changed_from_source'])

    def test_destination_scoped_override_cannot_undo_inherited_recipe(self):
        self.current['supervised']['aux'] = {
            'danger_enabled': False, 'danger_weight': 0.0, 'opponent_state_weight': 0.0,
        }
        resolved, record = resolve_bootstrap_auxiliary_recipe(self.current, self.source)
        self.assertNotIn('aux', resolved['supervised'])
        self.assertEqual(self.source['aux'], resolve_effective_aux_cfg(resolved, 'supervised'))
        self.assertEqual(effective_auxiliary_recipe(self.source), record['effective_training_recipe'])

    def test_source_scoped_nested_settings_are_preserved_with_training_precedence(self):
        self.source['aux']['opponent_turn_weighting'] = {'early_factor': 0.2, 'late_factor': 1.6}
        self.source['supervised']['aux'] = {
            'danger_weight': 0.004,
            'opponent_turn_weighting': {'late_factor': 1.2},
        }
        resolved, _ = resolve_bootstrap_auxiliary_recipe(self.current, self.source)
        effective = resolve_effective_aux_cfg(resolved, 'supervised')
        self.assertEqual(0.004, effective['danger_weight'])
        self.assertEqual({'late_factor': 1.2}, effective['opponent_turn_weighting'])
        resolved['supervised']['aux']['danger_weight'] = 9
        self.assertEqual(0.004, self.source['supervised']['aux']['danger_weight'])

    def test_explicit_current_policy_keeps_scoped_overrides(self):
        self.current['supervised']['aux'] = {'danger_weight': 0.02}
        resolved, record = resolve_bootstrap_auxiliary_recipe(self.current, self.source, policy='current')
        self.assertEqual(self.current, resolved)
        self.assertEqual(0.02, record['effective_training_recipe']['aux']['danger_weight'])

    def test_recipe_record_compares_resolved_values_not_table_placement(self):
        current = deepcopy(self.source)
        current['supervised']['aux'] = {'danger_weight': current['aux'].pop('danger_weight')}
        _, record = resolve_bootstrap_auxiliary_recipe(current, self.source, policy='current')
        self.assertFalse(record['changed_from_source'])

    def test_resume_rejects_rank_coefficient_drift_with_identical_enabled_heads(self):
        current = deepcopy(self.source)
        current['supervised']['rank_aux']['base_weight'] = 0.03
        with self.assertRaisesRegex(RuntimeError, 'recipe mismatch: rank_aux'):
            validate_exact_resume_auxiliary_recipe({'config': self.source}, current)

    def test_resume_checks_danger_targets_and_scoped_overrides(self):
        for change in ({'danger_value_cap': 32000}, {'danger_weight': 0.0}, {'danger_ramp_steps': 999}):
            current = deepcopy(self.source)
            current['supervised']['aux'] = change
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'recipe mismatch: aux'):
                validate_exact_resume_auxiliary_recipe({'config': self.source}, current)

    def test_resume_accepts_equal_effective_recipe_and_runtime_changes(self):
        current = deepcopy(self.source)
        current['supervised']['aux'] = {'danger_weight': current['aux'].pop('danger_weight')}
        current['supervised']['num_workers'] = 2
        current['supervised']['file_index'] = 'another-path.pth'
        validate_exact_resume_auxiliary_recipe({'config': self.source}, current)

    def test_resume_cannot_claim_an_unknown_recipe_or_section_matches(self):
        with self.assertRaisesRegex(RuntimeError, 'saved auxiliary config'):
            validate_exact_resume_auxiliary_recipe({}, self.current)
        with self.assertRaisesRegex(RuntimeError, 'same config section'):
            validate_exact_resume_auxiliary_recipe(
                {'config': self.source, 'config_section': 'another'}, self.source,
            )

    def test_auxiliary_ramp_survives_bootstrap_resume_and_another_phase(self):
        source_updates = 2_878_912
        offset = auxiliary_step_offset_from_state(
            {}, optimizer_steps=0, source_optimizer_steps=source_updates,
        )
        self.assertEqual(source_updates, offset)
        self.assertEqual(1.0, min(offset / 1000, 1.0))
        saved = {'auxiliary_optimizer_steps': offset + 49_985}
        resumed = auxiliary_step_offset_from_state(
            saved, optimizer_steps=49_985, source_optimizer_steps=49_985,
        )
        self.assertEqual(offset, resumed)
        self.assertEqual(source_updates + 49_986, resumed + 49_986)
        # A later weights-only branch starts its own optimizer at zero again.
        next_offset = auxiliary_step_offset_from_state(
            saved, optimizer_steps=0, source_optimizer_steps=49_985,
        )
        self.assertEqual(saved['auxiliary_optimizer_steps'], next_offset)

    def test_invalid_auxiliary_clock_is_rejected(self):
        for value in (-1, True, 2.5, '12'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                auxiliary_step_offset_from_state(
                    {'auxiliary_optimizer_steps': value}, optimizer_steps=0,
                    source_optimizer_steps=0,
                )

    def test_adaptive_handoff_rejects_objective_change_before_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pth'
            destination = Path(directory) / 'target' / 'checkpoints' / 'latest.pth'
            torch.save({
                'checkpoint_id': 'parent', 'run_provenance': {'plan_id': 'source'},
                'config': self.source,
            }, source)
            plan = {'parent_checkpoint_id': 'parent', 'parent_plan_id': 'source'}
            for target in (self.current, {
                **self.source,
                'supervised': {**self.source['supervised'], 'aux': {'danger_weight': 0.0}},
            }):
                with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'semantics'):
                    migrate_adaptive_phase_handoff(
                        source_path=source, target_path=destination,
                        expected_plan=plan, target_cfg=target,
                    )
                self.assertFalse(destination.parent.exists())

    def test_invalid_recipe_is_rejected(self):
        for source in ({'aux': []}, {'supervised': []}, {'supervised': {'rank_aux': []}},
                       {'supervised': {'aux': []}}):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'mapping'):
                resolve_bootstrap_auxiliary_recipe(self.current, source)
        with self.assertRaisesRegex(ValueError, 'policy'):
            resolve_bootstrap_auxiliary_recipe(self.current, self.source, policy='silent')


if __name__ == '__main__':
    unittest.main()

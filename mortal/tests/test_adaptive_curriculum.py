import math
import unittest

from mortal.core.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    inherit_adaptive_curriculum_baseline,
    MetricSpec,
    normalize_adaptive_curriculum_state,
    observe_adaptive_curriculum,
    paired_cluster_summary,
)


def records(values):
    return [[index, float(value), 1] for index, value in enumerate(values)]


class PairedClusterSummaryTests(unittest.TestCase):
    def test_matches_sample_weighted_paired_mean(self):
        summary = paired_cluster_summary(
            [[1, 3.0, 2], [2, 2.0, 1]],
            [[1, 2.0, 2], [2, 1.0, 1]],
        )

        self.assertTrue(math.isclose(2.0 / 3.0, summary['mean']))
        self.assertEqual(3, summary['num_samples'])
        self.assertEqual(2, summary['num_games'])

    def test_rejects_mismatched_game_sets(self):
        with self.assertRaisesRegex(ValueError, 'different game ids'):
            paired_cluster_summary(
                [[1, 1.0, 1]],
                [[2, 1.0, 1]],
            )


class AdaptiveCurriculumTests(unittest.TestCase):
    def make_config(self, *, final_phase=False, guardrails=()):
        return AdaptiveCurriculumConfig(
            phase_name='phase_a',
            primary=MetricSpec(
                name='primary_loss',
                direction='lower',
                meaningful_delta=0.01,
            ),
            final_phase=final_phase,
            gate_every_steps=50,
            required_futile_gates=2,
            primary_noninferiority_margin=0.01,
            guardrails=guardrails,
            lr_levels=(0.1, 0.05) if final_phase else (),
        )

    def observe(self, state, step, values, *, config=None, guardrail_values=None):
        config = config or self.make_config()
        metrics = {'primary_loss': sum(values) / len(values)}
        clustered = {'primary_loss': records(values)}
        if guardrail_values is not None:
            metrics['tail_loss'] = sum(guardrail_values) / len(guardrail_values)
            clustered['tail_loss'] = records(guardrail_values)
        return observe_adaptive_curriculum(
            state,
            config,
            optimizer_steps=step,
            metrics=metrics,
            cluster_records=clustered,
        )

    def test_first_gate_sets_baseline(self):
        decision = self.observe(None, 50, [1.0, 1.0, 1.0])

        self.assertEqual('continue', decision.action)
        self.assertEqual(50, decision.state['best_step'])
        self.assertEqual('set_baseline', decision.state['last_action'])

    def test_new_phase_inherits_best_as_zero_gate_baseline(self):
        source = self.observe(None, 50, [1.0, 1.0, 1.0]).state
        destination_config = AdaptiveCurriculumConfig(
            phase_name='phase_b',
            primary=self.make_config().primary,
        )

        inherited = inherit_adaptive_curriculum_baseline(
            destination_config,
            source,
        )

        self.assertEqual('phase_b', inherited['phase_name'])
        self.assertEqual(50, inherited['best_step'])
        self.assertEqual(0, inherited['gate_index'])
        self.assertIsNone(inherited['last_gate_step'])
        self.assertFalse(inherited['completed'])
        decision = self.observe(
            inherited,
            100,
            [1.0, 1.0, 1.0],
            config=destination_config,
        )
        self.assertEqual(1, decision.state['consecutive_futile_gates'])

    def test_clear_meaningful_improvement_updates_best(self):
        state = self.observe(None, 50, [1.0, 1.0, 1.0]).state
        decision = self.observe(state, 100, [0.95, 0.95, 0.95])

        self.assertEqual('update_best', decision.action)
        self.assertEqual(100, decision.state['best_step'])
        self.assertEqual(0, decision.state['consecutive_futile_gates'])

    def test_uncertain_interval_keeps_phase_open(self):
        state = self.observe(None, 50, [1.0, 1.0, 1.0, 1.0]).state
        decision = self.observe(state, 100, [0.8, 1.2, 0.8, 1.2])

        self.assertEqual('continue', decision.action)
        self.assertEqual(0, decision.state['consecutive_futile_gates'])

    def test_two_futile_gates_transition_nonfinal_phase(self):
        state = self.observe(None, 50, [1.0, 1.0, 1.0]).state
        state = self.observe(state, 100, [1.0, 1.0, 1.0]).state
        decision = self.observe(state, 150, [1.0, 1.0, 1.0])

        self.assertEqual('transition', decision.action)
        self.assertTrue(decision.state['completed'])

    def test_guardrail_compensation_prevents_futility(self):
        guardrail = MetricSpec('tail_loss', 'lower', 0.01)
        config = self.make_config(guardrails=(guardrail,))
        state = self.observe(
            None,
            50,
            [1.0, 1.0, 1.0],
            config=config,
            guardrail_values=[2.0, 2.0, 2.0],
        ).state
        decision = self.observe(
            state,
            100,
            [1.0, 1.0, 1.0],
            config=config,
            guardrail_values=[1.9, 1.9, 1.9],
        )

        self.assertEqual('continue', decision.action)
        self.assertEqual(0, decision.state['consecutive_futile_gates'])

    def test_final_phase_reduces_then_stops(self):
        config = self.make_config(final_phase=True)
        state = self.observe(None, 50, [1.0, 1.0], config=config).state
        state = self.observe(state, 100, [1.0, 1.0], config=config).state
        decision = self.observe(state, 150, [1.0, 1.0], config=config)
        self.assertEqual('reduce_lr', decision.action)
        self.assertEqual(0.05, decision.target_lr)
        self.assertEqual(50, decision.state['best_step'])

        state = self.observe(
            decision.state,
            200,
            [1.0, 1.0],
            config=config,
        ).state
        decision = self.observe(state, 250, [1.0, 1.0], config=config)
        self.assertEqual('stop', decision.action)
        self.assertTrue(decision.state['completed'])

    def test_state_phase_mismatch_is_rejected(self):
        config = self.make_config()
        state = self.observe(None, 50, [1.0, 1.0], config=config).state
        other = AdaptiveCurriculumConfig(
            phase_name='phase_b',
            primary=config.primary,
        )
        with self.assertRaisesRegex(ValueError, 'phase mismatch'):
            normalize_adaptive_curriculum_state(state, other)


if __name__ == '__main__':
    unittest.main()

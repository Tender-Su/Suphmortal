import unittest

from mortal.supervised.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    MetricSpec,
    observe_adaptive_curriculum,
)


def records(value):
    return [[index, float(value), 1] for index in range(4)]


class SupervisedAdaptiveCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.config = AdaptiveCurriculumConfig(
            phase_name='phase_c',
            primary=MetricSpec('policy_loss', 'lower', 0.01),
            final_phase=True,
            gate_every_steps=50,
            required_futile_gates=2,
            lr_levels=(0.1, 0.05, 0.025),
        )

    def observe(self, state, step, value):
        return observe_adaptive_curriculum(
            state,
            self.config,
            optimizer_steps=step,
            metrics={'policy_loss': value},
            cluster_records={'policy_loss': records(value)},
        )

    def enter_second_lr(self):
        state = self.observe(None, 50, 1.0).state
        state = self.observe(state, 100, 1.0).state
        decision = self.observe(state, 150, 1.0)
        self.assertEqual('reduce_lr', decision.action)
        self.assertFalse(decision.state['lr_level_has_improved'])
        return decision.state

    def test_failed_lower_lr_stops_without_cascading(self):
        state = self.enter_second_lr()
        state = self.observe(state, 200, 1.0).state
        decision = self.observe(state, 250, 1.0)

        self.assertEqual('stop', decision.action)
        self.assertEqual(1, decision.state['lr_level_index'])
        self.assertEqual(0.05, decision.target_lr)
        self.assertIn('instead of cascading', decision.reason)

    def test_successful_lower_lr_can_earn_another_reduction(self):
        state = self.enter_second_lr()
        decision = self.observe(state, 200, 0.95)
        self.assertEqual('update_best', decision.action)
        self.assertTrue(decision.state['lr_level_has_improved'])

        state = self.observe(decision.state, 250, 0.95).state
        decision = self.observe(state, 300, 0.95)
        self.assertEqual('reduce_lr', decision.action)
        self.assertEqual(0.025, decision.target_lr)


if __name__ == '__main__':
    unittest.main()

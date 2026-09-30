import math
import unittest

import torch

from mortal.core.lr_scheduler import LinearWarmUpCosineAnnealingLR
from mortal.supervised.convergence import (
    ConvergenceConfig,
    observe_convergence,
    select_bounded_pareto_candidates,
)


class ConvergenceControllerTests(unittest.TestCase):
    def make_config(self):
        return ConvergenceConfig(
            core_optimizer_steps=100,
            tail_lr_levels=(1e-2, 5e-3, 1e-3),
            smoothing_checks=2,
            improvement_delta=1e-3,
            reduce_patience_steps=20,
            stop_patience_steps=30,
            min_level_steps=10,
        )

    def observe(self, state, step, value):
        return observe_convergence(
            state,
            self.make_config(),
            optimizer_steps=step,
            metric_value=value,
        )

    def test_never_stops_or_reduces_during_cosine_core(self):
        state = None
        for step, value in ((10, 1.0), (50, 1.1), (99, 1.2)):
            decision = self.observe(state, step, value)
            state = decision.state
            self.assertEqual('continue', decision.action)
            self.assertFalse(state['tail_started'])

    def test_reduces_each_tail_level_then_stops_at_final_level(self):
        state = None
        actions = []
        observations = (
            (100, 1.0),
            (110, 1.0),
            (120, 1.0),
            (130, 1.0),
            (140, 1.0),
            (150, 1.0),
            (160, 1.0),
            (170, 1.0),
            (180, 1.0),
            (190, 1.0),
            (200, 1.0),
            (210, 1.0),
            (220, 1.0),
        )
        for step, value in observations:
            decision = self.observe(state, step, value)
            state = decision.state
            actions.append((step, decision.action, decision.target_lr))

        self.assertIn((130, 'reduce_lr', 5e-3), actions)
        self.assertIn((170, 'reduce_lr', 1e-3), actions)
        self.assertIn((220, 'stop', 1e-3), actions)
        self.assertTrue(state['converged'])

    def test_meaningful_improvement_restarts_patience(self):
        state = None
        for step, value in ((100, 1.0), (110, 1.0), (120, 0.99), (130, 0.99)):
            decision = self.observe(state, step, value)
            state = decision.state

        self.assertEqual('continue', decision.action)
        self.assertEqual(130, state['last_meaningful_improvement_optimizer_step'])

    def test_rejects_non_descending_tail_levels(self):
        with self.assertRaisesRegex(ValueError, 'strictly descending'):
            ConvergenceConfig.from_mapping({
                'core_optimizer_steps': 100,
                'tail_lr_levels': [1e-3, 1e-3],
            })


class CandidatePortfolioTests(unittest.TestCase):
    def candidate(self, name, policy, action, old, step):
        return {
            'checkpoint_id': name,
            'policy_loss': policy,
            'action_quality_score': action,
            'old_regression_policy_loss': old,
            'step': step,
        }

    def test_dominated_candidates_are_removed(self):
        strong = self.candidate('strong', 0.4, 0.7, 0.5, 10)
        weak = self.candidate('weak', 0.5, 0.6, 0.6, 20)

        selected, removed = select_bounded_pareto_candidates(
            [strong, weak],
            limit=4,
        )

        self.assertEqual(['strong'], [item['checkpoint_id'] for item in selected])
        self.assertEqual(['weak'], [item['checkpoint_id'] for item in removed])

    def test_bounded_frontier_preserves_each_objective_extreme(self):
        candidates = [
            self.candidate('policy', 0.30, 0.30, 0.60, 10),
            self.candidate('middle-a', 0.35, 0.45, 0.55, 20),
            self.candidate('middle-b', 0.40, 0.60, 0.50, 30),
            self.candidate('action', 0.45, 0.80, 0.45, 40),
            self.candidate('old', 0.50, 0.75, 0.30, 50),
        ]

        selected, _ = select_bounded_pareto_candidates(candidates, limit=3)
        selected_ids = {item['checkpoint_id'] for item in selected}

        self.assertEqual({'policy', 'action', 'old'}, selected_ids)


class CosineTailSchedulerTests(unittest.TestCase):
    def make_scheduler(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = LinearWarmUpCosineAnnealingLR(
            optimizer,
            peak=1e-2,
            final=1e-3,
            warm_up_steps=0,
            max_steps=4,
        )
        return parameter, optimizer, scheduler

    def take_step(self, parameter, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()

    def test_legacy_state_defaults_tail_to_original_cosine_floor(self):
        parameter, optimizer, scheduler = self.make_scheduler()
        self.take_step(parameter, optimizer, scheduler)
        optimizer_state = optimizer.state_dict()
        legacy_state = scheduler.state_dict()
        legacy_state.pop('tail_lr')

        _, resumed_optimizer, resumed = self.make_scheduler()
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed.load_state_dict(legacy_state)

        self.assertTrue(math.isclose(1e-3, resumed.tail_lr, abs_tol=1e-12))
        self.assertEqual(scheduler.last_epoch, resumed.last_epoch)
        self.assertEqual(optimizer.param_groups[0]['lr'], scheduler.get_last_lr()[0])
        self.assertEqual(resumed_optimizer.param_groups[0]['lr'], resumed.get_last_lr()[0])

    def test_tail_lr_persists_and_controls_post_core_steps(self):
        parameter, optimizer, scheduler = self.make_scheduler()
        for _ in range(4):
            self.take_step(parameter, optimizer, scheduler)
        scheduler.set_tail_lr(5e-4)
        self.take_step(parameter, optimizer, scheduler)

        self.assertTrue(math.isclose(5e-4, optimizer.param_groups[0]['lr'], abs_tol=1e-12))
        self.assertEqual(5e-4, scheduler.state_dict()['tail_lr'])

    def test_tail_lr_preserves_param_group_lr_scales(self):
        parameters = [
            torch.nn.Parameter(torch.tensor(1.0)),
            torch.nn.Parameter(torch.tensor(2.0)),
        ]
        optimizer = torch.optim.SGD(
            [
                {'params': [parameters[0]], 'lr': 1.0},
                {'params': [parameters[1]], 'lr': 0.25},
            ]
        )
        scheduler = LinearWarmUpCosineAnnealingLR(
            optimizer,
            peak=1e-2,
            final=1e-3,
            warm_up_steps=0,
            max_steps=2,
        )
        for _ in range(2):
            for parameter in parameters:
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()

        scheduler.set_tail_lr(5e-4)

        self.assertEqual([5e-4, 1.25e-4], scheduler.get_last_lr())
        self.assertEqual([5e-4, 1.25e-4], [group['lr'] for group in optimizer.param_groups])

    def test_exact_resume_keeps_pre_core_and_tail_lr_trajectory(self):
        parameter, optimizer, scheduler = self.make_scheduler()
        for _ in range(2):
            self.take_step(parameter, optimizer, scheduler)

        _, resumed_optimizer, resumed = self.make_scheduler()
        resumed_optimizer.load_state_dict(optimizer.state_dict())
        resumed.load_state_dict(scheduler.state_dict())

        for _ in range(2):
            self.take_step(parameter, optimizer, scheduler)
            resumed_parameter = resumed_optimizer.param_groups[0]['params'][0]
            self.take_step(resumed_parameter, resumed_optimizer, resumed)
            self.assertEqual(
                optimizer.param_groups[0]['lr'],
                resumed_optimizer.param_groups[0]['lr'],
            )

        scheduler.set_tail_lr(5e-4)
        resumed.set_tail_lr(5e-4)
        self.take_step(parameter, optimizer, scheduler)
        resumed_parameter = resumed_optimizer.param_groups[0]['params'][0]
        self.take_step(resumed_parameter, resumed_optimizer, resumed)
        self.assertEqual(
            optimizer.param_groups[0]['lr'],
            resumed_optimizer.param_groups[0]['lr'],
        )


if __name__ == '__main__':
    unittest.main()

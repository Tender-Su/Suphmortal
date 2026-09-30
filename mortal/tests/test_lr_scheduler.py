import unittest

import torch

from mortal.core.lr_scheduler import (
    LinearWarmUpReduceOnPlateauLR,
    LinearWarmUpStableDecayLR,
    OptimizerLRSnapshot,
    build_lr_scheduler,
    normalize_scheduler_type,
)


def make_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.SGD([parameter], lr=1.0)


class LRSchedulerTests(unittest.TestCase):
    def test_wsd_has_distinct_warmup_stable_and_linear_decay_regions(self):
        optimizer = make_optimizer()
        scheduler = LinearWarmUpStableDecayLR(
            optimizer,
            init=0.0,
            peak=1.0,
            final=0.2,
            warm_up_steps=2,
            stable_steps=2,
            decay_steps=2,
        )

        observed = [scheduler.get_last_lr()[0]]
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            observed.append(scheduler.get_last_lr()[0])

        torch.testing.assert_close(
            torch.tensor(observed),
            torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0, 0.6, 0.2]),
        )

    def test_plateau_reduces_only_after_monitor_patience(self):
        optimizer = make_optimizer()
        scheduler = LinearWarmUpReduceOnPlateauLR(
            optimizer,
            init=0.0,
            peak=1.0,
            warm_up_steps=0,
            factor=0.5,
            patience_steps=20,
            threshold=0.01,
            min_lr=0.1,
        )

        self.assertEqual('improved', scheduler.observe(1.0, 10)['action'])
        self.assertEqual('hold', scheduler.observe(0.995, 20)['action'])
        decision = scheduler.observe(0.994, 30)

        self.assertEqual('reduce_lr', decision['action'])
        self.assertEqual(0.5, scheduler.get_last_lr()[0])

    def test_plateau_state_restores_lr_and_monitor_history(self):
        first_optimizer = make_optimizer()
        first = LinearWarmUpReduceOnPlateauLR(
            first_optimizer,
            peak=1.0,
            warm_up_steps=0,
            factor=0.5,
            patience_steps=10,
        )
        first.observe(1.0, 10)
        first.observe(1.0, 20)

        second_optimizer = make_optimizer()
        second = LinearWarmUpReduceOnPlateauLR(
            second_optimizer,
            peak=1.0,
            warm_up_steps=0,
            factor=0.5,
            patience_steps=10,
        )
        second.load_state_dict(first.state_dict())

        self.assertEqual(first.get_last_lr(), second.get_last_lr())
        self.assertEqual(first.best, second.best)
        self.assertEqual(first.num_reductions, second.num_reductions)

    def test_optimizer_owned_scheduler_reports_effective_lr(self):
        optimizer = make_optimizer()
        optimizer.param_groups[0]['scheduled_lr'] = 0.25
        scheduler = OptimizerLRSnapshot(optimizer)

        scheduler.step()

        self.assertEqual([0.25], scheduler.get_last_lr())
        self.assertEqual(1, scheduler.state_dict()['last_epoch'])

    def test_optimizer_owned_scheduler_applies_and_restores_tail_lr_by_group_scale(self):
        first = make_optimizer()
        first.add_param_group(
            {'params': [torch.nn.Parameter(torch.tensor(2.0))], 'lr': 0.5}
        )
        scheduler = OptimizerLRSnapshot(first)

        scheduler.set_tail_lr(0.25)

        self.assertEqual([0.25, 0.125], scheduler.get_last_lr())
        second = make_optimizer()
        second.add_param_group(
            {'params': [torch.nn.Parameter(torch.tensor(2.0))], 'lr': 0.5}
        )
        restored = OptimizerLRSnapshot(second)
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual([0.25, 0.125], restored.get_last_lr())

    def test_factory_rejects_unknown_scheduler(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            normalize_scheduler_type('mystery')

        scheduler = build_lr_scheduler(
            make_optimizer(),
            {
                'type': 'constant',
                'init': 0.0,
                'peak': 0.25,
                'warm_up_steps': 0,
            },
        )
        self.assertEqual([0.25], scheduler.get_last_lr())

    def test_constant_schedule_supports_checkpointed_dynamic_tail_lr(self):
        optimizer = make_optimizer()
        scheduler = build_lr_scheduler(
            optimizer,
            {
                'type': 'constant',
                'init': 0.0,
                'peak': 0.25,
                'warm_up_steps': 2,
            },
        )
        for _ in range(2):
            optimizer.step()
            scheduler.step()
        scheduler.set_tail_lr(0.1)

        restored_optimizer = make_optimizer()
        restored = build_lr_scheduler(
            restored_optimizer,
            {
                'type': 'constant',
                'init': 0.0,
                'peak': 0.25,
                'warm_up_steps': 2,
            },
        )
        restored.load_state_dict(scheduler.state_dict())

        self.assertEqual([0.1], scheduler.get_last_lr())
        self.assertEqual([0.1], restored.get_last_lr())


if __name__ == '__main__':
    unittest.main()

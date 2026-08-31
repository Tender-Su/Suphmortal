import unittest

import torch

from mortal.supervised.lr_scheduler import LinearWarmUpConstantLR


def make_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.SGD([parameter], lr=1.0)


class SupervisedLRSchedulerTests(unittest.TestCase):
    def test_linear_rewarm_reaches_constant_peak(self):
        optimizer = make_optimizer()
        scheduler = LinearWarmUpConstantLR(
            optimizer,
            init=0.25,
            peak=1.0,
            warm_up_steps=3,
        )

        observed = [scheduler.get_last_lr()[0]]
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            observed.append(scheduler.get_last_lr()[0])

        torch.testing.assert_close(
            torch.tensor(observed),
            torch.tensor([0.25, 0.5, 0.75, 1.0, 1.0]),
        )

    def test_dynamic_tail_survives_checkpoint_restore(self):
        first_optimizer = make_optimizer()
        first = LinearWarmUpConstantLR(
            first_optimizer,
            init=0.25,
            peak=1.0,
            warm_up_steps=2,
        )
        for _ in range(2):
            first_optimizer.step()
            first.step()
        first.set_tail_lr(0.5)

        second_optimizer = make_optimizer()
        second = LinearWarmUpConstantLR(
            second_optimizer,
            init=0.25,
            peak=1.0,
            warm_up_steps=2,
        )
        second.load_state_dict(first.state_dict())

        self.assertEqual([0.5], first.get_last_lr())
        self.assertEqual([0.5], second.get_last_lr())
        self.assertEqual(0.5, second_optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    unittest.main()

"""Runner integration: real public AdamW hook and CPU/CUDA GradScaler."""
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from mortal.core.update_clock import OptimizerUpdateClock, observed_scaler_step


@unittest.skipIf(torch is None, 'PyTorch is required for optimizer integration')
class TorchUpdateClockTests(unittest.TestCase):
    def exercise(self, device, enabled):
        param = torch.nn.Parameter(torch.tensor([1.0], device=device))
        optimizer = torch.optim.AdamW([param], lr=0.1, fused=False)
        scaler = torch.amp.GradScaler(device, enabled=enabled, init_scale=2.0, growth_interval=1)
        clock = OptimizerUpdateClock()
        for overflow in (False, True, False):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(param.square().sum()).backward()
            if overflow and enabled:
                param.grad.fill_(float('inf'))
            before = param.detach().clone()
            succeeded = observed_scaler_step(scaler, optimizer, clock)
            self.assertEqual(succeeded, not (overflow and enabled))
            self.assertEqual(torch.equal(before, param), not succeeded)
        self.assertEqual(clock.attempts, 3)
        self.assertEqual(clock.skips, int(enabled))
        self.assertEqual(clock.successes, 3 - int(enabled))
        self.assertEqual(int(optimizer.state[param]['step'].item()), clock.successes)

    def test_cpu_disabled(self):
        self.exercise('cpu', False)

    def test_cpu_amp_overflow_and_growth(self):
        self.exercise('cpu', True)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_amp_overflow_and_growth(self):
        self.exercise('cuda', True)


if __name__ == '__main__':
    unittest.main()

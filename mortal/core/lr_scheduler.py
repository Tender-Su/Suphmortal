import math
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmUpCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, *, peak, final, warm_up_steps, max_steps, init=1e-8, offset=0, epoch_size=0, **kwargs):
        assert peak >= final >= init >= 0
        assert max_steps >= warm_up_steps
        self.init = init
        self.peak = peak
        self.final = final
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.offset = offset
        self.epoch_size = epoch_size
        self.tail_lr = final
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Checkpoints created before convergence tails have no tail_lr. Their
        # exact continuation is the original cosine floor.
        if not hasattr(self, 'tail_lr'):
            self.tail_lr = self.final

    def set_tail_lr(self, value):
        value = float(value)
        if not 0 <= value <= self.final:
            raise ValueError(f'tail lr must be in [0, {self.final}], got {value}')
        self.tail_lr = value
        if self.last_epoch >= self.max_steps:
            self._last_lr = [value for _ in self.optimizer.param_groups]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

    def _step_inner(self, steps):
        steps += self.offset
        if self.epoch_size > 0:
            steps %= self.epoch_size
        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            return self.init + (self.peak - self.init) / self.warm_up_steps * steps
        if steps < self.max_steps:
            cos_steps = steps - self.warm_up_steps
            cos_max_steps = self.max_steps - self.warm_up_steps
            return self.final + 0.5 * (self.peak - self.final) * (1 + math.cos(cos_steps / cos_max_steps * math.pi))
        return self.tail_lr

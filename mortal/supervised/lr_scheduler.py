from torch.optim.lr_scheduler import LambdaLR


class LinearWarmUpConstantLR(LambdaLR):
    """Linear rewarm followed by a checkpointable constant tail LR."""

    def __init__(
        self,
        optimizer,
        *,
        peak,
        warm_up_steps,
        init=1e-8,
        **kwargs,
    ):
        if not peak >= init >= 0:
            raise ValueError('constant scheduler requires peak >= init >= 0')
        if warm_up_steps < 0:
            raise ValueError('constant scheduler warmup must be non-negative')
        self.init = float(init)
        self.peak = float(peak)
        self.warm_up_steps = int(warm_up_steps)
        self.tail_lr = self.peak
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if not hasattr(self, 'tail_lr'):
            self.tail_lr = self.peak
        if len(self._last_lr) != len(self.base_lrs):
            self._last_lr = [
                base_lr * self._step_inner(self.last_epoch)
                for base_lr in self.base_lrs
            ]
        self._apply_last_lr()

    def _apply_last_lr(self):
        for param_group, group_lr in zip(
            self.optimizer.param_groups,
            self._last_lr,
        ):
            param_group['lr'] = group_lr

    def set_tail_lr(self, value):
        value = float(value)
        if not 0 < value <= self.peak:
            raise ValueError(f'tail lr must be in (0, {self.peak}], got {value}')
        self.tail_lr = value
        if self.last_epoch >= self.warm_up_steps:
            self._last_lr = [base_lr * value for base_lr in self.base_lrs]
            self._apply_last_lr()

    def _step_inner(self, steps):
        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            return self.init + (
                (self.peak - self.init) * steps / self.warm_up_steps
            )
        return self.tail_lr

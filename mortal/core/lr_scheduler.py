import math
from torch.optim.lr_scheduler import LambdaLR


def normalize_scheduler_type(value):
    value = str(value or 'cosine').strip().lower().replace('-', '_')
    aliases = {
        'cos': 'cosine',
        'warmup_cosine': 'cosine',
        'stable_decay': 'wsd',
        'warmup_stable_decay': 'wsd',
        'reduce_on_plateau': 'plateau',
        'monitor_driven': 'plateau',
        'none': 'optimizer',
        'schedule_free': 'optimizer',
    }
    value = aliases.get(value, value)
    if value not in {'cosine', 'constant', 'wsd', 'plateau', 'optimizer'}:
        raise ValueError(f'unsupported lr scheduler type: {value!r}')
    return value

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
            self._last_lr = [base_lr * value for base_lr in self.base_lrs]
            for param_group, group_lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group['lr'] = group_lr

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


class LinearWarmUpStableDecayLR(LambdaLR):
    def __init__(
        self,
        optimizer,
        *,
        peak,
        final,
        warm_up_steps,
        stable_steps,
        decay_steps,
        init=1e-8,
        decay_style='linear',
        **kwargs,
    ):
        if not peak >= final >= init >= 0:
            raise ValueError('WSD requires peak >= final >= init >= 0')
        if min(warm_up_steps, stable_steps, decay_steps) < 0:
            raise ValueError('WSD step counts must be non-negative')
        decay_style = str(decay_style).strip().lower()
        if decay_style not in {'linear', 'cosine'}:
            raise ValueError('WSD decay_style must be linear or cosine')
        self.init = float(init)
        self.peak = float(peak)
        self.final = float(final)
        self.warm_up_steps = int(warm_up_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.decay_style = decay_style
        self.tail_lr = self.final
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if not hasattr(self, 'tail_lr'):
            self.tail_lr = self.final

    def set_tail_lr(self, value):
        if self.decay_steps != 0 or not math.isclose(
            self.final,
            self.peak,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError('dynamic tail lr is only supported by constant schedules')
        value = float(value)
        if not 0 < value <= self.peak:
            raise ValueError(f'tail lr must be in (0, {self.peak}], got {value}')
        self.tail_lr = value
        if self.last_epoch >= self.warm_up_steps:
            self._last_lr = [base_lr * value for base_lr in self.base_lrs]
            for param_group, group_lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group['lr'] = group_lr

    def _step_inner(self, steps):
        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            return self.init + (
                (self.peak - self.init) * steps / self.warm_up_steps
            )
        decay_start = self.warm_up_steps + self.stable_steps
        if self.decay_steps == 0:
            return self.tail_lr
        if steps < decay_start:
            return self.peak
        decay_progress = min(max(steps - decay_start, 0), self.decay_steps)
        ratio = decay_progress / self.decay_steps
        if self.decay_style == 'cosine':
            ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
        return self.peak + (self.final - self.peak) * ratio


class LinearWarmUpReduceOnPlateauLR:
    """Per-update warmup with validation-driven, checkpointable LR reductions."""

    def __init__(
        self,
        optimizer,
        *,
        peak,
        warm_up_steps,
        init=1e-8,
        factor=0.5,
        patience_steps=40000,
        threshold=0.0,
        min_lr=1e-6,
        **_kwargs,
    ):
        if not peak >= init >= 0:
            raise ValueError('plateau scheduler requires peak >= init >= 0')
        if not 0 < factor < 1:
            raise ValueError('plateau factor must be in (0, 1)')
        if warm_up_steps < 0 or patience_steps <= 0:
            raise ValueError('plateau warmup must be non-negative and patience positive')
        if not 0 <= min_lr <= peak:
            raise ValueError('plateau min_lr must be in [0, peak]')
        self.optimizer = optimizer
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.peak = float(peak)
        self.init = float(init)
        self.warm_up_steps = int(warm_up_steps)
        self.factor = float(factor)
        self.patience_steps = int(patience_steps)
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)
        self.plateau_lr = self.peak
        self.best = math.inf
        self.last_improvement_step = 0
        self.num_reductions = 0
        self.last_epoch = 0
        self._last_lr = []
        self._apply_lr(self._lr_at_step(0))

    def _lr_at_step(self, steps):
        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            return self.init + (
                (self.plateau_lr - self.init) * steps / self.warm_up_steps
            )
        return self.plateau_lr

    def _apply_lr(self, lr):
        self._last_lr = [base_lr * lr for base_lr in self.base_lrs]
        for group, group_lr in zip(self.optimizer.param_groups, self._last_lr):
            group['lr'] = group_lr

    def step(self):
        self.last_epoch += 1
        self._apply_lr(self._lr_at_step(self.last_epoch))

    def observe(self, metric, optimizer_steps):
        metric = float(metric)
        optimizer_steps = int(optimizer_steps)
        if not math.isfinite(metric):
            return {'action': 'ignore_non_finite', 'lr': self.plateau_lr}
        if metric < self.best - self.threshold:
            self.best = metric
            self.last_improvement_step = optimizer_steps
            return {'action': 'improved', 'lr': self.plateau_lr}
        if optimizer_steps - self.last_improvement_step < self.patience_steps:
            return {'action': 'hold', 'lr': self.plateau_lr}
        reduced = max(self.min_lr, self.plateau_lr * self.factor)
        if math.isclose(reduced, self.plateau_lr, rel_tol=0.0, abs_tol=1e-15):
            return {'action': 'at_min_lr', 'lr': self.plateau_lr}
        self.plateau_lr = reduced
        self.last_improvement_step = optimizer_steps
        self.num_reductions += 1
        self._apply_lr(self._lr_at_step(self.last_epoch))
        return {'action': 'reduce_lr', 'lr': self.plateau_lr}

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != 'optimizer'
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._apply_lr(self._lr_at_step(self.last_epoch))


class OptimizerLRSnapshot:
    """Checkpointable LR control for optimizers that own their warmup."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.tail_lr = max(self.base_lrs, default=0.0)

    def _apply_tail_lr(self):
        base_lr = max(self.base_lrs, default=0.0)
        if base_lr <= 0:
            raise ValueError('optimizer-owned LR control requires positive base lrs')
        lrs = [self.tail_lr * value / base_lr for value in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group['lr'] = lr
            group['scheduled_lr'] = lr

    def set_tail_lr(self, value):
        value = float(value)
        base_lr = max(self.base_lrs, default=0.0)
        if not 0 < value <= base_lr:
            raise ValueError(f'tail lr must be in (0, {base_lr}], got {value}')
        self.tail_lr = value
        self._apply_tail_lr()

    def step(self):
        self.last_epoch += 1

    def get_last_lr(self):
        return [
            float(group.get('scheduled_lr', group['lr']))
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': list(self.base_lrs),
            'tail_lr': self.tail_lr,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict.get('last_epoch', 0))
        self.base_lrs = [
            float(value)
            for value in state_dict.get('base_lrs', self.base_lrs)
        ]
        self.tail_lr = float(
            state_dict.get('tail_lr', max(self.base_lrs, default=0.0))
        )
        if 'tail_lr' in state_dict:
            self._apply_tail_lr()


def build_lr_scheduler(optimizer, scheduler_cfg):
    scheduler_cfg = dict(scheduler_cfg)
    scheduler_type = normalize_scheduler_type(scheduler_cfg.pop('type', 'cosine'))
    if scheduler_type == 'cosine':
        return LinearWarmUpCosineAnnealingLR(
            optimizer,
            **{
                key: scheduler_cfg[key]
                for key in ('peak', 'final', 'warm_up_steps', 'max_steps')
            },
            init=float(scheduler_cfg.get('init', 1e-8)),
        )
    if scheduler_type == 'constant':
        peak = float(scheduler_cfg['peak'])
        return LinearWarmUpStableDecayLR(
            optimizer,
            peak=peak,
            final=peak,
            warm_up_steps=int(scheduler_cfg.get('warm_up_steps', 0)),
            stable_steps=0,
            decay_steps=0,
            init=float(scheduler_cfg.get('init', 1e-8)),
        )
    if scheduler_type == 'wsd':
        return LinearWarmUpStableDecayLR(
            optimizer,
            **{
                key: scheduler_cfg[key]
                for key in (
                    'peak',
                    'final',
                    'warm_up_steps',
                    'stable_steps',
                    'decay_steps',
                )
            },
            init=float(scheduler_cfg.get('init', 1e-8)),
            decay_style=str(scheduler_cfg.get('decay_style', 'linear')),
        )
    if scheduler_type == 'plateau':
        return LinearWarmUpReduceOnPlateauLR(optimizer, **scheduler_cfg)
    return OptimizerLRSnapshot(optimizer)

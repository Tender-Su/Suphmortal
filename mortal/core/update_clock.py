"""Observed optimizer invocations, separate from the online microbatch clock.

For non-fused AdamW, GradScaler skips optimizer.step on overflow, so a public
post-step hook observes a real update without interpreting step's return value
or scale growth/backoff (which can saturate). This is not an LR-clock migration.
"""
from dataclasses import asdict, dataclass


@dataclass
class OptimizerUpdateClock:
    # Historical counts may include AMP skips. Never label this offset successful.
    legacy_attempt_offset: int = 0
    legacy_source: str = 'fresh'
    inherited_progress_offset: int = 0
    inherited_source: str = 'none'
    attempts: int = 0
    successes: int = 0
    skips: int = 0

    @property
    def progress(self):
        """Compatibility/danger-ramp clock, not an exact lifetime success count."""
        return self.legacy_attempt_offset + self.inherited_progress_offset + self.successes

    def record(self, succeeded):
        self.attempts += 1
        if succeeded:
            self.successes += 1
        else:
            self.skips += 1

    def state_dict(self):
        return {'version': 1, **asdict(self)}

    @classmethod
    def from_checkpoint(cls, state, *, opt_step_every, weights_only=False):
        saved = state.get('optimizer_update_clock')
        if saved is not None:
            if not isinstance(saved, dict) or saved.get('version') != 1:
                raise ValueError('unsupported optimizer update clock')
            missing = sorted(set(cls.__dataclass_fields__) - saved.keys())
            if missing:
                raise ValueError(f'optimizer update clock missing fields: {missing}')
            clock = cls(**{key: saved[key] for key in cls.__dataclass_fields__})
            for key in ('legacy_attempt_offset', 'inherited_progress_offset', 'attempts', 'successes', 'skips'):
                value = getattr(clock, key)
                if type(value) is not int or value < 0:
                    raise ValueError(f'invalid optimizer update clock {key}: {value!r}')
            if clock.attempts != clock.successes + clock.skips:
                raise ValueError('optimizer update clock attempts != successes + skips')
            if clock.legacy_source not in ('fresh', 'optimizer_steps', 'inferred_microbatches'):
                raise ValueError('invalid optimizer update clock legacy source')
            if clock.inherited_source not in ('none', 'weights_only'):
                raise ValueError('invalid optimizer update clock inherited source')
            if 'optimizer_steps' in state and state['optimizer_steps'] != clock.progress:
                raise ValueError('optimizer_steps disagrees with optimizer update clock progress')
            if weights_only:
                return clock.for_weights_only()
            return clock
        if 'optimizer_steps' in state:
            offset = int(state['optimizer_steps'])
            source = 'optimizer_steps'
        else:
            # Retain the old fallback, including its rounding, for ramp continuity.
            accumulation = max(int(opt_step_every), 1)
            offset = (int(state.get('steps', 0) or 0) + accumulation - 1) // accumulation
            source = 'inferred_microbatches'
        if offset < 0:
            raise ValueError('negative legacy optimizer update offset')
        clock = cls(legacy_attempt_offset=offset, legacy_source=source)
        return clock.for_weights_only() if weights_only else clock

    def for_weights_only(self):
        # Model history/ramp continuity survives; this optimizer starts at zero.
        return type(self)(
            legacy_attempt_offset=self.legacy_attempt_offset,
            legacy_source=self.legacy_source,
            inherited_progress_offset=self.inherited_progress_offset + self.successes,
            inherited_source='weights_only',
        )


def observed_scaler_step(scaler, optimizer, clock):
    """Step/update the scaler and record completion for a non-fused optimizer.

    Caller still zeroes gradients and advances its normal data/microbatch clocks
    on overflow. No gradient is an invalid attempt, even with AMP disabled.
    Fused optimizers may invoke step while suppressing the kernel update, so this
    observer deliberately rejects that configuration. Exceptions are not skips.
    """
    if optimizer.defaults.get('fused', False) or any(
        group.get('fused', False) for group in optimizer.param_groups
    ):
        raise ValueError('optimizer update observer requires non-fused optimizer')
    if not any(p.grad is not None for g in optimizer.param_groups for p in g['params']):
        raise ValueError('optimizer update attempt has no gradients')
    succeeded = False

    def after_step(_optimizer, _args, _kwargs):
        nonlocal succeeded
        succeeded = True

    handle = optimizer.register_step_post_hook(after_step)
    try:
        scaler.step(optimizer)
    finally:
        handle.remove()
    scaler.update()
    clock.record(succeeded)
    return succeeded

"""Explicit critic-only phases; clocks remain separate from legacy LR steps."""
from contextlib import nullcontext


def critic_only_enabled(config):
    value = config.get('value', {})
    enabled = bool(value.get('critic_only', False))
    if enabled and not bool(value.get('enabled', False)):
        raise ValueError('value.critic_only requires value.enabled=true')
    return enabled


def successful_optimizer_step_limit(config):
    limit = config.get('online', {}).get('max_successful_optimizer_steps', 0)
    if type(limit) is not int or limit < 0:
        raise ValueError('online.max_successful_optimizer_steps must be a nonnegative integer')
    return limit


def successful_optimizer_step_limit_reached(config, clock):
    limit = successful_optimizer_step_limit(config)
    # Never count legacy attempts or weights-only inherited model progress.
    return limit > 0 and clock.successes >= limit


def restore_actor_training_mode(mortal, policy_net, *, critic_only):
    mortal.train(not critic_only)
    policy_net.train(not critic_only)


def actor_forward_context(mortal, policy_net, *, frozen, critic_only):
    """Freeze buffers as well as autograd only for the explicit calibration phase.

    Ordinary policy-inactive alternating steps deliberately retain shared-trunk
    value gradients. Legacy finite warmup retains its pre-existing mode behavior.
    """
    if critic_only:
        restore_actor_training_mode(mortal, policy_net, critic_only=True)
    if frozen or critic_only:
        import torch
        return torch.no_grad()
    return nullcontext()


def validate_calibration_resume(state, config):
    """Full resume continues one phase; a new phase must use weights-only init."""
    if critic_only_enabled(state.get('config', {})) != critic_only_enabled(config):
        raise ValueError(
            'cannot change value.critic_only during full checkpoint resume; '
            'use a new control.state_file and online.init_state_file for a new phase'
        )
    if successful_optimizer_step_limit(config) and 'optimizer_update_clock' not in state:
        raise ValueError(
            'bounded full resume requires an exact optimizer_update_clock; '
            'use weights-only initialization for a fresh phase'
        )

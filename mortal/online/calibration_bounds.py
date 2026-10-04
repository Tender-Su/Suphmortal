"""Opt-in limits and raw complete-game MC targets for critic-only calibration."""
def calibration_limits(config):
    online = config.get('online', {})
    limits = []
    for name in ('max_replay_passes', 'max_optimizer_attempts'):
        value = online.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
        limits.append(value)
    raw = bool(config.get('value', {}).get('fixed_mc_targets', False))
    if any(limits) or raw:
        value, policy = config.get('value', {}), config.get('policy', {})
        if not (value.get('enabled') and value.get('critic_only') and value.get('oracle_critic')
                and value.get('target_mode') == 'all_players'
                and value.get('reward_source') == 'score_rank'
                and policy.get('gae_enabled') and policy.get('gae_gamma') == 1.
                and policy.get('gae_lambda') == 1.
                and policy.get('online_action_scope') == 'all'):
            raise ValueError('bounded calibration requires frozen actor and raw all_players MC')
        if online.get('importance_sampling', {}).get('enabled'):
            raise ValueError('bounded frozen-policy calibration cannot use replay importance sampling')
        if config.get('oracle_guiding', {}).get('actor_enabled'):
            raise ValueError('calibration actor must be visible only')
    return tuple(limits)


def calibration_limit_reason(limits, *, completed_passes, attempts):
    passes, max_attempts = limits
    if max_attempts and attempts >= max_attempts:
        return 'optimizer_attempt_limit'
    if passes and completed_passes >= passes:
        return 'replay_pass_limit'
    return None


def complete_mc_target(trajectory):
    import numpy as np
    from mortal.data.oracle_value import expand_kyoku_rewards_to_steps, discounted_returns_from_step_rewards
    rewards = expand_kyoku_rewards_to_steps(
        trajectory['kyoku_value_target'], trajectory['at_kyoku'])
    target = discounted_returns_from_step_rewards(rewards, 1.)
    if target.shape != (len(trajectory['obs']), 4) or not np.isfinite(target).all():
        raise ValueError('complete MC target requires finite four-head full trajectories')
    return target.astype(np.float32, copy=False)

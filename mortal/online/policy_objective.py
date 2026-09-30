"""Explicit PPO and IMPALA actor contracts (one importance correction)."""
import math

import torch


ACTOR_OBJECTIVE_VERSION = 'ppo_or_vtrace_pg_v2'


def validate_actor_objective(config, *, vtrace_enabled, gae_enabled, replay_is):
    policy = config.get('policy', {})
    objective = str(policy.get('actor_objective', 'ppo')).lower()
    if objective not in {'ppo', 'vtrace'}:
        raise ValueError('policy.actor_objective must be ppo or vtrace')
    if objective == 'ppo' and vtrace_enabled:
        raise ValueError('PPO cannot consume rho-weighted V-trace advantages; disable '
                         'V-trace or explicitly select actor_objective=vtrace')
    if objective == 'vtrace':
        sampling = config.get('online', {}).get('importance_sampling', {})
        if not (vtrace_enabled and gae_enabled and replay_is
                and sampling.get('vtrace_mode') == 'always'
                and sampling.get('drop_untracked_samples', False)):
            raise ValueError('V-trace actor requires full trajectories, tracked behavior '
                             'policies, drop_untracked_samples=true and vtrace_mode=always')
    for key, default in [('target_kl', 0.02), ('max_clip_fraction', 0.5)]:
        value = float(policy.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'policy.{key} must be finite and positive')
    return objective


def actor_surrogate(new_log_prob, ratio, advantage, *, objective, clip_ratio,
                    dual_clip=0.0, importance_rho_clip=0.0):
    if objective == 'vtrace':
        # advantage already includes detached rho from the V-trace recursion.
        # IMPALA eq. 4: rho * grad log pi * (r + gamma v_next - V).
        return new_log_prob * advantage.detach()
    if objective != 'ppo':
        raise ValueError(f'unsupported actor objective: {objective}')
    rho = ratio.clamp(max=importance_rho_clip) if importance_rho_clip > 0 else ratio
    result = torch.minimum(rho * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    if dual_clip > 1:
        result = torch.where(advantage < 0, torch.maximum(result, dual_clip * advantage), result)
    return result


def policy_drift(ratio, clip_ratio):
    ratio = ratio.detach().float()
    if ratio.numel() == 0 or not bool(torch.isfinite(ratio).all()) or bool((ratio <= 0).any()):
        raise ValueError('non-finite or zero policy ratio; no optimizer update is allowed')
    return {
        'approx_kl': ((ratio - 1) - ratio.log()).mean(),
        'clip_fraction': ((ratio - 1).abs() > clip_ratio).float().mean(),
    }


def normalized_behavior_version(value):
    return -1 if value is None else int(value)


def behavior_version_is_usable(value, history, *, published_version, max_gap):
    version = normalized_behavior_version(value)
    return (version >= 0 and version <= published_version and history.get(version) is not None
            and (max_gap < 0 or published_version - version <= max_gap))

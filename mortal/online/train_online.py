import math as _math
import numpy as _np
from collections import OrderedDict

from mortal.eval.oracle_experiments import (
    apply_oracle_experiment_to_config,
    apply_oracle_input_mode,
    normalize_oracle_input_mode,
)
from mortal.core.artifacts import atomic_torch_save
from mortal.core.training_stop import ONLINE_STOP_REQUEST_EXIT_CODE, training_stop_requested
from mortal.core.update_clock import OptimizerUpdateClock, observed_scaler_step
from mortal.core.turn_weighting import compute_turn_bucket_weights, resolve_turn_weighting_cfg
from mortal.online.policy_objective import (
    ACTOR_OBJECTIVE_VERSION, actor_surrogate, behavior_version_is_usable,
    normalized_behavior_version, policy_drift, validate_actor_objective,
)

from mortal.online.critic_calibration import (
    actor_forward_context, critic_only_enabled, restore_actor_training_mode,
    successful_optimizer_step_limit, successful_optimizer_step_limit_reached,
    validate_calibration_resume,
)

ONLINE_MAX_STEPS_EXIT_CODE = 86


def oracle_guiding_cfg(config):
    cfg = config.get('oracle_guiding', {})
    return cfg if isinstance(cfg, dict) else {}


def actor_oracle_guiding_enabled(config):
    return bool(oracle_guiding_cfg(config).get('actor_enabled', False))


def actor_oracle_guiding_source(config):
    cfg = oracle_guiding_cfg(config)
    enabled = bool(cfg.get('actor_enabled', False))
    return normalize_oracle_input_mode(
        cfg.get('actor_source', 'true' if enabled else 'zero'),
        field_name='oracle_guiding.actor_source',
    )


def actor_oracle_guiding_keep_prob(config, steps):
    cfg = oracle_guiding_cfg(config)
    if not bool(cfg.get('actor_enabled', False)):
        return 0.0

    schedule = str(cfg.get('schedule', 'linear') or 'linear').strip().lower()
    gamma_start = float(cfg.get('gamma_start', 1.0))
    gamma_end = float(cfg.get('gamma_end', 0.0))
    hold_steps = max(int(cfg.get('hold_steps', 0) or 0), 0)
    decay_steps = max(int(cfg.get('decay_steps', 0) or 0), 0)
    steps = max(int(steps), 0)

    if schedule == 'none':
        return max(0.0, min(1.0, gamma_start))
    if steps <= hold_steps:
        return max(0.0, min(1.0, gamma_start))
    if decay_steps <= 0:
        return max(0.0, min(1.0, gamma_end))

    progress = min(max(steps - hold_steps, 0) / decay_steps, 1.0)
    if schedule == 'cosine':
        cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        value = gamma_end + (gamma_start - gamma_end) * cosine
    else:
        value = gamma_start + (gamma_end - gamma_start) * progress
    return max(0.0, min(1.0, float(value)))


def actor_oracle_guiding_continuation_active(config, steps):
    return actor_oracle_guiding_enabled(config) and actor_oracle_guiding_keep_prob(config, steps) <= 0.0


def actor_oracle_guiding_lr_scale(config):
    cfg = oracle_guiding_cfg(config)
    return float(cfg.get('continuation_lr_scale', 0.1) or 0.1)


def actor_oracle_guiding_importance_threshold(config):
    cfg = oracle_guiding_cfg(config)
    return float(cfg.get('importance_weight_threshold', 0.0) or 0.0)


def actor_oracle_guiding_runtime_state(config, steps):
    return {
        'oracle_experiment_arm': str(
            config.get('oracle_experiments', {}).get('resolved_arm', 'current_config')
            if isinstance(config.get('oracle_experiments', {}), dict)
            else 'current_config'
        ),
        'actor_oracle_enabled': actor_oracle_guiding_enabled(config),
        'actor_oracle_keep_prob': actor_oracle_guiding_keep_prob(config, steps),
        'actor_oracle_continuation': actor_oracle_guiding_continuation_active(config, steps),
        'actor_oracle_source': actor_oracle_guiding_source(config),
        'oracle_critic_enabled': bool(
            isinstance(config.get('value', {}), dict)
            and config.get('value', {}).get('enabled', False)
            and config.get('value', {}).get('oracle_critic', True)
        ),
    }


def transform_actor_oracle_invisible_obs(invisible_obs, *, actor_source, keep_prob):
    from mortal.core.model import apply_oracle_obs_keep_prob

    transformed = apply_oracle_input_mode(invisible_obs, actor_source)
    return apply_oracle_obs_keep_prob(transformed, keep_prob)


def replay_importance_sampling_cfg(config):
    online_cfg = config.get('online', {})
    if not isinstance(online_cfg, dict):
        return {}
    cfg = online_cfg.get('importance_sampling', {})
    return cfg if isinstance(cfg, dict) else {}


def replay_importance_sampling_enabled(config):
    return bool(replay_importance_sampling_cfg(config).get('enabled', False))


def replay_importance_sampling_max_versions(config):
    cfg = replay_importance_sampling_cfg(config)
    return max(int(cfg.get('max_policy_versions', 8) or 8), 1)


def replay_importance_sampling_drop_untracked(config):
    cfg = replay_importance_sampling_cfg(config)
    return bool(cfg.get('drop_untracked_samples', False))


def replay_importance_sampling_vtrace_mode(config):
    cfg = replay_importance_sampling_cfg(config)
    raw_mode = str(cfg.get('vtrace_mode', 'auto') or 'auto').strip().lower()
    if raw_mode in ('auto', 'lag', 'replay_gap', 'stale_only'):
        return 'auto'
    if raw_mode in ('always', 'force', 'on', 'enabled', 'true'):
        return 'always'
    if raw_mode in ('disabled', 'off', 'false', 'never', 'none'):
        return 'disabled'
    raise ValueError(
        f"unsupported online.importance_sampling.vtrace_mode={raw_mode!r}; "
        "expected 'auto', 'always', or 'disabled'"
    )


def replay_importance_sampling_vtrace_min_version_gap(config):
    cfg = replay_importance_sampling_cfg(config)
    return max(int(cfg.get('vtrace_min_version_gap', 2) or 0), 0)


def replay_version_gap(*, published_param_version, replay_param_version):
    if published_param_version is None or replay_param_version is None:
        return None
    published = int(published_param_version)
    replay = int(replay_param_version)
    if published < 0 or replay < 0:
        return None
    return max(published - replay, 0)


def replay_importance_sampling_should_use_vtrace(
    config,
    *,
    published_param_version,
    replay_param_version,
):
    mode = replay_importance_sampling_vtrace_mode(config)
    if mode == 'disabled':
        return False
    if mode == 'always':
        return True
    version_gap = replay_version_gap(
        published_param_version=published_param_version,
        replay_param_version=replay_param_version,
    )
    if version_gap is None:
        return False
    return version_gap >= replay_importance_sampling_vtrace_min_version_gap(config)


def online_scheduler_max_steps(config):
    optim_cfg = config.get('optim', {})
    if not isinstance(optim_cfg, dict):
        return 0
    scheduler_cfg = optim_cfg.get('scheduler', {})
    if not isinstance(scheduler_cfg, dict):
        return 0
    return max(int(scheduler_cfg.get('max_steps', 0) or 0), 0)


def online_stop_at_max_steps(config):
    online_cfg = config.get('online', {})
    if not isinstance(online_cfg, dict):
        return True
    return bool(online_cfg.get('stop_at_max_steps', True))


def online_reached_max_steps(config, steps):
    max_steps = online_scheduler_max_steps(config)
    return online_stop_at_max_steps(config) and max_steps > 0 and int(steps) >= max_steps


def online_gae_inference_batch_size(config):
    """Physical forward block size; complete trajectories and shuffle chunks stay fixed."""
    value = int(config.get('online', {}).get('gae_inference_batch_size', 2048))
    if value <= 0:
        raise ValueError('online.gae_inference_batch_size must be positive')
    return value


def test_play_cfg(config):
    cfg = config.get('test_play', {})
    return cfg if isinstance(cfg, dict) else {}


def test_play_enabled(config):
    return bool(test_play_cfg(config).get('enable', True))


def test_play_games(config):
    return int(test_play_cfg(config).get('games', 0) or 0)


def initial_test_play_enabled(config):
    return bool(test_play_cfg(config).get('initial_enable', False))


def initial_test_play_games(config):
    cfg = test_play_cfg(config)
    return int(cfg.get('initial_games', cfg.get('games', 0)) or 0)


def periodic_test_play_due(*, enabled, steps, test_every):
    return bool(enabled) and int(test_every) > 0 and int(steps) % int(test_every) == 0


def old_policy_update_due(steps, old_update_every):
    return int(old_update_every) > 0 and int(steps) > 0 and int(steps) % int(old_update_every) == 0


def refresh_old_policy_snapshot(old_mortal, old_policy_net, mortal, policy_net):
    old_mortal.load_state_dict(mortal.state_dict())
    old_policy_net.load_state_dict(policy_net.state_dict())


def recorded_step0_baseline(config):
    profile_cfg = config.get('online_experiment_profile', {})
    if not isinstance(profile_cfg, dict):
        return None
    baseline = profile_cfg.get('recorded_step0_baseline')
    return baseline if isinstance(baseline, dict) else None


def policy_online_action_scope(config):
    policy_cfg = config.get('policy', {})
    if not isinstance(policy_cfg, dict):
        return 'all'
    scope = str(policy_cfg.get('online_action_scope', 'all') or 'all').strip().lower()
    if scope != 'all':
        raise ValueError(f"unsupported policy.online_action_scope={scope!r}; expected 'all'")
    return scope


def policy_online_action_keep_mask(actions, scope):
    if scope == 'all':
        return None
    raise ValueError(f'unsupported online action scope: {scope!r}')


def policy_training_cfg(config):
    cfg = config.get('policy', {})
    return cfg if isinstance(cfg, dict) else {}


def policy_logit_gate_threshold(config):
    return float(policy_training_cfg(config).get('logit_thres', 0.0) or 0.0)


def policy_importance_rho_clip(config):
    cfg = policy_training_cfg(config)
    if 'importance_rho_clip' in cfg:
        return float(cfg.get('importance_rho_clip', 0.0) or 0.0)
    return float(cfg.get('vtrace_rho_clip', 0.0) or 0.0)


def policy_importance_c_clip(config):
    cfg = policy_training_cfg(config)
    if 'importance_c_clip' in cfg:
        return float(cfg.get('importance_c_clip', 0.0) or 0.0)
    return float(cfg.get('vtrace_c_clip', 0.0) or 0.0)


def policy_vtrace_target_rho_clip(config):
    cfg = policy_training_cfg(config)
    if 'vtrace_target_rho_clip' in cfg:
        return float(cfg.get('vtrace_target_rho_clip', 0.0) or 0.0)
    if 'vtrace_rho_clip' in cfg:
        return float(cfg.get('vtrace_rho_clip', 0.0) or 0.0)
    return float(cfg.get('importance_rho_clip', 0.0) or 0.0)


def policy_vtrace_target_c_clip(config):
    cfg = policy_training_cfg(config)
    if 'vtrace_target_c_clip' in cfg:
        return float(cfg.get('vtrace_target_c_clip', 0.0) or 0.0)
    if 'vtrace_c_clip' in cfg:
        return float(cfg.get('vtrace_c_clip', 0.0) or 0.0)
    return float(cfg.get('importance_c_clip', 0.0) or 0.0)


def policy_entropy_floor(config):
    cfg = policy_training_cfg(config)
    if 'entropy_floor' in cfg:
        return float(cfg.get('entropy_floor', 0.0) or 0.0)
    return float(cfg.get('entropy_target', 0.0) or 0.0)


def policy_entropy_floor_start_step(config, *, default=0):
    cfg = policy_training_cfg(config)
    if 'entropy_floor_start_step' in cfg:
        return max(int(cfg.get('entropy_floor_start_step', default) or default), 0)
    return max(int(default or 0), 0)


def policy_actor_lr_scale(config):
    cfg = policy_training_cfg(config)
    return max(float(cfg.get('actor_lr_scale', 1.0) or 0.0), 0.0)


def policy_head_lr_scale(config):
    cfg = policy_training_cfg(config)
    return max(float(cfg.get('policy_head_lr_scale', 1.0) or 0.0), 0.0)


def policy_update_interval(config):
    cfg = policy_training_cfg(config)
    return max(int(cfg.get('update_interval', 1) or 1), 1)


def policy_update_phase(config):
    cfg = policy_training_cfg(config)
    return max(int(cfg.get('update_phase', 0) or 0), 0)


def policy_update_active(config, steps):
    interval = policy_update_interval(config)
    phase = policy_update_phase(config) % interval
    return int(steps) % interval == phase


def value_training_cfg(config):
    cfg = config.get('value', {})
    return cfg if isinstance(cfg, dict) else {}


def value_critic_warmup_steps(config):
    cfg = value_training_cfg(config)
    return max(int(cfg.get('critic_warmup_steps', cfg.get('actor_freeze_steps', 0)) or 0), 0)


def value_critic_warmup_active(config, steps):
    cfg = value_training_cfg(config)
    return critic_only_enabled(config) or (
        bool(cfg.get('enabled', False)) and int(steps) < value_critic_warmup_steps(config)
    )


def value_independent_actor_lr_clock(config):
    cfg = value_training_cfg(config)
    if not bool(cfg.get('enabled', False)):
        return False
    configured = cfg.get('independent_actor_lr_clock')
    if configured is not None:
        return bool(configured)
    return value_critic_warmup_steps(config) > 0


def actor_lr_clock_cfg(config):
    scheduler_cfg = dict(config.get('optim', {}).get('scheduler', {}))
    critic_warmup_steps = value_critic_warmup_steps(config)
    total_steps = max(int(scheduler_cfg.get('max_steps', 0) or 0), 1)
    actor_max_steps = max(total_steps - critic_warmup_steps, 1)
    scheduler_cfg['max_steps'] = actor_max_steps
    scheduler_cfg['warm_up_steps'] = min(
        max(int(scheduler_cfg.get('warm_up_steps', 0) or 0), 0),
        actor_max_steps,
    )
    return scheduler_cfg


def lr_at_step(scheduler_cfg, steps):
    init = float(scheduler_cfg.get('init', 1e-8))
    peak = float(scheduler_cfg.get('peak', 0.0))
    final = float(scheduler_cfg.get('final', 0.0))
    warm_up_steps = max(int(scheduler_cfg.get('warm_up_steps', 0) or 0), 0)
    max_steps = max(int(scheduler_cfg.get('max_steps', 0) or 0), warm_up_steps, 1)
    steps = max(int(steps), 0)
    if not peak >= final >= init >= 0:
        raise ValueError(
            f'invalid actor scheduler values: peak={peak}, final={final}, init={init}'
        )
    if warm_up_steps > 0 and steps < warm_up_steps:
        return init + (peak - init) / warm_up_steps * steps
    if steps < max_steps:
        cosine_steps = steps - warm_up_steps
        cosine_max_steps = max(max_steps - warm_up_steps, 1)
        return final + 0.5 * (peak - final) * (
            1 + _math.cos(cosine_steps / cosine_max_steps * _math.pi)
        )
    return final


def apply_independent_actor_lr_clock(optimizer, scheduler, config, *, steps):
    if not value_independent_actor_lr_clock(config):
        return None
    actor_steps = max(int(steps) - value_critic_warmup_steps(config), 0)
    actor_lr = lr_at_step(actor_lr_clock_cfg(config), actor_steps)
    changed = False
    for group in optimizer.param_groups:
        if group.get('schedule_role') != 'actor':
            continue
        group['lr'] = actor_lr * float(group.get('lr_scale', 1.0))
        changed = True
    if not changed:
        raise RuntimeError(
            'independent actor LR clock is enabled but optimizer has no actor parameter groups'
        )
    if hasattr(scheduler, '_last_lr'):
        scheduler._last_lr = [
            float(group.get('lr', 0.0) or 0.0)
            for group in optimizer.param_groups
        ]
    return actor_lr


def normalize_value_target_mode(value, *, oracle_critic):
    mode = str(value or 'auto').strip().lower()
    if mode == 'auto':
        return 'all_players' if oracle_critic else 'current_player'
    if mode in ('current', 'current_player', 'self', 'one'):
        return 'current_player'
    if mode in ('all', 'all_players', 'four_player', '4p'):
        return 'all_players'
    raise ValueError(
        f"unsupported value.target_mode={value!r}; expected 'auto', 'current_player', or 'all_players'"
    )


def value_target_mode(config):
    cfg = value_training_cfg(config)
    enabled = bool(cfg.get('enabled', False))
    oracle_critic = bool(cfg.get('oracle_critic', True)) if enabled else False
    raw_mode = cfg.get('target_mode')
    if raw_mode is None:
        legacy_num_players = int(cfg.get('num_players', 0) or 0)
        if legacy_num_players == 1:
            raw_mode = 'current_player'
        elif legacy_num_players == 4:
            raw_mode = 'all_players'
        else:
            raw_mode = 'auto'
    return normalize_value_target_mode(raw_mode, oracle_critic=oracle_critic)


def value_reward_source(config):
    cfg = value_training_cfg(config)
    enabled = bool(cfg.get('enabled', False))
    oracle_critic = bool(cfg.get('oracle_critic', True)) if enabled else False
    raw_source = cfg.get('reward_source')
    if raw_source is None:
        raw_source = 'score_rank' if oracle_critic else 'grp'
    from mortal.data.dataloader import normalize_value_reward_source

    return normalize_value_reward_source(raw_source)


def value_num_players_from_mode(target_mode):
    if target_mode == 'current_player':
        return 1
    if target_mode == 'all_players':
        return 4
    raise ValueError(f'unsupported normalized value target mode: {target_mode!r}')


def normalize_oracle_critic_arch(value):
    arch = str(value or 'single_tower').strip().lower()
    if arch in ('single', 'bridge', 'resnet', 'single_tower'):
        return 'single_tower'
    if arch in ('dual', 'two_tower', 'dual_tower'):
        return 'dual_tower'
    raise ValueError(
        f"unsupported value.oracle_critic_arch={arch!r}; "
        "expected 'single_tower' or 'dual_tower'"
    )


def infer_oracle_critic_arch_from_state_dict(state_dict, *, checkpoint_name):
    if not isinstance(state_dict, dict):
        raise ValueError(
            f'{checkpoint_name} metadata missing critic_arch and checkpoint has no '
            'oracle_brain state dict to infer it from'
        )
    keys = tuple(str(key) for key in state_dict.keys())
    if any(
        key.startswith('visible_encoder.')
        or key.startswith('oracle_encoder.')
        or key.startswith('fusion.')
        for key in keys
    ):
        return 'dual_tower'
    if any(key.startswith('encoder.') for key in keys):
        return 'single_tower'
    raise ValueError(
        f'{checkpoint_name} metadata missing critic_arch and oracle_brain keys do '
        'not match a known Oracle critic architecture'
    )


def centered_reward_signature(config):
    pts = config.get('env', {}).get('pts')
    if pts is None:
        return None
    values = _np.asarray(pts, dtype=_np.float64)
    if values.shape != (4,) or not _np.isfinite(values).all():
        raise ValueError('env.pts must contain four finite rank rewards')
    return tuple((values - values.mean()).tolist())


def online_value_architecture(config):
    cfg = value_training_cfg(config)
    return {
        'oracle_fusion_mode': str(cfg.get('oracle_fusion_mode', 'linear')),
        'oracle_fusion_hidden': int(cfg.get('oracle_fusion_hidden', 512)),
        'value_head_hidden': int(cfg.get('value_head_hidden', 256)),
        'value_loss_mode': str(cfg.get('value_loss_mode', 'mse')),
    }


def build_online_value_models(config, *, device):
    from mortal.core.model import Brain, OracleDualTowerBrain, ValueHead
    cfg = value_training_cfg(config)
    arch = online_value_architecture(config)
    if arch['value_loss_mode'] != 'mse':
        raise ValueError('online value training currently requires the scalar MSE head')
    oracle = None
    if cfg.get('oracle_critic', True):
        kwargs = dict(version=config['control']['version'], **config['resnet'], Norm='GN')
        if normalize_oracle_critic_arch(cfg.get('oracle_critic_arch')) == 'dual_tower':
            oracle = OracleDualTowerBrain(
                **kwargs, oracle_fusion_mode=arch['oracle_fusion_mode'],
                oracle_fusion_hidden=arch['oracle_fusion_hidden'],
            )
        else:
            oracle = Brain(**kwargs, is_oracle=True)
        oracle = oracle.to(device)
    head = ValueHead(num_players=value_num_players_from_mode(value_target_mode(config)),
                     hidden_size=arch['value_head_hidden'], zero_sum=bool(cfg.get('exact_zero_sum', False)))
    return oracle, head.to(device)


def validate_oracle_critic_init_checkpoint(
    state,
    config,
    *,
    checkpoint_name='value.oracle_critic_state_file',
):
    if not isinstance(state, dict):
        raise TypeError(f'{checkpoint_name} must be a checkpoint dict')

    pretrain_cfg = state.get('oracle_critic_pretrain')
    if isinstance(pretrain_cfg, dict):
        required_fields = (
            'target_mode',
            'return_mode',
            'discount_gamma',
        )
        missing_fields = [
            field for field in required_fields
            if field not in pretrain_cfg
        ]
        if missing_fields:
            raise ValueError(
                f'{checkpoint_name} metadata missing required field(s): '
                f'{", ".join(missing_fields)}'
            )

        expected_target_mode = value_target_mode(config)
        actual_target_mode = normalize_value_target_mode(
            pretrain_cfg['target_mode'],
            oracle_critic=True,
        )
        if actual_target_mode != expected_target_mode:
            raise ValueError(
                f'{checkpoint_name} target mismatch: checkpoint target_mode='
                f'{actual_target_mode!r}, current value target_mode={expected_target_mode!r}'
            )

        expected_reward_source = value_reward_source(config)
        from mortal.data.oracle_value import normalize_oracle_return_mode

        actual_return_mode = normalize_oracle_return_mode(
            pretrain_cfg['return_mode']
        )
        if expected_reward_source != 'score_rank':
            raise ValueError(
                f'{checkpoint_name} reward mismatch: checkpoint return_mode='
                f'{actual_return_mode!r}, current value.reward_source='
                f'{expected_reward_source!r}; explicit Oracle critic checkpoints '
                'must match the online value target semantics'
            )
        if actual_return_mode != 'score_rank_mc':
            raise ValueError(
                f'{checkpoint_name} return mismatch: checkpoint return_mode='
                f'{actual_return_mode!r}, current value.reward_source='
                f'{expected_reward_source!r} expects score_rank_mc'
            )

        policy_cfg = config.get('policy', {})
        policy_cfg = policy_cfg if isinstance(policy_cfg, dict) else {}
        expected_gamma = float(policy_cfg.get('gae_gamma', 0.999))
        actual_gamma = float(pretrain_cfg['discount_gamma'])
        if abs(actual_gamma - expected_gamma) > 1e-6:
            raise ValueError(
                f'{checkpoint_name} discount mismatch: checkpoint discount_gamma='
                f'{actual_gamma:g}, current policy.gae_gamma={expected_gamma:g}'
            )

        value_cfg = value_training_cfg(config)
        expected_exact_zero_sum = bool(value_cfg.get('exact_zero_sum', False))
        actual_exact_zero_sum = bool(pretrain_cfg.get('exact_zero_sum', False))
        if actual_exact_zero_sum != expected_exact_zero_sum:
            raise ValueError(
                f'{checkpoint_name} zero-sum mismatch: checkpoint exact_zero_sum='
                f'{actual_exact_zero_sum}, current value.exact_zero_sum='
                f'{expected_exact_zero_sum}'
            )
        expected_arch = normalize_oracle_critic_arch(
            value_cfg.get('oracle_critic_arch', 'single_tower')
        )
        actual_arch = (
            normalize_oracle_critic_arch(pretrain_cfg['critic_arch'])
            if 'critic_arch' in pretrain_cfg
            else infer_oracle_critic_arch_from_state_dict(
                state.get('oracle_brain'),
                checkpoint_name=checkpoint_name,
            )
        )
        if actual_arch != expected_arch:
            raise ValueError(
                f'{checkpoint_name} arch mismatch: checkpoint critic_arch='
                f'{actual_arch!r}, current value.oracle_critic_arch={expected_arch!r}'
            )
        expected_layout = online_value_architecture(config)
        for field, expected in expected_layout.items():
            defaults = {'oracle_fusion_mode': 'linear', 'oracle_fusion_hidden': 512,
                        'value_head_hidden': 256, 'value_loss_mode': 'mse'}
            if pretrain_cfg.get(field, defaults[field]) != expected:
                raise ValueError(f'{checkpoint_name} {field} mismatch: '
                                 f'{pretrain_cfg.get(field, defaults[field])!r} != {expected!r}')
        if (pretrain_cfg.get('state_fold_backend') == 'native_hash'
            and int(pretrain_cfg.get('state_fold_count', 1)) > 1 and actual_gamma != 1.0
            and not state.get('training_contract', {}).get('target_clock_version')):
            raise ValueError(f'{checkpoint_name} used legacy folded target clocks; requalify under corrected targets')
        expected_points = centered_reward_signature(config)
        actual_points = centered_reward_signature(state.get('config') or {})
        if expected_points is not None and actual_points != expected_points:
            raise ValueError(
                f'{checkpoint_name} rank reward mismatch: checkpoint centered env.pts='
                f'{actual_points!r}, current centered env.pts={expected_points!r}'
            )
        return {
            'source': 'oracle_critic_pretrain',
            'target_mode': actual_target_mode,
            'return_mode': actual_return_mode,
            'discount_gamma': actual_gamma,
            'critic_arch': actual_arch,
        }

    saved_config = state.get('config')
    if isinstance(saved_config, dict):
        saved_signature = online_resume_model_signature(saved_config)
        current_signature = online_resume_model_signature(config)
        if saved_signature is not None and saved_signature == current_signature:
            return {'source': 'online_resume_signature'}
        raise ValueError(
            f'{checkpoint_name} has no oracle_critic_pretrain metadata and its '
            'saved online model signature does not match the current value setup'
        )

    raise ValueError(
        f'{checkpoint_name} has no oracle_critic_pretrain metadata; refusing to '
        'silently load a same-shape Oracle critic with unknown target semantics'
    )


def masked_optional_tensor(value, mask, cpu_mask):
    import torch as _torch

    if value is None:
        return None
    if isinstance(value, _torch.Tensor):
        return value[cpu_mask] if value.device.type == 'cpu' else value[mask]
    return value


def prepare_policy_advantage_and_value_target(advantage, v_target, *, device, gae_enabled):
    import torch as _torch

    advantage = advantage.to(dtype=_torch.float32, device=device)
    if gae_enabled:
        # A replay filter can leave one sample; unbiased std is undefined there.
        adv_std = advantage.std() if advantage.numel() > 1 else advantage.new_zeros(())
        adv_std = adv_std.clamp(min=1e-8)
        adv_mean = advantage.mean()
        normalized_advantage = (advantage - adv_mean) / adv_std
    else:
        normalized_advantage = advantage
    value_target = (
        v_target.to(dtype=_torch.float32, device=device)
        if v_target is not None
        else None
    )
    return advantage, normalized_advantage, value_target


def compute_policy_objective_loss(clip_loss, entropy, entropy_weight):
    if clip_loss.shape != entropy.shape:
        raise ValueError(
            'clip_loss and entropy must have identical shapes; '
            f'got {tuple(clip_loss.shape)} and {tuple(entropy.shape)}'
        )
    return -(clip_loss + entropy * float(entropy_weight)).mean()


ONLINE_CONTEXT_META_SPECS = {
    'at_turn': 0,
    'round_stage': 1,
    'is_dealer': 2,
    'is_all_last': 3,
    'self_rank': 4,
    'opp_riichi_count': 5,
    'up_gap_100': 6,
    'down_gap_100': 7,
}


def _dict_section(node, key):
    if not isinstance(node, dict):
        return {}
    child = node.get(key, {})
    return child if isinstance(child, dict) else {}


def compute_context_turn_weights(context_meta, weighting_cfg, *, device):
    return compute_turn_bucket_weights(
        context_meta[:, ONLINE_CONTEXT_META_SPECS['at_turn']],
        early_factor=weighting_cfg['early_factor'],
        mid_factor=weighting_cfg['mid_factor'],
        late_factor=weighting_cfg['late_factor'],
        early_max_turn=weighting_cfg['early_max_turn'],
        late_min_turn=weighting_cfg['late_min_turn'],
    ).to(device=device, non_blocking=True)


def compute_rank_aux_sample_weights(
    context_meta,
    *,
    device,
    base_weight,
    south_factor,
    all_last_factor,
    gap_focus_points,
    gap_close_bonus,
    max_weight,
    turn_weighting,
):
    import torch as _torch

    weights = _torch.full(
        (context_meta.shape[0],),
        float(base_weight),
        dtype=_torch.float32,
        device=device,
    )
    weights.mul_(compute_context_turn_weights(context_meta, turn_weighting, device=device))
    if south_factor > 0 and south_factor != 1.0:
        weights[context_meta[:, ONLINE_CONTEXT_META_SPECS['round_stage']] == 1] *= float(south_factor)
    if all_last_factor > 0 and all_last_factor != 1.0:
        weights[context_meta[:, ONLINE_CONTEXT_META_SPECS['is_all_last']].to(_torch.bool)] *= float(all_last_factor)
    if gap_focus_points > 0 and gap_close_bonus > 0:
        nearest_gap = _torch.minimum(
            context_meta[:, ONLINE_CONTEXT_META_SPECS['up_gap_100']],
            context_meta[:, ONLINE_CONTEXT_META_SPECS['down_gap_100']],
        ).to(_torch.float32)
        nearest_gap.mul_(100.0)
        closeness = (1.0 - nearest_gap / float(gap_focus_points)).clamp_(0.0, 1.0)
        weights.mul_(1.0 + float(gap_close_bonus) * closeness)
    if max_weight > 0:
        weights.clamp_(max=float(max_weight))
    return weights


def balanced_bce_per_sample_with_logits(logits, targets, eligible, *, focal_gamma=0.0):
    import torch as _torch
    from torch import nn as _nn

    loss = _nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        reduction='none',
    )
    if focal_gamma > 0:
        probs = logits.sigmoid()
        focal_factor = _torch.where(targets.to(_torch.bool), 1 - probs, probs)
        loss = loss * focal_factor.pow(float(focal_gamma))

    reduce_dims = tuple(range(1, loss.ndim))
    targets_bool = targets.to(_torch.bool)
    pos_mask = eligible & targets_bool
    neg_mask = eligible & ~targets_bool
    pos_weight = pos_mask.to(dtype=loss.dtype)
    neg_weight = neg_mask.to(dtype=loss.dtype)
    pos_count = pos_weight.sum(dim=reduce_dims)
    neg_count = neg_weight.sum(dim=reduce_dims)
    pos_present = (pos_count > 0).to(dtype=loss.dtype)
    neg_present = (neg_count > 0).to(dtype=loss.dtype)
    present_count = pos_present + neg_present
    pos_mean = (loss * pos_weight).sum(dim=reduce_dims) / pos_count.clamp_min(1.0)
    neg_mean = (loss * neg_weight).sum(dim=reduce_dims) / neg_count.clamp_min(1.0)
    per_sample = (pos_mean * pos_present + neg_mean * neg_present) / present_count.clamp_min(1.0)
    eligible_count = eligible.to(dtype=loss.dtype).sum(dim=reduce_dims)
    return _torch.where(eligible_count > 0, per_sample, _torch.zeros_like(per_sample))


def init_binary_metric_dict(*, device):
    import torch as _torch

    return {
        'correct': _torch.zeros((), dtype=_torch.int64, device=device),
        'count': _torch.zeros((), dtype=_torch.int64, device=device),
        'tp': _torch.zeros((), dtype=_torch.int64, device=device),
        'tn': _torch.zeros((), dtype=_torch.int64, device=device),
        'pos_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'neg_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'pred_pos_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'pos_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'neg_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
    }


def update_binary_metric(result, eligible, target_positive, pred_positive, positive_prob):
    import torch as _torch

    target_positive = target_positive.to(_torch.bool) & eligible
    target_negative = eligible & ~target_positive
    pred_positive = pred_positive.to(_torch.bool) & eligible
    pred_negative = eligible & ~pred_positive

    tp = (pred_positive & target_positive).to(_torch.int64).sum()
    tn = (pred_negative & target_negative).to(_torch.int64).sum()
    pos_count = target_positive.to(_torch.int64).sum()
    neg_count = target_negative.to(_torch.int64).sum()
    eligible_count = pos_count + neg_count

    result['count'] = eligible_count
    result['correct'] = tp + tn
    result['tp'] = tp
    result['tn'] = tn
    result['pos_count'] = pos_count
    result['neg_count'] = neg_count
    result['pred_pos_count'] = pred_positive.to(_torch.int64).sum()

    positive_prob = positive_prob.clamp(1e-6, 1 - 1e-6)
    result['pos_loss_sum'] = -positive_prob[target_positive].log().sum().to(_torch.float64)
    result['neg_loss_sum'] = -_torch.log1p(-positive_prob[target_negative]).sum().to(_torch.float64)


def merge_binary_metric(target, source):
    for key in ('correct', 'count', 'tp', 'tn', 'pos_count', 'neg_count', 'pred_pos_count'):
        target[key] += source[key]
    for key in ('pos_loss_sum', 'neg_loss_sum'):
        target[key] += source[key]


def finalize_binary_metric(prefix, stat, output):
    count = int(stat['count'].item())
    if count <= 0:
        return

    pos_count = int(stat['pos_count'].item())
    neg_count = int(stat['neg_count'].item())
    output[f'{prefix}_count'] = count
    output[f'{prefix}_pos_count'] = pos_count
    output[f'{prefix}_neg_count'] = neg_count
    output[f'{prefix}_acc'] = stat['correct'].item() / count
    output[f'{prefix}_pred_rate'] = stat['pred_pos_count'].item() / count
    output[f'{prefix}_target_rate'] = stat['pos_count'].item() / count

    balanced_acc_terms = []
    balanced_bce_terms = []
    if pos_count > 0:
        pos_recall = stat['tp'].item() / pos_count
        pos_bce = stat['pos_loss_sum'].item() / pos_count
        output[f'{prefix}_pos_recall'] = pos_recall
        output[f'{prefix}_pos_bce'] = pos_bce
        balanced_acc_terms.append(pos_recall)
        balanced_bce_terms.append(pos_bce)
    if neg_count > 0:
        neg_recall = stat['tn'].item() / neg_count
        neg_bce = stat['neg_loss_sum'].item() / neg_count
        output[f'{prefix}_neg_recall'] = neg_recall
        output[f'{prefix}_neg_bce'] = neg_bce
        balanced_acc_terms.append(neg_recall)
        balanced_bce_terms.append(neg_bce)
    if balanced_acc_terms:
        output[f'{prefix}_balanced_acc'] = sum(balanced_acc_terms) / len(balanced_acc_terms)
    if balanced_bce_terms:
        output[f'{prefix}_balanced_bce'] = sum(balanced_bce_terms) / len(balanced_bce_terms)


def init_online_aux_monitor_stats(*, device):
    import torch as _torch

    return {
        'rank_correct': _torch.zeros((), dtype=_torch.int64, device=device),
        'rank_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'rank_aux_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'rank_aux_raw_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'rank_aux_weight_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'opponent_sample_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'opponent_aux_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'opponent_turn_weight_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'opponent_shanten_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'opponent_tenpai_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'opponent_count': _torch.zeros((3,), dtype=_torch.int64, device=device),
        'opponent_shanten_correct': _torch.zeros((3,), dtype=_torch.int64, device=device),
        'opponent_tenpai_correct': _torch.zeros((3,), dtype=_torch.int64, device=device),
        'danger_sample_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'danger_aux_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'danger_turn_weight_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'danger_any_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'danger_value_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'danger_player_loss_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'danger_any_stats': init_binary_metric_dict(device=device),
        'danger_player_stats': init_binary_metric_dict(device=device),
        'danger_value_pos_count': _torch.zeros((), dtype=_torch.int64, device=device),
        'danger_value_abs_err_sum': _torch.zeros((), dtype=_torch.float64, device=device),
        'danger_value_sq_err_sum': _torch.zeros((), dtype=_torch.float64, device=device),
    }


def finalize_online_aux_monitor_stats(stats):
    finalized = {}

    rank_count = int(stats['rank_count'].item())
    if rank_count > 0:
        finalized['aux_loss'] = stats['rank_aux_loss_sum'].item() / rank_count
        finalized['rank_aux_raw_loss'] = stats['rank_aux_raw_loss_sum'].item() / rank_count
        finalized['rank_aux_weight_mean'] = stats['rank_aux_weight_sum'].item() / rank_count
        finalized['rank_acc'] = stats['rank_correct'].item() / rank_count

    opponent_sample_count = int(stats['opponent_sample_count'].item())
    if opponent_sample_count > 0:
        finalized['opponent_aux_loss'] = stats['opponent_aux_loss_sum'].item() / opponent_sample_count
        finalized['opponent_turn_weight_mean'] = stats['opponent_turn_weight_sum'].item() / opponent_sample_count
        finalized['opponent_shanten_loss'] = stats['opponent_shanten_loss_sum'].item() / opponent_sample_count
        finalized['opponent_tenpai_loss'] = stats['opponent_tenpai_loss_sum'].item() / opponent_sample_count

    opponent_count = stats['opponent_count'].detach().cpu()
    opponent_shanten_correct = stats['opponent_shanten_correct'].detach().cpu()
    opponent_tenpai_correct = stats['opponent_tenpai_correct'].detach().cpu()
    shanten_accs = []
    tenpai_accs = []
    for idx in range(3):
        count = opponent_count[idx].item()
        if count <= 0:
            continue
        shanten_acc = opponent_shanten_correct[idx].item() / count
        tenpai_acc = opponent_tenpai_correct[idx].item() / count
        finalized[f'opp{idx + 1}_shanten_acc'] = shanten_acc
        finalized[f'opp{idx + 1}_tenpai_acc'] = tenpai_acc
        shanten_accs.append(shanten_acc)
        tenpai_accs.append(tenpai_acc)
    if shanten_accs:
        finalized['opponent_shanten_macro_acc'] = sum(shanten_accs) / len(shanten_accs)
    if tenpai_accs:
        finalized['opponent_tenpai_macro_acc'] = sum(tenpai_accs) / len(tenpai_accs)

    danger_sample_count = int(stats['danger_sample_count'].item())
    if danger_sample_count > 0:
        finalized['danger_aux_loss'] = stats['danger_aux_loss_sum'].item() / danger_sample_count
        finalized['danger_turn_weight_mean'] = stats['danger_turn_weight_sum'].item() / danger_sample_count
        finalized['danger_any_loss'] = stats['danger_any_loss_sum'].item() / danger_sample_count
        finalized['danger_value_loss'] = stats['danger_value_loss_sum'].item() / danger_sample_count
        finalized['danger_player_loss'] = stats['danger_player_loss_sum'].item() / danger_sample_count

    danger_summary = {}
    finalize_binary_metric('danger_any', stats['danger_any_stats'], danger_summary)
    finalize_binary_metric('danger_player', stats['danger_player_stats'], danger_summary)
    value_count = int(stats['danger_value_pos_count'].item())
    if value_count > 0:
        danger_summary['danger_value_mae'] = stats['danger_value_abs_err_sum'].item() / value_count
        danger_summary['danger_value_rmse'] = _math.sqrt(
            stats['danger_value_sq_err_sum'].item() / value_count
        )
    finalized.update(danger_summary)
    return finalized


def init_online_stats(*, device):
    import torch as _torch

    def scalar():
        return _torch.zeros((), dtype=_torch.float32, device=device)

    return {
        'important_ratio': scalar(),
        'approx_kl': scalar(),
        'clip_fraction': scalar(),
        'ratio_var': scalar(),
        'ratio_batch_max_sum': scalar(),
        'ratio_window_max': scalar(),
        'clipped_ratio_window_max': scalar(),
        'entropy': scalar(),
        'policy_logit_gate_fraction': scalar(),
        'policy_update_active': scalar(),
        'loss': scalar(),
        'aux_loss': scalar(),
        'opp_loss': scalar(),
        'danger_loss': scalar(),
        'value_loss': scalar(),
        'exp_reward_loss': scalar(),
        'tile_eff_loss': scalar(),
        'furo_regret_loss': scalar(),
        'hand_value_regret_loss': scalar(),
        'importance_reject_fraction': scalar(),
        'policy_scope_reject_fraction': scalar(),
        'replay_is_coverage': scalar(),
        'replay_is_coverage_min': _torch.ones((), dtype=_torch.float32, device=device),
        'replay_is_missing_fraction': scalar(),
        'replay_is_missing_fraction_max': scalar(),
        'replay_is_version_gap': scalar(),
        'replay_is_version_gap_max': scalar(),
        'aux_monitor': init_online_aux_monitor_stats(device=device),
    }


def resolve_effective_online_aux_training_cfg(current_config, *, checkpoint_config=None):
    source_config = checkpoint_config if isinstance(checkpoint_config, dict) else current_config
    aux_cfg = _dict_section(source_config, 'aux')
    current_aux_cfg = _dict_section(current_config, 'aux')
    supervised_cfg = _dict_section(source_config, 'supervised')
    current_supervised_cfg = _dict_section(current_config, 'supervised')
    rank_aux_cfg = _dict_section(supervised_cfg, 'rank_aux')
    current_rank_aux_cfg = _dict_section(current_supervised_cfg, 'rank_aux')
    source_label = 'checkpoint' if isinstance(checkpoint_config, dict) else 'current_config'

    return {
        'source': source_label,
        'rank_base_weight': float(
            rank_aux_cfg.get(
                'base_weight',
                current_rank_aux_cfg.get('base_weight', current_aux_cfg.get('next_rank_weight', 0.0)),
            ) or 0.0
        ),
        'rank_south_factor': float(
            rank_aux_cfg.get(
                'south_factor',
                current_rank_aux_cfg.get('south_factor', 1.0),
            ) or 1.0
        ),
        'rank_all_last_factor': float(
            rank_aux_cfg.get(
                'all_last_factor',
                current_rank_aux_cfg.get('all_last_factor', 1.0),
            ) or 1.0
        ),
        'rank_gap_focus_points': float(
            rank_aux_cfg.get(
                'gap_focus_points',
                current_rank_aux_cfg.get('gap_focus_points', 0.0),
            ) or 0.0
        ),
        'rank_gap_close_bonus': float(
            rank_aux_cfg.get(
                'gap_close_bonus',
                current_rank_aux_cfg.get('gap_close_bonus', 0.0),
            ) or 0.0
        ),
        'rank_max_weight': float(
            rank_aux_cfg.get(
                'max_weight',
                current_rank_aux_cfg.get('max_weight', 0.0),
            ) or 0.0
        ),
        'rank_turn_weighting': resolve_turn_weighting_cfg(
            rank_aux_cfg.get('turn_weighting', current_rank_aux_cfg.get('turn_weighting', {})),
            default_early_factor=1.0,
            default_mid_factor=1.05,
            default_late_factor=1.15,
            default_early_max_turn=4,
            default_late_min_turn=12,
        ),
        'opponent_state_weight': float(aux_cfg.get('opponent_state_weight', current_aux_cfg.get('opponent_state_weight', 0.0)) or 0.0),
        'opponent_shanten_weight': float(aux_cfg.get('opponent_shanten_weight', current_aux_cfg.get('opponent_shanten_weight', 1.0)) or 1.0),
        'opponent_tenpai_weight': float(aux_cfg.get('opponent_tenpai_weight', current_aux_cfg.get('opponent_tenpai_weight', 1.0)) or 1.0),
        'opponent_turn_weighting': resolve_turn_weighting_cfg(
            aux_cfg.get('opponent_turn_weighting', current_aux_cfg.get('opponent_turn_weighting', {})),
            default_early_factor=0.20,
            default_mid_factor=1.0,
            default_late_factor=1.60,
            default_early_max_turn=4,
            default_late_min_turn=12,
        ),
        'danger_weight': float(aux_cfg.get('danger_weight', current_aux_cfg.get('danger_weight', 0.0)) or 0.0),
        'danger_any_weight': float(aux_cfg.get('danger_any_weight', current_aux_cfg.get('danger_any_weight', 0.09042179466099699)) or 0.0),
        'danger_value_weight': float(aux_cfg.get('danger_value_weight', current_aux_cfg.get('danger_value_weight', 0.8180402859274302)) or 0.0),
        'danger_player_weight': float(aux_cfg.get('danger_player_weight', current_aux_cfg.get('danger_player_weight', 0.09153791941157279)) or 0.0),
        'danger_focal_gamma': float(aux_cfg.get('danger_focal_gamma', current_aux_cfg.get('danger_focal_gamma', 0.0)) or 0.0),
        'danger_ramp_steps': int(aux_cfg.get('danger_ramp_steps', current_aux_cfg.get('danger_ramp_steps', 0)) or 0),
        'danger_value_cap': float(aux_cfg.get('danger_value_cap', current_aux_cfg.get('danger_value_cap', 96000.0)) or 96000.0),
        'danger_turn_weighting': resolve_turn_weighting_cfg(
            aux_cfg.get('danger_turn_weighting', current_aux_cfg.get('danger_turn_weighting', {})),
            default_early_factor=0.05,
            default_mid_factor=1.0,
            default_late_factor=2.50,
            default_early_max_turn=4,
            default_late_min_turn=12,
        ),
    }


class PublishedPolicyHistory:
    def __init__(self, max_versions):
        self.max_versions = max(int(max_versions or 1), 1)
        self._items = OrderedDict()

    def remember(self, version, *, mortal_state, policy_state, runtime):
        if version is None:
            return
        version = int(version)
        self._items[version] = {
            'mortal': mortal_state,
            'policy_net': policy_state,
            'runtime': dict(runtime) if isinstance(runtime, dict) else {},
        }
        self._items.move_to_end(version)
        while len(self._items) > self.max_versions:
            self._items.popitem(last=False)

    def get(self, version):
        if version is None:
            return None
        return self._items.get(int(version))

    def __len__(self):
        return len(self._items)

    def versions(self):
        return tuple(self._items.keys())


def tracked_replay_versions_mask(replay_versions, policy_history):
    import torch as _torch

    versions = replay_versions.to(dtype=_torch.int64, device='cpu')
    tracked = _torch.zeros_like(versions, dtype=_torch.bool)
    for version in versions.unique().tolist():
        if version >= 0 and policy_history.get(version) is not None:
            tracked |= versions.eq(version)
    return tracked


def compute_gae_advantages(kyoku_adv, at_kyoku, v_pred, gamma, lam):
    """Compute step-level GAE advantages from a complete game trajectory."""
    step_rewards = expand_sparse_kyoku_reward_to_steps(kyoku_adv, at_kyoku)
    return compute_gae_advantages_from_step_rewards(step_rewards, v_pred, gamma, lam)


def expand_sparse_kyoku_reward_to_steps(kyoku_adv, at_kyoku):
    n = len(at_kyoku)
    r = _np.zeros(n, dtype=_np.float32)
    kyoku_adv = _np.asarray(kyoku_adv, dtype=_np.float32)
    for t in range(n):
        k = int(at_kyoku[t])
        next_k = len(kyoku_adv) if t == n - 1 else int(at_kyoku[t + 1])
        if next_k < k:
            raise ValueError(f'at_kyoku must be non-decreasing, got transition {k}->{next_k}')
        if next_k != k and k < len(kyoku_adv):
            r[t] = float(kyoku_adv[k:min(next_k, len(kyoku_adv))].sum())
    return r


def compute_gae_advantages_from_step_rewards(step_rewards, v_pred, gamma, lam):
    n = len(step_rewards)
    gae = 0.0
    advantages = _np.zeros(n, dtype=_np.float32)
    for t in reversed(range(n)):
        v_next = float(v_pred[t + 1]) if t + 1 < n else 0.0
        delta = float(step_rewards[t]) + gamma * v_next - float(v_pred[t])
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


def compute_vtrace_targets_from_step_rewards(
    step_rewards,
    value_pred,
    log_rhos,
    gamma,
    *,
    rho_clip=0.0,
    c_clip=0.0,
):
    rewards = _np.asarray(step_rewards, dtype=_np.float32)
    values = _np.asarray(value_pred, dtype=_np.float32).reshape(-1)
    log_rhos = _np.asarray(log_rhos, dtype=_np.float32).reshape(-1)
    if rewards.shape[0] != values.shape[0] or rewards.shape[0] != log_rhos.shape[0]:
        raise ValueError('step_rewards, value_pred, and log_rhos must have the same length')

    rhos = _np.exp(log_rhos.astype(_np.float64)).astype(_np.float32)
    clipped_rhos = _np.minimum(rhos, float(rho_clip)) if rho_clip and rho_clip > 0 else rhos
    clipped_cs = _np.minimum(rhos, float(c_clip)) if c_clip and c_clip > 0 else rhos

    vs = _np.zeros_like(values, dtype=_np.float32)
    pg_advantages = _np.zeros_like(values, dtype=_np.float32)
    next_vs = 0.0
    bootstrap_value = 0.0

    for t in reversed(range(values.shape[0])):
        value_next = float(values[t + 1]) if t + 1 < values.shape[0] else bootstrap_value
        delta = clipped_rhos[t] * (
            float(rewards[t]) + gamma * value_next - float(values[t])
        )
        correction = gamma * clipped_cs[t] * (next_vs - value_next)
        vs[t] = float(values[t]) + delta + correction
        next_vs = float(vs[t])

    for t in range(values.shape[0]):
        next_vtrace = float(vs[t + 1]) if t + 1 < values.shape[0] else bootstrap_value
        pg_advantages[t] = clipped_rhos[t] * (
            float(rewards[t]) + gamma * next_vtrace - float(values[t])
        )

    return vs.astype(_np.float32), pg_advantages.astype(_np.float32)


def online_resume_model_signature(config):
    if not isinstance(config, dict):
        return None

    control_cfg = config.get('control', {})
    resnet_cfg = config.get('resnet', {})
    aux_cfg = config.get('aux', {})
    value_cfg = config.get('value', {})
    policy_cfg = config.get('policy', {})
    exp_reward_cfg = config.get('expected_reward', {})
    if control_cfg and not isinstance(control_cfg, dict):
        return None
    if resnet_cfg and not isinstance(resnet_cfg, dict):
        return None
    if aux_cfg and not isinstance(aux_cfg, dict):
        return None
    if policy_cfg and not isinstance(policy_cfg, dict):
        return None

    # `control.online` changes runtime behavior, not the model/optimizer layout
    # expected by train_online checkpoints, so it must not block exact resume.
    return {
        'version': control_cfg.get('version'),
        'centered_rank_rewards': centered_reward_signature(config),
        'actor_objective_contract': policy_cfg.get('actor_objective_contract', 'legacy'),
        'actor_objective': policy_cfg.get('actor_objective', 'legacy_hybrid'),
        'value_architecture': online_value_architecture(config),
        'resnet': dict(resnet_cfg),
        'oracle_experiment_arm': (
            config.get('oracle_experiments', {}).get('resolved_arm', 'current_config')
            if isinstance(config.get('oracle_experiments', {}), dict)
            else 'current_config'
        ),
        'actor_oracle_enabled': actor_oracle_guiding_enabled(config),
        'actor_oracle_source': actor_oracle_guiding_source(config),
        'policy_action_scope': policy_online_action_scope(config),
        'aux_enabled': float(aux_cfg.get('next_rank_weight', 0.0) or 0.0) > 0.0,
        'opp_enabled': float(aux_cfg.get('opponent_state_weight', 0.0) or 0.0) > 0.0,
        'danger_enabled': bool(aux_cfg.get('danger_enabled', False)) or float(aux_cfg.get('danger_weight', 0.0) or 0.0) > 0.0,
        'value_enabled': bool(value_cfg.get('enabled', False) if isinstance(value_cfg, dict) else False),
        'oracle_critic': bool(value_cfg.get('oracle_critic', True) if isinstance(value_cfg, dict) and value_cfg.get('enabled', False) else False),
        'oracle_critic_arch': (
            str(value_cfg.get('oracle_critic_arch', 'single_tower'))
            if isinstance(value_cfg, dict) and value_cfg.get('enabled', False)
            else 'single_tower'
        ),
        'independent_actor_lr_clock': value_independent_actor_lr_clock(config),
        'value_target_mode': value_target_mode(config),
        'value_reward_source': value_reward_source(config),
        'gae_enabled': bool(policy_cfg.get('gae_enabled', False))
        and bool(value_cfg.get('enabled', False) if isinstance(value_cfg, dict) else False),
        'gae_gamma': float(policy_cfg.get('gae_gamma', 0.999)),
        'gae_lambda': float(policy_cfg.get('gae_lambda', 0.95)),
        'tile_eff_enabled': float(aux_cfg.get('tile_efficiency_weight', 0.0) or 0.0) > 0.0,
        'furo_regret_enabled': float(aux_cfg.get('furo_regret_weight', 0.0) or 0.0) > 0.0,
        'hand_value_regret_enabled': float(aux_cfg.get('hand_value_regret_weight', 0.0) or 0.0) > 0.0,
        'exp_reward_enabled': bool(exp_reward_cfg.get('enabled', False) if isinstance(exp_reward_cfg, dict) else False),
    }


def checkpoint_matches_online_model_signature(state, *, current_config):
    if not isinstance(state, dict):
        return False
    saved_config = state.get('config', {})
    if saved_config and not isinstance(saved_config, dict):
        return False
    saved_signature = online_resume_model_signature(saved_config)
    current_signature = online_resume_model_signature(current_config)
    return (
        saved_signature is not None
        and current_signature is not None
        and saved_signature == current_signature
    )


def should_load_oracle_critic_init_checkpoint(
    *,
    state_file_exists,
    state_file_model_signature_matches,
):
    return not state_file_exists or not state_file_model_signature_matches


def optimizer_state_matches_current_layout(saved_optimizer_state, optimizer):
    if not isinstance(saved_optimizer_state, dict):
        return False

    saved_param_groups = saved_optimizer_state.get('param_groups')
    if not isinstance(saved_param_groups, list):
        return False
    if len(saved_param_groups) != len(optimizer.param_groups):
        return False

    for saved_group, current_group in zip(saved_param_groups, optimizer.param_groups):
        if not isinstance(saved_group, dict):
            return False
        saved_params = saved_group.get('params')
        current_params = current_group.get('params')
        if not isinstance(saved_params, list):
            return False
        if len(saved_params) != len(current_params):
            return False

    return True


def checkpoint_supports_online_resume(state, *, current_config, optimizer):
    if not isinstance(state, dict):
        return False
    if state.get('resume_supported') is False:
        return False
    saved_config = state.get('config', {})
    if saved_config and not isinstance(saved_config, dict):
        return False
    saved_control = saved_config.get('control', {})
    if saved_control and not isinstance(saved_control, dict):
        return False
    required_keys = ('optimizer', 'scheduler', 'scaler', 'best_perf', 'steps')
    if not all(key in state for key in required_keys):
        return False

    if not checkpoint_matches_online_model_signature(state, current_config=current_config):
        return False

    return optimizer_state_matches_current_layout(state['optimizer'], optimizer)


def _split_decay_params(model):
    from torch import nn as _nn

    params_dict = {}
    to_decay = set()
    for mod_name, mod in model.named_modules():
        for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
            params_dict[name] = param
            if isinstance(mod, (_nn.Linear, _nn.Conv1d)) and name.endswith('weight'):
                to_decay.add(name)
    decay = [params_dict[name] for name in sorted(to_decay)]
    no_decay = [params_dict[name] for name in sorted(params_dict.keys() - to_decay)]
    return decay, no_decay


def _append_optimizer_groups(
    param_groups,
    name,
    decay_params,
    no_decay_params,
    *,
    weight_decay,
    lr_scale=1.0,
    schedule_role=None,
):
    shared = {
        'lr': float(lr_scale),
        'lr_scale': float(lr_scale),
    }
    if schedule_role is not None:
        shared['schedule_role'] = str(schedule_role)
    if decay_params:
        param_groups.append({
            'name': f'{name}_decay',
            'params': decay_params,
            'weight_decay': weight_decay,
            **shared,
        })
    if no_decay_params:
        param_groups.append({
            'name': f'{name}_no_decay',
            'params': no_decay_params,
            **shared,
        })


def reconcile_loaded_scheduler_state(scheduler, optimizer, scheduler_cfg, *, steps):
    if not isinstance(scheduler_cfg, dict):
        return {}

    loaded_lrs = [float(group.get('lr', 0.0) or 0.0) for group in optimizer.param_groups]
    original_values = {}
    changed = {}
    int_keys = {'warm_up_steps', 'max_steps', 'offset', 'epoch_size'}
    for key in ('init', 'peak', 'final', 'warm_up_steps', 'max_steps', 'offset', 'epoch_size'):
        if key not in scheduler_cfg or not hasattr(scheduler, key):
            continue
        old_value = getattr(scheduler, key)
        raw_value = scheduler_cfg[key]
        new_value = int(raw_value) if key in int_keys else float(raw_value)
        if old_value != new_value:
            original_values[key] = old_value
            setattr(scheduler, key, new_value)
            changed[key] = (old_value, new_value)

    if not changed:
        return changed

    last_epoch = int(steps)
    scheduler.last_epoch = last_epoch
    if hasattr(scheduler, '_step_inner'):
        scale = float(scheduler._step_inner(last_epoch))
        base_lrs = list(getattr(scheduler, 'base_lrs', []))
        if len(base_lrs) != len(optimizer.param_groups):
            base_lrs = [group.get('initial_lr', 1.0) for group in optimizer.param_groups]
            scheduler.base_lrs = base_lrs
        lrs = [float(base_lr) * scale for base_lr in base_lrs]
        if any(new_lr > loaded_lr * (1.0 + 1e-12) for new_lr, loaded_lr in zip(lrs, loaded_lrs)):
            for key, old_value in original_values.items():
                setattr(scheduler, key, old_value)
            scheduler._last_lr = loaded_lrs
            changed['lr_increase_guard'] = {
                'loaded_lrs': loaded_lrs,
                'proposed_lrs': lrs,
                'ignored_scheduler_changes': {
                    key: value
                    for key, value in changed.items()
                    if key != 'lr_increase_guard'
                },
            }
            return changed
        for group, lr in zip(optimizer.param_groups, lrs):
            group['lr'] = lr
        scheduler._last_lr = lrs

    return changed


def resolve_online_init_state_file(config):
    if not isinstance(config, dict):
        return ''

    online_cfg = config.get('online', {})
    if isinstance(online_cfg, dict):
        init_state_file = str(online_cfg.get('init_state_file', '') or '').strip()
        if init_state_file:
            return init_state_file

    supervised_cfg = config.get('supervised', {})
    if not isinstance(supervised_cfg, dict):
        return ''

    return str(
        supervised_cfg.get('best_loss_state_file', '')
        or supervised_cfg.get('best_state_file', '')
        or ''
    ).strip()


def resolve_oracle_critic_init_state_file(config):
    if not isinstance(config, dict):
        return ''

    value_cfg = config.get('value', {})
    if not isinstance(value_cfg, dict):
        return ''
    for key in ('oracle_critic_state_file', 'critic_state_file', 'pretrained_state_file'):
        state_file = str(value_cfg.get(key, '') or '').strip()
        if state_file:
            return state_file

    pretrain_cfg = config.get('oracle_critic_pretrain', {})
    if isinstance(pretrain_cfg, dict):
        state_file = str(pretrain_cfg.get('best_state_file', '') or '').strip()
        if state_file:
            return state_file
    return ''


def ensure_online_init_state_file_ready(init_state_file):
    if not init_state_file:
        return

    from os import path
    import mortal.supervised.run_sl_formal as sl_formal

    sl_formal.ensure_supervised_canonical_handoff_ready(init_state_file)
    if not path.exists(init_state_file):
        raise FileNotFoundError(f'online.init_state_file does not exist: {init_state_file}')


def ensure_parent_dir_for_file(file_path):
    if not file_path:
        return

    import os

    parent = os.path.dirname(str(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def train():
    import mortal.core.prelude
    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    import math
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, TensorDataset
    from torch.distributions import Categorical
    from torch.utils.tensorboard import SummaryWriter
    from mortal.core.common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from mortal.eval.oracle_eval import (
        evaluate_oracle_dependency_modes,
        oracle_dependency_eval_enabled,
        oracle_dependency_eval_log_dir,
        oracle_dependency_eval_modes,
        summarize_stat,
        write_oracle_dependency_report,
    )
    from mortal.eval.player import TestPlayer
    from mortal.data.dataloader import FileDatasetsIter, worker_init_fn
    import numpy as np
    from collections import defaultdict
    from mortal.core.lr_scheduler import LinearWarmUpCosineAnnealingLR
    from mortal.core.model import (
        Brain,
        CategoricalPolicy,
        AuxNet,
        OracleDualTowerBrain,
        apply_oracle_obs_keep_prob,
    )
    from libriichi.consts import obs_shape, oracle_obs_shape
    from mortal.config import config
    from mortal.core.checkpoint_utils import (
        BRAIN_IS_ORACLE_KEY,
        FIRST_CONV_KEY,
        DEPLOY_ZERO_ORACLE_KEY,
        load_brain_state_strict,
        load_brain_state_with_input_bridge,
    )
    from copy import deepcopy
    from mortal.core.repro import apply_reproducibility, effective_cudnn_benchmark

    def load_oracle_critic_state(
        oracle_model,
        source_state,
        *,
        fallback_visible_state=None,
        strict=False,
        checkpoint_name='Oracle critic checkpoint',
    ):
        if oracle_model is None or source_state is None:
            return None
        if strict:
            return load_brain_state_strict(
                oracle_model,
                source_state,
                checkpoint_name=checkpoint_name,
            )
        try:
            oracle_model.load_state_dict(source_state)
            return {
                'loaded_keys': tuple(source_state.keys()),
                'skipped_keys': (),
                'expanded_input_keys': (),
                'extra_input_init_scale': 0.0,
                'direct': True,
                'strict': False,
            }
        except RuntimeError:
            pass
        if isinstance(oracle_model, OracleDualTowerBrain):
            visible_state = fallback_visible_state or source_state
            target_state = oracle_model.visible_encoder.state_dict()
            visible_channels = obs_shape(oracle_model.version)[0]
            with torch.no_grad():
                for key, target_tensor in target_state.items():
                    source_tensor = visible_state.get(f'encoder.{key}')
                    if source_tensor is not None and source_tensor.shape == target_tensor.shape:
                        target_tensor.copy_(source_tensor)
                oracle_model.visible_encoder.load_state_dict(target_state)

                oracle_state = oracle_model.oracle_encoder.state_dict()
                first_conv = visible_state.get(FIRST_CONV_KEY)
                for key, target_tensor in oracle_state.items():
                    source_tensor = visible_state.get(f'encoder.{key}')
                    if key == 'net.0.weight':
                        if torch.is_tensor(first_conv):
                            target_tensor.zero_()
                            out_channels = min(target_tensor.shape[0], first_conv.shape[0])
                            kernel = min(target_tensor.shape[2], first_conv.shape[2])
                            source_channels = first_conv.shape[1]
                            target_channels = target_tensor.shape[1]
                            oracle_start = visible_channels
                            oracle_end = oracle_start + target_channels
                            if source_channels >= oracle_end:
                                target_tensor[:out_channels, :, :kernel].copy_(
                                    first_conv[:out_channels, oracle_start:oracle_end, :kernel]
                                )
                            elif source_channels == target_channels:
                                target_tensor[:out_channels, :, :kernel].copy_(
                                    first_conv[:out_channels, :, :kernel]
                                )
                            else:
                                mean_weight = first_conv[:out_channels, :, :kernel].mean(dim=1, keepdim=True)
                                target_tensor[:out_channels, :, :kernel].copy_(
                                    mean_weight.expand(-1, target_channels, -1)
                                )
                                target_tensor.mul_(0.25)
                        continue
                    if source_tensor is not None and source_tensor.shape == target_tensor.shape:
                        target_tensor.copy_(source_tensor)
                oracle_model.oracle_encoder.load_state_dict(oracle_state)
            return {
                'direct': False,
                'strict': False,
                'bridge': 'dual_tower_visible_or_oracle_state',
            }
        try:
            return load_brain_state_with_input_bridge(oracle_model, source_state)
        except (KeyError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                'failed to load Oracle critic state; check value.oracle_critic_arch '
                'matches the checkpoint structure'
            ) from exc

    oracle_experiment_arm, oracle_artifact_suffix = apply_oracle_experiment_to_config(config)
    repro_runtime = apply_reproducibility(config, process_name='trainer')

    # --- Value Head / Oracle Critic imports ---
    value_cfg = value_training_cfg(config)
    value_enabled = value_cfg.get('enabled', False)
    critic_only = critic_only_enabled(config)
    successful_step_limit = successful_optimizer_step_limit(config)
    value_weight = value_cfg.get('weight', 0.5) if value_enabled else 0.0
    oracle_critic = value_cfg.get('oracle_critic', True) if value_enabled else False
    oracle_critic_arch = normalize_oracle_critic_arch(
        value_cfg.get('oracle_critic_arch', 'single_tower')
    )
    resolved_value_target_mode = value_target_mode(config) if value_enabled else 'current_player'
    resolved_value_reward_source = value_reward_source(config) if value_enabled else 'grp'
    value_num_players = value_num_players_from_mode(resolved_value_target_mode) if value_enabled else 1
    exact_zero_sum = bool(value_cfg.get('exact_zero_sum', False)) if value_enabled else False
    if exact_zero_sum and resolved_value_target_mode != 'all_players':
        raise ValueError('value.exact_zero_sum requires value.target_mode=all_players')
    zero_sum_weight = (
        value_cfg.get('zero_sum_weight', 0.01)
        if value_enabled and resolved_value_target_mode == 'all_players'
        else 0.0
    )
    critic_warmup_steps = value_critic_warmup_steps(config) if value_enabled else 0
    independent_actor_lr_clock = value_independent_actor_lr_clock(config)
    actor_oracle_enabled = actor_oracle_guiding_enabled(config)
    actor_oracle_source = actor_oracle_guiding_source(config)
    actor_oracle_lr_scale = actor_oracle_guiding_lr_scale(config)
    actor_oracle_importance_threshold = actor_oracle_guiding_importance_threshold(config)
    need_oracle_obs = bool(actor_oracle_enabled or oracle_critic)
    dependency_eval_enabled = oracle_dependency_eval_enabled(config)
    dependency_eval_modes = oracle_dependency_eval_modes(config)
    dependency_eval_log_dir = oracle_dependency_eval_log_dir(config)
    if oracle_experiment_arm.oracle_critic_enabled and not value_enabled:
        raise ValueError(
            f'Oracle experiment arm {oracle_experiment_arm.name!r} requires value.enabled=true'
        )

    # --- Local Regret Heads config ---
    tile_eff_weight = config.get('aux', {}).get('tile_efficiency_weight', 0.0)
    furo_regret_weight = config.get('aux', {}).get('furo_regret_weight', 0.0)
    hand_value_regret_weight = config.get('aux', {}).get('hand_value_regret_weight', 0.0)
    online_regret_enabled = (tile_eff_weight > 0 or furo_regret_weight > 0 or hand_value_regret_weight > 0)

    # --- Expected Reward Network config ---
    exp_reward_cfg = config.get('expected_reward', {})
    exp_reward_enabled = exp_reward_cfg.get('enabled', False)
    exp_reward_weight = exp_reward_cfg.get('weight', 0.1) if exp_reward_enabled else 0.0
    exp_reward_warmup = exp_reward_cfg.get('warmup_steps', 10000) if exp_reward_enabled else 0

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    old_update_every= config['control']['old_update_every']
    test_play_eval_enabled = test_play_enabled(config)
    test_games = test_play_games(config)
    initial_test_eval_enabled = initial_test_play_enabled(config)
    initial_test_games = initial_test_play_games(config)
    recorded_initial_baseline = recorded_step0_baseline(config)
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = effective_cudnn_benchmark(config)
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']
    if repro_runtime.enabled:
        logging.info(
            'repro mode active: process=%s base_seed=%s process_seed=%s train_key=%s train_seed_start=%s cudnn_benchmark=%s strict_cuda=%s',
            repro_runtime.process_name,
            repro_runtime.base_seed,
            repro_runtime.process_seed,
            repro_runtime.train_key,
            repro_runtime.train_seed_start,
            repro_runtime.cudnn_benchmark,
            repro_runtime.strict_cuda,
        )

    pts = config['env']['pts']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    prefetch_factor = config['dataset'].get('prefetch_factor', 2)
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    policy_cfg = policy_training_cfg(config)
    scheduler_cfg = config.get('optim', {}).get('scheduler', {})
    warm_up_steps = (
        max(int(scheduler_cfg.get('warm_up_steps', 0) or 0), 0)
        if isinstance(scheduler_cfg, dict)
        else 0
    )
    entropy_weight = policy_cfg['entropy_weight']
    entropy_floor = policy_entropy_floor(config)
    entropy_floor_start_step = policy_entropy_floor_start_step(
        config,
        default=warm_up_steps,
    )
    entropy_adjust_rate = float(policy_cfg.get('entropy_adjust_rate', 1e-4) or 0.0)
    clip_ratio = policy_cfg['clip_ratio']
    dual_clip = policy_cfg['dual_clip']
    actor_lr_scale = policy_actor_lr_scale(config)
    policy_lr_scale = policy_head_lr_scale(config)
    policy_interval = policy_update_interval(config)
    policy_phase = policy_update_phase(config) % policy_interval
    logit_gate_threshold = policy_logit_gate_threshold(config)
    importance_rho_clip = policy_importance_rho_clip(config)
    importance_c_clip = policy_importance_c_clip(config)
    vtrace_target_rho_clip = policy_vtrace_target_rho_clip(config)
    vtrace_target_c_clip = policy_vtrace_target_c_clip(config)
    policy_action_scope = policy_online_action_scope(config)
    # --- Step-Level GAE ---
    gae_enabled = config['policy'].get('gae_enabled', False) and value_enabled
    gae_gamma   = float(config['policy'].get('gae_gamma',  0.999))
    gae_lambda  = float(config['policy'].get('gae_lambda', 0.95))
    online_replay_is = online and replay_importance_sampling_enabled(config)
    replay_is_max_versions = replay_importance_sampling_max_versions(config)
    replay_is_drop_untracked = replay_importance_sampling_drop_untracked(config)
    replay_is_vtrace_mode = replay_importance_sampling_vtrace_mode(config)
    replay_is_vtrace_min_version_gap = replay_importance_sampling_vtrace_min_version_gap(config)
    aux_cfg = config.get('aux', {})
    next_rank_weight = aux_cfg.get('next_rank_weight', 0.0)
    next_rank_enabled = float(next_rank_weight or 0.0) > 0.0
    # --- Opponent State + Danger aux heads for online ---
    online_opponent_weight = aux_cfg.get('opponent_state_weight', 0.0)
    online_danger_weight = aux_cfg.get('danger_weight', 0.0)
    online_danger_enabled = bool(aux_cfg.get('danger_enabled', False)) or online_danger_weight > 0
    online_opp_enabled = online_opponent_weight > 0
    online_context_meta_enabled = bool(next_rank_enabled or online_opp_enabled or online_danger_enabled)
    effective_aux_training_cfg = resolve_effective_online_aux_training_cfg(config)
    danger_mix_weights = [
        effective_aux_training_cfg['danger_any_weight'],
        effective_aux_training_cfg['danger_value_weight'],
        effective_aux_training_cfg['danger_player_weight'],
    ]
    danger_value_cap = effective_aux_training_cfg['danger_value_cap']

    dynamic_entropy_weight = entropy_weight
    log_entropy_alpha = math.log(max(entropy_weight, 1e-8))
    vtrace_requested = bool(
        online
        and gae_enabled
        and value_enabled
        and (vtrace_target_rho_clip > 0 or vtrace_target_c_clip > 0)
    )
    vtrace_recursion_enabled = bool(
        vtrace_requested
        and (
            replay_is_vtrace_mode == 'always'
            or (replay_is_vtrace_mode == 'auto' and online_replay_is)
        )
    )
    actor_objective = validate_actor_objective(
        config, vtrace_enabled=vtrace_recursion_enabled,
        gae_enabled=gae_enabled, replay_is=online_replay_is,
    )
    # Missing v2 metadata must not be interpreted as an exact resume of the old actor loss.
    policy_cfg['actor_objective'] = actor_objective
    policy_cfg['actor_objective_contract'] = ACTOR_OBJECTIVE_VERSION
    if vtrace_recursion_enabled:
        if replay_is_vtrace_mode == 'always':
            logging.info(
                'true V-trace recursion enabled for online trajectory preprocessing: '
                'mode=always target_rho_clip=%.4f target_c_clip=%.4f gamma=%.4f; '
                'gae_lambda is not used in this path',
                vtrace_target_rho_clip,
                vtrace_target_c_clip,
                gae_gamma,
            )
        else:
            logging.info(
                'true V-trace recursion armed for stale replay trajectories only: '
                'mode=auto min_version_gap=%s target_rho_clip=%.4f target_c_clip=%.4f gamma=%.4f; '
                'fresh replay keeps plain GAE and gae_lambda, stale replay uses V-trace',
                replay_is_vtrace_min_version_gap,
                vtrace_target_rho_clip,
                vtrace_target_c_clip,
                gae_gamma,
            )
    elif vtrace_requested and replay_is_vtrace_mode == 'auto' and not online_replay_is:
        logging.info(
            'true V-trace recursion not armed: mode=auto requires '
            'online.importance_sampling.enabled=true; current run keeps plain GAE'
        )
    elif vtrace_requested and replay_is_vtrace_mode == 'disabled':
        logging.info(
            'true V-trace recursion disabled explicitly by '
            'online.importance_sampling.vtrace_mode=disabled'
        )
    elif vtrace_target_c_clip > 0:
        logging.warning(
            'policy.vtrace_target_c_clip/vtrace_c_clip=%.4f is set, but V-trace recursion '
            'only runs on the online value+GAE path; current run will not use c_clip',
            vtrace_target_c_clip,
        )
    if logit_gate_threshold > 0:
        logging.info(
            'policy logit gate enabled: threshold=%.4f (chosen-action gradient gate; rollout logits stay unchanged)',
            logit_gate_threshold,
        )
    if entropy_floor > 0 and entropy_adjust_rate > 0:
        logging.info(
            'entropy floor enabled: floor=%.4f start_step=%s adjust_rate=%.6f',
            entropy_floor,
            entropy_floor_start_step,
            entropy_adjust_rate,
        )
    if policy_interval > 1:
        logging.info(
            'policy update throttle enabled: interval=%s phase=%s',
            policy_interval,
            policy_phase,
        )
    if critic_warmup_steps > 0:
        if not value_enabled:
            raise ValueError('value.critic_warmup_steps requires value.enabled=true')
        logging.info(
            'Oracle critic actor-freeze warmup enabled: steps=%s (policy/aux heads frozen, value loss only)',
            critic_warmup_steps,
        )
    if independent_actor_lr_clock:
        logging.info(
            'independent actor LR clock enabled: critic warmup does not consume actor schedule'
        )

    mortal = Brain(version=version, is_oracle=actor_oracle_enabled, **config['resnet'], Norm="GN").to(device)
    policy_net = CategoricalPolicy().to(device)
    aux_net = AuxNet(dims=(4,)).to(device) if next_rank_enabled else None

    # --- Opponent State + Danger aux heads ---
    if online_opp_enabled:
        from mortal.core.model import OpponentStateAuxNet
        opponent_aux_net = OpponentStateAuxNet().to(device)
    else:
        opponent_aux_net = None

    if online_danger_enabled:
        from mortal.core.model import DangerAuxNet
        danger_aux_net = DangerAuxNet().to(device)
    else:
        danger_aux_net = None

    # --- Oracle Critic + Value Head ---
    if value_enabled:
        oracle_brain, value_net = build_online_value_models(config, device=device)
    else:
        oracle_brain = None
        value_net = None

    # --- Local Regret Heads ---
    if tile_eff_weight > 0:
        from mortal.core.model import TileEfficiencyRegretHead
        tile_eff_net = TileEfficiencyRegretHead().to(device)
    else:
        tile_eff_net = None

    if furo_regret_weight > 0:
        from mortal.core.model import FuroRegretHead
        furo_regret_net = FuroRegretHead().to(device)
    else:
        furo_regret_net = None

    if hand_value_regret_weight > 0:
        from mortal.core.model import HandValueRegretHead
        hand_value_regret_net = HandValueRegretHead().to(device)
    else:
        hand_value_regret_net = None

    # --- Expected Reward Network ---
    if exp_reward_enabled:
        from mortal.core.model import ExpectedRewardNet
        exp_reward_net = ExpectedRewardNet(num_players=value_num_players).to(device)
        grp_label_smoothing = config.get('grp', {}).get('label_smoothing', 0.0)
        if grp_label_smoothing > 0:
            logging.info(
                'note: both ExpectedRewardNet and label_smoothing (%.2f) are active; '
                'label_smoothing is only a historical cheap smoothing fallback, not an RVR-equivalent '
                'variance-reduction implementation; prefer validating ExpectedRewardNet separately and '
                'keeping label_smoothing=0 on the mainline',
                grp_label_smoothing,
            )
    else:
        exp_reward_net = None

    all_models_list = [mortal, policy_net]
    if aux_net is not None:
        all_models_list.append(aux_net)
    if opponent_aux_net is not None:
        all_models_list.append(opponent_aux_net)
    if danger_aux_net is not None:
        all_models_list.append(danger_aux_net)
    if oracle_brain is not None:
        all_models_list.append(oracle_brain)
    if value_net is not None:
        all_models_list.append(value_net)
    if tile_eff_net is not None:
        all_models_list.append(tile_eff_net)
    if furo_regret_net is not None:
        all_models_list.append(furo_regret_net)
    if hand_value_regret_net is not None:
        all_models_list.append(hand_value_regret_net)
    if exp_reward_net is not None:
        all_models_list.append(exp_reward_net)
    all_models = tuple(all_models_list)
    if enable_compile:
        for m in all_models:
            m.compile()

    if critic_only:
        restore_actor_training_mode(mortal, policy_net, critic_only=True)
    Old_mortal = deepcopy(mortal)
    Old_policy_net = deepcopy(policy_net)
    behavior_mortal = deepcopy(mortal).eval() if online_replay_is else None
    behavior_policy_net = deepcopy(policy_net).eval() if online_replay_is else None
    published_policy_history = PublishedPolicyHistory(replay_is_max_versions)
    published_param_version = -1
    loaded_behavior_version = None

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy_net):,}')
    if aux_net is not None:
        logging.info(f'aux params: {parameter_count(aux_net):,}')
    if opponent_aux_net is not None:
        logging.info(f'opponent_aux params: {parameter_count(opponent_aux_net):,}')
    if danger_aux_net is not None:
        logging.info(f'danger_aux params: {parameter_count(danger_aux_net):,}')
    if oracle_brain is not None:
        logging.info(f'oracle_brain params: {parameter_count(oracle_brain):,}')
    if value_net is not None:
        logging.info(f'value_net params: {parameter_count(value_net):,}')
    if tile_eff_net is not None:
        logging.info(f'tile_eff_net params: {parameter_count(tile_eff_net):,}')
    if furo_regret_net is not None:
        logging.info(f'furo_regret_net params: {parameter_count(furo_regret_net):,}')
    if exp_reward_net is not None:
        logging.info(f'exp_reward_net params: {parameter_count(exp_reward_net):,}')

    use_policy_lr_scales = actor_lr_scale != 1.0 or policy_lr_scale != 1.0
    use_named_optimizer_groups = use_policy_lr_scales or independent_actor_lr_clock
    param_groups = []
    if use_named_optimizer_groups:
        logging.info(
            'named optimizer groups enabled: actor_lr_scale=%.4g policy_head_lr_scale=%.4g independent_actor_lr_clock=%s',
            actor_lr_scale,
            policy_lr_scale,
            independent_actor_lr_clock,
        )
        for group_name, model, lr_scale in (
            ('actor', mortal, actor_lr_scale),
            ('policy_head', policy_net, policy_lr_scale),
        ):
            decay_params, no_decay_params = _split_decay_params(model)
            _append_optimizer_groups(
                param_groups,
                group_name,
                decay_params,
                no_decay_params,
                weight_decay=weight_decay,
                lr_scale=lr_scale,
                schedule_role='actor',
            )
        models_for_optim = []
    else:
        decay_params = []
        no_decay_params = []
        models_for_optim = [mortal, policy_net]
    if aux_net is not None:
        models_for_optim.append(aux_net)
    if opponent_aux_net is not None:
        models_for_optim.append(opponent_aux_net)
    if danger_aux_net is not None:
        models_for_optim.append(danger_aux_net)
    if oracle_brain is not None:
        models_for_optim.append(oracle_brain)
    if value_net is not None:
        models_for_optim.append(value_net)
    if tile_eff_net is not None:
        models_for_optim.append(tile_eff_net)
    if furo_regret_net is not None:
        models_for_optim.append(furo_regret_net)
    if hand_value_regret_net is not None:
        models_for_optim.append(hand_value_regret_net)
    if exp_reward_net is not None:
        models_for_optim.append(exp_reward_net)
    if use_named_optimizer_groups:
        critic_models = tuple(
            model
            for model in (oracle_brain, value_net, exp_reward_net)
            if model is not None
        )
        actor_other_idx = 0
        critic_idx = 0
        for model in models_for_optim:
            is_critic_model = any(model is candidate for candidate in critic_models)
            if is_critic_model:
                group_name = f'critic_{critic_idx}'
                critic_idx += 1
                schedule_role = 'critic'
            else:
                group_name = f'actor_aux_{actor_other_idx}'
                actor_other_idx += 1
                schedule_role = 'actor'
            decay_params, no_decay_params = _split_decay_params(model)
            _append_optimizer_groups(
                param_groups,
                group_name,
                decay_params,
                no_decay_params,
                weight_decay=weight_decay,
                schedule_role=schedule_role,
            )
    else:
        for model in models_for_optim:
            model_decay_params, model_no_decay_params = _split_decay_params(model)
            decay_params.extend(model_decay_params)
            no_decay_params.extend(model_no_decay_params)
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params},
        ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps, fused=False)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    apply_independent_actor_lr_clock(optimizer, scheduler, config, steps=0)
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    update_clock = OptimizerUpdateClock()
    state_file = config['control']['state_file']
    init_state_file = resolve_online_init_state_file(config)
    oracle_critic_init_state_file = (
        resolve_oracle_critic_init_state_file(config)
        if value_enabled and oracle_critic
        else ''
    )
    best_state_file = config['control']['best_state_file']
    loaded_config_for_aux_alignment = None
    reward_target_metadata = {
        'mode': 'raw_delta_pt',
        'mean': 0.0,
        'variance': 1.0,
        'std_dev': 1.0,
        'count': 0,
    }
    state_file_exists = path.exists(state_file)
    state_file_model_signature_matches = False
    if state_file_exists:
        state = torch.load(state_file, weights_only=False, map_location=device)
        loaded_config_for_aux_alignment = state.get('config') if isinstance(state.get('config'), dict) else None
        state_file_model_signature_matches = checkpoint_matches_online_model_signature(
            state,
            current_config=config,
        )
        if (
            value_enabled
            and oracle_critic
            and not state_file_model_signature_matches
            and not oracle_critic_init_state_file
        ):
            raise ValueError(
                f'{state_file} has Oracle/value weights but its saved value / GAE '
                'semantics do not match the current config, and no explicit '
                'value.oracle_critic_state_file was provided'
            )
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        stored_reward_target_metadata = state.get('reward_target_metadata')
        if isinstance(stored_reward_target_metadata, dict):
            reward_target_metadata.update(stored_reward_target_metadata)
        bridge_info = load_brain_state_with_input_bridge(mortal, state['mortal'])
        Old_mortal.load_state_dict(mortal.state_dict())
        policy_net.load_state_dict(state['policy_net'])
        Old_policy_net.load_state_dict(state['policy_net'])
        if aux_net is not None and 'aux_net' in state:
            aux_net.load_state_dict(state['aux_net'])
        if opponent_aux_net is not None and 'opponent_aux_net' in state:
            opponent_aux_net.load_state_dict(state['opponent_aux_net'])
        if danger_aux_net is not None and 'danger_aux_net' in state:
            danger_aux_net.load_state_dict(state['danger_aux_net'])
        if (
            oracle_brain is not None
            and 'oracle_brain' in state
            and state_file_model_signature_matches
        ):
            load_oracle_critic_state(
                oracle_brain,
                state['oracle_brain'],
                fallback_visible_state=state.get('mortal'),
            )
        elif oracle_brain is not None and 'oracle_brain' in state:
            logging.info(
                'skipping Oracle critic weights from mismatched online state_file; '
                'expecting explicit value.oracle_critic_state_file initialization'
            )
        if (
            value_net is not None
            and 'value_net' in state
            and state_file_model_signature_matches
        ):
            value_net.load_state_dict(state['value_net'])
        elif value_net is not None and 'value_net' in state:
            logging.info(
                'skipping value_net weights from mismatched online state_file; '
                'expecting explicit value.oracle_critic_state_file initialization'
            )
        if tile_eff_net is not None and 'tile_eff_net' in state:
            tile_eff_net.load_state_dict(state['tile_eff_net'])
        if furo_regret_net is not None and 'furo_regret_net' in state:
            furo_regret_net.load_state_dict(state['furo_regret_net'])
        if hand_value_regret_net is not None and 'hand_value_regret_net' in state:
            hand_value_regret_net.load_state_dict(state['hand_value_regret_net'])
        if exp_reward_net is not None and 'exp_reward_net' in state:
            exp_reward_net.load_state_dict(state['exp_reward_net'])
        if checkpoint_supports_online_resume(state, current_config=config, optimizer=optimizer):
            validate_calibration_resume(state, config)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scheduler_changes = reconcile_loaded_scheduler_state(
                scheduler,
                optimizer,
                config['optim'].get('scheduler', {}),
                steps=int(state.get('steps', 0) or 0),
            )
            if state.get('shared_stats') is not None:
                logging.info(
                    'ignoring legacy online reward standardization stats from checkpoint; '
                    'reward targets now stay on fixed raw delta_pt scale'
                )
            scaler.load_state_dict(state['scaler'])
            best_perf = state['best_perf']
            steps = state['steps']
            update_clock = OptimizerUpdateClock.from_checkpoint(
                state, opt_step_every=opt_step_every,
            )
            if 'dynamic_entropy_weight' in state:
                dynamic_entropy_weight = state['dynamic_entropy_weight']
            if 'log_entropy_alpha' in state:
                log_entropy_alpha = state['log_entropy_alpha']
            else:
                log_entropy_alpha = math.log(max(dynamic_entropy_weight, 1e-8))
            if scheduler_changes:
                logging.info(
                    'reconciled scheduler state with current config: %s',
                    scheduler_changes,
                )
            logging.info('resumed optimizer/scheduler state from checkpoint')
        else:
            if state_file_model_signature_matches and 'steps' in state:
                steps = int(state.get('steps', 0) or 0)
                update_clock = OptimizerUpdateClock.from_checkpoint(
                    state, opt_step_every=opt_step_every, weights_only=True,
                )
            logging.info(
                'initialized training from checkpoint weights only; '
                'optimizer/scheduler/scaler/best_perf were reset; '
                'steps=%s optimizer_progress=%s '
                '(brain bridge loaded=%s skipped=%s)',
                steps,
                update_clock.progress,
                len(bridge_info['loaded_keys']),
                len(bridge_info['skipped_keys']),
            )
    elif init_state_file:
        ensure_online_init_state_file_ready(init_state_file)
        state = torch.load(init_state_file, weights_only=False, map_location=device)
        loaded_config_for_aux_alignment = state.get('config') if isinstance(state.get('config'), dict) else None
        bridge_info = load_brain_state_with_input_bridge(mortal, state['mortal'])
        Old_mortal.load_state_dict(mortal.state_dict())
        policy_net.load_state_dict(state['policy_net'])
        Old_policy_net.load_state_dict(state['policy_net'])
        if aux_net is not None and state.get('aux_net') is not None:
            aux_net.load_state_dict(state['aux_net'])
        if opponent_aux_net is not None and state.get('opponent_aux_net') is not None:
            opponent_aux_net.load_state_dict(state['opponent_aux_net'])
        if danger_aux_net is not None and state.get('danger_aux_net') is not None:
            danger_aux_net.load_state_dict(state['danger_aux_net'])
        if value_net is not None and state.get('value_net') is not None:
            value_net.load_state_dict(state['value_net'])
        if tile_eff_net is not None and state.get('tile_eff_net') is not None:
            tile_eff_net.load_state_dict(state['tile_eff_net'])
        if furo_regret_net is not None and state.get('furo_regret_net') is not None:
            furo_regret_net.load_state_dict(state['furo_regret_net'])
        if hand_value_regret_net is not None and state.get('hand_value_regret_net') is not None:
            hand_value_regret_net.load_state_dict(state['hand_value_regret_net'])
        if exp_reward_net is not None and state.get('exp_reward_net') is not None:
            exp_reward_net.load_state_dict(state['exp_reward_net'])
        if oracle_brain is not None:
            if state.get('oracle_brain') is not None:
                load_oracle_critic_state(
                    oracle_brain,
                    state['oracle_brain'],
                    fallback_visible_state=state.get('mortal'),
                )
            else:
                # Initialize oracle brain from supervised brain weights via bridge
                load_oracle_critic_state(oracle_brain, state['mortal'])
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(
            'initialized online weights from supervised checkpoint: %s (%s); '
            'brain bridge loaded=%s skipped=%s',
            init_state_file,
            timestamp,
            len(bridge_info['loaded_keys']),
            len(bridge_info['skipped_keys']),
        )
    if oracle_critic_init_state_file and oracle_brain is not None and value_net is not None:
        if not should_load_oracle_critic_init_checkpoint(
            state_file_exists=state_file_exists,
            state_file_model_signature_matches=state_file_model_signature_matches,
        ):
            oracle_critic_init_state_file = ''
        elif not path.exists(oracle_critic_init_state_file):
            raise FileNotFoundError(
                f'value.oracle_critic_state_file does not exist: {oracle_critic_init_state_file}'
            )
        else:
            oracle_state = torch.load(oracle_critic_init_state_file, weights_only=False, map_location=device)
            if oracle_state.get('oracle_brain') is None or oracle_state.get('value_net') is None:
                raise ValueError(
                    f'oracle critic init checkpoint must contain oracle_brain and value_net: '
                    f'{oracle_critic_init_state_file}'
                )
            init_metadata = validate_oracle_critic_init_checkpoint(
                oracle_state,
                config,
                checkpoint_name='value.oracle_critic_state_file',
            )
            load_oracle_critic_state(
                oracle_brain,
                oracle_state['oracle_brain'],
                fallback_visible_state=oracle_state.get('mortal'),
                strict=True,
                checkpoint_name='value.oracle_critic_state_file',
            )
            value_net.load_state_dict(oracle_state['value_net'])
            logging.info(
                'initialized Oracle critic/value head from pretrain checkpoint: %s (%s)',
                oracle_critic_init_state_file,
                init_metadata,
            )

    logging.info('optimizer update clock (exact counts exclude legacy/inherited offsets): %s', update_clock.state_dict())
    apply_independent_actor_lr_clock(optimizer, scheduler, config, steps=steps)
    effective_aux_training_cfg = resolve_effective_online_aux_training_cfg(
        config,
        checkpoint_config=loaded_config_for_aux_alignment,
    )
    opponent_shanten_weight = effective_aux_training_cfg['opponent_shanten_weight']
    opponent_tenpai_weight = effective_aux_training_cfg['opponent_tenpai_weight']
    danger_value_cap = effective_aux_training_cfg['danger_value_cap']
    danger_mix_weights = [
        effective_aux_training_cfg['danger_any_weight'],
        effective_aux_training_cfg['danger_value_weight'],
        effective_aux_training_cfg['danger_player_weight'],
    ]
    danger_mix_total = sum(max(float(weight), 0.0) for weight in danger_mix_weights)
    if danger_mix_total <= 0:
        danger_mix_weights = [0.09042179466099699, 0.8180402859274302, 0.09153791941157279]
        danger_mix_total = sum(danger_mix_weights)
    danger_mix_weights = [float(weight) / danger_mix_total for weight in danger_mix_weights]
    logging.info(
        'effective online aux alignment source=%s rank(base=%.6f max=%.6f south=%.3f all_last=%.3f gap_focus=%.0f gap_bonus=%.3f) '
        'opp(weight=%.6f shanten=%.6f tenpai=%.6f) '
        'danger(weight=%.6f mix=[%.6f, %.6f, %.6f] ramp=%s focal=%.3f)',
        effective_aux_training_cfg['source'],
        effective_aux_training_cfg['rank_base_weight'],
        effective_aux_training_cfg['rank_max_weight'],
        effective_aux_training_cfg['rank_south_factor'],
        effective_aux_training_cfg['rank_all_last_factor'],
        effective_aux_training_cfg['rank_gap_focus_points'],
        effective_aux_training_cfg['rank_gap_close_bonus'],
        effective_aux_training_cfg['opponent_state_weight'],
        effective_aux_training_cfg['opponent_shanten_weight'],
        effective_aux_training_cfg['opponent_tenpai_weight'],
        effective_aux_training_cfg['danger_weight'],
        danger_mix_weights[0],
        danger_mix_weights[1],
        danger_mix_weights[2],
        effective_aux_training_cfg['danger_ramp_steps'],
        effective_aux_training_cfg['danger_focal_gamma'],
    )

    optimizer.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')
    logging.info(
        'oracle experiment arm=%s actor_enabled=%s actor_source=%s oracle_critic=%s value_target_mode=%s value_reward_source=%s action_scope=%s artifact_suffix=%s',
        oracle_experiment_arm.name,
        actor_oracle_enabled,
        actor_oracle_source,
        oracle_critic,
        resolved_value_target_mode,
        resolved_value_reward_source,
        policy_action_scope,
        oracle_artifact_suffix or '(none)',
    )
    if online and online_stop_at_max_steps(config):
        logging.info('online training will stop automatically at steps=%s', online_scheduler_max_steps(config))
    if online_reached_max_steps(config, steps):
        logging.info('configured online max steps already reached at steps=%s; stopping', steps)
        sys.exit(ONLINE_MAX_STEPS_EXIT_CODE)

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    writer.add_text('oracle_experiment/arm', oracle_experiment_arm.name, steps)
    writer.add_text('oracle_experiment/description', oracle_experiment_arm.description, steps)
    writer.add_text('oracle_experiment/actor_source', actor_oracle_source, steps)
    writer.add_text('policy/action_scope', policy_action_scope, steps)
    if recorded_initial_baseline is not None:
        writer.add_text(
            'test_play/recorded_step0_baseline',
            (
                f"games={recorded_initial_baseline.get('games')}, "
                f"avg_rank={recorded_initial_baseline.get('avg_rank')}, "
                f"avg_pt={recorded_initial_baseline.get('avg_pt')}, "
                f"source_run={recorded_initial_baseline.get('source_run')}"
            ),
            steps,
        )
    stats = init_online_stats(device=device)
    idx = 0

    def clone_state_dict_to_cpu(module):
        return OrderedDict(
            (key, value.detach().to(device='cpu').clone())
            for key, value in module.state_dict().items()
        )

    def remember_published_policy(param_version, runtime):
        nonlocal published_param_version
        if not online_replay_is or param_version is None:
            return
        published_param_version = int(param_version)
        published_policy_history.remember(
            published_param_version,
            mortal_state=clone_state_dict_to_cpu(mortal),
            policy_state=clone_state_dict_to_cpu(policy_net),
            runtime=runtime,
        )

    def build_published_aux_payload():
        payload = {}
        optional_modules = {
            'opponent_aux_net': opponent_aux_net,
            'danger_aux_net': danger_aux_net,
            'tile_eff_net': tile_eff_net,
            'furo_regret_net': furo_regret_net,
            'hand_value_regret_net': hand_value_regret_net,
            'oracle_brain': oracle_brain,
            'value_net': value_net,
            'exp_reward_net': exp_reward_net,
        }
        for key, module in optional_modules.items():
            if module is not None:
                payload[key] = clone_state_dict_to_cpu(module)
        search_cfg = config.get('search', {})
        if isinstance(search_cfg, dict) and payload:
            payload['search_cfg'] = dict(search_cfg)
        return payload if payload else None

    def publish_current_policy(*, is_idle):
        runtime = actor_oracle_guiding_runtime_state(config, steps)
        param_version = submit_param(
            mortal,
            policy_net,
            is_idle=is_idle,
            runtime=runtime,
            aux_payload=build_published_aux_payload(),
        )
        remember_published_policy(param_version, runtime)
        return param_version

    if online:
        published_version = publish_current_policy(is_idle=True)
        logging.info('param has been submitted: version=%s', published_version)

    def build_zero_oracle_export_state(live_state):
        export_state = dict(live_state)
        export_state.pop('optimizer', None)
        export_state.pop('scheduler', None)
        export_state.pop('scaler', None)
        export_state['resume_supported'] = False
        export_state['exported_from_actor_oracle'] = True
        export_state['exported_actor_oracle_keep_prob'] = actor_oracle_guiding_keep_prob(config, steps)
        export_state[BRAIN_IS_ORACLE_KEY] = bool(mortal.is_oracle)
        export_state[DEPLOY_ZERO_ORACLE_KEY] = True
        export_state['oracle_guiding_runtime'] = {
            'oracle_experiment_arm': str(
                config.get('oracle_experiments', {}).get('resolved_arm', 'current_config')
                if isinstance(config.get('oracle_experiments', {}), dict)
                else 'current_config'
            ),
            'actor_oracle_enabled': False,
            'actor_oracle_keep_prob': 0.0,
            'actor_oracle_continuation': True,
            'actor_oracle_source': 'zero',
            'oracle_critic_enabled': oracle_critic,
        }
        export_config = deepcopy(export_state.get('config', config))
        oracle_cfg = export_config.get('oracle_guiding')
        if isinstance(oracle_cfg, dict):
            oracle_cfg['actor_enabled'] = False
            oracle_cfg['actor_source'] = 'zero'
        export_state['config'] = export_config
        return export_state

    def prepare_eval_mortal():
        mortal.eval()
        return mortal

    def restore_training_mode():
        restore_actor_training_mode(mortal, policy_net, critic_only=critic_only)
        if aux_net is not None:
            aux_net.train()
        if opponent_aux_net is not None:
            opponent_aux_net.train()
        if danger_aux_net is not None:
            danger_aux_net.train()
        if oracle_brain is not None:
            oracle_brain.train()
        if value_net is not None:
            value_net.train()

    def run_test_play_evaluation(*, stats_dict=None, save_best_checkpoint=False, eval_games=None):
        target_games = int(test_games if eval_games is None else eval_games)
        eval_mortal = prepare_eval_mortal()
        stat = test_player.test_play(target_games // 4, eval_mortal, policy_net, device)
        restore_training_mode()

        avg_pt = stat.avg_pt([90, 45, 0, -135])  # for display only, never used in training
        better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
        if better:
            past_best = best_perf.copy()
            best_perf['avg_pt'] = avg_pt
            best_perf['avg_rank'] = stat.avg_rank
        else:
            past_best = None

        logging.info(f'avg rank: {stat.avg_rank:.6}')
        logging.info(f'avg pt: {avg_pt:.6}')
        writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
        writer.add_scalar('test_play/avg_pt', avg_pt, steps)
        writer.add_scalars('test_play/ranking', {
            '1st': stat.rank_1_rate,
            '2nd': stat.rank_2_rate,
            '3rd': stat.rank_3_rate,
            '4th': stat.rank_4_rate,
        }, steps)
        writer.add_scalars('test_play/behavior', {
            'agari': stat.agari_rate,
            'houjuu': stat.houjuu_rate,
            'fuuro': stat.fuuro_rate,
            'riichi': stat.riichi_rate,
        }, steps)
        writer.add_scalars('test_play/agari_point', {
            'overall': stat.avg_point_per_agari,
            'riichi': stat.avg_point_per_riichi_agari,
            'fuuro': stat.avg_point_per_fuuro_agari,
            'dama': stat.avg_point_per_dama_agari,
        }, steps)
        writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
        writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
        writer.add_scalars('test_play/key_step', {
            'agari_jun': stat.avg_agari_jun,
            'houjuu_jun': stat.avg_houjuu_jun,
            'riichi_jun': stat.avg_riichi_jun,
        }, steps)
        writer.add_scalars('test_play/riichi', {
            'agari_after_riichi': stat.agari_rate_after_riichi,
            'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
            'chasing_riichi': stat.chasing_riichi_rate,
            'riichi_chased': stat.riichi_chased_rate,
        }, steps)
        writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
        writer.add_scalars('test_play/fuuro', {
            'agari_after_fuuro': stat.agari_rate_after_fuuro,
            'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
        }, steps)
        writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
        writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
        if dependency_eval_enabled:
            zero_summary = summarize_stat(stat)
            dependency_results = evaluate_oracle_dependency_modes(
                test_player,
                eval_mortal,
                policy_net,
                device,
                seed_count=target_games // 4,
                modes=dependency_eval_modes,
                search_runtime_bundle=None,
                precomputed_zero=zero_summary,
            )
            writer.add_scalar(
                'oracle_dependency/checkpoint_brain_is_oracle',
                float(dependency_results['checkpoint_brain_is_oracle']),
                steps,
            )
            for mode_name, summary in dependency_results['modes'].items():
                writer.add_scalar(
                    f'oracle_dependency/{mode_name}/avg_rank',
                    summary['avg_rank'],
                    steps,
                )
                writer.add_scalar(
                    f'oracle_dependency/{mode_name}/avg_pt',
                    summary['avg_pt'],
                    steps,
                )
                writer.add_scalar(
                    f'oracle_dependency/{mode_name}/delta_vs_zero_avg_rank',
                    summary['delta_vs_zero_avg_rank'],
                    steps,
                )
                writer.add_scalar(
                    f'oracle_dependency/{mode_name}/delta_vs_zero_avg_pt',
                    summary['delta_vs_zero_avg_pt'],
                    steps,
                )
                writer.add_scalar(
                    f'oracle_dependency/{mode_name}/agari_rate',
                    summary['agari_rate'],
                    steps,
                )
                writer.add_scalar(
                    f'oracle_dependency/{mode_name}/houjuu_rate',
                    summary['houjuu_rate'],
                    steps,
                )
            if dependency_eval_log_dir:
                write_oracle_dependency_report(
                    path.join(dependency_eval_log_dir, f'step_{steps:08d}.json'),
                    {
                        'steps': int(steps),
                        'oracle_experiment_arm': oracle_experiment_arm.name,
                        'actor_oracle_source': actor_oracle_source,
                        'results': dependency_results,
                    },
                )
        writer.flush()

        if better and save_best_checkpoint:
            state = persist_live_training_state(
                reward_target_metadata_dict=stats_dict,
            )
            logging.info(
                'a new record has been made, '
                f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                f'saving to {best_state_file}'
            )
            ensure_parent_dir_for_file(best_state_file)
            if actor_oracle_enabled:
                atomic_torch_save(build_zero_oracle_export_state(state), best_state_file)
            else:
                atomic_torch_save(state, best_state_file)
        return stat

    def build_live_training_state(reward_target_metadata_dict=None):
        return {
            'mortal': mortal.state_dict(),
            'policy_net': policy_net.state_dict(),
            'aux_net': aux_net.state_dict() if aux_net is not None else None,
            'opponent_aux_net': opponent_aux_net.state_dict() if opponent_aux_net is not None else None,
            'danger_aux_net': danger_aux_net.state_dict() if danger_aux_net is not None else None,
            'oracle_brain': oracle_brain.state_dict() if oracle_brain is not None else None,
            'value_net': value_net.state_dict() if value_net is not None else None,
            'tile_eff_net': tile_eff_net.state_dict() if tile_eff_net is not None else None,
            'furo_regret_net': furo_regret_net.state_dict() if furo_regret_net is not None else None,
            'hand_value_regret_net': hand_value_regret_net.state_dict() if hand_value_regret_net is not None else None,
            'exp_reward_net': exp_reward_net.state_dict() if exp_reward_net is not None else None,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            # Compatibility progress includes legacy/inherited offsets, if any.
            'optimizer_steps': update_clock.progress,
            'optimizer_update_clock': update_clock.state_dict(),
            'timestamp': datetime.now().timestamp(),
            'best_perf': best_perf,
            'config': config,
            'reward_target_metadata': reward_target_metadata_dict,
            'dynamic_entropy_weight': dynamic_entropy_weight,
            'log_entropy_alpha': log_entropy_alpha,
            'oracle_guiding_runtime': actor_oracle_guiding_runtime_state(config, steps),
            BRAIN_IS_ORACLE_KEY: bool(mortal.is_oracle),
            DEPLOY_ZERO_ORACLE_KEY: False,
        }

    def persist_live_training_state(reward_target_metadata_dict=None):
        state = build_live_training_state(
            reward_target_metadata_dict=reward_target_metadata_dict,
        )
        ensure_parent_dir_for_file(state_file)
        atomic_torch_save(state, state_file)
        return state

    def stop_at_successful_optimizer_step_limit():
        if online and successful_optimizer_step_limit_reached(config, update_clock):
            persist_live_training_state(reward_target_metadata_dict=dict(reward_target_metadata))
            writer.flush()
            logging.info(
                'reached exact successful optimizer update limit=%s at microbatch=%s; '
                'saved latest at optimizer boundary (critic readiness is not implied)',
                successful_step_limit, steps,
            )
            sys.exit(ONLINE_MAX_STEPS_EXIT_CODE)

    def stop_after_checkpoint_if_requested():
        if training_stop_requested():
            persist_live_training_state(reward_target_metadata_dict=dict(reward_target_metadata))
            writer.flush()
            logging.info(
                'supervisor stop requested; saved latest checkpoint at microbatch=%s '
                '(optimizer boundary; replay cursor/old-policy snapshot are not persisted)',
                steps,
            )
            sys.exit(ONLINE_STOP_REQUEST_EXIT_CODE)

    if critic_only:
        logging.info('critic-only calibration phase: actor/policy eval + no_grad; no in-run PPO transition')
    if online and successful_step_limit:
        logging.info('exact successful optimizer update limit=%s; resumed successes=%s; LR remains microbatch-based',
                     successful_step_limit, update_clock.successes)
    stop_at_successful_optimizer_step_limit()
    stop_after_checkpoint_if_requested()

    if online and steps == 0 and test_play_eval_enabled and initial_test_eval_enabled:
        logging.info(
            'running initial step-0 test_play evaluation before training (games=%s)',
            initial_test_games,
        )
        run_test_play_evaluation(
            stats_dict=None,
            save_best_checkpoint=True,
            eval_games=initial_test_games,
        )
    elif online and steps == 0 and recorded_initial_baseline is not None:
        logging.info(
            'skipping initial step-0 test_play evaluation; recorded baseline: games=%s avg_rank=%s avg_pt=%s source_run=%s',
            recorded_initial_baseline.get('games'),
            recorded_initial_baseline.get('avg_rank'),
            recorded_initial_baseline.get('avg_pt'),
            recorded_initial_baseline.get('source_run'),
        )

    def train_epoch():
        nonlocal steps
        nonlocal idx
        nonlocal stats
        nonlocal Old_mortal
        nonlocal Old_policy_net
        nonlocal dynamic_entropy_weight
        nonlocal log_entropy_alpha
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), sorted(os.listdir(dirname))))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        # --- GAE: chunked batch generator for step-level advantages ---
        def _gae_chunk_iter(fl, chunk_size=50):
            """Generator: processes drain files in chunks to avoid OOM.
            Yields DataLoader batches with step-level GAE advantages.
            chunk_size=50 → ~1.1 GB obs per chunk; peak ~4 GB with copies."""
            traj_loader = FileDatasetsIter(
                version=version, file_list=fl, pts=pts,
                oracle=need_oracle_obs, player_names=player_names,
                emit_opponent_state_labels=online_opp_enabled,
                track_danger_labels=online_danger_enabled,
                track_regret_labels=online_regret_enabled,
                value_target_mode=resolved_value_target_mode,
                value_reward_source=resolved_value_reward_source,
                emit_context_meta=online_context_meta_enabled,
            )
            total_steps = 0
            total_games = 0
            infer_chunk = online_gae_inference_batch_size(config)

            for chunk_start in range(0, len(fl), chunk_size):
                chunk_files = fl[chunk_start:chunk_start + chunk_size]
                chunk_trajs = list(traj_loader.iter_game_trajectories(chunk_files))
                if online_replay_is and replay_is_drop_untracked:
                    before_count = len(chunk_trajs)
                    chunk_trajs = [traj for traj in chunk_trajs if behavior_version_is_usable(
                        traj.get('replay_param_version'), published_policy_history,
                        published_version=published_param_version,
                        max_gap=int(policy_cfg.get('max_behavior_version_gap', 1)),
                    )]
                    if len(chunk_trajs) != before_count:
                        logging.warning('dropped %s unknown or stale replay trajectories before targets',
                                        before_count - len(chunk_trajs))
                if not chunk_trajs:
                    continue

                # Batch target-policy / behavior-policy / value inference for this chunk
                all_obs_np = np.concatenate([t['obs'] for t in chunk_trajs], axis=0)
                obs_dev = torch.from_numpy(all_obs_np).float().to(device)
                del all_obs_np  # free CPU copy; GPU has its own copy
                all_actions_np = np.concatenate([t['actions'] for t in chunk_trajs], axis=0)
                actions_dev = torch.from_numpy(all_actions_np).to(dtype=torch.int64, device=device)
                del all_actions_np
                all_masks_np = np.concatenate([t['masks'] for t in chunk_trajs], axis=0)
                masks_dev = torch.from_numpy(all_masks_np).to(dtype=torch.bool, device=device)
                del all_masks_np
                replay_param_version_cpu = None
                if online_replay_is:
                    replay_param_version_cpu = torch.tensor(
                        np.concatenate([
                            np.full(
                                (len(t['actions']),),
                                (
                                    int(t['replay_param_version'])
                                    if t.get('replay_param_version', None) is not None
                                    else -1
                                ),
                                dtype=np.int64,
                            )
                            for t in chunk_trajs
                        ]),
                        dtype=torch.int64,
                    )
                current_actor_keep_prob = actor_oracle_guiding_keep_prob(config, steps)
                base_invisible_obs_dev = None
                if need_oracle_obs:
                    all_inv_np = np.concatenate([t['invisible_obs'] for t in chunk_trajs], axis=0)
                    base_invisible_obs_dev = torch.from_numpy(all_inv_np).float().to(device)
                    del all_inv_np
                actor_invisible_obs_dev, base_invisible_obs_dev = prepare_actor_invisible_obs(
                    obs_dev,
                    base_invisible_obs_dev,
                    current_actor_keep_prob,
                )
                traj_vtrace_flags = None
                chunk_uses_vtrace = False
                if vtrace_recursion_enabled:
                    traj_vtrace_flags = [
                        replay_importance_sampling_should_use_vtrace(
                            config,
                            published_param_version=published_param_version,
                            replay_param_version=traj.get('replay_param_version', -1),
                        )
                        for traj in chunk_trajs
                    ]
                    chunk_uses_vtrace = any(traj_vtrace_flags)
                v_list = []
                new_log_prob_list = []
                old_log_prob_list = []
                with torch.no_grad():
                    for s in range(0, obs_dev.shape[0], infer_chunk):
                        e = min(s + infer_chunk, obs_dev.shape[0])
                        obs_slice = obs_dev[s:e]
                        actions_slice = actions_dev[s:e]
                        masks_slice = masks_dev[s:e]
                        actor_invisible_slice = (
                            actor_invisible_obs_dev[s:e]
                            if actor_invisible_obs_dev is not None
                            else None
                        )
                        with torch.autocast(device.type, enabled=enable_amp):
                            current_phi = (
                                mortal(obs_slice, invisible_obs=actor_invisible_slice)
                                if actor_invisible_slice is not None
                                else mortal(obs_slice)
                            )
                            if value_net is not None:
                                if oracle_critic and oracle_brain is not None and base_invisible_obs_dev is not None:
                                    oracle_phi = oracle_brain(
                                        obs_slice,
                                        invisible_obs=base_invisible_obs_dev[s:e],
                                    )
                                    value_pred = value_net(oracle_phi)
                                else:
                                    value_pred = value_net(current_phi)
                                v_list.append(value_pred.cpu().float().numpy())
                            if chunk_uses_vtrace:
                                current_logits = policy_net.logits(current_phi, masks_slice)
                                new_log_prob_list.append(
                                    Categorical(logits=current_logits).log_prob(actions_slice).cpu().float().numpy()
                                )
                                old_phi = (
                                    Old_mortal(obs_slice, invisible_obs=actor_invisible_slice)
                                    if actor_invisible_slice is not None
                                    else Old_mortal(obs_slice)
                                )
                                old_logits = Old_policy_net.logits(old_phi, masks_slice)
                                old_log_prob_list.append(
                                    Categorical(logits=old_logits).log_prob(actions_slice).cpu().float().numpy()
                                )
                    if value_net is None:
                        v_list.append(np.zeros((obs_dev.shape[0], value_num_players), dtype=np.float32))
                if chunk_uses_vtrace:
                    new_log_prob_all = np.concatenate(new_log_prob_list, axis=0)
                    old_log_prob_tensor = torch.from_numpy(
                        np.concatenate(old_log_prob_list, axis=0)
                    ).to(dtype=torch.float32, device=device)
                    if online_replay_is and replay_param_version_cpu is not None:
                        old_log_prob_tensor, _, _, _, _, _ = apply_replay_importance_sampling(
                            old_log_prob=old_log_prob_tensor,
                            obs=obs_dev,
                            masks=masks_dev,
                            actions=actions_dev,
                            invisible_obs_dev=base_invisible_obs_dev,
                            replay_param_version=replay_param_version_cpu,
                        )
                    old_log_prob_all = old_log_prob_tensor.detach().cpu().float().numpy()
                    log_rhos_all = (new_log_prob_all - old_log_prob_all).astype(np.float32, copy=False)
                    del old_log_prob_tensor
                else:
                    log_rhos_all = None
                del obs_dev
                del actions_dev
                del masks_dev
                del actor_invisible_obs_dev
                del base_invisible_obs_dev
                v_all = np.concatenate(v_list, axis=0)

                # Compute GAE and build tensors for this chunk
                buf = defaultdict(list)
                offset = 0
                for traj_idx, traj in enumerate(chunk_trajs):
                    n = len(traj['obs'])
                    v_pred = v_all[offset:offset + n]
                    traj_use_vtrace = bool(
                        traj_vtrace_flags[traj_idx]
                        if traj_vtrace_flags is not None
                        else False
                    )
                    traj_log_rhos = (
                        log_rhos_all[offset:offset + n]
                        if traj_use_vtrace and log_rhos_all is not None
                        else None
                    )
                    offset += n
                    if traj_use_vtrace:
                        current_step_rewards = expand_sparse_kyoku_reward_to_steps(
                            traj['kyoku_advantage'],
                            traj['at_kyoku'],
                        )
                        current_v_target, gae_adv = compute_vtrace_targets_from_step_rewards(
                            current_step_rewards,
                            v_pred[:, 0],
                            traj_log_rhos,
                            gae_gamma,
                            rho_clip=vtrace_target_rho_clip,
                            c_clip=vtrace_target_c_clip,
                        )
                    else:
                        gae_adv = compute_gae_advantages(
                            traj['kyoku_advantage'],
                            traj['at_kyoku'],
                            v_pred[:, 0],
                            gae_gamma,
                            gae_lambda,
                        )
                        current_v_target = (gae_adv + v_pred[:, 0]).astype(np.float32)
                    v_tgt = np.empty_like(v_pred, dtype=np.float32)
                    v_tgt[:, 0] = current_v_target
                    for player_idx in range(1, v_pred.shape[1]):
                        if traj_use_vtrace:
                            player_step_rewards = expand_sparse_kyoku_reward_to_steps(
                                traj['kyoku_value_target'][:, player_idx],
                                traj['at_kyoku'],
                            )
                            player_v_target, _ = compute_vtrace_targets_from_step_rewards(
                                player_step_rewards,
                                v_pred[:, player_idx],
                                traj_log_rhos,
                                gae_gamma,
                                rho_clip=vtrace_target_rho_clip,
                                c_clip=vtrace_target_c_clip,
                            )
                        else:
                            player_adv = compute_gae_advantages(
                                traj['kyoku_value_target'][:, player_idx],
                                traj['at_kyoku'],
                                v_pred[:, player_idx],
                                gae_gamma,
                                gae_lambda,
                            )
                            player_v_target = player_adv + v_pred[:, player_idx]
                        v_tgt[:, player_idx] = np.asarray(player_v_target, dtype=np.float32)
                    buf['obs'].append(traj['obs'])
                    if traj['invisible_obs'] is not None:
                        buf['invisible_obs'].append(traj['invisible_obs'])
                    buf['actions'].append(traj['actions'])
                    buf['masks'].append(traj['masks'])
                    buf['advantage'].append(gae_adv.astype(np.float32))
                    buf['v_target'].append(v_tgt)
                    buf['player_rank'].extend([traj['player_rank']] * n)
                    if traj.get('context_meta') is not None:
                        buf['context_meta'].append(traj['context_meta'])
                    if online_replay_is:
                        buf['replay_param_version'].extend([
                            normalized_behavior_version(traj.get('replay_param_version'))
                        ] * n)
                    for key in ('opp_shanten', 'opp_tenpai',
                                'danger_valid', 'danger_any', 'danger_value', 'danger_player_mask',
                                'tile_eff_valid', 'tile_eff_shanten_delta',
                                'furo_valid', 'furo_label', 'hand_value_valid', 'hand_value_points'):
                        if traj.get(key) is not None:
                            buf[key].append(traj[key])

                n_chunk_games = len(chunk_trajs)
                del chunk_trajs, v_all

                def cat(key):
                    return np.concatenate(buf[key]) if buf[key] else None

                tensors = [torch.from_numpy(cat('obs')).float()]
                if buf['invisible_obs']:
                    tensors.append(torch.from_numpy(cat('invisible_obs')).float())
                tensors += [
                    torch.from_numpy(cat('actions')),
                    torch.from_numpy(cat('masks')),
                    torch.from_numpy(cat('advantage')).float(),
                    torch.from_numpy(cat('v_target')).float(),
                    torch.tensor(buf['player_rank'], dtype=torch.int64),
                ]
                if buf['context_meta']:
                    tensors.append(torch.from_numpy(cat('context_meta')).to(dtype=torch.int64))
                if online_replay_is:
                    tensors.append(torch.tensor(buf['replay_param_version'], dtype=torch.int64))
                for key in (['opp_shanten', 'opp_tenpai'] if online_opp_enabled else []):
                    tensors.append(torch.from_numpy(cat(key)))
                for key in (['danger_valid', 'danger_any', 'danger_value', 'danger_player_mask'] if online_danger_enabled else []):
                    tensors.append(torch.from_numpy(cat(key)))
                for key in (['tile_eff_valid', 'tile_eff_shanten_delta',
                             'furo_valid', 'furo_label', 'hand_value_valid', 'hand_value_points'] if online_regret_enabled else []):
                    tensors.append(torch.from_numpy(cat(key)))
                del buf

                # Shuffle within chunk
                perm = torch.randperm(tensors[0].shape[0])
                tensors = [t[perm] for t in tensors]
                n_steps = tensors[0].shape[0]
                total_steps += n_steps
                total_games += n_chunk_games
                logging.info(f'GAE chunk {chunk_start // chunk_size + 1}: {n_steps:,} steps from {n_chunk_games:,} games')

                chunk_ds = TensorDataset(*tensors)
                del tensors
                sub_loader = DataLoader(chunk_ds, batch_size=batch_size, drop_last=False,
                                        shuffle=False, num_workers=0, pin_memory=True)
                yield from sub_loader
                # DataLoader owns its dataset. Release both before encoding the
                # next chunk, otherwise two full chunks coexist during setup.
                del sub_loader, chunk_ds
                gc.collect()

            logging.info(f'GAE total: {total_steps:,} steps from {total_games:,} games')

        if gae_enabled and online:
            if not file_list:
                return
            data_loader = _gae_chunk_iter(file_list)
        else:
            if num_workers > 1:
                random.shuffle(file_list)
            file_data = FileDatasetsIter(
                version = version,
                file_list = file_list,
                pts = pts,
                oracle = need_oracle_obs,
                file_batch_size = file_batch_size,
                reserve_ratio = reserve_ratio,
                player_names = player_names,
                num_epochs = num_epochs,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
                emit_opponent_state_labels = online_opp_enabled,
                track_danger_labels = online_danger_enabled,
                track_regret_labels = online_regret_enabled,
                emit_value_targets = value_enabled,
                value_target_mode = resolved_value_target_mode,
                value_reward_source = resolved_value_reward_source,
                emit_context_meta = online_context_meta_enabled,
                emit_replay_param_version = online_replay_is,
            )
            data_loader_kwargs = {
                'dataset': file_data,
                'batch_size': batch_size,
                'drop_last': False,
                'num_workers': num_workers,
                'pin_memory': True,
                'worker_init_fn': worker_init_fn,
            }
            if num_workers > 0:
                data_loader_kwargs['persistent_workers'] = True
                data_loader_kwargs['prefetch_factor'] = prefetch_factor
            data_loader = iter(DataLoader(**data_loader_kwargs))

        remaining_obs = []
        remaining_invisible_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_advantage = []
        remaining_player_rank = []
        remaining_extra = []  # opp/danger labels (variable-length tail)
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def prepare_actor_invisible_obs(obs_tensor, base_invisible_obs, keep_prob):
            if not actor_oracle_enabled:
                return None, base_invisible_obs
            if base_invisible_obs is None:
                oracle_channels = oracle_obs_shape(version)[0]
                base_invisible_obs = torch.zeros(
                    (obs_tensor.shape[0], oracle_channels, obs_tensor.shape[-1]),
                    dtype=obs_tensor.dtype,
                    device=device,
                )
            actor_invisible_obs = transform_actor_oracle_invisible_obs(
                base_invisible_obs,
                actor_source=actor_oracle_source,
                keep_prob=keep_prob,
            )
            return actor_invisible_obs, base_invisible_obs

        def apply_replay_importance_sampling(
            *,
            old_log_prob,
            obs,
            masks,
            actions,
            invisible_obs_dev,
            replay_param_version,
        ):
            nonlocal loaded_behavior_version
            coverage = torch.tensor(0.0, device=device)
            missing_fraction = torch.tensor(0.0, device=device)
            gap_mean = torch.tensor(0.0, device=device)
            gap_max = torch.tensor(0.0, device=device)
            tracked_mask_cpu = None
            if not online_replay_is or replay_param_version is None:
                return old_log_prob, coverage, missing_fraction, gap_mean, gap_max, tracked_mask_cpu

            replay_param_version_cpu = replay_param_version.to(dtype=torch.int64, device='cpu')
            tracked_mask_cpu = tracked_replay_versions_mask(
                replay_param_version_cpu, published_policy_history,
            )
            valid_mask_cpu = replay_param_version_cpu.ge(0)
            valid_count = int(valid_mask_cpu.sum().item())
            if valid_count <= 0:
                return old_log_prob, coverage, missing_fraction, gap_mean, gap_max, tracked_mask_cpu

            if published_param_version >= 0:
                version_gaps = (
                    published_param_version - replay_param_version_cpu[valid_mask_cpu]
                ).clamp(min=0)
                gap_mean = torch.tensor(float(version_gaps.float().mean().item()), device=device)
                gap_max = torch.tensor(float(version_gaps.max().item()), device=device)

            used_behavior_mask_cpu = torch.zeros_like(valid_mask_cpu, dtype=torch.bool)
            valid_indices_cpu = torch.arange(replay_param_version_cpu.shape[0], dtype=torch.int64)
            unique_versions = replay_param_version_cpu[valid_mask_cpu].unique(sorted=True).tolist()
            for version_id in unique_versions:
                payload = published_policy_history.get(version_id)
                if payload is None:
                    continue
                if loaded_behavior_version != version_id:
                    behavior_mortal.load_state_dict(payload['mortal'])
                    behavior_policy_net.load_state_dict(payload['policy_net'])
                    loaded_behavior_version = version_id
                sample_mask_cpu = replay_param_version_cpu.eq(version_id)
                if not bool(sample_mask_cpu.any()):
                    continue
                sample_indices = valid_indices_cpu[sample_mask_cpu].to(device=device)
                behavior_runtime = payload.get('runtime', {})
                behavior_keep_prob = float(
                    behavior_runtime.get(
                        'actor_oracle_keep_prob',
                        1.0 if actor_oracle_enabled else 0.0,
                    )
                )
                sample_invisible_obs = (
                    invisible_obs_dev.index_select(0, sample_indices)
                    if invisible_obs_dev is not None
                    else None
                )
                behavior_invisible_obs, sample_invisible_obs = prepare_actor_invisible_obs(
                    obs.index_select(0, sample_indices),
                    sample_invisible_obs,
                    behavior_keep_prob,
                )
                with torch.no_grad():
                    with torch.autocast(device.type, enabled=enable_amp):
                        behavior_phi = (
                            behavior_mortal(
                                obs.index_select(0, sample_indices),
                                invisible_obs=behavior_invisible_obs,
                            )
                            if actor_oracle_enabled
                            else behavior_mortal(obs.index_select(0, sample_indices))
                        )
                        behavior_logits = behavior_policy_net.logits(
                            behavior_phi,
                            masks.index_select(0, sample_indices),
                        )
                        behavior_dist = Categorical(logits=behavior_logits)
                        old_log_prob[sample_indices] = behavior_dist.log_prob(
                            actions.index_select(0, sample_indices)
                        )
                used_behavior_mask_cpu |= sample_mask_cpu

            coverage = torch.tensor(
                float(used_behavior_mask_cpu.float().mean().item()),
                device=device,
            )
            missing_fraction = torch.tensor(
                float((valid_mask_cpu & ~used_behavior_mask_cpu).float().mean().item()),
                device=device,
            )
            tracked_mask_cpu = used_behavior_mask_cpu
            return old_log_prob, coverage, missing_fraction, gap_mean, gap_max, tracked_mask_cpu

        def train_batch(obs, actions, masks, advantage, player_rank,
                        invisible_obs=None, replay_param_version=None,
                        context_meta=None,
                        opp_shanten=None, opp_tenpai=None,
                        danger_valid=None, danger_any=None,
                        danger_value=None, danger_player_mask=None,
                        tile_eff_valid=None, tile_eff_delta=None,
                        furo_valid=None, furo_label=None,
                        hand_value_valid=None, hand_value_points=None,
                        v_target=None):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal stats
            nonlocal Old_mortal
            nonlocal Old_policy_net
            nonlocal dynamic_entropy_weight
            nonlocal log_entropy_alpha

            aux_monitor = stats['aux_monitor']
            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            advantage = advantage.to(dtype=torch.float32, device=device)
            player_rank = player_rank.to(dtype=torch.int64, device=device)
            context_meta = (
                context_meta.to(dtype=torch.int64, device=device)
                if context_meta is not None
                else None
            )
            invisible_obs_dev = (
                invisible_obs.to(dtype=torch.float32, device=device)
                if invisible_obs is not None
                else None
            )
            replay_param_version_cpu = (
                replay_param_version.to(dtype=torch.int64)
                if replay_param_version is not None
                else None
            )
            policy_scope_reject_fraction = torch.tensor(0.0, device=device)
            current_actor_keep_prob = actor_oracle_guiding_keep_prob(config, steps)
            actor_oracle_continuation = actor_oracle_guiding_continuation_active(config, steps)
            actor_invisible_obs, invisible_obs_dev = prepare_actor_invisible_obs(
                obs,
                invisible_obs_dev,
                current_actor_keep_prob,
            )
            policy_keep_mask = policy_online_action_keep_mask(actions, policy_action_scope)
            if policy_keep_mask is not None:
                if not bool(policy_keep_mask.any()):
                    return
                if not bool(policy_keep_mask.all()):
                    policy_scope_reject_fraction = 1.0 - policy_keep_mask.float().mean()
                    cpu_policy_keep_mask = policy_keep_mask.cpu()
                    obs = obs[policy_keep_mask]
                    actions = actions[policy_keep_mask]
                    masks = masks[policy_keep_mask]
                    advantage = advantage[policy_keep_mask]
                    player_rank = player_rank[policy_keep_mask]
                    context_meta = masked_optional_tensor(
                        context_meta,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    invisible_obs_dev = masked_optional_tensor(
                        invisible_obs_dev,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    replay_param_version_cpu = masked_optional_tensor(
                        replay_param_version_cpu,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    opp_shanten = masked_optional_tensor(
                        opp_shanten,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    opp_tenpai = masked_optional_tensor(
                        opp_tenpai,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    danger_valid = masked_optional_tensor(
                        danger_valid,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    danger_any = masked_optional_tensor(
                        danger_any,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    danger_value = masked_optional_tensor(
                        danger_value,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    danger_player_mask = masked_optional_tensor(
                        danger_player_mask,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    tile_eff_valid = masked_optional_tensor(
                        tile_eff_valid,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    tile_eff_delta = masked_optional_tensor(
                        tile_eff_delta,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    furo_valid = masked_optional_tensor(
                        furo_valid,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    furo_label = masked_optional_tensor(
                        furo_label,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    hand_value_valid = masked_optional_tensor(
                        hand_value_valid,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    hand_value_points = masked_optional_tensor(
                        hand_value_points,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
                    v_target = masked_optional_tensor(
                        v_target,
                        policy_keep_mask,
                        cpu_policy_keep_mask,
                    )
            replay_is_coverage = torch.tensor(0.0, device=device)
            replay_is_missing_fraction = torch.tensor(0.0, device=device)
            replay_is_gap_mean = torch.tensor(0.0, device=device)
            replay_is_gap_max = torch.tensor(0.0, device=device)
            replay_is_tracked_mask_cpu = None
            critic_warmup = value_critic_warmup_active(config, steps)
            policy_step_active = (not critic_warmup) and policy_update_active(config, steps)

            with torch.no_grad():
                with torch.autocast(device.type, enabled=enable_amp):
                    old_phi = (
                        Old_mortal(obs, invisible_obs=actor_invisible_obs)
                        if actor_oracle_enabled
                        else Old_mortal(obs)
                    )
                    old_logits = Old_policy_net.logits(old_phi, masks)
                    old_dist = Categorical(logits=old_logits)
                    old_log_prob = old_dist.log_prob(actions)
                old_log_prob, replay_is_coverage, replay_is_missing_fraction, replay_is_gap_mean, replay_is_gap_max, replay_is_tracked_mask_cpu = apply_replay_importance_sampling(
                    old_log_prob=old_log_prob,
                    obs=obs,
                    masks=masks,
                    actions=actions,
                    invisible_obs_dev=invisible_obs_dev,
                    replay_param_version=replay_param_version_cpu,
                )

            actor_grad_context = actor_forward_context(
                mortal, policy_net, frozen=critic_warmup, critic_only=critic_only,
            )
            with actor_grad_context:
                with torch.autocast(device.type, enabled=enable_amp):
                    phi = (
                        mortal(obs, invisible_obs=actor_invisible_obs)
                        if actor_oracle_enabled
                        else mortal(obs)
                    )
                    logits = policy_net.logits(phi, masks)
            with torch.autocast(device.type, enabled=enable_amp):
                dist = Categorical(logits=logits)
                new_log_prob = dist.log_prob(actions)
                ratio = (new_log_prob - old_log_prob).exp()
                clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                importance_reject_fraction = torch.tensor(0.0, device=device)
                if replay_is_drop_untracked and replay_is_tracked_mask_cpu is not None:
                    keep_mask = replay_is_tracked_mask_cpu.to(device=device)
                    if not bool(keep_mask.any()):
                        logging.warning('dropping replay batch with no tracked behavior policy')
                        return
                    if not bool(keep_mask.all()):
                        obs = obs[keep_mask]
                        actions = actions[keep_mask]
                        masks = masks[keep_mask]
                        advantage = advantage[keep_mask]
                        player_rank = player_rank[keep_mask]
                        if context_meta is not None:
                            context_meta = context_meta[keep_mask]
                        old_log_prob = old_log_prob[keep_mask]
                        new_log_prob = new_log_prob[keep_mask]
                        ratio = ratio[keep_mask]
                        clipped_ratio = clipped_ratio[keep_mask]
                        phi = phi[keep_mask]
                        logits = logits[keep_mask]
                        if actor_invisible_obs is not None:
                            actor_invisible_obs = actor_invisible_obs[keep_mask]
                        if invisible_obs_dev is not None:
                            invisible_obs_dev = invisible_obs_dev[keep_mask]
                        if replay_param_version_cpu is not None:
                            replay_param_version_cpu = replay_param_version_cpu[keep_mask.cpu()]
                        if v_target is not None:
                            v_target = v_target[keep_mask]
                        if opp_shanten is not None:
                            opp_shanten = opp_shanten[keep_mask]
                        if opp_tenpai is not None:
                            opp_tenpai = opp_tenpai[keep_mask]
                        if danger_valid is not None:
                            danger_valid = danger_valid[keep_mask]
                        if danger_any is not None:
                            danger_any = danger_any[keep_mask]
                        if danger_value is not None:
                            danger_value = danger_value[keep_mask]
                        if danger_player_mask is not None:
                            danger_player_mask = danger_player_mask[keep_mask]
                        if tile_eff_valid is not None:
                            tile_eff_valid = tile_eff_valid[keep_mask]
                        if tile_eff_delta is not None:
                            tile_eff_delta = tile_eff_delta[keep_mask]
                        if furo_valid is not None:
                            furo_valid = furo_valid[keep_mask]
                        if furo_label is not None:
                            furo_label = furo_label[keep_mask]
                        if hand_value_valid is not None:
                            hand_value_valid = hand_value_valid[keep_mask]
                        if hand_value_points is not None:
                            hand_value_points = hand_value_points[keep_mask]
                        dist = Categorical(logits=logits)
                if actor_oracle_continuation and actor_oracle_importance_threshold > 0:
                    keep_mask = torch.isfinite(ratio) & ratio.le(actor_oracle_importance_threshold)
                    if not keep_mask.any():
                        keep_mask = torch.ones_like(ratio, dtype=torch.bool)
                    if not bool(keep_mask.all()):
                        importance_reject_fraction = 1.0 - keep_mask.float().mean()
                        obs = obs[keep_mask]
                        actions = actions[keep_mask]
                        masks = masks[keep_mask]
                        advantage = advantage[keep_mask]
                        player_rank = player_rank[keep_mask]
                        if context_meta is not None:
                            context_meta = context_meta[keep_mask]
                        old_log_prob = old_log_prob[keep_mask]
                        new_log_prob = new_log_prob[keep_mask]
                        ratio = ratio[keep_mask]
                        clipped_ratio = clipped_ratio[keep_mask]
                        phi = phi[keep_mask]
                        logits = logits[keep_mask]
                        if actor_invisible_obs is not None:
                            actor_invisible_obs = actor_invisible_obs[keep_mask]
                        if invisible_obs_dev is not None:
                            invisible_obs_dev = invisible_obs_dev[keep_mask]
                        if replay_param_version_cpu is not None:
                            replay_param_version_cpu = replay_param_version_cpu[keep_mask.cpu()]
                        if v_target is not None:
                            v_target = v_target[keep_mask]
                        if opp_shanten is not None:
                            opp_shanten = opp_shanten[keep_mask]
                        if opp_tenpai is not None:
                            opp_tenpai = opp_tenpai[keep_mask]
                        if danger_valid is not None:
                            danger_valid = danger_valid[keep_mask]
                        if danger_any is not None:
                            danger_any = danger_any[keep_mask]
                        if danger_value is not None:
                            danger_value = danger_value[keep_mask]
                        if danger_player_mask is not None:
                            danger_player_mask = danger_player_mask[keep_mask]
                        if tile_eff_valid is not None:
                            tile_eff_valid = tile_eff_valid[keep_mask]
                        if tile_eff_delta is not None:
                            tile_eff_delta = tile_eff_delta[keep_mask]
                        if furo_valid is not None:
                            furo_valid = furo_valid[keep_mask]
                        if furo_label is not None:
                            furo_label = furo_label[keep_mask]
                        if hand_value_valid is not None:
                            hand_value_valid = hand_value_valid[keep_mask]
                        if hand_value_points is not None:
                            hand_value_points = hand_value_points[keep_mask]
                        dist = Categorical(logits=logits)

                raw_advantage, advantage, v_target = prepare_policy_advantage_and_value_target(
                    advantage,
                    v_target,
                    device=device,
                    gae_enabled=gae_enabled,
                )
                current_bs = actions.shape[0]
                assert masks[range(current_bs), actions].all()

                drift = policy_drift(ratio, clip_ratio)
                if (
                    drift['approx_kl'].item() > float(policy_cfg.get('target_kl', 0.02))
                    or drift['clip_fraction'].item() > float(policy_cfg.get('max_clip_fraction', 0.5))
                ):
                    logging.warning('skipping drifted batch before optimizer update: KL=%.6g clip_fraction=%.4f',
                                    drift['approx_kl'].item(), drift['clip_fraction'].item())
                    writer.add_scalar('policy_drift/rejected_kl', drift['approx_kl'], steps)
                    writer.add_scalar('policy_drift/rejected_clip_fraction', drift['clip_fraction'], steps)
                    return
                clip_loss = actor_surrogate(
                    new_log_prob, ratio,
                    raw_advantage if actor_objective == 'vtrace' else advantage,
                    objective=actor_objective, clip_ratio=clip_ratio,
                    dual_clip=dual_clip, importance_rho_clip=importance_rho_clip,
                )
                policy_logit_gate_fraction = torch.tensor(0.0, device=device)
                if logit_gate_threshold > 0:
                    chosen_logits = logits.gather(1, actions.view(-1, 1)).squeeze(1)
                    raw_adv_flat = raw_advantage.reshape(-1)
                    gate_keep = ~(
                        ((raw_adv_flat > 0) & chosen_logits.ge(logit_gate_threshold))
                        | ((raw_adv_flat < 0) & chosen_logits.le(-logit_gate_threshold))
                    )
                    policy_logit_gate_fraction = 1.0 - gate_keep.float().mean()
                    gate_tensor = gate_keep.to(dtype=clip_loss.dtype)
                    while gate_tensor.ndim < clip_loss.ndim:
                        gate_tensor = gate_tensor.unsqueeze(-1)
                    clip_loss = clip_loss * gate_tensor
                entropy = dist.entropy()

                if not policy_step_active:
                    loss = torch.zeros((), dtype=phi.dtype, device=device)
                else:
                    loss = compute_policy_objective_loss(
                        clip_loss,
                        entropy,
                        dynamic_entropy_weight,
                    )

                if online_context_meta_enabled and context_meta is None:
                    raise RuntimeError('online auxiliary heads enabled but context_meta is missing')

                # AuxNet auxiliary loss (rank prediction) with the supervised-stage
                # context weighting: turn bucket, south/all-last emphasis, gap focus,
                # and max-weight clipping.
                aux_loss_val = torch.tensor(0.0, device=device)
                if aux_net is not None and policy_step_active:
                    rank_logits = aux_net(phi)[0]
                    rank_aux_weights = compute_rank_aux_sample_weights(
                        context_meta,
                        device=device,
                        base_weight=effective_aux_training_cfg['rank_base_weight'],
                        south_factor=effective_aux_training_cfg['rank_south_factor'],
                        all_last_factor=effective_aux_training_cfg['rank_all_last_factor'],
                        gap_focus_points=effective_aux_training_cfg['rank_gap_focus_points'],
                        gap_close_bonus=effective_aux_training_cfg['rank_gap_close_bonus'],
                        max_weight=effective_aux_training_cfg['rank_max_weight'],
                        turn_weighting=effective_aux_training_cfg['rank_turn_weighting'],
                    )
                    rank_loss_vec = nn.functional.cross_entropy(
                        rank_logits,
                        player_rank,
                        reduction='none',
                    )
                    aux_loss_val = (rank_loss_vec * rank_aux_weights).mean()
                    loss = loss + aux_loss_val
                    with torch.inference_mode():
                        aux_monitor['rank_correct'] += (
                            rank_logits.argmax(-1) == player_rank
                        ).to(torch.int64).sum()
                        aux_monitor['rank_count'] += torch.tensor(
                            rank_loss_vec.shape[0],
                            dtype=torch.int64,
                            device=device,
                        )
                        aux_monitor['rank_aux_loss_sum'] += (
                            aux_loss_val.detach().to(torch.float64) * rank_loss_vec.shape[0]
                        )
                        aux_monitor['rank_aux_raw_loss_sum'] += rank_loss_vec.detach().to(torch.float64).sum()
                        aux_monitor['rank_aux_weight_sum'] += rank_aux_weights.detach().to(torch.float64).sum()

                # Opponent State auxiliary loss
                opp_loss_val = torch.tensor(0.0, device=device)
                if opponent_aux_net is not None and opp_shanten is not None and policy_step_active:
                    opp_shanten_dev = opp_shanten.to(dtype=torch.int64, device=device)
                    opp_tenpai_dev = opp_tenpai.to(dtype=torch.int64, device=device)
                    shanten_logits, tenpai_logits = opponent_aux_net(phi)
                    opponent_turn_weights = compute_context_turn_weights(
                        context_meta,
                        effective_aux_training_cfg['opponent_turn_weighting'],
                        device=device,
                    )
                    shanten_losses = []
                    tenpai_losses = []
                    shanten_correct = torch.zeros((3,), dtype=torch.int64, device=device)
                    tenpai_correct = torch.zeros((3,), dtype=torch.int64, device=device)
                    for i in range(3):
                        shanten_losses.append(
                            nn.functional.cross_entropy(
                                shanten_logits[i],
                                opp_shanten_dev[:, i],
                                reduction='none',
                            )
                        )
                        tenpai_losses.append(
                            nn.functional.cross_entropy(
                                tenpai_logits[i],
                                opp_tenpai_dev[:, i],
                                reduction='none',
                            )
                        )
                        shanten_correct[i] = (
                            shanten_logits[i].argmax(-1) == opp_shanten_dev[:, i]
                        ).to(torch.int64).sum()
                        tenpai_correct[i] = (
                            tenpai_logits[i].argmax(-1) == opp_tenpai_dev[:, i]
                        ).to(torch.int64).sum()
                    shanten_loss_vec = torch.stack(shanten_losses).mean(dim=0)
                    tenpai_loss_vec = torch.stack(tenpai_losses).mean(dim=0)
                    raw_opp_loss_vec = (
                        effective_aux_training_cfg['opponent_shanten_weight'] * shanten_loss_vec
                        + effective_aux_training_cfg['opponent_tenpai_weight'] * tenpai_loss_vec
                    )
                    opp_loss_val = effective_aux_training_cfg['opponent_state_weight'] * (
                        raw_opp_loss_vec * opponent_turn_weights
                    ).mean()
                    loss = loss + opp_loss_val
                    with torch.inference_mode():
                        aux_monitor['opponent_sample_count'] += torch.tensor(
                            shanten_loss_vec.shape[0],
                            dtype=torch.int64,
                            device=device,
                        )
                        aux_monitor['opponent_aux_loss_sum'] += (
                            opp_loss_val.detach().to(torch.float64) * shanten_loss_vec.shape[0]
                        )
                        aux_monitor['opponent_turn_weight_sum'] += opponent_turn_weights.detach().to(torch.float64).sum()
                        aux_monitor['opponent_shanten_loss_sum'] += shanten_loss_vec.detach().to(torch.float64).sum()
                        aux_monitor['opponent_tenpai_loss_sum'] += tenpai_loss_vec.detach().to(torch.float64).sum()
                        aux_monitor['opponent_count'] += torch.full_like(
                            aux_monitor['opponent_count'],
                            shanten_loss_vec.shape[0],
                        )
                        aux_monitor['opponent_shanten_correct'] += shanten_correct
                        aux_monitor['opponent_tenpai_correct'] += tenpai_correct

                # Danger auxiliary loss
                danger_loss_val = torch.tensor(0.0, device=device)
                if danger_aux_net is not None and danger_valid is not None and policy_step_active:
                    danger_valid_dev = danger_valid.to(dtype=torch.bool, device=device)
                    danger_any_target = danger_any.to(dtype=torch.bool, device=device)
                    danger_any_dev = danger_any_target.to(dtype=torch.float32)
                    danger_value_dev = danger_value.to(dtype=torch.float32, device=device)
                    danger_player_target = danger_player_mask.to(dtype=torch.bool, device=device)
                    danger_player_dev = danger_player_target.to(dtype=torch.float32)
                    any_logits, value_pred, player_logits = danger_aux_net(phi)
                    danger_turn_weights = compute_context_turn_weights(
                        context_meta,
                        effective_aux_training_cfg['danger_turn_weighting'],
                        device=device,
                    )
                    # Eligible mask: valid steps AND legal discards (first 37 actions)
                    eligible = danger_valid_dev.unsqueeze(-1) & masks[:, :37]
                    any_loss_vec = balanced_bce_per_sample_with_logits(
                        any_logits,
                        danger_any_dev,
                        eligible,
                        focal_gamma=effective_aux_training_cfg['danger_focal_gamma'],
                    )
                    # value loss: smooth L1 on positive-danger tiles
                    _dvc_log = _math.log1p(danger_value_cap)
                    value_positive = eligible & danger_any_target
                    positive_target = torch.log1p(danger_value_dev.clamp(min=0, max=danger_value_cap)) / _dvc_log
                    value_loss_map = nn.functional.smooth_l1_loss(
                        value_pred.sigmoid(), positive_target, reduction='none'
                    )
                    value_positive_weight = value_positive.to(dtype=value_loss_map.dtype)
                    value_positive_count = value_positive_weight.sum(dim=1)
                    value_loss_vec = (
                        (value_loss_map * value_positive_weight).sum(dim=1)
                        / value_positive_count.clamp_min(1.0)
                    )
                    value_loss_vec = torch.where(
                        value_positive_count > 0,
                        value_loss_vec,
                        torch.zeros_like(value_loss_vec),
                    )
                    # player loss: balanced BCE on eligible tiles
                    eligible_player = eligible.unsqueeze(-1).expand_as(player_logits)
                    player_loss_vec = balanced_bce_per_sample_with_logits(
                        player_logits,
                        danger_player_dev,
                        eligible_player,
                        focal_gamma=effective_aux_training_cfg['danger_focal_gamma'],
                    )
                    raw_danger_loss_vec = (
                        danger_mix_weights[0] * any_loss_vec
                        + danger_mix_weights[1] * value_loss_vec
                        + danger_mix_weights[2] * player_loss_vec
                    )
                    if effective_aux_training_cfg['danger_ramp_steps'] > 0:
                        danger_ramp = min(
                            float(update_clock.progress) / max(float(effective_aux_training_cfg['danger_ramp_steps']), 1.0),
                            1.0,
                        )
                    else:
                        danger_ramp = 1.0
                    danger_loss_val = (
                        (raw_danger_loss_vec * danger_turn_weights).mean()
                        * effective_aux_training_cfg['danger_weight']
                        * danger_ramp
                    )
                    loss = loss + danger_loss_val
                    with torch.inference_mode():
                        aux_monitor['danger_sample_count'] += torch.tensor(
                            raw_danger_loss_vec.shape[0],
                            dtype=torch.int64,
                            device=device,
                        )
                        aux_monitor['danger_aux_loss_sum'] += (
                            danger_loss_val.detach().to(torch.float64) * raw_danger_loss_vec.shape[0]
                        )
                        aux_monitor['danger_turn_weight_sum'] += danger_turn_weights.detach().to(torch.float64).sum()
                        aux_monitor['danger_any_loss_sum'] += any_loss_vec.detach().to(torch.float64).sum()
                        aux_monitor['danger_value_loss_sum'] += value_loss_vec.detach().to(torch.float64).sum()
                        aux_monitor['danger_player_loss_sum'] += player_loss_vec.detach().to(torch.float64).sum()

                        any_prob = any_logits.sigmoid()
                        any_pred = any_prob >= 0.5
                        batch_any_stats = init_binary_metric_dict(device=device)
                        update_binary_metric(
                            batch_any_stats,
                            eligible,
                            danger_any_target,
                            any_pred,
                            any_prob,
                        )
                        merge_binary_metric(aux_monitor['danger_any_stats'], batch_any_stats)

                        player_prob = player_logits.sigmoid()
                        player_pred = player_prob >= 0.5
                        batch_player_stats = init_binary_metric_dict(device=device)
                        update_binary_metric(
                            batch_player_stats,
                            eligible_player,
                            danger_player_target,
                            player_pred,
                            player_prob,
                        )
                        merge_binary_metric(aux_monitor['danger_player_stats'], batch_player_stats)

                        aux_monitor['danger_value_pos_count'] += value_positive.sum(dtype=torch.int64)
                        positive_value_pred = value_pred[value_positive].sigmoid()
                        positive_value = danger_value_dev[value_positive]
                        if positive_value_pred.numel() > 0:
                            point_error = torch.expm1(positive_value_pred * _dvc_log) - positive_value
                            aux_monitor['danger_value_abs_err_sum'] += point_error.abs().sum().to(torch.float64)
                            aux_monitor['danger_value_sq_err_sum'] += point_error.square().sum().to(torch.float64)

                # Compute oracle features once, reuse for ValueHead + ExpectedRewardNet
                oracle_phi_cached = None
                if oracle_critic and oracle_brain is not None and invisible_obs_dev is not None:
                    # Critic keeps full oracle information throughout training.
                    oracle_phi_cached = oracle_brain(obs, invisible_obs=invisible_obs_dev)

                # Oracle Critic / Value Head loss (RVR-style)
                value_loss_val = torch.tensor(0.0, device=device)
                if value_enabled and value_net is not None:
                    if oracle_phi_cached is not None:
                        value_pred = value_net(oracle_phi_cached)
                    else:
                        # Visible value head trains on detached features during true warmup;
                        # ordinary alternating training still shapes the shared actor trunk.
                        value_pred = value_net(phi)
                    if v_target is not None:
                        value_target = v_target
                    else:
                        value_target = torch.zeros_like(value_pred)
                        value_target[:, 0] = advantage
                    value_loss_val = nn.functional.mse_loss(value_pred, value_target)
                    # Zero-sum regularization only makes sense on the 4-player target path.
                    if zero_sum_weight > 0:
                        value_sum = value_pred.sum(dim=-1)
                        value_loss_val = value_loss_val + zero_sum_weight * value_sum.square().mean()
                    loss = loss + value_weight * value_loss_val

                # Local Regret Heads
                tile_eff_loss_val = torch.tensor(0.0, device=device)
                if tile_eff_net is not None and tile_eff_valid is not None and policy_step_active:
                    tile_eff_pred = tile_eff_net(phi.detach())
                    te_valid = tile_eff_valid.to(device=device)
                    if te_valid.any():
                        te_target = tile_eff_delta.to(dtype=torch.float32, device=device)
                        tile_eff_loss_val = nn.functional.smooth_l1_loss(
                            tile_eff_pred[te_valid], te_target[te_valid])
                        loss = loss + tile_eff_weight * tile_eff_loss_val

                furo_regret_loss_val = torch.tensor(0.0, device=device)
                if furo_regret_net is not None and furo_valid is not None and policy_step_active:
                    furo_pred = furo_regret_net(phi.detach())
                    fr_valid = furo_valid.to(device=device)
                    if fr_valid.any():
                        fr_target = furo_label.to(dtype=torch.float32, device=device)
                        # Furo label is [called, shanten_before, shanten_after]
                        # FuroRegretHead outputs 2-dim [call_regret, pass_regret]
                        # Map: call label -> target = [shanten_before - shanten_after, 0]
                        #       pass label -> target = [0, shanten_after - shanten_before]
                        called = fr_target[fr_valid, 0]  # 1.0 or 0.0
                        sh_before = fr_target[fr_valid, 1]
                        sh_after = fr_target[fr_valid, 2]
                        sh_delta = sh_before - sh_after  # positive = shanten improved
                        furo_target = torch.zeros_like(furo_pred[fr_valid])
                        furo_target[:, 0] = called * sh_delta  # call regret
                        furo_target[:, 1] = (1.0 - called) * (-sh_delta)  # pass regret
                        furo_regret_loss_val = nn.functional.smooth_l1_loss(
                            furo_pred[fr_valid], furo_target)
                        loss = loss + furo_regret_weight * furo_regret_loss_val

                hand_value_regret_loss_val = torch.tensor(0.0, device=device)
                if hand_value_regret_net is not None and hand_value_valid is not None and policy_step_active:
                    hv_pred = hand_value_regret_net(phi.detach())
                    hv_valid = hand_value_valid.to(device=device)
                    if hv_valid.any():
                        hv_target = hand_value_points.to(dtype=torch.float32, device=device)
                        # Normalize to [0, 1] range for training stability
                        hv_target_norm = hv_target[hv_valid] / 32000.0
                        hv_pred_norm = hv_pred[hv_valid].sigmoid()  # output in [0, 1]
                        hand_value_regret_loss_val = nn.functional.smooth_l1_loss(
                            hv_pred_norm, hv_target_norm)
                        loss = loss + hand_value_regret_weight * hand_value_regret_loss_val

                # Expected Reward Network (reuses cached oracle features)
                exp_reward_loss_val = torch.tensor(0.0, device=device)
                if exp_reward_net is not None and steps >= exp_reward_warmup and policy_step_active:
                    if oracle_phi_cached is not None:
                        oracle_phi_for_reward = oracle_phi_cached.detach()
                    else:
                        oracle_phi_for_reward = phi.detach()
                    reward_pred = exp_reward_net(oracle_phi_for_reward)
                    trainee_reward = reward_pred[:, 0]
                    exp_reward_loss_val = nn.functional.smooth_l1_loss(trainee_reward, advantage)
                    loss = loss + exp_reward_weight * exp_reward_loss_val

            scaler.scale(loss / opt_step_every).backward()

            # Single-sided entropy floor: only push entropy up when it drops below the floor.
            if (
                entropy_floor > 0
                and entropy_adjust_rate > 0
                and steps >= entropy_floor_start_step
            ):
                entropy_gap = entropy_floor - entropy.mean().item()
                if entropy_gap > 0:
                    log_entropy_alpha += entropy_adjust_rate * entropy_gap
                    log_entropy_alpha = max(math.log(1e-4), min(math.log(1e-2), log_entropy_alpha))
                    dynamic_entropy_weight = math.exp(log_entropy_alpha)

            with torch.inference_mode():
                stats['important_ratio'] += ratio.mean()
                stats['approx_kl'] += drift['approx_kl']
                stats['clip_fraction'] += drift['clip_fraction']
                stats['ratio_var'] += ratio.var(unbiased=False)
                ratio_max = ratio.max()
                stats['ratio_batch_max_sum'] += ratio_max
                stats['ratio_window_max'] = torch.maximum(stats['ratio_window_max'], ratio_max)
                stats['clipped_ratio_window_max'] = torch.maximum(
                    stats['clipped_ratio_window_max'],
                    clipped_ratio.max(),
                )
                stats['entropy'] += entropy.mean()
                stats['policy_logit_gate_fraction'] += policy_logit_gate_fraction
                stats['policy_update_active'] += torch.tensor(
                    float(policy_step_active),
                    dtype=stats['policy_update_active'].dtype,
                    device=device,
                )
                stats['loss'] += loss
                stats['aux_loss'] += aux_loss_val
                stats['opp_loss'] += opp_loss_val
                stats['danger_loss'] += danger_loss_val
                stats['value_loss'] += value_loss_val
                stats['exp_reward_loss'] += exp_reward_loss_val
                stats['tile_eff_loss'] += tile_eff_loss_val
                stats['furo_regret_loss'] += furo_regret_loss_val
                stats['hand_value_regret_loss'] += hand_value_regret_loss_val
                stats['importance_reject_fraction'] += importance_reject_fraction
                stats['policy_scope_reject_fraction'] += policy_scope_reject_fraction
                stats['replay_is_coverage'] += replay_is_coverage
                stats['replay_is_coverage_min'] = torch.minimum(
                    stats['replay_is_coverage_min'],
                    replay_is_coverage,
                )
                stats['replay_is_missing_fraction'] += replay_is_missing_fraction
                stats['replay_is_missing_fraction_max'] = torch.maximum(
                    stats['replay_is_missing_fraction_max'],
                    replay_is_missing_fraction,
                )
                stats['replay_is_version_gap'] += replay_is_gap_mean
                stats['replay_is_version_gap_max'] = torch.maximum(
                    stats['replay_is_version_gap_max'],
                    replay_is_gap_max,
                )

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                observed_scaler_step(scaler, optimizer, update_clock)
                optimizer.zero_grad(set_to_none=True)
            # Deliberately preserve existing microbatch-based LR/phase clocks.
            # Migrating schedules to successful updates needs separate calibration.
            scheduler.step()
            apply_independent_actor_lr_clock(
                optimizer,
                scheduler,
                config,
                steps=steps,
            )
            if actor_oracle_guiding_continuation_active(config, steps):
                scaled_lrs = scheduler.get_last_lr()
                has_schedule_roles = any(
                    group.get('schedule_role') is not None
                    for group in optimizer.param_groups
                )
                for param_group, base_lr in zip(optimizer.param_groups, scaled_lrs):
                    if has_schedule_roles and param_group.get('schedule_role') != 'actor':
                        continue
                    param_group['lr'] = base_lr * actor_oracle_lr_scale
            pb.update(1)

            if idx % opt_step_every == 0:
                # Check before network publication/evaluation can block shutdown.
                stop_at_successful_optimizer_step_limit()
                stop_after_checkpoint_if_requested()

            if old_policy_update_due(steps, old_update_every):
                refresh_old_policy_snapshot(Old_mortal, Old_policy_net, mortal, policy_net)

            if online and steps % submit_every == 0:
                published_version = publish_current_policy(is_idle=False)
                logging.info('param has been submitted: version=%s', published_version)

            if steps % save_every == 0:
                pb.close()
                aux_metrics = finalize_online_aux_monitor_stats(stats['aux_monitor'])
                for name in ('attempts', 'successes', 'skips', 'legacy_attempt_offset', 'inherited_progress_offset'):
                    writer.add_scalar(f'optimizer_clock/{name}', getattr(update_clock, name), steps)
                writer.add_scalar('optimizer_clock/progress', update_clock.progress, steps)
                logging.info('optimizer update clock: %s', update_clock.state_dict())

                writer.add_scalar('important_ratio/ratio', stats['important_ratio'] / save_every, steps)
                writer.add_scalar('policy_drift/approx_kl', stats['approx_kl'] / save_every, steps)
                writer.add_scalar('policy_drift/clip_fraction', stats['clip_fraction'] / save_every, steps)
                writer.add_scalar('important_ratio/variance', stats['ratio_var'] / save_every, steps)
                writer.add_scalar('important_ratio/max', stats['ratio_batch_max_sum'] / save_every, steps)
                writer.add_scalar('important_ratio/batch_max_mean', stats['ratio_batch_max_sum'] / save_every, steps)
                writer.add_scalar('important_ratio/window_max', stats['ratio_window_max'], steps)
                writer.add_scalar('important_ratio/clipped_window_max', stats['clipped_ratio_window_max'], steps)
                writer.add_scalar('entropy/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('entropy/dynamic_weight', dynamic_entropy_weight, steps)
                writer.add_scalar(
                    'policy/logit_gate_fraction',
                    stats['policy_logit_gate_fraction'] / save_every,
                    steps,
                )
                writer.add_scalar(
                    'policy/update_active',
                    stats['policy_update_active'] / save_every,
                    steps,
                )
                writer.add_scalar('loss', stats['loss'] / save_every, steps)
                writer.add_scalar(
                    'oracle_guiding/actor_keep_prob',
                    actor_oracle_guiding_keep_prob(config, steps),
                    steps,
                )
                writer.add_scalar(
                    'oracle_guiding/continuation',
                    float(actor_oracle_guiding_continuation_active(config, steps)),
                    steps,
                )
                writer.add_scalar(
                    'important_ratio/rejected_fraction',
                    stats['importance_reject_fraction'] / save_every,
                    steps,
                )
                writer.add_scalar(
                    'policy/action_scope_reject_fraction',
                    stats['policy_scope_reject_fraction'] / save_every,
                    steps,
                )
                if online_replay_is:
                    writer.add_scalar(
                        'replay_is/coverage',
                        stats['replay_is_coverage'] / save_every,
                        steps,
                    )
                    writer.add_scalar(
                        'replay_is/coverage_min',
                        stats['replay_is_coverage_min'],
                        steps,
                    )
                    writer.add_scalar(
                        'replay_is/missing_fraction',
                        stats['replay_is_missing_fraction'] / save_every,
                        steps,
                    )
                    writer.add_scalar(
                        'replay_is/missing_fraction_max',
                        stats['replay_is_missing_fraction_max'],
                        steps,
                    )
                    writer.add_scalar(
                        'replay_is/version_gap_mean',
                        stats['replay_is_version_gap'] / save_every,
                        steps,
                    )
                    writer.add_scalar(
                        'replay_is/version_gap_max',
                        stats['replay_is_version_gap_max'],
                        steps,
                    )
                    writer.add_scalar(
                        'replay_is/cached_versions',
                        len(published_policy_history),
                        steps,
                    )
                for metric_name, value in aux_metrics.items():
                    writer.add_scalar(metric_name, value, steps)
                if 'opponent_aux_loss' in aux_metrics:
                    writer.add_scalar('opp_loss', aux_metrics['opponent_aux_loss'], steps)
                if 'danger_aux_loss' in aux_metrics:
                    writer.add_scalar('danger_loss', aux_metrics['danger_aux_loss'], steps)
                if value_enabled:
                    writer.add_scalar('value_loss', stats['value_loss'] / save_every, steps)
                if exp_reward_enabled:
                    writer.add_scalar('exp_reward_loss', stats['exp_reward_loss'] / save_every, steps)
                if tile_eff_weight > 0:
                    writer.add_scalar('tile_eff_loss', stats['tile_eff_loss'] / save_every, steps)
                if furo_regret_weight > 0:
                    writer.add_scalar('furo_regret_loss', stats['furo_regret_loss'] / save_every, steps)
                if hand_value_regret_weight > 0:
                    writer.add_scalar('hand_value_regret_loss', stats['hand_value_regret_loss'] / save_every, steps)
                if not online:
                    pass
                writer.flush()

                stats = init_online_stats(device=device)
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')
                stats_dict = save_reward_target_metadata(steps, writer)
                state = persist_live_training_state(
                    reward_target_metadata_dict=stats_dict,
                )

                if online and steps % submit_every != 0:
                    published_version = publish_current_policy(is_idle=False)
                    logging.info('param has been submitted: version=%s', published_version)

                if periodic_test_play_due(
                    enabled=test_play_eval_enabled,
                    steps=steps,
                    test_every=test_every,
                ):
                    run_test_play_evaluation(
                        stats_dict=stats_dict,
                        save_best_checkpoint=True,
                    )
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

            if online and online_reached_max_steps(config, steps):
                if steps % save_every != 0:
                    persist_live_training_state(reward_target_metadata_dict=None)
                writer.flush()
                published_version = publish_current_policy(is_idle=True)
                logging.info(
                    'reached configured max steps=%s; stopping online training after publishing version=%s',
                    steps,
                    published_version,
                )
                pb.close()
                sys.exit(ONLINE_MAX_STEPS_EXIT_CODE)

        def _unpack_batch(batch):
            """Unpack variable-length data tuple into named fields."""
            it = iter(batch)
            obs = next(it)
            invisible_obs_batch = next(it) if need_oracle_obs else None
            actions = next(it)
            masks = next(it)
            advantage = next(it)
            v_target = next(it) if value_enabled else None
            player_rank = next(it)
            context_meta = next(it) if online_context_meta_enabled else None
            replay_param_version = next(it) if online_replay_is else None
            opp_shanten = next(it) if online_opp_enabled else None
            opp_tenpai = next(it) if online_opp_enabled else None
            danger_valid = next(it) if online_danger_enabled else None
            danger_any = next(it) if online_danger_enabled else None
            danger_value_b = next(it) if online_danger_enabled else None
            danger_player = next(it) if online_danger_enabled else None
            tile_eff_valid = next(it) if online_regret_enabled else None
            tile_eff_delta = next(it) if online_regret_enabled else None
            furo_valid = next(it) if online_regret_enabled else None
            furo_label = next(it) if online_regret_enabled else None
            hand_value_valid = next(it) if online_regret_enabled else None
            hand_value_points = next(it) if online_regret_enabled else None
            return {
                'obs': obs, 'invisible_obs': invisible_obs_batch,
                'actions': actions, 'masks': masks,
                'advantage': advantage, 'v_target': v_target, 'player_rank': player_rank,
                'context_meta': context_meta,
                'replay_param_version': replay_param_version,
                'opp_shanten': opp_shanten, 'opp_tenpai': opp_tenpai,
                'danger_valid': danger_valid, 'danger_any': danger_any,
                'danger_value': danger_value_b, 'danger_player_mask': danger_player,
                'tile_eff_valid': tile_eff_valid, 'tile_eff_delta': tile_eff_delta,
                'furo_valid': furo_valid, 'furo_label': furo_label,
                'hand_value_valid': hand_value_valid, 'hand_value_points': hand_value_points,
            }

        def _call_train_batch(fields, start=None, end=None):
            """Call train_batch from a fields dict, optionally slicing."""
            if start is not None:
                s = {k: (v[start:end] if v is not None else None) for k, v in fields.items()}
            else:
                s = fields
            train_batch(
                s['obs'], s['actions'], s['masks'], s['advantage'], s['player_rank'],
                invisible_obs=s['invisible_obs'],
                replay_param_version=s.get('replay_param_version'),
                context_meta=s.get('context_meta'),
                opp_shanten=s['opp_shanten'], opp_tenpai=s['opp_tenpai'],
                danger_valid=s['danger_valid'], danger_any=s['danger_any'],
                danger_value=s['danger_value'], danger_player_mask=s['danger_player_mask'],
                tile_eff_valid=s['tile_eff_valid'], tile_eff_delta=s['tile_eff_delta'],
                furo_valid=s['furo_valid'], furo_label=s['furo_label'],
                hand_value_valid=s['hand_value_valid'], hand_value_points=s['hand_value_points'],
                v_target=s.get('v_target'),
            )

        remaining_fields = []  # list of field dicts

        for batch in data_loader:
            fields = _unpack_batch(batch)
            bs = fields['obs'].shape[0]
            if bs != batch_size:
                remaining_fields.append(fields)
                remaining_bs += bs
                continue
            _call_train_batch(fields)

        if remaining_bs >= batch_size and remaining_fields:
            # Concatenate all remaining fields
            cat_fields = {}
            for key in remaining_fields[0]:
                tensors = [f[key] for f in remaining_fields if f[key] is not None]
                cat_fields[key] = torch.cat(tensors, dim=0) if tensors else None

            start = 0
            end = batch_size
            while end <= remaining_bs:
                _call_train_batch(cat_fields, start, end)
                start = end
                end += batch_size
        pb.close()

        if online:
            published_version = publish_current_policy(is_idle=True)
            logging.info('param has been submitted: version=%s', published_version)

    def save_reward_target_metadata(steps, writer):
        writer.add_scalar('reward_target/raw_delta_pt_scale', 1.0, steps)
        return dict(reward_target_metadata)


    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break
    

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from mortal.config import config

    apply_oracle_experiment_to_config(config)

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, '-m', 'mortal.online.train_online')
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        if training_stop_requested():
            return
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        code = child.wait()
        if code in (ONLINE_MAX_STEPS_EXIT_CODE, ONLINE_STOP_REQUEST_EXIT_CODE):
            return
        if training_stop_requested():
            return
        if code != 0:
            sys.exit(code)
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

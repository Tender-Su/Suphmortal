import math

import torch

from libriichi.consts import obs_shape, oracle_obs_shape


FIRST_CONV_KEY = 'encoder.net.0.weight'
BRAIN_IS_ORACLE_KEY = 'brain_is_oracle_structure'
DEPLOY_ZERO_ORACLE_KEY = 'deploy_zero_oracle'
DEFAULT_EXTRA_INPUT_INIT_SCALE = 0.02


def checkpoint_brain_is_oracle_structure(state):
    if not isinstance(state, dict):
        return False

    if BRAIN_IS_ORACLE_KEY in state:
        return bool(state[BRAIN_IS_ORACLE_KEY])

    mortal_state = state.get('mortal')
    config = state.get('config', {})
    if not isinstance(mortal_state, dict) or not isinstance(config, dict):
        return False

    control_cfg = config.get('control', {})
    if not isinstance(control_cfg, dict):
        return False

    version = control_cfg.get('version')
    first_conv = mortal_state.get(FIRST_CONV_KEY)
    if version is None or not torch.is_tensor(first_conv) or first_conv.ndim != 3:
        oracle_cfg = config.get('oracle_guiding', {})
        return bool(oracle_cfg.get('actor_enabled', False)) if isinstance(oracle_cfg, dict) else False

    visible_channels = int(obs_shape(version)[0])
    oracle_channels = int(oracle_obs_shape(version)[0])
    total_oracle_channels = visible_channels + oracle_channels
    input_channels = int(first_conv.shape[1])

    if input_channels == total_oracle_channels:
        return True
    if input_channels == visible_channels:
        return False

    oracle_cfg = config.get('oracle_guiding', {})
    return bool(oracle_cfg.get('actor_enabled', False)) if isinstance(oracle_cfg, dict) else False


def _init_extra_input_slice_(target_tensor, source_channels, *, scale):
    extra_slice = target_tensor[:, source_channels:, :]
    if extra_slice.numel() == 0:
        return False
    if scale <= 0:
        extra_slice.zero_()
        return False

    torch.nn.init.kaiming_uniform_(extra_slice, a=math.sqrt(5))
    extra_slice.mul_(float(scale))
    return True


def load_brain_state_strict(target_brain, source_state_dict, *, checkpoint_name='checkpoint'):
    """Load an explicitly selected brain checkpoint without shape bridging."""
    if not isinstance(source_state_dict, dict):
        raise TypeError(f'{checkpoint_name} brain state must be a state_dict')
    try:
        target_brain.load_state_dict(source_state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f'{checkpoint_name} brain state does not match the requested model structure'
        ) from exc
    return {
        'loaded_keys': tuple(source_state_dict.keys()),
        'skipped_keys': (),
        'expanded_input_keys': (),
        'extra_input_init_scale': 0.0,
        'direct': True,
        'strict': True,
    }


def load_brain_state_with_input_bridge(
    target_brain,
    source_state_dict,
    *,
    extra_input_init_scale=DEFAULT_EXTRA_INPUT_INIT_SCALE,
):
    target_state = target_brain.state_dict()
    loaded_keys = []
    skipped_keys = []
    expanded_input_keys = []

    with torch.no_grad():
        for key, target_tensor in target_state.items():
            if key not in source_state_dict:
                skipped_keys.append(key)
                continue

            source_tensor = source_state_dict[key]
            if source_tensor.shape == target_tensor.shape:
                target_tensor.copy_(source_tensor)
                loaded_keys.append(key)
                continue

            if (
                key == FIRST_CONV_KEY
                and source_tensor.ndim == 3
                and target_tensor.ndim == 3
                and source_tensor.shape[0] == target_tensor.shape[0]
                and source_tensor.shape[2] == target_tensor.shape[2]
            ):
                if source_tensor.shape[1] <= target_tensor.shape[1]:
                    target_tensor.zero_()
                    target_tensor[:, :source_tensor.shape[1], :].copy_(source_tensor)
                    if _init_extra_input_slice_(
                        target_tensor,
                        source_tensor.shape[1],
                        scale=extra_input_init_scale,
                    ):
                        expanded_input_keys.append(key)
                    loaded_keys.append(key)
                    continue
                if source_tensor.shape[1] >= target_tensor.shape[1]:
                    target_tensor.copy_(source_tensor[:, :target_tensor.shape[1], :])
                    loaded_keys.append(key)
                    continue

            skipped_keys.append(key)

    target_brain.load_state_dict(target_state)
    return {
        'loaded_keys': loaded_keys,
        'skipped_keys': skipped_keys,
        'expanded_input_keys': expanded_input_keys,
        'extra_input_init_scale': float(extra_input_init_scale),
    }

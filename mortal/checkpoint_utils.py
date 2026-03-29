import torch


FIRST_CONV_KEY = 'encoder.net.0.weight'


def load_brain_state_with_input_bridge(target_brain, source_state_dict):
    target_state = target_brain.state_dict()
    loaded_keys = []
    skipped_keys = []

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
    }


def make_normal_checkpoint_from_oracle_checkpoint(oracle_checkpoint, normal_brain):
    # Avoid deep-copying the full live training checkpoint here. On CUDA runs
    # that would briefly duplicate optimizer/model state on GPU during export.
    checkpoint = dict(oracle_checkpoint)
    bridge_info = load_brain_state_with_input_bridge(normal_brain, checkpoint['mortal'])
    checkpoint['mortal'] = normal_brain.state_dict()
    checkpoint['bridge_info'] = bridge_info
    return checkpoint

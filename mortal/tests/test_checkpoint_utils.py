import sys
import unittest
from pathlib import Path

import torch
from libriichi.consts import obs_shape, oracle_obs_shape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mortal.core.checkpoint_utils import (
    BRAIN_IS_ORACLE_KEY,
    DEPLOY_ZERO_ORACLE_KEY,
    FIRST_CONV_KEY,
    checkpoint_brain_is_oracle_structure,
    load_brain_state_strict,
    load_brain_state_with_input_bridge,
)


class FakeBrain:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        for key, target_tensor in self._state.items():
            source_tensor = state.get(key)
            if source_tensor is None or source_tensor.shape != target_tensor.shape:
                raise RuntimeError(f'size mismatch for {key}')
        self._state = state


class CheckpointBridgeTests(unittest.TestCase):
    def test_input_bridge_expansion_uses_small_random_init_for_extra_slice(self):
        torch.manual_seed(0)
        source_mortal = {
            FIRST_CONV_KEY: torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        }
        oracle_brain = FakeBrain(
            {
                FIRST_CONV_KEY: torch.zeros(2, 4, 3, dtype=torch.float32),
            }
        )

        bridge_info = load_brain_state_with_input_bridge(
            oracle_brain,
            source_mortal,
            extra_input_init_scale=0.02,
        )

        loaded = oracle_brain.state_dict()[FIRST_CONV_KEY]
        self.assertTrue(torch.equal(loaded[:, :2, :], source_mortal[FIRST_CONV_KEY]))
        self.assertEqual([FIRST_CONV_KEY], bridge_info['expanded_input_keys'])
        self.assertGreater(float(loaded[:, 2:, :].abs().max()), 0.0)
        self.assertLess(float(loaded[:, 2:, :].abs().max()), 0.05)

    def test_strict_brain_load_rejects_input_shape_mismatch_without_bridge(self):
        source_mortal = {
            FIRST_CONV_KEY: torch.ones(2, 2, 3, dtype=torch.float32),
        }
        oracle_brain = FakeBrain(
            {
                FIRST_CONV_KEY: torch.zeros(2, 4, 3, dtype=torch.float32),
            }
        )

        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            load_brain_state_strict(
                oracle_brain,
                source_mortal,
                checkpoint_name='value.oracle_critic_state_file',
            )

        self.assertTrue(torch.equal(
            oracle_brain.state_dict()[FIRST_CONV_KEY],
            torch.zeros(2, 4, 3, dtype=torch.float32),
        ))

    def test_checkpoint_brain_is_oracle_structure_prefers_metadata(self):
        self.assertTrue(checkpoint_brain_is_oracle_structure({BRAIN_IS_ORACLE_KEY: True}))
        self.assertFalse(checkpoint_brain_is_oracle_structure({BRAIN_IS_ORACLE_KEY: False}))

    def test_checkpoint_brain_is_oracle_structure_infers_from_first_conv_shape(self):
        version = 4
        visible_channels = obs_shape(version)[0]
        oracle_channels = oracle_obs_shape(version)[0]
        state = {
            'config': {
                'control': {'version': version},
                'oracle_guiding': {'actor_enabled': False},
            },
            'mortal': {
                FIRST_CONV_KEY: torch.zeros(2, visible_channels + oracle_channels, 3, dtype=torch.float32),
            },
            DEPLOY_ZERO_ORACLE_KEY: True,
        }
        self.assertTrue(checkpoint_brain_is_oracle_structure(state))


if __name__ == '__main__':
    unittest.main()

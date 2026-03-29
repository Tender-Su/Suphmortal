import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_utils import FIRST_CONV_KEY, make_normal_checkpoint_from_oracle_checkpoint


class NoDeepCopy:
    def __deepcopy__(self, memo):
        raise AssertionError('unexpected deepcopy')


class FakeBrain:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state


class MakeNormalCheckpointTests(unittest.TestCase):
    def test_make_normal_checkpoint_does_not_deepcopy_unrelated_training_state(self):
        source_mortal = {
            FIRST_CONV_KEY: torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        }
        normal_brain = FakeBrain(
            {
                FIRST_CONV_KEY: torch.zeros(2, 1, 3, dtype=torch.float32),
            }
        )
        optimizer_state = NoDeepCopy()
        oracle_checkpoint = {
            'mortal': source_mortal,
            'optimizer': optimizer_state,
            'steps': 123,
        }

        export_checkpoint = make_normal_checkpoint_from_oracle_checkpoint(
            oracle_checkpoint,
            normal_brain,
        )

        self.assertIsNot(export_checkpoint, oracle_checkpoint)
        self.assertIs(export_checkpoint['optimizer'], optimizer_state)
        self.assertEqual(export_checkpoint['steps'], 123)
        self.assertTrue(torch.equal(oracle_checkpoint['mortal'][FIRST_CONV_KEY], source_mortal[FIRST_CONV_KEY]))
        self.assertEqual(
            export_checkpoint['bridge_info']['loaded_keys'],
            [FIRST_CONV_KEY],
        )
        self.assertEqual(export_checkpoint['bridge_info']['skipped_keys'], [])
        self.assertTrue(
            torch.equal(
                export_checkpoint['mortal'][FIRST_CONV_KEY],
                source_mortal[FIRST_CONV_KEY][:, :1, :],
            )
        )


if __name__ == '__main__':
    unittest.main()

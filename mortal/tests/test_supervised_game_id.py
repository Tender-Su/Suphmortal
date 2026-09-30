import unittest
from unittest.mock import patch

import numpy as np

from mortal.data.dataloader import (
    SupervisedFileDatasetsIter,
    stable_source_game_id,
)


class _FakeGrp:
    def take_rank_by_player(self):
        return np.array([0, 1, 2, 3], dtype=np.int64)


class _FakeGame:
    def take_obs_batch(self):
        return np.zeros((2, 1), dtype=np.float32)

    def take_actions_batch(self):
        return np.array([1, 2], dtype=np.int64)

    def take_masks_batch(self):
        return np.ones((2, 3), dtype=np.bool_)

    def take_context_meta_batch(self):
        return np.zeros((2, 1), dtype=np.int64)

    def take_grp(self):
        return _FakeGrp()

    def take_player_id(self):
        return 1


class SupervisedGameIdTests(unittest.TestCase):
    def test_stable_source_game_id_is_deterministic_and_path_sensitive(self):
        self.assertEqual(
            stable_source_game_id('game-a.json'),
            stable_source_game_id('game-a.json'),
        )
        self.assertNotEqual(
            stable_source_game_id('game-a.json'),
            stable_source_game_id('game-b.json'),
        )

    def test_validation_dataset_can_append_source_game_id(self):
        dataset = SupervisedFileDatasetsIter(
            version=4,
            file_list=['game-a.json'],
            emit_opponent_state_labels=False,
            track_danger_labels=False,
            emit_game_id=True,
            shuffle_files=False,
        )
        dataset.buffer = []
        dataset.loader = object()

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('game-a.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['game-a.json'])

        expected = stable_source_game_id('game-a.json')
        self.assertEqual(2, len(dataset.buffer))
        self.assertTrue(all(len(row) == 6 for row in dataset.buffer))
        self.assertTrue(all(int(row[-1]) == expected for row in dataset.buffer))


if __name__ == '__main__':
    unittest.main()

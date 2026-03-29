import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import SupervisedFileDatasetsIter


class SupervisedFileDatasetsIterTests(unittest.TestCase):
    def test_danger_only_mode_does_not_request_opponent_label_emission(self):
        with patch('dataloader.GameplayLoader') as loader_cls:
            dataset = SupervisedFileDatasetsIter(
                version=4,
                file_list=[],
                emit_opponent_state_labels=False,
                track_danger_labels=True,
                shuffle_files=False,
            )
            list(dataset.load_files(False))

        kwargs = loader_cls.call_args.kwargs
        self.assertFalse(kwargs['track_opponent_states'])
        self.assertTrue(kwargs['track_danger_labels'])

    def test_opponent_aux_mode_still_requests_opponent_labels(self):
        with patch('dataloader.GameplayLoader') as loader_cls:
            dataset = SupervisedFileDatasetsIter(
                version=4,
                file_list=[],
                emit_opponent_state_labels=True,
                track_danger_labels=False,
                shuffle_files=False,
            )
            list(dataset.load_files(False))

        kwargs = loader_cls.call_args.kwargs
        self.assertTrue(kwargs['track_opponent_states'])
        self.assertFalse(kwargs['track_danger_labels'])


if __name__ == '__main__':
    unittest.main()

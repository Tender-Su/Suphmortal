import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mortal.eval.summarize_duplicate_logs import load_challenger_game


class LoadChallengerGameTest(unittest.TestCase):
    def test_resolves_unique_challenger_seat_and_seed(self):
        events = [
            {
                'type': 'start_game',
                'names': ['champion', 'challenger', 'champion', 'champion'],
                'seed': [123, 456],
            },
            {'type': 'end_game'},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / 'game.json.gz'
            with gzip.open(log_path, 'wt', encoding='utf-8') as log_file:
                for event in events:
                    log_file.write(json.dumps(event) + '\n')
            with mock.patch(
                'mortal.eval.summarize_duplicate_logs.Stat.from_log',
                return_value=mock.Mock(avg_rank=3.0),
            ):
                game = load_challenger_game(log_path, 'challenger')

        self.assertEqual(123, game['seed'])
        self.assertEqual(1, game['challenger_seat'])
        self.assertEqual(3, game['challenger_rank'])


if __name__ == '__main__':
    unittest.main()

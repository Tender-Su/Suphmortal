import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.eval.player as player


class TrainBaselinePoolTests(unittest.TestCase):
    def test_build_train_baseline_pool_entries_keeps_legacy_single_baseline(self):
        entries = player.build_train_baseline_pool_entries({
            'state_file': 'C:/tmp/baseline.pth',
        })

        self.assertEqual(1, len(entries))
        self.assertEqual(Path('C:/tmp/baseline.pth').resolve(), Path(entries[0]['state_file']))
        self.assertEqual(1.0, entries[0]['weight'])
        self.assertEqual(('legacy',), entries[0]['labels'])

    def test_build_train_baseline_pool_entries_uses_requested_mixture(self):
        entries = player.build_train_baseline_pool_entries({
            'champion_state_file': 'C:/tmp/champion.pth',
            'anchor_state_file': 'C:/tmp/anchor.pth',
            'history_state_files': [
                'C:/tmp/h1.pth',
                'C:/tmp/h2.pth',
            ],
            'champion_prob': 0.50,
            'anchor_prob': 0.25,
            'history_prob': 0.25,
        })

        weights = {
            Path(entry['state_file']).name: entry['weight']
            for entry in entries
        }
        self.assertAlmostEqual(0.50, weights['champion.pth'])
        self.assertAlmostEqual(0.25, weights['anchor.pth'])
        self.assertAlmostEqual(0.125, weights['h1.pth'])
        self.assertAlmostEqual(0.125, weights['h2.pth'])

    def test_build_train_baseline_pool_entries_merges_duplicate_paths(self):
        entries = player.build_train_baseline_pool_entries({
            'state_file': 'C:/tmp/baseline.pth',
            'champion_state_file': 'C:/tmp/shared.pth',
            'anchor_state_file': 'C:/tmp/shared.pth',
            'history_state_files': [
                'C:/tmp/shared.pth',
                'C:/tmp/h1.pth',
                'C:/tmp/h1.pth',
            ],
            'champion_prob': 0.50,
            'anchor_prob': 0.25,
            'history_prob': 0.25,
        })

        weights = {
            Path(entry['state_file']).name: entry['weight']
            for entry in entries
        }
        self.assertAlmostEqual(0.875, weights['shared.pth'])
        self.assertAlmostEqual(0.125, weights['h1.pth'])


if __name__ == '__main__':
    unittest.main()

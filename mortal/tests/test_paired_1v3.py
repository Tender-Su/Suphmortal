import unittest

from mortal.eval.paired_1v3 import compare_games, duplicate_sets


def games(ranks, *, key=41):
    return [
        {'seed': seed, 'seed_key': key, 'challenger_seat': seat, 'challenger_rank': rank}
        for seed, group in enumerate(ranks)
        for seat, rank in enumerate(group)
    ]


class PairedOneVsThreeTests(unittest.TestCase):
    def test_identical_policies_have_exactly_zero_paired_interval(self):
        sample = games([[1, 2, 3, 4], [4, 4, 1, 1]])
        result = compare_games(sample, list(reversed(sample)), replicates=1000)
        self.assertEqual(result['delta_pt_candidate_minus_reference']['ci95'], [0.0, 0.0])
        self.assertEqual(result['changed_game_outcomes'], 0)

    def test_four_correlated_seats_are_one_cluster(self):
        candidate = games([[1] * 4, [4] * 4])
        reference = games([[2] * 4, [3] * 4])
        result = compare_games(candidate, reference, replicates=1000)
        self.assertEqual(result['independent_seed_sets'], 2)
        self.assertEqual(result['games_per_arm'], 8)
        self.assertAlmostEqual(result['delta_pt_candidate_minus_reference']['mean'], -45.0)
        self.assertAlmostEqual(result['delta_pt_candidate_minus_reference']['cluster_se'], 90.0)

    def test_seed_key_prevents_collision_across_rounds(self):
        sample = games([[1, 2, 3, 4]], key=1) + games([[4, 3, 2, 1]], key=2)
        self.assertEqual(len(duplicate_sets(sample)), 2)

    def test_rejects_missing_rotation_duplicate_and_missing_key(self):
        sample = games([[1, 2, 3, 4], [4, 3, 2, 1]])
        for invalid in (sample[:-1], sample + sample[:1], [{**g, 'seed_key': None} for g in sample]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    duplicate_sets(invalid)

    def test_rejects_unpaired_seed_sets(self):
        sample = games([[1, 2, 3, 4], [4, 3, 2, 1]])
        with self.assertRaisesRegex(ValueError, 'exactly the same'):
            compare_games(sample, [{**g, 'seed_key': 42} for g in sample])


if __name__ == '__main__':
    unittest.main()

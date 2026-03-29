import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_selection_heuristics as audit


class AnalyzeSelectionHeuristicsTests(unittest.TestCase):
    def test_round_up_step_uses_half_milli_precision(self):
        self.assertAlmostEqual(0.0030, audit.round_up_step(0.0028734))
        self.assertAlmostEqual(0.0035, audit.round_up_step(0.0034440))

    def test_build_scenario_factor_grid_keeps_dense_range_and_extra_points(self):
        grid = audit.build_scenario_factor_grid(0.0, 0.01, 0.005, extra_points=[0.5, 0.005])
        self.assertEqual([0.0, 0.005, 0.01, 0.5], grid)

    def test_collect_noise_stats_rounds_policy_and_old_regression_separately(self):
        rounds = [
            {
                "payload": {
                    "ranking": [
                        {
                            "seed_summaries": [
                                {
                                    "valid": True,
                                    "recent_policy_loss": 0.5000,
                                    "old_regression_policy_loss": 0.6000,
                                    "full_recent_loss": 0.7000,
                                },
                                {
                                    "valid": True,
                                    "recent_policy_loss": 0.5029,
                                    "old_regression_policy_loss": 0.6033,
                                    "full_recent_loss": 0.7028,
                                },
                            ]
                        },
                        {
                            "seed_summaries": [
                                {
                                    "valid": True,
                                    "recent_policy_loss": 0.5100,
                                    "old_regression_policy_loss": 0.6100,
                                    "full_recent_loss": 0.7100,
                                },
                                {
                                    "valid": True,
                                    "recent_policy_loss": 0.5127,
                                    "old_regression_policy_loss": 0.6134,
                                    "full_recent_loss": 0.7129,
                                },
                            ]
                        },
                    ]
                }
            }
        ]

        stats = audit.collect_noise_stats(rounds)

        self.assertAlmostEqual(0.0030, stats["policy"]["suggested_epsilon_roundup_p90"])
        self.assertAlmostEqual(0.0035, stats["old_regression"]["suggested_epsilon_roundup_p90"])

    def test_selection_sort_key_uses_configurable_scenario_factor(self):
        seed = {
            "action_quality_score": -0.20,
            "scenario_quality_score": -0.10,
            "recent_policy_loss": 0.50,
            "old_regression_policy_loss": 0.60,
        }
        zero = audit.selection_sort_key(seed, 0.0)[0]
        half = audit.selection_sort_key(seed, 0.5)[0]
        self.assertAlmostEqual(-0.20, zero)
        self.assertAlmostEqual(-0.25, half)

    def test_recommend_scenario_factor_uses_midpoint_of_best_plateau(self):
        recommendation = audit.recommend_scenario_factor(
            [
                {
                    "scenario_factor": 0.199,
                    "aggregate_winner_match_rate": 0.375,
                    "aggregate_winner_total": 8,
                    "pairwise_agreement": 0.459,
                },
                {
                    "scenario_factor": 0.205,
                    "aggregate_winner_match_rate": 0.375,
                    "aggregate_winner_total": 8,
                    "pairwise_agreement": 0.456,
                },
                {
                    "scenario_factor": 0.21,
                    "aggregate_winner_match_rate": 0.375,
                    "aggregate_winner_total": 8,
                    "pairwise_agreement": 0.456,
                },
                {
                    "scenario_factor": 0.5,
                    "aggregate_winner_match_rate": 0.5,
                    "aggregate_winner_total": 8,
                    "pairwise_agreement": 0.381,
                },
            ],
            current_factor=0.20,
            search_min=0.0,
            search_max=0.25,
        )
        self.assertAlmostEqual(0.20, recommendation["recommended_value"])
        self.assertEqual([0.199, 0.21], recommendation["best_dense_band"])
        self.assertAlmostEqual(0.381, recommendation["legacy_pairwise_agreement_at_0_5"])


if __name__ == "__main__":
    unittest.main()

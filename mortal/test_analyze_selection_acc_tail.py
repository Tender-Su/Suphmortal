import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_selection_acc_tail as audit


class AnalyzeSelectionAccTailTests(unittest.TestCase):
    def test_summarize_rounds_reports_no_tail_usage_when_selection_scores_are_unique(self):
        rounds = [
            {
                "path": Path("sample.json"),
                "payload": {
                    "round_name": "p1_solo_round",
                    "ranking": [
                        {
                            "arm_name": "arm_a",
                            "valid": True,
                            "eligible": True,
                            "selection_quality_score": -0.10,
                            "comparison_recent_loss": 0.50,
                            "comparison_old_regression_loss": 0.60,
                            "sort_key": [1, 1, -0.10, 0.70, -0.50, -0.60, -0.50],
                            "candidate_meta": {"protocol_arm": "proto"},
                        },
                        {
                            "arm_name": "arm_b",
                            "valid": True,
                            "eligible": True,
                            "selection_quality_score": -0.11,
                            "comparison_recent_loss": 0.49,
                            "comparison_old_regression_loss": 0.59,
                            "sort_key": [1, 1, -0.11, 0.90, -0.49, -0.59, -0.49],
                            "candidate_meta": {"protocol_arm": "proto"},
                        },
                    ],
                },
            }
        ]

        summary = audit.summarize_rounds(rounds)

        self.assertEqual(0, summary["top1_selection_ties"])
        self.assertEqual(0, summary["selection_equal_pair_count"])
        self.assertEqual(0, summary["winner_changed_without_acc_tail"])
        self.assertEqual(0, summary["order_changed_without_acc_tail"])
        self.assertAlmostEqual(0.01, summary["min_top12_gap"])

    def test_summarize_rounds_detects_when_acc_tail_changes_winner(self):
        rounds = [
            {
                "path": Path("sample.json"),
                "payload": {
                    "round_name": "p1_solo_round",
                    "ranking": [
                        {
                            "arm_name": "arm_a",
                            "valid": True,
                            "eligible": True,
                            "selection_quality_score": -0.10,
                            "comparison_recent_loss": 0.51,
                            "comparison_old_regression_loss": 0.60,
                            "sort_key": [1, 1, -0.10, 0.90, -0.51, -0.60, -0.51],
                            "candidate_meta": {"protocol_arm": "proto"},
                        },
                        {
                            "arm_name": "arm_b",
                            "valid": True,
                            "eligible": True,
                            "selection_quality_score": -0.10,
                            "comparison_recent_loss": 0.50,
                            "comparison_old_regression_loss": 0.60,
                            "sort_key": [1, 1, -0.10, 0.80, -0.50, -0.60, -0.50],
                            "candidate_meta": {"protocol_arm": "proto"},
                        },
                    ],
                },
            }
        ]

        summary = audit.summarize_rounds(rounds)

        self.assertEqual(1, summary["top1_selection_ties"])
        self.assertEqual(1, summary["groups_with_any_selection_equal_pairs"])
        self.assertEqual(1, summary["selection_equal_pair_count"])
        self.assertEqual(1, summary["winner_changed_without_acc_tail"])
        self.assertEqual(1, summary["order_changed_without_acc_tail"])


if __name__ == "__main__":
    unittest.main()

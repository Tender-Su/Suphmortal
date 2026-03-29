import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_aux_heuristics as audit


class AnalyzeAuxHeuristicsTests(unittest.TestCase):
    def test_parse_round_metrics_keeps_same_arm_across_main_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fidelity_dir = Path(tmpdir)
            shared_arm_name = "proto_a__all_three__budget_030"

            (fidelity_dir / "p1_protocol_decide_round.json").write_text(
                json.dumps(
                    {
                        "round_name": "p1_protocol_decide_round",
                        "ranking": [
                            {
                                "arm_name": shared_arm_name,
                                "full_recent_metrics": {
                                    "opponent_aux_loss": 0.11,
                                    "opponent_shanten_loss": 0.12,
                                    "opponent_tenpai_loss": 0.13,
                                    "opponent_shanten_macro_acc": 0.71,
                                    "opponent_tenpai_macro_acc": 0.72,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (fidelity_dir / "p1_winner_refine_round.json").write_text(
                json.dumps(
                    {
                        "round_name": "p1_winner_refine_round",
                        "ranking": [
                            {
                                "arm_name": shared_arm_name,
                                "full_recent_metrics": {
                                    "opponent_aux_loss": 0.21,
                                    "opponent_shanten_loss": 0.22,
                                    "opponent_tenpai_loss": 0.23,
                                    "opponent_shanten_macro_acc": 0.81,
                                    "opponent_tenpai_macro_acc": 0.82,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (fidelity_dir / "p1_protocol_decide_round__s1505.json").write_text(
                json.dumps(
                    {
                        "round_name": "p1_protocol_decide_round__s1505",
                        "ranking": [
                            {
                                "arm_name": shared_arm_name,
                                "full_recent_metrics": {
                                    "opponent_aux_loss": 9.99,
                                    "opponent_shanten_loss": 9.99,
                                    "opponent_tenpai_loss": 9.99,
                                    "opponent_shanten_macro_acc": 0.0,
                                    "opponent_tenpai_macro_acc": 0.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            metrics = audit.parse_round_metrics(fidelity_dir)

            self.assertEqual(2, len(metrics.opponent_runs))
            self.assertEqual(
                ["p1_protocol_decide_round", "p1_winner_refine_round"],
                [row["round_name"] for row in metrics.opponent_runs],
            )
            self.assertEqual(
                [0.11, 0.21],
                [row["opponent_aux_loss"] for row in metrics.opponent_runs],
            )


if __name__ == "__main__":
    unittest.main()

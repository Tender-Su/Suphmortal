import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_oracle_critic_checkpoints import (
    game_subset_mask,
    normalize_game_subset,
    paired_cluster_summary,
    parse_checkpoint_specs,
    parts_to_row_losses,
    resolve_eval_state_fold_count,
    resolve_split_provenance,
    summarize_regression_slices,
    summarize_target_tensor,
)


class PairedClusterSummaryTest(unittest.TestCase):
    def test_accepts_single_checkpoint_for_absolute_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "finalist.pth"
            checkpoint.touch()

            specs = parse_checkpoint_specs([f"finalist={checkpoint}"])

        self.assertEqual([("finalist", checkpoint.resolve())], specs)

    def test_rejects_empty_checkpoint_list(self):
        with self.assertRaisesRegex(ValueError, "at least one checkpoint"):
            parse_checkpoint_specs([])

    def test_partitions_signed_game_ids_deterministically(self):
        modulus, remainders = normalize_game_subset(5, [3, 0, 3, 1, 2])
        mask = game_subset_mask(
            torch.tensor([-1, 0, 1, 2, 3, 4, 5]),
            modulus=modulus,
            remainders=remainders,
        )

        self.assertEqual((5, (0, 1, 2, 3)), (modulus, remainders))
        torch.testing.assert_close(
            mask,
            torch.tensor([False, True, True, True, True, False, True]),
        )

    def test_rejects_invalid_game_partition(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            normalize_game_subset(0, [])
        with self.assertRaisesRegex(ValueError, "must be in"):
            normalize_game_subset(5, [5])

    def test_resolves_eval_state_fold_override(self):
        cfg = {"val_state_fold_count": 128}

        self.assertEqual(128, resolve_eval_state_fold_count(cfg, None))
        self.assertEqual(32, resolve_eval_state_fold_count(cfg, 32))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_eval_state_fold_count(cfg, 0)

    def test_split_provenance_requires_explicit_reason_for_mismatch(self):
        saved = {"sha256": "old"}
        evaluation = {"sha256": "new"}

        with self.assertRaisesRegex(ValueError, "eval-split-override-reason"):
            resolve_split_provenance(saved, evaluation, split_name="dev")

    def test_split_provenance_records_audited_ood_override(self):
        saved = {"sha256": "old"}
        evaluation = {"sha256": "new"}

        result = resolve_split_provenance(
            saved,
            evaluation,
            split_name="dev",
            override_reason="actor replay sid0 selection",
        )

        self.assertEqual("explicit_ood_override", result["mode"])
        self.assertTrue(result["override_used"])
        self.assertEqual(saved, result["checkpoint_split"])
        self.assertEqual(evaluation, result["evaluation_split"])
        self.assertEqual("actor replay sid0 selection", result["override_reason"])

    def test_split_provenance_keeps_matching_guard_strict(self):
        split = {"sha256": "same"}

        result = resolve_split_provenance(
            split,
            split_name="dev",
            evaluation_split=split,
            override_reason="unused reason",
        )

        self.assertEqual("matched", result["mode"])
        self.assertFalse(result["override_used"])
        self.assertEqual("", result["override_reason"])

    def test_reports_state_and_game_balanced_differences(self):
        result = paired_cluster_summary(
            torch.tensor([3.0, 1.0, 5.0]),
            torch.tensor([1.0, 1.0, 2.0]),
            torch.tensor([10, 10, 20]),
        )

        self.assertAlmostEqual(result["mean"], 5.0 / 3.0)
        self.assertAlmostEqual(result["game_balanced_mean"], 2.0)
        self.assertEqual(result["num_samples"], 3)
        self.assertEqual(result["num_games"], 2)
        self.assertGreater(result["cluster_se"], 0.0)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "same one-dimensional shape"):
            paired_cluster_summary(
                torch.tensor([1.0]),
                torch.tensor([1.0, 2.0]),
                torch.tensor([1]),
            )

    def test_can_select_primary_value_output(self):
        parts = [{
            "pred": torch.tensor([[2.0, 5.0], [4.0, 1.0]]),
            "target": torch.tensor([[1.0, 1.0], [2.0, 1.0]]),
            "game_id": torch.tensor([10, 20]),
        }]

        overall, game_id = parts_to_row_losses(parts)
        primary, _ = parts_to_row_losses(parts, output_index=0)

        torch.testing.assert_close(overall, torch.tensor([8.5, 2.0], dtype=torch.float64))
        torch.testing.assert_close(primary, torch.tensor([1.0, 4.0], dtype=torch.float64))
        torch.testing.assert_close(game_id, torch.tensor([10, 20]))

    def test_summarizes_target_support_and_zero_sum_contract(self):
        target = torch.tensor([
            [3.0, 1.0, -1.0, -3.0],
            [1.0, -1.0, 3.0, -3.0],
        ])

        summary = summarize_target_tensor(target)

        self.assertEqual(0.0, summary["zero_sum_max_abs"])
        self.assertEqual(-3.0, summary["all"]["quantiles"]["q0"])
        self.assertEqual(3.0, summary["all"]["quantiles"]["q1"])
        self.assertAlmostEqual(5.0 ** 0.5, summary["all"]["std"])
        self.assertEqual(0.0, summary["all"]["exact_zero_fraction"])
        self.assertEqual(4, summary["all"]["unique_count"])

    def test_summarizes_zero_nonzero_and_tail_errors(self):
        summary = summarize_regression_slices(
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, -2.0, 4.0, -5.0]),
        )

        self.assertEqual(5, summary["all"]["count"])
        self.assertEqual(1, summary["exact_zero"]["count"])
        self.assertEqual(4, summary["nonzero"]["count"])
        self.assertEqual(3, summary["abs_ge_2"]["count"])
        self.assertEqual(2, summary["abs_ge_4"]["count"])
        self.assertAlmostEqual(1.0, summary["exact_zero"]["loss"])
        self.assertAlmostEqual(22.5, summary["abs_ge_4"]["loss"])


if __name__ == "__main__":
    unittest.main()

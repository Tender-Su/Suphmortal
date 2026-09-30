import tempfile
import unittest
from pathlib import Path

from scripts.build_oracle_temporal_indexes import (
    build_temporal_splits,
    discover_month_files,
    validate_month,
)


class OracleTemporalIndexTests(unittest.TestCase):
    def test_builds_disjoint_chronological_splits_with_seeded_train_order(self):
        by_month = {
            "202510": ["old_b.json", "old_a.json"],
            "202511": ["recent.json"],
            "202512": ["dev.json"],
            "202601": ["test.json"],
        }

        train_a, dev_a, test_a = build_temporal_splits(
            by_month,
            dev_month="202512",
            test_month="202601",
            seed=7,
        )
        train_b, dev_b, test_b = build_temporal_splits(
            by_month,
            dev_month="202512",
            test_month="202601",
            seed=7,
        )

        self.assertEqual(train_a, train_b)
        self.assertCountEqual(train_a, ["old_b.json", "old_a.json", "recent.json"])
        self.assertEqual(["dev.json"], dev_a)
        self.assertEqual(["test.json"], test_a)
        self.assertEqual(dev_a, dev_b)
        self.assertEqual(test_a, test_b)
        self.assertFalse(set(train_a) & set(dev_a))
        self.assertFalse(set(train_a) & set(test_a))

    def test_rejects_unassigned_month_after_dev_cutoff(self):
        with self.assertRaisesRegex(ValueError, "assigned explicitly"):
            build_temporal_splits(
                {
                    "202511": ["train.json"],
                    "202512": ["dev.json"],
                    "202601": ["test.json"],
                    "202602": ["future.json"],
                },
                dev_month="202512",
                test_month="202601",
                seed=1,
            )

    def test_discovers_month_directories_and_checks_filename_month(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            month_dir = root / "2025" / "202512"
            month_dir.mkdir(parents=True)
            source = month_dir / "20251201-game.json"
            source.write_text("{}", encoding="utf-8")

            discovered = discover_month_files(root)

            self.assertEqual([str(source.resolve())], discovered["202512"])

    def test_validate_month_rejects_invalid_calendar_month(self):
        with self.assertRaises(ValueError):
            validate_month("202513", name="month")


if __name__ == "__main__":
    unittest.main()

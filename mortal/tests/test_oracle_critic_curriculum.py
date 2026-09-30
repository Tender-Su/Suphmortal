import unittest

import torch

from mortal.online.pretrain_oracle_critic import (
    resolve_convergence_config,
    validate_convergence_schedule,
)
from mortal.research.oracle_critic_curriculum import (
    build_phase_file_lists,
    checkpoint_component_hashes,
    classify_source_files,
    migrate_checkpoint_for_phase,
)
from scripts.run_oracle_critic_curriculum import (
    INTERIM_UNIFORM_SANITY_STEP,
    PHASE_LENGTHS,
    PHASE_TARGETS,
)


BOUNDARIES = {
    "mid_start": "202112",
    "old_regression_start": "202212",
    "recent_24_start": "202312",
    "recent_12_start": "202412",
    "train_end": "202511",
}


def source(month: str, suffix: str = "a") -> str:
    return rf"D:\mahjong_data\dataset_json\{month[:4]}\{month}\{month}_{suffix}.json"


def fake_checkpoint() -> dict:
    return {
        "oracle_brain": {"weight": torch.arange(4, dtype=torch.float32)},
        "value_net": {"weight": torch.arange(3, dtype=torch.float32)},
        "optimizer": {
            "state": {0: {"step": torch.tensor(100)}},
            "param_groups": [{"lr": 2e-4}],
        },
        "scheduler": {"last_epoch": 100, "tail_lr": 2e-4},
        "scaler": {"scale": 2048.0},
        "steps": 100,
        "training_contract": {"optimizer": {"type": "schedule_free_adamw"}},
        "convergence_state": None,
        "resume_supported": True,
        "file_splits": {
            "seed": 7,
            "train": {"files": 2, "sha256": "old"},
            "dev": {"files": 1, "sha256": "dev"},
            "test": {"files": 1, "sha256": "test"},
        },
        "data_progress": {
            "signature": {"batch_size": 640, "num_workers": 2},
            "cycle": 4,
            "resume_cursors": {0: [10, 20]},
            "samples_consumed": 64_000,
            "batches_consumed": 100,
        },
        "config": {"oracle_critic_pretrain": {"run_name": "source"}},
        "oracle_critic_pretrain": {"run_name": "source"},
        "init_info": {"source": "sl.pth", "loaded": True},
    }


class OracleCriticCurriculumTests(unittest.TestCase):
    def test_formal_curriculum_uses_s35_duration_after_interim_sanity(self):
        self.assertEqual(
            {"phase_a": 630_000, "phase_b": 420_000, "phase_c": 210_000},
            PHASE_LENGTHS,
        )
        self.assertEqual(
            {"phase_a": 630_000, "phase_b": 1_050_000, "phase_c": 1_260_000},
            PHASE_TARGETS,
        )
        self.assertEqual(190_000, INTERIM_UNIFORM_SANITY_STEP)
        self.assertLess(INTERIM_UNIFORM_SANITY_STEP, PHASE_TARGETS["phase_a"])

    def test_classifies_shifted_sl_windows_and_filters_quarantine(self):
        invalid = source("202201", "invalid")
        buckets = classify_source_files(
            [
                source("201901"),
                source("202112"),
                invalid,
                source("202212"),
                source("202312"),
                source("202412"),
            ],
            boundaries=BOUNDARIES,
            invalid_sources=[invalid],
        )

        self.assertEqual([source("201901")], buckets["early"])
        self.assertEqual([source("202112")], buckets["mid"])
        self.assertEqual([source("202212")], buckets["old_regression"])
        self.assertEqual([source("202312")], buckets["recent_older12"])
        self.assertEqual([source("202412")], buckets["recent_12"])

    def test_builds_exact_deterministic_strong_phase_weights(self):
        cache_buckets = {
            "early": ["e0", "e1"],
            "mid": ["m0", "m1"],
            "recent_older12": ["r24a", "r24b"],
            "recent_12": ["r12a", "r12b"],
        }

        first, counts = build_phase_file_lists(
            cache_buckets, target_size=1000, seed=17
        )
        second, second_counts = build_phase_file_lists(
            cache_buckets, target_size=1000, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual(counts, second_counts)
        self.assertEqual({"recent": 600, "mid": 250, "early": 150}, counts["phase_a"])
        self.assertEqual({"recent": 900, "replay": 100}, counts["phase_b"])
        self.assertEqual({"recent": 980, "replay": 20}, counts["phase_c"])
        self.assertEqual(980, sum(item.startswith("r12") for item in first["phase_c"]))

    def test_phase_migration_preserves_training_state_and_resets_only_cursor(self):
        state = fake_checkpoint()
        before = checkpoint_component_hashes(state)
        destination_splits = {
            "seed": 7,
            "train": {"files": 3, "sha256": "new"},
            "dev": state["file_splits"]["dev"],
            "test": state["file_splits"]["test"],
        }

        migrated = migrate_checkpoint_for_phase(
            state,
            destination_config={
                "oracle_critic_pretrain": {"run_name": "phase_b"}
            },
            destination_file_splits=destination_splits,
            source_phase="phase_a",
            destination_phase="phase_b",
            source_checkpoint="step_095000.pth",
            source_checkpoint_sha256="abc",
            created_at_utc="2026-09-01T00:00:00+00:00",
        )

        self.assertEqual(before, checkpoint_component_hashes(migrated))
        self.assertEqual(destination_splits, migrated["file_splits"])
        self.assertEqual(0, migrated["data_progress"]["cycle"])
        self.assertEqual({}, migrated["data_progress"]["resume_cursors"])
        self.assertEqual(64_000, migrated["data_progress"]["samples_consumed"])
        self.assertEqual(100, migrated["data_progress"]["batches_consumed"])
        record = migrated["init_info"]["curriculum_provenance"][-1]
        self.assertEqual("phase_b", record["destination_phase"])
        self.assertEqual(
            "reset_for_declared_curriculum_phase", record["data_cursor_action"]
        )

    def test_same_phase_convergence_migration_preserves_exact_cursor(self):
        state = fake_checkpoint()
        migrated = migrate_checkpoint_for_phase(
            state,
            destination_config={
                "oracle_critic_pretrain": {"run_name": "convergence"}
            },
            destination_file_splits=state["file_splits"],
            source_phase="phase_c",
            destination_phase="phase_c_convergence",
            source_checkpoint="step_190000.pth",
            source_checkpoint_sha256="def",
            reset_data_cursor=False,
        )

        self.assertEqual(state["data_progress"], migrated["data_progress"])
        record = migrated["init_info"]["curriculum_provenance"][-1]
        self.assertEqual("preserve_for_same_train_split", record["data_cursor_action"])

    def test_changed_train_split_cannot_preserve_cursor(self):
        state = fake_checkpoint()
        destination_splits = dict(state["file_splits"])
        destination_splits["train"] = {"files": 3, "sha256": "new"}
        with self.assertRaisesRegex(ValueError, "requires resetting"):
            migrate_checkpoint_for_phase(
                state,
                destination_config={
                    "oracle_critic_pretrain": {"run_name": "phase_b"}
                },
                destination_file_splits=destination_splits,
                source_phase="phase_a",
                destination_phase="phase_b",
                source_checkpoint="step_095000.pth",
                source_checkpoint_sha256="abc",
                reset_data_cursor=False,
            )

    def test_schedule_free_convergence_tail_contract_is_explicit(self):
        convergence = resolve_convergence_config(
            {
                "convergence": {
                    "enabled": True,
                    "core_optimizer_steps": 100,
                    "tail_lr_levels": [0.005, 0.001],
                    "metric": "primary_loss",
                }
            }
        )

        validate_convergence_schedule(
            convergence,
            scheduler_cfg={"type": "optimizer"},
            optimizer_cfg={"type": "schedule_free_adamw", "lr": 0.01},
            scheduler_horizon_steps=100,
            max_steps=200,
        )
        with self.assertRaisesRegex(ValueError, "Schedule-Free"):
            validate_convergence_schedule(
                convergence,
                scheduler_cfg={"type": "optimizer"},
                optimizer_cfg={"type": "adamw"},
                scheduler_horizon_steps=100,
                max_steps=200,
            )


if __name__ == "__main__":
    unittest.main()

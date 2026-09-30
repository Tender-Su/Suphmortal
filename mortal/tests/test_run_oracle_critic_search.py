import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_oracle_critic_search import (
    EXTERNAL_PAUSE_EXIT_CODE,
    case_summary,
    case_scheduler_type,
    case_uses_convergence,
    make_case_config,
    preserve_stage_checkpoint,
)


class PreserveStageCheckpointTest(unittest.TestCase):
    def test_external_pause_exit_code_is_reserved(self):
        self.assertEqual(75, EXTERNAL_PAUSE_EXIT_CODE)

    def test_global_case_replaces_legacy_cosine_and_uses_monitor_subset(self):
        args = SimpleNamespace(
            search_name="recipe",
            seed=7,
            split_seed=8,
            num_workers=2,
            file_batch_size=5,
            prefetch_factor=2,
            val_num_workers=0,
            val_file_batch_size=8,
            val_prefetch_factor=1,
            stage_steps=200000,
            scheduler_horizon_steps=2500000,
            save_every=10000,
            val_every_steps=20000,
            dependency_val_every_steps=0,
            val_batches=256,
            test_batches=1024,
            final_test=False,
            val_game_id_modulus=5,
            val_game_id_remainder=[0],
        )
        case = {
            "name": "wsd",
            "optimizer": {"type": "adamw"},
            "scheduler": {
                "type": "wsd",
                "init": 1e-8,
                "peak": 5e-5,
                "final": 1e-6,
                "warm_up_steps": 2000,
                "stable_steps": 800000,
                "decay_steps": 200000,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = make_case_config(
                {"oracle_critic_pretrain": {"scheduler": {"max_steps": 99}}},
                case,
                Path(tmp_dir),
                args,
            )

        pretrain = cfg["oracle_critic_pretrain"]
        self.assertEqual("wsd", case_scheduler_type(case))
        self.assertNotIn("max_steps", pretrain["scheduler"])
        self.assertEqual([0], pretrain["val_game_id_remainders"])
        self.assertFalse(pretrain["convergence"]["enabled"])

    def test_preserves_only_the_exact_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir)
            checkpoint_dir = case_dir / "checkpoints"
            checkpoint_dir.mkdir()
            latest = checkpoint_dir / "latest.pth"
            latest.write_bytes(b"checkpoint")
            (case_dir / "metrics.jsonl").write_text(
                json.dumps({"steps": 5000, "val": {"loss": 1.0}}) + "\n",
                encoding="utf-8",
            )

            preserved = preserve_stage_checkpoint(case_dir, 5000)

            self.assertEqual(checkpoint_dir / "step_005000.pth", preserved)
            self.assertEqual(b"checkpoint", preserved.read_bytes())
            self.assertIsNone(preserve_stage_checkpoint(case_dir, 4000))

    def test_convergence_case_and_summary_are_recognized(self):
        case = {
            "name": "formal",
            "pretrain": {"convergence": {"enabled": True}},
        }
        self.assertTrue(case_uses_convergence(case))
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir)
            (case_dir / "metrics.jsonl").write_text(
                json.dumps({
                    "steps": 123,
                    "val": {
                        "loss": 1.0,
                        "outputs": {"relative_player_0": {"loss": 0.9}},
                    },
                    "convergence": {"converged": True},
                }) + "\n",
                encoding="utf-8",
            )

            summary = case_summary(case, case_dir)

        self.assertEqual("converged", summary["status"])
        self.assertEqual(123, summary["steps"])


if __name__ == "__main__":
    unittest.main()

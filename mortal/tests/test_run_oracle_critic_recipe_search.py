import unittest
from copy import deepcopy
from pathlib import Path

from scripts.run_oracle_critic_recipe_search import (
    constant_scheduler_config,
    current_schedule_lr,
    fork_checkpoint_state,
    make_branch_config,
    normalize_output_weights,
    parse_arm_specs,
    parse_p0_weight_arm_specs,
)


def fake_anchor_state():
    return {
        "steps": 100,
        "oracle_brain": {},
        "value_net": {},
        "optimizer": {
            "param_groups": [
                {"name": "encoder", "lr": 0.01, "initial_lr": 1.0},
                {"name": "value", "lr": 0.005, "initial_lr": 0.5},
            ]
        },
        "scheduler": {
            "init": 1e-4,
            "peak": 0.02,
            "final": 0.001,
            "warm_up_steps": 10,
            "max_steps": 1000,
            "offset": 0,
            "epoch_size": 0,
            "tail_lr": 0.001,
            "base_lrs": [1.0, 0.5],
            "last_epoch": 100,
            "_step_count": 101,
            "_last_lr": [0.01, 0.005],
            "lr_lambdas": [{}, {}],
        },
        "scaler": {},
        "data_progress": {"signature": {}, "cycle": 3},
        "training_contract": {
            "critic_arch": "dual_tower",
            "weight_decay": 0.03,
            "scheduler": {
                "init": 1e-4,
                "peak": 0.02,
                "final": 0.001,
                "warm_up_steps": 10,
                "max_steps": 1000,
            },
            "convergence": {"core_optimizer_steps": 1000},
        },
        "convergence_state": {"tail_started": False},
        "resume_supported": True,
    }


def fake_config():
    return {
        "oracle_critic_pretrain": {
            "scheduler": {
                "init": 1e-4,
                "peak": 0.02,
                "final": 0.001,
                "warm_up_steps": 10,
                "max_steps": 1000,
            }
        }
    }


class OracleCriticRecipeSearchTest(unittest.TestCase):
    def test_custom_constant_arms_replace_defaults(self):
        self.assertEqual(
            [
                {"name": "hold", "kind": "constant", "lr_multiplier": 1.0},
                {"name": "half", "kind": "constant", "lr_multiplier": 0.5},
            ],
            parse_arm_specs(["hold=1", "half=0.5"]),
        )

    def test_custom_constant_arms_reject_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate constant arm name"):
            parse_arm_specs(["half=0.5", "half=0.25"])

    def test_p0_weight_arms_share_lr_and_keep_raw_output_weights(self):
        self.assertEqual(
            [
                {
                    "name": "p0x1p5",
                    "kind": "constant",
                    "lr_multiplier": 0.25,
                    "target_output_weights": [1.5, 1.0, 1.0, 1.0],
                },
                {
                    "name": "p0x2",
                    "kind": "constant",
                    "lr_multiplier": 0.25,
                    "target_output_weights": [2.0, 1.0, 1.0, 1.0],
                },
            ],
            parse_p0_weight_arm_specs(
                ["p0x1p5=1.5", "p0x2=2"],
                lr_multiplier=0.25,
            ),
        )

    def test_output_weight_normalization_preserves_mean_weight(self):
        self.assertEqual(
            [1.6, 0.8, 0.8, 0.8],
            normalize_output_weights([2.0, 1.0, 1.0, 1.0]),
        )

    def test_reads_consistent_current_lr(self):
        self.assertEqual(0.01, current_schedule_lr(fake_anchor_state()))

    def test_constant_scheduler_is_exactly_flat(self):
        self.assertEqual(
            {
                "init": 0.01,
                "peak": 0.01,
                "final": 0.01,
                "warm_up_steps": 0,
                "max_steps": 120,
            },
            constant_scheduler_config(0.01, 120),
        )

    def test_constant_fork_preserves_state_but_replaces_scheduler_policy(self):
        state = fake_anchor_state()
        original_data_progress = deepcopy(state["data_progress"])
        arm = {"name": "constant_high", "kind": "constant", "lr_multiplier": 2.0}
        branch_cfg, scheduler_cfg = make_branch_config(
            fake_config(),
            arm=arm,
            arm_dir=Path("recipe") / "constant_high",
            search_name="recipe",
            anchor_step=100,
            target_step=120,
            current_lr=0.01,
            in_training_val_batches=8,
        )

        result = fork_checkpoint_state(
            state,
            branch_cfg=branch_cfg,
            scheduler_cfg=scheduler_cfg,
            arm=arm,
            anchor_path=Path("anchor.pth"),
            anchor_sha256="abc",
            target_step=120,
        )

        self.assertEqual(original_data_progress, result["data_progress"])
        self.assertEqual([0.02, 0.01], [g["lr"] for g in result["optimizer"]["param_groups"]])
        self.assertEqual([0.02, 0.01], result["scheduler"]["_last_lr"])
        self.assertEqual(0.02, result["scheduler"]["peak"])
        self.assertNotIn("convergence", result["training_contract"])
        self.assertIsNone(result["convergence_state"])

    def test_p0_weight_fork_updates_config_and_resume_contract(self):
        state = fake_anchor_state()
        arm = {
            "name": "p0x2",
            "kind": "constant",
            "lr_multiplier": 0.5,
            "target_output_weights": [2.0, 1.0, 1.0, 1.0],
        }
        branch_cfg, scheduler_cfg = make_branch_config(
            fake_config(),
            arm=arm,
            arm_dir=Path("recipe") / "p0x2",
            search_name="recipe",
            anchor_step=100,
            target_step=120,
            current_lr=0.01,
            in_training_val_batches=8,
        )

        result = fork_checkpoint_state(
            state,
            branch_cfg=branch_cfg,
            scheduler_cfg=scheduler_cfg,
            arm=arm,
            anchor_path=Path("anchor.pth"),
            anchor_sha256="abc",
            target_step=120,
        )

        self.assertEqual(
            [2.0, 1.0, 1.0, 1.0],
            branch_cfg["oracle_critic_pretrain"]["target_output_weights"],
        )
        self.assertEqual(
            [1.6, 0.8, 0.8, 0.8],
            result["training_contract"]["target_output_weights"],
        )

    def test_cosine_arm_preserves_exact_scheduler_state(self):
        state = fake_anchor_state()
        original_scheduler = deepcopy(state["scheduler"])
        arm = {"name": "cosine_inherited", "kind": "continue", "lr_multiplier": 1.0}
        branch_cfg, scheduler_cfg = make_branch_config(
            fake_config(),
            arm=arm,
            arm_dir=Path("recipe") / "cosine_inherited",
            search_name="recipe",
            anchor_step=100,
            target_step=120,
            current_lr=0.01,
            in_training_val_batches=8,
        )

        result = fork_checkpoint_state(
            state,
            branch_cfg=branch_cfg,
            scheduler_cfg=scheduler_cfg,
            arm=arm,
            anchor_path=Path("anchor.pth"),
            anchor_sha256="abc",
            target_step=120,
        )

        self.assertEqual(original_scheduler, result["scheduler"])
        self.assertEqual(1000, result["training_contract"]["scheduler"]["max_steps"])


if __name__ == "__main__":
    unittest.main()

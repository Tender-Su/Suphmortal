from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mortal.research.run_online_fidelity as run_online_fidelity
from mortal.core.toml_utils import load_toml_file, write_toml_file


def _minimal_online_base_config() -> dict:
    return {
        "control": {
            "online": True,
            "state_file": "C:/tmp/mortal.pth",
            "best_state_file": "C:/tmp/best.pth",
            "tensorboard_dir": "C:/tmp/tb_log",
            "test_every": 20000,
        },
        "test_play": {
            "games": 3000,
        },
        "train_play": {
            "default": {
                "games": 800,
                "log_dir": "C:/tmp/train_play",
            },
        },
        "policy": {
            "online_action_scope": "all",
            "gae_enabled": False,
            "logit_thres": 2.0,
            "importance_rho_clip": 1.5,
            "importance_c_clip": 1.0,
            "entropy_floor": 0.8,
            "entropy_floor_start_step": 999,
            "entropy_target": 1.0,
            "entropy_adjust_rate": 1e-4,
        },
        "grp": {
            "label_smoothing": 0.1,
        },
        "value": {
            "enabled": False,
            "oracle_critic": False,
            "weight": 0.0,
            "zero_sum_weight": 0.0,
        },
        "oracle_guiding": {
            "actor_enabled": False,
            "actor_source": "zero",
            "decay_steps": 200000,
        },
        "online": {
            "remote": {"host": "127.0.0.1", "port": 5000},
            "server": {"buffer_dir": "C:/tmp/buffer", "drain_dir": "C:/tmp/drain"},
            "importance_sampling": {
                "enabled": False,
                "max_policy_versions": 0,
            },
        },
        "optim": {
            "scheduler": {
                "max_steps": 400000,
            },
        },
        "baseline": {
            "train": {
                "state_file": "C:/tmp/baseline.pth",
            },
            "train_presets": {
                "validation": {
                    "state_file": "C:/tmp/baseline_anchor.pth",
                    "champion_state_file": "C:/tmp/sl_canonical.pth",
                    "anchor_state_file": "C:/tmp/baseline_anchor.pth",
                    "history_state_files": ["C:/tmp/history.pth"],
                    "champion_prob": 0.2,
                    "anchor_prob": 0.7,
                    "history_prob": 0.1,
                }
            },
        },
        "1v3": {
            "log_dir": "C:/tmp/1v3",
        },
        "oracle_dependency_eval": {
            "log_dir": "C:/tmp/oracle_dep",
        },
    }


class OnlineFidelityTests(unittest.TestCase):
    def test_derive_weight_from_target_share(self):
        weight = run_online_fidelity.derive_weight_from_target_share(
            observed_total_loss=0.020121,
            observed_value_loss=0.313727,
            current_value_weight=0.05,
            target_share=0.25,
        )
        self.assertGreater(weight, 0.0)
        self.assertLess(weight, 0.05)

    def test_protocol_decide_ranking_prefers_higher_avg_pt(self):
        ranked = run_online_fidelity.protocol_decide_ranking_from_results(
            {
                "results": [
                    {
                        "arm_name": "visible_w0050",
                        "oracle_critic": False,
                        "value_weight": 0.005,
                        "formal_avg_pt": -1.0,
                        "formal_avg_rank": 2.53,
                        "pt_stderr": 0.2,
                    },
                    {
                        "arm_name": "oracle_w0050",
                        "oracle_critic": True,
                        "value_weight": 0.005,
                        "formal_avg_pt": 0.4,
                        "formal_avg_rank": 2.50,
                        "pt_stderr": 0.2,
                    },
                ]
            },
            gap_threshold=0.1,
        )
        self.assertEqual("oracle_w0050", ranked["winner"]["arm_name"])
        self.assertFalse(ranked["ambiguous"])

    def test_build_protocol_decide_manifest_writes_candidate_configs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_config_path = root / "base.toml"
            write_toml_file(base_config_path, _minimal_online_base_config())
            manifest = run_online_fidelity.build_protocol_decide_manifest(
                base_config_path=base_config_path,
                runtime_root=root / "pd",
                base_experiment_profile="ms_rl1_add_value_gae_is_500",
                calibration_payload={
                    "search_space": {
                        "protocol_decide_oracle_critic_options": [False, True],
                        "protocol_decide_value_weights": [0.005, 0.01],
                    }
                },
                opponent_pool_preset="validation",
            )
            self.assertEqual(4, len(manifest["candidates"]))
            oracle_candidates = [
                candidate for candidate in manifest["candidates"] if candidate["oracle_critic"]
            ]
            self.assertTrue(oracle_candidates)
            for candidate in manifest["candidates"]:
                config_path = Path(candidate["config_path"])
                self.assertTrue(config_path.exists())

    def test_build_protocol_decide_manifest_can_be_limited_to_oracle_critic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_config_path = root / "base.toml"
            write_toml_file(base_config_path, _minimal_online_base_config())
            manifest = run_online_fidelity.build_protocol_decide_manifest(
                base_config_path=base_config_path,
                runtime_root=root / "pd",
                base_experiment_profile="ms_rl1_add_value_gae_is_oracle_critic_500",
                calibration_payload={
                    "search_space": {
                        "protocol_decide_oracle_critic_options": [False, True],
                        "protocol_decide_value_weights": [0.002234, 0.01266],
                    }
                },
                opponent_pool_preset="validation",
                oracle_critic_mode="oracle",
            )
            self.assertEqual(2, len(manifest["candidates"]))
            self.assertEqual("oracle", manifest["oracle_critic_mode"])
            for candidate in manifest["candidates"]:
                self.assertTrue(candidate["oracle_critic"])
                self.assertIn("protocol_decide_oracle_w", candidate["arm_name"])
            self.assertEqual(
                [
                    "protocol_decide_oracle_w002234",
                    "protocol_decide_oracle_w012660",
                ],
                [candidate["arm_name"] for candidate in manifest["candidates"]],
            )

    def test_build_protocol_decide_manifest_resolves_relative_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_config = _minimal_online_base_config()
            base_config["baseline"]["test"] = {"state_file": "./checkpoints/baseline.pth"}
            base_config["control"]["state_file"] = "./checkpoints/mortal.pth"
            base_config["control"]["best_state_file"] = "./checkpoints/best.pth"
            base_config["online"]["server"]["buffer_dir"] = "./server/buffer"
            base_config["online"]["server"]["drain_dir"] = "./server/drain"
            base_config_path = root / "base.toml"
            write_toml_file(base_config_path, base_config)
            manifest = run_online_fidelity.build_protocol_decide_manifest(
                base_config_path=base_config_path,
                runtime_root=root / "pd",
                base_experiment_profile="ms_rl1_add_value_gae_is_oracle_critic_500",
                calibration_payload={
                    "search_space": {
                        "protocol_decide_oracle_critic_options": [True],
                        "protocol_decide_value_weights": [0.005],
                    }
                },
                opponent_pool_preset="validation",
                oracle_critic_mode="oracle",
            )
            candidate_cfg = Path(manifest["candidates"][0]["config_path"])
            candidate_config = load_toml_file(candidate_cfg)
            self.assertEqual(
                str((root / "checkpoints" / "baseline.pth").resolve()),
                candidate_config["baseline"]["test"]["state_file"],
            )


if __name__ == "__main__":
    unittest.main()

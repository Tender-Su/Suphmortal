import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.online.online_machine_modes as online_machine_modes


class OnlineMachineModesTests(unittest.TestCase):
    def test_scaled_online_warmup_steps_uses_micro_gate_schedule(self):
        self.assertEqual(50, online_machine_modes._scaled_online_warmup_steps(500))
        self.assertEqual(100, online_machine_modes._scaled_online_warmup_steps(1500))
        self.assertEqual(150, online_machine_modes._scaled_online_warmup_steps(3000))
        self.assertEqual(200, online_machine_modes._scaled_online_warmup_steps(4000))

    def test_independent_arm_config_isolates_server_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "base"
            base_dir.mkdir(parents=True, exist_ok=True)
            config_path = base_dir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[control]",
                        "online = true",
                        "state_file = './checkpoints/mortal.pth'",
                        "best_state_file = './checkpoints/best.pth'",
                        "tensorboard_dir = './tb_log'",
                        "",
                        "[test_play]",
                        "log_dir = './logs/test_play'",
                        "",
                        "[train_play.default]",
                        "log_dir = './logs/train_play'",
                        "",
                        "[online.remote]",
                        "host = '192.0.2.10'",
                        "port = 5000",
                        "",
                        "[online.server]",
                        "buffer_dir = './buffer'",
                        "drain_dir = './drain'",
                        "",
                        "[1v3]",
                        "log_dir = './logs/1v3'",
                        "",
                        "[oracle_dependency_eval]",
                        "log_dir = './logs/oracle_dependency'",
                    ]
                ),
                encoding="utf-8",
                newline="\n",
            )

            base_config = online_machine_modes.load_resolved_base_config(config_path)
            runtime_root = Path(tmp_dir) / "runtime" / "independent"
            built = online_machine_modes.build_independent_arm_config(
                base_config,
                runtime_root=runtime_root,
                remote_host="127.0.0.1",
                remote_port=5100,
            )

            self.assertEqual(str((runtime_root / "checkpoints" / "mortal.pth").resolve()), built["control"]["state_file"])
            self.assertEqual(str((runtime_root / "checkpoints" / "best.pth").resolve()), built["control"]["best_state_file"])
            self.assertEqual(str((runtime_root / "tb_log").resolve()), built["control"]["tensorboard_dir"])
            self.assertEqual("127.0.0.1", built["online"]["remote"]["host"])
            self.assertEqual(5100, built["online"]["remote"]["port"])
            self.assertEqual(str((runtime_root / "server" / "buffer").resolve()), built["online"]["server"]["buffer_dir"])
            self.assertEqual(str((runtime_root / "server" / "drain").resolve()), built["online"]["server"]["drain_dir"])
            self.assertEqual(str((runtime_root / "logs" / "train_play" / "default").resolve()), built["train_play"]["default"]["log_dir"])
            self.assertEqual("independent_arm", built["online_machine_mode"]["mode"])
            self.assertTrue((runtime_root / "checkpoints").is_dir())
            self.assertTrue((runtime_root / "server" / "buffer").is_dir())
            self.assertTrue((runtime_root / "logs" / "train_play" / "default").is_dir())

    def test_worker_config_preserves_model_paths_but_redirects_local_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "base"
            base_dir.mkdir(parents=True, exist_ok=True)
            config_path = base_dir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[control]",
                        "online = true",
                        "state_file = './checkpoints/mortal.pth'",
                        "best_state_file = './checkpoints/best.pth'",
                        "tensorboard_dir = './tb_log'",
                        "",
                        "[baseline.train]",
                        "state_file = './checkpoints/baseline.pth'",
                        "champion_state_file = './checkpoints/champion.pth'",
                        "anchor_state_file = './checkpoints/anchor.pth'",
                        "history_state_files = ['./checkpoints/h1.pth', './checkpoints/h2.pth']",
                        "",
                        "[online.remote]",
                        "host = '192.0.2.10'",
                        "port = 5000",
                        "",
                        "[online.server]",
                        "buffer_dir = './buffer'",
                        "drain_dir = './drain'",
                        "",
                        "[test_play]",
                        "log_dir = './logs/test_play'",
                        "",
                        "[train_play.default]",
                        "log_dir = './logs/train_play'",
                        "",
                        "[1v3]",
                        "log_dir = './logs/1v3'",
                    ]
                ),
                encoding="utf-8",
                newline="\n",
            )

            base_config = online_machine_modes.load_resolved_base_config(config_path)
            expected_state = str((base_dir / "checkpoints" / "mortal.pth").resolve())
            expected_baseline = str((base_dir / "checkpoints" / "baseline.pth").resolve())
            expected_champion = str((base_dir / "checkpoints" / "champion.pth").resolve())
            expected_anchor = str((base_dir / "checkpoints" / "anchor.pth").resolve())
            expected_history = [
                str((base_dir / "checkpoints" / "h1.pth").resolve()),
                str((base_dir / "checkpoints" / "h2.pth").resolve()),
            ]
            runtime_root = Path(tmp_dir) / "runtime" / "worker"
            built = online_machine_modes.build_worker_mode_config(
                base_config,
                runtime_root=runtime_root,
                remote_host="192.0.2.20",
                remote_port=5200,
            )

            self.assertEqual(expected_state, built["control"]["state_file"])
            self.assertEqual(expected_baseline, built["baseline"]["train"]["state_file"])
            self.assertEqual(expected_champion, built["baseline"]["train"]["champion_state_file"])
            self.assertEqual(expected_anchor, built["baseline"]["train"]["anchor_state_file"])
            self.assertEqual(expected_history, built["baseline"]["train"]["history_state_files"])
            self.assertEqual("192.0.2.20", built["online"]["remote"]["host"])
            self.assertEqual(5200, built["online"]["remote"]["port"])
            self.assertEqual(str((runtime_root / "logs" / "train_play" / "default").resolve()), built["train_play"]["default"]["log_dir"])
            self.assertEqual(str((runtime_root / "server" / "buffer").resolve()), built["online"]["server"]["buffer_dir"])
            self.assertEqual("worker", built["online_machine_mode"]["mode"])
            self.assertTrue((runtime_root / "server" / "buffer").is_dir())
            self.assertTrue((runtime_root / "logs" / "train_play" / "default").is_dir())

    def test_independent_arm_config_applies_validation_opponent_pool_preset(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
            },
            "baseline": {
                "train": {
                    "device": "cuda:0",
                    "enable_compile": False,
                    "state_file": "C:/tmp/checkpoints/base_default.pth",
                    "champion_state_file": "C:/tmp/checkpoints/base_default.pth",
                    "anchor_state_file": "C:/tmp/checkpoints/base_default.pth",
                    "history_state_files": [],
                    "champion_prob": 0.50,
                    "anchor_prob": 0.25,
                    "history_prob": 0.25,
                    "reload_each_session": True,
                },
                "train_presets": {
                    "validation": {
                        "state_file": "C:/tmp/checkpoints/anchor.pth",
                        "champion_state_file": "C:/tmp/checkpoints/validation_champion.pth",
                        "anchor_state_file": "C:/tmp/checkpoints/anchor.pth",
                        "history_state_files": ["C:/tmp/checkpoints/h1.pth"],
                        "champion_prob": 0.20,
                        "anchor_prob": 0.70,
                        "history_prob": 0.10,
                    },
                },
                "test": {
                    "state_file": "C:/tmp/checkpoints/baseline_test.pth",
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            opponent_pool_preset="validation",
        )

        self.assertEqual(
            "C:/tmp/checkpoints/validation_champion.pth",
            built["baseline"]["train"]["champion_state_file"],
        )
        self.assertEqual(0.20, built["baseline"]["train"]["champion_prob"])
        self.assertEqual(0.70, built["baseline"]["train"]["anchor_prob"])
        self.assertEqual(
            ["C:/tmp/checkpoints/h1.pth"],
            built["baseline"]["train"]["history_state_files"],
        )
        self.assertEqual(
            "C:/tmp/checkpoints/baseline_test.pth",
            built["baseline"]["test"]["state_file"],
        )
        self.assertEqual(
            "validation",
            built["online_opponent_pool_preset"]["name"],
        )
        self.assertEqual(
            "baseline.train_presets.validation",
            built["online_opponent_pool_preset"]["source"],
        )

    def test_independent_arm_config_can_force_repro_seed(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
            },
            "repro": {
                "enabled": False,
                "seed": 1,
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            repro_seed=2026041501,
            train_key=123456789,
            train_seed_start=7000,
        )

        self.assertTrue(built["repro"]["enabled"])
        self.assertEqual(2026041501, built["repro"]["seed"])
        self.assertEqual(123456789, built["repro"]["train_key"])
        self.assertEqual(7000, built["repro"]["train_seed_start"])

    def test_worker_config_applies_formal_opponent_pool_preset(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
            },
            "baseline": {
                "train": {
                    "device": "cuda:0",
                    "enable_compile": False,
                    "state_file": "C:/tmp/checkpoints/base_default.pth",
                    "champion_state_file": "C:/tmp/checkpoints/base_default.pth",
                    "anchor_state_file": "C:/tmp/checkpoints/base_default.pth",
                    "history_state_files": [],
                    "champion_prob": 0.50,
                    "anchor_prob": 0.25,
                    "history_prob": 0.25,
                    "reload_each_session": True,
                },
                "train_presets": {
                    "formal": {
                        "state_file": "C:/tmp/checkpoints/anchor.pth",
                        "champion_state_file": "C:/tmp/checkpoints/train_champion.pth",
                        "anchor_state_file": "C:/tmp/checkpoints/anchor.pth",
                        "history_state_files": [
                            "C:/tmp/checkpoints/h1.pth",
                            "C:/tmp/checkpoints/h2.pth",
                        ],
                        "champion_prob": 0.50,
                        "anchor_prob": 0.25,
                        "history_prob": 0.25,
                    },
                },
            },
        }

        built = online_machine_modes.build_worker_mode_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            opponent_pool_preset="formal",
        )

        self.assertEqual(
            "C:/tmp/checkpoints/train_champion.pth",
            built["baseline"]["train"]["champion_state_file"],
        )
        self.assertEqual(
            ["C:/tmp/checkpoints/h1.pth", "C:/tmp/checkpoints/h2.pth"],
            built["baseline"]["train"]["history_state_files"],
        )
        self.assertEqual(0.25, built["baseline"]["train"]["history_prob"])
        self.assertEqual("formal", built["online_opponent_pool_preset"]["name"])

    def test_unknown_opponent_pool_preset_raises_clear_error(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
            },
            "baseline": {
                "train": {
                    "state_file": "C:/tmp/checkpoints/base_default.pth",
                },
                "train_presets": {},
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            r"requires baseline\.train_presets\.validation",
        ):
            online_machine_modes.build_independent_arm_config(
                base_config,
                runtime_root=Path("C:/tmp/runtime"),
                opponent_pool_preset="validation",
            )

    def test_experiment_profile_applies_ms_rl2_sanity_overrides(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
            },
            "value": {
                "enabled": True,
                "oracle_critic": True,
            },
            "oracle_guiding": {
                "actor_enabled": False,
                "actor_source": "zero",
                "decay_steps": 200000,
            },
            "search": {
                "enabled": True,
            },
            "search_distill": {
                "enabled": True,
            },
            "oracle_dependency_eval": {
                "enabled": True,
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl2_sanity_120k",
        )

        self.assertTrue(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("true", built["oracle_guiding"]["actor_source"])
        self.assertEqual(80000, built["oracle_guiding"]["decay_steps"])
        self.assertFalse(built["value"]["oracle_critic"])
        self.assertFalse(built["search"]["enabled"])
        self.assertFalse(built["search_distill"]["enabled"])
        self.assertFalse(built["oracle_dependency_eval"]["enabled"])
        self.assertEqual(120000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual("ms_rl2_sanity_120k", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl2_minimal_overrides(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
            },
            "policy": {
                "gae_enabled": True,
            },
            "aux": {
                "next_rank_weight": 0.2,
                "opponent_state_weight": 0.03,
                "danger_enabled": True,
                "danger_weight": 0.05,
                "tile_efficiency_weight": 0.01,
                "furo_regret_weight": 0.01,
                "hand_value_regret_weight": 0.01,
            },
            "value": {
                "enabled": True,
                "oracle_critic": True,
                "weight": 0.05,
                "zero_sum_weight": 0.01,
            },
            "oracle_guiding": {
                "actor_enabled": False,
                "actor_source": "zero",
                "decay_steps": 200000,
            },
            "search": {
                "enabled": True,
            },
            "search_distill": {
                "enabled": True,
            },
            "oracle_dependency_eval": {
                "enabled": True,
            },
            "online": {
                "importance_sampling": {
                    "enabled": True,
                    "max_policy_versions": 8,
                },
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl2_minimal_100k",
        )

        self.assertTrue(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("true", built["oracle_guiding"]["actor_source"])
        self.assertEqual(60000, built["oracle_guiding"]["decay_steps"])
        self.assertFalse(built["policy"]["gae_enabled"])
        self.assertEqual(0.0, built["aux"]["next_rank_weight"])
        self.assertEqual(0.0, built["aux"]["opponent_state_weight"])
        self.assertFalse(built["aux"]["danger_enabled"])
        self.assertEqual(0.0, built["aux"]["danger_weight"])
        self.assertEqual(0.0, built["aux"]["tile_efficiency_weight"])
        self.assertEqual(0.0, built["aux"]["furo_regret_weight"])
        self.assertEqual(0.0, built["aux"]["hand_value_regret_weight"])
        self.assertFalse(built["value"]["enabled"])
        self.assertFalse(built["value"]["oracle_critic"])
        self.assertEqual(0.0, built["value"]["weight"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["vtrace_rho_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_c_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])
        self.assertEqual(0.0, built["policy"]["entropy_adjust_rate"])
        self.assertFalse(built["search"]["enabled"])
        self.assertFalse(built["search_distill"]["enabled"])
        self.assertFalse(built["oracle_dependency_eval"]["enabled"])
        self.assertTrue(built["online"]["stop_at_max_steps"])
        self.assertFalse(built["online"]["importance_sampling"]["enabled"])
        self.assertEqual(0, built["online"]["importance_sampling"]["max_policy_versions"])
        self.assertEqual(5000, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(100000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual("ms_rl2_minimal_100k", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl2_smoke_overrides(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
                "test_every": 20000,
            },
            "oracle_guiding": {
                "actor_enabled": False,
                "actor_source": "zero",
                "decay_steps": 200000,
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl2_smoke_20k",
        )

        self.assertTrue(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("true", built["oracle_guiding"]["actor_source"])
        self.assertEqual(12000, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(5000, built["control"]["test_every"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(1000, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(20000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual("ms_rl2_smoke_20k", built["online_experiment_profile"]["name"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["vtrace_rho_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])

    def test_experiment_profile_applies_ms_rl2_smoke_10k_overrides(self):
        base_config = {
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
                },
            },
            "oracle_guiding": {
                "actor_enabled": False,
                "actor_source": "zero",
                "decay_steps": 200000,
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl2_smoke_10k",
        )

        self.assertTrue(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("true", built["oracle_guiding"]["actor_source"])
        self.assertEqual(6000, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(2500, built["control"]["test_every"])
        self.assertEqual(400, built["train_play"]["default"]["games"])
        self.assertEqual(1000, built["test_play"]["games"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(500, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(10000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual("ms_rl2_smoke_10k", built["online_experiment_profile"]["name"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["vtrace_rho_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])

    def test_experiment_profile_applies_ms_rl1_add_rank_opp_danger_500_overrides(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
                "test_every": 20000,
            },
            "policy": {
                "gae_enabled": True,
            },
            "aux": {
                "next_rank_weight": 0.0,
                "opponent_state_weight": 0.0,
                "danger_enabled": False,
                "danger_weight": 0.0,
            },
            "value": {
                "enabled": True,
                "oracle_critic": True,
                "weight": 0.05,
                "zero_sum_weight": 0.01,
            },
            "oracle_guiding": {
                "actor_enabled": True,
                "actor_source": "true",
                "decay_steps": 200000,
            },
            "search": {
                "enabled": True,
            },
            "search_distill": {
                "enabled": True,
            },
            "online": {
                "importance_sampling": {
                    "enabled": True,
                    "max_policy_versions": 8,
                },
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl1_add_rank_opp_danger_500",
        )

        self.assertFalse(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("zero", built["oracle_guiding"]["actor_source"])
        self.assertFalse(built["policy"]["gae_enabled"])
        self.assertEqual(0.2, built["aux"]["next_rank_weight"])
        self.assertEqual(0.03, built["aux"]["opponent_state_weight"])
        self.assertTrue(built["aux"]["danger_enabled"])
        self.assertEqual(0.05, built["aux"]["danger_weight"])
        self.assertFalse(built["value"]["enabled"])
        self.assertFalse(built["online"]["importance_sampling"]["enabled"])
        self.assertFalse(built["search"]["enabled"])
        self.assertFalse(built["search_distill"]["enabled"])
        self.assertEqual(50, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(500, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual("ms_rl1_add_rank_opp_danger_500", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl2_smoke_4k_overrides(self):
        base_config = {
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
                },
            },
            "policy": {
                "logit_thres": 2.0,
                "vtrace_rho_clip": 1.5,
                "vtrace_c_clip": 1.0,
                "entropy_target": 1.0,
                "entropy_adjust_rate": 1e-4,
            },
            "oracle_guiding": {
                "actor_enabled": False,
                "actor_source": "zero",
                "decay_steps": 200000,
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl2_smoke_4k",
        )

        self.assertTrue(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("true", built["oracle_guiding"]["actor_source"])
        self.assertEqual(2400, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(1000, built["control"]["test_every"])
        self.assertEqual(400, built["train_play"]["default"]["games"])
        self.assertEqual(800, built["test_play"]["games"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(200, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(4000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["importance_rho_clip"])
        self.assertEqual(0.0, built["policy"]["importance_c_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_rho_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_c_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_floor"])
        self.assertEqual(0, built["policy"]["entropy_floor_start_step"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])
        self.assertEqual(0.0, built["policy"]["entropy_adjust_rate"])
        self.assertEqual("ms_rl2_smoke_4k", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl1_mortal_policy_smoke_4k_overrides(self):
        base_config = {
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
                },
            },
            "policy": {
                "online_action_scope": "all",
                "logit_thres": 2.0,
                "vtrace_rho_clip": 1.5,
                "vtrace_c_clip": 1.0,
                "entropy_target": 1.0,
                "entropy_adjust_rate": 1e-4,
            },
            "grp": {
                "label_smoothing": 0.1,
            },
            "oracle_guiding": {
                "actor_enabled": True,
                "actor_source": "true",
                "decay_steps": 200000,
            },
            "optim": {
                "scheduler": {
                    "max_steps": 400000,
                },
            },
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl1_mortal_policy_smoke_4k",
        )

        self.assertFalse(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("zero", built["oracle_guiding"]["actor_source"])
        self.assertEqual(2400, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(1000, built["control"]["test_every"])
        self.assertEqual(400, built["train_play"]["default"]["games"])
        self.assertEqual(800, built["test_play"]["games"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(0.0, built["grp"]["label_smoothing"])
        self.assertEqual(200, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(4000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["importance_rho_clip"])
        self.assertEqual(0.0, built["policy"]["importance_c_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_rho_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_c_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_floor"])
        self.assertEqual(0, built["policy"]["entropy_floor_start_step"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])
        self.assertEqual(0.0, built["policy"]["entropy_adjust_rate"])
        self.assertEqual("ms_rl1_mortal_policy_smoke_4k", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl2_addback_opp_danger_overrides(self):
        base_config = {
            "control": {
                "online": True,
                "state_file": "C:/tmp/mortal.pth",
                "best_state_file": "C:/tmp/best.pth",
                "tensorboard_dir": "C:/tmp/tb_log",
                "test_every": 20000,
            },
            "policy": {
                "online_action_scope": "all",
                "gae_enabled": False,
            },
            "grp": {
                "label_smoothing": 0.1,
            },
            "aux": {
                "next_rank_weight": 0.0,
                "opponent_state_weight": 0.0,
                "danger_enabled": False,
                "danger_weight": 0.0,
                "tile_efficiency_weight": 0.0,
                "furo_regret_weight": 0.0,
                "hand_value_regret_weight": 0.0,
            },
            "value": {
                "enabled": False,
                "oracle_critic": True,
                "weight": 0.0,
                "zero_sum_weight": 0.0,
            },
            "oracle_guiding": {
                "actor_enabled": False,
                "actor_source": "zero",
                "decay_steps": 200000,
            },
            "online": {
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
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl2_add_value_gae_is_rank_opp_danger_40k",
        )

        self.assertTrue(built["value"]["enabled"])
        self.assertFalse(built["value"]["oracle_critic"])
        self.assertFalse(built["search"]["enabled"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(0.0, built["grp"]["label_smoothing"])
        self.assertFalse(built["test_play"]["initial_enable"])
        self.assertEqual(600, built["test_play"]["initial_games"])
        self.assertTrue(built["policy"]["gae_enabled"])
        self.assertTrue(built["online"]["importance_sampling"]["enabled"])
        self.assertEqual(8, built["online"]["importance_sampling"]["max_policy_versions"])
        self.assertEqual(0.2, built["aux"]["next_rank_weight"])
        self.assertEqual(0.03, built["aux"]["opponent_state_weight"])
        self.assertTrue(built["aux"]["danger_enabled"])
        self.assertEqual(0.05, built["aux"]["danger_weight"])
        self.assertEqual(0.0, built["aux"]["tile_efficiency_weight"])
        self.assertEqual(10000, built["control"]["test_every"])
        self.assertEqual(40000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual("ms_rl2_add_value_gae_is_rank_opp_danger_40k", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl1_add_value_gae_is_20k_overrides(self):
        base_config = {
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
                },
            },
            "policy": {
                "online_action_scope": "all",
                "gae_enabled": False,
                "logit_thres": 2.0,
                "vtrace_rho_clip": 1.5,
                "vtrace_c_clip": 1.0,
                "entropy_target": 1.0,
                "entropy_adjust_rate": 1e-4,
            },
            "grp": {
                "label_smoothing": 0.1,
            },
            "value": {
                "enabled": False,
                "oracle_critic": True,
                "weight": 0.0,
                "zero_sum_weight": 0.0,
            },
            "oracle_guiding": {
                "actor_enabled": True,
                "actor_source": "true",
                "decay_steps": 200000,
            },
            "online": {
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
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl1_add_value_gae_is_20k",
        )

        self.assertFalse(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("zero", built["oracle_guiding"]["actor_source"])
        self.assertEqual(12000, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(5000, built["control"]["test_every"])
        self.assertEqual(400, built["train_play"]["default"]["games"])
        self.assertEqual(600, built["test_play"]["games"])
        self.assertEqual("all", built["policy"]["online_action_scope"])
        self.assertEqual(0.0, built["grp"]["label_smoothing"])
        self.assertTrue(built["value"]["enabled"])
        self.assertTrue(built["policy"]["gae_enabled"])
        self.assertTrue(built["online"]["importance_sampling"]["enabled"])
        self.assertEqual(8, built["online"]["importance_sampling"]["max_policy_versions"])
        self.assertFalse(built["value"]["oracle_critic"])
        self.assertEqual(1000, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(20000, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["importance_rho_clip"])
        self.assertEqual(0.0, built["policy"]["importance_c_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_rho_clip"])
        self.assertEqual(0.0, built["policy"]["vtrace_c_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_floor"])
        self.assertEqual(0, built["policy"]["entropy_floor_start_step"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])
        self.assertEqual(0.0, built["policy"]["entropy_adjust_rate"])
        self.assertFalse(built["test_play"]["initial_enable"])
        self.assertEqual(600, built["test_play"]["initial_games"])
        self.assertEqual(600, built["online_experiment_profile"]["recorded_step0_baseline"]["games"])
        self.assertEqual(-2.475, built["online_experiment_profile"]["recorded_step0_baseline"]["avg_pt"])
        self.assertEqual("ms_rl1_add_value_gae_is_20k", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl1_add_value_gae_is_500_overrides(self):
        base_config = {
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
                "oracle_critic": True,
                "weight": 0.0,
                "zero_sum_weight": 0.0,
            },
            "oracle_guiding": {
                "actor_enabled": True,
                "actor_source": "true",
                "decay_steps": 200000,
            },
            "online": {
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
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl1_add_value_gae_is_500",
        )

        self.assertFalse(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("zero", built["oracle_guiding"]["actor_source"])
        self.assertEqual("grp", built["value"]["reward_source"])
        self.assertEqual(300, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(500, built["control"]["test_every"])
        self.assertEqual(200, built["train_play"]["default"]["games"])
        self.assertEqual(200, built["test_play"]["games"])
        self.assertEqual(50, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(500, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual(0.0, built["policy"]["logit_thres"])
        self.assertEqual(0.0, built["policy"]["importance_rho_clip"])
        self.assertEqual(0.0, built["policy"]["importance_c_clip"])
        self.assertEqual(0.0, built["policy"]["entropy_floor"])
        self.assertEqual(0, built["policy"]["entropy_floor_start_step"])
        self.assertEqual(0.0, built["policy"]["entropy_target"])
        self.assertEqual(0.0, built["policy"]["entropy_adjust_rate"])
        self.assertEqual(600, built["online_experiment_profile"]["recorded_step0_baseline"]["games"])
        self.assertEqual("ms_rl1_add_value_gae_is_500", built["online_experiment_profile"]["name"])

    def test_experiment_profile_applies_ms_rl1_add_value_gae_is_oracle_critic_500_overrides(self):
        base_config = {
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
                "actor_enabled": True,
                "actor_source": "true",
                "decay_steps": 200000,
            },
            "online": {
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
        }

        built = online_machine_modes.build_independent_arm_config(
            base_config,
            runtime_root=Path("C:/tmp/runtime"),
            experiment_profile="ms_rl1_add_value_gae_is_oracle_critic_500",
        )

        self.assertFalse(built["oracle_guiding"]["actor_enabled"])
        self.assertEqual("zero", built["oracle_guiding"]["actor_source"])
        self.assertTrue(built["value"]["enabled"])
        self.assertTrue(built["value"]["oracle_critic"])
        self.assertEqual(0.05, built["value"]["weight"])
        self.assertEqual(0.01, built["value"]["zero_sum_weight"])
        self.assertEqual("score_rank", built["value"]["reward_source"])
        self.assertEqual(300, built["oracle_guiding"]["decay_steps"])
        self.assertEqual(500, built["control"]["test_every"])
        self.assertEqual(200, built["train_play"]["default"]["games"])
        self.assertEqual(200, built["test_play"]["games"])
        self.assertEqual(50, built["optim"]["scheduler"]["warm_up_steps"])
        self.assertEqual(500, built["optim"]["scheduler"]["max_steps"])
        self.assertEqual(600, built["online_experiment_profile"]["recorded_step0_baseline"]["games"])
        self.assertEqual(-2.475, built["online_experiment_profile"]["recorded_step0_baseline"]["avg_pt"])
        self.assertEqual("ms_rl1_add_value_gae_is_oracle_critic_500", built["online_experiment_profile"]["name"])


if __name__ == "__main__":
    unittest.main()

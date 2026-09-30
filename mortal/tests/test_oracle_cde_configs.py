import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.online import oracle_cde_configs


def make_base_config(tmp_path: Path) -> dict:
    return {
        "control": {
            "version": 4,
            "online": True,
            "state_file": str(tmp_path / "old" / "mortal.pth"),
            "best_state_file": str(tmp_path / "old" / "best.pth"),
            "tensorboard_dir": str(tmp_path / "old" / "tb"),
            "save_every": 1000,
            "test_every": 20000,
        },
        "policy": {
            "gae_enabled": False,
            "logit_thres": 2.0,
        },
        "value": {
            "enabled": True,
            "oracle_critic": True,
            "oracle_critic_state_file": str(tmp_path / "stale" / "oracle.pth"),
            "critic_state_file": str(tmp_path / "stale" / "critic.pth"),
            "pretrained_state_file": str(tmp_path / "stale" / "pretrained.pth"),
            "critic_warmup_steps": 999,
        },
        "oracle_critic_pretrain": {
            "state_file": str(tmp_path / "stale" / "latest.pth"),
            "best_state_file": str(tmp_path / "stale" / "best.pth"),
            "init_state_file": str(tmp_path / "stale" / "init.pth"),
        },
        "oracle_guiding": {
            "actor_enabled": True,
            "actor_source": "true",
            "gamma_start": 1.0,
            "decay_steps": 10000,
        },
        "aux": {
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
            "tile_efficiency_weight": 0.1,
            "furo_regret_weight": 0.1,
            "hand_value_regret_weight": 0.1,
        },
        "expected_reward": {
            "enabled": True,
        },
        "search": {
            "enabled": True,
        },
        "search_distill": {
            "enabled": True,
        },
        "online": {
            "init_state_file": str(tmp_path / "sl_canonical.pth"),
            "stop_at_max_steps": False,
            "remote": {
                "host": "192.0.2.10",
                "port": 5000,
            },
            "server": {
                "buffer_dir": str(tmp_path / "old" / "buffer"),
                "drain_dir": str(tmp_path / "old" / "drain"),
            },
            "importance_sampling": {
                "enabled": False,
                "max_policy_versions": 0,
                "drop_untracked_samples": True,
            },
        },
        "baseline": {
            "train": {
                "state_file": str(tmp_path / "baseline.pth"),
            },
            "train_presets": {
                "validation": {
                    "state_file": str(tmp_path / "validation.pth"),
                    "champion_prob": 0.2,
                },
            },
        },
        "test_play": {
            "enable": True,
            "log_dir": str(tmp_path / "old" / "test_play"),
        },
        "train_play": {
            "default": {
                "games": 800,
                "log_dir": str(tmp_path / "old" / "train_play"),
            },
        },
        "1v3": {
            "log_dir": str(tmp_path / "old" / "1v3"),
        },
        "oracle_dependency_eval": {
            "enabled": True,
            "log_dir": str(tmp_path / "old" / "oracle_dependency"),
        },
        "optim": {
            "scheduler": {
                "warm_up_steps": 5000,
                "max_steps": 400000,
            },
        },
    }


class OracleCdeConfigTests(unittest.TestCase):
    def test_c_arm_uses_bridge_warmup_without_oracle_pretrain_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runtime_root = tmp_path / "runtime"
            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=runtime_root,
                tower="single",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                remote_port=5100,
            )

            self.assertEqual("C", built["oracle_cde_experiment"]["arm"])
            self.assertEqual("single_tower", built["value"]["oracle_critic_arch"])
            self.assertTrue(built["value"]["enabled"])
            self.assertTrue(built["value"]["oracle_critic"])
            for key in ("oracle_critic_state_file", "critic_state_file", "pretrained_state_file"):
                self.assertNotIn(key, built["value"])
            for key in ("best_state_file", "state_file", "init_state_file"):
                self.assertNotIn(key, built["oracle_critic_pretrain"])
            self.assertEqual(1000, built["value"]["critic_warmup_steps"])
            self.assertEqual(1000, built["oracle_cde_experiment"]["critic_warmup_steps"])
            self.assertEqual(1000, built["oracle_cde_experiment"]["critic_warmup_requested_steps"])
            self.assertEqual(0, built["oracle_cde_experiment"]["critic_warmup_resume_steps"])
            self.assertFalse(built["oracle_guiding"]["actor_enabled"])
            self.assertEqual("zero", built["oracle_guiding"]["actor_source"])
            self.assertTrue(built["policy"]["gae_enabled"])
            self.assertEqual("score_rank", built["value"]["reward_source"])
            self.assertEqual(0.05, built["value"]["weight"])
            self.assertTrue(built["online"]["stop_at_max_steps"])
            self.assertEqual(3000, built["optim"]["scheduler"]["max_steps"])
            self.assertEqual(150, built["optim"]["scheduler"]["warm_up_steps"])
            self.assertEqual(500, built["control"]["save_every"])
            self.assertEqual(3000, built["control"]["test_every"])
            self.assertEqual(512, built["control"]["batch_size"])
            self.assertEqual(16, built["online"]["importance_sampling"]["max_policy_versions"])
            self.assertEqual(0.0, built["policy"]["vtrace_target_rho_clip"])
            self.assertEqual(0.0, built["policy"]["vtrace_target_c_clip"])
            self.assertEqual("auto", built["online"]["importance_sampling"]["vtrace_mode"])
            self.assertEqual(2, built["online"]["importance_sampling"]["vtrace_min_version_gap"])
            self.assertEqual(512, built["oracle_cde_experiment"]["batch_size"])
            self.assertEqual(0.05, built["oracle_cde_experiment"]["value_weight"])
            self.assertFalse(built["test_play"]["enable"])
            self.assertEqual(str((runtime_root / "checkpoints" / "mortal.pth").resolve()), built["control"]["state_file"])
            self.assertEqual(str((runtime_root / "server" / "buffer").resolve()), built["online"]["server"]["buffer_dir"])

    def test_vtrace_target_options_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                vtrace_target_rho_clip=1.0,
                vtrace_target_c_clip=0.8,
                vtrace_mode="always",
                vtrace_min_version_gap=4,
            )

            self.assertEqual(1.0, built["policy"]["vtrace_target_rho_clip"])
            self.assertEqual(0.8, built["policy"]["vtrace_target_c_clip"])
            self.assertEqual(0.0, built["policy"]["importance_rho_clip"])
            self.assertEqual(0.0, built["policy"]["importance_c_clip"])
            self.assertEqual("always", built["online"]["importance_sampling"]["vtrace_mode"])
            self.assertEqual(4, built["online"]["importance_sampling"]["vtrace_min_version_gap"])
            self.assertEqual(1.0, built["oracle_cde_experiment"]["vtrace_target_rho_clip"])
            self.assertEqual(0.8, built["oracle_cde_experiment"]["vtrace_target_c_clip"])

    def test_d_arm_requires_and_uses_explicit_pretrain_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "oracle_best.pth"
            checkpoint.write_bytes(b"checkpoint")

            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="D",
                max_steps=3000,
                oracle_critic_state_file=checkpoint,
                critic_warmup_steps=123,
            )

            self.assertEqual("D", built["oracle_cde_experiment"]["arm"])
            self.assertEqual("dual_tower", built["value"]["oracle_critic_arch"])
            self.assertEqual(192, built["control"]["batch_size"])
            self.assertEqual(192, built["oracle_cde_experiment"]["batch_size"])
            self.assertEqual(str(checkpoint.resolve()), built["value"]["oracle_critic_state_file"])
            self.assertEqual(0, built["value"]["critic_warmup_steps"])
            self.assertEqual(str(checkpoint.resolve()), built["oracle_cde_experiment"]["oracle_critic_state_file"])

    def test_explicit_batch_size_override_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                batch_size=320,
            )

            self.assertEqual(320, built["control"]["batch_size"])
            self.assertEqual(320, built["oracle_cde_experiment"]["batch_size"])

    def test_explicit_value_weight_override_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                value_weight=0.02,
            )

            self.assertEqual(0.02, built["value"]["weight"])
            self.assertEqual(0.02, built["oracle_cde_experiment"]["value_weight"])

    def test_policy_stability_overrides_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                policy_clip_ratio=0.1,
                actor_lr_scale=0.5,
                policy_head_lr_scale=0.25,
                policy_update_interval=2,
                policy_update_phase=1,
                entropy_floor=0.47,
                entropy_adjust_rate=0.001,
                entropy_floor_start_step=8000,
            )

            policy = built["policy"]
            cde = built["oracle_cde_experiment"]
            self.assertEqual(0.1, policy["clip_ratio"])
            self.assertEqual(0.47, policy["entropy_floor"])
            self.assertEqual(0.47, policy["entropy_target"])
            self.assertEqual(0.001, policy["entropy_adjust_rate"])
            self.assertEqual(8000, policy["entropy_floor_start_step"])
            self.assertEqual(0.5, policy["actor_lr_scale"])
            self.assertEqual(0.25, policy["policy_head_lr_scale"])
            self.assertEqual(2, policy["update_interval"])
            self.assertEqual(1, policy["update_phase"])
            self.assertEqual(0.1, cde["policy_clip_ratio"])
            self.assertEqual(0.5, cde["actor_lr_scale"])
            self.assertEqual(0.25, cde["policy_head_lr_scale"])
            self.assertEqual(2, cde["policy_update_interval"])
            self.assertEqual(1, cde["policy_update_phase"])
            self.assertEqual(0.47, cde["entropy_floor"])
            self.assertEqual(0.001, cde["entropy_adjust_rate"])
            self.assertEqual(8000, cde["entropy_floor_start_step"])

    def test_invalid_policy_stability_overrides_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with self.assertRaisesRegex(ValueError, "policy_clip_ratio"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_clip",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    policy_clip_ratio=0.0,
                )

            with self.assertRaisesRegex(ValueError, "entropy_floor"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_entropy_floor",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    entropy_floor=-0.1,
                )

            with self.assertRaisesRegex(ValueError, "entropy_adjust_rate"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_entropy_rate",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    entropy_adjust_rate=-0.1,
                )

            with self.assertRaisesRegex(ValueError, "actor_lr_scale"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_actor_lr",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    actor_lr_scale=-0.1,
                )

            with self.assertRaisesRegex(ValueError, "policy_head_lr_scale"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_policy_lr",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    policy_head_lr_scale=-0.1,
                )

            with self.assertRaisesRegex(ValueError, "policy_update_interval"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_policy_interval",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    policy_update_interval=0,
                )

            with self.assertRaisesRegex(ValueError, "policy_update_phase"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_policy_phase",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    policy_update_phase=-1,
                )

    def test_explicit_repro_can_allow_cudnn_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                repro_seed=123,
                train_key=456,
                train_seed_start=789,
                allow_cudnn_benchmark=True,
            )

            self.assertTrue(built["repro"]["enabled"])
            self.assertTrue(built["repro"]["allow_cudnn_benchmark"])
            self.assertTrue(built["oracle_cde_experiment"]["allow_cudnn_benchmark"])

    def test_negative_value_weight_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with self.assertRaisesRegex(ValueError, "value_weight"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime",
                    tower="dual",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=1000,
                    value_weight=-0.01,
                )

    def test_d_and_e_fail_fast_without_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertRaisesRegex(ValueError, "requires --oracle-critic-state-file"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_d",
                    tower="single_tower",
                    arm="D",
                    max_steps=3000,
                )
            with self.assertRaises(FileNotFoundError):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime_e",
                    tower="single_tower",
                    arm="E",
                    max_steps=3000,
                    oracle_critic_state_file=tmp_path / "missing.pth",
                    critic_warmup_steps=3000,
                )

    def test_c_arm_requires_positive_actor_freeze_warmup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with self.assertRaisesRegex(ValueError, "positive critic_warmup_steps"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime",
                    tower="single_tower",
                    arm="C",
                    max_steps=3000,
                    critic_warmup_steps=0,
                )

            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime_ok",
                tower="single_tower",
                arm="C",
                max_steps=6000,
                critic_warmup_steps=3000,
            )
            self.assertEqual(3000, built["value"]["critic_warmup_steps"])
            self.assertEqual(6000, built["optim"]["scheduler"]["max_steps"])

    def test_e_arm_requires_positive_actor_freeze_warmup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "oracle_best.pth"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaisesRegex(ValueError, "positive critic_warmup_steps"):
                oracle_cde_configs.build_oracle_cde_config(
                    make_base_config(tmp_path),
                    runtime_root=tmp_path / "runtime",
                    tower="single_tower",
                    arm="E",
                    max_steps=3000,
                    oracle_critic_state_file=checkpoint,
                    critic_warmup_steps=0,
                )

            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime_ok",
                tower="single_tower",
                arm="E",
                max_steps=6000,
                oracle_critic_state_file=checkpoint,
                critic_warmup_steps=3000,
            )
            self.assertEqual(3000, built["value"]["critic_warmup_steps"])
            self.assertEqual(6000, built["optim"]["scheduler"]["max_steps"])

    def test_e_arm_adds_resume_steps_to_effective_warmup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "oracle_best.pth"
            checkpoint.write_bytes(b"checkpoint")

            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="E",
                max_steps=12000,
                oracle_critic_state_file=checkpoint,
                critic_warmup_steps=3000,
                resume_steps=5000,
            )

            cde = built["oracle_cde_experiment"]
            self.assertEqual(8000, built["value"]["critic_warmup_steps"])
            self.assertEqual(8000, cde["critic_warmup_steps"])
            self.assertEqual(3000, cde["critic_warmup_requested_steps"])
            self.assertEqual(5000, cde["critic_warmup_resume_steps"])

    def test_e_arm_allows_zero_extra_warmup_when_resuming(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "oracle_best.pth"
            checkpoint.write_bytes(b"checkpoint")

            built = oracle_cde_configs.build_oracle_cde_config(
                make_base_config(tmp_path),
                runtime_root=tmp_path / "runtime",
                tower="dual",
                arm="E",
                max_steps=9000,
                oracle_critic_state_file=checkpoint,
                critic_warmup_steps=0,
                resume_steps=8000,
            )

            cde = built["oracle_cde_experiment"]
            self.assertEqual(8000, built["value"]["critic_warmup_steps"])
            self.assertEqual(8000, cde["critic_warmup_steps"])
            self.assertEqual(0, cde["critic_warmup_requested_steps"])
            self.assertEqual(8000, cde["critic_warmup_resume_steps"])

    def test_write_config_emits_manifest_and_resolved_toml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base.toml"
            output_path = tmp_path / "runtime" / "config.toml"
            write_toml_file(base_config_path, make_base_config(tmp_path))

            _, manifest = oracle_cde_configs.write_oracle_cde_config(
                base_config_path=base_config_path,
                output_path=output_path,
                runtime_root=tmp_path / "runtime",
                run_name="unit_C_w1000_s3000",
                tower="single_tower",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                remote_port=5101,
            )

            self.assertTrue(output_path.is_file())
            manifest_path = output_path.with_name("manifest.json")
            self.assertTrue(manifest_path.is_file())
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loaded_config = load_toml_file(output_path)
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual("unit_C_w1000_s3000", loaded_manifest["run_name"])
            self.assertEqual("current_config", loaded_manifest["recommended_role_args"][-1])
            self.assertEqual(500, loaded_manifest["monitor_every_steps"])
            self.assertEqual(512, loaded_manifest["batch_size"])
            self.assertEqual(0.05, loaded_manifest["value_weight"])
            self.assertEqual(0.2, loaded_manifest["policy_clip_ratio"])
            self.assertEqual(0.0, loaded_manifest["entropy_floor"])
            self.assertEqual(0.0, loaded_manifest["entropy_adjust_rate"])
            self.assertEqual(0, loaded_manifest["entropy_floor_start_step"])
            self.assertEqual(0.0, loaded_manifest["vtrace_target_rho_clip"])
            self.assertEqual(0.0, loaded_manifest["vtrace_target_c_clip"])
            self.assertEqual("auto", loaded_manifest["vtrace_mode"])
            self.assertEqual(2, loaded_manifest["vtrace_min_version_gap"])
            self.assertEqual(1000, loaded_manifest["critic_warmup_steps"])
            self.assertEqual(1000, loaded_manifest["critic_warmup_requested_steps"])
            self.assertEqual(0, loaded_manifest["critic_warmup_resume_steps"])
            self.assertEqual("validation", loaded_manifest["opponent_pool_preset"])
            self.assertFalse(loaded_manifest["allow_cudnn_benchmark"])
            self.assertEqual(3000, loaded_manifest["test_every_steps"])
            self.assertEqual(5101, loaded_config["online"]["remote"]["port"])
            self.assertEqual("C", loaded_config["oracle_cde_experiment"]["arm"])

    def test_write_e_config_uses_resume_checkpoint_steps_for_warmup(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base.toml"
            output_path = tmp_path / "runtime" / "config.toml"
            resume_state_file = tmp_path / "source" / "mortal.pth"
            oracle_checkpoint = tmp_path / "oracle_best.pth"
            resume_state_file.parent.mkdir(parents=True)
            torch.save({"steps": 5000, "optimizer": {}, "scheduler": {}}, resume_state_file)
            oracle_checkpoint.write_bytes(b"oracle-checkpoint")
            write_toml_file(base_config_path, make_base_config(tmp_path))

            _, manifest = oracle_cde_configs.write_oracle_cde_config(
                base_config_path=base_config_path,
                output_path=output_path,
                runtime_root=tmp_path / "runtime",
                run_name="dual_E_w3000_s12000_resume5000",
                tower="dual_tower",
                arm="E",
                max_steps=12000,
                oracle_critic_state_file=oracle_checkpoint,
                critic_warmup_steps=3000,
                remote_port=5103,
                resume_state_file=resume_state_file,
            )

            loaded_config = load_toml_file(output_path)
            self.assertEqual(8000, loaded_config["value"]["critic_warmup_steps"])
            self.assertEqual(8000, loaded_config["oracle_cde_experiment"]["critic_warmup_steps"])
            self.assertEqual(3000, manifest["critic_warmup_requested_steps"])
            self.assertEqual(5000, manifest["critic_warmup_resume_steps"])
            self.assertEqual(8000, manifest["critic_warmup_steps"])

    def test_write_config_can_stage_resume_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base.toml"
            output_path = tmp_path / "runtime" / "config.toml"
            resume_state_file = tmp_path / "source" / "mortal.pth"
            resume_state_file.parent.mkdir(parents=True)
            torch.save({"steps": 500, "optimizer": {}, "scheduler": {}}, resume_state_file)
            write_toml_file(base_config_path, make_base_config(tmp_path))

            _, manifest = oracle_cde_configs.write_oracle_cde_config(
                base_config_path=base_config_path,
                output_path=output_path,
                runtime_root=tmp_path / "runtime",
                run_name="unit_C_w1000_s3000_resume",
                tower="single_tower",
                arm="C",
                max_steps=3000,
                critic_warmup_steps=1000,
                remote_port=5102,
                resume_state_file=resume_state_file,
            )

            loaded_config = load_toml_file(output_path)
            staged_state_file = Path(loaded_config["control"]["state_file"])
            staged_state = torch.load(staged_state_file, weights_only=False, map_location="cpu")
            self.assertEqual(500, staged_state["steps"])
            self.assertEqual(str(resume_state_file.resolve()), manifest["resume_state_file"])
            self.assertEqual(1500, loaded_config["value"]["critic_warmup_steps"])
            self.assertEqual(1500, manifest["critic_warmup_steps"])
            self.assertEqual(1000, manifest["critic_warmup_requested_steps"])
            self.assertEqual(500, manifest["critic_warmup_resume_steps"])

    def test_stage_resume_checkpoint_refuses_to_overwrite_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.pth"
            target = tmp_path / "runtime" / "checkpoints" / "mortal.pth"
            source.write_bytes(b"new")
            target.parent.mkdir(parents=True)
            target.write_bytes(b"old")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                oracle_cde_configs.stage_resume_state_file(source, target)
            self.assertEqual(b"old", target.read_bytes())


if __name__ == "__main__":
    unittest.main()

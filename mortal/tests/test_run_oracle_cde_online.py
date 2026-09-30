import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_oracle_cde_online
from mortal.core.toml_utils import write_toml_file


class RunOracleCdeOnlineTests(unittest.TestCase):
    def test_normalize_runner_arm_accepts_matching_cde_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            write_toml_file(config_path, {"oracle_cde_experiment": {"arm": "D"}})

            self.assertEqual(
                "current_config",
                run_oracle_cde_online.normalize_runner_arm(config_path, "D"),
            )

    def test_normalize_runner_arm_rejects_mismatched_cde_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            write_toml_file(config_path, {"oracle_cde_experiment": {"arm": "D"}})

            with self.assertRaisesRegex(ValueError, "does not match"):
                run_oracle_cde_online.normalize_runner_arm(config_path, "E")

    def test_normalize_runner_arm_leaves_oracle_experiment_arm_intact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            write_toml_file(config_path, {"oracle_cde_experiment": {"arm": "D"}})

            self.assertEqual(
                "critic_only",
                run_oracle_cde_online.normalize_runner_arm(config_path, "critic_only"),
            )

    def test_parse_console_summary_extracts_progress_and_rank(self):
        console = {
            "trainer": "\n".join(
                [
                    "2026 INFO total steps: 500 (~0)",
                    "2026 INFO param has been submitted: version=3",
                    "2026 INFO total steps: 1,000 (~0)",
                    "2026 INFO reached configured max steps=3,000; stopping online training after publishing version=9",
                ]
            ),
            "client": "2026 INFO trainee rankings: [10 20 30 40] (3.000000, -45.000000pt)",
            "server": "",
        }

        parsed = run_oracle_cde_online.parse_console_summary(console)

        self.assertEqual(3000, parsed["latest_step"])
        self.assertEqual(3, parsed["last_submit_version"])
        self.assertEqual([10, 20, 30, 40], parsed["last_client_rank"]["rankings"])
        self.assertEqual(3.0, parsed["last_client_rank"]["avg_rank"])
        self.assertFalse(parsed["trainer_error"])

    def test_parse_console_summary_marks_traceback_as_error(self):
        parsed = run_oracle_cde_online.parse_console_summary(
            {
                "trainer": "Traceback (most recent call last): RuntimeError: boom",
                "client": "",
                "server": "",
            }
        )

        self.assertTrue(parsed["trainer_error"])
        self.assertEqual(0, parsed["latest_step"])

    def test_parse_console_summary_can_ignore_shutdown_disconnect(self):
        parsed = run_oracle_cde_online.parse_console_summary(
            {
                "trainer": "2026 INFO reached configured max steps=50,000; stopping online training",
                "client": "\n".join(
                    [
                        "Traceback (most recent call last):",
                        "  File \"client.py\", line 89, in main",
                        "    rsp = recv_msg(conn)",
                        "ConnectionResetError: [WinError 10054] remote host closed",
                    ]
                ),
                "server": "",
            },
            allow_shutdown_disconnect=True,
        )

        self.assertFalse(parsed["client_error"])
        self.assertEqual(50000, parsed["latest_step"])

    def test_parse_console_summary_keeps_shutdown_disconnect_error_by_default(self):
        parsed = run_oracle_cde_online.parse_console_summary(
            {
                "trainer": "",
                "client": "Traceback (most recent call last):\nConnectionResetError: [WinError 10054]",
                "server": "",
            }
        )

        self.assertTrue(parsed["client_error"])

    def test_resource_stop_reason_requires_consecutive_soft_breaches(self):
        args = Namespace(
            max_system_mem_percent=85.0,
            max_gpu_mem_mb=15000.0,
            hard_system_mem_percent=92.0,
            hard_gpu_mem_mb=16000.0,
            resource_breach_samples=3,
        )
        counts = {}

        for _ in range(2):
            reason = run_oracle_cde_online.resource_stop_reason(
                sample={"system_mem_percent": 86.0, "gpu_mem_used_mb": 12000.0},
                reached_target=False,
                args=args,
                breach_counts=counts,
            )
            self.assertEqual("", reason)

        reason = run_oracle_cde_online.resource_stop_reason(
            sample={"system_mem_percent": 86.0, "gpu_mem_used_mb": 12000.0},
            reached_target=False,
            args=args,
            breach_counts=counts,
        )
        self.assertEqual("system_mem_percent>=85.0x3", reason)

    def test_resource_stop_reason_resets_soft_breach_count(self):
        args = Namespace(
            max_system_mem_percent=85.0,
            max_gpu_mem_mb=15000.0,
            hard_system_mem_percent=92.0,
            hard_gpu_mem_mb=16000.0,
            resource_breach_samples=2,
        )
        counts = {}

        run_oracle_cde_online.resource_stop_reason(
            sample={"system_mem_percent": 86.0, "gpu_mem_used_mb": 12000.0},
            reached_target=False,
            args=args,
            breach_counts=counts,
        )
        reason = run_oracle_cde_online.resource_stop_reason(
            sample={"system_mem_percent": 80.0, "gpu_mem_used_mb": 12000.0},
            reached_target=False,
            args=args,
            breach_counts=counts,
        )
        self.assertEqual("", reason)
        self.assertEqual(0, counts["system_mem_percent"])

    def test_resource_stop_reason_hard_limit_stops_immediately(self):
        args = Namespace(
            max_system_mem_percent=85.0,
            max_gpu_mem_mb=15000.0,
            hard_system_mem_percent=92.0,
            hard_gpu_mem_mb=16000.0,
            resource_breach_samples=3,
        )

        reason = run_oracle_cde_online.resource_stop_reason(
            sample={"system_mem_percent": 93.0, "gpu_mem_used_mb": 12000.0},
            reached_target=False,
            args=args,
            breach_counts={},
        )
        self.assertEqual("system_mem_percent>=92.0", reason)

    def test_resource_stop_reason_ignores_limits_after_target(self):
        args = Namespace(
            max_system_mem_percent=85.0,
            max_gpu_mem_mb=15000.0,
            hard_system_mem_percent=92.0,
            hard_gpu_mem_mb=16000.0,
            resource_breach_samples=3,
        )
        counts = {"system_mem_percent": 2}

        reason = run_oracle_cde_online.resource_stop_reason(
            sample={"system_mem_percent": 99.0, "gpu_mem_used_mb": 20000.0},
            reached_target=True,
            args=args,
            breach_counts=counts,
        )
        self.assertEqual("", reason)
        self.assertEqual({}, counts)

    def test_process_snapshot_uses_max_system_memory_seen_across_roles(self):
        class DummyProc:
            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return None

        role_metrics = {
            10: (
                {
                    "process_count": 1,
                    "tree_cpu_percent": 10.0,
                    "tree_rss_gb": 1.0,
                    "system_cpu_percent": 20.0,
                    "system_mem_used_gb": 20.0,
                    "system_mem_percent": 70.0,
                },
                {10},
            ),
            20: (
                {
                    "process_count": 1,
                    "tree_cpu_percent": 20.0,
                    "tree_rss_gb": 2.0,
                    "system_cpu_percent": 80.0,
                    "system_mem_used_gb": 28.0,
                    "system_mem_percent": 88.0,
                },
                {20},
            ),
            30: (
                {
                    "process_count": 1,
                    "tree_cpu_percent": 30.0,
                    "tree_rss_gb": 3.0,
                    "system_cpu_percent": 60.0,
                    "system_mem_used_gb": 25.0,
                    "system_mem_percent": 82.0,
                },
                {30},
            ),
        }

        with patch.object(
            run_oracle_cde_online,
            "merge_windows_and_tree_metrics",
            side_effect=lambda pid: role_metrics[pid],
        ), patch.object(
            run_oracle_cde_online,
            "query_gpu_metrics",
            return_value={},
        ):
            snapshot, pids = run_oracle_cde_online.process_snapshot(
                {
                    "server": DummyProc(10),
                    "trainer": DummyProc(20),
                    "client": DummyProc(30),
                }
            )

        self.assertEqual({10, 20, 30}, pids)
        self.assertEqual(3, snapshot["process_count"])
        self.assertEqual(60.0, snapshot["tree_cpu_percent"])
        self.assertEqual(6.0, snapshot["tree_rss_gb"])
        self.assertEqual(80.0, snapshot["system_cpu_percent"])
        self.assertEqual(28.0, snapshot["system_mem_used_gb"])
        self.assertEqual(88.0, snapshot["system_mem_percent"])


if __name__ == "__main__":
    unittest.main()

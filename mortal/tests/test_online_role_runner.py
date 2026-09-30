import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.config as config_module
import mortal.online.online_role_runner as online_role_runner


class OnlineRoleRunnerTests(unittest.TestCase):
    def test_runtime_config_overrides_update_canonical_config_module(self):
        cfg = config_module.config
        original_control = copy.deepcopy(cfg.get("control"))
        original_baseline = copy.deepcopy(cfg.get("baseline"))
        try:
            online_role_runner.apply_runtime_config_overrides(
                control_device="cuda:runner",
                baseline_train_device="cuda:baseline-train",
                baseline_test_device="cuda:baseline-test",
                train_play_profile=None,
            )

            self.assertEqual("cuda:runner", config_module.config["control"]["device"])
            self.assertEqual(
                "cuda:baseline-train",
                config_module.config["baseline"]["train"]["device"],
            )
            self.assertEqual(
                "cuda:baseline-test",
                config_module.config["baseline"]["test"]["device"],
            )
        finally:
            if original_control is None:
                cfg.pop("control", None)
            else:
                cfg["control"] = original_control
            if original_baseline is None:
                cfg.pop("baseline", None)
            else:
                cfg["baseline"] = original_baseline


if __name__ == "__main__":
    unittest.main()

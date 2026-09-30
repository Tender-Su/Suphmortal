import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mortal.core.toml_utils import load_toml_file, write_toml_file
from scripts.build_oracle_cde_1v3_config import build_1v3_config


class BuildOracleCde1v3ConfigTests(unittest.TestCase):
    def test_relative_eval_paths_are_resolved_against_base_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_dir = tmp_path / "base"
            output_dir = tmp_path / "runtime" / "eval"
            challenger_state = tmp_path / "runs" / "mortal.pth"
            base_config = base_dir / "config.toml"
            output = output_dir / "config.toml"
            log_dir = output_dir / "logs"
            challenger_state.parent.mkdir(parents=True)
            base_dir.mkdir(parents=True)
            write_toml_file(
                base_config,
                {
                    "1v3": {
                        "log_dir": "./logs/old",
                        "challenger": {
                            "state_file": "./checkpoints/old_challenger.pth",
                        },
                        "champion": {
                            "state_file": "./checkpoints/baseline.pth",
                        },
                    },
                },
            )

            build_1v3_config(
                base_config=base_config,
                output=output,
                challenger_state=challenger_state,
                log_dir=log_dir,
            )

            cfg = load_toml_file(output)["1v3"]
            self.assertEqual(str(log_dir.resolve()), cfg["log_dir"])
            self.assertEqual(str(challenger_state.resolve()), cfg["challenger"]["state_file"])
            self.assertEqual(
                str((base_dir / "checkpoints" / "baseline.pth").resolve()),
                cfg["champion"]["state_file"],
            )


if __name__ == "__main__":
    unittest.main()

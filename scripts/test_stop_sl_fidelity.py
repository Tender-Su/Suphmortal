import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stop_sl_fidelity as stop_helper


class StopStage05FidelityTests(unittest.TestCase):
    def test_matches_explicit_run_name(self):
        command = (
            '"C:\\ProgramData\\anaconda3\\envs\\mortal\\python.exe" '
            'mortal\\run_sl_fidelity.py --run-name experiment_a'
        )
        self.assertTrue(stop_helper.command_targets_run(command, 'experiment_a'))
        self.assertFalse(stop_helper.command_targets_run(command, 'experiment_b'))

    def test_matches_default_run_name_when_flag_is_omitted(self):
        command = (
            '"C:\\ProgramData\\anaconda3\\envs\\mortal\\python.exe" '
            'mortal\\run_sl_fidelity.py --skip-formal'
        )
        self.assertTrue(stop_helper.command_targets_run(command, stop_helper.DEFAULT_RUN_NAME))

    def test_omitted_run_name_does_not_match_non_default_target(self):
        command = (
            '"C:\\ProgramData\\anaconda3\\envs\\mortal\\python.exe" '
            'mortal\\run_sl_fidelity.py --skip-formal'
        )
        self.assertFalse(stop_helper.command_targets_run(command, 'custom_run'))


if __name__ == '__main__':
    unittest.main()

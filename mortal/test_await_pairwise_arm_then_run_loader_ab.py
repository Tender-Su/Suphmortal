import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import await_pairwise_arm_then_run_loader_ab as watcher


class AwaitPairwiseArmThenRunLoaderAbTests(unittest.TestCase):
    def test_wait_for_arm_completion_allows_watcher_to_start_before_pairwise_dir_exists(self):
        run_name = 'unit_run'
        arm_name = 'arm_a'

        with tempfile.TemporaryDirectory() as tmp_dir:
            pairwise_root = Path(tmp_dir)
            expected_pairwise_dir = pairwise_root / f'{run_name}_p1_pairwise_001'
            expected_arm_result = expected_pairwise_dir / arm_name / 'arm_result.json'
            sleep_calls: list[int] = []

            def fake_sleep(seconds: int) -> None:
                sleep_calls.append(seconds)
                expected_arm_result.parent.mkdir(parents=True, exist_ok=True)
                expected_arm_result.write_text('{}', encoding='utf-8', newline='\n')

            with (
                patch.object(watcher, 'PAIRWISE_ROOT', pairwise_root),
                patch.object(watcher.time, 'sleep', side_effect=fake_sleep),
            ):
                pairwise_dir, arm_result_path = watcher.wait_for_arm_completion(
                    run_name,
                    arm_name,
                    poll_seconds=7,
                )

        self.assertEqual(expected_pairwise_dir, pairwise_dir)
        self.assertEqual(expected_arm_result, arm_result_path)
        self.assertEqual([7], sleep_calls)

    def test_stop_owned_pairwise_controller_kills_only_live_owned_lock_pid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / 'run.lock.json').write_text(
                json.dumps({'pid': 4242}, ensure_ascii=False, indent=2),
                encoding='utf-8',
                newline='\n',
            )

            with (
                patch.object(watcher.fidelity, 'lock_belongs_to_running_process', return_value=True),
                patch.object(watcher, 'kill_process_tree') as kill_mock,
                patch.object(watcher.time, 'sleep') as sleep_mock,
            ):
                pid = watcher.stop_owned_pairwise_controller(run_dir)

        self.assertEqual(4242, pid)
        kill_mock.assert_called_once_with(4242)
        sleep_mock.assert_called_once_with(3)

    def test_stop_owned_pairwise_controller_skips_stale_lock_pid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / 'run.lock.json').write_text(
                json.dumps({'pid': 4242}, ensure_ascii=False, indent=2),
                encoding='utf-8',
                newline='\n',
            )

            with (
                patch.object(watcher.fidelity, 'lock_belongs_to_running_process', return_value=False),
                patch.object(watcher, 'kill_process_tree') as kill_mock,
                patch.object(watcher.time, 'sleep') as sleep_mock,
            ):
                pid = watcher.stop_owned_pairwise_controller(run_dir)

        self.assertIsNone(pid)
        kill_mock.assert_not_called()
        sleep_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()

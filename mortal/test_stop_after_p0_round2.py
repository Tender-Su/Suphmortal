import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / 'scripts' / 'stop_after_p0_round2.ps1'
POWERSHELL_EXE = shutil.which('powershell') or shutil.which('pwsh')


def watcher_command(run_name: str) -> list[str]:
    command = [POWERSHELL_EXE]
    if not Path(POWERSHELL_EXE).name.lower().startswith('pwsh'):
        command.extend(['-ExecutionPolicy', 'Bypass'])
    command.extend([
        '-File',
        str(SCRIPT_PATH),
        '-RunName',
        run_name,
        '-PollSeconds',
        '1',
    ])
    return command


@unittest.skipIf(POWERSHELL_EXE is None, 'PowerShell is not available')
class StopAfterP0Round2WatcherTests(unittest.TestCase):
    def test_watcher_ignores_stale_existing_round2_file(self):
        run_name = f'test_stale_round2_{os.getpid()}_{time.time_ns()}'
        run_dir = REPO_ROOT / 'logs' / 'sl_fidelity' / run_name
        target_json = run_dir / 'p0_round2.json'
        log_path = run_dir / 'stop_after_round2.log'
        run_dir.mkdir(parents=True, exist_ok=True)
        target_json.write_text('{}\n', encoding='utf-8', newline='\n')
        stale_time = time.time() - 3600
        os.utime(target_json, (stale_time, stale_time))

        proc = subprocess.Popen(
            watcher_command(run_name),
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)

        log_text = log_path.read_text(encoding='utf-8')
        self.assertIn('Ignoring stale existing p0_round2.json', log_text)
        self.assertNotIn('Detected p0_round2.json. Stopping fidelity runner before round3.', log_text)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

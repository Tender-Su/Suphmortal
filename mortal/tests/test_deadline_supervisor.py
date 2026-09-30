"""No GPU dependencies: real short-lived processes exercise lease ownership."""
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts' / 'run_deadline_supervisor.py'
SPEC = importlib.util.spec_from_file_location('deadline_supervisor', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DeadlineSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup_directory)
        self.root = Path(self.tmp.name)
        self.output = self.root / 'run'
        self.source = self.root / 'source'
        self.source.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        (self.source / 'fixture.txt').write_text('clean fixture')
        subprocess.run(['git', '-C', str(self.source), 'add', 'fixture.txt'], check=True)
        subprocess.run(['git', '-C', str(self.source), '-c', 'user.name=Test', '-c',
                        'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        self.commit = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], text=True).strip()

    def cleanup_directory(self):
        # A detached owner writes its final manifest just before interpreter exit.
        # Windows can still hold supervisor.log for that short interval. Retry only
        # a bounded sharing violation; a persistent process/handle leak still fails.
        until = time.monotonic() + 3
        while True:
            try:
                self.tmp.cleanup()
                return
            except PermissionError as exc:
                if os.name != 'nt' or getattr(exc, 'winerror', None) != 32 or time.monotonic() >= until:
                    raise
                time.sleep(.05)

    def spec(self, commands, seconds=4, grace=0.5):
        value = {'deadline_utc': (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)).isoformat(),
                 'grace_seconds': grace, 'posix_process_group_acknowledged': True,
                 'roles': [{'name': f'role{i}', 'argv': [sys.executable, '-c', code], 'cwd': str(self.source)}
                           for i, code in enumerate(commands)]}
        path = self.root / 'spec.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def run_spec(self, path, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), 'run', '--spec', str(path),
                               '--output', str(self.output), *extra], capture_output=True, text=True, timeout=12)

    def manifest(self):
        return json.loads((self.output / 'manifest.json').read_text())

    def wait_finished(self):
        until = time.monotonic() + 12
        while time.monotonic() < until:
            result = self.manifest()
            if result.get('finished_utc'):
                return result
            time.sleep(0.05)
        self.fail('supervisor did not finish')

    def test_rejects_late_start_and_naive_deadline(self):
        path = self.spec(['pass'], seconds=-1)
        result = self.run_spec(path)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            MODULE.parse_deadline('2026-10-08T11:00:00')

    def test_atomic_manifest_retries_transient_windows_reader_conflict(self):
        target = self.root / 'status.json'
        MODULE.atomic_json(target, {'state': 'old'})
        original_replace = os.replace
        failures = [5, 32]
        def replace(source, destination):
            if failures:
                error = PermissionError('temporary Windows sharing conflict')
                error.winerror = failures.pop(0)
                raise error
            original_replace(source, destination)
        with patch.object(MODULE.os, 'replace', side_effect=replace), patch.object(MODULE.time, 'sleep'):
            MODULE.atomic_json(target, {'state': 'new'})
        self.assertEqual(json.loads(target.read_text()), {'state': 'new'})

    def test_atomic_manifest_persistent_denial_preserves_previous_file(self):
        target = self.root / 'status.json'
        MODULE.atomic_json(target, {'state': 'old'})
        error = PermissionError('persistent denial')
        error.winerror = 5
        with patch.object(MODULE.os, 'replace', side_effect=error) as replace, patch.object(MODULE.time, 'sleep'):
            with self.assertRaises(PermissionError):
                MODULE.atomic_json(target, {'state': 'new'})
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(json.loads(target.read_text()), {'state': 'old'})
        self.assertEqual(list(self.root.glob('status.json.*.tmp')), [])

    def test_rejects_dirty_source_unless_explicit_smoke_override(self):
        path = self.spec(['pass'])
        (self.source / 'fixture.txt').write_text('dirty')
        result = self.run_spec(path, '--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('dirty source refused', result.stderr)
        spec = json.loads(path.read_text())
        spec['allow_dirty'] = True
        path.write_text(json.dumps(spec))
        result = self.run_spec(path, '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['roles'][0]['allow_dirty'])

    def test_expected_commit_global_and_role_override(self):
        path = self.spec(['pass'])
        spec = json.loads(path.read_text())
        spec['expected_commit'] = '0' * 40
        path.write_text(json.dumps(spec))
        result = self.run_spec(path, '--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('expected_commit mismatch', result.stderr)
        spec['roles'][0]['expected_commit'] = self.commit
        path.write_text(json.dumps(spec))
        result = self.run_spec(path, '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launch_rechecks_source_and_config(self):
        path = self.spec(['pass'])
        config = self.root / 'config.toml'
        config.write_text('x=1')
        spec = json.loads(path.read_text())
        spec['roles'][0]['config'] = str(config)
        path.write_text(json.dumps(spec))
        manifest = MODULE.prepare(path, self.output)
        config.write_text('x=2')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            MODULE.verify_role(manifest['roles'][0])
        config.write_text('x=1')
        (self.source / 'fixture.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'dirty source refused'):
            MODULE.verify_role(manifest['roles'][0])

    def test_dry_run_does_not_create_run_or_launch(self):
        path = self.spec(['raise RuntimeError("must not run")'])
        result = self.run_spec(path, '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest['roles'][0]['argv'][0], sys.executable)
        self.assertTrue(manifest['roles'][0]['source_commit'])

    def test_multiple_roles_complete_and_record_identity(self):
        path = self.spec(['print("one")', 'print("two")'])
        result = self.run_spec(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(len(manifest['role_results']), 2)
        self.assertTrue(manifest['owner_pid'])
        self.assertEqual((self.output / 'role0.stdout.log').read_text().strip(), 'one')
        # Existing leases cannot be overwritten.
        self.assertEqual(self.run_spec(path).returncode, 2)

    def test_deadline_graceful_stop_and_sl_pause_contract(self):
        code = ('import os,time,pathlib,sys; '
                'p=pathlib.Path(os.environ["MORTAL_STOP_FILE"]); '
                'assert str(p)==os.environ["MORTAL_ORACLE_PAUSE_FILE"]; '
                '\nwhile not p.exists(): time.sleep(.02)\n'
                'pathlib.Path("' + str(self.root / 'saved').replace('\\', '\\\\') + '").write_text("partial"); '
                'sys.exit(75)')
        result = self.run_spec(self.spec([code], seconds=2, grace=0.6))
        self.assertEqual(result.returncode, 124, result.stderr)
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'deadline')
        self.assertTrue(manifest['graceful_exit'])
        self.assertTrue((self.root / 'saved').exists())

    def test_manual_stop_accepts_rl_terminal_code(self):
        code = ('import os,pathlib,time; p=pathlib.Path(os.environ["MORTAL_STOP_FILE"]); '
                'p.touch(); time.sleep(.2); raise SystemExit(87)')
        result = self.run_spec(self.spec([code]))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.manifest()['status'], 'interrupted')
        self.assertEqual(self.manifest()['role_results']['role0']['returncode'], 87)
        self.assertTrue(self.manifest()['graceful_exit'])

    def test_noncooperative_deadline_preserves_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: (unrelated.kill(), unrelated.wait()) if unrelated.poll() is None else None)
        started = time.monotonic()
        result = self.run_spec(self.spec(['import time; time.sleep(30)'], seconds=2, grace=0.3))
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(self.manifest()['forced_stop'])
        self.assertIsNone(unrelated.poll())

    def test_failure_stops_other_roles(self):
        result = self.run_spec(self.spec(['raise SystemExit(7)', 'import time; time.sleep(30)']))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.manifest()['status'], 'external_failure')
        self.assertEqual(self.manifest()['role_results']['role0']['returncode'], 7)

    def test_designated_completion_stops_owned_services_only(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: (unrelated.kill(), unrelated.wait()) if unrelated.poll() is None else None)
        path = self.spec(['print("trainer complete")', 'import time; time.sleep(30)'], seconds=8, grace=.3)
        spec = json.loads(path.read_text())
        spec['roles'][0]['stop_when_complete'] = True
        path.write_text(json.dumps(spec))
        started = time.monotonic()
        result = self.run_spec(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(time.monotonic() - started, 5)
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(manifest['completion_roles'], ['role0'])
        self.assertEqual(manifest['role_results']['role0']['returncode'], 0)
        self.assertTrue(manifest['forced_stop'])
        self.assertIsNone(unrelated.poll())

    def test_ordinary_zero_exit_does_not_complete_other_roles(self):
        result = self.run_spec(self.spec(['pass', 'import time; time.sleep(30)'], seconds=2, grace=.3))
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertEqual(self.manifest()['status'], 'deadline')

    def test_designated_failure_is_not_success(self):
        path = self.spec(['raise SystemExit(7)', 'import time; time.sleep(30)'])
        spec = json.loads(path.read_text())
        spec['roles'][0]['stop_when_complete'] = True
        path.write_text(json.dumps(spec))
        result = self.run_spec(path)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.manifest()['status'], 'external_failure')

    def test_detached_owner_outlives_launcher(self):
        result = self.run_spec(self.spec(['import time; time.sleep(30)'], seconds=2, grace=0.3), '--detach')
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.wait_finished()
        self.assertEqual(manifest['status'], 'deadline')
        self.assertTrue(manifest['forced_stop'])

    def test_owner_crash_closes_pipe_and_stops_descendants(self):
        heartbeat = self.root / 'heartbeat'
        grandchild = 'import pathlib,time; p=pathlib.Path(' + repr(str(heartbeat)) + ');\nwhile True: p.write_text(str(time.time())); time.sleep(.03)'
        code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(grandchild) + ']); time.sleep(30)'
        path = self.spec([code], seconds=9)
        owner = subprocess.Popen([sys.executable, str(SCRIPT), 'run', '--spec', str(path),
                                  '--output', str(self.output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (owner.kill(), owner.wait()) if owner.poll() is None else None)
        until = time.monotonic() + 4
        while not heartbeat.exists() and time.monotonic() < until:
            time.sleep(.03)
        self.assertTrue(heartbeat.exists())
        owner.kill()  # Exact Popen process handle on Windows, unreaped child on POSIX.
        owner.wait(timeout=3)
        time.sleep(.25)
        last = heartbeat.read_text()
        time.sleep(.2)
        self.assertEqual(heartbeat.read_text(), last)


if __name__ == '__main__':
    unittest.main()

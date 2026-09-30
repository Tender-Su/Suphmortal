"""Opt-in Windows desktop smoke tests; no training or GPU work is started.

Run with MORTAL_TEST_DETACHED_LAUNCH=1 in an authorized desktop session.
"""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/start_oracle_critic_detached.ps1"


@unittest.skipUnless(
    os.name == "nt" and os.environ.get("MORTAL_TEST_DETACHED_LAUNCH") == "1",
    "requires an explicitly enabled Windows desktop session",
)
class DetachedLauncherTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="detached launcher ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spec = self.root / "spec with spaces.json"
        self.spec.write_text(json.dumps({
            "repo_root": str(ROOT),
            "search_root": str(self.root),
            "status_file": str(self.root / "status.json"),
        }), encoding="utf-8")
        self.pwsh = shutil.which("pwsh")
        self.assertIsNotNone(self.pwsh)
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            ("CreateFileW", [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                             wintypes.HANDLE], wintypes.HANDLE),
        ):
            function = getattr(self.api, name)
            function.argtypes = args
            function.restype = result

    def wait_receipt(self, launch_id, state=None):
        path = self.root / f"detached_launch_{launch_id}.json"
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if state is None or receipt["state"] == state:
                    return receipt
            time.sleep(0.1)
        self.fail(f"Missing receipt state {state}: {path}")

    def test_survives_caller_job_termination(self):
        launch_id = uuid.uuid4().hex
        quote = lambda s: "'" + str(s).replace("'", "''") + "'"
        command = (
            f"Start-Sleep -Seconds 2; & {quote(LAUNCHER)} -Probe "
            f"-SpecPath {quote(self.spec)} -LaunchId '{launch_id}'; Start-Sleep -Seconds 30"
        )
        job = self.api.CreateJobObjectW(None, None)
        self.assertTrue(job)
        process = subprocess.Popen(
            [self.pwsh, "-NoProfile", "-Command", command],
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(self.api.AssignProcessToJobObject(job, int(process._handle)))
            receipt = self.wait_receipt(launch_id, "isolated")
            self.assertFalse(receipt["in_job"])
            self.assertGreater(receipt["session_id"], 0)
            self.assertTrue(self.api.TerminateJobObject(job, 99))
            self.assertEqual(process.wait(timeout=5), 99)
            self.wait_receipt(launch_id, "probe_completed")
        finally:
            self.api.TerminateJobObject(job, 99)
            self.api.CloseHandle(job)
            process.wait(timeout=5)

    def test_existing_supervisor_lock_refuses_duplicate(self):
        launch_id = uuid.uuid4().hex
        lock = self.api.CreateFileW(
            str(self.root / "apex_supervisor.lock"), 0xC0000000, 0, None, 4, 0x80, None
        )
        self.assertNotEqual(lock, ctypes.c_void_p(-1).value)
        try:
            result = subprocess.run(
                [self.pwsh, "-NoProfile", "-File", str(LAUNCHER),
                 "-SpecPath", str(self.spec), "-LaunchId", launch_id],
                creationflags=subprocess.CREATE_NO_WINDOW,
                capture_output=True, text=True, timeout=35,
            )
            self.assertNotEqual(result.returncode, 0)
            receipt = self.wait_receipt(launch_id, "error")
            self.assertIn("apex_supervisor.lock", receipt["detail"])
        finally:
            self.api.CloseHandle(lock)


if __name__ == "__main__":
    unittest.main()

"""CPU-only guard I/O tests, including real Windows file-sharing conflicts."""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from mortal.core import artifacts
from scripts.run_sl_phase_fork_control import existing_guard, read_guard_json, write_guard_json


UPSTREAM = '''import json, os
from pathlib import Path
D = Path('.')
LEASE = D / 'lease'
LIMITS = {'minimum_available_ram_bytes': 2147483648, 'minimum_global_vram_free_mib': 1024}
APEX_NAMES = {'r5apex.exe', 'r5apex_dx12.exe'}
def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
def put(name, value):
    path = D / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)
def guard():
    try:
        put('guard_status.json', {'samples': 2})
        return 0
    except BaseException as exc:
        put('guard_failure.json', {'error': repr(exc)})
        (LEASE / 'STOP').touch()
        return 76
'''


@contextmanager
def windows_reader(path, share):
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, share, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield handle
    finally:
        kernel.CloseHandle(handle)


class GuardIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'attempt'
        self.directory.mkdir()
        (self.directory / 'lease').mkdir()
        self.source = self.root / 'upstream.py'
        self.source.write_text(UPSTREAM, encoding='utf8')
        self.digest = artifacts.file_sha256(self.source)
        self.guard = existing_guard(self.source, self.digest, self.directory)
        self.target = self.directory / 'guard_status.json'
        self.guard.put('guard_status.json', {'samples': 1})

    def test_adapter_preserves_source_identity_and_resource_contract(self):
        self.assertEqual(artifacts.file_sha256(self.source), self.digest)
        self.assertEqual(self.guard.LIMITS, {'minimum_available_ram_bytes': 2 * 1024**3,
                                           'minimum_global_vram_free_mib': 1024})
        self.assertEqual(self.guard.APEX_NAMES, {'r5apex.exe', 'r5apex_dx12.exe'})
        self.assertEqual(self.guard.LEASE, self.directory / 'lease')
        with self.assertRaisesRegex(ValueError, 'guard source changed'):
            existing_guard(self.source, 'foreign', self.directory)

    def test_shared_reader_and_strict_json_writer(self):
        self.guard.put('guard_status.json', {'message': '中文', 'samples': 3})
        self.assertEqual(self.guard.read(self.target), {'message': '中文', 'samples': 3})
        before = self.target.read_bytes()
        with self.assertRaises(ValueError):
            self.guard.put('guard_status.json', {'bad': float('nan')})
        self.assertEqual(self.target.read_bytes(), before)
        text = self.root / 'tail.txt'
        text.write_text('first\nlast', encoding='utf8')
        self.assertEqual(artifacts.read_shared_text(text, tail=4), 'last')
        self.assertEqual(artifacts.read_shared_text(text, tail=0), '')
        with self.assertRaises(ValueError):
            artifacts.read_shared_text(text, tail=-1)
        text.write_text('last中', encoding='utf8')
        self.assertEqual(artifacts.read_shared_text(text, tail=1), '\ufffd')
        with self.assertRaises(FileNotFoundError):
            read_guard_json(self.root / 'missing.json')
        text.write_text('{', encoding='utf8')
        with self.assertRaises(json.JSONDecodeError):
            read_guard_json(text)
        text.unlink()  # Parsing failures leave no open Windows handle.

    @unittest.skipUnless(os.name == 'nt', 'Windows retry policy')
    def test_transient_winerrors_retry_and_publish_new_status(self):
        original = os.replace
        for code in (5, 32):
            with self.subTest(winerror=code):
                attempts = []
                def replace(source, target):
                    attempts.append((source, target))
                    if len(attempts) < 3:
                        raise ctypes.WinError(code)
                    return original(source, target)
                with patch.object(artifacts.os, 'replace', side_effect=replace), \
                        patch.object(artifacts.time, 'sleep') as sleep:
                    self.guard.put('guard_status.json', {'samples': code})
                self.assertEqual(len(attempts), 3)
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(self.guard.read(self.target), {'samples': code})

    def test_unrelated_io_error_is_not_retried_or_hidden(self):
        before = self.target.read_bytes()
        with patch.object(artifacts.os, 'replace', side_effect=OSError('disk failure')) as replace, \
                patch.object(artifacts.time, 'sleep') as sleep:
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.guard.put('guard_status.json', {'samples': 99})
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.target.read_bytes(), before)

    @unittest.skipUnless(os.name == 'nt', 'real Windows file-sharing conflict')
    def test_real_blocking_reader_reproduces_original_replace_failure(self):
        candidate = self.root / 'original.tmp'
        candidate.write_text('new', encoding='utf8')
        with windows_reader(self.target, 3):
            with self.assertRaises(PermissionError) as raised:
                os.replace(candidate, self.target)
        self.assertIn(raised.exception.winerror, (5, 32))
        self.assertEqual(self.guard.read(self.target), {'samples': 1})

    @unittest.skipUnless(os.name == 'nt', 'real Windows file-sharing conflict')
    def test_real_transient_reader_lock_recovers_without_false_stop(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        def reader():
            try:
                with windows_reader(self.target, 3):
                    entered.set()
                    release.wait(5)
            except BaseException as exc:
                errors.append(exc)
                entered.set()
        thread = threading.Thread(target=reader)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(errors)
            def release_on_retry(delay):
                release.set()
                thread.join(2)
                self.assertFalse(thread.is_alive())
            with patch.object(artifacts.time, 'sleep', side_effect=release_on_retry) as retry:
                self.assertEqual(self.guard.guard(), 0)
            self.assertGreaterEqual(retry.call_count, 1)
            self.assertEqual(self.guard.read(self.target), {'samples': 2})
            self.assertFalse((self.directory / 'guard_failure.json').exists())
            self.assertFalse((self.directory / 'lease/STOP').exists())
        finally:
            release.set()
            thread.join(5)

    @unittest.skipUnless(os.name == 'nt', 'real Windows file-sharing conflict')
    def test_persistent_reader_lock_still_fails_closed_and_preserves_evidence(self):
        before = self.target.read_bytes()
        stale = self.target.with_suffix('.json.tmp')
        stale.write_bytes(b'original failed-attempt evidence')
        with windows_reader(self.target, 3), patch.object(artifacts.time, 'sleep') as retry:
            self.assertEqual(self.guard.guard(), 76)
        self.assertEqual(retry.call_count, 9)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(stale.read_bytes(), b'original failed-attempt evidence')
        self.assertTrue((self.directory / 'lease/STOP').is_file())
        self.assertIn('PermissionError', self.guard.read(self.directory / 'guard_failure.json')['error'])
        self.assertFalse(list(self.directory.glob('.guard_status.json.*.tmp')))

    @unittest.skipUnless(os.name == 'nt', 'real Windows shared-delete reader')
    def test_shared_reader_requests_delete_sharing_and_releases_handle(self):
        original = ctypes.WinDLL
        shares = []
        class Proxy:
            def __init__(self):
                self.real = original('kernel32', use_last_error=True)
            def __getattr__(self, name):
                return getattr(self.real, name)
            def CreateFileW(self, *args):
                shares.append(args[2])
                return self.real.CreateFileW(*args)
        proxy = Proxy()
        # Bound methods cannot carry argtypes: use a normal callable as the hook.
        callback = proxy.CreateFileW
        proxy.CreateFileW = lambda *args: callback(*args)
        from ctypes import wintypes
        proxy.real.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                          ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        proxy.real.CreateFileW.restype = wintypes.HANDLE
        with patch.object(ctypes, 'WinDLL', return_value=proxy):
            self.assertEqual(read_guard_json(self.target), {'samples': 1})
        self.assertEqual(shares, [7])
        with patch.object(artifacts.time, 'sleep') as sleep:
            write_guard_json(self.directory, 'guard_status.json', {'samples': 4})
        sleep.assert_not_called()
        self.assertEqual(read_guard_json(self.target), {'samples': 4})


if __name__ == '__main__':
    unittest.main()

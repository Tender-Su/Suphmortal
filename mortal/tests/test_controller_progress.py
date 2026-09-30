"""CPU-only progress failures never become workload/completion failures."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/controller_progress.py'
SPEC = importlib.util.spec_from_file_location('controller_progress', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControllerProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'progress.jsonl'
        self.progress = MODULE.ProgressJournal(self.path)

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def test_append_never_replaces_a_file_and_preserves_previous_records(self):
        with patch('os.replace', side_effect=AssertionError('no replacement')):
            self.assertTrue(self.progress.publish({'stage': 'first'}))
            self.assertTrue(self.progress.publish({'stage': 'second'}))
        self.assertEqual(self.records(), [{'stage': 'first'}, {'stage': 'second'}])
        self.assertEqual(self.progress.diagnostics(), {'write_failures': 0, 'last_error': None})

    def test_windows_denials_drop_without_retry_and_warn_once_then_recover(self):
        stderr = io.StringIO()
        for winerror in (5, 32):
            error = PermissionError('Windows observer conflict')
            error.winerror = winerror
            with patch.object(Path, 'open', side_effect=error) as opening, patch.object(MODULE.sys, 'stderr', stderr):
                self.assertFalse(self.progress.publish({'status': 'running'}))
            self.assertEqual(opening.call_count, 1)
        self.assertEqual(self.progress.failures, 2)
        self.assertEqual(stderr.getvalue().count('workload continues'), 1)
        self.assertTrue(self.progress.publish({'status': 'complete'}))
        self.assertEqual(self.records(), [{'status': 'complete'}])

    def test_failed_write_and_closed_stderr_do_not_interrupt_work(self):
        with patch.object(Path, 'open', side_effect=OSError('read-only journal')), \
                patch.object(MODULE, 'print', side_effect=BrokenPipeError(), create=True):
            self.assertFalse(self.progress.publish({'stage': 'healthy child'}))
        self.assertEqual(self.progress.failures, 1)

    def test_serialization_errors_are_not_silenced(self):
        with self.assertRaises(TypeError):
            self.progress.publish({'bad': object()})
        self.assertFalse(self.path.exists())

    def test_actual_closed_stderr_does_not_mask_a_diagnostic_write_failure(self):
        stderr = io.StringIO()
        stderr.close()
        with patch.object(Path, 'open', side_effect=PermissionError('observer conflict')), \
                patch.object(MODULE.sys, 'stderr', stderr):
            self.assertFalse(self.progress.publish({'stage': 'healthy child'}))
            self.assertFalse(MODULE.preserve_traceback(self.root / 'trace.txt', 'original failure'))

    def test_partial_previous_append_does_not_corrupt_next_complete_record(self):
        self.path.write_bytes(b'{"stage":"partial')
        self.progress.publish({'stage': 'next'})
        lines = self.path.read_text().splitlines()
        with self.assertRaises(json.JSONDecodeError):
            json.loads(lines[0])
        self.assertEqual(json.loads(lines[1]), {'stage': 'next'})

    def test_short_write_is_counted_without_retry(self):
        with patch.object(Path, 'open') as opening, patch.object(MODULE.sys, 'stderr', io.StringIO()):
            writing = opening.return_value.__enter__.return_value.write
            writing.return_value = 1
            self.assertFalse(self.progress.publish({'stage': 'running'}))
            self.assertEqual(writing.call_count, 1)
        self.assertEqual(self.progress.failures, 1)

    def test_original_trace_survives_unavailable_trace_and_status_outputs(self):
        original = RuntimeError('original child failure')
        stderr = io.StringIO()
        with self.assertRaises(RuntimeError) as raised:
            try:
                raise original
            except RuntimeError:
                text = traceback.format_exc()
                with patch.object(Path, 'open', side_effect=PermissionError('diagnostics locked')), \
                        patch.object(MODULE.sys, 'stderr', stderr):
                    self.assertFalse(MODULE.preserve_traceback(self.root / 'trace.txt', text))
                    self.assertFalse(self.progress.publish({'status': 'failed'}))
                raise
        self.assertIs(raised.exception, original)
        self.assertIn('original child failure', stderr.getvalue())

    def test_trace_saved_once_without_overwriting_previous_failure(self):
        path = self.root / 'trace.txt'
        self.assertTrue(MODULE.preserve_traceback(path, 'original'))
        with patch.object(MODULE.sys, 'stderr', io.StringIO()):
            self.assertFalse(MODULE.preserve_traceback(path, 'later'))
        self.assertEqual(path.read_text(), 'original')

    @unittest.skipUnless(os.name == 'nt', 'real Windows sharing handles')
    def test_windows_reader_without_delete_sharing_allows_progress_appends(self):
        import ctypes
        from ctypes import wintypes
        self.progress.publish({'stage': 'first'})
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # A reader sharing read/write but not delete reproduces the relevant
        # Windows mechanism without guessing who held the historical handle.
        handle = kernel.CreateFileW(str(self.path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self.assertTrue(self.progress.publish({'stage': 'second'}))
            self.assertTrue(self.progress.publish({'stage': 'third'}))
        finally:
            kernel.CloseHandle(handle)
        self.assertEqual(len(self.records()), 3)


if __name__ == '__main__':
    unittest.main()

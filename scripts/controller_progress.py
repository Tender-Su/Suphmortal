"""Best-effort controller diagnostics, never completion or ownership evidence.

One owner appends snapshots to its own JSONL journal. Readers must use complete
lines and tolerate a malformed/truncated line after a failed write. No existing
file is replaced, no write is retried, and no diagnostic error changes the
workload's result. An independent supervisor still owns the compute deadline.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def _warn(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # A closed diagnostic stream must not replace the workload error.


class ProgressJournal:
    """Single-writer, append-only progress; not a durable result/checkpoint API."""

    def __init__(self, path):
        self.path = Path(path)
        self.failures = 0
        self.last_error = None

    def publish(self, value):
        # Serialization errors are programming errors, not observer/file sharing
        # failures. Keep them visible rather than swallowing arbitrary exceptions.
        payload = ('\n' + json.dumps(value, ensure_ascii=True, separators=(',', ':')) + '\n').encode('utf-8')
        try:
            # Leading newline isolates a previous interrupted/partial append.
            # One unbuffered write, no fsync/backoff/replacement in the heartbeat.
            with self.path.open('ab', buffering=0) as stream:
                if stream.write(payload) != len(payload):
                    raise OSError('incomplete progress journal write')
        except OSError as exc:
            self.failures += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            if self.failures == 1:
                _warn(f'progress telemetry unavailable ({self.path}): {self.last_error}; workload continues')
            return False
        return True

    def diagnostics(self):
        return {'write_failures': self.failures, 'last_error': self.last_error}


def preserve_traceback(path, text):
    """Save the already-captured original trace before fallible status publishing.

    Refuse to overwrite older failure evidence. A diagnostic write failure falls
    back to stderr and never replaces the exception the caller is re-raising.
    """
    try:
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(text)
    except OSError as exc:
        _warn(f'could not save original traceback to {path}: {exc}\n{text}')
        return False
    return True

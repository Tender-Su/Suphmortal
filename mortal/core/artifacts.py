from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


WINDOWS_REPLACE_ERRORS = frozenset({5, 32})


def _replace_with_retries(
    source: Path,
    target: Path,
    *,
    attempts: int,
) -> None:
    if attempts < 1:
        raise ValueError('replace attempts must be positive')
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            retryable = (
                os.name == 'nt'
                and getattr(exc, 'winerror', None) in WINDOWS_REPLACE_ERRORS
                and attempt < attempts
            )
            if not retryable:
                raise
            time.sleep(min(1.0, 0.1 * attempt))


@contextmanager
def atomic_output_path(
    target: str | Path,
    *,
    replace_attempts: int = 10,
) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically install it on success."""
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(
        f'.{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    )
    try:
        yield temporary
        _replace_with_retries(
            temporary,
            target_path,
            attempts=replace_attempts,
        )
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_shared_text(path: str | Path, *, tail: int | None = None) -> str:
    """Read live telemetry with delete sharing and short-lived Windows handles.

    Keep the FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE contract
    used by the existing remote SL observer. The open handle is always closed;
    writer-side retries remain necessary for transient replacement failures.
    """
    if tail is not None and tail < 0:
        raise ValueError('tail must be nonnegative')
    path = Path(path)
    errors = 'replace' if tail is not None else 'strict'
    if os.name != 'nt':
        with path.open('rb') as handle:
            if tail is not None:
                handle.seek(max(0, path.stat().st_size - tail))
            return handle.read().decode('utf-8-sig', errors=errors)
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                       ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    kernel.SetFilePointerEx.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if tail is not None and not kernel.SetFilePointerEx(handle, max(0, path.stat().st_size - tail), None, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        parts = []
        while True:
            buffer = ctypes.create_string_buffer(65536)
            count = wintypes.DWORD()
            if not kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not count.value:
                return b''.join(parts).decode('utf-8-sig', errors=errors)
            parts.append(buffer.raw[:count.value])
    finally:
        kernel.CloseHandle(handle)


def atomic_write_text(target: str | Path, text: str) -> None:
    with atomic_output_path(target) as temporary:
        temporary.write_text(text, encoding='utf-8', newline='\n')


def atomic_write_json(
    target: str | Path,
    payload: Any,
    *,
    indent: int = 2,
) -> None:
    atomic_write_text(
        target,
        json.dumps(payload, ensure_ascii=False, indent=indent),
    )


def atomic_torch_save(payload: Any, target: str | Path) -> None:
    import torch

    with atomic_output_path(target) as temporary:
        torch.save(payload, temporary)


def atomic_write_toml(target: str | Path, payload: dict[str, Any]) -> None:
    from mortal.core.toml_utils import write_toml_file

    with atomic_output_path(target) as temporary:
        write_toml_file(temporary, payload)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

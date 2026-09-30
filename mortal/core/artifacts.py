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

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


EXTERNAL_PAUSE_ENV_VAR = 'MORTAL_ORACLE_PAUSE_FILE'
EXTERNAL_PAUSE_EXIT_CODE = 75


def resolve_external_pause_file(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    source = os.environ if environ is None else environ
    value = source.get(EXTERNAL_PAUSE_ENV_VAR, '').strip()
    return Path(value).resolve() if value else None


def external_pause_requested(file_path: Path | None) -> bool:
    return file_path is not None and file_path.is_file()

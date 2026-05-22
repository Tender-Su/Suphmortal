from __future__ import annotations

from pathlib import Path
from typing import Any

import toml


def load_toml_file(path: str | Path) -> dict[str, Any]:
    return toml.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_toml_file(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(toml.dumps(data), encoding='utf-8', newline='\n')

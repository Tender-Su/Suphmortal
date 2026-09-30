from __future__ import annotations

import sys
from pathlib import Path


MORTAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MORTAL_ROOT.parent


def ensure_legacy_import_path() -> None:
    """Expose repo-root and domain folders for script-style imports."""
    paths = [
        REPO_ROOT,
        MORTAL_ROOT,
        MORTAL_ROOT / "core",
        MORTAL_ROOT / "data",
        MORTAL_ROOT / "supervised",
        MORTAL_ROOT / "online",
        MORTAL_ROOT / "eval",
        MORTAL_ROOT / "research",
        REPO_ROOT / "scripts",
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mortal.core.toml_utils import load_toml_file, write_toml_file


PATH_KEYS = {
    "dir",
    "log_dir",
    "state_file",
    "tactics",
}


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_paths(node, base_dir: Path):
    if isinstance(node, dict):
        return {
            key: _resolve_paths(value, base_dir)
            if isinstance(value, dict)
            else _resolve_path(value, base_dir)
            if key in PATH_KEYS and isinstance(value, str) and value
            else value
            for key, value in node.items()
        }
    return node


def build_1v3_config(base_config: Path, output: Path, challenger_state: Path, log_dir: Path) -> None:
    base_config = base_config.resolve()
    cfg = _resolve_paths(load_toml_file(base_config), base_config.parent)
    one_vs_three = cfg.setdefault("1v3", {})
    challenger = one_vs_three.setdefault("challenger", {})

    one_vs_three["seed_key"] = 2026052350
    one_vs_three["games_per_iter"] = 2000
    one_vs_three["iters"] = 1
    one_vs_three["log_dir"] = str(log_dir.resolve())
    challenger["state_file"] = str(challenger_state.resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    write_toml_file(output, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--challenger-state", required=True)
    parser.add_argument("--log-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_1v3_config(
        base_config=Path(args.base_config),
        output=Path(args.output),
        challenger_state=Path(args.challenger_state),
        log_dir=Path(args.log_dir),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--remove", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_path)
    payload = torch.load(index_path, weights_only=True)
    remove_set = {str(Path(item)) for item in args.remove}
    removed = {}
    for key in (
        "train_files",
        "val_files",
        "monitor_recent_files",
        "full_recent_files",
        "old_regression_files",
    ):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        before = len(values)
        payload[key] = [item for item in values if str(Path(item)) not in remove_set]
        removed[key] = before - len(payload[key])
    torch.save(payload, index_path)
    print(json.dumps({"index_path": str(index_path), "removed": removed}, ensure_ascii=False))


if __name__ == "__main__":
    main()

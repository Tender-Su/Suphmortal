from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


MONTH_PATTERN = re.compile(r"^\d{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic chronological train/dev/test file indexes for "
            "Oracle critic pretraining."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dev-month", required=True, help="Held-out dev month as YYYYMM.")
    parser.add_argument("--test-month", required=True, help="Untouched test month as YYYYMM.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260416)
    return parser.parse_args()


def validate_month(value: str, *, name: str) -> str:
    value = str(value).strip()
    if not MONTH_PATTERN.fullmatch(value) or not 1 <= int(value[4:]) <= 12:
        raise ValueError(f"{name} must use YYYYMM with a valid month, got {value!r}")
    return value


def file_list_fingerprint(file_list: list[str]) -> str:
    digest = hashlib.sha256()
    for filename in file_list:
        digest.update(str(filename).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def discover_month_files(dataset_root: Path) -> dict[str, list[str]]:
    by_month: dict[str, list[str]] = {}
    for year_dir in sorted(dataset_root.iterdir()):
        if not year_dir.is_dir() or not re.fullmatch(r"\d{4}", year_dir.name):
            continue
        for month_dir in sorted(year_dir.iterdir()):
            month = month_dir.name
            if not month_dir.is_dir() or not MONTH_PATTERN.fullmatch(month):
                continue
            if not month.startswith(year_dir.name) or not 1 <= int(month[4:]) <= 12:
                raise ValueError(f"invalid year/month directory pairing: {month_dir}")
            files = []
            for source in sorted(month_dir.glob("*.json")):
                if not source.name.startswith(month):
                    raise ValueError(
                        f"JSON filename month does not match its directory: {source}"
                    )
                files.append(str(source.resolve()))
            if files:
                by_month[month] = files
    if not by_month:
        raise FileNotFoundError(f"no monthly JSON files found under {dataset_root}")
    return by_month


def build_temporal_splits(
    by_month: dict[str, list[str]],
    *,
    dev_month: str,
    test_month: str,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    dev_month = validate_month(dev_month, name="dev_month")
    test_month = validate_month(test_month, name="test_month")
    if dev_month >= test_month:
        raise ValueError("dev_month must be earlier than test_month")
    if dev_month not in by_month:
        raise ValueError(f"dev month {dev_month} is absent from the dataset")
    if test_month not in by_month:
        raise ValueError(f"test month {test_month} is absent from the dataset")

    unsupported = sorted(month for month in by_month if month > dev_month and month != test_month)
    if unsupported:
        raise ValueError(
            "months newer than the dev cutoff must be assigned explicitly; "
            f"found {unsupported}"
        )

    train_files = [
        filename
        for month in sorted(by_month)
        if month < dev_month
        for filename in by_month[month]
    ]
    dev_files = list(by_month[dev_month])
    test_files = list(by_month[test_month])
    if not train_files or not dev_files or not test_files:
        raise ValueError("temporal train/dev/test splits must all be non-empty")

    random.Random(int(seed)).shuffle(train_files)
    split_sets = tuple(map(set, (train_files, dev_files, test_files)))
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("temporal train/dev/test splits overlap")
    return train_files, dev_files, test_files


def atomic_write_json(destination: Path, payload: Any) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def atomic_torch_save(destination: Path, payload: Any) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    dev_month = validate_month(args.dev_month, name="--dev-month")
    test_month = validate_month(args.test_month, name="--test-month")
    by_month = discover_month_files(dataset_root)
    train_files, dev_files, test_files = build_temporal_splits(
        by_month,
        dev_month=dev_month,
        test_month=test_month,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {
        "train_file_index": output_dir / "train_index.pth",
        "dev_file_index": output_dir / "dev_index.pth",
        "test_file_index": output_dir / "test_index.pth",
    }
    for split, files in (
        ("train", train_files),
        ("dev", dev_files),
        ("test", test_files),
    ):
        atomic_torch_save(index_paths[f"{split}_file_index"], {"file_list": files})

    manifest = {
        "format": "oracle_temporal_file_indexes_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "seed": int(args.seed),
        "split_rule": {
            "train": f"month < {dev_month}",
            "dev": f"month == {dev_month}",
            "test": f"month == {test_month}",
        },
        "month_counts": {month: len(files) for month, files in sorted(by_month.items())},
        "splits": {
            split: {
                "files": len(files),
                "sha256": file_list_fingerprint(files),
                "index": str(index_paths[f"{split}_file_index"]),
            }
            for split, files in (
                ("train", train_files),
                ("dev", dev_files),
                ("test", test_files),
            )
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

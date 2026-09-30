from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable event cache for only the Oracle critic train split; "
            "dev and test indexes continue to reference the original logs."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--files-per-chunk", type=int, default=32)
    parser.add_argument(
        "--max-source-files",
        type=int,
        default=0,
        help="Limit the train split for a cache benchmark; zero builds all files.",
    )
    parser.add_argument("--log-every-chunks", type=int, default=100)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def file_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def chunked(values: list[str], chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield start // chunk_size, values[start:start + chunk_size]


def main() -> None:
    args = parse_args()
    if args.files_per_chunk <= 0:
        raise ValueError("--files-per-chunk must be positive")
    if args.max_source_files < 0:
        raise ValueError("--max-source-files must be non-negative")

    config_path = resolve_path(args.config).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MORTAL_CFG"] = str(config_path)

    import libriichi.libriichi as native
    from libriichi.dataset import GameplayLoader
    from mortal.online.pretrain_oracle_critic import (
        build_file_splits,
        file_list_fingerprint,
        oracle_pretrain_cfg,
    )

    cfg = dict(oracle_pretrain_cfg())
    full_train_files, dev_files, test_files = build_file_splits(cfg)
    source_files = list(full_train_files)
    if args.max_source_files > 0:
        source_files = source_files[: args.max_source_files]
    if not source_files:
        raise ValueError("selected Oracle critic train split is empty")

    native_path = Path(native.__file__).resolve()
    expected_chunks = (len(source_files) + args.files_per_chunk - 1) // args.files_per_chunk
    contract = {
        "format": "oracle_event_cache_v1",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "version": 4,
        "files_per_chunk": int(args.files_per_chunk),
        "full_train_files": len(full_train_files),
        "full_train_sha256": file_list_fingerprint(full_train_files),
        "source_files": len(source_files),
        "source_sha256": file_list_fingerprint(source_files),
        "dev_files": len(dev_files),
        "dev_sha256": file_list_fingerprint(dev_files),
        "test_files": len(test_files),
        "test_sha256": file_list_fingerprint(test_files),
        "expected_chunks": expected_chunks,
        "native_artifact": str(native_path),
        "native_sha256": file_sha256(native_path),
    }
    manifest_path = output_dir / "manifest.json"
    saved_quarantine = None
    quarantined_chunk_names = set()
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in contract.items():
            if saved.get(key) != expected:
                raise RuntimeError(
                    f"event-cache contract mismatch for {key}: "
                    f"saved={saved.get(key)!r} current={expected!r}"
                )
        saved_quarantine = saved.get("quarantine")
        if isinstance(saved_quarantine, dict):
            quarantined_chunk_names = {
                Path(record["cache_file"]).name
                for record in saved_quarantine.get("repairs", [])
            }
    else:
        atomic_write_json(manifest_path, {**contract, "complete": False})

    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    started_at = time.perf_counter()
    built_chunks = 0
    cache_files = []
    for chunk_index, source_chunk in chunked(source_files, args.files_per_chunk):
        destination = chunks_dir / f"chunk_{chunk_index:06d}.events.zst"
        cache_files.append(str(destination))
        if destination.exists():
            continue
        if destination.name in quarantined_chunk_names:
            raise RuntimeError(
                f"quarantined cache chunk is missing: {destination}; "
                "restore its replacement or rerun sanitize_oracle_event_cache.py"
            )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        file_count = loader.build_event_cache_file(source_chunk, str(temporary))
        if file_count != len(source_chunk):
            raise RuntimeError(
                f"cache chunk {chunk_index} wrote {file_count} files, "
                f"expected {len(source_chunk)}"
            )
        temporary.replace(destination)
        built_chunks += 1
        completed_chunks = chunk_index + 1
        if (
            args.log_every_chunks > 0
            and (completed_chunks % args.log_every_chunks == 0 or completed_chunks == expected_chunks)
        ):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            print(
                f"cache chunks={completed_chunks}/{expected_chunks} "
                f"built_this_run={built_chunks} elapsed={elapsed:.1f}s "
                f"chunks_per_s={built_chunks / elapsed:.3f}",
                flush=True,
            )

    indexes = {
        "train_file_index": output_dir / "cache_train_index.pth",
        "source_train_file_index": output_dir / "source_train_index.pth",
        "dev_file_index": output_dir / "dev_index.pth",
        "test_file_index": output_dir / "test_index.pth",
    }
    atomic_torch_save(indexes["train_file_index"], {"file_list": cache_files})
    atomic_torch_save(indexes["source_train_file_index"], {"file_list": source_files})
    atomic_torch_save(indexes["dev_file_index"], {"file_list": dev_files})
    atomic_torch_save(indexes["test_file_index"], {"file_list": test_files})
    total_bytes = sum(Path(cache_file).stat().st_size for cache_file in cache_files)
    elapsed = time.perf_counter() - started_at
    result = {
        **contract,
        "complete": True,
        "cache_bytes": total_bytes,
        "elapsed_seconds_this_run": elapsed,
        "built_chunks_this_run": built_chunks,
        "indexes": {key: str(value) for key, value in indexes.items()},
    }
    if saved_quarantine is not None:
        result["quarantine"] = saved_quarantine
        result["effective_source_files"] = int(
            saved_quarantine.get("effective_source_files", len(source_files))
        )
    atomic_write_json(manifest_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

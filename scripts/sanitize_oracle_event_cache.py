from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_NAME_RE = re.compile(r"^chunk_(\d{6})\.events\.zst$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay every Oracle event-cache entry, identify malformed source logs, "
            "and optionally quarantine them without changing the train-index paths."
        )
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-chunks", type=int, default=128)
    parser.add_argument("--fold-count", type=int, default=65536)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=20260416)
    parser.add_argument("--save-every-batches", type=int, default=10)
    parser.add_argument("--repair-in-place", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def atomic_write_json(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def file_sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_list_sha256(file_list: list[str]) -> str:
    digest = hashlib.sha256()
    for filename in file_list:
        digest.update(str(filename).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_file_index(filename: Path) -> list[str]:
    payload = torch.load(filename, weights_only=False, map_location="cpu")
    file_list = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(file_list, list) or not all(isinstance(item, str) for item in file_list):
        raise ValueError(f"invalid file index: {filename}")
    return file_list


def exception_text(exc: BaseException) -> str:
    text = str(exc)
    return text if len(text) <= 8000 else text[:8000] + "\n[truncated]"


def load_and_count(loader: Any, filenames: list[str]) -> int:
    loaded = loader.load_log_files(filenames)
    count = len(loaded)
    del loaded
    return count


def source_range(chunk_position: int, files_per_chunk: int, source_count: int) -> range:
    start = chunk_position * files_per_chunk
    return range(start, min(start + files_per_chunk, source_count))


def validate_chunk_mapping(cache_files: list[str], files_per_chunk: int, source_count: int) -> None:
    expected_chunks = (source_count + files_per_chunk - 1) // files_per_chunk
    if len(cache_files) != expected_chunks:
        raise ValueError(
            f"cache index has {len(cache_files)} chunks, expected {expected_chunks}"
        )
    for position, filename in enumerate(cache_files):
        match = CHUNK_NAME_RE.match(Path(filename).name)
        if match is None or int(match.group(1)) != position:
            raise ValueError(
                f"cache index position {position} does not match chunk path {filename!r}"
            )


def audit_contract(
    cache_dir: Path,
    manifest: dict[str, Any],
    cache_files: list[str],
    source_files: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": "oracle_event_cache_validity_audit_v1",
        "cache_dir": str(cache_dir),
        "cache_format": manifest.get("format"),
        "cache_version": manifest.get("version"),
        "files_per_chunk": manifest.get("files_per_chunk"),
        "declared_source_sha256": manifest.get("source_sha256"),
        "cache_index_sha256": file_list_sha256(cache_files),
        "source_index_sha256": file_list_sha256(source_files),
        "cache_chunks": len(cache_files),
        "source_files": len(source_files),
        "fold": [int(args.fold_count), int(args.fold_index), int(args.fold_seed)],
    }


def initial_audit(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        **contract,
        "complete": False,
        "next_chunk": 0,
        "validated_chunks": 0,
        "validated_source_files": 0,
        "invalid_chunks": [],
    }


def load_or_initialize_audit(output: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not output.exists():
        return initial_audit(contract)
    saved = json.loads(output.read_text(encoding="utf-8"))
    for key, value in contract.items():
        if saved.get(key) != value:
            raise RuntimeError(
                f"audit resume contract mismatch for {key}: "
                f"saved={saved.get(key)!r} current={value!r}"
            )
    return saved


def identify_invalid_chunk(
    loader: Any,
    cache_file: str,
    source_files: list[str],
    files_per_chunk: int,
    chunk_position: int,
    cache_error: BaseException,
) -> dict[str, Any]:
    invalid_sources = []
    valid_sources = []
    positions = source_range(chunk_position, files_per_chunk, len(source_files))
    for source_position in positions:
        source_file = source_files[source_position]
        try:
            loaded_count = load_and_count(loader, [source_file])
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            invalid_sources.append(
                {
                    "source_position": source_position,
                    "path": source_file,
                    "error": exception_text(exc),
                }
            )
        else:
            if loaded_count != 1:
                raise RuntimeError(
                    f"source {source_file} yielded {loaded_count} games, expected 1"
                )
            valid_sources.append(source_file)
    return {
        "chunk_position": chunk_position,
        "cache_file": cache_file,
        "cache_sha256": file_sha256(Path(cache_file)),
        "cache_error": exception_text(cache_error),
        "source_files": len(valid_sources) + len(invalid_sources),
        "valid_source_files": valid_sources,
        "invalid_sources": invalid_sources,
    }


def scan_cache(
    loader: Any,
    cache_files: list[str],
    source_files: list[str],
    files_per_chunk: int,
    audit: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    if audit.get("complete"):
        return
    started_at = time.perf_counter()
    start_chunk = int(audit.get("next_chunk", 0))
    batches = 0
    for batch_start in range(start_chunk, len(cache_files), args.batch_chunks):
        batch_end = min(batch_start + args.batch_chunks, len(cache_files))
        batch_files = cache_files[batch_start:batch_end]
        expected_games = sum(
            len(source_range(position, files_per_chunk, len(source_files)))
            for position in range(batch_start, batch_end)
        )
        try:
            loaded_count = load_and_count(loader, batch_files)
        except BaseException as batch_exc:
            if isinstance(batch_exc, (KeyboardInterrupt, SystemExit)):
                raise
            for position, cache_file in enumerate(batch_files, start=batch_start):
                expected_chunk_games = len(
                    source_range(position, files_per_chunk, len(source_files))
                )
                try:
                    chunk_count = load_and_count(loader, [cache_file])
                except BaseException as chunk_exc:
                    if isinstance(chunk_exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    record = identify_invalid_chunk(
                        loader,
                        cache_file,
                        source_files,
                        files_per_chunk,
                        position,
                        chunk_exc,
                    )
                    audit["invalid_chunks"].append(record)
                    print(
                        f"invalid chunk={position} "
                        f"invalid_sources={len(record['invalid_sources'])} "
                        f"path={cache_file}",
                        flush=True,
                    )
                else:
                    if chunk_count != expected_chunk_games:
                        raise RuntimeError(
                            f"chunk {position} yielded {chunk_count} games, "
                            f"expected {expected_chunk_games}"
                        )
        else:
            if loaded_count != expected_games:
                raise RuntimeError(
                    f"chunks [{batch_start}, {batch_end}) yielded {loaded_count} games, "
                    f"expected {expected_games}"
                )

        audit["next_chunk"] = batch_end
        audit["validated_chunks"] = batch_end
        audit["validated_source_files"] = min(
            batch_end * files_per_chunk,
            len(source_files),
        )
        batches += 1
        if batches % args.save_every_batches == 0 or batch_end == len(cache_files):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            audit["elapsed_seconds_this_run"] = elapsed
            atomic_write_json(output, audit)
            print(
                f"audit chunks={batch_end}/{len(cache_files)} "
                f"invalid={len(audit['invalid_chunks'])} "
                f"chunks_per_s={(batch_end - start_chunk) / elapsed:.2f}",
                flush=True,
            )
        if batches % 25 == 0:
            gc.collect()

    audit["complete"] = True
    audit["invalid_source_files"] = sum(
        len(record["invalid_sources"]) for record in audit["invalid_chunks"]
    )
    audit["effective_source_files"] = (
        len(source_files) - int(audit["invalid_source_files"])
    )
    atomic_write_json(output, audit)


def repair_cache(
    loader: Any,
    cache_dir: Path,
    manifest_path: Path,
    source_files: list[str],
    audit: dict[str, Any],
    output: Path,
) -> None:
    if not audit.get("complete"):
        raise RuntimeError("refusing to repair before the full audit is complete")
    backup_dir = cache_dir / "quarantine" / "original_chunks"
    backup_dir.mkdir(parents=True, exist_ok=True)
    builder = type(loader)(version=4, oracle=False, augmented=False)
    repairs = []
    invalid_paths = set()

    for record in audit["invalid_chunks"]:
        cache_file = Path(record["cache_file"])
        for item in record["invalid_sources"]:
            invalid_paths.add(item["path"])
            item.setdefault("source_sha256", file_sha256(Path(item["path"])))
        valid_sources = list(record["valid_source_files"])
        original_sha256 = record["cache_sha256"]
        backup_file = backup_dir / cache_file.name
        if backup_file.exists():
            if file_sha256(backup_file) != original_sha256:
                raise RuntimeError(f"quarantine backup hash mismatch: {backup_file}")
        else:
            if file_sha256(cache_file) != original_sha256:
                raise RuntimeError(f"cache changed since audit: {cache_file}")
            shutil.copy2(cache_file, backup_file)

        temporary = cache_file.with_name(cache_file.name + ".repair.events.zst")
        if temporary.exists():
            temporary.unlink()
        written = builder.build_event_cache_file(valid_sources, str(temporary))
        if written != len(valid_sources):
            raise RuntimeError(
                f"repair wrote {written} entries, expected {len(valid_sources)}"
            )
        replacement_count = load_and_count(loader, [str(temporary)])
        if replacement_count != len(valid_sources):
            raise RuntimeError(
                f"repair validates as {replacement_count} games, expected {len(valid_sources)}"
            )
        replacement_sha256 = file_sha256(temporary)
        os.replace(temporary, cache_file)
        if file_sha256(cache_file) != replacement_sha256:
            raise RuntimeError(f"replacement hash mismatch after install: {cache_file}")
        repairs.append(
            {
                "chunk_position": record["chunk_position"],
                "cache_file": str(cache_file),
                "backup_file": str(backup_file),
                "original_sha256": original_sha256,
                "replacement_sha256": replacement_sha256,
                "retained_source_files": len(valid_sources),
            }
        )

    effective_sources = [source for source in source_files if source not in invalid_paths]
    audit["repair_complete"] = True
    audit["repairs"] = repairs
    atomic_write_json(output, audit)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quarantine = {
        "format": "oracle_event_cache_quarantine_v1",
        "audit_file": str(output.resolve()),
        "audit_sha256": file_sha256(output),
        "invalid_source_files": len(invalid_paths),
        "invalid_sources": [
            {
                "path": item["path"],
                "source_position": item["source_position"],
                "source_sha256": item["source_sha256"],
            }
            for record in audit["invalid_chunks"]
            for item in record["invalid_sources"]
        ],
        "effective_source_files": len(effective_sources),
        "effective_source_sha256": file_list_sha256(effective_sources),
        "repairs": repairs,
    }
    manifest["quarantine"] = quarantine
    manifest["effective_source_files"] = len(effective_sources)
    manifest["cache_bytes"] = sum(
        Path(filename).stat().st_size
        for filename in load_file_index(cache_dir / "cache_train_index.pth")
    )
    atomic_write_json(manifest_path, manifest)


def main() -> None:
    args = parse_args()
    if args.batch_chunks <= 0 or args.fold_count <= 0:
        raise ValueError("--batch-chunks and --fold-count must be positive")
    if not 0 <= args.fold_index < args.fold_count:
        raise ValueError("--fold-index must be in [0, fold-count)")
    if args.save_every_batches <= 0:
        raise ValueError("--save-every-batches must be positive")

    cache_dir = resolve_path(args.cache_dir).resolve()
    output = resolve_path(args.output).resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_per_chunk = int(manifest.get("files_per_chunk", 0) or 0)
    if files_per_chunk <= 0:
        raise ValueError(f"invalid files_per_chunk in {manifest_path}")
    cache_files = load_file_index(cache_dir / "cache_train_index.pth")
    source_files = load_file_index(cache_dir / "source_train_index.pth")
    validate_chunk_mapping(cache_files, files_per_chunk, len(source_files))

    contract = audit_contract(
        cache_dir,
        manifest,
        cache_files,
        source_files,
        args,
    )
    audit = load_or_initialize_audit(output, contract)

    from libriichi.dataset import GameplayLoader

    loader = GameplayLoader(version=4, oracle=True, augmented=False)
    loader.set_sample_fold(args.fold_count, args.fold_index, args.fold_seed)
    scan_cache(
        loader,
        cache_files,
        source_files,
        files_per_chunk,
        audit,
        output,
        args,
    )
    if args.repair_in_place:
        repair_cache(loader, cache_dir, manifest_path, source_files, audit, output)
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

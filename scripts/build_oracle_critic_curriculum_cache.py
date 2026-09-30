from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mortal.research.oracle_critic_curriculum import (
    CURRICULUM_PHASES,
    build_phase_file_lists,
    classify_source_files,
    file_list_fingerprint,
    normalize_boundaries,
)


EXTERNAL_PAUSE_EXIT_CODE = 75
DEFAULT_SOURCE_CACHE_ROOT = (
    REPO_ROOT
    / "logs/oracle_event_cache/s70_temporal_dev202512_test202601_chunk16_v1"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "logs/oracle_event_cache/"
    "s70_broad_to_recent_strong24m12m_20260901_r1"
)
DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "logs/oracle_critic_search/"
    "s70_temporal_scalar_hand_wd003_formal_20260818_r1/"
    "visible_transfer_hand_aligned_wd003/config.toml"
)
DEFAULT_RUNTIME_OVERLAY = (
    REPO_ROOT / "logs/runtime_overlays/libriichi_native_fold_capacity_v2"
)
DEFAULT_BOUNDARIES = {
    "mid_start": "202112",
    "old_regression_start": "202212",
    "recent_24_start": "202312",
    "recent_12_start": "202412",
    "train_end": "202511",
}
BUCKET_ORDER = (
    "early",
    "mid",
    "old_regression",
    "recent_older12",
    "recent_12",
)


class ExternalPauseRequested(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build resumable, era-pure Oracle event-cache buckets and the "
            "SL-derived broad-to-recent phase indexes."
        )
    )
    parser.add_argument("--source-cache-root", default=str(DEFAULT_SOURCE_CACHE_ROOT))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--runtime-overlay", default=str(DEFAULT_RUNTIME_OVERLAY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--files-per-chunk", type=int, default=16)
    parser.add_argument("--target-pool-size", type=int, default=0)
    parser.add_argument("--old-regression-eval-chunks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--log-every-chunks", type=int, default=100)
    parser.add_argument("--mid-start", default=DEFAULT_BOUNDARIES["mid_start"])
    parser.add_argument(
        "--old-regression-start",
        default=DEFAULT_BOUNDARIES["old_regression_start"],
    )
    parser.add_argument(
        "--recent-24-start", default=DEFAULT_BOUNDARIES["recent_24_start"]
    )
    parser.add_argument(
        "--recent-12-start", default=DEFAULT_BOUNDARIES["recent_12_start"]
    )
    parser.add_argument("--train-end", default=DEFAULT_BOUNDARIES["train_end"])
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def file_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def atomic_torch_save(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_file_index(source: Path) -> list[str]:
    payload = torch.load(source, weights_only=False, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("file_list")
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"file index has no file_list: {source}")
    result = [str(filename) for filename in payload]
    if not result:
        raise ValueError(f"file index is empty: {source}")
    return result


def pause_file() -> Path | None:
    value = os.environ.get("MORTAL_ORACLE_PAUSE_FILE", "").strip()
    return Path(value) if value else None


def pause_requested() -> bool:
    marker = pause_file()
    return marker is not None and marker.is_file()


def require_not_paused() -> None:
    if pause_requested():
        raise ExternalPauseRequested("Apex pause requested")


def guarded_files(files: Iterable[str]) -> Iterable[str]:
    for index, filename in enumerate(files):
        if index % 8192 == 0:
            require_not_paused()
        yield filename


def invalid_source_paths(source_manifest: dict[str, Any]) -> list[str]:
    quarantine = source_manifest.get("quarantine", {})
    if not isinstance(quarantine, dict):
        return []
    return [
        str(record["path"])
        for record in quarantine.get("invalid_sources", [])
        if isinstance(record, dict) and record.get("path")
    ]


def ordered_bucket_sources(
    buckets: dict[str, list[str]], *, seed: int
) -> dict[str, list[str]]:
    ordered = {}
    for offset, bucket in enumerate(BUCKET_ORDER):
        require_not_paused()
        files = sorted(buckets[bucket])
        random.Random(int(seed) + 10_007 * (offset + 1)).shuffle(files)
        ordered[bucket] = files
    return ordered


def chunked(values: list[str], chunk_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), chunk_size):
        yield start // chunk_size, values[start : start + chunk_size]


def cache_files_for_bucket(
    output_dir: Path, bucket: str, source_files: list[str], files_per_chunk: int
) -> list[str]:
    chunk_count = (len(source_files) + files_per_chunk - 1) // files_per_chunk
    chunks_dir = output_dir / "buckets" / bucket / "chunks"
    return [
        str((chunks_dir / f"chunk_{index:06d}.events.zst").resolve())
        for index in range(chunk_count)
    ]


def runtime_artifact(runtime_overlay: Path) -> Path:
    package_dir = runtime_overlay / "libriichi"
    if not (package_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"runtime overlay is incomplete: {runtime_overlay}")
    candidates = sorted(package_dir.glob("libriichi*.pyd"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one native libriichi artifact under {package_dir}, got {candidates}"
        )
    return candidates[0].resolve()


def configure_runtime_overlay(runtime_overlay: Path, base_config: Path) -> None:
    overlay_text = str(runtime_overlay)
    if overlay_text not in sys.path:
        sys.path.insert(0, overlay_text)
    existing = [
        item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [overlay_text, *[item for item in existing if item != overlay_text]]
    )
    os.environ["MORTAL_CFG"] = str(base_config)


def write_progress(
    output_dir: Path,
    *,
    state: str,
    bucket: str = "",
    completed_chunks: int = 0,
    total_chunks: int = 0,
    detail: str = "",
) -> None:
    atomic_write_json(
        output_dir / "build_status.json",
        {
            "format": "oracle_critic_curriculum_cache_status_v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "bucket": bucket,
            "completed_chunks": int(completed_chunks),
            "total_chunks": int(total_chunks),
            "detail": detail,
        },
    )


def build_bucket_cache(
    loader: Any,
    *,
    output_dir: Path,
    bucket: str,
    source_files: list[str],
    files_per_chunk: int,
    log_every_chunks: int,
) -> tuple[list[str], int]:
    cache_files = cache_files_for_bucket(
        output_dir, bucket, source_files, files_per_chunk
    )
    chunks_dir = output_dir / "buckets" / bucket / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    built_this_run = 0
    total_chunks = len(cache_files)
    started_at = time.perf_counter()
    for chunk_index, source_chunk in chunked(source_files, files_per_chunk):
        require_not_paused()
        destination = Path(cache_files[chunk_index])
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            written = loader.build_event_cache_file(source_chunk, str(temporary))
            if int(written) != len(source_chunk):
                raise RuntimeError(
                    f"{bucket} chunk {chunk_index} wrote {written} files, "
                    f"expected {len(source_chunk)}"
                )
            temporary.replace(destination)
            built_this_run += 1
        completed = chunk_index + 1
        if (
            log_every_chunks > 0
            and (completed % log_every_chunks == 0 or completed == total_chunks)
        ):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            print(
                f"cache bucket={bucket} chunks={completed}/{total_chunks} "
                f"built_this_run={built_this_run} elapsed={elapsed:.1f}s",
                flush=True,
            )
            write_progress(
                output_dir,
                state="building",
                bucket=bucket,
                completed_chunks=completed,
                total_chunks=total_chunks,
            )
        require_not_paused()
    return cache_files, built_this_run


def main() -> int:
    args = parse_args()
    if args.files_per_chunk <= 0:
        raise ValueError("--files-per-chunk must be positive")
    if args.target_pool_size < 0:
        raise ValueError("--target-pool-size must be non-negative")
    if args.old_regression_eval_chunks <= 0:
        raise ValueError("--old-regression-eval-chunks must be positive")

    source_cache_root = resolve_path(args.source_cache_root)
    base_config = resolve_path(args.base_config)
    runtime_overlay = resolve_path(args.runtime_overlay)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    boundaries = normalize_boundaries(
        {
            "mid_start": args.mid_start,
            "old_regression_start": args.old_regression_start,
            "recent_24_start": args.recent_24_start,
            "recent_12_start": args.recent_12_start,
            "train_end": args.train_end,
        }
    )

    try:
        require_not_paused()
        source_manifest_path = source_cache_root / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        indexes = source_manifest.get("indexes", {})
        source_index = resolve_path(indexes["source_train_file_index"])
        dev_index = resolve_path(indexes["dev_file_index"])
        test_index = resolve_path(indexes["test_file_index"])
        source_files = load_file_index(source_index)
        dev_files = load_file_index(dev_index)
        test_files = load_file_index(test_index)
        invalid_sources = invalid_source_paths(source_manifest)
        invalid_source_set = set(invalid_sources)
        effective_source_files = [
            filename
            for filename in guarded_files(source_files)
            if filename not in invalid_source_set
        ]
        classified = classify_source_files(
            guarded_files(source_files),
            boundaries=boundaries,
            invalid_sources=invalid_sources,
        )
        bucket_sources = ordered_bucket_sources(classified, seed=args.seed)
        cache_buckets = {
            bucket: cache_files_for_bucket(
                output_dir,
                bucket,
                bucket_sources[bucket],
                args.files_per_chunk,
            )
            for bucket in BUCKET_ORDER
        }
        default_pool_size = sum(
            len(cache_buckets[bucket])
            for bucket in ("early", "mid", "recent_older12", "recent_12")
        )
        target_pool_size = int(args.target_pool_size or default_pool_size)
        phase_lists, phase_counts = build_phase_file_lists(
            cache_buckets,
            target_size=target_pool_size,
            seed=args.seed,
        )
        old_regression_eval = list(cache_buckets["old_regression"])
        random.Random(args.seed + 70_001).shuffle(old_regression_eval)
        old_regression_eval = old_regression_eval[
            : min(args.old_regression_eval_chunks, len(old_regression_eval))
        ]
        native_artifact = runtime_artifact(runtime_overlay)

        index_paths = {
            "phase_a": output_dir / "phase_a_train_index.pth",
            "phase_b": output_dir / "phase_b_train_index.pth",
            "phase_c": output_dir / "phase_c_train_index.pth",
            "old_regression_full": output_dir / "old_regression_full_index.pth",
            "old_regression_eval": output_dir / "old_regression_eval_index.pth",
            "dev": output_dir / "dev_index.pth",
            "test": output_dir / "test_index.pth",
        }
        contract = {
            "format": "oracle_critic_curriculum_cache_v1",
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "curriculum_module_sha256": file_sha256(
                REPO_ROOT / "mortal/research/oracle_critic_curriculum.py"
            ),
            "source_cache_manifest": str(source_manifest_path.resolve()),
            "source_cache_manifest_sha256": file_sha256(source_manifest_path),
            "source_train_index": str(source_index),
            "source_train_index_sha256": file_sha256(source_index),
            "source_files": len(source_files),
            "source_sha256": file_list_fingerprint(source_files),
            "invalid_sources": sorted(invalid_sources),
            "effective_source_files": len(effective_source_files),
            "effective_source_sha256": file_list_fingerprint(effective_source_files),
            "dev_files": len(dev_files),
            "dev_sha256": file_list_fingerprint(dev_files),
            "test_files": len(test_files),
            "test_sha256": file_list_fingerprint(test_files),
            "base_config": str(base_config),
            "base_config_sha256": file_sha256(base_config),
            "runtime_overlay": str(runtime_overlay),
            "native_artifact": str(native_artifact),
            "native_artifact_sha256": file_sha256(native_artifact),
            "version": 4,
            "files_per_chunk": int(args.files_per_chunk),
            "seed": int(args.seed),
            "boundaries": boundaries,
            "phase_profiles": {
                phase: {name: weight for name, weight in profile}
                for phase, profile in CURRICULUM_PHASES.items()
            },
            "target_pool_size": target_pool_size,
            "old_regression_eval_chunks": len(old_regression_eval),
            "bucket_sources": {
                bucket: {
                    "files": len(bucket_sources[bucket]),
                    "sha256": file_list_fingerprint(bucket_sources[bucket]),
                    "cache_chunks": len(cache_buckets[bucket]),
                }
                for bucket in BUCKET_ORDER
            },
            "phase_counts": phase_counts,
            "phase_sha256": {
                phase: file_list_fingerprint(files)
                for phase, files in phase_lists.items()
            },
            "indexes": {name: str(path.resolve()) for name, path in index_paths.items()},
        }
        if manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("contract") != contract:
                raise RuntimeError(
                    "curriculum cache contract changed; use a new output directory"
                )
            created_at_utc = saved.get("created_at_utc")
        else:
            created_at_utc = datetime.now(timezone.utc).isoformat()

        for bucket in BUCKET_ORDER:
            require_not_paused()
            bucket_dir = output_dir / "buckets" / bucket
            atomic_torch_save(
                bucket_dir / "source_index.pth",
                {"file_list": bucket_sources[bucket]},
            )
            atomic_torch_save(
                bucket_dir / "cache_index.pth",
                {"file_list": cache_buckets[bucket]},
            )
            require_not_paused()
        for phase in ("phase_a", "phase_b", "phase_c"):
            require_not_paused()
            atomic_torch_save(index_paths[phase], {"file_list": phase_lists[phase]})
        atomic_torch_save(
            index_paths["old_regression_full"],
            {"file_list": cache_buckets["old_regression"]},
        )
        atomic_torch_save(
            index_paths["old_regression_eval"],
            {"file_list": old_regression_eval},
        )
        atomic_torch_save(index_paths["dev"], {"file_list": dev_files})
        atomic_torch_save(index_paths["test"], {"file_list": test_files})
        atomic_write_json(
            manifest_path,
            {
                "contract": contract,
                "created_at_utc": created_at_utc,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete": False,
                "prepared": True,
            },
        )
        if args.prepare_only:
            write_progress(output_dir, state="prepared", detail="cache files not built")
            print(json.dumps({"contract": contract, "prepared": True}, sort_keys=True))
            return 0

        configure_runtime_overlay(runtime_overlay, base_config)
        from libriichi.dataset import GameplayLoader

        loader = GameplayLoader(version=4, oracle=False, augmented=False)
        total_built = 0
        started_at = time.perf_counter()
        for bucket in BUCKET_ORDER:
            _, built = build_bucket_cache(
                loader,
                output_dir=output_dir,
                bucket=bucket,
                source_files=bucket_sources[bucket],
                files_per_chunk=args.files_per_chunk,
                log_every_chunks=args.log_every_chunks,
            )
            total_built += built
        require_not_paused()
        all_cache_files = [
            filename
            for bucket in BUCKET_ORDER
            for filename in cache_buckets[bucket]
        ]
        missing = [filename for filename in all_cache_files if not Path(filename).is_file()]
        if missing:
            raise RuntimeError(f"curriculum cache is incomplete; missing {missing[:3]}")
        result = {
            "contract": contract,
            "created_at_utc": created_at_utc,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "prepared": True,
            "cache_bytes": sum(Path(filename).stat().st_size for filename in all_cache_files),
            "built_chunks_this_run": total_built,
            "elapsed_seconds_this_run": time.perf_counter() - started_at,
        }
        atomic_write_json(manifest_path, result)
        write_progress(output_dir, state="completed")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except ExternalPauseRequested as exc:
        write_progress(output_dir, state="paused", detail=str(exc))
        print("curriculum cache paused for Apex", flush=True)
        return EXTERNAL_PAUSE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())

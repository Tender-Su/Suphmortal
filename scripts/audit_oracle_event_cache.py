from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ORACLE_PRIVATE_CHANNELS = 51
ORACLE_WALL_SLOT_CHANNELS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that an event cache reproduces its source gameplay tensors."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--cache",
        required=True,
        help="One .events.zst file or an event-cache directory containing chunks/.",
    )
    parser.add_argument("--max-files", type=int, required=True)
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="Start at this cache chunk when --cache is a directory.",
    )
    parser.add_argument("--fold-count", type=int, default=64)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=20260416)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def take_gameplay(gameplay) -> dict[str, object]:
    fields = {
        "obs": np.asarray(gameplay.take_obs_batch()),
        "invisible_obs": np.asarray(gameplay.take_invisible_obs_batch()),
        "actions": np.asarray(gameplay.take_actions_batch()),
        "masks": np.asarray(gameplay.take_masks_batch()),
        "at_kyoku": np.asarray(gameplay.take_at_kyoku_batch()),
    }
    grp = gameplay.take_grp()
    fields["grp_feature"] = np.asarray(grp.take_feature())
    fields["rank_by_player"] = np.asarray(grp.take_rank_by_player())
    fields["player_id"] = int(gameplay.take_player_id())
    return fields


def wall_multiset(invisible_obs: np.ndarray) -> np.ndarray:
    wall = invisible_obs[:, ORACLE_PRIVATE_CHANNELS:, :]
    if wall.shape[1] % ORACLE_WALL_SLOT_CHANNELS:
        raise ValueError(f"unexpected Oracle wall channel count: {wall.shape[1]}")
    slots = wall.reshape(
        wall.shape[0],
        wall.shape[1] // ORACLE_WALL_SLOT_CHANNELS,
        ORACLE_WALL_SLOT_CHANNELS,
        wall.shape[2],
    )
    return slots.sum(axis=1)


def compare_gameplay_batches(source_batches, cache_batches) -> dict[str, int]:
    if len(source_batches) != len(cache_batches):
        raise AssertionError(
            f"gameplay count differs: source={len(source_batches)} cache={len(cache_batches)}"
        )

    exact_fields = (
        "obs",
        "actions",
        "masks",
        "at_kyoku",
        "grp_feature",
        "rank_by_player",
    )
    full_invisible_differences = 0
    gameplays = 0
    states = 0
    for game_index, (source_gameplay_batch, cache_gameplay_batch) in enumerate(
        zip(source_batches, cache_batches, strict=True)
    ):
        if len(source_gameplay_batch) != len(cache_gameplay_batch):
            raise AssertionError(
                f"player-view count differs at game {game_index}: "
                f"source={len(source_gameplay_batch)} cache={len(cache_gameplay_batch)}"
            )
        for player_index, (source_gameplay, cache_gameplay) in enumerate(
            zip(source_gameplay_batch, cache_gameplay_batch, strict=True)
        ):
            location = f"game {game_index} player-view {player_index}"
            source = take_gameplay(source_gameplay)
            cached = take_gameplay(cache_gameplay)
            for field in exact_fields:
                if not np.array_equal(source[field], cached[field]):
                    raise AssertionError(f"{field} differs at {location}")
            if source["player_id"] != cached["player_id"]:
                raise AssertionError(f"player_id differs at {location}")

            source_invisible = source["invisible_obs"]
            cached_invisible = cached["invisible_obs"]
            if source_invisible.shape != cached_invisible.shape:
                raise AssertionError(f"invisible_obs shape differs at {location}")
            if not np.array_equal(
                source_invisible[:, :ORACLE_PRIVATE_CHANNELS, :],
                cached_invisible[:, :ORACLE_PRIVATE_CHANNELS, :],
            ):
                raise AssertionError(f"opponent private features differ at {location}")
            if not np.array_equal(
                wall_multiset(source_invisible),
                wall_multiset(cached_invisible),
            ):
                raise AssertionError(f"Oracle wall tile multiset differs at {location}")
            full_invisible_differences += int(
                np.count_nonzero(source_invisible != cached_invisible)
            )
            gameplays += 1
            states += int(source_invisible.shape[0])
    return {
        "games": len(source_batches),
        "gameplays": gameplays,
        "states": states,
        "full_invisible_differences": full_invisible_differences,
    }


def main() -> None:
    args = parse_args()
    if args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.start_chunk < 0:
        raise ValueError("--start-chunk must be non-negative")
    if args.fold_count <= 0 or not 0 <= args.fold_index < args.fold_count:
        raise ValueError("fold index must be in [0, fold-count)")

    config_path = resolve_path(args.config).resolve()
    cache_path = resolve_path(args.cache).resolve()
    os.environ["MORTAL_CFG"] = str(config_path)

    from libriichi.dataset import GameplayLoader
    from mortal.online.pretrain_oracle_critic import (
        build_file_splits,
        oracle_pretrain_cfg,
    )

    cfg = dict(oracle_pretrain_cfg())
    train_files, _dev_files, _test_files = build_file_splits(cfg)
    loader_kwargs = {
        "version": 4,
        "oracle": True,
        "augmented": False,
        "track_opponent_states": False,
        "track_danger_labels": False,
        "track_regret_labels": False,
    }
    source_loader = GameplayLoader(**loader_kwargs)
    cache_loader = GameplayLoader(**loader_kwargs)
    source_loader.set_sample_fold(args.fold_count, args.fold_index, args.fold_seed)
    cache_loader.set_sample_fold(args.fold_count, args.fold_index, args.fold_seed)
    if cache_path.is_dir():
        chunks_dir = cache_path / "chunks"
        manifest_path = cache_path / "manifest.json"
        if not chunks_dir.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"event-cache directory is missing chunks/ or manifest.json: {cache_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_per_chunk = int(manifest.get("files_per_chunk", 0) or 0)
        if files_per_chunk <= 0:
            raise ValueError(f"invalid files_per_chunk in {manifest_path}")
        source_start = int(args.start_chunk) * files_per_chunk
        if source_start >= len(train_files):
            raise ValueError(
                f"start chunk {args.start_chunk} begins past {len(train_files)} source files"
            )
        source_file_count = min(int(args.max_files), len(train_files) - source_start)
        required_chunks = (source_file_count + files_per_chunk - 1) // files_per_chunk
        source_file_count = min(
            required_chunks * files_per_chunk,
            len(train_files) - source_start,
        )
        source_files = train_files[source_start:source_start + source_file_count]
        cache_files = [
            str(source)
            for source in sorted(chunks_dir.glob("*.events.zst"))[
                args.start_chunk:args.start_chunk + required_chunks
            ]
        ]
        if len(cache_files) != required_chunks:
            raise FileNotFoundError(
                f"cache has {len(cache_files)} required chunks, expected {required_chunks}"
            )
    else:
        if args.start_chunk:
            raise ValueError("--start-chunk requires a cache directory")
        source_files = train_files[: args.max_files]
        files_per_chunk = len(source_files)
        cache_files = [str(cache_path)]

    totals = {
        "games": 0,
        "gameplays": 0,
        "states": 0,
        "full_invisible_differences": 0,
    }
    for cache_index, cache_file in enumerate(cache_files):
        start = cache_index * files_per_chunk
        source_chunk = source_files[start:start + files_per_chunk]
        source_batches = source_loader.load_log_files(source_chunk)
        cache_batches = cache_loader.load_log_files([cache_file])
        chunk_totals = compare_gameplay_batches(source_batches, cache_batches)
        for key, value in chunk_totals.items():
            totals[key] += value
        del source_batches, cache_batches
        gc.collect()

    print(json.dumps({
        "cache": str(cache_path),
        "cache_files": len(cache_files),
        "start_chunk": int(args.start_chunk),
        "source_files": len(source_files),
        "games": totals["games"],
        "gameplays": totals["gameplays"],
        "states": totals["states"],
        "fold": [args.fold_count, args.fold_index, args.fold_seed],
        "exact_fields": [
            "obs",
            "actions",
            "masks",
            "at_kyoku",
            "grp_feature",
            "rank_by_player",
        ],
        "oracle_private_channels": ORACLE_PRIVATE_CHANNELS,
        "wall_multiset_exact": True,
        "full_invisible_differences": totals["full_invisible_differences"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

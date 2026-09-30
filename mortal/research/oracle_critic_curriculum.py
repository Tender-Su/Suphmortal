from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


MONTH_PATTERN = re.compile(r"^\d{6}$")
CURRICULUM_PHASES = {
    "phase_a": (("recent", 0.60), ("mid", 0.25), ("early", 0.15)),
    "phase_b": (("recent", 0.90), ("replay", 0.10)),
    "phase_c": (("recent", 0.98), ("replay", 0.02)),
}
PRESERVED_CHECKPOINT_FIELDS = (
    "oracle_brain",
    "value_net",
    "optimizer",
    "scheduler",
    "scaler",
    "steps",
    "training_contract",
    "convergence_state",
)
PRESERVED_TRAINING_STATE_FIELDS = (
    "oracle_brain",
    "value_net",
    "optimizer",
    "scheduler",
    "scaler",
    "steps",
)


def validate_month(value: str, *, name: str) -> str:
    value = str(value).strip()
    if not MONTH_PATTERN.fullmatch(value) or not 1 <= int(value[4:]) <= 12:
        raise ValueError(f"{name} must be a valid YYYYMM month, got {value!r}")
    return value


def source_month(filename: str | Path) -> str:
    source = Path(filename)
    month = source.parent.name
    year = source.parent.parent.name
    validate_month(month, name="source month")
    if year != month[:4] or not source.name.startswith(month):
        raise ValueError(f"source path does not follow year/month naming: {source}")
    return month


def normalize_boundaries(boundaries: Mapping[str, str]) -> dict[str, str]:
    required = (
        "mid_start",
        "old_regression_start",
        "recent_24_start",
        "recent_12_start",
        "train_end",
    )
    missing = [key for key in required if key not in boundaries]
    if missing:
        raise ValueError(f"curriculum boundaries are missing {missing}")
    normalized = {
        key: validate_month(str(boundaries[key]), name=key) for key in required
    }
    ordered = [normalized[key] for key in required]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError(f"curriculum boundaries must be strictly increasing: {normalized}")
    return normalized


def classify_source_files(
    source_files: Iterable[str],
    *,
    boundaries: Mapping[str, str],
    invalid_sources: Iterable[str] = (),
) -> dict[str, list[str]]:
    bounds = normalize_boundaries(boundaries)
    invalid = {str(Path(filename)) for filename in invalid_sources}
    buckets = {
        "early": [],
        "mid": [],
        "old_regression": [],
        "recent_older12": [],
        "recent_12": [],
    }
    seen = set()
    for raw_filename in source_files:
        filename = str(Path(raw_filename))
        if filename in invalid:
            continue
        if filename in seen:
            raise ValueError(f"duplicate Oracle source file: {filename}")
        seen.add(filename)
        month = source_month(filename)
        if month > bounds["train_end"]:
            raise ValueError(
                f"source month {month} is newer than train_end {bounds['train_end']}"
            )
        if month < bounds["mid_start"]:
            bucket = "early"
        elif month < bounds["old_regression_start"]:
            bucket = "mid"
        elif month < bounds["recent_24_start"]:
            bucket = "old_regression"
        elif month < bounds["recent_12_start"]:
            bucket = "recent_older12"
        else:
            bucket = "recent_12"
        buckets[bucket].append(filename)

    empty = [name for name, files in buckets.items() if not files]
    if empty:
        raise ValueError(f"curriculum source buckets are empty: {empty}")
    return buckets


def file_list_fingerprint(file_list: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for filename in file_list:
        digest.update(str(filename).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def allocate_weighted_counts(
    weights: Mapping[str, float], target_size: int
) -> dict[str, int]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if not weights:
        raise ValueError("weights must not be empty")
    normalized = {name: float(weight) for name, weight in weights.items()}
    if any(weight <= 0 for weight in normalized.values()):
        raise ValueError(f"weights must be positive: {normalized}")
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to one, got {total}")

    raw = {name: target_size * weight for name, weight in normalized.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = target_size - sum(counts.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def weighted_file_pool(
    buckets: Mapping[str, list[str]],
    weights: Mapping[str, float],
    *,
    target_size: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    counts = allocate_weighted_counts(weights, target_size)
    selected = []
    for offset, name in enumerate(sorted(counts)):
        files = list(buckets.get(name, ()))
        if not files:
            raise ValueError(f"weighted bucket {name!r} is empty")
        random.Random(int(seed) + 1009 * (offset + 1)).shuffle(files)
        selected.extend(files[index % len(files)] for index in range(counts[name]))
    random.Random(int(seed)).shuffle(selected)
    return selected, counts


def build_phase_file_lists(
    cache_buckets: Mapping[str, list[str]],
    *,
    target_size: int,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    early = list(cache_buckets["early"])
    mid = list(cache_buckets["mid"])
    recent_older12 = list(cache_buckets["recent_older12"])
    recent_12 = list(cache_buckets["recent_12"])
    phase_sources = {
        "phase_a": {
            "recent": recent_older12 + recent_12,
            "mid": mid,
            "early": early,
        },
        "phase_b": {
            "recent": recent_older12 + recent_12,
            "replay": early + mid,
        },
        "phase_c": {
            "recent": recent_12,
            "replay": early + mid,
        },
    }
    phase_lists = {}
    phase_counts = {}
    for phase_offset, (phase, profile) in enumerate(CURRICULUM_PHASES.items()):
        weights = dict(profile)
        phase_lists[phase], phase_counts[phase] = weighted_file_pool(
            phase_sources[phase],
            weights,
            target_size=target_size,
            seed=int(seed) + 100_003 * (phase_offset + 1),
        )
    return phase_lists, phase_counts


def split_summary(
    train_files: list[str],
    dev_files: list[str],
    test_files: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "train": {
            "files": len(train_files),
            "sha256": file_list_fingerprint(train_files),
        },
        "dev": {
            "files": len(dev_files),
            "sha256": file_list_fingerprint(dev_files),
        },
        "test": {
            "files": len(test_files),
            "sha256": file_list_fingerprint(test_files),
        },
    }


def _hash_update(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_update(digest, item)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(repr(value).encode("utf-8"))


def stable_object_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def checkpoint_component_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    missing = [key for key in PRESERVED_CHECKPOINT_FIELDS if key not in state]
    if missing:
        raise ValueError(f"source checkpoint is missing required state: {missing}")
    return {
        key: stable_object_sha256(state[key]) for key in PRESERVED_CHECKPOINT_FIELDS
    }


def checkpoint_training_state_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    missing = [key for key in PRESERVED_TRAINING_STATE_FIELDS if key not in state]
    if missing:
        raise ValueError(f"source checkpoint is missing training state: {missing}")
    return {
        key: stable_object_sha256(state[key])
        for key in PRESERVED_TRAINING_STATE_FIELDS
    }


def migrate_checkpoint_for_phase(
    state: dict[str, Any],
    *,
    destination_config: dict[str, Any],
    destination_file_splits: dict[str, Any],
    source_phase: str,
    destination_phase: str,
    source_checkpoint: str,
    source_checkpoint_sha256: str,
    reset_data_cursor: bool = True,
    destination_training_contract: Mapping[str, Any] | None = None,
    destination_adaptive_curriculum_state: Mapping[str, Any] | None = None,
    destination_adaptive_state_action: str = "reset_for_destination_phase",
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if not bool(state.get("resume_supported", False)):
        raise ValueError("curriculum phase migration requires a resumable checkpoint")
    checkpoint_component_hashes(state)
    pretrain = copy.deepcopy(destination_config.get("oracle_critic_pretrain", {}))
    if not pretrain:
        raise ValueError("destination config has no oracle_critic_pretrain section")
    saved_progress = copy.deepcopy(state.get("data_progress"))
    if not isinstance(saved_progress, dict) or "signature" not in saved_progress:
        raise ValueError("source checkpoint has no resumable data_progress signature")

    source_splits = copy.deepcopy(state.get("file_splits"))
    if not isinstance(source_splits, dict):
        raise ValueError("source checkpoint has no file split provenance")
    if source_splits.get("dev") != destination_file_splits.get("dev"):
        raise ValueError("curriculum phase migration must preserve the dev split")
    if source_splits.get("test") != destination_file_splits.get("test"):
        raise ValueError("curriculum phase migration must preserve the test split")
    train_split_changed = source_splits.get("train") != destination_file_splits.get(
        "train"
    )
    if train_split_changed and not reset_data_cursor:
        raise ValueError("a changed train split requires resetting the data cursor")

    migrated = dict(state)
    migrated["config"] = copy.deepcopy(destination_config)
    migrated["oracle_critic_pretrain"] = pretrain
    migrated["file_splits"] = copy.deepcopy(destination_file_splits)
    if destination_training_contract is not None:
        migrated["source_training_contract"] = copy.deepcopy(
            state.get("training_contract")
        )
        migrated["training_contract"] = copy.deepcopy(
            destination_training_contract
        )
    if destination_adaptive_curriculum_state is not None:
        migrated["adaptive_curriculum_state"] = copy.deepcopy(
            destination_adaptive_curriculum_state
        )
    if reset_data_cursor:
        migrated["data_progress"] = {
            "signature": copy.deepcopy(saved_progress["signature"]),
            "cycle": 0,
            "resume_cursors": {},
            "samples_consumed": int(saved_progress.get("samples_consumed", 0)),
            "batches_consumed": int(saved_progress.get("batches_consumed", 0)),
        }
        data_cursor_action = "reset_for_declared_curriculum_phase"
    else:
        migrated["data_progress"] = saved_progress
        data_cursor_action = "preserve_for_same_train_split"
    record = {
        "format": "oracle_critic_curriculum_phase_transition_v1",
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "source_phase": str(source_phase),
        "destination_phase": str(destination_phase),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "steps": int(state["steps"]),
        "source_file_splits": source_splits,
        "destination_file_splits": copy.deepcopy(destination_file_splits),
        "data_cursor_action": data_cursor_action,
        "cumulative_samples_consumed": int(saved_progress.get("samples_consumed", 0)),
        "cumulative_batches_consumed": int(saved_progress.get("batches_consumed", 0)),
        "preserved_state": list(
            PRESERVED_TRAINING_STATE_FIELDS
            if destination_training_contract is not None
            or destination_adaptive_curriculum_state is not None
            else PRESERVED_CHECKPOINT_FIELDS
        ),
        "training_contract_action": (
            "replace_for_audited_adaptive_phase"
            if destination_training_contract is not None
            else "preserve"
        ),
        "adaptive_state_action": (
            str(destination_adaptive_state_action)
            if destination_adaptive_curriculum_state is not None
            else "preserve"
        ),
    }
    provenance = list(migrated.get("curriculum_provenance", ()))
    provenance.append(record)
    migrated["curriculum_provenance"] = provenance
    init_info = copy.deepcopy(migrated.get("init_info", {}))
    init_provenance = list(init_info.get("curriculum_provenance", ()))
    init_provenance.append(copy.deepcopy(record))
    init_info["curriculum_provenance"] = init_provenance
    migrated["init_info"] = init_info

    hash_function = (
        checkpoint_training_state_hashes
        if destination_training_contract is not None
        or destination_adaptive_curriculum_state is not None
        else checkpoint_component_hashes
    )
    before = hash_function(state)
    after = hash_function(migrated)
    if before != after:
        raise RuntimeError("curriculum phase migration changed protected checkpoint state")
    return migrated

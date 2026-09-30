from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
import tomllib
from pathlib import Path

import torch


PLATEAU_KEYS = (
    "init",
    "peak",
    "warm_up_steps",
    "factor",
    "patience_steps",
    "threshold",
    "min_lr",
    "metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an auditable plateau-scheduler branch from an AdamW "
            "checkpoint whose pre-branch LR trajectory is identical."
        )
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-metrics", required=True)
    parser.add_argument("--destination-config", required=True)
    parser.add_argument("--destination-dir", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_primary_history(metrics_path: Path, max_step: int) -> list[dict[str, object]]:
    by_step: dict[int, dict[str, object]] = {}
    with metrics_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["steps"])
            if step > max_step:
                continue
            metric = float(row["val"]["outputs"]["relative_player_0"]["loss"])
            if not math.isfinite(metric):
                raise ValueError(
                    f"non-finite primary metric at {metrics_path}:{line_number}"
                )
            previous = by_step.get(step)
            if previous is not None and not math.isclose(
                float(previous["metric"]), metric, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"conflicting monitor metrics at step {step}")
            by_step[step] = {
                "step": step,
                "metric": metric,
                "source_line": line_number,
                "row": row,
            }
    history = [by_step[step] for step in sorted(by_step)]
    if not history or int(history[-1]["step"]) != max_step:
        raise ValueError(f"monitor history does not end at exact step {max_step}")
    return history


def replay_plateau_history(
    history: list[dict[str, object]],
    *,
    peak: float,
    factor: float,
    patience_steps: int,
    threshold: float,
    min_lr: float,
) -> dict[str, object]:
    best = math.inf
    last_improvement_step = 0
    plateau_lr = float(peak)
    num_reductions = 0
    observations = []
    for item in history:
        step = int(item["step"])
        metric = float(item["metric"])
        if metric < best - threshold:
            best = metric
            last_improvement_step = step
            action = "improved"
        elif step - last_improvement_step < patience_steps:
            action = "hold"
        else:
            reduced = max(float(min_lr), plateau_lr * float(factor))
            if math.isclose(reduced, plateau_lr, rel_tol=0.0, abs_tol=1e-15):
                action = "at_min_lr"
            else:
                plateau_lr = reduced
                last_improvement_step = step
                num_reductions += 1
                action = "reduce_lr"
        observations.append(
            {
                "step": step,
                "metric": metric,
                "action": action,
                "lr_after_observation": plateau_lr,
            }
        )
    return {
        "best": best,
        "last_improvement_step": last_improvement_step,
        "plateau_lr": plateau_lr,
        "num_reductions": num_reductions,
        "observations": observations,
    }


def plateau_contract(scheduler_cfg: dict[str, object]) -> dict[str, object]:
    if str(scheduler_cfg.get("type", "")).strip().lower() != "plateau":
        raise ValueError("destination config must use scheduler.type=plateau")
    missing = [key for key in PLATEAU_KEYS if key not in scheduler_cfg]
    if missing:
        raise ValueError(f"destination plateau scheduler is missing {missing}")
    return {"type": "plateau", **{key: scheduler_cfg[key] for key in PLATEAU_KEYS}}


def branch_checkpoint(
    *,
    source_checkpoint: Path,
    source_metrics: Path,
    destination_config: Path,
    destination_dir: Path,
    expected_step: int,
    overwrite: bool,
) -> dict[str, object]:
    destination_checkpoints = destination_dir / "checkpoints"
    anchor_path = destination_checkpoints / f"step_{expected_step:06d}.pth"
    latest_path = destination_checkpoints / "latest.pth"
    metrics_path = destination_dir / "metrics.jsonl"
    manifest_path = destination_dir / "plateau_branch_manifest.json"
    outputs = (anchor_path, latest_path, metrics_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"destination artifacts already exist: {existing}")

    with destination_config.open("rb") as source:
        destination_cfg = tomllib.load(source)
    oracle_cfg = destination_cfg["oracle_critic_pretrain"]
    scheduler_cfg = oracle_cfg["scheduler"]
    destination_scheduler_contract = plateau_contract(scheduler_cfg)

    state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    step = int(state.get("steps", -1))
    if step != expected_step:
        raise ValueError(f"source checkpoint step {step} != expected {expected_step}")
    if not bool(state.get("resume_supported", False)):
        raise ValueError("source checkpoint does not support exact resume")
    for key in ("optimizer", "scaler", "data_progress", "scheduler"):
        if not state.get(key):
            raise ValueError(f"source checkpoint is missing complete {key} state")

    training_contract = copy.deepcopy(state["training_contract"])
    source_scheduler_contract = training_contract.get("scheduler", {})
    if source_scheduler_contract.get("type") != "wsd":
        raise ValueError("source scheduler must be the shared WSD AdamW trunk")
    optimizer_contract = training_contract.get("optimizer", {})
    if optimizer_contract.get("type") != "adamw":
        raise ValueError("plateau branch requires an AdamW source optimizer")

    warm_up_steps = int(source_scheduler_contract["warm_up_steps"])
    stable_steps = int(source_scheduler_contract["stable_steps"])
    if step > warm_up_steps + stable_steps:
        raise ValueError("source WSD scheduler has already left its constant-LR segment")

    peak = float(scheduler_cfg["peak"])
    source_scheduler = state["scheduler"]
    source_lrs = [float(value) for value in source_scheduler.get("_last_lr", [])]
    optimizer_lrs = [
        float(group["lr"]) for group in state["optimizer"].get("param_groups", [])
    ]
    if not source_lrs or not optimizer_lrs:
        raise ValueError("source checkpoint does not record optimizer LRs")
    for label, values in (("scheduler", source_lrs), ("optimizer", optimizer_lrs)):
        if any(not math.isclose(value, peak, rel_tol=0.0, abs_tol=1e-12) for value in values):
            raise ValueError(f"source {label} LRs are not identical to plateau peak {peak}")

    history = load_primary_history(source_metrics, expected_step)
    replay = replay_plateau_history(
        history,
        peak=peak,
        factor=float(scheduler_cfg["factor"]),
        patience_steps=int(scheduler_cfg["patience_steps"]),
        threshold=float(scheduler_cfg["threshold"]),
        min_lr=float(scheduler_cfg["min_lr"]),
    )
    if int(replay["num_reductions"]) != 0:
        raise ValueError(
            "replayed plateau history would already have reduced LR; "
            "the WSD trunk is not an equivalent branch point"
        )
    saved_best = float(state.get("best_primary_loss", math.inf))
    if not math.isclose(saved_best, float(replay["best"]), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"checkpoint best_primary_loss {saved_best} != replayed {replay['best']}"
        )

    base_lrs = [float(value) for value in source_scheduler["base_lrs"]]
    plateau_lr = float(replay["plateau_lr"])
    state["scheduler"] = {
        "base_lrs": base_lrs,
        "peak": peak,
        "init": float(scheduler_cfg["init"]),
        "warm_up_steps": int(scheduler_cfg["warm_up_steps"]),
        "factor": float(scheduler_cfg["factor"]),
        "patience_steps": int(scheduler_cfg["patience_steps"]),
        "threshold": float(scheduler_cfg["threshold"]),
        "min_lr": float(scheduler_cfg["min_lr"]),
        "plateau_lr": plateau_lr,
        "best": float(replay["best"]),
        "last_improvement_step": int(replay["last_improvement_step"]),
        "num_reductions": 0,
        "last_epoch": expected_step,
        "_last_lr": [base_lr * plateau_lr for base_lr in base_lrs],
    }
    training_contract["scheduler"] = destination_scheduler_contract
    state["training_contract"] = training_contract
    state["config"] = copy.deepcopy(destination_cfg)
    state["oracle_critic_pretrain"] = copy.deepcopy(oracle_cfg)
    state["timestamp"] = time.time()

    source_checkpoint_sha256 = file_sha256(source_checkpoint)
    source_metrics_sha256 = file_sha256(source_metrics)
    provenance = {
        "format": "oracle_critic_scheduler_branch_v1",
        "created_at": state["timestamp"],
        "source_checkpoint": {
            "path": str(source_checkpoint.resolve()),
            "sha256": source_checkpoint_sha256,
            "steps": expected_step,
        },
        "source_metrics": {
            "path": str(source_metrics.resolve()),
            "sha256": source_metrics_sha256,
        },
        "source_scheduler_contract": source_scheduler_contract,
        "destination_scheduler_contract": destination_scheduler_contract,
        "monitor_replay": replay,
        "preserved_state": [
            "oracle_brain",
            "value_net",
            "optimizer",
            "scaler",
            "data_progress",
        ],
    }
    init_info = copy.deepcopy(state.get("init_info", {}))
    init_info["scheduler_branch"] = provenance
    state["init_info"] = init_info

    destination_checkpoints.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        if output.exists():
            output.unlink()
    temporary = anchor_path.with_suffix(anchor_path.suffix + f".{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, anchor_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        os.link(anchor_path, latest_path)
    except OSError:
        shutil.copy2(anchor_path, latest_path)

    metrics_lines = [
        json.dumps(item["row"], sort_keys=True) for item in history
    ]
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    manifest = {
        **provenance,
        "destination": {
            "directory": str(destination_dir.resolve()),
            "config": str(destination_config.resolve()),
            "checkpoint": str(anchor_path.resolve()),
            "checkpoint_sha256": file_sha256(anchor_path),
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": file_sha256(metrics_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = branch_checkpoint(
        source_checkpoint=Path(args.source_checkpoint).resolve(),
        source_metrics=Path(args.source_metrics).resolve(),
        destination_config=Path(args.destination_config).resolve(),
        destination_dir=Path(args.destination_dir).resolve(),
        expected_step=args.expected_step,
        overwrite=args.overwrite,
    )
    replay = manifest["monitor_replay"]
    print(
        "created plateau branch "
        f"step={args.expected_step} best={replay['best']:.9f} "
        f"last_improvement_step={replay['last_improvement_step']} "
        f"lr={replay['plateau_lr']:.9g} reductions={replay['num_reductions']}"
    )
    print(f"wrote {manifest['destination']['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

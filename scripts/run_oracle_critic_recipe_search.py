from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_CONTRACT_KEYS = ("init", "peak", "final", "warm_up_steps", "max_steps")
SOURCE_FILES = (
    "mortal/core/lr_scheduler.py",
    "mortal/core/model.py",
    "mortal/core/toml_utils.py",
    "mortal/data/dataloader.py",
    "mortal/data/oracle_value.py",
    "mortal/online/pretrain_oracle_critic.py",
    "scripts/evaluate_oracle_critic_checkpoints.py",
    "scripts/run_oracle_critic_recipe_search.py",
)
ARM_SPECS = (
    {"name": "cosine_inherited", "kind": "continue", "lr_multiplier": 1.0},
    {"name": "constant_current", "kind": "constant", "lr_multiplier": 1.0},
    {"name": "constant_low", "kind": "constant", "lr_multiplier": 2.0 ** -0.5},
    {"name": "constant_high", "kind": "constant", "lr_multiplier": 2.0 ** 0.5},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a neutral, same-checkpoint Oracle critic tail-schedule rung. "
            "Every arm preserves model, optimizer moments, AMP scaler, and data progress."
        )
    )
    parser.add_argument("--anchor-config", required=True)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--runtime-overlay", required=True)
    parser.add_argument("--search-name", required=True)
    parser.add_argument("--output-root", default="logs/oracle_critic_recipe_search")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--rung-steps", type=int, default=20_000)
    parser.add_argument("--order-seed", type=int, default=20260824)
    parser.add_argument("--in-training-val-batches", type=int, default=32)
    parser.add_argument("--selection-game-modulus", type=int, default=5)
    parser.add_argument(
        "--selection-game-remainder",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--constant-arm",
        action="append",
        default=[],
        metavar="NAME=MULTIPLIER",
        help=(
            "Replace the default arms with a constant-LR arm at "
            "current_lr * MULTIPLIER. Repeat for a local LR search."
        ),
    )
    parser.add_argument(
        "--p0-weight-arm",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help=(
            "Replace the default arms with a constant-LR arm using target output "
            "weights [WEIGHT, 1, 1, 1]. Repeat for a late-stage p0 allocation search."
        ),
    )
    parser.add_argument(
        "--p0-weight-lr-multiplier",
        type=float,
        help="Use current_lr * MULTIPLIER for every --p0-weight-arm.",
    )
    return parser.parse_args()


def parse_arm_specs(values: list[str]) -> list[dict[str, Any]]:
    if not values:
        return [dict(arm) for arm in ARM_SPECS]

    arms = []
    names = set()
    for value in values:
        name, separator, multiplier_text = value.partition("=")
        if not separator:
            raise ValueError(
                f"constant arm must use NAME=MULTIPLIER syntax: {value!r}"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid constant arm name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate constant arm name: {name!r}")
        try:
            multiplier = float(multiplier_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid constant arm multiplier: {multiplier_text!r}"
            ) from exc
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(
                f"constant arm multiplier must be finite and positive: {multiplier!r}"
            )
        names.add(name)
        arms.append(
            {"name": name, "kind": "constant", "lr_multiplier": multiplier}
        )
    return arms


def parse_p0_weight_arm_specs(
    values: list[str],
    *,
    lr_multiplier: float | None,
) -> list[dict[str, Any]]:
    if not values:
        return []
    if lr_multiplier is None:
        raise ValueError(
            "--p0-weight-lr-multiplier is required with --p0-weight-arm"
        )
    if not math.isfinite(lr_multiplier) or lr_multiplier <= 0.0:
        raise ValueError("p0-weight LR multiplier must be finite and positive")

    arms = []
    names = set()
    for value in values:
        name, separator, weight_text = value.partition("=")
        if not separator:
            raise ValueError(
                f"p0 weight arm must use NAME=WEIGHT syntax: {value!r}"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid p0 weight arm name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate p0 weight arm name: {name!r}")
        try:
            p0_weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"invalid p0 output weight: {weight_text!r}") from exc
        if not math.isfinite(p0_weight) or p0_weight <= 0.0:
            raise ValueError("p0 output weight must be finite and positive")
        names.add(name)
        arms.append(
            {
                "name": name,
                "kind": "constant",
                "lr_multiplier": float(lr_multiplier),
                "target_output_weights": [p0_weight, 1.0, 1.0, 1.0],
            }
        )
    return arms


def normalize_output_weights(weights: list[float]) -> list[float]:
    if len(weights) != 4:
        raise ValueError("target output weights must contain four values")
    values = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in values):
        raise ValueError("target output weights must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("target output weights must contain a positive value")
    return [weight * 4.0 / total for weight in values]


def resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def configure_runtime_overlay(value: str | Path) -> Path:
    overlay = resolve_path(value).resolve()
    package_init = overlay / "libriichi" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"libriichi runtime overlay is incomplete: {overlay}")
    overlay_text = str(overlay)
    if overlay_text not in sys.path:
        sys.path.insert(0, overlay_text)
    current_python_path = os.environ.get("PYTHONPATH", "")
    python_path_parts = [
        part for part in current_python_path.split(os.pathsep) if part
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [overlay_text, *[part for part in python_path_parts if part != overlay_text]]
    )
    importlib.invalidate_caches()
    return overlay


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def scheduler_contract(scheduler_cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: scheduler_cfg.get(key) for key in SCHEDULER_CONTRACT_KEYS}


def current_schedule_lr(state: dict[str, Any]) -> float:
    scheduler_state = state.get("scheduler", {})
    optimizer_state = state.get("optimizer", {})
    scheduler_lrs = [float(value) for value in scheduler_state.get("_last_lr", [])]
    optimizer_lrs = [
        float(group["lr"])
        for group in optimizer_state.get("param_groups", [])
    ]
    if not scheduler_lrs or scheduler_lrs != optimizer_lrs:
        raise ValueError("anchor optimizer and scheduler learning rates do not match")
    if len(scheduler_lrs) != len(scheduler_state.get("base_lrs", [])):
        raise ValueError("anchor scheduler base LR count does not match optimizer groups")
    return max(scheduler_lrs)


def validate_anchor_state(state: dict[str, Any]) -> None:
    required = (
        "oracle_brain",
        "value_net",
        "optimizer",
        "scheduler",
        "scaler",
        "data_progress",
        "training_contract",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"anchor checkpoint is missing: {', '.join(missing)}")
    if not bool(state.get("resume_supported", False)):
        raise ValueError("anchor checkpoint does not support exact resume")
    if int(state.get("steps", 0)) <= 0:
        raise ValueError("anchor checkpoint has no completed optimizer steps")
    current_schedule_lr(state)


def constant_scheduler_config(lr: float, target_step: int) -> dict[str, Any]:
    lr = float(lr)
    if lr <= 0.0:
        raise ValueError("constant scheduler LR must be positive")
    return {
        "init": lr,
        "peak": lr,
        "final": lr,
        "warm_up_steps": 0,
        "max_steps": int(target_step),
    }


def make_branch_config(
    base_cfg: dict[str, Any],
    *,
    arm: dict[str, Any],
    arm_dir: Path,
    search_name: str,
    anchor_step: int,
    target_step: int,
    current_lr: float,
    in_training_val_batches: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = deepcopy(base_cfg)
    pretrain = cfg.setdefault("oracle_critic_pretrain", {})
    arm_name = str(arm["name"])
    if arm["kind"] == "continue":
        saved_scheduler = pretrain.get("scheduler", {})
        scheduler_cfg = scheduler_contract(saved_scheduler)
        if any(value is None for value in scheduler_cfg.values()):
            raise ValueError("anchor config does not contain a complete scheduler contract")
    elif arm["kind"] == "constant":
        scheduler_cfg = constant_scheduler_config(
            current_lr * float(arm["lr_multiplier"]),
            target_step,
        )
    else:
        raise ValueError(f"unsupported recipe arm kind: {arm['kind']!r}")

    target_output_weights = arm.get("target_output_weights")
    if target_output_weights is not None:
        pretrain["target_output_weights"] = list(target_output_weights)
        pretrain.pop("target_output_weights_initial", None)
        pretrain.pop("target_output_weight_ramp_start_steps", None)
        pretrain.pop("target_output_weight_ramp_end_steps", None)

    run_name = f"{search_name}__{arm_name}"
    recipe_fork = {
        "search_name": search_name,
        "arm": arm_name,
        "kind": str(arm["kind"]),
        "lr_multiplier": float(arm["lr_multiplier"]),
        "anchor_step": int(anchor_step),
        "target_step": int(target_step),
    }
    if target_output_weights is not None:
        recipe_fork["target_output_weights"] = list(target_output_weights)

    pretrain.update(
        {
            "run_name": run_name,
            "state_file": str((arm_dir / "checkpoints" / "latest.pth").resolve()),
            "best_state_file": str((arm_dir / "checkpoints" / "best_dev.pth").resolve()),
            "best_primary_state_file": str(
                (arm_dir / "checkpoints" / "best_primary.pth").resolve()
            ),
            "tensorboard_dir": str((arm_dir / "tb_log").resolve()),
            "metrics_file": str((arm_dir / "metrics.jsonl").resolve()),
            "max_steps": int(target_step),
            "scheduler_horizon_steps": int(scheduler_cfg["max_steps"]),
            "scheduler": scheduler_cfg,
            "convergence": {"enabled": False},
            "save_every": min(10_000, target_step - anchor_step),
            "val_every_steps": int(target_step),
            "dependency_val_every_steps": 0,
            "val_batches": int(in_training_val_batches),
            "eval_input_modes": ["true"],
            "final_test_enabled": False,
            "progress_bar": False,
            "recipe_fork": recipe_fork,
        }
    )
    return cfg, scheduler_cfg


def fork_checkpoint_state(
    state: dict[str, Any],
    *,
    branch_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    arm: dict[str, Any],
    anchor_path: Path,
    anchor_sha256: str,
    target_step: int,
) -> dict[str, Any]:
    validate_anchor_state(state)
    anchor_step = int(state["steps"])
    if target_step <= anchor_step:
        raise ValueError(
            f"target step must exceed anchor step: {target_step} <= {anchor_step}"
        )
    contract = deepcopy(state["training_contract"])
    contract.pop("convergence", None)
    contract["scheduler"] = scheduler_contract(scheduler_cfg)
    target_output_weights = arm.get("target_output_weights")
    if target_output_weights is not None:
        contract["target_output_weights"] = normalize_output_weights(
            target_output_weights
        )
        contract.pop("target_output_weight_schedule", None)
    state["training_contract"] = contract
    state["convergence_state"] = None
    state["config"] = deepcopy(branch_cfg)
    state["oracle_critic_pretrain"] = deepcopy(
        branch_cfg["oracle_critic_pretrain"]
    )

    if arm["kind"] == "constant":
        target_lr = float(scheduler_cfg["peak"])
        scheduler_state = state["scheduler"]
        base_lrs = [float(value) for value in scheduler_state["base_lrs"]]
        if not math.isclose(max(base_lrs), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "constant recipe fork requires normalized scheduler base LRs"
            )
        group_lrs = [base_lr * target_lr for base_lr in base_lrs]
        optimizer_groups = state["optimizer"]["param_groups"]
        if len(optimizer_groups) != len(group_lrs):
            raise ValueError("optimizer and scheduler group counts differ")
        for group, group_lr in zip(optimizer_groups, group_lrs):
            group["lr"] = group_lr
        scheduler_state.update(
            {
                **scheduler_cfg,
                "offset": 0,
                "epoch_size": 0,
                "tail_lr": target_lr,
                "_last_lr": group_lrs,
            }
        )

    fork_record = {
        "format": "oracle_critic_recipe_fork_v1",
        "timestamp": time.time(),
        "anchor_path": str(anchor_path.resolve()),
        "anchor_sha256": anchor_sha256,
        "anchor_step": anchor_step,
        "target_step": int(target_step),
        "arm": str(arm["name"]),
        "kind": str(arm["kind"]),
        "lr_multiplier": float(arm["lr_multiplier"]),
        "scheduler": scheduler_contract(scheduler_cfg),
    }
    if target_output_weights is not None:
        fork_record["target_output_weights"] = list(target_output_weights)
    state["recipe_fork"] = fork_record
    return state


def checkpoint_step(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    step = int(state.get("steps", 0))
    del state
    return step


def source_fingerprint(runtime_overlay: Path) -> dict[str, Any]:
    records = {}
    for relative in SOURCE_FILES:
        source = REPO_ROOT / relative
        records[relative] = file_sha256(source)
    import libriichi.libriichi as native
    from libriichi.dataset import GameplayLoader

    loader = GameplayLoader(version=4, oracle=True)
    if not hasattr(loader, "set_sample_fold"):
        raise RuntimeError(
            "configured libriichi runtime overlay does not provide set_sample_fold"
        )

    native_path = Path(native.__file__).resolve()
    if runtime_overlay not in native_path.parents:
        raise RuntimeError(
            f"libriichi imported from {native_path}, expected overlay {runtime_overlay}"
        )
    return {
        "files": records,
        "libriichi_native": {
            "path": str(native_path),
            "sha256": file_sha256(native_path),
        },
    }


def run_arm(
    *,
    arm: dict[str, Any],
    arm_dir: Path,
    config_path: Path,
    python_exe: str,
    target_step: int,
) -> None:
    latest = arm_dir / "checkpoints" / "latest.pth"
    if checkpoint_step(latest) >= target_step:
        return
    environment = os.environ.copy()
    environment["MORTAL_CFG"] = str(config_path.resolve())
    log_path = arm_dir / f"train_to_{target_step}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        result = subprocess.run(
            [python_exe, "-m", "mortal.online.pretrain_oracle_critic"],
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"recipe arm {arm['name']} exited with code {result.returncode}; "
            f"see {log_path}"
        )
    final_step = checkpoint_step(latest)
    if final_step != target_step:
        raise RuntimeError(
            f"recipe arm {arm['name']} stopped at {final_step}, expected {target_step}"
        )


def run_paired_selection_eval(
    *,
    search_dir: Path,
    ordered_arms: list[dict[str, Any]],
    python_exe: str,
    eval_state_fold_count: int,
    game_modulus: int,
    game_remainders: list[int],
) -> Path:
    output_path = search_dir / "paired_selection_dev.json"
    config_path = search_dir / ordered_arms[0]["name"] / "config.toml"
    command = [
        python_exe,
        str(REPO_ROOT / "scripts" / "evaluate_oracle_critic_checkpoints.py"),
        "--config",
        str(config_path),
        "--split",
        "dev",
        "--device",
        "cuda",
        "--input-mode",
        "true",
        "--eval-state-fold-count",
        str(eval_state_fold_count),
        "--game-id-modulus",
        str(game_modulus),
        "--output",
        str(output_path),
        "--no-print-result",
    ]
    for remainder in game_remainders:
        command.extend(("--game-id-remainder", str(remainder)))
    for arm in ordered_arms:
        checkpoint = search_dir / arm["name"] / "checkpoints" / "latest.pth"
        command.extend(("--checkpoint", f"{arm['name']}={checkpoint}"))
    log_path = search_dir / "paired_selection_dev.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"paired selection eval failed; see {log_path}")
    return output_path


def main() -> int:
    args = parse_args()
    if args.rung_steps <= 0:
        raise ValueError("rung steps must be positive")
    if args.in_training_val_batches <= 0:
        raise ValueError("in-training val batches must be positive")
    if args.selection_game_modulus <= 1:
        raise ValueError("selection game modulus must be greater than one")
    if args.constant_arm and args.p0_weight_arm:
        raise ValueError("--constant-arm and --p0-weight-arm cannot be combined")
    selection_remainders = args.selection_game_remainder or list(
        range(args.selection_game_modulus - 1)
    )
    if any(
        remainder < 0 or remainder >= args.selection_game_modulus
        for remainder in selection_remainders
    ):
        raise ValueError("selection game remainder is outside its modulus")

    sys.path.insert(0, str(REPO_ROOT))
    from mortal.core.toml_utils import load_toml_file, write_toml_file

    anchor_config = resolve_path(args.anchor_config).resolve()
    anchor_checkpoint = resolve_path(args.anchor_checkpoint).resolve()
    runtime_overlay = configure_runtime_overlay(args.runtime_overlay)
    search_dir = resolve_path(args.output_root).resolve() / args.search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_toml_file(anchor_config)
    anchor_sha256 = file_sha256(anchor_checkpoint)
    anchor_state = torch.load(anchor_checkpoint, map_location="cpu", weights_only=False)
    validate_anchor_state(anchor_state)
    anchor_step = int(anchor_state["steps"])
    current_lr = current_schedule_lr(anchor_state)
    target_step = anchor_step + int(args.rung_steps)
    del anchor_state

    ordered_arms = (
        parse_p0_weight_arm_specs(
            args.p0_weight_arm,
            lr_multiplier=args.p0_weight_lr_multiplier,
        )
        if args.p0_weight_arm
        else parse_arm_specs(args.constant_arm)
    )
    random.Random(args.order_seed).shuffle(ordered_arms)
    manifest = {
        "format": "oracle_critic_recipe_search_v1",
        "created_at": time.time(),
        "search_name": args.search_name,
        "anchor": {
            "config": str(anchor_config),
            "config_sha256": file_sha256(anchor_config),
            "checkpoint": str(anchor_checkpoint),
            "checkpoint_sha256": anchor_sha256,
            "step": anchor_step,
            "current_lr": current_lr,
        },
        "rung_steps": int(args.rung_steps),
        "target_step": target_step,
        "order_seed": int(args.order_seed),
        "arm_order": [arm["name"] for arm in ordered_arms],
        "arms": ordered_arms,
        "selection_game_subset": {
            "modulus": int(args.selection_game_modulus),
            "remainders": selection_remainders,
        },
        "runtime_overlay": str(runtime_overlay),
        "source": source_fingerprint(runtime_overlay),
    }
    manifest_path = search_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "search_name",
            "anchor",
            "rung_steps",
            "target_step",
            "order_seed",
            "arm_order",
            "arms",
            "selection_game_subset",
            "runtime_overlay",
            "source",
        )
        mismatched_keys = [
            key
            for key in comparable_keys
            if existing.get(key) != manifest.get(key)
        ]
        if mismatched_keys:
            raise ValueError(
                "existing recipe-search manifest does not match this run: "
                + ", ".join(mismatched_keys)
            )
    else:
        atomic_write_json(manifest_path, manifest)

    status_path = search_dir / "status.json"
    configs = {}
    for arm in ordered_arms:
        arm_dir = search_dir / arm["name"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        branch_cfg, scheduler_cfg = make_branch_config(
            base_cfg,
            arm=arm,
            arm_dir=arm_dir,
            search_name=args.search_name,
            anchor_step=anchor_step,
            target_step=target_step,
            current_lr=current_lr,
            in_training_val_batches=args.in_training_val_batches,
        )
        config_path = arm_dir / "config.toml"
        write_toml_file(config_path, branch_cfg)
        configs[arm["name"]] = config_path
        latest = arm_dir / "checkpoints" / "latest.pth"
        if not latest.exists():
            state = torch.load(anchor_checkpoint, map_location="cpu", weights_only=False)
            fork_checkpoint_state(
                state,
                branch_cfg=branch_cfg,
                scheduler_cfg=scheduler_cfg,
                arm=arm,
                anchor_path=anchor_checkpoint,
                anchor_sha256=anchor_sha256,
                target_step=target_step,
            )
            atomic_torch_save(latest, state)
            del state

    atomic_write_json(
        status_path,
        {
            "status": "prepared",
            "anchor_step": anchor_step,
            "target_step": target_step,
            "arm_order": [arm["name"] for arm in ordered_arms],
        },
    )
    if args.prepare_only:
        return 0

    completed = []
    for arm in ordered_arms:
        atomic_write_json(
            status_path,
            {
                "status": "training",
                "active_arm": arm["name"],
                "completed_arms": completed,
                "anchor_step": anchor_step,
                "target_step": target_step,
                "arm_order": [item["name"] for item in ordered_arms],
            },
        )
        run_arm(
            arm=arm,
            arm_dir=search_dir / arm["name"],
            config_path=configs[arm["name"]],
            python_exe=args.python_exe,
            target_step=target_step,
        )
        completed.append(arm["name"])

    atomic_write_json(
        status_path,
        {
            "status": "paired_eval",
            "completed_arms": completed,
            "anchor_step": anchor_step,
            "target_step": target_step,
            "arm_order": [arm["name"] for arm in ordered_arms],
        },
    )
    eval_state_fold_count = int(
        base_cfg["oracle_critic_pretrain"].get("val_state_fold_count", 1) or 1
    )
    output_path = run_paired_selection_eval(
        search_dir=search_dir,
        ordered_arms=ordered_arms,
        python_exe=args.python_exe,
        eval_state_fold_count=eval_state_fold_count,
        game_modulus=args.selection_game_modulus,
        game_remainders=selection_remainders,
    )
    atomic_write_json(
        status_path,
        {
            "status": "complete",
            "completed_arms": completed,
            "anchor_step": anchor_step,
            "target_step": target_step,
            "arm_order": [arm["name"] for arm in ordered_arms],
            "paired_selection_dev": str(output_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

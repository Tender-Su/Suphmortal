from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PAUSE_EXIT_CODE = 75
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mortal.core.toml_utils import load_toml_file, write_toml_file


DEFAULT_CASES = (
    {
        "name": "base_lr50_wd100_v100",
        "scheduler_peak": 5e-5,
        "weight_decay": 0.1,
        "visible_lr_scale": 1.0,
        "reserve_ratio": 0.0,
    },
    {
        "name": "wd030",
        "scheduler_peak": 5e-5,
        "weight_decay": 0.03,
        "visible_lr_scale": 1.0,
        "reserve_ratio": 0.0,
    },
    {
        "name": "wd010",
        "scheduler_peak": 5e-5,
        "weight_decay": 0.01,
        "visible_lr_scale": 1.0,
        "reserve_ratio": 0.0,
    },
    {
        "name": "lr025",
        "scheduler_peak": 2.5e-5,
        "weight_decay": 0.1,
        "visible_lr_scale": 1.0,
        "reserve_ratio": 0.0,
    },
    {
        "name": "lr075",
        "scheduler_peak": 7.5e-5,
        "weight_decay": 0.1,
        "visible_lr_scale": 1.0,
        "reserve_ratio": 0.0,
    },
    {
        "name": "visible025",
        "scheduler_peak": 5e-5,
        "weight_decay": 0.1,
        "visible_lr_scale": 0.25,
        "reserve_ratio": 0.0,
    },
    {
        "name": "visible050",
        "scheduler_peak": 5e-5,
        "weight_decay": 0.1,
        "visible_lr_scale": 0.5,
        "reserve_ratio": 0.0,
    },
    {
        "name": "reserve050",
        "scheduler_peak": 5e-5,
        "weight_decay": 0.1,
        "visible_lr_scale": 1.0,
        "reserve_ratio": 0.5,
    },
)

SOURCE_FINGERPRINT_FILES = (
    "libriichi/src/dataset/gameplay.rs",
    "mortal/core/common.py",
    "mortal/core/adaptive_curriculum.py",
    "mortal/core/lr_scheduler.py",
    "mortal/core/model.py",
    "mortal/core/toml_utils.py",
    "mortal/data/dataloader.py",
    "mortal/data/oracle_value.py",
    "mortal/online/pretrain_oracle_critic.py",
    "mortal/supervised/convergence.py",
    "scripts/run_oracle_critic_search.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable, same-base Oracle critic hyperparameter arms."
    )
    parser.add_argument(
        "--base-config",
        default=(
            "logs/oracle_critic_formal/"
            "s70_dual_all_score_rank_mc_s2500000_20260726_r1/config.toml"
        ),
    )
    parser.add_argument("--search-name", default="s70_dual_stage0_20260816")
    parser.add_argument("--output-root", default="logs/oracle_critic_search")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--runtime-overlay",
        default="logs/runtime_overlays/libriichi_native_fold_capacity_v2",
    )
    parser.add_argument("--stage-steps", type=int, default=5000)
    parser.add_argument("--scheduler-horizon-steps", type=int, default=2500000)
    parser.add_argument("--val-every-steps", type=int, default=2500)
    parser.add_argument("--val-batches", type=int, default=256)
    parser.add_argument("--test-batches", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--file-batch-size", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--val-file-batch-size", type=int, default=8)
    parser.add_argument("--val-prefetch-factor", type=int, default=5)
    parser.add_argument("--dependency-val-every-steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--split-seed", type=int, default=20260416)
    parser.add_argument("--val-game-id-modulus", type=int, default=5)
    parser.add_argument(
        "--val-game-id-remainder",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--paired-selection-eval", action="store_true")
    parser.add_argument("--selection-game-id-modulus", type=int, default=5)
    parser.add_argument(
        "--selection-game-id-remainder",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--selection-max-batches", type=int, default=0)
    parser.add_argument("--selection-state-fold-count", type=int, default=128)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--case-file", default="")
    parser.add_argument("--train-file-index", default="")
    parser.add_argument("--dev-file-index", default="")
    parser.add_argument("--test-file-index", default="")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def configure_runtime_overlay(value: str | Path) -> Path:
    overlay = resolve_path(value).resolve()
    package_init = overlay / "libriichi" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"libriichi runtime overlay is incomplete: {overlay}")
    overlay_text = str(overlay)
    if overlay_text not in sys.path:
        sys.path.insert(0, overlay_text)
    python_path = [
        part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [overlay_text, *[part for part in python_path if part != overlay_text]]
    )
    importlib.invalidate_caches()
    return overlay


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(resolved),
    }


def runtime_artifact_record() -> dict[str, Any]:
    native = importlib.import_module("libriichi.libriichi")
    native_file = getattr(native, "__file__", None)
    if not native_file:
        raise RuntimeError("loaded libriichi extension does not expose __file__")
    return artifact_record(Path(native_file))


def optimizer_package_records(cases: list[dict[str, Any]]) -> dict[str, str]:
    optimizer_types = {
        str(case.get("optimizer", {}).get("type", "adamw"))
        .strip()
        .lower()
        .replace("-", "_")
        for case in cases
    }
    if not optimizer_types.intersection(
        {"schedulefree", "schedule_free_adamw", "adamw_schedule_free"}
    ):
        return {}
    try:
        version = importlib.metadata.version("schedulefree")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "schedule-free search cases require schedulefree==1.4.1"
        ) from exc
    if version != "1.4.1":
        raise RuntimeError(
            f"schedule-free recipe audit is pinned to schedulefree==1.4.1, got {version}"
        )
    return {"schedulefree": version}


def base_artifact_records(base_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pretrain = base_cfg.get("oracle_critic_pretrain", {})
    records = {}
    index_keys = (
        ("train_file_index", "dev_file_index", "test_file_index")
        if pretrain.get("train_file_index")
        else ("file_index",)
    )
    for key in ("init_state_file", *index_keys):
        raw_value = str(pretrain.get(key, "") or "")
        if not raw_value and key == "test_file_index":
            continue
        if not raw_value:
            raise ValueError(f"base config is missing oracle_critic_pretrain.{key}")
        artifact = Path(raw_value)
        if not artifact.is_file():
            raise FileNotFoundError(f"base artifact does not exist: {artifact}")
        records[key] = artifact_record(artifact)
    return records


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def source_fingerprint() -> tuple[str, dict[str, str]]:
    hashes = {}
    digest = hashlib.sha256()
    for relative in SOURCE_FINGERPRINT_FILES:
        source = REPO_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"training source file does not exist: {source}")
        sha256 = file_sha256(source)
        hashes[relative] = sha256
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), hashes


def load_case_file(case_file: str) -> list[dict[str, Any]]:
    case_path = resolve_path(case_file).resolve()
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Oracle critic case file must contain a non-empty JSON list")
    cases = []
    for item in payload:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            raise ValueError("every Oracle critic case must be an object with a name")
        cases.append(deepcopy(item))
    case_names = [case["name"] for case in cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError("Oracle critic case names must be unique")
    return cases


def selected_cases(names: list[str], case_file: str = "") -> list[dict[str, Any]]:
    cases = load_case_file(case_file) if case_file else [dict(case) for case in DEFAULT_CASES]
    if not names:
        return cases
    requested = set(names)
    selected = [case for case in cases if case["name"] in requested]
    missing = sorted(requested - {case["name"] for case in selected})
    if missing:
        raise ValueError(f"unknown Oracle critic search case(s): {', '.join(missing)}")
    return selected


def case_uses_convergence(case: dict[str, Any]) -> bool:
    pretrain = case.get("pretrain", {})
    if not isinstance(pretrain, dict):
        return False
    convergence = pretrain.get("convergence", {})
    return isinstance(convergence, dict) and bool(convergence.get("enabled", False))


def case_scheduler_type(case: dict[str, Any]) -> str:
    scheduler = case.get("scheduler", {})
    if not isinstance(scheduler, dict):
        raise ValueError(f"case {case['name']!r} scheduler must be an object")
    value = str(scheduler.get("type", "cosine")).strip().lower().replace("-", "_")
    aliases = {"schedule_free": "optimizer", "none": "optimizer"}
    return aliases.get(value, value)


def make_case_config(
    base_cfg: dict[str, Any],
    case: dict[str, Any],
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    pretrain = cfg.setdefault("oracle_critic_pretrain", {})
    scheduler: dict[str, Any] = {}
    pretrain["scheduler"] = scheduler
    run_name = f"{args.search_name}__{case['name']}__seed{args.seed}"
    pretrain.update(
        {
            "run_name": run_name,
            "state_file": str((case_dir / "checkpoints" / "latest.pth").resolve()),
            "best_state_file": str((case_dir / "checkpoints" / "best_dev.pth").resolve()),
            "best_primary_state_file": str(
                (case_dir / "checkpoints" / "best_primary.pth").resolve()
            ),
            "adaptive_best_state_file": str(
                (case_dir / "checkpoints" / "adaptive_best.pth").resolve()
            ),
            "tensorboard_dir": str((case_dir / "tb_log").resolve()),
            "metrics_file": str((case_dir / "metrics.jsonl").resolve()),
            "critic_arch": str(case.get("critic_arch", pretrain.get("critic_arch", "dual_tower"))),
            "train_scope": "all",
            "target_mode": "all_players",
            "return_mode": "score_rank_mc",
            "discount_gamma": 0.999,
            "value_loss_mode": str(case.get("value_loss_mode", "mse")),
            "value_head_hidden": int(case.get("value_head_hidden", 256)),
            "batch_size": 640,
            "num_workers": int(args.num_workers),
            "file_batch_size": int(args.file_batch_size),
            "prefetch_factor": int(args.prefetch_factor),
            "val_num_workers": int(args.val_num_workers),
            "val_file_batch_size": int(args.val_file_batch_size),
            "val_prefetch_factor": int(args.val_prefetch_factor),
            "max_steps": int(args.stage_steps),
            "scheduler_horizon_steps": int(args.scheduler_horizon_steps),
            "log_every": 100,
            "save_every": int(args.save_every),
            "val_every_steps": int(args.val_every_steps),
            "dependency_val_every_steps": int(args.dependency_val_every_steps),
            "eval_input_modes": ["true"],
            "val_batches": int(args.val_batches),
            "test_batches": int(args.test_batches),
            "val_ratio": 0.01,
            "test_ratio": 0.01,
            "min_val_files": 1024,
            "min_test_files": 2048,
            "max_val_files": 0,
            "max_test_files": 0,
            "max_train_files": 0,
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "data_shuffle_seed": int(args.seed),
            "num_epochs": 1,
            "enable_augmentation": False,
            "augmented_first": False,
            "reserve_ratio": float(case.get("reserve_ratio", 0.0)),
            "zero_sum_weight": float(
                case.get("zero_sum_weight", pretrain.get("zero_sum_weight", 0.0))
            ),
            "visible_lr_scale": float(case.get("visible_lr_scale", 1.0)),
            "oracle_lr_scale": float(case.get("oracle_lr_scale", 1.0)),
            "fusion_lr_scale": float(case.get("fusion_lr_scale", 1.0)),
            "value_lr_scale": float(case.get("value_lr_scale", 1.0)),
            "weight_decay": float(case.get("weight_decay", 0.1)),
            "final_test_enabled": bool(args.final_test),
            "progress_bar": False,
            "val_game_id_modulus": int(args.val_game_id_modulus),
            "val_game_id_remainders": list(
                args.val_game_id_remainder or [0]
            ),
        }
    )
    pretrain.update(deepcopy(case.get("pretrain", {})))
    if "convergence" not in case.get("pretrain", {}):
        pretrain["convergence"] = {"enabled": False}
    pretrain["optimizer"] = deepcopy(case.get("optimizer", {"type": "adamw"}))
    scheduler.update(
        {
            "type": "cosine",
            "peak": float(case.get("scheduler_peak", 5e-5)),
            "final": 1e-5,
            "warm_up_steps": 2000,
            "max_steps": int(args.scheduler_horizon_steps),
            "init": 1e-8,
        }
    )
    case_scheduler = deepcopy(case.get("scheduler", {}))
    if case_scheduler:
        scheduler.clear()
        scheduler.update(case_scheduler)
    return cfg


def resolve_base_artifact_paths(base_cfg: dict[str, Any], base_config: Path) -> None:
    pretrain = base_cfg.get("oracle_critic_pretrain", {})
    if not isinstance(pretrain, dict):
        return
    for key in (
        "file_index",
        "train_file_index",
        "dev_file_index",
        "test_file_index",
        "init_state_file",
    ):
        raw_value = str(pretrain.get(key, "") or "")
        if not raw_value or Path(raw_value).is_absolute():
            continue
        relative = raw_value[2:] if raw_value.startswith("./") else raw_value
        candidate = (base_config.parent / relative).resolve()
        if candidate.exists():
            pretrain[key] = str(candidate)


def apply_explicit_split_overrides(base_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    values = {
        "train_file_index": args.train_file_index,
        "dev_file_index": args.dev_file_index,
        "test_file_index": args.test_file_index,
    }
    if not any(values.values()):
        return
    if not values["train_file_index"] or not values["dev_file_index"]:
        raise ValueError(
            "search split overrides require --train-file-index and --dev-file-index"
        )
    pretrain = base_cfg.setdefault("oracle_critic_pretrain", {})
    for key, value in values.items():
        pretrain[key] = str(resolve_path(value).resolve()) if value else ""


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def case_summary(case: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    rows = [row for row in read_metrics(case_dir / "metrics.jsonl") if "val" in row]
    result: dict[str, Any] = {"case": case["name"], "spec": case}
    if not rows:
        result.update({"steps": 0, "status": "no_metrics"})
        return result
    final = max(rows, key=lambda row: int(row.get("steps", 0)))
    best = min(rows, key=lambda row: float(row["val"]["loss"]))
    best_primary = min(
        rows,
        key=lambda row: float(
            row["val"]["outputs"]["relative_player_0"]["loss"]
        ),
    )
    final_convergence = final.get("convergence")
    converged = isinstance(final_convergence, dict) and bool(
        final_convergence.get("converged", False)
    )
    final_adaptive = final.get("adaptive_curriculum")
    adaptive_complete = isinstance(final_adaptive, dict) and bool(
        final_adaptive.get("completed", False)
    )
    result.update(
        {
            "status": (
                "adaptive_complete"
                if adaptive_complete
                else "converged"
                if converged
                else "complete"
            ),
            "steps": int(final["steps"]),
            "final_val": final["val"],
            "best_dev_steps": int(best["steps"]),
            "best_dev": best["val"],
            "best_primary_steps": int(best_primary["steps"]),
            "best_primary": best_primary["val"]["outputs"]["relative_player_0"],
            "final_oracle_dependency": final.get("oracle_dependency", {}),
            "final_convergence": final_convergence,
            "final_adaptive_curriculum": final_adaptive,
        }
    )
    return result


def write_search_summary(search_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [case_summary(case, search_dir / case["name"]) for case in cases]
    summaries.sort(
        key=lambda item: (
            item.get("status") != "complete",
            float(
                item.get("final_val", {})
                .get("outputs", {})
                .get("relative_player_0", {})
                .get("loss", float("inf"))
            ),
        )
    )
    atomic_write_json(search_dir / "summary.json", summaries)
    return summaries


def preserve_stage_checkpoint(case_dir: Path, stage_steps: int) -> Path | None:
    summary_rows = [
        row for row in read_metrics(case_dir / "metrics.jsonl") if "val" in row
    ]
    if not summary_rows:
        return None
    completed_steps = max(int(row.get("steps", 0)) for row in summary_rows)
    if completed_steps != int(stage_steps):
        return None

    latest = case_dir / "checkpoints" / "latest.pth"
    if not latest.is_file():
        raise FileNotFoundError(
            f"completed stage {stage_steps} has no latest checkpoint: {latest}"
        )
    destination = latest.with_name(f"step_{int(stage_steps):06d}.pth")
    if destination.exists():
        return destination

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(latest, temporary)
    temporary.replace(destination)
    return destination


def prepare_search_snapshot(
    search_dir: Path,
    base_config: Path,
    base_cfg: dict[str, Any],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    search_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = search_dir / "manifest.json"
    patch_file = search_dir / "source.patch"
    source_sha256, source_files = source_fingerprint()
    base_config_sha256 = file_sha256(base_config)
    runtime_artifact = runtime_artifact_record()
    runtime_overlay = resolve_path(args.runtime_overlay).resolve()
    optimizer_packages = optimizer_package_records(cases)
    base_artifacts = base_artifact_records(base_cfg)
    existing = None
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        checks = {
            "source_fingerprint_sha256": source_sha256,
            "base_config_sha256": base_config_sha256,
            "scheduler_horizon_steps": int(args.scheduler_horizon_steps),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "val_every_steps": int(args.val_every_steps),
            "dependency_val_every_steps": int(args.dependency_val_every_steps),
            "val_batches": int(args.val_batches),
            "test_batches": int(args.test_batches),
            "num_workers": int(args.num_workers),
            "file_batch_size": int(args.file_batch_size),
            "prefetch_factor": int(args.prefetch_factor),
            "val_num_workers": int(args.val_num_workers),
            "val_file_batch_size": int(args.val_file_batch_size),
            "val_prefetch_factor": int(args.val_prefetch_factor),
            "runtime_artifact_sha256": runtime_artifact["sha256"],
            "runtime_overlay": str(runtime_overlay),
            "optimizer_packages": optimizer_packages,
            "base_artifacts": base_artifacts,
            "val_game_id_modulus": int(args.val_game_id_modulus),
            "val_game_id_remainders": list(args.val_game_id_remainder or [0]),
            "selection_game_id_modulus": int(args.selection_game_id_modulus),
            "selection_game_id_remainders": list(
                args.selection_game_id_remainder or [1, 2, 3]
            ),
            "selection_max_batches": int(args.selection_max_batches),
            "selection_state_fold_count": int(args.selection_state_fold_count),
        }
        for key, expected in checks.items():
            if existing.get(key) != expected:
                raise RuntimeError(
                    f"search directory contract mismatch for {key}: "
                    f"saved={existing.get(key)!r} current={expected!r}; use a new search name"
                )
        existing_cases = {
            str(case["name"]): case
            for case in existing.get("cases", [])
        }
        for case in cases:
            saved_case = existing_cases.get(str(case["name"]))
            if saved_case is not None and saved_case != case:
                raise RuntimeError(
                    f"search case {case['name']!r} changed; use a new search name"
                )
            existing_cases[str(case["name"])] = case
        manifest_cases = list(existing_cases.values())
    else:
        manifest_cases = cases
    if not patch_file.exists():
        patch_file.write_text(
            git_output("diff", "--binary", "--", *SOURCE_FINGERPRINT_FILES),
            encoding="utf-8",
        )
    atomic_write_json(
        manifest_file,
        {
            "format": "oracle_critic_search_v2",
            "updated_at": time.time(),
            "source_commit": git_output("rev-parse", "HEAD").strip(),
            "source_patch": patch_file.name,
            "source_patch_sha256": file_sha256(patch_file),
            "source_fingerprint_sha256": source_sha256,
            "source_files": source_files,
            "base_config": str(base_config),
            "base_config_sha256": base_config_sha256,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "runtime_artifact": runtime_artifact,
            "runtime_artifact_sha256": runtime_artifact["sha256"],
            "runtime_overlay": str(runtime_overlay),
            "optimizer_packages": optimizer_packages,
            "base_artifacts": base_artifacts,
            "search_name": args.search_name,
            "stage_steps": max(
                int(args.stage_steps),
                int(existing.get("stage_steps", 0)) if existing else 0,
            ),
            "scheduler_horizon_steps": int(args.scheduler_horizon_steps),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "val_every_steps": int(args.val_every_steps),
            "dependency_val_every_steps": int(args.dependency_val_every_steps),
            "val_batches": int(args.val_batches),
            "test_batches": int(args.test_batches),
            "num_workers": int(args.num_workers),
            "file_batch_size": int(args.file_batch_size),
            "prefetch_factor": int(args.prefetch_factor),
            "val_num_workers": int(args.val_num_workers),
            "val_file_batch_size": int(args.val_file_batch_size),
            "val_prefetch_factor": int(args.val_prefetch_factor),
            "val_game_id_modulus": int(args.val_game_id_modulus),
            "val_game_id_remainders": list(args.val_game_id_remainder or [0]),
            "selection_game_id_modulus": int(args.selection_game_id_modulus),
            "selection_game_id_remainders": list(
                args.selection_game_id_remainder or [1, 2, 3]
            ),
            "selection_max_batches": int(args.selection_max_batches),
            "selection_state_fold_count": int(args.selection_state_fold_count),
            "cases": manifest_cases,
        },
    )


def run_case(case: dict[str, Any], case_dir: Path, config_path: Path, args: argparse.Namespace) -> int:
    summary = case_summary(case, case_dir)
    if summary.get("status") in {"converged", "adaptive_complete"}:
        print(
            f"[{case['name']}] already {summary['status']} at "
            f"step {summary['steps']}; skipping"
        )
        return 0
    if int(summary.get("steps", 0)) >= args.stage_steps:
        preserve_stage_checkpoint(case_dir, args.stage_steps)
        print(f"[{case['name']}] already reached step {summary['steps']}; skipping")
        return 0

    log_path = case_dir / f"train_to_{args.stage_steps}.log"
    environment = os.environ.copy()
    environment["MORTAL_CFG"] = str(config_path.resolve())
    command = [args.python_exe, "-m", "mortal.online.pretrain_oracle_critic"]
    print(f"[{case['name']}] launching to step {args.stage_steps}")
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if " val_loss=" in line or "input=true val_loss=" in line or "final test" in line:
                print(f"[{case['name']}] {line.rstrip()}")
        return_code = process.wait()
    print(f"[{case['name']}] exited with code {return_code}")
    if return_code == 0:
        preserved = preserve_stage_checkpoint(case_dir, args.stage_steps)
        if preserved is not None:
            print(f"[{case['name']}] preserved {preserved.name}")
    return return_code


def run_paired_selection_eval(
    cases: list[dict[str, Any]],
    search_dir: Path,
    configs: dict[str, Path],
    args: argparse.Namespace,
) -> int:
    if len(cases) < 2:
        raise ValueError("paired selection evaluation requires at least two cases")
    output = search_dir / f"paired_selection_step_{int(args.stage_steps):07d}.json"
    if output.is_file():
        print(f"paired selection already exists: {output}")
        return 0
    checkpoints = []
    for case in cases:
        checkpoint = (
            search_dir
            / case["name"]
            / "checkpoints"
            / f"step_{int(args.stage_steps):06d}.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"selection checkpoint is missing for {case['name']}: {checkpoint}"
            )
        checkpoints.append((case["name"], checkpoint))

    command = [
        args.python_exe,
        str(REPO_ROOT / "scripts" / "evaluate_oracle_critic_checkpoints.py"),
        "--config",
        str(configs[cases[0]["name"]]),
        "--split",
        "dev",
        "--input-mode",
        "true",
        "--eval-state-fold-count",
        str(args.selection_state_fold_count),
        "--game-id-modulus",
        str(args.selection_game_id_modulus),
        "--max-batches",
        str(args.selection_max_batches),
        "--output",
        str(output),
        "--no-print-result",
    ]
    for remainder in args.selection_game_id_remainder or [1, 2, 3]:
        command.extend(["--game-id-remainder", str(remainder)])
    for name, checkpoint in checkpoints:
        command.extend(["--checkpoint", f"{name}={checkpoint}"])

    log_path = output.with_suffix(".log")
    print(f"launching paired selection for {len(checkpoints)} cases")
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if "progress" in line.lower() or "wrote" in line.lower():
                print(f"[selection] {line.rstrip()}")
        return_code = process.wait()
    print(f"paired selection exited with code {return_code}")
    return return_code


def main() -> int:
    args = parse_args()
    configure_runtime_overlay(args.runtime_overlay)
    cases = selected_cases(args.case, args.case_file)
    if args.stage_steps <= 0:
        raise ValueError("stage_steps must be positive")
    if (
        args.scheduler_horizon_steps < args.stage_steps
        and any(
            case_scheduler_type(case) == "cosine"
            and not case_uses_convergence(case)
            for case in cases
        )
    ):
        raise ValueError(
            "scheduler_horizon_steps must be >= stage_steps for cosine cases "
            "without convergence tails"
        )
    if args.num_workers < 0 or args.val_num_workers < 0:
        raise ValueError("DataLoader worker counts must be non-negative")
    if args.file_batch_size <= 0 or args.val_file_batch_size <= 0:
        raise ValueError("file batch sizes must be positive")
    if args.prefetch_factor <= 0 or args.val_prefetch_factor <= 0:
        raise ValueError("prefetch factors must be positive")
    for modulus, remainders, label in (
        (
            args.val_game_id_modulus,
            args.val_game_id_remainder or [0],
            "monitor validation",
        ),
        (
            args.selection_game_id_modulus,
            args.selection_game_id_remainder or [1, 2, 3],
            "selection validation",
        ),
    ):
        if modulus <= 0 or not remainders or min(remainders) < 0 or max(remainders) >= modulus:
            raise ValueError(f"invalid {label} game id subset")

    base_config = resolve_path(args.base_config).resolve()
    output_root = resolve_path(args.output_root).resolve()
    search_dir = output_root / args.search_name
    base_cfg = load_toml_file(base_config)
    resolve_base_artifact_paths(base_cfg, base_config)
    apply_explicit_split_overrides(base_cfg, args)
    prepare_search_snapshot(search_dir, base_config, base_cfg, cases, args)

    configs: dict[str, Path] = {}
    for case in cases:
        case_dir = search_dir / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        config_path = case_dir / "config.toml"
        write_toml_file(config_path, make_case_config(base_cfg, case, case_dir, args))
        configs[case["name"]] = config_path
    write_search_summary(search_dir, cases)
    if args.prepare_only:
        print(f"prepared {len(cases)} cases under {search_dir}")
        return 0

    failures = []
    for case in cases:
        return_code = run_case(
            case,
            search_dir / case["name"],
            configs[case["name"]],
            args,
        )
        summaries = write_search_summary(search_dir, cases)
        leader = next((item for item in summaries if item.get("status") == "complete"), None)
        if leader is not None:
            print(
                f"monitor_only_leader={leader['case']} step={leader['steps']} "
                f"primary_loss={leader['final_val']['outputs']['relative_player_0']['loss']:.6f} "
                f"all_loss={leader['final_val']['loss']:.6f} "
                f"corr={leader['final_val']['corr']:.4f}"
            )
        if return_code != 0:
            if return_code == EXTERNAL_PAUSE_EXIT_CODE:
                print(
                    f"[{case['name']}] paused after an external checkpoint request"
                )
                return EXTERNAL_PAUSE_EXIT_CODE
            failures.append({"case": case["name"], "returncode": return_code})
            if args.stop_on_error:
                break

    if failures:
        atomic_write_json(search_dir / "failures.json", failures)
        return 1
    if args.paired_selection_eval:
        return run_paired_selection_eval(cases, search_dir, configs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

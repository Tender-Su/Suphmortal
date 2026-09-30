from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.research.oracle_critic_curriculum import (
    checkpoint_component_hashes,
    migrate_checkpoint_for_phase,
    split_summary,
)


EXTERNAL_PAUSE_EXIT_CODE = 75
CASE_NAME = "sf_lr200_wd000"
PHASE_LENGTHS = {
    "phase_a": 630_000,
    "phase_b": 420_000,
    "phase_c": 210_000,
}
PHASE_TARGETS = {
    "phase_a": PHASE_LENGTHS["phase_a"],
    "phase_b": PHASE_LENGTHS["phase_a"] + PHASE_LENGTHS["phase_b"],
    "phase_c": sum(PHASE_LENGTHS.values()),
}
INTERIM_UNIFORM_SANITY_STEP = 190_000
DEFAULT_HARD_MAX_STEPS = 5_000_000
SCHEDULER_HORIZON_STEPS = 2_500_000
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "logs/oracle_critic_formal/"
    "s70_broad_to_recent_strong24m12m_s35duration_sf200_wd0_20260901_r1"
)
DEFAULT_CACHE_ROOT = (
    REPO_ROOT
    / "logs/oracle_event_cache/"
    "s70_broad_to_recent_strong24m12m_20260901_r1"
)
DEFAULT_SOURCE_CACHE_ROOT = (
    REPO_ROOT
    / "logs/oracle_event_cache/s70_temporal_dev202512_test202601_chunk16_v1"
)
BASE_CONFIG = (
    REPO_ROOT
    / "logs/oracle_critic_search/"
    "s70_temporal_scalar_hand_wd003_formal_20260818_r1/"
    "visible_transfer_hand_aligned_wd003/config.toml"
)
RUNTIME_OVERLAY = (
    REPO_ROOT / "logs/runtime_overlays/libriichi_native_fold_capacity_v2"
)
CASE_FILE = REPO_ROOT / "scripts/oracle_critic_curriculum_case_v1.json"
CONVERGENCE_CASE_FILE = (
    REPO_ROOT / "scripts/oracle_critic_curriculum_convergence_case_v1.json"
)
AUDITED_UNIFORM_CHECKPOINT = (
    REPO_ROOT
    / "logs/oracle_critic_search/"
    "s70_sf_wd_stage1_s10000_20260828_r2/sf_lr200_wd000/"
    "checkpoints/step_190000.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable broad-to-recent Oracle critic curriculum with the "
            "audited Schedule-Free LR/WD recipe and Apex-aware exits."
        )
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--source-cache-root", default=str(DEFAULT_SOURCE_CACHE_ROOT))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--hard-max-steps", type=int, default=DEFAULT_HARD_MAX_STEPS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--emit-supervisor-spec", action="store_true")
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
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    temporary.replace(destination)


def pause_file() -> Path | None:
    value = os.environ.get("MORTAL_ORACLE_PAUSE_FILE", "").strip()
    return Path(value) if value else None


def pause_requested() -> bool:
    marker = pause_file()
    return marker is not None and marker.is_file()


def child_environment() -> dict[str, str]:
    if not (RUNTIME_OVERLAY / "libriichi/__init__.py").is_file():
        raise FileNotFoundError(f"libriichi runtime overlay is incomplete: {RUNTIME_OVERLAY}")
    environment = os.environ.copy()
    overlay_text = str(RUNTIME_OVERLAY.resolve())
    existing = [
        item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        [overlay_text, *[item for item in existing if item != overlay_text]]
    )
    return environment


def run_child(arguments: list[str], *, python_exe: Path) -> int:
    if pause_requested():
        return EXTERNAL_PAUSE_EXIT_CODE
    completed = subprocess.run(
        [str(python_exe), "-u", *arguments],
        cwd=REPO_ROOT,
        env=child_environment(),
        check=False,
    )
    return int(completed.returncode)


def run_restartable_evaluation(
    arguments: list[str], *, python_exe: Path
) -> int:
    if pause_requested():
        return EXTERNAL_PAUSE_EXIT_CODE
    process = subprocess.Popen(
        [str(python_exe), "-u", *arguments],
        cwd=REPO_ROOT,
        env=child_environment(),
    )
    while process.poll() is None:
        if pause_requested():
            print("evaluation interrupted for Apex; safe to restart", flush=True)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return EXTERNAL_PAUSE_EXIT_CODE
        time.sleep(1)
    return int(process.returncode)


def curriculum_design(
    *, run_root: Path, cache_root: Path, source_cache_root: Path, hard_max_steps: int
) -> dict[str, Any]:
    return {
        "format": "oracle_critic_formal_curriculum_v2",
        "objective": "strongest standalone Oracle critic before actor integration",
        "initialization": {
            "type": "sl_policy_checkpoint",
            "source": str(
                (
                    REPO_ROOT
                    / "logs/sl_fidelity/"
                    "sl_anchor_longabc_s70_20260609_r1_1v3_compare/"
                    "best_action_score.pth"
                ).resolve()
            ),
            "note": "fresh Oracle critic initialization, not the ad-hoc 1.6M critic",
        },
        "audited_recipe": {
            "optimizer": "Schedule-Free AdamW",
            "lr": 0.0002,
            "weight_decay": 0.0,
            "warmup_steps": 2000,
            "evidence_checkpoint": str(AUDITED_UNIFORM_CHECKPOINT.resolve()),
        },
        "curriculum": {
            "profile": "SL broad_to_recent strong 24m_12m, s35 duration",
            "duration_basis": {
                "source_sl_profile": "longabc_s70",
                "source_sl_phase_lengths": {
                    "phase_a": 1_260_000,
                    "phase_b": 840_000,
                    "phase_c": 420_000,
                },
                "transfer_rule": (
                    "half the source SL duration because the visible trunk is "
                    "already pretrained; train the new Oracle/value path for a "
                    "full s35 curriculum before recent-data convergence"
                ),
                "critic_phase_lengths": PHASE_LENGTHS,
            },
            "phase_a": {
                "target_step": PHASE_TARGETS["phase_a"],
                "weights": {"recent_24m": 0.60, "mid": 0.25, "early": 0.15},
            },
            "phase_b": {
                "target_step": PHASE_TARGETS["phase_b"],
                "weights": {"recent_24m": 0.90, "replay": 0.10},
            },
            "phase_c": {
                "target_step": PHASE_TARGETS["phase_c"],
                "weights": {"recent_12m": 0.98, "replay": 0.02},
            },
        },
        "convergence": {
            "starts_from_step": PHASE_TARGETS["phase_c"],
            "core_optimizer_steps": SCHEDULER_HORIZON_STEPS,
            "tail_lr_levels": [0.0001, 0.00005, 0.000025, 0.00001],
            "hard_max_steps": int(hard_max_steps),
            "metric": "primary_loss",
            "human_sealed_test": "closed",
        },
        "acceptance": {
            "step": PHASE_TARGETS["phase_c"],
            "interim_uniform_sanity_step": INTERIM_UNIFORM_SANITY_STEP,
            "interim_uniform_sanity_scope": "phase_a broad-data transfer only",
            "dev_game_id_modulus": 5,
            "dev_remainders": [1, 2, 3],
            "historical_regression_guard": True,
            "actor_replay_sid0_sid1": "consumed_do_not_reuse",
        },
        "run_root": str(run_root),
        "cache_root": str(cache_root),
        "source_cache_root": str(source_cache_root),
        "source_fingerprints": {
            "orchestrator_sha256": file_sha256(Path(__file__).resolve()),
            "cache_builder_sha256": file_sha256(
                REPO_ROOT / "scripts/build_oracle_critic_curriculum_cache.py"
            ),
            "curriculum_module_sha256": file_sha256(
                REPO_ROOT / "mortal/research/oracle_critic_curriculum.py"
            ),
            "case_file_sha256": file_sha256(CASE_FILE),
            "convergence_case_file_sha256": file_sha256(CONVERGENCE_CASE_FILE),
            "lr_scheduler_sha256": file_sha256(
                REPO_ROOT / "mortal/core/lr_scheduler.py"
            ),
            "convergence_controller_sha256": file_sha256(
                REPO_ROOT / "mortal/supervised/convergence.py"
            ),
            "oracle_trainer_sha256": file_sha256(
                REPO_ROOT / "mortal/online/pretrain_oracle_critic.py"
            ),
            "search_runner_sha256": file_sha256(
                REPO_ROOT / "scripts/run_oracle_critic_search.py"
            ),
        },
    }


def write_runtime_files(
    *,
    run_root: Path,
    cache_root: Path,
    source_cache_root: Path,
    python_exe: Path,
    hard_max_steps: int,
) -> tuple[Path, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    design_path = run_root / "curriculum_design.json"
    design = curriculum_design(
        run_root=run_root,
        cache_root=cache_root,
        source_cache_root=source_cache_root,
        hard_max_steps=hard_max_steps,
    )
    if design_path.is_file():
        saved = json.loads(design_path.read_text(encoding="utf-8"))
        if saved != design:
            raise RuntimeError("formal curriculum design changed; use a new run root")
    else:
        atomic_write_json(design_path, design)

    spec_path = run_root / "apex_supervisor_spec.json"
    spec = {
        "format": "oracle_critic_apex_supervisor_spec_v1",
        "repo_root": str(REPO_ROOT.resolve()),
        "search_root": str(run_root.resolve()),
        "python_executable": str(python_exe.resolve()),
        "pause_file": str((run_root / "apex_pause.request").resolve()),
        "status_file": str((run_root / "apex_supervisor_status.json").resolve()),
        "log_file": str((run_root / "apex_supervisor.log").resolve()),
        "runner_arguments": [
            "-u",
            "scripts/run_oracle_critic_curriculum.py",
            "--run-root",
            str(run_root.resolve()),
            "--cache-root",
            str(cache_root.resolve()),
            "--source-cache-root",
            str(source_cache_root.resolve()),
            "--python-exe",
            str(python_exe.resolve()),
            "--hard-max-steps",
            str(int(hard_max_steps)),
        ],
    }
    if spec_path.is_file():
        saved = json.loads(spec_path.read_text(encoding="utf-8"))
        if saved != spec:
            raise RuntimeError("Apex supervisor spec changed; use a new run root")
    else:
        atomic_write_json(spec_path, spec)
    return design_path, spec_path


def validate_static_inputs(
    *, source_cache_root: Path, python_exe: Path, hard_max_steps: int
) -> None:
    required_files = (
        python_exe,
        BASE_CONFIG,
        CASE_FILE,
        CONVERGENCE_CASE_FILE,
        AUDITED_UNIFORM_CHECKPOINT,
        source_cache_root / "manifest.json",
        REPO_ROOT / "scripts/build_oracle_critic_curriculum_cache.py",
        REPO_ROOT / "scripts/run_oracle_critic_search.py",
        REPO_ROOT / "scripts/evaluate_oracle_critic_checkpoints.py",
        REPO_ROOT / "scripts/supervise_oracle_critic_around_apex.ps1",
    )
    missing = [str(source) for source in required_files if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"formal curriculum inputs are missing: {missing}")
    if not (RUNTIME_OVERLAY / "libriichi/__init__.py").is_file():
        raise FileNotFoundError(f"runtime overlay is incomplete: {RUNTIME_OVERLAY}")
    if hard_max_steps <= SCHEDULER_HORIZON_STEPS:
        raise ValueError(
            f"hard max must exceed convergence core {SCHEDULER_HORIZON_STEPS}"
        )


def cache_indexes(cache_root: Path) -> dict[str, Path]:
    return {
        "phase_a": cache_root / "phase_a_train_index.pth",
        "phase_b": cache_root / "phase_b_train_index.pth",
        "phase_c": cache_root / "phase_c_train_index.pth",
        "old_regression": cache_root / "old_regression_eval_index.pth",
        "dev": cache_root / "dev_index.pth",
        "test": cache_root / "test_index.pth",
    }


def run_cache_builder(
    *, cache_root: Path, source_cache_root: Path, python_exe: Path
) -> int:
    return run_child(
        [
            "scripts/build_oracle_critic_curriculum_cache.py",
            "--source-cache-root",
            str(source_cache_root),
            "--base-config",
            str(BASE_CONFIG),
            "--runtime-overlay",
            str(RUNTIME_OVERLAY),
            "--output-dir",
            str(cache_root),
            "--files-per-chunk",
            "16",
            "--old-regression-eval-chunks",
            "64",
            "--seed",
            "20260416",
        ],
        python_exe=python_exe,
    )


def phase_search_dir(run_root: Path, phase: str) -> Path:
    return run_root / "phases" / phase


def phase_case_dir(run_root: Path, phase: str) -> Path:
    return phase_search_dir(run_root, phase) / CASE_NAME


def phase_config(run_root: Path, phase: str) -> Path:
    return phase_case_dir(run_root, phase) / "config.toml"


def phase_checkpoint(run_root: Path, phase: str, steps: int) -> Path:
    return phase_case_dir(run_root, phase) / "checkpoints" / f"step_{steps:06d}.pth"


def search_arguments(
    *,
    run_root: Path,
    phase: str,
    target_steps: int,
    train_index: Path,
    dev_index: Path,
    test_index: Path,
    convergence: bool,
    prepare_only: bool,
) -> list[str]:
    arguments = [
        "scripts/run_oracle_critic_search.py",
        "--base-config",
        str(BASE_CONFIG),
        "--output-root",
        str(run_root / "phases"),
        "--search-name",
        phase,
        "--runtime-overlay",
        str(RUNTIME_OVERLAY),
        "--case-file",
        str(CONVERGENCE_CASE_FILE if convergence else CASE_FILE),
        "--case",
        CASE_NAME,
        "--stage-steps",
        str(int(target_steps)),
        "--scheduler-horizon-steps",
        str(SCHEDULER_HORIZON_STEPS),
        "--val-every-steps",
        "10000",
        "--val-batches",
        "256",
        "--test-batches",
        "1024",
        "--num-workers",
        "2",
        "--file-batch-size",
        "6",
        "--prefetch-factor",
        "2",
        "--val-num-workers",
        "0",
        "--val-file-batch-size",
        "8",
        "--val-prefetch-factor",
        "5",
        "--dependency-val-every-steps",
        "0",
        "--save-every",
        "10000",
        "--seed",
        "20260416",
        "--split-seed",
        "20260416",
        "--val-game-id-modulus",
        "5",
        "--val-game-id-remainder",
        "0",
        "--selection-game-id-modulus",
        "5",
        "--selection-game-id-remainder",
        "1",
        "--selection-game-id-remainder",
        "2",
        "--selection-game-id-remainder",
        "3",
        "--selection-max-batches",
        "0",
        "--selection-state-fold-count",
        "128",
        "--train-file-index",
        str(train_index),
        "--dev-file-index",
        str(dev_index),
        "--test-file-index",
        str(test_index),
        "--stop-on-error",
    ]
    if prepare_only:
        arguments.append("--prepare-only")
    return arguments


def load_index(source: Path) -> list[str]:
    payload = torch.load(source, weights_only=False, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("file_list")
    if not isinstance(payload, (list, tuple)) or not payload:
        raise ValueError(f"invalid or empty file index: {source}")
    return [str(filename) for filename in payload]


def destination_file_splits(config: dict[str, Any]) -> dict[str, Any]:
    pretrain = config["oracle_critic_pretrain"]
    return split_summary(
        load_index(resolve_path(pretrain["train_file_index"])),
        load_index(resolve_path(pretrain["dev_file_index"])),
        load_index(resolve_path(pretrain["test_file_index"])),
        seed=int(pretrain.get("split_seed", pretrain.get("seed", 20260416))),
    )


def verify_existing_migration(
    destination: Path,
    *,
    minimum_steps: int,
    expected_splits: dict[str, Any],
    destination_phase: str,
) -> None:
    state = torch.load(destination, weights_only=False, map_location="cpu")
    if int(state.get("steps", -1)) < int(minimum_steps):
        raise RuntimeError(f"existing migration regressed below its source step: {destination}")
    if state.get("file_splits") != expected_splits:
        raise RuntimeError(f"existing migration has wrong split: {destination}")
    provenance = state.get("curriculum_provenance", [])
    if not provenance:
        provenance = state.get("init_info", {}).get("curriculum_provenance", [])
    if not provenance or provenance[-1].get("destination_phase") != destination_phase:
        raise RuntimeError(f"existing migration lacks destination provenance: {destination}")


def seed_best_artifacts(source_case_dir: Path, destination_case_dir: Path) -> list[dict[str, Any]]:
    records = []
    destination_checkpoints = destination_case_dir / "checkpoints"
    destination_checkpoints.mkdir(parents=True, exist_ok=True)
    for name in ("best_dev.pth", "best_primary.pth"):
        source = source_case_dir / "checkpoints" / name
        destination = destination_checkpoints / name
        if not source.is_file():
            continue
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        records.append(
            {
                "role": name,
                "source": str(source.resolve()),
                "source_sha256": file_sha256(source),
                "destination": str(destination.resolve()),
                "destination_sha256": file_sha256(destination),
                "provenance": "historical best on the unchanged fixed dev split",
            }
        )
    return records


def migrate_latest_checkpoint(
    *,
    source: Path,
    source_case_dir: Path,
    destination_case_dir: Path,
    source_phase: str,
    destination_phase: str,
    expected_steps: int,
    reset_data_cursor: bool,
) -> Path:
    destination = destination_case_dir / "checkpoints/latest.pth"
    destination_config_path = destination_case_dir / "config.toml"
    destination_config = load_toml_file(destination_config_path)
    splits = destination_file_splits(destination_config)
    if destination.is_file():
        verify_existing_migration(
            destination,
            minimum_steps=expected_steps,
            expected_splits=splits,
            destination_phase=destination_phase,
        )
        return destination
    if pause_requested():
        raise InterruptedError("Apex pause requested before phase migration")
    source_sha256 = file_sha256(source)
    state = torch.load(source, weights_only=False, map_location="cpu")
    if int(state.get("steps", -1)) != int(expected_steps):
        raise RuntimeError(
            f"phase source must be exact step {expected_steps}: {source}"
        )
    protected_before = checkpoint_component_hashes(state)
    migrated = migrate_checkpoint_for_phase(
        state,
        destination_config=destination_config,
        destination_file_splits=splits,
        source_phase=source_phase,
        destination_phase=destination_phase,
        source_checkpoint=str(source.resolve()),
        source_checkpoint_sha256=source_sha256,
        reset_data_cursor=reset_data_cursor,
    )
    atomic_torch_save(destination, migrated)
    reloaded = torch.load(destination, weights_only=False, map_location="cpu")
    protected_after = checkpoint_component_hashes(reloaded)
    if protected_before != protected_after:
        raise RuntimeError("serialized phase migration changed protected state")
    best_records = seed_best_artifacts(source_case_dir, destination_case_dir)
    transition = {
        "format": "oracle_critic_curriculum_phase_transition_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase": source_phase,
        "destination_phase": destination_phase,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "destination_checkpoint": str(destination.resolve()),
        "destination_checkpoint_sha256": file_sha256(destination),
        "steps": int(expected_steps),
        "reset_data_cursor": bool(reset_data_cursor),
        "protected_component_sha256": protected_after,
        "destination_file_splits": splits,
        "seeded_best_artifacts": best_records,
    }
    atomic_write_json(destination_case_dir / "phase_transition_manifest.json", transition)
    return destination


def run_phase(
    *,
    run_root: Path,
    phase: str,
    target_steps: int,
    indexes: dict[str, Path],
    python_exe: Path,
    convergence: bool = False,
    train_index_name: str | None = None,
    source_phase: str | None = None,
    source_checkpoint: Path | None = None,
    source_steps: int | None = None,
    reset_data_cursor: bool = True,
) -> int:
    resolved_train_index_name = train_index_name or (
        "phase_c" if convergence else phase
    )
    if resolved_train_index_name not in indexes:
        raise KeyError(f"unknown train index {resolved_train_index_name!r}")
    prepare_arguments = search_arguments(
        run_root=run_root,
        phase=phase,
        target_steps=target_steps,
        train_index=indexes[resolved_train_index_name],
        dev_index=indexes["dev"],
        test_index=indexes["test"],
        convergence=convergence,
        prepare_only=True,
    )
    return_code = run_child(prepare_arguments, python_exe=python_exe)
    if return_code != 0:
        return return_code
    if source_checkpoint is not None:
        if source_steps is None:
            raise ValueError("source_steps is required with source_checkpoint")
        try:
            migrate_latest_checkpoint(
                source=source_checkpoint,
                source_case_dir=source_checkpoint.parent.parent,
                destination_case_dir=phase_case_dir(run_root, phase),
                source_phase=str(source_phase),
                destination_phase=phase,
                expected_steps=int(source_steps),
                reset_data_cursor=reset_data_cursor,
            )
        except InterruptedError:
            return EXTERNAL_PAUSE_EXIT_CODE
    return run_child(
        search_arguments(
            run_root=run_root,
            phase=phase,
            target_steps=target_steps,
            train_index=indexes[resolved_train_index_name],
            dev_index=indexes["dev"],
            test_index=indexes["test"],
            convergence=convergence,
            prepare_only=False,
        ),
        python_exe=python_exe,
    )


def evaluation_arguments(
    *,
    config: Path,
    output: Path,
    checkpoints: list[tuple[str, Path]],
    split_override_reason: str = "",
) -> list[str]:
    arguments = [
        "scripts/evaluate_oracle_critic_checkpoints.py",
        "--config",
        str(config),
        "--split",
        "dev",
        "--input-mode",
        "true",
        "--eval-state-fold-count",
        "128",
        "--game-id-modulus",
        "5",
        "--game-id-remainder",
        "1",
        "--game-id-remainder",
        "2",
        "--game-id-remainder",
        "3",
        "--max-batches",
        "0",
        "--output",
        str(output),
        "--no-print-result",
    ]
    for name, checkpoint in checkpoints:
        arguments.extend(["--checkpoint", f"{name}={checkpoint}"])
    if split_override_reason:
        arguments.extend(["--eval-split-override-reason", split_override_reason])
    return arguments


def run_guard_evaluations(
    *,
    evaluation_dir: Path,
    config: Path,
    checkpoints: list[tuple[str, Path]],
    indexes: dict[str, Path],
    python_exe: Path,
    historical_reason: str,
) -> int:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    dev_output = evaluation_dir / "paired_dev_remainders123.json"
    if not dev_output.is_file():
        return_code = run_restartable_evaluation(
            evaluation_arguments(
                config=config,
                output=dev_output,
                checkpoints=checkpoints,
            ),
            python_exe=python_exe,
        )
        if return_code != 0:
            return return_code

    historical_config = evaluation_dir / "historical_regression_config.toml"
    if not historical_config.is_file():
        config_payload = load_toml_file(config)
        pretrain = config_payload["oracle_critic_pretrain"]
        pretrain["dev_file_index"] = str(indexes["old_regression"].resolve())
        pretrain["max_val_files"] = 0
        write_toml_file(historical_config, config_payload)
    historical_output = evaluation_dir / "paired_historical_regression.json"
    if not historical_output.is_file():
        return run_restartable_evaluation(
            evaluation_arguments(
                config=historical_config,
                output=historical_output,
                checkpoints=checkpoints,
                split_override_reason=historical_reason,
            ),
            python_exe=python_exe,
        )
    return 0


def run_interim_uniform_sanity_evaluations(
    *, run_root: Path, indexes: dict[str, Path], python_exe: Path
) -> int:
    checkpoint = phase_checkpoint(
        run_root, "phase_a_probe", INTERIM_UNIFORM_SANITY_STEP
    )
    return run_guard_evaluations(
        evaluation_dir=(
            run_root / f"uniform_sanity_step_{INTERIM_UNIFORM_SANITY_STEP:07d}"
        ),
        config=phase_config(run_root, "phase_a_probe"),
        checkpoints=[
            ("curriculum_phase_a_sf200_wd0", checkpoint),
            ("uniform_sf200_wd0", AUDITED_UNIFORM_CHECKPOINT),
        ],
        indexes=indexes,
        python_exe=python_exe,
        historical_reason=(
            "predeclared 202212-202311 historical regression guard for the "
            "190k broad-data transfer sanity check"
        ),
    )


def run_course_gate_evaluations(
    *, run_root: Path, indexes: dict[str, Path], python_exe: Path
) -> int:
    checkpoint = phase_checkpoint(run_root, "phase_c", PHASE_TARGETS["phase_c"])
    return run_guard_evaluations(
        evaluation_dir=(run_root / f"course_gate_step_{PHASE_TARGETS['phase_c']:07d}"),
        config=phase_config(run_root, "phase_c"),
        checkpoints=[("curriculum_sf200_wd0", checkpoint)],
        indexes=indexes,
        python_exe=python_exe,
        historical_reason=(
            "predeclared 202212-202311 historical regression guard for the "
            "completed SL-derived Oracle critic curriculum"
        ),
    )


def final_candidate_checkpoints(run_root: Path) -> list[tuple[str, Path]]:
    checkpoint_dir = phase_case_dir(run_root, "phase_c_convergence") / "checkpoints"
    candidates = []
    seen_sha256 = set()
    for name, filename in (
        ("latest", "latest.pth"),
        ("best_primary", "best_primary.pth"),
        ("best_dev", "best_dev.pth"),
    ):
        checkpoint = checkpoint_dir / filename
        if not checkpoint.is_file():
            continue
        sha256 = file_sha256(checkpoint)
        if sha256 in seen_sha256:
            continue
        seen_sha256.add(sha256)
        candidates.append((name, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"no final candidate checkpoints under {checkpoint_dir}")
    return candidates


def run_final_candidate_evaluations(
    *, run_root: Path, indexes: dict[str, Path], python_exe: Path
) -> int:
    selection_dir = run_root / "final_candidate_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    candidates = final_candidate_checkpoints(run_root)
    config = phase_config(run_root, "phase_c_convergence")
    dev_output = selection_dir / "paired_dev_remainders123.json"
    if not dev_output.is_file():
        return_code = run_restartable_evaluation(
            evaluation_arguments(
                config=config,
                output=dev_output,
                checkpoints=candidates,
            ),
            python_exe=python_exe,
        )
        if return_code != 0:
            return return_code

    historical_config = selection_dir / "historical_regression_config.toml"
    if not historical_config.is_file():
        config_payload = load_toml_file(config)
        pretrain = config_payload["oracle_critic_pretrain"]
        pretrain["dev_file_index"] = str(indexes["old_regression"].resolve())
        pretrain["max_val_files"] = 0
        write_toml_file(historical_config, config_payload)
    historical_output = selection_dir / "paired_historical_regression.json"
    if not historical_output.is_file():
        return run_restartable_evaluation(
            evaluation_arguments(
                config=historical_config,
                output=historical_output,
                checkpoints=candidates,
                split_override_reason=(
                    "predeclared 202212-202311 historical regression guard for "
                    "final Oracle critic checkpoint selection"
                ),
            ),
            python_exe=python_exe,
        )
    return 0


def write_completion(run_root: Path) -> None:
    latest = phase_case_dir(run_root, "phase_c_convergence") / "checkpoints/latest.pth"
    state = torch.load(latest, weights_only=False, map_location="cpu")
    convergence = state.get("convergence_state") or {}
    atomic_write_json(
        run_root / "completion.json",
        {
            "format": "oracle_critic_formal_curriculum_completion_v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(latest.resolve()),
            "checkpoint_sha256": file_sha256(latest),
            "steps": int(state["steps"]),
            "converged": bool(convergence.get("converged", False)),
            "convergence_state": convergence,
            "candidate_checkpoints": [
                {
                    "name": name,
                    "path": str(checkpoint.resolve()),
                    "sha256": file_sha256(checkpoint),
                }
                for name, checkpoint in final_candidate_checkpoints(run_root)
            ],
            "candidate_selection": str(
                (run_root / "final_candidate_selection").resolve()
            ),
            "human_sealed_test": "closed",
            "actor_replay_sid0_sid1": "not_reused",
        },
    )


def main() -> int:
    args = parse_args()
    run_root = resolve_path(args.run_root)
    cache_root = resolve_path(args.cache_root)
    source_cache_root = resolve_path(args.source_cache_root)
    python_exe = resolve_path(args.python_exe)
    validate_static_inputs(
        source_cache_root=source_cache_root,
        python_exe=python_exe,
        hard_max_steps=args.hard_max_steps,
    )
    design_path, spec_path = write_runtime_files(
        run_root=run_root,
        cache_root=cache_root,
        source_cache_root=source_cache_root,
        python_exe=python_exe,
        hard_max_steps=args.hard_max_steps,
    )
    if args.validate_only or args.emit_supervisor_spec:
        print(
            json.dumps(
                {
                    "validated": True,
                    "design": str(design_path),
                    "supervisor_spec": str(spec_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if pause_requested():
        return EXTERNAL_PAUSE_EXIT_CODE

    return_code = run_cache_builder(
        cache_root=cache_root,
        source_cache_root=source_cache_root,
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code
    cache_manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if not bool(cache_manifest.get("complete", False)):
        raise RuntimeError("curriculum cache builder returned without a complete cache")
    indexes = cache_indexes(cache_root)
    missing_indexes = [str(source) for source in indexes.values() if not source.is_file()]
    if missing_indexes:
        raise FileNotFoundError(f"curriculum indexes are missing: {missing_indexes}")

    print(
        "[phase_a_probe] fresh SL initialization -> interim uniform sanity "
        f"step {INTERIM_UNIFORM_SANITY_STEP}",
        flush=True,
    )
    return_code = run_phase(
        run_root=run_root,
        phase="phase_a_probe",
        target_steps=INTERIM_UNIFORM_SANITY_STEP,
        indexes=indexes,
        python_exe=python_exe,
        train_index_name="phase_a",
    )
    if return_code != 0:
        return return_code

    print("[uniform_sanity] paired dev and historical regression guard", flush=True)
    return_code = run_interim_uniform_sanity_evaluations(
        run_root=run_root,
        indexes=indexes,
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code

    print(
        f"[phase_a] preserve phase-A cursor {INTERIM_UNIFORM_SANITY_STEP} "
        f"-> {PHASE_TARGETS['phase_a']}",
        flush=True,
    )
    return_code = run_phase(
        run_root=run_root,
        phase="phase_a",
        target_steps=PHASE_TARGETS["phase_a"],
        indexes=indexes,
        python_exe=python_exe,
        train_index_name="phase_a",
        source_phase="phase_a_probe",
        source_checkpoint=phase_checkpoint(
            run_root, "phase_a_probe", INTERIM_UNIFORM_SANITY_STEP
        ),
        source_steps=INTERIM_UNIFORM_SANITY_STEP,
        reset_data_cursor=False,
    )
    if return_code != 0:
        return return_code

    print(
        f"[phase_b] migrate exact {PHASE_TARGETS['phase_a']} "
        f"-> {PHASE_TARGETS['phase_b']}",
        flush=True,
    )
    return_code = run_phase(
        run_root=run_root,
        phase="phase_b",
        target_steps=PHASE_TARGETS["phase_b"],
        indexes=indexes,
        python_exe=python_exe,
        source_phase="phase_a",
        source_checkpoint=phase_checkpoint(
            run_root, "phase_a", PHASE_TARGETS["phase_a"]
        ),
        source_steps=PHASE_TARGETS["phase_a"],
        reset_data_cursor=True,
    )
    if return_code != 0:
        return return_code

    print(
        f"[phase_c] migrate exact {PHASE_TARGETS['phase_b']} "
        f"-> course gate {PHASE_TARGETS['phase_c']}",
        flush=True,
    )
    return_code = run_phase(
        run_root=run_root,
        phase="phase_c",
        target_steps=PHASE_TARGETS["phase_c"],
        indexes=indexes,
        python_exe=python_exe,
        source_phase="phase_b",
        source_checkpoint=phase_checkpoint(
            run_root, "phase_b", PHASE_TARGETS["phase_b"]
        ),
        source_steps=PHASE_TARGETS["phase_b"],
        reset_data_cursor=True,
    )
    if return_code != 0:
        return return_code

    print("[course_gate] fixed dev and historical regression guard", flush=True)
    return_code = run_course_gate_evaluations(
        run_root=run_root,
        indexes=indexes,
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code

    print(
        f"[phase_c_convergence] exact {PHASE_TARGETS['phase_c']} "
        "-> dynamic convergence or hard cap",
        flush=True,
    )
    return_code = run_phase(
        run_root=run_root,
        phase="phase_c_convergence",
        target_steps=args.hard_max_steps,
        indexes=indexes,
        python_exe=python_exe,
        convergence=True,
        source_phase="phase_c",
        source_checkpoint=phase_checkpoint(
            run_root, "phase_c", PHASE_TARGETS["phase_c"]
        ),
        source_steps=PHASE_TARGETS["phase_c"],
        reset_data_cursor=False,
    )
    if return_code != 0:
        return return_code
    print("[final_selection] latest/best_primary/best_dev paired guards", flush=True)
    return_code = run_final_candidate_evaluations(
        run_root=run_root,
        indexes=indexes,
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code
    write_completion(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

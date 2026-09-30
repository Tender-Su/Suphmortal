from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from mortal._repo import MORTAL_ROOT, REPO_ROOT
from mortal.core.toml_utils import write_toml_file
from mortal.online.online_machine_modes import (
    _cfg_section,
    _deep_merge_dict,
    _ensure_runtime_root_layout,
    build_independent_arm_config,
    load_resolved_base_config,
)


CDE_ARMS = frozenset({"C", "D", "E"})
TOWER_ARCHES = frozenset({"single_tower", "dual_tower"})
DEFAULT_BATCH_SIZE_BY_TOWER = {
    "single_tower": 512,
    "dual_tower": 192,
}


def _normalize_arm(value: str) -> str:
    arm = str(value or "").strip().upper()
    if arm not in CDE_ARMS:
        raise ValueError(f"arm must be one of {sorted(CDE_ARMS)}, got {value!r}")
    return arm


def _normalize_tower(value: str) -> str:
    tower = str(value or "").strip().lower()
    if tower in {"single", "single_tower"}:
        return "single_tower"
    if tower in {"dual", "two_tower", "dual_tower"}:
        return "dual_tower"
    raise ValueError(f"tower must be one of {sorted(TOWER_ARCHES)}, got {value!r}")


def default_run_name(*, tower: str, arm: str, max_steps: int) -> str:
    return f"{_normalize_tower(tower)}_{_normalize_arm(arm)}_s{int(max_steps)}"


def _abs_path(value: str | Path) -> str:
    return str(Path(value).resolve())


def stage_resume_state_file(
    resume_state_file: str | Path,
    target_state_file: str | Path,
) -> Path:
    source = Path(resume_state_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {source}")

    target = Path(target_state_file).resolve()
    if target.exists():
        try:
            if source.samefile(target):
                return target
        except OSError:
            pass
        raise FileExistsError(
            f"target control.state_file already exists; refusing to overwrite: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def load_online_checkpoint_steps(state_file: str | Path) -> int:
    checkpoint = Path(state_file).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")

    import torch

    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if not isinstance(state, dict) or "steps" not in state:
        raise ValueError(f"resume checkpoint is missing online steps: {checkpoint}")
    steps = int(state.get("steps", 0) or 0)
    if steps < 0:
        raise ValueError(f"resume checkpoint has negative steps: {checkpoint}")
    return steps


def _drop_oracle_pretrain_paths(config_dict: dict[str, Any]) -> None:
    value_cfg = _cfg_section(config_dict, "value")
    for key in ("oracle_critic_state_file", "critic_state_file", "pretrained_state_file"):
        value_cfg.pop(key, None)

    pretrain_cfg = _cfg_section(config_dict, "oracle_critic_pretrain")
    for key in ("best_state_file", "state_file", "init_state_file"):
        pretrain_cfg.pop(key, None)


def _apply_oracle_critic_common(
    config_dict: dict[str, Any],
    *,
    tower: str,
    max_steps: int,
    monitor_every: int,
    batch_size: int | None,
    value_weight: float,
    policy_clip_ratio: float | None = None,
    actor_lr_scale: float = 1.0,
    policy_head_lr_scale: float = 1.0,
    policy_update_interval: int = 1,
    policy_update_phase: int = 0,
    entropy_floor: float = 0.0,
    entropy_adjust_rate: float = 0.0,
    entropy_floor_start_step: int = 0,
    vtrace_target_rho_clip: float = 0.0,
    vtrace_target_c_clip: float = 0.0,
    vtrace_mode: str = "auto",
    vtrace_min_version_gap: int = 2,
) -> None:
    control_cfg = _cfg_section(config_dict, "control")
    control_cfg["online"] = True
    control_cfg["save_every"] = int(monitor_every)
    control_cfg["test_every"] = int(max_steps)
    if batch_size is not None:
        control_cfg["batch_size"] = int(batch_size)

    policy_cfg = _cfg_section(config_dict, "policy")
    _deep_merge_dict(
        policy_cfg,
        {
            "online_action_scope": "all",
            "gae_enabled": True,
            "clip_ratio": 0.2,
            "logit_thres": 0.0,
            "importance_rho_clip": 0.0,
            "importance_c_clip": 0.0,
            "vtrace_rho_clip": 0.0,
            "vtrace_c_clip": 0.0,
            "vtrace_target_rho_clip": float(vtrace_target_rho_clip),
            "vtrace_target_c_clip": float(vtrace_target_c_clip),
            "entropy_floor": float(entropy_floor),
            "entropy_floor_start_step": int(entropy_floor_start_step),
            "entropy_target": float(entropy_floor),
            "entropy_adjust_rate": float(entropy_adjust_rate),
            "actor_lr_scale": float(actor_lr_scale),
            "policy_head_lr_scale": float(policy_head_lr_scale),
            "update_interval": int(policy_update_interval),
            "update_phase": int(policy_update_phase),
        },
    )
    if policy_clip_ratio is not None:
        policy_cfg["clip_ratio"] = float(policy_clip_ratio)

    value_cfg = _cfg_section(config_dict, "value")
    _deep_merge_dict(
        value_cfg,
        {
            "enabled": True,
            "weight": float(value_weight),
            "zero_sum_weight": 0.01,
            "target_mode": "auto",
            "reward_source": "score_rank",
            "oracle_critic": True,
            "oracle_critic_arch": tower,
            "independent_actor_lr_clock": True,
        },
    )

    oracle_cfg = _cfg_section(config_dict, "oracle_guiding")
    _deep_merge_dict(
        oracle_cfg,
        {
            "actor_enabled": False,
            "actor_source": "zero",
            "schedule": "linear",
            "gamma_start": 0.0,
            "gamma_end": 0.0,
            "hold_steps": 0,
            "decay_steps": 0,
        },
    )

    aux_cfg = _cfg_section(config_dict, "aux")
    _deep_merge_dict(
        aux_cfg,
        {
            "next_rank_weight": 0.0,
            "opponent_state_weight": 0.0,
            "danger_enabled": False,
            "danger_weight": 0.0,
            "tile_efficiency_weight": 0.0,
            "furo_regret_weight": 0.0,
            "hand_value_regret_weight": 0.0,
        },
    )

    _deep_merge_dict(_cfg_section(config_dict, "expected_reward"), {"enabled": False})
    _deep_merge_dict(_cfg_section(config_dict, "search"), {"enabled": False})
    _deep_merge_dict(_cfg_section(config_dict, "search_distill"), {"enabled": False})
    _deep_merge_dict(_cfg_section(config_dict, "oracle_dependency_eval"), {"enabled": False})

    online_cfg = _cfg_section(config_dict, "online")
    online_cfg["stop_at_max_steps"] = True
    _deep_merge_dict(
        _cfg_section(online_cfg, "importance_sampling"),
        {
            "enabled": True,
            "max_policy_versions": 16,
            "drop_untracked_samples": False,
            "vtrace_mode": str(vtrace_mode),
            "vtrace_min_version_gap": int(vtrace_min_version_gap),
        },
    )

    scheduler_cfg = _cfg_section(_cfg_section(config_dict, "optim"), "scheduler")
    scheduler_cfg["max_steps"] = int(max_steps)
    scheduler_cfg["warm_up_steps"] = max(50, min(5000, int(max_steps) // 20))

    test_play_cfg = _cfg_section(config_dict, "test_play")
    test_play_cfg["enable"] = False
    test_play_cfg["initial_enable"] = False
    test_play_cfg["initial_games"] = 600


def build_oracle_cde_config(
    base_config: dict[str, Any],
    *,
    runtime_root: str | Path,
    tower: str,
    arm: str,
    max_steps: int,
    oracle_critic_state_file: str | Path | None = None,
    critic_warmup_steps: int = 0,
    remote_host: str = "127.0.0.1",
    remote_port: int = 5000,
    opponent_pool_preset: str = "validation",
    repro_seed: int | None = None,
    train_key: int | None = None,
    train_seed_start: int | None = None,
    allow_cudnn_benchmark: bool = False,
    monitor_every: int = 500,
    batch_size: int | None = None,
    value_weight: float = 0.05,
    resume_steps: int = 0,
    policy_clip_ratio: float | None = None,
    actor_lr_scale: float = 1.0,
    policy_head_lr_scale: float = 1.0,
    policy_update_interval: int = 1,
    policy_update_phase: int = 0,
    entropy_floor: float = 0.0,
    entropy_adjust_rate: float = 0.0,
    entropy_floor_start_step: int = 0,
    vtrace_target_rho_clip: float = 0.0,
    vtrace_target_c_clip: float = 0.0,
    vtrace_mode: str = "auto",
    vtrace_min_version_gap: int = 2,
) -> dict[str, Any]:
    arm = _normalize_arm(arm)
    tower = _normalize_tower(tower)
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if int(monitor_every) <= 0 or int(monitor_every) % 500 != 0:
        raise ValueError("monitor_every must keep the 500-step monitoring grid")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE_BY_TOWER[tower]
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    value_weight = float(value_weight)
    if value_weight < 0:
        raise ValueError("value_weight must be non-negative")
    if policy_clip_ratio is not None:
        policy_clip_ratio = float(policy_clip_ratio)
        if policy_clip_ratio <= 0:
            raise ValueError("policy_clip_ratio must be positive")
    actor_lr_scale = float(actor_lr_scale)
    if actor_lr_scale < 0:
        raise ValueError("actor_lr_scale must be non-negative")
    policy_head_lr_scale = float(policy_head_lr_scale)
    if policy_head_lr_scale < 0:
        raise ValueError("policy_head_lr_scale must be non-negative")
    policy_update_interval = int(policy_update_interval)
    if policy_update_interval <= 0:
        raise ValueError("policy_update_interval must be positive")
    policy_update_phase = int(policy_update_phase)
    if policy_update_phase < 0:
        raise ValueError("policy_update_phase must be non-negative")
    entropy_floor = float(entropy_floor)
    entropy_adjust_rate = float(entropy_adjust_rate)
    entropy_floor_start_step = max(int(entropy_floor_start_step), 0)
    if entropy_floor < 0:
        raise ValueError("entropy_floor must be non-negative")
    if entropy_adjust_rate < 0:
        raise ValueError("entropy_adjust_rate must be non-negative")
    vtrace_target_rho_clip = float(vtrace_target_rho_clip)
    vtrace_target_c_clip = float(vtrace_target_c_clip)
    if vtrace_target_rho_clip < 0 or vtrace_target_c_clip < 0:
        raise ValueError("V-trace target clips must be non-negative")
    vtrace_mode = str(vtrace_mode or "auto").strip().lower()
    if vtrace_mode not in {"auto", "always", "disabled"}:
        raise ValueError("vtrace_mode must be one of: auto, always, disabled")
    vtrace_min_version_gap = max(int(vtrace_min_version_gap), 0)

    runtime_root = Path(runtime_root).resolve()
    config_dict = build_independent_arm_config(
        base_config,
        runtime_root=runtime_root,
        remote_host=remote_host,
        remote_port=remote_port,
        experiment_profile="default",
        opponent_pool_preset=opponent_pool_preset,
        repro_seed=repro_seed,
        train_key=train_key,
        train_seed_start=train_seed_start,
    )
    if allow_cudnn_benchmark and (
        repro_seed is not None or train_key is not None or train_seed_start is not None
    ):
        _cfg_section(config_dict, "repro")["allow_cudnn_benchmark"] = True
    _apply_oracle_critic_common(
        config_dict,
        tower=tower,
        max_steps=max_steps,
        monitor_every=int(monitor_every),
        batch_size=batch_size,
        value_weight=value_weight,
        policy_clip_ratio=policy_clip_ratio,
        actor_lr_scale=actor_lr_scale,
        policy_head_lr_scale=policy_head_lr_scale,
        policy_update_interval=policy_update_interval,
        policy_update_phase=policy_update_phase,
        entropy_floor=entropy_floor,
        entropy_adjust_rate=entropy_adjust_rate,
        entropy_floor_start_step=entropy_floor_start_step,
        vtrace_target_rho_clip=vtrace_target_rho_clip,
        vtrace_target_c_clip=vtrace_target_c_clip,
        vtrace_mode=vtrace_mode,
        vtrace_min_version_gap=vtrace_min_version_gap,
    )

    value_cfg = _cfg_section(config_dict, "value")
    warmup_steps = int(critic_warmup_steps or 0)
    if warmup_steps < 0:
        raise ValueError("critic_warmup_steps must be non-negative")
    resume_steps = int(resume_steps or 0)
    if resume_steps < 0:
        raise ValueError("resume_steps must be non-negative")

    if arm == "C":
        _drop_oracle_pretrain_paths(config_dict)
        if warmup_steps <= 0 and resume_steps <= 0:
            raise ValueError(
                "arm C requires positive critic_warmup_steps unless resuming "
                "from an already-warmed checkpoint"
            )
        effective_warmup_steps = resume_steps + warmup_steps
        value_cfg["critic_warmup_steps"] = effective_warmup_steps
    elif arm == "D":
        if not oracle_critic_state_file:
            raise ValueError(f"arm {arm} requires --oracle-critic-state-file")
        checkpoint = Path(oracle_critic_state_file).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"oracle critic checkpoint does not exist: {checkpoint}")
        _drop_oracle_pretrain_paths(config_dict)
        value_cfg["oracle_critic_state_file"] = str(checkpoint)
        effective_warmup_steps = 0
        value_cfg["critic_warmup_steps"] = effective_warmup_steps
    else:
        if not oracle_critic_state_file:
            raise ValueError(f"arm {arm} requires --oracle-critic-state-file")
        checkpoint = Path(oracle_critic_state_file).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"oracle critic checkpoint does not exist: {checkpoint}")
        _drop_oracle_pretrain_paths(config_dict)
        value_cfg["oracle_critic_state_file"] = str(checkpoint)
        if arm == "E" and warmup_steps <= 0 and resume_steps <= 0:
            raise ValueError(
                "arm E requires positive critic_warmup_steps unless resuming "
                "from an already-warmed checkpoint"
            )
        effective_warmup_steps = resume_steps + warmup_steps
        value_cfg["critic_warmup_steps"] = effective_warmup_steps

    config_dict["oracle_cde_experiment"] = {
        "arm": arm,
        "tower": tower,
        "max_steps": max_steps,
        "critic_warmup_steps": int(effective_warmup_steps),
        "critic_warmup_requested_steps": warmup_steps if arm in {"C", "E"} else 0,
        "critic_warmup_resume_steps": resume_steps if arm in {"C", "E"} else 0,
        "monitor_every": int(monitor_every),
        "batch_size": int(_cfg_section(config_dict, "control")["batch_size"]),
        "value_weight": float(value_cfg.get("weight", 0.0) or 0.0),
        "policy_clip_ratio": float(_cfg_section(config_dict, "policy")["clip_ratio"]),
        "actor_lr_scale": float(_cfg_section(config_dict, "policy").get("actor_lr_scale", 1.0) or 0.0),
        "policy_head_lr_scale": float(
            _cfg_section(config_dict, "policy").get("policy_head_lr_scale", 1.0) or 0.0
        ),
        "policy_update_interval": int(
            _cfg_section(config_dict, "policy").get("update_interval", 1) or 1
        ),
        "policy_update_phase": int(
            _cfg_section(config_dict, "policy").get("update_phase", 0) or 0
        ),
        "entropy_floor": float(_cfg_section(config_dict, "policy").get("entropy_floor", 0.0) or 0.0),
        "entropy_adjust_rate": float(
            _cfg_section(config_dict, "policy").get("entropy_adjust_rate", 0.0) or 0.0
        ),
        "entropy_floor_start_step": int(
            _cfg_section(config_dict, "policy").get("entropy_floor_start_step", 0) or 0
        ),
        "vtrace_target_rho_clip": vtrace_target_rho_clip,
        "vtrace_target_c_clip": vtrace_target_c_clip,
        "vtrace_mode": vtrace_mode,
        "vtrace_min_version_gap": vtrace_min_version_gap,
        "oracle_critic_state_file": value_cfg.get("oracle_critic_state_file", ""),
        "opponent_pool_preset": opponent_pool_preset,
        "remote_host": remote_host,
        "remote_port": int(remote_port),
        "runtime_root": str(runtime_root),
        "allow_cudnn_benchmark": bool(
            _cfg_section(config_dict, "repro").get("allow_cudnn_benchmark", False)
        ),
    }
    _ensure_runtime_root_layout(config_dict, runtime_root)
    return config_dict


def build_manifest(
    *,
    run_name: str,
    base_config_path: str | Path,
    output_config_path: str | Path,
    config_dict: dict[str, Any],
    resume_state_file: str | Path | None = None,
) -> dict[str, Any]:
    cde = config_dict["oracle_cde_experiment"]
    return {
        "run_name": run_name,
        "base_config_path": _abs_path(base_config_path),
        "config_path": _abs_path(output_config_path),
        "runtime_root": cde["runtime_root"],
        "arm": cde["arm"],
        "tower": cde["tower"],
        "max_steps": cde["max_steps"],
        "monitor_every_steps": int(config_dict["control"]["save_every"]),
        "batch_size": int(config_dict["control"]["batch_size"]),
        "value_weight": float(cde["value_weight"]),
        "policy_clip_ratio": float(cde["policy_clip_ratio"]),
        "actor_lr_scale": float(cde["actor_lr_scale"]),
        "policy_head_lr_scale": float(cde["policy_head_lr_scale"]),
        "policy_update_interval": int(cde["policy_update_interval"]),
        "policy_update_phase": int(cde["policy_update_phase"]),
        "entropy_floor": float(cde["entropy_floor"]),
        "entropy_adjust_rate": float(cde["entropy_adjust_rate"]),
        "entropy_floor_start_step": int(cde["entropy_floor_start_step"]),
        "vtrace_target_rho_clip": float(cde["vtrace_target_rho_clip"]),
        "vtrace_target_c_clip": float(cde["vtrace_target_c_clip"]),
        "vtrace_mode": cde["vtrace_mode"],
        "vtrace_min_version_gap": int(cde["vtrace_min_version_gap"]),
        "allow_cudnn_benchmark": bool(cde["allow_cudnn_benchmark"]),
        "test_every_steps": int(config_dict["control"]["test_every"]),
        "critic_warmup_steps": cde["critic_warmup_steps"],
        "critic_warmup_requested_steps": cde["critic_warmup_requested_steps"],
        "critic_warmup_resume_steps": cde["critic_warmup_resume_steps"],
        "oracle_critic_state_file": cde["oracle_critic_state_file"],
        "opponent_pool_preset": cde["opponent_pool_preset"],
        "control_state_file": config_dict["control"]["state_file"],
        "control_best_state_file": config_dict["control"]["best_state_file"],
        "resume_state_file": _abs_path(resume_state_file) if resume_state_file else "",
        "online_init_state_file": config_dict.get("online", {}).get("init_state_file", ""),
        "server_buffer_dir": config_dict["online"]["server"]["buffer_dir"],
        "server_drain_dir": config_dict["online"]["server"]["drain_dir"],
        "recommended_role_args": [
            "-m",
            "mortal.online.online_role_runner",
            "<server|trainer|client>",
            "--config",
            _abs_path(output_config_path),
            "--arm",
            "current_config",
        ],
    }


def write_oracle_cde_config(
    *,
    base_config_path: str | Path,
    output_path: str | Path,
    runtime_root: str | Path,
    run_name: str,
    tower: str,
    arm: str,
    max_steps: int,
    oracle_critic_state_file: str | Path | None = None,
    critic_warmup_steps: int = 0,
    remote_host: str = "127.0.0.1",
    remote_port: int = 5000,
    opponent_pool_preset: str = "validation",
    repro_seed: int | None = None,
    train_key: int | None = None,
    train_seed_start: int | None = None,
    allow_cudnn_benchmark: bool = False,
    monitor_every: int = 500,
    batch_size: int | None = None,
    value_weight: float = 0.05,
    resume_state_file: str | Path | None = None,
    policy_clip_ratio: float | None = None,
    actor_lr_scale: float = 1.0,
    policy_head_lr_scale: float = 1.0,
    policy_update_interval: int = 1,
    policy_update_phase: int = 0,
    entropy_floor: float = 0.0,
    entropy_adjust_rate: float = 0.0,
    entropy_floor_start_step: int = 0,
    vtrace_target_rho_clip: float = 0.0,
    vtrace_target_c_clip: float = 0.0,
    vtrace_mode: str = "auto",
    vtrace_min_version_gap: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_config_path = Path(base_config_path).resolve()
    output_path = Path(output_path).resolve()
    base_config = load_resolved_base_config(base_config_path)
    resume_steps = 0
    if resume_state_file and _normalize_arm(arm) in {"C", "E"}:
        resume_steps = load_online_checkpoint_steps(resume_state_file)
    config_dict = build_oracle_cde_config(
        base_config,
        runtime_root=runtime_root,
        tower=tower,
        arm=arm,
        max_steps=max_steps,
        oracle_critic_state_file=oracle_critic_state_file,
        critic_warmup_steps=critic_warmup_steps,
        remote_host=remote_host,
        remote_port=remote_port,
        opponent_pool_preset=opponent_pool_preset,
        repro_seed=repro_seed,
        train_key=train_key,
        train_seed_start=train_seed_start,
        allow_cudnn_benchmark=allow_cudnn_benchmark,
        monitor_every=monitor_every,
        batch_size=batch_size,
        value_weight=value_weight,
        resume_steps=resume_steps,
        policy_clip_ratio=policy_clip_ratio,
        actor_lr_scale=actor_lr_scale,
        policy_head_lr_scale=policy_head_lr_scale,
        policy_update_interval=policy_update_interval,
        policy_update_phase=policy_update_phase,
        entropy_floor=entropy_floor,
        entropy_adjust_rate=entropy_adjust_rate,
        entropy_floor_start_step=entropy_floor_start_step,
        vtrace_target_rho_clip=vtrace_target_rho_clip,
        vtrace_target_c_clip=vtrace_target_c_clip,
        vtrace_mode=vtrace_mode,
        vtrace_min_version_gap=vtrace_min_version_gap,
    )
    if resume_state_file:
        stage_resume_state_file(resume_state_file, config_dict["control"]["state_file"])
    write_toml_file(output_path, config_dict)
    manifest = build_manifest(
        run_name=run_name,
        base_config_path=base_config_path,
        output_config_path=output_path,
        config_dict=config_dict,
        resume_state_file=resume_state_file,
    )
    manifest_path = output_path.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return config_dict, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an isolated online config for single/dual-tower Oracle critic C/D/E probes.",
    )
    parser.add_argument("--base-config", default=str((MORTAL_ROOT / "config.toml").resolve()))
    parser.add_argument("--output", default=None)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tower", choices=tuple(sorted(TOWER_ARCHES)), required=True)
    parser.add_argument("--arm", choices=tuple(sorted(CDE_ARMS)), required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--oracle-critic-state-file", default=None)
    parser.add_argument("--critic-warmup-steps", type=int, default=0)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=5000)
    parser.add_argument("--opponent-pool-preset", default="validation")
    parser.add_argument("--repro-seed", type=int, default=None)
    parser.add_argument("--train-key", type=int, default=None)
    parser.add_argument("--train-seed-start", type=int, default=None)
    parser.add_argument("--allow-cudnn-benchmark", action="store_true")
    parser.add_argument("--monitor-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=0.05)
    parser.add_argument("--policy-clip-ratio", type=float, default=None)
    parser.add_argument("--actor-lr-scale", type=float, default=1.0)
    parser.add_argument("--policy-head-lr-scale", type=float, default=1.0)
    parser.add_argument("--policy-update-interval", type=int, default=1)
    parser.add_argument("--policy-update-phase", type=int, default=0)
    parser.add_argument("--entropy-floor", type=float, default=0.0)
    parser.add_argument("--entropy-adjust-rate", type=float, default=0.0)
    parser.add_argument("--entropy-floor-start-step", type=int, default=0)
    parser.add_argument("--vtrace-target-rho-clip", type=float, default=0.0)
    parser.add_argument("--vtrace-target-c-clip", type=float, default=0.0)
    parser.add_argument("--vtrace-mode", choices=("auto", "always", "disabled"), default="auto")
    parser.add_argument("--vtrace-min-version-gap", type=int, default=2)
    parser.add_argument(
        "--resume-state-file",
        default=None,
        help="Copy this online checkpoint to the generated [control].state_file before launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or default_run_name(
        tower=args.tower,
        arm=args.arm,
        max_steps=args.max_steps,
    )
    runtime_root = Path(args.runtime_root or (REPO_ROOT / "logs" / "oracle_cde" / run_name)).resolve()
    output_path = Path(args.output or (runtime_root / "config.toml")).resolve()
    _, manifest = write_oracle_cde_config(
        base_config_path=args.base_config,
        output_path=output_path,
        runtime_root=runtime_root,
        run_name=run_name,
        tower=args.tower,
        arm=args.arm,
        max_steps=args.max_steps,
        oracle_critic_state_file=args.oracle_critic_state_file,
        critic_warmup_steps=args.critic_warmup_steps,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        opponent_pool_preset=args.opponent_pool_preset,
        repro_seed=args.repro_seed,
        train_key=args.train_key,
        train_seed_start=args.train_seed_start,
        allow_cudnn_benchmark=args.allow_cudnn_benchmark,
        monitor_every=args.monitor_every,
        batch_size=args.batch_size,
        value_weight=args.value_weight,
        resume_state_file=args.resume_state_file,
        policy_clip_ratio=args.policy_clip_ratio,
        actor_lr_scale=args.actor_lr_scale,
        policy_head_lr_scale=args.policy_head_lr_scale,
        policy_update_interval=args.policy_update_interval,
        policy_update_phase=args.policy_update_phase,
        entropy_floor=args.entropy_floor,
        entropy_adjust_rate=args.entropy_adjust_rate,
        entropy_floor_start_step=args.entropy_floor_start_step,
        vtrace_target_rho_clip=args.vtrace_target_rho_clip,
        vtrace_target_c_clip=args.vtrace_target_c_clip,
        vtrace_mode=args.vtrace_mode,
        vtrace_min_version_gap=args.vtrace_min_version_gap,
    )
    print(manifest["config_path"])


if __name__ == "__main__":
    main()

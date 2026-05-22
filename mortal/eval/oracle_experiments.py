from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import torch


VALID_ORACLE_INPUT_MODES = frozenset({"true", "zero", "shuffled"})


@dataclass(frozen=True)
class OracleExperimentArm:
    name: str
    actor_oracle_enabled: bool
    actor_oracle_source: str
    oracle_critic_enabled: bool
    description: str


_DEFAULT_ARM_SPECS = {
    "visible_only": OracleExperimentArm(
        name="visible_only",
        actor_oracle_enabled=False,
        actor_oracle_source="zero",
        oracle_critic_enabled=False,
        description="Visible-only actor baseline with no Oracle critic.",
    ),
    "actor_true": OracleExperimentArm(
        name="actor_true",
        actor_oracle_enabled=True,
        actor_oracle_source="true",
        oracle_critic_enabled=True,
        description="True Oracle actor guiding with zero-Oracle continuation and Oracle critic.",
    ),
    "actor_shuffled": OracleExperimentArm(
        name="actor_shuffled",
        actor_oracle_enabled=True,
        actor_oracle_source="shuffled",
        oracle_critic_enabled=True,
        description="Fake/shuffled Oracle actor guiding with zero-Oracle continuation and Oracle critic.",
    ),
    "critic_only": OracleExperimentArm(
        name="critic_only",
        actor_oracle_enabled=False,
        actor_oracle_source="zero",
        oracle_critic_enabled=True,
        description="Visible-only actor with Oracle critic only.",
    ),
}

_ARM_ALIASES = {
    "baseline": "visible_only",
    "visible": "visible_only",
    "true": "actor_true",
    "actor_true_continuation": "actor_true",
    "oracle_actor_true": "actor_true",
    "shuffled": "actor_shuffled",
    "fake": "actor_shuffled",
    "fake_oracle": "actor_shuffled",
    "oracle_actor_shuffled": "actor_shuffled",
    "oracle_critic_only": "critic_only",
}


def _cfg_section(config_dict: Any, key: str) -> dict[str, Any]:
    if not isinstance(config_dict, dict):
        return {}
    section = config_dict.get(key, {})
    return section if isinstance(section, dict) else {}


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    parsed = str(value).strip().lower()
    if parsed in {"1", "true", "yes", "on"}:
        return True
    if parsed in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_oracle_input_mode(mode: Any, *, field_name: str = "oracle_input_mode") -> str:
    parsed = str(mode or "zero").strip().lower()
    if parsed not in VALID_ORACLE_INPUT_MODES:
        raise ValueError(
            f"{field_name} must be one of {sorted(VALID_ORACLE_INPUT_MODES)}, got {mode!r}"
        )
    return parsed


def apply_oracle_input_mode(
    invisible_obs: Optional[torch.Tensor],
    mode: Any,
) -> Optional[torch.Tensor]:
    if invisible_obs is None:
        return None
    normalized = normalize_oracle_input_mode(mode)
    if normalized == "true":
        return invisible_obs
    if normalized == "zero":
        return torch.zeros_like(invisible_obs)

    if invisible_obs.ndim < 2:
        raise ValueError(
            f"shuffled oracle input expects at least 2 dims, got shape {tuple(invisible_obs.shape)}"
        )
    if invisible_obs.shape[0] > 1:
        return torch.roll(invisible_obs, shifts=1, dims=0)
    return torch.roll(invisible_obs, shifts=1, dims=1)


def derive_current_config_arm(config: Mapping[str, Any]) -> OracleExperimentArm:
    oracle_cfg = _cfg_section(config, "oracle_guiding")
    value_cfg = _cfg_section(config, "value")
    actor_enabled = bool(oracle_cfg.get("actor_enabled", False))
    actor_source = normalize_oracle_input_mode(
        oracle_cfg.get("actor_source", "true" if actor_enabled else "zero"),
        field_name="oracle_guiding.actor_source",
    )
    value_enabled = bool(value_cfg.get("enabled", False))
    oracle_critic_enabled = value_enabled and bool(value_cfg.get("oracle_critic", True))
    return OracleExperimentArm(
        name="current_config",
        actor_oracle_enabled=actor_enabled,
        actor_oracle_source=actor_source,
        oracle_critic_enabled=oracle_critic_enabled,
        description="Use the Oracle settings already present in config.toml.",
    )


def resolve_oracle_experiment_arm(
    config: Mapping[str, Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> OracleExperimentArm:
    env_map = os.environ if env is None else env
    exp_cfg = _cfg_section(config, "oracle_experiments")
    selected = (
        env_map.get("MORTAL_ORACLE_ARM")
        or exp_cfg.get("selected_arm")
        or exp_cfg.get("default_arm")
        or "current_config"
    )
    selected_name = str(selected).strip().lower()
    selected_name = _ARM_ALIASES.get(selected_name, selected_name)
    if selected_name in {"", "current_config", "config", "default", "mainline"}:
        return derive_current_config_arm(config)
    if selected_name not in _DEFAULT_ARM_SPECS:
        raise ValueError(
            f"unknown Oracle experiment arm {selected!r}; expected one of "
            f"{['current_config', *sorted(_DEFAULT_ARM_SPECS.keys())]}"
        )
    return _DEFAULT_ARM_SPECS[selected_name]


def resolve_oracle_artifact_suffix(
    config: Mapping[str, Any],
    arm: OracleExperimentArm,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    env_map = os.environ if env is None else env
    if "MORTAL_ORACLE_ARTIFACT_SUFFIX" in env_map:
        return str(env_map["MORTAL_ORACLE_ARTIFACT_SUFFIX"]).strip()
    exp_cfg = _cfg_section(config, "oracle_experiments")
    if "artifact_suffix" in exp_cfg:
        return str(exp_cfg.get("artifact_suffix", "") or "").strip()
    if arm.name == "current_config":
        return ""
    suffix_enabled = _coerce_bool(exp_cfg.get("suffix_artifacts", True), default=True)
    return arm.name if suffix_enabled else ""


def _suffix_path_string(path_value: str, suffix: str) -> str:
    if not path_value or not suffix:
        return path_value
    path = Path(path_value)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))
    return str(path.with_name(f"{path.name}_{suffix}"))


def _suffix_config_path(node: MutableMapping[str, Any], key: str, suffix: str) -> None:
    value = node.get(key)
    if isinstance(value, str) and value:
        node[key] = _suffix_path_string(value, suffix)


def apply_oracle_experiment_to_config(
    config: MutableMapping[str, Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[OracleExperimentArm, str]:
    exp_cfg = config.setdefault("oracle_experiments", {})
    if not isinstance(exp_cfg, dict):
        raise ValueError("oracle_experiments config section must be a table")

    if exp_cfg.get("_applied", False):
        arm_name = str(exp_cfg.get("resolved_arm", "current_config") or "current_config")
        arm = (
            derive_current_config_arm(config)
            if arm_name == "current_config"
            else _DEFAULT_ARM_SPECS.get(arm_name, derive_current_config_arm(config))
        )
        return arm, str(exp_cfg.get("artifact_suffix", "") or "")

    arm = resolve_oracle_experiment_arm(config, env=env)
    suffix = resolve_oracle_artifact_suffix(config, arm, env=env)

    oracle_cfg = config.setdefault("oracle_guiding", {})
    if not isinstance(oracle_cfg, dict):
        raise ValueError("oracle_guiding config section must be a table")
    oracle_cfg["actor_enabled"] = bool(arm.actor_oracle_enabled)
    oracle_cfg["actor_source"] = arm.actor_oracle_source

    value_cfg = config.setdefault("value", {})
    if not isinstance(value_cfg, dict):
        raise ValueError("value config section must be a table")
    value_cfg["oracle_critic"] = bool(arm.oracle_critic_enabled)

    if suffix:
        control_cfg = _cfg_section(config, "control")
        test_play_cfg = _cfg_section(config, "test_play")
        one_vs_three_cfg = _cfg_section(config, "1v3")
        online_cfg = _cfg_section(config, "online")
        online_server_cfg = _cfg_section(online_cfg, "server")
        _suffix_config_path(control_cfg, "state_file", suffix)
        _suffix_config_path(control_cfg, "best_state_file", suffix)
        _suffix_config_path(control_cfg, "tensorboard_dir", suffix)
        _suffix_config_path(test_play_cfg, "log_dir", suffix)
        _suffix_config_path(one_vs_three_cfg, "log_dir", suffix)
        _suffix_config_path(online_server_cfg, "buffer_dir", suffix)
        _suffix_config_path(online_server_cfg, "drain_dir", suffix)
        train_play_cfg = config.get("train_play", {})
        if isinstance(train_play_cfg, dict):
            for profile_cfg in train_play_cfg.values():
                if isinstance(profile_cfg, dict):
                    _suffix_config_path(profile_cfg, "log_dir", suffix)
        dep_eval_cfg = _cfg_section(config, "oracle_dependency_eval")
        _suffix_config_path(dep_eval_cfg, "log_dir", suffix)

    exp_cfg["selected_arm"] = arm.name
    exp_cfg["resolved_arm"] = arm.name
    exp_cfg["artifact_suffix"] = suffix
    exp_cfg["description"] = arm.description
    exp_cfg["_applied"] = True
    return arm, suffix


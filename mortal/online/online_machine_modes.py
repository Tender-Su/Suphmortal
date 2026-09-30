from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mortal._repo import MORTAL_ROOT
from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.core.config_utils import (
    deep_merge_dict as _deep_merge_dict,
    ensure_dict_section as _cfg_section,
)


PATH_KEYS = {
    "init_state_file",
    "state_file",
    "anchor_state_file",
    "champion_state_file",
    "latest_state_file",
    "best_loss_state_file",
    "best_acc_state_file",
    "best_state_file",
    "oracle_critic_state_file",
    "critic_state_file",
    "pretrained_state_file",
    "tensorboard_dir",
    "log_dir",
    "file_index",
    "buffer_dir",
    "drain_dir",
    "dir",
    "tactics",
}

GLOB_KEYS = {
    "globs",
    "train_globs",
    "val_globs",
}

PATH_LIST_KEYS = {
    "history_state_files",
}


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    description: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class OpponentPoolPreset:
    name: str
    description: str
    config_key: str | None


UNIFIED_STEP0_BASELINE_GAMES = 600


RECORDED_STEP0_BASELINES: dict[str, dict[str, Any]] = {
    "ms_rl1_minimal_500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_minimal_1500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_minimal_3000": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_1500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_3000": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_rank_opp_danger_500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_rank_opp_danger_1500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_rank_opp_danger_3000": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_shared_stack_500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_shared_stack_1500": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_shared_stack_3000": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_minimal_20k": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_20k": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_add_value_gae_is_rank_opp_danger_20k": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
    "ms_rl1_shared_stack_20k": {
        "games": UNIFIED_STEP0_BASELINE_GAMES,
        "avg_rank": 2.525,
        "avg_pt": -2.475,
        "source_run": "rl1_add_value_gae_is_20k_20260412_002033",
        "note": "Canonical RL-1 visible-only all-action baseline for the short-window add-back family.",
    },
}


def lookup_recorded_step0_baseline(profile_name: str) -> dict[str, Any] | None:
    baseline = RECORDED_STEP0_BASELINES.get(profile_name)
    if baseline is not None:
        return baseline
    fallback_name = str(profile_name).replace("_oracle_critic", "")
    if fallback_name != profile_name:
        baseline = RECORDED_STEP0_BASELINES.get(fallback_name)
        if baseline is not None:
            return baseline
    return None


def _clean_policy_ppo_overrides(*, action_scope: str = "all") -> dict[str, Any]:
    return {
        "policy": {
            "online_action_scope": action_scope,
            "logit_thres": 0.0,
            "importance_rho_clip": 0.0,
            "importance_c_clip": 0.0,
            "vtrace_rho_clip": 0.0,
            "vtrace_c_clip": 0.0,
            "entropy_floor": 0.0,
            "entropy_floor_start_step": 0,
            "entropy_target": 0.0,
            "entropy_adjust_rate": 0.0,
        },
    }


def _scaled_online_warmup_steps(max_steps: int) -> int:
    max_steps = max(int(max_steps), 1)
    if max_steps <= 500:
        return 50
    if max_steps <= 1500:
        return 100
    if max_steps <= 3000:
        return 150
    return max(200, min(5000, max_steps // 20))


def _oracle_sanity_profile_overrides(
    *,
    actor_oracle_enabled: bool,
    decay_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    overrides = {
        "expected_reward": {
            "enabled": False,
        },
        "value": {
            "enabled": True,
            "oracle_critic": False,
        },
        "oracle_guiding": {
            "actor_enabled": actor_oracle_enabled,
            "actor_source": "true" if actor_oracle_enabled else "zero",
            "schedule": "linear",
            "gamma_start": 1.0 if actor_oracle_enabled else 0.0,
            "gamma_end": 0.0,
            "hold_steps": 0,
            "decay_steps": int(decay_steps),
        },
        "search": {
            "enabled": False,
        },
        "search_distill": {
            "enabled": False,
        },
        "oracle_dependency_eval": {
            "enabled": False,
        },
        "optim": {
            "scheduler": {
                "warm_up_steps": _scaled_online_warmup_steps(max_steps),
                "max_steps": int(max_steps),
            },
        },
        "test_play": {
            "initial_enable": False,
            "initial_games": UNIFIED_STEP0_BASELINE_GAMES,
        },
    }
    overrides["policy"] = dict(_clean_policy_ppo_overrides()["policy"])
    return overrides


def _oracle_minimal_profile_overrides(
    *,
    actor_oracle_enabled: bool,
    decay_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    return _oracle_component_profile_overrides(
        actor_oracle_enabled=actor_oracle_enabled,
        decay_steps=decay_steps,
        max_steps=max_steps,
    )


def _oracle_component_profile_overrides(
    *,
    actor_oracle_enabled: bool,
    decay_steps: int,
    max_steps: int,
    test_every: int | None = None,
    train_play_games: int | None = None,
    test_play_games: int | None = None,
    policy_action_scope: str = "all",
    grp_label_smoothing: float | None = None,
    value_enabled: bool = False,
    oracle_critic: bool = False,
    value_weight: float | None = None,
    zero_sum_weight: float | None = None,
    value_target_mode: str | None = None,
    value_reward_source: str | None = None,
    gae_enabled: bool = False,
    replay_is_enabled: bool = False,
    next_rank_weight: float = 0.0,
    opponent_state_weight: float = 0.0,
    danger_enabled: bool = False,
    danger_weight: float = 0.0,
    tile_efficiency_weight: float = 0.0,
    furo_regret_weight: float = 0.0,
    hand_value_regret_weight: float = 0.0,
) -> dict[str, Any]:
    if gae_enabled and not value_enabled:
        raise ValueError("gae_enabled requires value_enabled")
    if oracle_critic and not value_enabled:
        raise ValueError("oracle_critic requires value_enabled")

    resolved_value_weight = float(
        value_weight if value_weight is not None else (0.05 if value_enabled else 0.0)
    )
    resolved_zero_sum_weight = float(
        zero_sum_weight if zero_sum_weight is not None else (0.01 if value_enabled else 0.0)
    )

    overrides = {
        "expected_reward": {
            "enabled": False,
        },
        "policy": {
            "gae_enabled": bool(gae_enabled),
        },
        "aux": {
            "next_rank_weight": float(next_rank_weight),
            "opponent_state_weight": float(opponent_state_weight),
            "danger_enabled": bool(danger_enabled),
            "danger_weight": float(danger_weight),
            "tile_efficiency_weight": float(tile_efficiency_weight),
            "furo_regret_weight": float(furo_regret_weight),
            "hand_value_regret_weight": float(hand_value_regret_weight),
        },
        "value": {
            "enabled": bool(value_enabled),
            "oracle_critic": bool(oracle_critic),
            "weight": resolved_value_weight,
            "zero_sum_weight": resolved_zero_sum_weight,
        },
        "oracle_guiding": {
            "actor_enabled": actor_oracle_enabled,
            "actor_source": "true" if actor_oracle_enabled else "zero",
            "schedule": "linear",
            "gamma_start": 1.0 if actor_oracle_enabled else 0.0,
            "gamma_end": 0.0,
            "hold_steps": 0,
            "decay_steps": int(decay_steps),
        },
        "search": {
            "enabled": False,
        },
        "search_distill": {
            "enabled": False,
        },
        "oracle_dependency_eval": {
            "enabled": False,
        },
        "online": {
            "stop_at_max_steps": True,
            "importance_sampling": {
                "enabled": bool(replay_is_enabled),
                "max_policy_versions": 8 if replay_is_enabled else 0,
                "drop_untracked_samples": False,
            },
        },
        "optim": {
            "scheduler": {
                "warm_up_steps": _scaled_online_warmup_steps(max_steps),
                "max_steps": int(max_steps),
            },
        },
    }
    overrides["policy"].update(
        _clean_policy_ppo_overrides(action_scope=policy_action_scope)["policy"]
    )
    if grp_label_smoothing is not None:
        overrides["grp"] = {
            "label_smoothing": float(grp_label_smoothing),
        }
    if value_target_mode is not None:
        overrides["value"]["target_mode"] = str(value_target_mode)
    if value_enabled:
        overrides["value"]["reward_source"] = str(
            value_reward_source
            if value_reward_source is not None
            else ("score_rank" if oracle_critic else "grp")
        )
    elif value_reward_source is not None:
        overrides["value"]["reward_source"] = str(value_reward_source)
    if test_every is not None:
        overrides["control"] = {
            "test_every": int(test_every),
        }
    if train_play_games is not None:
        overrides["train_play"] = {
            "default": {
                "games": int(train_play_games),
            },
        }
    test_play_overrides = overrides.get("test_play")
    if not isinstance(test_play_overrides, dict):
        test_play_overrides = {}
        overrides["test_play"] = test_play_overrides
    test_play_overrides["initial_enable"] = False
    test_play_overrides["initial_games"] = UNIFIED_STEP0_BASELINE_GAMES
    if test_play_games is not None:
        test_play_overrides["games"] = int(test_play_games)

    return overrides


def _build_incremental_profile(
    *,
    actor_oracle_enabled: bool,
    slug: str,
    description: str,
    decay_steps: int,
    max_steps: int,
    test_every: int,
    train_play_games: int | None = None,
    test_play_games: int | None = None,
    policy_action_scope: str = "all",
    grp_label_smoothing: float | None = None,
    value_enabled: bool = False,
    oracle_critic: bool = False,
    value_weight: float | None = None,
    zero_sum_weight: float | None = None,
    value_target_mode: str | None = None,
    value_reward_source: str | None = None,
    gae_enabled: bool = False,
    replay_is_enabled: bool = False,
    next_rank_weight: float = 0.0,
    opponent_state_weight: float = 0.0,
    danger_enabled: bool = False,
    danger_weight: float = 0.0,
    tile_efficiency_weight: float = 0.0,
    furo_regret_weight: float = 0.0,
    hand_value_regret_weight: float = 0.0,
) -> tuple[str, ExperimentProfile]:
    rl_name = "RL-2" if actor_oracle_enabled else "RL-1"
    actor_desc = "actor Oracle guiding" if actor_oracle_enabled else "visible-only actor"
    profile_name = f"ms_{'rl2' if actor_oracle_enabled else 'rl1'}_{slug}"
    profile = ExperimentProfile(
        name=profile_name,
        description=(
            f"Incremental {rl_name} probe: {description}; {actor_desc}; "
            f"decay={decay_steps}, max_steps={max_steps}, test_every={test_every}."
        ),
        overrides=_oracle_component_profile_overrides(
            actor_oracle_enabled=actor_oracle_enabled,
            decay_steps=decay_steps,
            max_steps=max_steps,
            test_every=test_every,
            train_play_games=train_play_games,
            test_play_games=test_play_games,
            policy_action_scope=policy_action_scope,
            grp_label_smoothing=grp_label_smoothing,
            value_enabled=value_enabled,
            oracle_critic=oracle_critic,
            value_weight=value_weight,
            zero_sum_weight=zero_sum_weight,
            value_target_mode=value_target_mode,
            value_reward_source=value_reward_source,
            gae_enabled=gae_enabled,
            replay_is_enabled=replay_is_enabled,
            next_rank_weight=next_rank_weight,
            opponent_state_weight=opponent_state_weight,
            danger_enabled=danger_enabled,
            danger_weight=danger_weight,
            tile_efficiency_weight=tile_efficiency_weight,
            furo_regret_weight=furo_regret_weight,
            hand_value_regret_weight=hand_value_regret_weight,
        ),
    )
    return profile_name, profile


def _register_incremental_profiles(profiles: dict[str, ExperimentProfile]) -> None:
    stage_specs: list[dict[str, Any]] = []

    micro_gate_specs = [
        {
            "suffix": "500",
            "description_suffix": "500-step micro gate",
            "decay_steps": 300,
            "max_steps": 500,
            "test_every": 500,
            "train_play_games": 200,
            "test_play_games": 200,
        },
        {
            "suffix": "1500",
            "description_suffix": "1500-step micro gate",
            "decay_steps": 900,
            "max_steps": 1500,
            "test_every": 1500,
            "train_play_games": 300,
            "test_play_games": 400,
        },
        {
            "suffix": "3000",
            "description_suffix": "3000-step micro gate",
            "decay_steps": 1800,
            "max_steps": 3000,
            "test_every": 3000,
            "train_play_games": 400,
            "test_play_games": 600,
        },
    ]
    micro_ladder_specs = [
        {
            "slug_prefix": "minimal",
            "description_prefix": "all-action minimal visible-only probe",
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
        },
        {
            "slug_prefix": "add_rank_opp_danger",
            "description_prefix": "all-action probe with rank aux and opponent/danger heads",
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
        },
        {
            "slug_prefix": "add_value_gae_is",
            "description_prefix": "all-action probe with value head, GAE, and replay importance sampling",
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
        },
        {
            "slug_prefix": "add_value_gae_is_oracle_critic",
            "description_prefix": "all-action probe with Oracle critic, value head, GAE, and replay importance sampling",
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "oracle_critic": True,
            "value_reward_source": "score_rank",
            "gae_enabled": True,
            "replay_is_enabled": True,
        },
        {
            "slug_prefix": "add_value_gae_is_rank_opp_danger",
            "description_prefix": "all-action probe with value head, GAE, replay IS, rank aux, and opponent/danger heads",
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
        },
        {
            "slug_prefix": "shared_stack",
            "description_prefix": "all-action shared stack probe without search or oracle critic",
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
            "tile_efficiency_weight": 0.01,
            "furo_regret_weight": 0.01,
            "hand_value_regret_weight": 0.01,
        },
    ]
    for gate in micro_gate_specs:
        for ladder in micro_ladder_specs:
            stage_specs.append(
                {
                    "slug": f"{ladder['slug_prefix']}_{gate['suffix']}",
                    "description": f"{ladder['description_prefix']} on the {gate['description_suffix']}",
                    "decay_steps": gate["decay_steps"],
                    "max_steps": gate["max_steps"],
                    "test_every": gate["test_every"],
                    "train_play_games": gate["train_play_games"],
                    "test_play_games": gate["test_play_games"],
                    **{
                        key: value
                        for key, value in ladder.items()
                        if key not in {"slug_prefix", "description_prefix"}
                    },
                }
            )

    stage_specs.extend([
        {
            "slug": "mortal_policy_smoke_4k",
            "description": "Mortal-Policy-style unified PPO smoke: all actions, no value stack, no reward smoothing",
            "decay_steps": 2400,
            "max_steps": 4000,
            "test_every": 1000,
            "train_play_games": 400,
            "test_play_games": 800,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
        },
        {
            "slug": "smoke_4k",
            "description": "ultra-short clean smoke for PPO direction check",
            "decay_steps": 2400,
            "max_steps": 4000,
            "test_every": 1000,
            "train_play_games": 400,
            "test_play_games": 800,
        },
        {
            "slug": "smoke_10k",
            "description": "shortest clean smoke for fast RL-1/RL-2 direction check",
            "decay_steps": 6000,
            "max_steps": 10000,
            "test_every": 2500,
            "train_play_games": 400,
            "test_play_games": 1000,
        },
        {
            "slug": "smoke_20k",
            "description": "minimal smoke for early positive-direction check",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
        },
        {
            "slug": "minimal_20k",
            "description": "all-action baseline with a shorter 20k validation window",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
            "train_play_games": 400,
            "test_play_games": 600,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
        },
        {
            "slug": "add_rank_opp_danger_20k",
            "description": "all-action baseline plus rank aux and opponent/danger heads on a shorter 20k window",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
            "train_play_games": 400,
            "test_play_games": 600,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
        },
        {
            "slug": "add_value_gae_is_20k",
            "description": "all-action baseline plus value head, GAE, and replay importance sampling on a shorter 20k window",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
            "train_play_games": 400,
            "test_play_games": 600,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
        },
        {
            "slug": "add_value_gae_is_oracle_critic_20k",
            "description": "all-action baseline plus Oracle critic, value head, GAE, and replay importance sampling on a shorter 20k window",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
            "train_play_games": 400,
            "test_play_games": 600,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "oracle_critic": True,
            "value_reward_source": "score_rank",
            "gae_enabled": True,
            "replay_is_enabled": True,
        },
        {
            "slug": "add_value_gae_is_rank_opp_danger_20k",
            "description": "all-action baseline plus value head, GAE, replay IS, rank aux, and opponent/danger heads on a shorter 20k window",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
            "train_play_games": 400,
            "test_play_games": 600,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
        },
        {
            "slug": "shared_stack_20k",
            "description": "all-action shared stack without search or oracle critic on a shorter 20k window",
            "decay_steps": 12000,
            "max_steps": 20000,
            "test_every": 5000,
            "train_play_games": 400,
            "test_play_games": 600,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
            "tile_efficiency_weight": 0.01,
            "furo_regret_weight": 0.01,
            "hand_value_regret_weight": 0.01,
        },
        {
            "slug": "minimal_40k",
            "description": "Mortal-Policy-style all-action baseline with a longer 40k window",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
        },
        {
            "slug": "add_rank_opp_danger_40k",
            "description": "all-action baseline plus rank aux and opponent/danger heads",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
        },
        {
            "slug": "add_value_40k",
            "description": "all-action baseline plus visible-only value head",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
        },
        {
            "slug": "add_value_gae_40k",
            "description": "all-action baseline plus value head and step-level GAE",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
        },
        {
            "slug": "add_value_gae_is_40k",
            "description": "all-action baseline plus value head, GAE, and replay importance sampling",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
        },
        {
            "slug": "add_value_gae_is_oracle_critic_40k",
            "description": "all-action baseline plus Oracle critic, value head, GAE, and replay importance sampling",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "oracle_critic": True,
            "value_reward_source": "score_rank",
            "gae_enabled": True,
            "replay_is_enabled": True,
        },
        {
            "slug": "add_value_gae_is_rank_40k",
            "description": "all-action baseline plus value head, GAE, replay IS, and next-rank auxiliary head",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
        },
        {
            "slug": "add_value_gae_is_rank_opp_danger_40k",
            "description": "all-action baseline plus value head, GAE, replay IS, rank aux, and opponent/danger heads",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
        },
        {
            "slug": "shared_stack_40k",
            "description": "all-action shared stack without search or oracle critic",
            "decay_steps": 24000,
            "max_steps": 40000,
            "test_every": 10000,
            "policy_action_scope": "all",
            "grp_label_smoothing": 0.0,
            "value_enabled": True,
            "gae_enabled": True,
            "replay_is_enabled": True,
            "next_rank_weight": 0.2,
            "opponent_state_weight": 0.03,
            "danger_enabled": True,
            "danger_weight": 0.05,
            "tile_efficiency_weight": 0.01,
            "furo_regret_weight": 0.01,
            "hand_value_regret_weight": 0.01,
        },
    ])
    for actor_oracle_enabled in (False, True):
        for spec in stage_specs:
            profile_name, profile = _build_incremental_profile(
                actor_oracle_enabled=actor_oracle_enabled,
                slug=spec["slug"],
                description=spec["description"],
                decay_steps=spec["decay_steps"],
                max_steps=spec["max_steps"],
                test_every=spec["test_every"],
                train_play_games=spec.get("train_play_games"),
                test_play_games=spec.get("test_play_games"),
                policy_action_scope=spec.get("policy_action_scope", "all"),
                grp_label_smoothing=spec.get("grp_label_smoothing"),
                value_enabled=spec.get("value_enabled", False),
                oracle_critic=spec.get("oracle_critic", False),
                value_weight=spec.get("value_weight"),
                zero_sum_weight=spec.get("zero_sum_weight"),
                value_target_mode=spec.get("value_target_mode"),
                value_reward_source=spec.get("value_reward_source"),
                gae_enabled=spec.get("gae_enabled", False),
                replay_is_enabled=spec.get("replay_is_enabled", False),
                next_rank_weight=spec.get("next_rank_weight", 0.0),
                opponent_state_weight=spec.get("opponent_state_weight", 0.0),
                danger_enabled=spec.get("danger_enabled", False),
                danger_weight=spec.get("danger_weight", 0.0),
                tile_efficiency_weight=spec.get("tile_efficiency_weight", 0.0),
                furo_regret_weight=spec.get("furo_regret_weight", 0.0),
                hand_value_regret_weight=spec.get("hand_value_regret_weight", 0.0),
            )
            profiles[profile_name] = profile


EXPERIMENT_PROFILES = {
    "default": ExperimentProfile(
        name="default",
        description="No experiment-specific overrides; keep the base config as-is.",
        overrides={},
    ),
    "ms_rl1_pilot_100k": ExperimentProfile(
        name="ms_rl1_pilot_100k",
        description=(
            "Microsoft/Suphx-style RL-1 quick pilot: visible-only actor, no Oracle critic, "
            "no search, no search distill, 60k decay placeholder, 100k LR schedule."
        ),
        overrides=_oracle_sanity_profile_overrides(
            actor_oracle_enabled=False,
            decay_steps=60000,
            max_steps=100000,
        ),
    ),
    "ms_rl2_pilot_100k": ExperimentProfile(
        name="ms_rl2_pilot_100k",
        description=(
            "Microsoft/Suphx-style RL-2 quick pilot: actor Oracle guiding only, no Oracle "
            "critic, no search, no search distill, 60k decay, 100k LR schedule."
        ),
        overrides=_oracle_sanity_profile_overrides(
            actor_oracle_enabled=True,
            decay_steps=60000,
            max_steps=100000,
        ),
    ),
    "ms_rl1_sanity_120k": ExperimentProfile(
        name="ms_rl1_sanity_120k",
        description=(
            "Microsoft/Suphx-style RL-1 sanity: visible-only actor, no Oracle critic, no "
            "search, no search distill, 80k decay placeholder, 120k LR schedule."
        ),
        overrides=_oracle_sanity_profile_overrides(
            actor_oracle_enabled=False,
            decay_steps=80000,
            max_steps=120000,
        ),
    ),
    "ms_rl2_sanity_120k": ExperimentProfile(
        name="ms_rl2_sanity_120k",
        description=(
            "Microsoft/Suphx-style RL-2 sanity: actor Oracle guiding only, no Oracle critic, "
            "no search, no search distill, 80k decay, 120k LR schedule."
        ),
        overrides=_oracle_sanity_profile_overrides(
            actor_oracle_enabled=True,
            decay_steps=80000,
            max_steps=120000,
        ),
    ),
    "ms_rl1_minimal_100k": ExperimentProfile(
        name="ms_rl1_minimal_100k",
        description=(
            "Minimal RL-1 reproduction: visible-only actor, policy-only PPO, no value/GAE, "
            "no replay importance sampling, no aux/regret heads, no search, 60k placeholder "
            "decay, 100k LR schedule."
        ),
        overrides=_oracle_minimal_profile_overrides(
            actor_oracle_enabled=False,
            decay_steps=60000,
            max_steps=100000,
        ),
    ),
    "ms_rl2_minimal_100k": ExperimentProfile(
        name="ms_rl2_minimal_100k",
        description=(
            "Minimal RL-2 reproduction: RL-1 minimal baseline plus actor Oracle guiding, no "
            "value/GAE, no replay importance sampling, no aux/regret heads, no search, 60k "
            "decay, 100k LR schedule."
        ),
        overrides=_oracle_minimal_profile_overrides(
            actor_oracle_enabled=True,
            decay_steps=60000,
            max_steps=100000,
        ),
    ),
    "ms_rl1_minimal_120k": ExperimentProfile(
        name="ms_rl1_minimal_120k",
        description=(
            "Minimal RL-1 reproduction: visible-only actor, policy-only PPO, no value/GAE, "
            "no replay importance sampling, no aux/regret heads, no search, 80k placeholder "
            "decay, 120k LR schedule."
        ),
        overrides=_oracle_minimal_profile_overrides(
            actor_oracle_enabled=False,
            decay_steps=80000,
            max_steps=120000,
        ),
    ),
    "ms_rl2_minimal_120k": ExperimentProfile(
        name="ms_rl2_minimal_120k",
        description=(
            "Minimal RL-2 reproduction: RL-1 minimal baseline plus actor Oracle guiding, no "
            "value/GAE, no replay importance sampling, no aux/regret heads, no search, 80k "
            "decay, 120k LR schedule."
        ),
        overrides=_oracle_minimal_profile_overrides(
            actor_oracle_enabled=True,
            decay_steps=80000,
            max_steps=120000,
        ),
    ),
}
_register_incremental_profiles(EXPERIMENT_PROFILES)


OPPONENT_POOL_PRESETS = {
    "default": OpponentPoolPreset(
        name="default",
        description="Keep baseline.train exactly as it appears in the base config.",
        config_key=None,
    ),
    "validation": OpponentPoolPreset(
        name="validation",
        description=(
            "Use baseline.train_presets.validation as the active baseline.train pool "
            "for clean causal probes, Oracle ablations, and short-window A/Bs."
        ),
        config_key="validation",
    ),
    "formal": OpponentPoolPreset(
        name="formal",
        description=(
            "Use baseline.train_presets.formal as the active baseline.train pool for "
            "longer formal training and ceiling-pushing runs."
        ),
        config_key="formal",
    ),
}


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_config_paths(node: Any, base_dir: Path) -> Any:
    if isinstance(node, dict):
        resolved: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(value, dict):
                resolved[key] = resolve_config_paths(value, base_dir)
            elif key in PATH_KEYS and isinstance(value, str):
                resolved[key] = _resolve_path(value, base_dir)
            elif key in PATH_LIST_KEYS and isinstance(value, list):
                resolved[key] = [
                    _resolve_path(item, base_dir) if isinstance(item, str) else item
                    for item in value
                ]
            elif key in GLOB_KEYS and isinstance(value, list):
                resolved[key] = [
                    _resolve_path(item, base_dir) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved
    return node


def load_resolved_base_config(base_config_path: str | Path) -> dict[str, Any]:
    base_path = Path(base_config_path).resolve()
    return resolve_config_paths(load_toml_file(base_path), base_path.parent)


def _available_opponent_pool_preset_names(config_dict: dict[str, Any]) -> list[str]:
    names = set(OPPONENT_POOL_PRESETS)
    baseline_cfg = config_dict.get("baseline", {})
    if isinstance(baseline_cfg, dict):
        train_presets = baseline_cfg.get("train_presets", {})
        if isinstance(train_presets, dict):
            for key, value in train_presets.items():
                if isinstance(value, dict):
                    names.add(str(key).strip().lower())
    return sorted(names)


def apply_experiment_profile(
    config_dict: dict[str, Any],
    experiment_profile: str,
) -> dict[str, Any]:
    profile_name = str(experiment_profile or "default").strip().lower()
    try:
        profile = EXPERIMENT_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown experiment profile {experiment_profile!r}; expected one of "
            f"{sorted(EXPERIMENT_PROFILES)}"
        ) from exc

    if profile.overrides:
        _deep_merge_dict(config_dict, profile.overrides)

    config_dict["online_experiment_profile"] = {
        "name": profile.name,
        "description": profile.description,
    }
    recorded_step0 = lookup_recorded_step0_baseline(profile.name)
    if recorded_step0 is not None:
        config_dict["online_experiment_profile"]["recorded_step0_baseline"] = deepcopy(
            recorded_step0
        )
    return config_dict


def apply_opponent_pool_preset(
    config_dict: dict[str, Any],
    opponent_pool_preset: str,
) -> dict[str, Any]:
    preset_name = str(opponent_pool_preset or "default").strip().lower()
    preset = OPPONENT_POOL_PRESETS.get(preset_name)
    baseline_cfg = _cfg_section(config_dict, "baseline")
    train_cfg = _cfg_section(baseline_cfg, "train")

    if preset_name == "default":
        description = (
            preset.description
            if preset is not None
            else "Keep baseline.train exactly as it appears in the base config."
        )
        config_dict["online_opponent_pool_preset"] = {
            "name": preset_name,
            "description": description,
            "source": "baseline.train",
        }
        return config_dict

    train_presets = baseline_cfg.get("train_presets", {})
    preset_cfg = train_presets.get(preset_name) if isinstance(train_presets, dict) else None
    if not isinstance(preset_cfg, dict):
        if preset is not None and preset.config_key is not None:
            raise ValueError(
                f"opponent pool preset {opponent_pool_preset!r} requires "
                f"baseline.train_presets.{preset.config_key} in the base config"
            )
        available = _available_opponent_pool_preset_names(config_dict)
        raise ValueError(
            f"unknown opponent pool preset {opponent_pool_preset!r}; expected one of {available}"
        )

    _deep_merge_dict(train_cfg, preset_cfg)
    config_dict["online_opponent_pool_preset"] = {
        "name": preset_name,
        "description": (
            preset.description
            if preset is not None
            else f"Apply baseline.train_presets.{preset_name} to baseline.train."
        ),
        "source": f"baseline.train_presets.{preset_name}",
    }
    return config_dict


def apply_repro_overrides(
    config_dict: dict[str, Any],
    *,
    repro_seed: int | None = None,
    train_key: int | None = None,
    train_seed_start: int | None = None,
) -> dict[str, Any]:
    if repro_seed is None and train_key is None and train_seed_start is None:
        return config_dict

    repro_cfg = _cfg_section(config_dict, "repro")
    repro_cfg["enabled"] = True
    if repro_seed is not None:
        repro_cfg["seed"] = int(repro_seed)
    if train_key is not None:
        repro_cfg["train_key"] = int(train_key)
    if train_seed_start is not None:
        repro_cfg["train_seed_start"] = int(train_seed_start)
    return config_dict


def _set_train_play_log_dirs(config_dict: dict[str, Any], root: Path) -> None:
    train_play_cfg = _cfg_section(config_dict, "train_play")
    for profile_name, profile_cfg in train_play_cfg.items():
        if isinstance(profile_cfg, dict):
            profile_cfg["log_dir"] = str((root / "logs" / "train_play" / profile_name).resolve())


def _set_shared_local_paths(config_dict: dict[str, Any], runtime_root: Path) -> None:
    control_cfg = _cfg_section(config_dict, "control")
    control_cfg["tensorboard_dir"] = str((runtime_root / "tb_log").resolve())

    test_play_cfg = _cfg_section(config_dict, "test_play")
    test_play_cfg["log_dir"] = str((runtime_root / "logs" / "test_play").resolve())

    one_vs_three_cfg = _cfg_section(config_dict, "1v3")
    one_vs_three_cfg["log_dir"] = str((runtime_root / "logs" / "1v3").resolve())

    dep_eval_cfg = _cfg_section(config_dict, "oracle_dependency_eval")
    dep_eval_cfg["log_dir"] = str((runtime_root / "logs" / "oracle_dependency").resolve())

    online_cfg = _cfg_section(config_dict, "online")
    server_cfg = _cfg_section(online_cfg, "server")
    server_cfg["buffer_dir"] = str((runtime_root / "server" / "buffer").resolve())
    server_cfg["drain_dir"] = str((runtime_root / "server" / "drain").resolve())

    _set_train_play_log_dirs(config_dict, runtime_root)


def _ensure_runtime_root_layout(config_dict: dict[str, Any], runtime_root: Path) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)

    dirs_to_create = {
        runtime_root / "checkpoints",
        runtime_root / "tb_log",
        runtime_root / "logs",
        runtime_root / "server",
        runtime_root / "server" / "buffer",
        runtime_root / "server" / "drain",
    }

    control_cfg = _cfg_section(config_dict, "control")
    for key in ("state_file", "best_state_file", "tensorboard_dir"):
        value = control_cfg.get(key)
        if isinstance(value, str) and value:
            target = Path(value)
            dirs_to_create.add(target if key == "tensorboard_dir" else target.parent)

    for section_name in ("test_play", "1v3", "oracle_dependency_eval"):
        section = _cfg_section(config_dict, section_name)
        value = section.get("log_dir")
        if isinstance(value, str) and value:
            dirs_to_create.add(Path(value))

    train_play_cfg = _cfg_section(config_dict, "train_play")
    for profile_cfg in train_play_cfg.values():
        if isinstance(profile_cfg, dict):
            value = profile_cfg.get("log_dir")
            if isinstance(value, str) and value:
                dirs_to_create.add(Path(value))

    online_cfg = _cfg_section(config_dict, "online")
    server_cfg = _cfg_section(online_cfg, "server")
    for key in ("buffer_dir", "drain_dir"):
        value = server_cfg.get(key)
        if isinstance(value, str) and value:
            dirs_to_create.add(Path(value))

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)


def build_independent_arm_config(
    base_config: dict[str, Any],
    *,
    runtime_root: str | Path,
    remote_host: str = "127.0.0.1",
    remote_port: int = 5000,
    experiment_profile: str = "default",
    opponent_pool_preset: str = "default",
    repro_seed: int | None = None,
    train_key: int | None = None,
    train_seed_start: int | None = None,
) -> dict[str, Any]:
    runtime_root = Path(runtime_root).resolve()
    config_dict = deepcopy(base_config)

    control_cfg = _cfg_section(config_dict, "control")
    control_cfg["online"] = True
    control_cfg["state_file"] = str((runtime_root / "checkpoints" / "mortal.pth").resolve())
    control_cfg["best_state_file"] = str((runtime_root / "checkpoints" / "best.pth").resolve())

    _set_shared_local_paths(config_dict, runtime_root)

    online_cfg = _cfg_section(config_dict, "online")
    remote_cfg = _cfg_section(online_cfg, "remote")
    remote_cfg["host"] = remote_host
    remote_cfg["port"] = int(remote_port)

    config_dict["online_machine_mode"] = {
        "mode": "independent_arm",
        "runtime_root": str(runtime_root),
        "remote_host": remote_host,
        "remote_port": int(remote_port),
    }
    config_dict = apply_experiment_profile(config_dict, experiment_profile)
    config_dict = apply_opponent_pool_preset(config_dict, opponent_pool_preset)
    apply_repro_overrides(
        config_dict,
        repro_seed=repro_seed,
        train_key=train_key,
        train_seed_start=train_seed_start,
    )
    _ensure_runtime_root_layout(config_dict, runtime_root)
    return config_dict


def build_worker_mode_config(
    base_config: dict[str, Any],
    *,
    runtime_root: str | Path,
    remote_host: str | None = None,
    remote_port: int | None = None,
    experiment_profile: str = "default",
    opponent_pool_preset: str = "default",
    repro_seed: int | None = None,
    train_key: int | None = None,
    train_seed_start: int | None = None,
) -> dict[str, Any]:
    runtime_root = Path(runtime_root).resolve()
    config_dict = deepcopy(base_config)

    control_cfg = _cfg_section(config_dict, "control")
    control_cfg["online"] = True

    _set_shared_local_paths(config_dict, runtime_root)

    online_cfg = _cfg_section(config_dict, "online")
    remote_cfg = _cfg_section(online_cfg, "remote")
    resolved_host = str(remote_host or remote_cfg.get("host", "127.0.0.1"))
    resolved_port = int(remote_port if remote_port is not None else remote_cfg.get("port", 5000))
    remote_cfg["host"] = resolved_host
    remote_cfg["port"] = resolved_port

    config_dict["online_machine_mode"] = {
        "mode": "worker",
        "runtime_root": str(runtime_root),
        "remote_host": resolved_host,
        "remote_port": resolved_port,
    }
    config_dict = apply_experiment_profile(config_dict, experiment_profile)
    config_dict = apply_opponent_pool_preset(config_dict, opponent_pool_preset)
    apply_repro_overrides(
        config_dict,
        repro_seed=repro_seed,
        train_key=train_key,
        train_seed_start=train_seed_start,
    )
    _ensure_runtime_root_layout(config_dict, runtime_root)
    return config_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a machine-scoped online config for independent-arm or worker mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("independent_arm", "worker"),
        required=True,
    )
    parser.add_argument(
        "--base-config",
        default=str((MORTAL_ROOT / "config.toml").resolve()),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-port", type=int, default=None)
    parser.add_argument(
        "--experiment-profile",
        choices=tuple(EXPERIMENT_PROFILES),
        default="default",
    )
    parser.add_argument(
        "--opponent-pool-preset",
        default="default",
        help="Opponent pool preset name. Built-ins: default, validation, formal.",
    )
    parser.add_argument("--repro-seed", type=int, default=None)
    parser.add_argument("--train-key", type=int, default=None)
    parser.add_argument("--train-seed-start", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_resolved_base_config(args.base_config)
    if args.mode == "independent_arm":
        config_dict = build_independent_arm_config(
            base_config,
            runtime_root=args.runtime_root,
            remote_host=args.remote_host or "127.0.0.1",
            remote_port=args.remote_port if args.remote_port is not None else 5000,
            experiment_profile=args.experiment_profile,
            opponent_pool_preset=args.opponent_pool_preset,
            repro_seed=args.repro_seed,
            train_key=args.train_key,
            train_seed_start=args.train_seed_start,
        )
    else:
        config_dict = build_worker_mode_config(
            base_config,
            runtime_root=args.runtime_root,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            experiment_profile=args.experiment_profile,
            opponent_pool_preset=args.opponent_pool_preset,
            repro_seed=args.repro_seed,
            train_key=args.train_key,
            train_seed_start=args.train_seed_start,
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_toml_file(output_path, config_dict)
    print(output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from mortal.eval.oracle_experiments import normalize_oracle_input_mode


DEFAULT_DEPENDENCY_EVAL_MODES = ("true", "zero", "shuffled")
DEFAULT_PT_RULE = (90, 45, 0, -135)


def _cfg_section(config_dict: Any, key: str) -> dict[str, Any]:
    if not isinstance(config_dict, dict):
        return {}
    section = config_dict.get(key, {})
    return section if isinstance(section, dict) else {}


def oracle_dependency_eval_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    return _cfg_section(config, "oracle_dependency_eval")


def oracle_dependency_eval_enabled(config: Mapping[str, Any]) -> bool:
    return bool(oracle_dependency_eval_cfg(config).get("enabled", False))


def oracle_dependency_eval_modes(config: Mapping[str, Any]) -> tuple[str, ...]:
    cfg = oracle_dependency_eval_cfg(config)
    raw_modes = cfg.get("modes")
    if not raw_modes:
        return DEFAULT_DEPENDENCY_EVAL_MODES
    if not isinstance(raw_modes, (list, tuple)):
        raise ValueError("oracle_dependency_eval.modes must be a list")
    return tuple(
        normalize_oracle_input_mode(mode, field_name="oracle_dependency_eval.modes")
        for mode in raw_modes
    )


def oracle_dependency_eval_log_dir(config: Mapping[str, Any]) -> str:
    cfg = oracle_dependency_eval_cfg(config)
    value = cfg.get("log_dir", "")
    return str(value or "").strip()


def summarize_stat(stat, *, pt_rule: Sequence[int] = DEFAULT_PT_RULE) -> dict[str, float]:
    return {
        "avg_rank": float(stat.avg_rank),
        "avg_pt": float(stat.avg_pt(list(pt_rule))),
        "rank_1_rate": float(stat.rank_1_rate),
        "rank_2_rate": float(stat.rank_2_rate),
        "rank_3_rate": float(stat.rank_3_rate),
        "rank_4_rate": float(stat.rank_4_rate),
        "agari_rate": float(stat.agari_rate),
        "houjuu_rate": float(stat.houjuu_rate),
        "riichi_rate": float(stat.riichi_rate),
        "fuuro_rate": float(stat.fuuro_rate),
        "avg_point_per_round": float(stat.avg_point_per_round),
    }


def evaluate_oracle_dependency_modes(
    test_player,
    mortal,
    dqn,
    device,
    *,
    seed_count: int,
    modes: Optional[Iterable[str]] = None,
    search_runtime_bundle=None,
    precomputed_zero: Optional[Mapping[str, float]] = None,
    pt_rule: Sequence[int] = DEFAULT_PT_RULE,
) -> dict[str, Any]:
    requested_modes = tuple(modes or DEFAULT_DEPENDENCY_EVAL_MODES)
    normalized_modes = tuple(
        normalize_oracle_input_mode(mode, field_name="oracle_dependency_eval.mode")
        for mode in requested_modes
    )
    results: dict[str, Any] = {
        "checkpoint_brain_is_oracle": bool(getattr(mortal, "is_oracle", False)),
        "modes": {},
    }

    for mode in normalized_modes:
        effective_runtime_is_oracle = bool(getattr(mortal, "is_oracle", False) and mode != "zero")
        if mode == "zero" and precomputed_zero is not None:
            summary = dict(precomputed_zero)
        else:
            stat = test_player.test_play(
                seed_count,
                mortal,
                dqn,
                device,
                search_runtime_bundle=search_runtime_bundle,
                oracle_input_mode=mode,
            )
            summary = summarize_stat(stat, pt_rule=pt_rule)
        summary["effective_runtime_is_oracle"] = float(effective_runtime_is_oracle)
        results["modes"][mode] = summary

    zero_summary = results["modes"].get("zero")
    if zero_summary is not None:
        for mode, summary in results["modes"].items():
            summary["delta_vs_zero_avg_pt"] = float(summary["avg_pt"] - zero_summary["avg_pt"])
            summary["delta_vs_zero_avg_rank"] = float(summary["avg_rank"] - zero_summary["avg_rank"])

    return results


def write_oracle_dependency_report(
    output_path: str,
    payload: Mapping[str, Any],
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


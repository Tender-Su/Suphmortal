from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:  # pragma: no cover - handled by CLI error path
    EventAccumulator = None

from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.core.config_utils import (
    deep_merge_dict as _deep_merge_dict,
    get_dict_section as _cfg_section,
)
from mortal.online.online_machine_modes import build_independent_arm_config, load_resolved_base_config


DEFAULT_TARGET_VALUE_TERM_SHARES = (0.15, 0.25, 0.35, 0.50)
DEFAULT_PROTOCOL_DECIDE_ORACLE_CRITIC_OPTIONS = (False, True)
DEFAULT_WINNER_REFINE_SCALE_FACTORS = (0.80, 1.00, 1.20)
DEFAULT_PROTOCOL_DECIDE_GAP_THRESHOLD = 0.5
WEIGHT_SLUG_SCALE = 1_000_000


def _round_weight(value: float) -> float:
    return round(float(value), 6)


def _weight_slug(value: float) -> str:
    scaled = int(round(float(value) * WEIGHT_SLUG_SCALE))
    return f"w{scaled:06d}"


def _candidate_name(prefix: str, *, oracle_critic: bool, value_weight: float) -> str:
    critic_tag = "oracle" if oracle_critic else "visible"
    return f"{prefix}_{critic_tag}_{_weight_slug(value_weight)}"


def _resolve_protocol_decide_oracle_options(
    mode: str,
    *,
    calibration_payload: dict[str, Any],
) -> list[bool]:
    normalized_mode = str(mode or "both").strip().lower()
    if normalized_mode == "oracle":
        return [True]
    if normalized_mode == "visible":
        return [False]
    if normalized_mode != "both":
        raise ValueError(
            f"unsupported oracle critic mode {mode!r}; expected one of ['both', 'oracle', 'visible']"
        )

    search_space = _cfg_section(calibration_payload, "search_space")
    resolved = [
        bool(item) for item in search_space.get(
            "protocol_decide_oracle_critic_options",
            DEFAULT_PROTOCOL_DECIDE_ORACLE_CRITIC_OPTIONS,
        )
    ]
    if not resolved:
        raise ValueError("protocol_decide oracle critic options resolved to an empty list")
    return list(dict.fromkeys(resolved))


def _primary_event_file(tb_root: Path) -> Path:
    event_files = sorted(
        tb_root.glob("events.out.tfevents.*"),
        key=lambda path: (path.stat().st_mtime_ns, path.stat().st_size),
    )
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event files found under {tb_root}")
    return event_files[-1]


def load_latest_run_scalars(run_root: Path, tags: list[str]) -> dict[str, float]:
    if EventAccumulator is None:
        raise RuntimeError(
            "TensorBoard is not installed in the current environment; cannot read run scalars"
        )
    tb_root = run_root / "tb_log"
    event_file = _primary_event_file(tb_root)
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    values: dict[str, float] = {}
    for tag in tags:
        if tag not in scalar_tags:
            continue
        series = accumulator.Scalars(tag)
        if series:
            values[tag] = float(series[-1].value)
    return values


def derive_weight_from_target_share(
    *,
    observed_total_loss: float,
    observed_value_loss: float,
    current_value_weight: float,
    target_share: float,
) -> float:
    target_share = float(target_share)
    if not 0.0 < target_share < 1.0:
        raise ValueError("target_share must be in (0, 1)")
    observed_value_loss = float(observed_value_loss)
    if observed_value_loss <= 0.0:
        return _round_weight(current_value_weight)
    weighted_value_term = float(current_value_weight) * observed_value_loss
    non_value_loss = max(float(observed_total_loss) - weighted_value_term, 0.0)
    denom = max((1.0 - target_share) * observed_value_loss, 1e-12)
    return _round_weight(target_share * non_value_loss / denom)


def derive_online_calibration(
    *,
    run_root: Path,
    target_value_term_shares: tuple[float, ...] = DEFAULT_TARGET_VALUE_TERM_SHARES,
) -> dict[str, Any]:
    config = load_toml_file(run_root / "config.toml")
    current_value_weight = float(_cfg_section(config, "value").get("weight", 0.0) or 0.0)
    scalars = load_latest_run_scalars(run_root, ["loss", "value_loss", "test_play/avg_pt"])
    observed_total_loss = float(scalars["loss"])
    observed_value_loss = float(scalars["value_loss"])
    weighted_value_term = current_value_weight * observed_value_loss
    non_value_loss = max(observed_total_loss - weighted_value_term, 0.0)
    current_share = (
        weighted_value_term / observed_total_loss if observed_total_loss > 0.0 else 0.0
    )
    candidate_weights = {_round_weight(current_value_weight)}
    for share in target_value_term_shares:
        candidate_weights.add(
            derive_weight_from_target_share(
                observed_total_loss=observed_total_loss,
                observed_value_loss=observed_value_loss,
                current_value_weight=current_value_weight,
                target_share=share,
            )
        )
    candidate_weights = {weight for weight in candidate_weights if weight > 0.0}
    return {
        "stage": "calibration",
        "calibration_mode": "loss_share_from_tensorboard",
        "source_run_root": str(run_root),
        "source_profile": _cfg_section(config, "online_experiment_profile").get("name"),
        "source_test_play_avg_pt": scalars.get("test_play/avg_pt"),
        "observed": {
            "loss": observed_total_loss,
            "value_loss": observed_value_loss,
            "current_value_weight": current_value_weight,
            "weighted_value_term": weighted_value_term,
            "estimated_non_value_loss": non_value_loss,
            "current_weighted_value_share": current_share,
        },
        "search_space": {
            "protocol_decide_oracle_critic_options": list(
                DEFAULT_PROTOCOL_DECIDE_ORACLE_CRITIC_OPTIONS
            ),
            "protocol_decide_value_weights": sorted(candidate_weights),
            "winner_refine_scale_factors": list(DEFAULT_WINNER_REFINE_SCALE_FACTORS),
            "protocol_decide_gap_threshold": DEFAULT_PROTOCOL_DECIDE_GAP_THRESHOLD,
        },
    }


def _online_candidate_override(
    *,
    value_weight: float,
    oracle_critic: bool,
    zero_sum_weight: float = 0.01,
) -> dict[str, Any]:
    return {
        "value": {
            "enabled": True,
            "weight": float(value_weight),
            "oracle_critic": bool(oracle_critic),
            "zero_sum_weight": float(zero_sum_weight),
        },
        "online_fidelity": {
            "candidate_kind": "value_weight",
            "oracle_critic": bool(oracle_critic),
            "value_weight": float(value_weight),
        },
    }


def build_protocol_decide_manifest(
    *,
    base_config_path: Path,
    runtime_root: Path,
    base_experiment_profile: str,
    calibration_payload: dict[str, Any],
    opponent_pool_preset: str = "validation",
    zero_sum_weight: float = 0.01,
    oracle_critic_mode: str = "both",
) -> dict[str, Any]:
    base_config = load_resolved_base_config(base_config_path)
    search_space = _cfg_section(calibration_payload, "search_space")
    value_weights = [
        _round_weight(weight)
        for weight in search_space.get("protocol_decide_value_weights", [])
        if float(weight) > 0.0
    ]
    oracle_options = _resolve_protocol_decide_oracle_options(
        oracle_critic_mode,
        calibration_payload=calibration_payload,
    )
    candidates = []
    for oracle_critic in oracle_options:
        for value_weight in value_weights:
            candidate_name = _candidate_name(
                "protocol_decide",
                oracle_critic=oracle_critic,
                value_weight=value_weight,
            )
            candidate_root = (runtime_root / candidate_name).resolve()
            config_dict = build_independent_arm_config(
                base_config,
                runtime_root=candidate_root,
                experiment_profile=base_experiment_profile,
                opponent_pool_preset=opponent_pool_preset,
            )
            _deep_merge_dict(
                config_dict,
                _online_candidate_override(
                    value_weight=value_weight,
                    oracle_critic=oracle_critic,
                    zero_sum_weight=zero_sum_weight,
                ),
            )
            config_path = candidate_root / "config.toml"
            write_toml_file(config_path, config_dict)
            candidates.append(
                {
                    "arm_name": candidate_name,
                    "config_path": str(config_path),
                    "runtime_root": str(candidate_root),
                    "base_experiment_profile": base_experiment_profile,
                    "opponent_pool_preset": opponent_pool_preset,
                    "oracle_critic": oracle_critic,
                    "value_weight": value_weight,
                }
            )
    return {
        "stage": "protocol_decide",
        "base_config_path": str(base_config_path),
        "base_experiment_profile": base_experiment_profile,
        "opponent_pool_preset": opponent_pool_preset,
        "oracle_critic_mode": str(oracle_critic_mode),
        "runtime_root": str(runtime_root),
        "source_calibration": calibration_payload,
        "candidates": candidates,
    }


def protocol_decide_ranking_from_results(
    results_payload: dict[str, Any],
    *,
    gap_threshold: float | None = None,
) -> dict[str, Any]:
    candidates = list(results_payload.get("results", []))
    if not candidates:
        raise ValueError("results payload must contain a non-empty results list")

    def ranking_key(entry: dict[str, Any]) -> tuple[float, float, str]:
        avg_pt = float(entry.get("formal_avg_pt", entry.get("avg_pt", float("-inf"))))
        avg_rank = float(entry.get("formal_avg_rank", entry.get("avg_rank", float("inf"))))
        return (-avg_pt, avg_rank, str(entry.get("arm_name", "")))

    ranking = sorted(candidates, key=ranking_key)
    winner = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    resolved_gap_threshold = (
        float(gap_threshold)
        if gap_threshold is not None
        else float(results_payload.get("gap_threshold", DEFAULT_PROTOCOL_DECIDE_GAP_THRESHOLD))
    )
    ambiguous = False
    detail = None
    if runner_up is not None:
        winner_pt = float(winner.get("formal_avg_pt", winner.get("avg_pt", float("-inf"))))
        runner_pt = float(runner_up.get("formal_avg_pt", runner_up.get("avg_pt", float("-inf"))))
        gap = winner_pt - runner_pt
        winner_stderr = abs(float(winner.get("pt_stderr", 0.0) or 0.0))
        runner_stderr = abs(float(runner_up.get("pt_stderr", 0.0) or 0.0))
        flipped = (winner_pt - winner_stderr) <= (runner_pt + runner_stderr)
        ambiguous = bool(flipped or gap <= resolved_gap_threshold)
        detail = {
            "winner": winner.get("arm_name"),
            "runner_up": runner_up.get("arm_name"),
            "avg_pt_gap": gap,
            "gap_threshold": resolved_gap_threshold,
            "winner_flipped_by_stderr": flipped,
            "winner_pt_stderr": winner_stderr,
            "runner_up_pt_stderr": runner_stderr,
        }
    return {
        "stage": "protocol_decide_ranking",
        "winner": winner,
        "ranking": ranking,
        "ambiguous": ambiguous,
        "ambiguity_detail": detail,
    }


def build_winner_refine_manifest(
    *,
    base_config_path: Path,
    runtime_root: Path,
    base_experiment_profile: str,
    protocol_ranking_payload: dict[str, Any],
    opponent_pool_preset: str = "validation",
    scale_factors: tuple[float, ...] = DEFAULT_WINNER_REFINE_SCALE_FACTORS,
    zero_sum_weight: float = 0.01,
) -> dict[str, Any]:
    winner = protocol_ranking_payload.get("winner", {})
    if not isinstance(winner, dict) or not winner:
        raise ValueError("protocol ranking payload is missing winner")
    base_weight = float(winner["value_weight"])
    oracle_critic = bool(winner["oracle_critic"])
    candidate_weights = sorted({
        _round_weight(base_weight * float(scale))
        for scale in scale_factors
        if base_weight * float(scale) > 0.0
    })
    base_config = load_resolved_base_config(base_config_path)
    candidates = []
    for value_weight in candidate_weights:
        candidate_name = _candidate_name(
            "winner_refine",
            oracle_critic=oracle_critic,
            value_weight=value_weight,
        )
        candidate_root = (runtime_root / candidate_name).resolve()
        config_dict = build_independent_arm_config(
            base_config,
            runtime_root=candidate_root,
            experiment_profile=base_experiment_profile,
            opponent_pool_preset=opponent_pool_preset,
        )
        _deep_merge_dict(
            config_dict,
            _online_candidate_override(
                value_weight=value_weight,
                oracle_critic=oracle_critic,
                zero_sum_weight=zero_sum_weight,
            ),
        )
        config_path = candidate_root / "config.toml"
        write_toml_file(config_path, config_dict)
        candidates.append(
            {
                "arm_name": candidate_name,
                "config_path": str(config_path),
                "runtime_root": str(candidate_root),
                "base_experiment_profile": base_experiment_profile,
                "opponent_pool_preset": opponent_pool_preset,
                "oracle_critic": oracle_critic,
                "value_weight": value_weight,
                "winner_scale_factor": _round_weight(value_weight / base_weight),
            }
        )
    return {
        "stage": "winner_refine",
        "base_config_path": str(base_config_path),
        "base_experiment_profile": base_experiment_profile,
        "opponent_pool_preset": opponent_pool_preset,
        "runtime_root": str(runtime_root),
        "source_protocol_decide": protocol_ranking_payload,
        "candidates": candidates,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RL online calibration / protocol_decide / winner_refine artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    calibration = sub.add_parser("calibration")
    calibration.add_argument("--source-run", required=True)
    calibration.add_argument("--output-json", required=True)

    protocol_decide = sub.add_parser("emit_protocol_decide")
    protocol_decide.add_argument("--base-config", required=True)
    protocol_decide.add_argument("--base-experiment-profile", required=True)
    protocol_decide.add_argument("--calibration-json", required=True)
    protocol_decide.add_argument("--runtime-root", required=True)
    protocol_decide.add_argument("--output-json", required=True)
    protocol_decide.add_argument("--opponent-pool-preset", default="validation")
    protocol_decide.add_argument(
        "--oracle-critic-mode",
        choices=("both", "oracle", "visible"),
        default="both",
    )

    rank_results = sub.add_parser("rank_protocol_decide")
    rank_results.add_argument("--results-json", required=True)
    rank_results.add_argument("--output-json", required=True)
    rank_results.add_argument("--gap-threshold", type=float, default=None)

    winner_refine = sub.add_parser("emit_winner_refine")
    winner_refine.add_argument("--base-config", required=True)
    winner_refine.add_argument("--base-experiment-profile", required=True)
    winner_refine.add_argument("--protocol-json", required=True)
    winner_refine.add_argument("--runtime-root", required=True)
    winner_refine.add_argument("--output-json", required=True)
    winner_refine.add_argument("--opponent-pool-preset", default="validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match args.command:
        case "calibration":
            payload = derive_online_calibration(run_root=Path(args.source_run).resolve())
            _write_json(Path(args.output_json).resolve(), payload)
        case "emit_protocol_decide":
            payload = build_protocol_decide_manifest(
                base_config_path=Path(args.base_config).resolve(),
                runtime_root=Path(args.runtime_root).resolve(),
                base_experiment_profile=str(args.base_experiment_profile),
                calibration_payload=_load_json(Path(args.calibration_json).resolve()),
                opponent_pool_preset=str(args.opponent_pool_preset),
                oracle_critic_mode=str(args.oracle_critic_mode),
            )
            _write_json(Path(args.output_json).resolve(), payload)
        case "rank_protocol_decide":
            payload = protocol_decide_ranking_from_results(
                _load_json(Path(args.results_json).resolve()),
                gap_threshold=args.gap_threshold,
            )
            _write_json(Path(args.output_json).resolve(), payload)
        case "emit_winner_refine":
            payload = build_winner_refine_manifest(
                base_config_path=Path(args.base_config).resolve(),
                runtime_root=Path(args.runtime_root).resolve(),
                base_experiment_profile=str(args.base_experiment_profile),
                protocol_ranking_payload=_load_json(Path(args.protocol_json).resolve()),
                opponent_pool_preset=str(args.opponent_pool_preset),
            )
            _write_json(Path(args.output_json).resolve(), payload)
        case _:
            raise RuntimeError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    main()

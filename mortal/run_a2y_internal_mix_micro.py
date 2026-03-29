from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import toml

import run_rank_shape_probe as rsp
import run_stage05_ab as ab
import run_stage05_fidelity as fidelity


PROTOCOL_ARM = "C_A2y_cosine_broad_to_recent_strong_12m_6m"
OPP_WEIGHT = 0.004
DANGER_WEIGHT = 0.009
RANK_SCALE = 0.10
STEP_SCALE = 0.20
SEED = ab.BASE_SCREENING["seed"]
SEED_OFFSETS = [0]


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = fidelity.FIDELITY_ROOT / f"s05_fidelity_a2y_internal_mix_micro1s_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_protocol() -> fidelity.CandidateSpec:
    return fidelity.CandidateSpec(
        arm_name=PROTOCOL_ARM,
        scheduler_profile="cosine",
        curriculum_profile="broad_to_recent",
        weight_profile="strong",
        window_profile="12m_6m",
        cfg_overrides={},
        meta={"stage": "P1_internal_mix_probe", "protocol_arm": PROTOCOL_ARM},
    )


def build_ce_only(protocol: fidelity.CandidateSpec) -> fidelity.CandidateSpec:
    cfg = ab.merge_dict(deepcopy(protocol.cfg_overrides), fidelity.build_rank_override(0.0))
    cfg = ab.merge_dict(cfg, fidelity.build_aux_override(opp_weight=0.0, danger_weight=0.0, danger_enabled=False))
    return fidelity.CandidateSpec(
        arm_name=f"{PROTOCOL_ARM}__CE_ONLY",
        scheduler_profile=protocol.scheduler_profile,
        curriculum_profile=protocol.curriculum_profile,
        weight_profile=protocol.weight_profile,
        window_profile=protocol.window_profile,
        cfg_overrides=cfg,
        meta={"stage": "ce_only", "protocol_arm": PROTOCOL_ARM},
    )


def build_opp_candidates(protocol: fidelity.CandidateSpec) -> list[fidelity.CandidateSpec]:
    specs = [
        ("OPP_EQ_CURRENT", 1.0, 1.0),
        ("OPP_18K_STAT", 2 * 0.45663374805593215, 2 * 0.543366251944068),
        ("OPP_HYBRID_GRAD", 2 * 0.4253284204036321, 2 * 0.5746715795963679),
    ]
    candidates: list[fidelity.CandidateSpec] = []
    for name, shanten_weight, tenpai_weight in specs:
        cfg = ab.merge_dict(deepcopy(protocol.cfg_overrides), fidelity.build_rank_override(0.0))
        cfg = ab.merge_dict(
            cfg,
            {
                "aux": {
                    "opponent_state_weight": OPP_WEIGHT,
                    "opponent_shanten_weight": shanten_weight,
                    "opponent_tenpai_weight": tenpai_weight,
                    "danger_enabled": False,
                    "danger_weight": 0.0,
                    "danger_any_weight": 0.45,
                    "danger_value_weight": 0.35,
                    "danger_player_weight": 0.20,
                    "danger_focal_gamma": 0.0,
                    "danger_ramp_steps": 1000,
                    "danger_value_cap": 96000.0,
                }
            },
        )
        candidates.append(
            fidelity.CandidateSpec(
                arm_name=f"{PROTOCOL_ARM}__{name}",
                scheduler_profile=protocol.scheduler_profile,
                curriculum_profile=protocol.curriculum_profile,
                weight_profile=protocol.weight_profile,
                window_profile=protocol.window_profile,
                cfg_overrides=cfg,
                meta={
                    "stage": "opp_internal_mix",
                    "protocol_arm": PROTOCOL_ARM,
                    "opponent_state_weight": OPP_WEIGHT,
                    "shanten_weight": shanten_weight,
                    "tenpai_weight": tenpai_weight,
                },
            )
        )
    return candidates


def build_rank_candidates(
    protocol: fidelity.CandidateSpec,
    *,
    version: int,
    full_recent_files: list[str],
    file_batch_size: int,
) -> tuple[list[fidelity.CandidateSpec], float]:
    target_mean = rsp.rank_weight_mean_for_files(
        full_recent_files,
        version=version,
        file_batch_size=file_batch_size,
        template={
            **fidelity.RANK_TEMPLATE,
            "base_weight": fidelity.RANK_TEMPLATE["base_weight"] * RANK_SCALE,
            "max_weight": fidelity.RANK_TEMPLATE["max_weight"] * RANK_SCALE,
        },
    )
    specs = [
        ("RANK_CURRENT", {"south_factor": 1.4, "all_last_factor": 1.8, "gap_focus_points": 4000.0, "gap_close_bonus": 1.5}),
        ("RANK_18K_ROUND_ONLY", {"south_factor": 1.590, "all_last_factor": 1.617, "gap_focus_points": 4000.0, "gap_close_bonus": 0.0}),
        ("RANK_18K_WIDE_GAP", {"south_factor": 1.590, "all_last_factor": 1.617, "gap_focus_points": 12000.0, "gap_close_bonus": 1.531}),
    ]
    candidates: list[fidelity.CandidateSpec] = []
    for name, templ in specs:
        template = {
            **templ,
            "base_weight": fidelity.RANK_TEMPLATE["base_weight"],
            "max_weight": fidelity.RANK_TEMPLATE["max_weight"],
        }
        template_mean = rsp.rank_weight_mean_for_files(
            full_recent_files,
            version=version,
            file_batch_size=file_batch_size,
            template=template,
        )
        scale = target_mean / template_mean
        cfg = ab.merge_dict(
            deepcopy(protocol.cfg_overrides),
            {
                "supervised": {
                    "rank_aux": {
                        "base_weight": template["base_weight"] * scale,
                        "south_factor": template["south_factor"],
                        "all_last_factor": template["all_last_factor"],
                        "gap_focus_points": template["gap_focus_points"],
                        "gap_close_bonus": template["gap_close_bonus"],
                        "max_weight": template["max_weight"] * scale,
                    }
                }
            },
        )
        cfg = ab.merge_dict(cfg, fidelity.build_aux_override(opp_weight=0.0, danger_weight=0.0, danger_enabled=False))
        candidates.append(
            fidelity.CandidateSpec(
                arm_name=f"{PROTOCOL_ARM}__{name}",
                scheduler_profile=protocol.scheduler_profile,
                curriculum_profile=protocol.curriculum_profile,
                weight_profile=protocol.weight_profile,
                window_profile=protocol.window_profile,
                cfg_overrides=cfg,
                meta={
                    "stage": "rank_shape_micro",
                    "protocol_arm": PROTOCOL_ARM,
                    "target_rank_weight_mean": target_mean,
                    "template_mean_before_scale": template_mean,
                    "rank_shape_scale": scale,
                    **templ,
                },
            )
        )
    return candidates, target_mean


def build_danger_candidates(protocol: fidelity.CandidateSpec) -> list[fidelity.CandidateSpec]:
    specs = [
        ("DANGER_CURRENT", 0.45, 0.35, 0.20),
        ("DANGER_18K_STAT", 0.09042179466099699, 0.8180402859274302, 0.09153791941157279),
        ("DANGER_HYBRID_GRAD", 0.04077534231271209, 0.9157590704571368, 0.04346558723015105),
    ]
    candidates: list[fidelity.CandidateSpec] = []
    for name, any_weight, value_weight, player_weight in specs:
        cfg = ab.merge_dict(deepcopy(protocol.cfg_overrides), fidelity.build_rank_override(0.0))
        cfg = ab.merge_dict(
            cfg,
            {
                "aux": {
                    "opponent_state_weight": 0.0,
                    "opponent_shanten_weight": 1.0,
                    "opponent_tenpai_weight": 1.0,
                    "danger_enabled": True,
                    "danger_weight": DANGER_WEIGHT,
                    "danger_any_weight": any_weight,
                    "danger_value_weight": value_weight,
                    "danger_player_weight": player_weight,
                    "danger_focal_gamma": 0.0,
                    "danger_ramp_steps": 1000,
                    "danger_value_cap": 96000.0,
                }
            },
        )
        candidates.append(
            fidelity.CandidateSpec(
                arm_name=f"{PROTOCOL_ARM}__{name}",
                scheduler_profile=protocol.scheduler_profile,
                curriculum_profile=protocol.curriculum_profile,
                weight_profile=protocol.weight_profile,
                window_profile=protocol.window_profile,
                cfg_overrides=cfg,
                meta={
                    "stage": "danger_internal_mix",
                    "protocol_arm": PROTOCOL_ARM,
                    "danger_weight": DANGER_WEIGHT,
                    "danger_any_weight": any_weight,
                    "danger_value_weight": value_weight,
                    "danger_player_weight": player_weight,
                },
            )
        )
    return candidates


def run_round(
    *,
    run_dir: Path,
    round_name: str,
    ab_name: str,
    base_cfg: dict,
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    candidates: list[fidelity.CandidateSpec],
) -> tuple[dict, list[dict]]:
    print(f"[round-start] {round_name} arms={len(candidates)}", flush=True)
    summary = fidelity.execute_round_multiseed(
        run_dir=run_dir,
        round_name=round_name,
        ab_name=ab_name,
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=candidates,
        seed=SEED,
        seed_offsets=SEED_OFFSETS,
        step_scale=STEP_SCALE,
        selector_weights=fidelity.ACTION_SCORE_WEIGHTS,
        ranking_mode="policy_quality",
        eligibility_group_key="protocol_arm",
    )
    concise = [
        {
            "arm_name": row.get("arm_name"),
            "full_recent_loss": row.get("full_recent_loss"),
            "selection_quality_score": row.get("selection_quality_score"),
            "recent_policy_loss": row.get("recent_policy_loss"),
            "valid": row.get("valid"),
        }
        for row in summary.get("ranking", [])[:4]
    ]
    print(f"[round-done] {round_name}", flush=True)
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)
    return summary, concise


def attach_baseline_deltas(
    rows: list[dict],
    *,
    base_policy_loss: float | None,
    base_full_loss: float | None,
    base_sel: float | None,
) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        policy_loss = row.get("recent_policy_loss")
        loss = row.get("full_recent_loss")
        sel = row.get("selection_quality_score")
        if base_policy_loss is not None and policy_loss is not None:
            item["delta_vs_ce_policy_loss"] = policy_loss - base_policy_loss
        if base_full_loss is not None and loss is not None:
            item["delta_vs_ce_full_loss"] = loss - base_full_loss
        if base_sel is not None and sel is not None:
            item["delta_vs_ce_selection"] = sel - base_sel
        out.append(item)
    return out


def main() -> None:
    run_dir = make_run_dir()
    print(f"[start] run_dir={run_dir}", flush=True)

    base_cfg = toml.load(ab.BASE_CFG_PATH)
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(grouped, SEED, ab.BASE_SCREENING["eval_files"])
    version = int(base_cfg["control"]["version"])
    file_batch_size = int(ab.BASE_SCREENING["file_batch_size"])

    protocol = build_protocol()
    baseline_candidate = build_ce_only(protocol)
    opp_candidates = build_opp_candidates(protocol)
    rank_candidates, rank_target_mean = build_rank_candidates(
        protocol,
        version=version,
        full_recent_files=eval_splits["full_recent_files"],
        file_batch_size=file_batch_size,
    )
    danger_candidates = build_danger_candidates(protocol)

    plan = {
        "protocol_arm": PROTOCOL_ARM,
        "step_scale": STEP_SCALE,
        "seed": SEED,
        "seed_offsets": SEED_OFFSETS,
        "opp_weight": OPP_WEIGHT,
        "danger_weight": DANGER_WEIGHT,
        "rank_scale_target": RANK_SCALE,
        "rank_target_mean": rank_target_mean,
        "baseline": baseline_candidate.meta | {"arm_name": baseline_candidate.arm_name},
        "groups": {
            "opp_internal_mix_round": [c.meta | {"arm_name": c.arm_name} for c in opp_candidates],
            "rank_shape_round": [c.meta | {"arm_name": c.arm_name} for c in rank_candidates],
            "danger_internal_mix_round": [c.meta | {"arm_name": c.arm_name} for c in danger_candidates],
        },
    }
    (run_dir / "a2y_internal_mix_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_summary, baseline_rows = run_round(
        run_dir=run_dir,
        round_name="baseline_round",
        ab_name="s05_a2y_baseline_micro",
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=[baseline_candidate],
    )
    baseline_entry = baseline_summary["ranking"][0] if baseline_summary.get("ranking") else {}
    base_policy_loss = baseline_entry.get("comparison_recent_loss", baseline_entry.get("recent_policy_loss"))
    base_full_loss = baseline_entry.get("full_recent_loss")
    base_sel = baseline_entry.get("selection_quality_score")

    final = {"baseline_round": baseline_rows}
    for round_name, ab_name, candidates in [
        ("opp_internal_mix_round", "s05_a2y_opp_internal_mix_micro", opp_candidates),
        ("rank_shape_round", "s05_a2y_rank_shape_micro", rank_candidates),
        ("danger_internal_mix_round", "s05_a2y_danger_internal_mix_micro", danger_candidates),
    ]:
        _, rows = run_round(
            run_dir=run_dir,
            round_name=round_name,
            ab_name=ab_name,
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=eval_splits,
            candidates=candidates,
        )
        final[round_name] = attach_baseline_deltas(
            rows,
            base_policy_loss=base_policy_loss,
            base_full_loss=base_full_loss,
            base_sel=base_sel,
        )

    (run_dir / "final_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", flush=True)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

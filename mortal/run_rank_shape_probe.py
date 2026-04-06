from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import run_sl_ab as ab
import run_sl_fidelity as fidelity
from dataloader import SupervisedFileDatasetsIter


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIDELITY_RUN_DIR = REPO_ROOT / "logs" / "sl_fidelity" / "sl_fidelity_main"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "logs" / "rank_shape_probe_plan.json"

DEFAULT_PROTOCOL_ARM = "C_B2z_cosine_recent_broad_recent_strong_6m_6m"
DEFAULT_AB_NAME = "sl_rank_shape_probe"

RANK_SHAPE_TEMPLATES = [
    {
        "name": "T0_current",
        "south_factor": 1.59,
        "all_last_factor": 1.617,
        "gap_focus_points": 4000.0,
        "gap_close_bonus": 0.0,
        "note": "current frozen default from the 2026-03-25 A2y shape freeze",
    },
    {
        "name": "T1_legacy_heuristic",
        "south_factor": 1.40,
        "all_last_factor": 1.80,
        "gap_focus_points": 4000.0,
        "gap_close_bonus": 1.50,
        "note": "legacy heuristic baseline kept only for historical comparison",
    },
    {
        "name": "T2_proxy_round_plus_current_gap",
        "south_factor": 1.59,
        "all_last_factor": 1.617,
        "gap_focus_points": 4000.0,
        "gap_close_bonus": 1.50,
        "note": "frozen round/all-last factors plus the legacy gap bonus",
    },
    {
        "name": "T3_proxy_round_plus_wide_gap",
        "south_factor": 1.59,
        "all_last_factor": 1.617,
        "gap_focus_points": 12000.0,
        "gap_close_bonus": 1.531,
        "note": "replace south/all-last and use change-rate gap proxy",
    },
    {
        "name": "T4_no_gap_legacy_round",
        "south_factor": 1.40,
        "all_last_factor": 1.80,
        "gap_focus_points": 4000.0,
        "gap_close_bonus": 0.0,
        "note": "ablate gap bonus under the legacy round factors",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_p0_round2_ranking(run_dir: Path) -> list[dict[str, Any]]:
    round2_path = run_dir / "p0_round2.json"
    if round2_path.exists():
        return load_json(round2_path).get("ranking", [])

    state_path = run_dir / "state.json"
    if state_path.exists():
        state = load_json(state_path)
        ranking = state.get("p0", {}).get("round2", {}).get("ranking", [])
        if ranking:
            return ranking

    raise FileNotFoundError(f"could not find p0 round2 ranking under {run_dir}")


def rank_weight_mean_for_files(files: list[str], *, version: int, file_batch_size: int, template: dict[str, Any]) -> float:
    dataset = SupervisedFileDatasetsIter(
        version=version,
        file_list=list(files),
        oracle=False,
        file_batch_size=file_batch_size,
        reserve_ratio=0.0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        shuffle_files=False,
        worker_torch_num_threads=1,
        worker_torch_num_interop_threads=1,
        rayon_num_threads=0,
        emit_opponent_state_labels=False,
        track_danger_labels=False,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=4096,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    total_weight = 0.0
    total_samples = 0
    for batch in loader:
        context_meta = torch.as_tensor(batch[4], dtype=torch.int64)
        weights = torch.full(
            (context_meta.shape[0],),
            float(template["base_weight"]),
            dtype=torch.float32,
        )
        south_mask = context_meta[:, 1] == 1
        weights = weights * torch.where(
            south_mask,
            torch.full_like(weights, float(template["south_factor"])),
            torch.ones_like(weights),
        )
        all_last_mask = context_meta[:, 3].to(torch.bool)
        weights = weights * torch.where(
            all_last_mask,
            torch.full_like(weights, float(template["all_last_factor"])),
            torch.ones_like(weights),
        )
        nearest_gap = torch.minimum(context_meta[:, 6], context_meta[:, 7]).to(torch.float32) * 100.0
        closeness = (
            1.0 - nearest_gap / float(template["gap_focus_points"])
            if float(template["gap_focus_points"]) > 0 else torch.zeros_like(nearest_gap)
        )
        closeness = closeness.clamp(min=0.0, max=1.0)
        weights = weights * (1.0 + float(template["gap_close_bonus"]) * closeness)
        weights = weights.clamp(max=float(template["max_weight"]))
        total_weight += float(weights.sum().item())
        total_samples += int(context_meta.shape[0])
    if total_samples <= 0:
        raise RuntimeError("rank_shape_probe found zero samples when estimating template mean")
    return total_weight / total_samples


def choose_protocol_candidate(run_dir: Path, protocol_arm: str) -> fidelity.CandidateSpec:
    for entry in load_p0_round2_ranking(run_dir):
        if entry.get("arm_name") == protocol_arm:
            return fidelity.candidate_from_entry(entry)
    raise ValueError(f"protocol arm not found in p0 round2 ranking: {protocol_arm}")


def build_rank_shape_candidates(
    protocol: fidelity.CandidateSpec,
    *,
    target_mean: float,
    version: int,
    full_recent_files: list[str],
    file_batch_size: int,
) -> list[fidelity.CandidateSpec]:
    candidates: list[fidelity.CandidateSpec] = []
    for template in RANK_SHAPE_TEMPLATES:
        templ = {
            **template,
            "base_weight": fidelity.RANK_TEMPLATE["base_weight"],
            "max_weight": fidelity.RANK_TEMPLATE["max_weight"],
        }
        template_mean = rank_weight_mean_for_files(
            full_recent_files,
            version=version,
            file_batch_size=file_batch_size,
            template=templ,
        )
        scale = target_mean / template_mean if template_mean > 0 else 1.0
        cfg_overrides = ab.merge_dict(
            deepcopy(protocol.cfg_overrides),
            {
                "supervised": {
                    "rank_aux": {
                        "base_weight": templ["base_weight"] * scale,
                        "south_factor": templ["south_factor"],
                        "all_last_factor": templ["all_last_factor"],
                        "gap_focus_points": templ["gap_focus_points"],
                        "gap_close_bonus": templ["gap_close_bonus"],
                        "max_weight": templ["max_weight"] * scale,
                    }
                },
                "aux": {
                    "opponent_state_weight": 0.0,
                    "danger_enabled": False,
                    "danger_weight": 0.0,
                },
            },
        )
        arm_name = f"{protocol.arm_name}__RSHAPE_{template['name']}"
        meta = {
            "stage": "rank_shape_probe",
            "protocol_arm": protocol.arm_name,
            "template_name": template["name"],
            "template_note": template["note"],
            "template_mean_before_scale": template_mean,
            "target_rank_weight_mean": target_mean,
            "rank_shape_scale": scale,
            "south_factor": templ["south_factor"],
            "all_last_factor": templ["all_last_factor"],
            "gap_focus_points": templ["gap_focus_points"],
            "gap_close_bonus": templ["gap_close_bonus"],
        }
        candidates.append(
            fidelity.CandidateSpec(
                arm_name=arm_name,
                scheduler_profile=protocol.scheduler_profile,
                curriculum_profile=protocol.curriculum_profile,
                weight_profile=protocol.weight_profile,
                window_profile=protocol.window_profile,
                cfg_overrides=cfg_overrides,
                meta=meta,
            )
        )
    return candidates


def build_plan(run_dir: Path, protocol_arm: str, *, step_scale: float) -> dict[str, Any]:
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(grouped, ab.BASE_SCREENING["seed"], ab.BASE_SCREENING["eval_files"])
    version = int(base_cfg["control"]["version"])
    target_mean = rank_weight_mean_for_files(
        eval_splits["full_recent_files"],
        version=version,
        file_batch_size=int(ab.BASE_SCREENING["file_batch_size"]),
        template={
            **fidelity.RANK_TEMPLATE,
            "base_weight": fidelity.RANK_TEMPLATE["base_weight"],
            "max_weight": fidelity.RANK_TEMPLATE["max_weight"],
        },
    )
    protocol = choose_protocol_candidate(run_dir, protocol_arm)
    candidates = build_rank_shape_candidates(
        protocol,
        target_mean=target_mean,
        version=version,
        full_recent_files=eval_splits["full_recent_files"],
        file_batch_size=int(ab.BASE_SCREENING["file_batch_size"]),
    )
    return {
        "run_dir": str(run_dir),
        "protocol_arm": protocol_arm,
        "step_scale": step_scale,
        "target_rank_weight_mean": target_mean,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "arm_name": candidate.arm_name,
                "meta": candidate.meta,
                "scheduler_profile": candidate.scheduler_profile,
                "curriculum_profile": candidate.curriculum_profile,
                "weight_profile": candidate.weight_profile,
                "window_profile": candidate.window_profile,
            }
            for candidate in candidates
        ],
    }, base_cfg, grouped, eval_splits, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_FIDELITY_RUN_DIR)
    parser.add_argument("--protocol-arm", type=str, default=DEFAULT_PROTOCOL_ARM)
    parser.add_argument("--ab-name", type=str, default=DEFAULT_AB_NAME)
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    plan, base_cfg, grouped, eval_splits, candidates = build_plan(
        args.run_dir,
        args.protocol_arm,
        step_scale=args.step_scale,
    )
    args.output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_json)

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    summary = fidelity.execute_round(
        run_dir=args.run_dir,
        round_name="rank_shape_probe",
        ab_name=args.ab_name,
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        candidates=candidates,
        seed=ab.BASE_SCREENING["seed"] + 4040,
        step_scale=args.step_scale,
        selector_weights=fidelity.ACTION_SCORE_WEIGHTS,
        ranking_mode="policy_quality",
        eligibility_group_key=None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

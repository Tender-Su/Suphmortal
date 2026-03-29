from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from libriichi.dataset import GameplayLoader
from model import AuxNet, Brain, CategoricalPolicy, DangerAuxNet, OpponentStateAuxNet


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_PATHS = REPO_ROOT / "logs" / "turn_weight_sample_paths.txt"
DEFAULT_FIDELITY_DIR = REPO_ROOT / "logs" / "stage05_fidelity" / "s05_fidelity_main"
DEFAULT_AB_ROOT = REPO_ROOT / "logs" / "stage05_ab"
DEFAULT_HEURISTIC_AUDIT = REPO_ROOT / "logs" / "aux_heuristic_audit.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "logs" / "aux_subhead_gradient_audit.json"
DEFAULT_REPORT_MD = REPO_ROOT / "logs" / "aux_subhead_gradient_audit.md"

DANGER_VALUE_CAP = 96000.0
DANGER_VALUE_CAP_LOG = math.log1p(DANGER_VALUE_CAP)


@dataclass(frozen=True)
class ArmRecord:
    family: str
    arm_name: str
    protocol_arm: str
    round_file: str
    ab_name: str
    arm_result_path: Path
    checkpoint_path: Path
    full_recent_loss: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_or_none(values: list[float]) -> float | None:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return median(positives)


def balanced_bce_per_sample_with_logits(logits: torch.Tensor, targets: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        reduction="none",
    )
    eligible_f = eligible.to(dtype=loss.dtype)
    pos_mask = (targets & eligible).to(dtype=loss.dtype)
    neg_mask = ((~targets) & eligible).to(dtype=loss.dtype)
    pos_count = pos_mask.sum(dim=tuple(range(1, pos_mask.dim())), keepdim=True)
    neg_count = neg_mask.sum(dim=tuple(range(1, neg_mask.dim())), keepdim=True)
    pos_weight = torch.where(pos_count > 0, 0.5 / pos_count, torch.zeros_like(pos_count))
    neg_weight = torch.where(neg_count > 0, 0.5 / neg_count, torch.zeros_like(neg_count))
    weights = pos_mask * pos_weight + neg_mask * neg_weight
    weighted = loss * weights * eligible_f
    return weighted.sum(dim=tuple(range(1, weighted.dim())))


def grad_rms(grad_tensor: torch.Tensor | None) -> float:
    if grad_tensor is None:
        return 0.0
    flat = grad_tensor.detach().float().reshape(-1)
    if flat.numel() <= 0:
        return 0.0
    return float(flat.square().mean().sqrt().item())


def normalize_weights_from_inverse(scale_by_task: dict[str, float]) -> dict[str, float]:
    inverse = {}
    for name, scale in scale_by_task.items():
        inverse[name] = 0.0 if scale <= 0 else 1.0 / scale
    total = sum(inverse.values())
    if total <= 0:
        return {name: 0.0 for name in scale_by_task}
    return {name: value / total for name, value in inverse.items()}


def hybrid_geomean_weights(loss_scale: dict[str, float], grad_scale: dict[str, float]) -> dict[str, float]:
    loss_weights = normalize_weights_from_inverse(loss_scale)
    grad_weights = normalize_weights_from_inverse(grad_scale)
    blended = {}
    for name in loss_weights:
        lw = max(loss_weights.get(name, 0.0), 0.0)
        gw = max(grad_weights.get(name, 0.0), 0.0)
        blended[name] = math.sqrt(lw * gw) if lw > 0 and gw > 0 else 0.0
    total = sum(blended.values())
    if total <= 0:
        return {name: 0.0 for name in blended}
    return {name: value / total for name, value in blended.items()}


def collect_arm_records(
    fidelity_dir: Path,
    ab_root: Path,
    *,
    family: str,
    keep_per_family: int,
) -> list[ArmRecord]:
    candidates_by_protocol: dict[str, list[ArmRecord]] = {}
    # Historical analysis script: family-specific records mostly come from the
    # old single-head rounds. Current mainline P1 no longer uses those rounds
    # as the default decision path.
    for round_path in sorted(fidelity_dir.glob("p1_solo_round__*.json")):
        round_payload = load_json(round_path)
        ab_name = str(round_payload.get("ab_name", ""))
        for entry in round_payload.get("ranking", []):
            meta = entry.get("candidate_meta", {})
            if meta.get("aux_family") != family or not entry.get("valid", False):
                continue
            protocol_arm = str(meta.get("protocol_arm", ""))
            arm_name = str(entry.get("arm_name", ""))
            if not protocol_arm or not arm_name:
                continue
            arm_result_path = ab_root / ab_name / arm_name / "arm_result.json"
            if not arm_result_path.exists():
                continue
            arm_payload = load_json(arm_result_path)
            if not arm_payload.get("ok", False):
                continue
            checkpoint_path = resolve_best_loss_checkpoint(arm_payload)
            if checkpoint_path is None or not checkpoint_path.exists():
                continue
            record = ArmRecord(
                family=family,
                arm_name=arm_name,
                protocol_arm=protocol_arm,
                round_file=round_path.name,
                ab_name=ab_name,
                arm_result_path=arm_result_path,
                checkpoint_path=checkpoint_path,
                full_recent_loss=float(entry.get("full_recent_loss", float("inf"))),
            )
            candidates_by_protocol.setdefault(protocol_arm, []).append(record)

    selected: list[ArmRecord] = []
    for protocol_arm, records in sorted(candidates_by_protocol.items()):
        best = min(records, key=lambda row: row.full_recent_loss)
        selected.append(best)
    selected.sort(key=lambda row: row.full_recent_loss)
    return selected[:keep_per_family]


def resolve_best_loss_checkpoint(arm_payload: dict[str, Any]) -> Path | None:
    run = arm_payload.get("run", {})
    final = run.get("final", {})
    best_loss = final.get("best_loss", {})
    path = best_loss.get("path")
    if isinstance(path, str) and path:
        return Path(path)
    phase_results = run.get("phase_results", {})
    if not phase_results:
        return None
    final_phase_name = run.get("phase_order", [])[-1]
    final_phase = phase_results.get(final_phase_name, {})
    path = final_phase.get("best_loss", {}).get("path")
    if isinstance(path, str) and path:
        return Path(path)
    return None


def load_sample_paths(sample_paths_file: Path, max_files: int) -> list[str]:
    paths = [line.strip() for line in sample_paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_files > 0:
        return paths[:max_files]
    return paths


def iter_minibatches(
    paths: list[str],
    *,
    file_batch_size: int,
    mini_batch_size: int,
) -> Any:
    loader = GameplayLoader(version=4, oracle=False, track_opponent_states=True, track_danger_labels=True)
    total_files = len(paths)
    for file_start in range(0, total_files, file_batch_size):
        batch_paths = paths[file_start:file_start + file_batch_size]
        print(
            f"[grad-audit] loading files {file_start + 1}-{file_start + len(batch_paths)} / {total_files}",
            flush=True,
        )
        nested = loader.load_log_files(batch_paths)
        for games in nested:
            for game in games:
                obs = np.asarray(game.take_obs_batch(), dtype=np.float32)
                masks = np.asarray(game.take_masks_batch(), dtype=bool)
                actions = np.asarray(game.take_actions_batch(), dtype=np.int64)
                player_rank = np.full((obs.shape[0],), int(game.take_grp().take_rank_by_player()[int(game.take_player_id())]), dtype=np.int64)
                opponent_shanten = np.asarray(game.take_opponent_shanten_batch(), dtype=np.int64)
                opponent_tenpai = np.asarray(game.take_opponent_tenpai_batch(), dtype=np.int64)
                danger_valid = np.asarray(game.take_danger_valid_batch(), dtype=bool)
                danger_any = np.asarray(game.take_danger_any_batch(), dtype=bool)
                danger_value = np.asarray(game.take_danger_value_batch(), dtype=np.float32)
                danger_player_mask = np.asarray(game.take_danger_player_mask_batch(), dtype=bool)

                for start in range(0, obs.shape[0], mini_batch_size):
                    end = start + mini_batch_size
                    yield {
                        "obs": torch.from_numpy(obs[start:end]),
                        "masks": torch.from_numpy(masks[start:end]),
                        "actions": torch.from_numpy(actions[start:end]),
                        "player_rank": torch.from_numpy(player_rank[start:end]),
                        "opponent_shanten": torch.from_numpy(opponent_shanten[start:end]),
                        "opponent_tenpai": torch.from_numpy(opponent_tenpai[start:end]),
                        "danger_valid": torch.from_numpy(danger_valid[start:end]),
                        "danger_any": torch.from_numpy(danger_any[start:end]),
                        "danger_value": torch.from_numpy(danger_value[start:end]),
                        "danger_player_mask": torch.from_numpy(danger_player_mask[start:end]),
                    }


def load_models(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)
    saved_cfg = state["config"]
    version = int(saved_cfg["control"]["version"])
    mortal = Brain(version=version, is_oracle=False, **saved_cfg["resnet"], Norm="GN").to(device)
    policy_net = CategoricalPolicy().to(device)
    aux_net = AuxNet(dims=(4,)).to(device)
    mortal.load_state_dict(state["mortal"])
    policy_net.load_state_dict(state["policy_net"])
    aux_net.load_state_dict(state["aux_net"])

    opponent_aux_net = None
    if state.get("opponent_aux_net") is not None:
        opponent_aux_net = OpponentStateAuxNet().to(device)
        opponent_aux_net.load_state_dict(state["opponent_aux_net"])

    danger_aux_net = None
    if state.get("danger_aux_net") is not None:
        danger_aux_net = DangerAuxNet().to(device)
        danger_aux_net.load_state_dict(state["danger_aux_net"])

    mortal.eval()
    policy_net.eval()
    aux_net.eval()
    if opponent_aux_net is not None:
        opponent_aux_net.eval()
    if danger_aux_net is not None:
        danger_aux_net.eval()

    return {
        "mortal": mortal,
        "policy_net": policy_net,
        "aux_net": aux_net,
        "opponent_aux_net": opponent_aux_net,
        "danger_aux_net": danger_aux_net,
    }


def audit_opponent_family(
    arm_records: list[ArmRecord],
    sample_paths: list[str],
    *,
    device: torch.device,
    file_batch_size: int,
    mini_batch_size: int,
    max_minibatches: int,
) -> dict[str, Any]:
    per_arm = []
    for arm_idx, record in enumerate(arm_records, start=1):
        print(f"[grad-audit][opp] arm {arm_idx}/{len(arm_records)} {record.arm_name}", flush=True)
        models = load_models(record.checkpoint_path, device)
        if models["opponent_aux_net"] is None:
            continue

        losses = {"shanten": [], "tenpai": [], "policy": []}
        grads = {"shanten": [], "tenpai": [], "policy": []}

        for batch_idx, batch in enumerate(iter_minibatches(sample_paths, file_batch_size=file_batch_size, mini_batch_size=mini_batch_size), start=1):
            if max_minibatches > 0 and batch_idx > max_minibatches:
                break
            tensors = {name: tensor.to(device=device, non_blocking=True) for name, tensor in batch.items()}

            obs = tensors["obs"].float()
            masks = tensors["masks"].bool()
            actions = tensors["actions"].long()
            opponent_shanten = tensors["opponent_shanten"].long()
            opponent_tenpai = tensors["opponent_tenpai"].long()

            phi = models["mortal"](obs)
            policy_logits = models["policy_net"].logits(phi, masks)
            policy_loss = F.cross_entropy(policy_logits, actions)
            shanten_logits, tenpai_logits = models["opponent_aux_net"](phi)
            shanten_loss = torch.stack(
                [
                    F.cross_entropy(shanten_logits[idx], opponent_shanten[:, idx], reduction="none")
                    for idx in range(3)
                ]
            ).mean()
            tenpai_loss = torch.stack(
                [
                    F.cross_entropy(tenpai_logits[idx], opponent_tenpai[:, idx], reduction="none")
                    for idx in range(3)
                ]
            ).mean()

            grad_policy = torch.autograd.grad(policy_loss, phi, retain_graph=True)[0]
            grad_shanten = torch.autograd.grad(shanten_loss, phi, retain_graph=True)[0]
            grad_tenpai = torch.autograd.grad(tenpai_loss, phi)[0]

            losses["policy"].append(float(policy_loss.item()))
            losses["shanten"].append(float(shanten_loss.item()))
            losses["tenpai"].append(float(tenpai_loss.item()))
            grads["policy"].append(grad_rms(grad_policy))
            grads["shanten"].append(grad_rms(grad_shanten))
            grads["tenpai"].append(grad_rms(grad_tenpai))

        per_arm.append(
            {
                "arm_name": record.arm_name,
                "protocol_arm": record.protocol_arm,
                "checkpoint_path": str(record.checkpoint_path),
                "median_policy_loss": positive_or_none(losses["policy"]),
                "median_shanten_loss": positive_or_none(losses["shanten"]),
                "median_tenpai_loss": positive_or_none(losses["tenpai"]),
                "median_policy_phi_grad_rms": positive_or_none(grads["policy"]),
                "median_shanten_phi_grad_rms": positive_or_none(grads["shanten"]),
                "median_tenpai_phi_grad_rms": positive_or_none(grads["tenpai"]),
                "mini_batches": len(losses["policy"]),
            }
        )

    aggregate = {
        "policy_phi_grad_rms": positive_or_none([row["median_policy_phi_grad_rms"] for row in per_arm]),
        "shanten_phi_grad_rms": positive_or_none([row["median_shanten_phi_grad_rms"] for row in per_arm]),
        "tenpai_phi_grad_rms": positive_or_none([row["median_tenpai_phi_grad_rms"] for row in per_arm]),
        "shanten_loss": positive_or_none([row["median_shanten_loss"] for row in per_arm]),
        "tenpai_loss": positive_or_none([row["median_tenpai_loss"] for row in per_arm]),
        "arms": len(per_arm),
    }
    return {
        "per_arm": per_arm,
        "aggregate": aggregate,
    }


def audit_danger_family(
    arm_records: list[ArmRecord],
    sample_paths: list[str],
    *,
    device: torch.device,
    file_batch_size: int,
    mini_batch_size: int,
    max_minibatches: int,
) -> dict[str, Any]:
    per_arm = []
    for arm_idx, record in enumerate(arm_records, start=1):
        print(f"[grad-audit][danger] arm {arm_idx}/{len(arm_records)} {record.arm_name}", flush=True)
        models = load_models(record.checkpoint_path, device)
        if models["danger_aux_net"] is None:
            continue

        losses = {"any": [], "value": [], "player": [], "policy": []}
        grads = {"any": [], "value": [], "player": [], "policy": []}

        for batch_idx, batch in enumerate(iter_minibatches(sample_paths, file_batch_size=file_batch_size, mini_batch_size=mini_batch_size), start=1):
            if max_minibatches > 0 and batch_idx > max_minibatches:
                break
            tensors = {name: tensor.to(device=device, non_blocking=True) for name, tensor in batch.items()}

            obs = tensors["obs"].float()
            masks = tensors["masks"].bool()
            actions = tensors["actions"].long()
            danger_valid = tensors["danger_valid"].bool()
            danger_any = tensors["danger_any"].bool()
            danger_value = tensors["danger_value"].float()
            danger_player_mask = tensors["danger_player_mask"].bool()

            phi = models["mortal"](obs)
            policy_logits = models["policy_net"].logits(phi, masks)
            policy_loss = F.cross_entropy(policy_logits, actions)
            any_logits, value_pred, player_logits = models["danger_aux_net"](phi)

            eligible = danger_valid.unsqueeze(-1) & masks[:, :37]
            any_loss = balanced_bce_per_sample_with_logits(any_logits, danger_any, eligible).mean()

            eligible_player = eligible.unsqueeze(-1).expand_as(danger_player_mask)
            player_loss = balanced_bce_per_sample_with_logits(player_logits, danger_player_mask, eligible_player).mean()

            value_positive = eligible & danger_any
            normalized_target = torch.log1p(danger_value.clamp(min=0.0, max=DANGER_VALUE_CAP)) / DANGER_VALUE_CAP_LOG
            value_loss_map = F.smooth_l1_loss(value_pred.sigmoid(), normalized_target, reduction="none")
            value_positive_weight = value_positive.to(dtype=value_loss_map.dtype)
            positive_count = value_positive_weight.sum(dim=1)
            value_loss_vec = (value_loss_map * value_positive_weight).sum(dim=1) / positive_count.clamp_min(1.0)
            value_loss_vec = torch.where(positive_count > 0, value_loss_vec, torch.zeros_like(value_loss_vec))
            value_loss = value_loss_vec.mean()

            grad_policy = torch.autograd.grad(policy_loss, phi, retain_graph=True)[0]
            grad_any = torch.autograd.grad(any_loss, phi, retain_graph=True)[0]
            grad_value = torch.autograd.grad(value_loss, phi, retain_graph=True, allow_unused=True)[0]
            grad_player = torch.autograd.grad(player_loss, phi, allow_unused=True)[0]

            losses["policy"].append(float(policy_loss.item()))
            losses["any"].append(float(any_loss.item()))
            losses["value"].append(float(value_loss.item()))
            losses["player"].append(float(player_loss.item()))
            grads["policy"].append(grad_rms(grad_policy))
            grads["any"].append(grad_rms(grad_any))
            grads["value"].append(grad_rms(grad_value))
            grads["player"].append(grad_rms(grad_player))

        per_arm.append(
            {
                "arm_name": record.arm_name,
                "protocol_arm": record.protocol_arm,
                "checkpoint_path": str(record.checkpoint_path),
                "median_policy_loss": positive_or_none(losses["policy"]),
                "median_any_loss": positive_or_none(losses["any"]),
                "median_value_loss": positive_or_none(losses["value"]),
                "median_player_loss": positive_or_none(losses["player"]),
                "median_policy_phi_grad_rms": positive_or_none(grads["policy"]),
                "median_any_phi_grad_rms": positive_or_none(grads["any"]),
                "median_value_phi_grad_rms": positive_or_none(grads["value"]),
                "median_player_phi_grad_rms": positive_or_none(grads["player"]),
                "mini_batches": len(losses["policy"]),
            }
        )

    aggregate = {
        "policy_phi_grad_rms": positive_or_none([row["median_policy_phi_grad_rms"] for row in per_arm]),
        "any_phi_grad_rms": positive_or_none([row["median_any_phi_grad_rms"] for row in per_arm]),
        "value_phi_grad_rms": positive_or_none([row["median_value_phi_grad_rms"] for row in per_arm]),
        "player_phi_grad_rms": positive_or_none([row["median_player_phi_grad_rms"] for row in per_arm]),
        "any_loss": positive_or_none([row["median_any_loss"] for row in per_arm]),
        "value_loss": positive_or_none([row["median_value_loss"] for row in per_arm]),
        "player_loss": positive_or_none([row["median_player_loss"] for row in per_arm]),
        "arms": len(per_arm),
    }
    return {
        "per_arm": per_arm,
        "aggregate": aggregate,
    }


def build_hybrid_suggestions(heuristic_audit: dict[str, Any], grad_audit: dict[str, Any]) -> dict[str, Any]:
    opp_loss_scale = {
        "shanten": float(heuristic_audit["opponent_state"]["normalized_scale"]["shanten_loss_over_entropy"]),
        "tenpai": float(heuristic_audit["opponent_state"]["normalized_scale"]["tenpai_loss_over_entropy"]),
    }
    opp_grad_scale = {
        "shanten": float(grad_audit["opponent_state"]["aggregate"]["shanten_phi_grad_rms"] or 0.0),
        "tenpai": float(grad_audit["opponent_state"]["aggregate"]["tenpai_phi_grad_rms"] or 0.0),
    }

    danger_loss_scale = {
        "any": float(heuristic_audit["danger"]["normalized_scale"]["danger_any_loss_over_entropy"]),
        "value": float(heuristic_audit["danger"]["normalized_scale"]["danger_value_loss_over_constant_baseline"]),
        "player": float(heuristic_audit["danger"]["normalized_scale"]["danger_player_loss_over_entropy"]),
    }
    danger_grad_scale = {
        "any": float(grad_audit["danger"]["aggregate"]["any_phi_grad_rms"] or 0.0),
        "value": float(grad_audit["danger"]["aggregate"]["value_phi_grad_rms"] or 0.0),
        "player": float(grad_audit["danger"]["aggregate"]["player_phi_grad_rms"] or 0.0),
    }

    return {
        "opponent_state": {
            "loss_scale": opp_loss_scale,
            "grad_scale": opp_grad_scale,
            "grad_only_weights": normalize_weights_from_inverse(opp_grad_scale),
            "hybrid_loss_grad_geomean_weights": hybrid_geomean_weights(opp_loss_scale, opp_grad_scale),
        },
        "danger": {
            "loss_scale": danger_loss_scale,
            "grad_scale": danger_grad_scale,
            "grad_only_weights": normalize_weights_from_inverse(danger_grad_scale),
            "hybrid_loss_grad_geomean_weights": hybrid_geomean_weights(danger_loss_scale, danger_grad_scale),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    hybrid = report["hybrid_suggestions"]
    opp = hybrid["opponent_state"]
    danger = hybrid["danger"]
    lines = [
        "# Auxiliary Subhead Gradient Audit",
        "",
        f"- sample_files: `{report['sample_files']}`",
        f"- max_minibatches_per_arm: `{report['max_minibatches_per_arm']}`",
        f"- device: `{report['device']}`",
        "",
        "## Opponent State",
        "",
        f"- selected arms: `{report['opponent_state']['aggregate']['arms']}`",
        f"- shanten phi-grad rms: `{opp['grad_scale']['shanten']:.8f}`",
        f"- tenpai phi-grad rms: `{opp['grad_scale']['tenpai']:.8f}`",
        f"- grad-only weights: `{json.dumps(opp['grad_only_weights'], ensure_ascii=False)}`",
        f"- hybrid loss+grad weights: `{json.dumps(opp['hybrid_loss_grad_geomean_weights'], ensure_ascii=False)}`",
        "",
        "## Danger",
        "",
        f"- selected arms: `{report['danger']['aggregate']['arms']}`",
        f"- any phi-grad rms: `{danger['grad_scale']['any']:.8f}`",
        f"- value phi-grad rms: `{danger['grad_scale']['value']:.8f}`",
        f"- player phi-grad rms: `{danger['grad_scale']['player']:.8f}`",
        f"- grad-only weights: `{json.dumps(danger['grad_only_weights'], ensure_ascii=False)}`",
        f"- hybrid loss+grad weights: `{json.dumps(danger['hybrid_loss_grad_geomean_weights'], ensure_ascii=False)}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-paths-file", type=Path, default=DEFAULT_SAMPLE_PATHS)
    parser.add_argument("--fidelity-dir", type=Path, default=DEFAULT_FIDELITY_DIR)
    parser.add_argument("--ab-root", type=Path, default=DEFAULT_AB_ROOT)
    parser.add_argument("--heuristic-audit-json", type=Path, default=DEFAULT_HEURISTIC_AUDIT)
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--file-batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--max-minibatches", type=int, default=48)
    parser.add_argument("--keep-per-family", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    sample_paths = load_sample_paths(args.sample_paths_file, args.max_files)
    heuristic_audit = load_json(args.heuristic_audit_json)
    device = torch.device(args.device)

    opponent_records = collect_arm_records(
        args.fidelity_dir,
        args.ab_root,
        family="opp",
        keep_per_family=args.keep_per_family,
    )
    danger_records = collect_arm_records(
        args.fidelity_dir,
        args.ab_root,
        family="danger",
        keep_per_family=args.keep_per_family,
    )

    opponent_report = audit_opponent_family(
        opponent_records,
        sample_paths,
        device=device,
        file_batch_size=args.file_batch_size,
        mini_batch_size=args.mini_batch_size,
        max_minibatches=args.max_minibatches,
    )
    danger_report = audit_danger_family(
        danger_records,
        sample_paths,
        device=device,
        file_batch_size=args.file_batch_size,
        mini_batch_size=args.mini_batch_size,
        max_minibatches=args.max_minibatches,
    )

    partial_report = {
        "sample_files": len(sample_paths),
        "max_minibatches_per_arm": args.max_minibatches,
        "device": str(device),
        "opponent_state": opponent_report,
        "danger": danger_report,
    }
    report = {
        **partial_report,
        "hybrid_suggestions": build_hybrid_suggestions(heuristic_audit, partial_report),
    }
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(args.report_json)
    print(args.report_md)


if __name__ == "__main__":
    main()

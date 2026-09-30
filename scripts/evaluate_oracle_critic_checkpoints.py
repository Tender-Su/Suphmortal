from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Oracle critic checkpoints on identical decoded states and "
            "report game-clustered paired differences."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint as NAME=PATH. Repeat for every model.",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--finalist-decision", default="")
    parser.add_argument("--imputation-seed", type=int, default=20260905)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--input-mode",
        action="append",
        choices=("true", "zero", "shuffled"),
        default=[],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every-batches", type=int, default=32)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--eval-state-fold-count",
        type=int,
        default=None,
        help=(
            "Override oracle_critic_pretrain.val_state_fold_count for a denser "
            "paired evaluation."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--game-id-modulus", type=int, default=1)
    parser.add_argument(
        "--game-id-remainder",
        type=int,
        action="append",
        default=[],
        help=(
            "Keep games whose non-negative modulo is one of these values. "
            "Repeat the option to select multiple partitions."
        ),
    )
    parser.add_argument(
        "--eval-split-override-reason",
        default="",
        help=(
            "Explicitly authorize evaluation on a file split that differs from "
            "the checkpoint provenance. The non-empty reason and both split "
            "records are written to the result."
        ),
    )
    parser.add_argument(
        "--print-result",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def resolve_eval_state_fold_count(cfg: dict[str, Any], override: int | None) -> int:
    fold_count = int(
        cfg.get("val_state_fold_count", 1)
        if override is None
        else override
    )
    if fold_count <= 0:
        raise ValueError("eval state fold count must be positive")
    return fold_count


def resolve_split_provenance(
    checkpoint_split: Any,
    evaluation_split: Any,
    *,
    split_name: str,
    override_reason: str = "",
) -> dict[str, Any]:
    reason = str(override_reason).strip()
    matches = checkpoint_split is None or checkpoint_split == evaluation_split
    if not matches and not reason:
        raise ValueError(
            f"checkpoint uses a different {split_name} file split; "
            "pass --eval-split-override-reason to authorize an audited OOD evaluation"
        )

    if checkpoint_split is None:
        mode = "checkpoint_split_unrecorded"
    elif matches:
        mode = "matched"
    else:
        mode = "explicit_ood_override"
    return {
        "mode": mode,
        "split": split_name,
        "checkpoint_split": checkpoint_split,
        "evaluation_split": evaluation_split,
        "override_used": not matches,
        "override_reason": reason if not matches else "",
    }


def normalize_game_subset(
    modulus: int,
    remainders: list[int] | tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    modulus = int(modulus)
    if modulus <= 0:
        raise ValueError("game id modulus must be positive")
    if not remainders:
        return modulus, tuple(range(modulus))
    normalized = tuple(sorted({int(remainder) for remainder in remainders}))
    if normalized[0] < 0 or normalized[-1] >= modulus:
        raise ValueError(
            f"game id remainders must be in [0, {modulus}), got {normalized}"
        )
    return modulus, normalized


def game_subset_mask(
    game_id: torch.Tensor,
    *,
    modulus: int,
    remainders: tuple[int, ...],
) -> torch.Tensor:
    game_id = torch.as_tensor(game_id).reshape(-1)
    if len(remainders) == modulus:
        return torch.ones_like(game_id, dtype=torch.bool)
    folded = torch.remainder(game_id, modulus)
    mask = torch.zeros_like(folded, dtype=torch.bool)
    for remainder in remainders:
        mask |= folded == remainder
    return mask


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    names = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"checkpoint must use NAME=PATH syntax, got {value!r}")
        if name in names:
            raise ValueError(f"duplicate checkpoint name: {name}")
        checkpoint = resolve_path(raw_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        names.add(name)
        specs.append((name, checkpoint))
    if not specs:
        raise ValueError("evaluation requires at least one checkpoint")
    return specs


def file_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paired_cluster_summary(
    row_loss_a: torch.Tensor,
    row_loss_b: torch.Tensor,
    game_id: torch.Tensor,
) -> dict[str, float | int]:
    row_loss_a = torch.as_tensor(row_loss_a, dtype=torch.float64).reshape(-1)
    row_loss_b = torch.as_tensor(row_loss_b, dtype=torch.float64).reshape(-1)
    game_id = torch.as_tensor(game_id).reshape(-1)
    if row_loss_a.shape != row_loss_b.shape or row_loss_a.shape != game_id.shape:
        raise ValueError(
            "paired losses and game ids must have the same one-dimensional shape"
        )
    if row_loss_a.numel() == 0:
        raise ValueError("paired comparison cannot be empty")

    difference = row_loss_a - row_loss_b
    mean = float(difference.mean().item())
    unique_games, inverse = torch.unique(game_id, sorted=True, return_inverse=True)
    game_sum = torch.zeros(unique_games.numel(), dtype=torch.float64).scatter_add_(
        0, inverse, difference
    )
    game_count = torch.zeros(unique_games.numel(), dtype=torch.float64).scatter_add_(
        0, inverse, torch.ones_like(difference)
    )
    game_mean = game_sum / game_count
    num_games = int(unique_games.numel())
    num_samples = int(difference.numel())
    if num_games > 1:
        residual = game_sum - mean * game_count
        cluster_se = math.sqrt(
            num_games
            / (num_games - 1)
            * float(residual.square().sum().item())
            / float(num_samples**2)
        )
        game_balanced_se = float(
            game_mean.std(unbiased=True).item() / math.sqrt(num_games)
        )
    else:
        cluster_se = 0.0
        game_balanced_se = 0.0
    return {
        "mean": mean,
        "cluster_se": cluster_se,
        "ci95_low": mean - 1.96 * cluster_se,
        "ci95_high": mean + 1.96 * cluster_se,
        "game_balanced_mean": float(game_mean.mean().item()),
        "game_balanced_se": game_balanced_se,
        "num_samples": num_samples,
        "num_games": num_games,
    }


def parts_to_row_losses(
    parts: list[dict[str, Any]],
    *,
    output_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.cat([part["pred"] for part in parts], dim=0).double()
    target = torch.cat([part["target"] for part in parts], dim=0).double()
    game_ids = [part.get("game_id") for part in parts]
    if not game_ids or any(game_id is None for game_id in game_ids):
        raise ValueError("paired evaluation requires game ids in every batch")
    game_id = torch.cat(game_ids, dim=0).reshape(-1)
    squared_error = (pred - target).square()
    if output_index is None:
        row_loss = squared_error.mean(dim=-1)
    else:
        if output_index < 0 or output_index >= squared_error.shape[1]:
            raise ValueError(f"output_index out of range: {output_index}")
        row_loss = squared_error[:, output_index]
    return row_loss, game_id


def summarize_target_tensor(target: torch.Tensor) -> dict[str, Any]:
    target = torch.as_tensor(target).detach().double().cpu()
    if target.ndim != 2 or target.shape[1] != 4:
        raise ValueError(f"target must have shape (samples, 4), got {tuple(target.shape)}")
    if target.numel() == 0:
        raise ValueError("target summary cannot be empty")
    if not torch.isfinite(target).all():
        raise ValueError("target summary requires finite values")

    levels = torch.tensor(TARGET_QUANTILES, dtype=target.dtype)

    def summarize(values: torch.Tensor) -> dict[str, Any]:
        flat_values = values.reshape(-1)
        quantiles = torch.quantile(flat_values, levels)
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "exact_zero_fraction": float((flat_values == 0.0).double().mean().item()),
            "unique_count": int(torch.unique(flat_values).numel()),
            "quantiles": {
                f"q{level:g}": float(value.item())
                for level, value in zip(TARGET_QUANTILES, quantiles)
            },
        }

    return {
        "all": summarize(target),
        "outputs": {
            f"relative_player_{idx}": summarize(target[:, idx])
            for idx in range(target.shape[1])
        },
        "zero_sum_max_abs": float(target.sum(dim=-1).abs().max().item()),
    }


def summarize_regression_slices(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    pred = torch.as_tensor(pred).detach().double().cpu().reshape(-1)
    target = torch.as_tensor(target).detach().double().cpu().reshape(-1)
    if pred.shape != target.shape or pred.numel() == 0:
        raise ValueError("slice prediction and target must be non-empty and aligned")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("slice prediction and target must be finite")

    masks = {
        "all": torch.ones_like(target, dtype=torch.bool),
        "exact_zero": target == 0.0,
        "nonzero": target != 0.0,
        "abs_ge_2": target.abs() >= 2.0,
        "abs_ge_4": target.abs() >= 4.0,
    }
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum().item())
        if count == 0:
            continue
        error = pred[mask] - target[mask]
        result[name] = {
            "count": count,
            "fraction": count / float(target.numel()),
            "loss": float(error.square().mean().item()),
            "mae": float(error.abs().mean().item()),
            "bias": float(error.mean().item()),
            "pred_mean": float(pred[mask].mean().item()),
            "target_mean": float(target[mask].mean().item()),
        }
    return result


def atomic_write_json(destination: Path, value: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    checkpoint_specs = parse_checkpoint_specs(args.checkpoint)
    os.environ["MORTAL_CFG"] = str(config_path)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from mortal.config import config
    from mortal.core.evidence_contract import validation_input_contract, require_finalist_decision, native_module_file
    from mortal.data.oracle_value import ORACLE_IMPUTATION_VERSION, ORACLE_TARGET_CLOCK_VERSION
    import libriichi
    if args.split == 'test':
        require_finalist_decision(args.finalist_decision, [item[1] for item in checkpoint_specs])
    from mortal.online.pretrain_oracle_critic import (
        batch_metrics,
        build_file_splits,
        build_models,
        finalize_metrics,
        make_dataset,
        make_loader,
        model_forward,
        normalize_critic_arch,
        normalize_eval_input_modes,
        normalize_oracle_fusion_mode,
        sanitize_sys_path_for_spawn,
        summarize_file_splits,
        transform_eval_invisible_obs,
    )

    sanitize_sys_path_for_spawn()
    cfg = dict(config.get("oracle_critic_pretrain", {}))
    cfg['val_oracle_imputation_seed'] = args.imputation_seed
    eval_state_fold_count = resolve_eval_state_fold_count(
        cfg,
        args.eval_state_fold_count,
    )
    game_id_modulus, game_id_remainders = normalize_game_subset(
        args.game_id_modulus,
        args.game_id_remainder,
    )
    cfg["val_state_fold_count"] = eval_state_fold_count
    modes = normalize_eval_input_modes(args.input_mode or ("true", "zero", "shuffled"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_files, dev_files, test_files = build_file_splits(cfg)
    split_info = summarize_file_splits(cfg, train_files, dev_files, test_files)
    eval_files = dev_files if args.split == "dev" else test_files
    if not eval_files:
        raise ValueError(f"configured {args.split} split is empty")
    input_contract = validation_input_contract(eval_files, native_file=native_module_file(libriichi), settings={
        'imputation_seed': args.imputation_seed, 'imputation_version': ORACLE_IMPUTATION_VERSION,
        'target_clock_version': ORACLE_TARGET_CLOCK_VERSION, 'rank_points': config['env']['pts'],
        'gamma': cfg.get('discount_gamma', 1.0), 'return_mode': cfg.get('return_mode', 'score_rank_mc'),
        'fold_count': eval_state_fold_count, 'max_batches': args.max_batches,
        'game_id_modulus': game_id_modulus, 'game_id_remainders': game_id_remainders,
    })

    models = {}
    checkpoint_records = {}
    expected_zero_sum = bool(cfg.get("exact_zero_sum", False))
    reference_target_contract = None
    for name, checkpoint_path in checkpoint_specs:
        state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        state_cfg = state.get("oracle_critic_pretrain", {})
        state_arch = normalize_critic_arch(
            state_cfg.get("critic_arch", cfg.get("critic_arch", "single_tower"))
        )
        state_zero_sum = bool(state_cfg.get("exact_zero_sum", expected_zero_sum))
        if state_zero_sum != expected_zero_sum:
            raise ValueError(
                f"{name} exact-zero-sum contract differs from evaluation config: "
                f"checkpoint={state_zero_sum}, config={expected_zero_sum}"
            )
        target_contract = {
            key: state_cfg.get(key, cfg.get(key))
            for key in ("target_mode", "return_mode", "discount_gamma")
        }
        if reference_target_contract is None:
            reference_target_contract = target_contract
        elif target_contract != reference_target_contract:
            raise ValueError(
                f"{name} target contract differs from the first checkpoint: "
                f"{target_contract!r} != {reference_target_contract!r}"
            )
        saved_split = state.get("file_splits", {}).get(args.split)
        split_provenance = resolve_split_provenance(
            saved_split,
            split_info[args.split],
            split_name=args.split,
            override_reason=args.eval_split_override_reason,
        )
        oracle_brain, value_net = build_models(
            device,
            critic_arch=state_arch,
            oracle_fusion_init=float(state_cfg.get("oracle_fusion_init", 0.5)),
            oracle_fusion_mode=normalize_oracle_fusion_mode(
                state_cfg.get("oracle_fusion_mode", "linear")
            ),
            oracle_fusion_hidden=int(state_cfg.get("oracle_fusion_hidden", 512) or 512),
            exact_zero_sum=expected_zero_sum,
            value_loss_mode=state_cfg.get("value_loss_mode", "mse"),
            value_head_hidden=int(state_cfg.get("value_head_hidden", 256) or 256),
            value_num_bins=int(state_cfg.get("value_num_bins", 100) or 100),
            value_target_min=float(state_cfg.get("value_target_min", -6.0)),
            value_target_max=float(state_cfg.get("value_target_max", 6.0)),
            value_sigma_to_bin_ratio=float(
                state_cfg.get("value_sigma_to_bin_ratio", 2.0)
            ),
            value_padding_sigma=float(state_cfg.get("value_padding_sigma", 3.0)),
        )
        oracle_brain.load_state_dict(state["oracle_brain"])
        value_net.load_state_dict(state["value_net"])
        oracle_brain.eval()
        value_net.eval()
        models[name] = (oracle_brain, value_net)
        checkpoint_records[name] = {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "steps": int(state.get("steps", 0)),
            "critic_arch": state_arch,
            "oracle_fusion_mode": normalize_oracle_fusion_mode(
                state_cfg.get("oracle_fusion_mode", "linear")
            ),
            "value_loss_mode": state_cfg.get("value_loss_mode", "mse"),
            "value_head_hidden": int(state_cfg.get("value_head_hidden", 256) or 256),
            "training_contract": state.get("training_contract", {}),
            "split_provenance": split_provenance,
        }
        del state
        gc.collect()

    loader = make_loader(make_dataset(eval_files, cfg, train=False), cfg, train=False)
    parts = {
        name: {mode: [] for mode in modes}
        for name in models
    }
    started_at = time.monotonic()
    samples_seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            obs, invisible_obs, target, _player_id = batch[:4]
            game_id = batch[4] if len(batch) > 4 else None
            if game_id is None:
                raise ValueError("evaluation loader did not emit game ids")
            subset_mask = game_subset_mask(
                game_id,
                modulus=game_id_modulus,
                remainders=game_id_remainders,
            )
            if not bool(subset_mask.any()):
                continue
            obs = obs[subset_mask]
            invisible_obs = invisible_obs[subset_mask]
            target = target[subset_mask]
            game_id = game_id[subset_mask]
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            invisible_obs = invisible_obs.to(
                dtype=torch.float32, device=device, non_blocking=True
            )
            target = target.to(dtype=torch.float32, device=device, non_blocking=True)
            transformed = {
                mode: transform_eval_invisible_obs(invisible_obs, mode)
                for mode in modes
            }
            for name, (oracle_brain, value_net) in models.items():
                for mode in modes:
                    pred = model_forward(
                        oracle_brain,
                        value_net,
                        obs,
                        transformed[mode],
                        enable_amp=bool(args.amp),
                        device_type=device.type,
                    )
                    parts[name][mode].append(
                        batch_metrics(pred, target, game_id=game_id)
                    )
            samples_seen += int(obs.shape[0])
            completed_batches = batch_idx + 1
            if (
                args.log_every_batches > 0
                and completed_batches % args.log_every_batches == 0
            ):
                elapsed = max(time.monotonic() - started_at, 1e-9)
                print(
                    f"paired eval batches={completed_batches} samples={samples_seen} "
                    f"elapsed={elapsed:.1f}s batches_per_s={completed_batches / elapsed:.3f}",
                    flush=True,
                )

    metrics = {
        name: {mode: finalize_metrics(mode_parts) for mode, mode_parts in by_mode.items()}
        for name, by_mode in parts.items()
    }
    row_losses = {}
    output_row_losses = {}
    shared_game_id = None
    for name, by_mode in parts.items():
        row_losses[name] = {}
        output_row_losses[name] = {}
        for mode, mode_parts in by_mode.items():
            losses, game_id = parts_to_row_losses(mode_parts)
            if shared_game_id is None:
                shared_game_id = game_id
            elif not torch.equal(shared_game_id, game_id):
                raise RuntimeError("paired evaluation sample order changed between models or modes")
            row_losses[name][mode] = losses
            output_row_losses[name][mode] = {
                f"relative_player_{output_index}": parts_to_row_losses(
                    mode_parts,
                    output_index=output_index,
                )[0]
                for output_index in range(4)
            }
    assert shared_game_id is not None
    model_names = list(models)
    first_name = model_names[0]
    first_mode = modes[0]
    shared_target = torch.cat(
        [part["target"] for part in parts[first_name][first_mode]],
        dim=0,
    ).double()
    target_summary = summarize_target_tensor(shared_target)
    primary_target = shared_target[:, 0]
    primary_slice_masks = {
        "all": torch.ones_like(primary_target, dtype=torch.bool),
        "exact_zero": primary_target == 0.0,
        "nonzero": primary_target != 0.0,
        "abs_ge_2": primary_target.abs() >= 2.0,
        "abs_ge_4": primary_target.abs() >= 4.0,
    }
    primary_slice_metrics = {
        name: {
            mode: summarize_regression_slices(
                torch.cat([part["pred"] for part in parts[name][mode]], dim=0)[:, 0],
                primary_target,
            )
            for mode in modes
        }
        for name in model_names
    }

    paired_models = {}
    paired_model_outputs = {}
    paired_primary_slices = {}
    for name_a, name_b in combinations(model_names, 2):
        comparison_name = f"{name_a}_minus_{name_b}"
        paired_models[comparison_name] = {
            mode: paired_cluster_summary(
                row_losses[name_a][mode],
                row_losses[name_b][mode],
                shared_game_id,
            )
            for mode in modes
        }
        paired_model_outputs[comparison_name] = {
            mode: {
                output_name: paired_cluster_summary(
                    output_row_losses[name_a][mode][output_name],
                    output_row_losses[name_b][mode][output_name],
                    shared_game_id,
                )
                for output_name in output_row_losses[name_a][mode]
            }
            for mode in modes
        }
        paired_primary_slices[comparison_name] = {
            mode: {
                slice_name: paired_cluster_summary(
                    output_row_losses[name_a][mode]["relative_player_0"][mask],
                    output_row_losses[name_b][mode]["relative_player_0"][mask],
                    shared_game_id[mask],
                )
                for slice_name, mask in primary_slice_masks.items()
                if bool(mask.any())
            }
            for mode in modes
        }

    oracle_dependency = {}
    oracle_dependency_outputs = {}
    if "true" in modes:
        for name in model_names:
            oracle_dependency[name] = {}
            oracle_dependency_outputs[name] = {}
            for baseline in ("zero", "shuffled"):
                if baseline not in modes:
                    continue
                comparison_name = f"{baseline}_minus_true"
                oracle_dependency[name][comparison_name] = paired_cluster_summary(
                    row_losses[name][baseline],
                    row_losses[name]["true"],
                    shared_game_id,
                )
                oracle_dependency_outputs[name][comparison_name] = {
                    output_name: paired_cluster_summary(
                        output_row_losses[name][baseline][output_name],
                        output_row_losses[name]["true"][output_name],
                        shared_game_id,
                    )
                    for output_name in output_row_losses[name]["true"]
                }

    ranking_mode = "true" if "true" in modes else modes[0]
    ranking = sorted(
        (
            {
                "name": name,
                "loss": metrics[name][ranking_mode]["loss"],
                "corr": metrics[name][ranking_mode]["corr"],
                "explained_variance": metrics[name][ranking_mode]["explained_variance"],
            }
            for name in model_names
        ),
        key=lambda item: item["loss"],
    )
    primary_ranking = sorted(
        (
            {
                "name": name,
                **metrics[name][ranking_mode]["outputs"]["relative_player_0"],
            }
            for name in model_names
        ),
        key=lambda item: item["loss"],
    )
    result = {
        "format": "oracle_critic_paired_eval_v2",
        "validation_input_contract": input_contract,
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "checkpoints": checkpoint_records,
        "split": args.split,
        "split_info": split_info[args.split],
        "eval_split_override_reason": str(args.eval_split_override_reason).strip(),
        "input_modes": list(modes),
        "eval_state_fold_count": eval_state_fold_count,
        "game_subset": {
            "modulus": game_id_modulus,
            "remainders": list(game_id_remainders),
        },
        "amp": bool(args.amp),
        "num_samples": int(shared_game_id.numel()),
        "num_games": int(torch.unique(shared_game_id).numel()),
        "target_summary": target_summary,
        "primary_slice_metrics": primary_slice_metrics,
        "elapsed_seconds": time.monotonic() - started_at,
        "metrics": metrics,
        "ranking": ranking,
        "primary_ranking": primary_ranking,
        "paired_model_differences": paired_models,
        "paired_model_output_differences": paired_model_outputs,
        "paired_primary_slice_differences": paired_primary_slices,
        "paired_oracle_dependency": oracle_dependency,
        "paired_oracle_dependency_outputs": oracle_dependency_outputs,
    }
    if args.output:
        output_path = resolve_path(args.output).resolve()
        atomic_write_json(output_path, result)
        print(f"wrote {output_path}", flush=True)
    if args.print_result:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

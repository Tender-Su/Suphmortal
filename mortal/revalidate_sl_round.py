from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

import run_sl_ab as ab
import run_sl_fidelity as fid
from sl_selection import SCENARIO_SCORE_VERSION, SCENARIO_SCORE_VERSION_FIELD


REQUIRED_SELECTOR_FIELDS = (
    'discard_count',
    'chi_exact_nll',
    'selection_quality_score',
)


def read_json(path: Path) -> dict[str, Any]:
    return fid.load_json(path)


def arm_result_paths(arm_root: Path) -> list[Path]:
    return sorted(path for path in arm_root.glob('*/arm_result.json') if path.is_file())


def payload_candidate(payload: dict[str, Any]) -> fid.CandidateSpec:
    run_payload = payload.get('run', {})
    return fid.CandidateSpec(
        arm_name=str(payload.get('arm_name', '')),
        scheduler_profile=str(run_payload.get('scheduler_profile', '')),
        curriculum_profile=str(run_payload.get('curriculum_profile', '')),
        weight_profile=str(run_payload.get('weight_profile', '')),
        window_profile=str(run_payload.get('window_profile', '')),
        cfg_overrides={},
        meta=dict(payload.get('candidate_meta') or {}),
    )


def final_phase_name(payload: dict[str, Any]) -> str:
    phase_order = payload.get('run', {}).get('phase_order') or []
    if not phase_order:
        raise ValueError(f'missing phase_order in payload for {payload.get("arm_name")}')
    return str(phase_order[-1])


def checkpoint_summary_ref(
    payload: dict[str, Any],
    *,
    checkpoint_kind: str,
    phase_name: str | None,
) -> tuple[str, dict[str, Any]]:
    selected_phase = final_phase_name(payload) if phase_name in (None, '', 'final') else str(phase_name)
    phase_results = payload.get('run', {}).get('phase_results') or {}
    if selected_phase not in phase_results:
        raise KeyError(f'phase `{selected_phase}` missing in {payload.get("arm_name")}')
    phase_payload = phase_results[selected_phase]
    if checkpoint_kind not in phase_payload:
        raise KeyError(f'checkpoint `{checkpoint_kind}` missing in {payload.get("arm_name")}:{selected_phase}')
    summary = phase_payload[checkpoint_kind]
    if not isinstance(summary, dict):
        raise TypeError(f'invalid summary for {payload.get("arm_name")}:{selected_phase}:{checkpoint_kind}')
    return selected_phase, summary


def summary_needs_revalidation(summary: dict[str, Any]) -> bool:
    metrics = summary.get('last_full_recent_metrics') or {}
    if metrics.get(SCENARIO_SCORE_VERSION_FIELD) != SCENARIO_SCORE_VERSION:
        return True
    return any(field not in metrics for field in REQUIRED_SELECTOR_FIELDS)


def start_epoch_for_state(state: dict[str, Any]) -> int:
    epoch = int(state.get('epoch', 0) or 0)
    epoch_complete = bool(state.get('epoch_complete', True))
    return epoch + 1 if epoch_complete else epoch


def clear_validation_history(state: dict[str, Any]) -> None:
    state['best_val_loss'] = float('inf')
    state['best_val_action_acc'] = 0.0
    state['best_val_action_score'] = float('-inf')
    state['best_val_rank_acc'] = 0.0
    state['best_full_recent_loss'] = float('inf')
    state['best_full_recent_action_acc'] = 0.0
    state['best_full_recent_action_score'] = float('-inf')
    state['best_full_recent_rank_acc'] = 0.0
    state['patience_counter'] = 0
    state['num_lr_reductions'] = 0
    state['validation_checks'] = 0
    state['full_validation_checks'] = 0
    state['old_regression_checks'] = 0
    state['last_monitor_recent_metrics'] = None
    state['last_full_recent_metrics'] = None
    state['last_old_regression_metrics'] = None


def build_eval_only_config(
    state: dict[str, Any],
    *,
    config_section: str,
    temp_dir: Path,
    start_epoch: int,
    eval_num_workers: int,
) -> dict[str, Any]:
    saved_config = copy.deepcopy(state.get('config') or {})
    if not isinstance(saved_config, dict):
        raise TypeError('checkpoint does not contain a valid config payload')

    if config_section in saved_config and isinstance(saved_config[config_section], dict):
        section_cfg = dict(saved_config[config_section])
    else:
        raise KeyError(f'config section `{config_section}` missing from checkpoint config')

    ckpt_dir = temp_dir / 'checkpoints'
    tb_dir = temp_dir / 'tb'
    section_cfg.update(
        {
            'state_file': str(ckpt_dir / 'latest.pth'),
            'best_state_file': str(ckpt_dir / 'best_loss.pth'),
            'best_loss_state_file': str(ckpt_dir / 'best_loss.pth'),
            'best_acc_state_file': str(ckpt_dir / 'best_action_score.pth'),
            'best_rank_state_file': str(ckpt_dir / 'best_rank.pth'),
            'tensorboard_dir': str(tb_dir),
            'max_epochs': start_epoch,
            'full_val_every_checks': 1,
            'old_regression_every_checks': 1,
            'num_workers': eval_num_workers,
        }
    )
    control_cfg = dict(saved_config.get('control') or {})
    control_cfg['enable_cuda_prefetch'] = False
    saved_config['control'] = control_cfg
    saved_config[config_section] = section_cfg
    return saved_config


def revalidate_checkpoint_summary(
    checkpoint_path: Path,
    *,
    temp_root: Path,
    label: str,
    eval_num_workers: int,
    keep_temp: bool = False,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config_section = str(state.get('config_section') or 'supervised')
    start_epoch = start_epoch_for_state(state)
    clear_validation_history(state)

    temp_dir = temp_root / label
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_ckpt_dir = temp_dir / 'checkpoints'
    temp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    temp_state_path = temp_ckpt_dir / 'latest.pth'
    torch.save(state, temp_state_path)

    cfg = build_eval_only_config(
        state,
        config_section=config_section,
        temp_dir=temp_dir,
        start_epoch=start_epoch,
        eval_num_workers=eval_num_workers,
    )
    cfg_path = temp_dir / 'config.toml'
    log_path = temp_dir / 'revalidate.log'
    ab.write_toml(cfg_path, cfg)

    env = os.environ.copy()
    env['MORTAL_CFG'] = str(cfg_path)
    with log_path.open('w', encoding='utf-8', newline='\n') as log_file:
        proc = subprocess.run(
            [sys.executable, 'train_supervised.py'],
            cwd=ab.MORTAL_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        log_tail = ab.read_log_tail(log_path)
        raise RuntimeError(
            f'eval-only revalidation failed for {checkpoint_path}; '
            f'see {log_path}\n{log_tail}'
        )

    refreshed = ab.load_state_summary(temp_state_path)
    refreshed['path'] = str(checkpoint_path)
    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return refreshed


def update_payload_summary(
    payload: dict[str, Any],
    *,
    phase_name: str,
    checkpoint_kind: str,
    refreshed_summary: dict[str, Any],
) -> None:
    payload['run']['phase_results'][phase_name][checkpoint_kind] = refreshed_summary
    if phase_name == final_phase_name(payload):
        payload['run']['final'][checkpoint_kind] = refreshed_summary
        if checkpoint_kind == 'best_loss':
            payload['run']['score'] = list(ab.score_summary(refreshed_summary))


def build_ranked_round_payload(
    *,
    arm_root: Path,
    payloads: list[dict[str, Any]],
    checkpoint_kind: str,
    phase_name: str | None,
    revalidated_count: int,
    stale_before_count: int,
) -> dict[str, Any]:
    entries = [
        fid.summarize_entry(payload['arm_name'], payload_candidate(payload), payload)
        for payload in payloads
    ]
    ranked = fid.rank_round_entries(entries)
    return {
        'round_name': f'revalidated_{checkpoint_kind}_{phase_name or "final"}',
        'source_dir': str(arm_root),
        'checkpoint_kind': checkpoint_kind,
        'phase_name': phase_name or 'final',
        'scenario_score_version': SCENARIO_SCORE_VERSION,
        'evaluated_arms': len(payloads),
        'stale_before_count': stale_before_count,
        'revalidated_count': revalidated_count,
        'ranking': ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arm-root',
        required=True,
        help='Directory containing one arm subdirectory per candidate',
    )
    parser.add_argument(
        '--checkpoint-kind',
        default='best_loss',
        choices=('latest', 'best_loss', 'best_acc', 'best_rank'),
    )
    parser.add_argument(
        '--phase',
        default='final',
        help='Phase name to refresh, or `final` for the last phase',
    )
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('--eval-num-workers', type=int, default=0)
    parser.add_argument(
        '--arm-name',
        action='append',
        default=[],
        help='Optional exact arm name filter; can be passed multiple times',
    )
    parser.add_argument(
        '--round-out',
        default='',
        help='Optional path for the reranked round summary JSON',
    )
    args = parser.parse_args()

    arm_root = Path(args.arm_root).resolve()
    if not arm_root.exists():
        raise FileNotFoundError(arm_root)

    temp_root = fid.FIDELITY_ROOT / '_revalidate_tmp'
    temp_root.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    stale_before_count = 0
    revalidated_count = 0
    arm_paths = arm_result_paths(arm_root)
    if not arm_paths:
        raise RuntimeError(f'no arm_result.json found under {arm_root}')
    if args.arm_name:
        allowed = set(args.arm_name)
        arm_paths = [path for path in arm_paths if path.parent.name in allowed]
        if not arm_paths:
            raise RuntimeError(f'no matching arm_result.json found for filters: {sorted(allowed)}')

    for arm_result_path in arm_paths:
        payload = read_json(arm_result_path)
        if not payload.get('ok'):
            payloads.append(payload)
            continue

        selected_phase, summary = checkpoint_summary_ref(
            payload,
            checkpoint_kind=args.checkpoint_kind,
            phase_name=args.phase,
        )
        needs_revalidation = summary_needs_revalidation(summary)
        if needs_revalidation:
            stale_before_count += 1

        if args.force or needs_revalidation:
            checkpoint_path = Path(summary['path'])
            label = f'{payload["arm_name"]}_{selected_phase}_{args.checkpoint_kind}'
            refreshed = revalidate_checkpoint_summary(
                checkpoint_path,
                temp_root=temp_root,
                label=label,
                eval_num_workers=args.eval_num_workers,
                keep_temp=args.keep_temp,
            )
            update_payload_summary(
                payload,
                phase_name=selected_phase,
                checkpoint_kind=args.checkpoint_kind,
                refreshed_summary=refreshed,
            )
            fid.atomic_write_json(arm_result_path, payload)
            revalidated_count += 1

        payloads.append(payload)

    round_payload = build_ranked_round_payload(
        arm_root=arm_root,
        payloads=payloads,
        checkpoint_kind=args.checkpoint_kind,
        phase_name=args.phase,
        revalidated_count=revalidated_count,
        stale_before_count=stale_before_count,
    )
    round_out = (
        Path(args.round_out).resolve()
        if args.round_out
        else arm_root / f'revalidated_{args.checkpoint_kind}_{args.phase or "final"}_round.json'
    )
    fid.atomic_write_json(round_out, round_payload)

    top_entries = round_payload['ranking'][:8]
    print(f'arm_root={arm_root}')
    print(f'stale_before={stale_before_count} revalidated={revalidated_count} total={len(payloads)}')
    print(f'round_out={round_out}')
    for idx, entry in enumerate(top_entries, start=1):
        print(
            f'{idx:02d}. {entry["arm_name"]} '
            f'loss={entry["full_recent_loss"]:.6f} '
            f'select={entry.get("selection_quality_score", float("nan")):.4f} '
            f'scenario={entry.get("scenario_quality_score", float("nan")):.4f} '
            f'action={entry.get("action_quality_score", float("nan")):.4f}'
        )


if __name__ == '__main__':
    main()

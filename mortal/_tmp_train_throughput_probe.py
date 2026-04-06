from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sl_ab as ab
import run_sl_fidelity as fidelity
import run_sl_loader_ab as loader_ab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-name', required=True)
    parser.add_argument('--round-name', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--affinity-label', required=True)
    parser.add_argument('--seed', type=int, default=20260326)
    parser.add_argument('--step-scale', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--file-batch-size', type=int, required=True)
    parser.add_argument('--prefetch-factor', type=int, required=True)
    parser.add_argument('--val-file-batch-size', type=int, default=7)
    parser.add_argument('--val-prefetch-factor', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--full-recent-files', type=int, default=0)
    parser.add_argument('--old-regression-files', type=int, default=0)
    parser.add_argument('--protocol-arm', default=loader_ab.DEFAULT_PROTOCOL_ARM)
    parser.add_argument('--rank-budget-ratio', type=float, default=loader_ab.DEFAULT_RANK_BUDGET_RATIO)
    parser.add_argument('--opp-budget-ratio', type=float, default=loader_ab.DEFAULT_OPP_BUDGET_RATIO)
    parser.add_argument('--danger-budget-ratio', type=float, default=loader_ab.DEFAULT_DANGER_BUDGET_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = ab.group_files_by_month(ab.load_all_files())
    pure_train_only = args.full_recent_files <= 0 and args.old_regression_files <= 0
    if pure_train_only:
        eval_splits = {
            'monitor_recent_files': [],
            'full_recent_files': [],
            'old_regression_files': [],
        }
    else:
        eval_splits = ab.build_eval_splits(
            grouped,
            args.seed + 77,
            {
                'full_recent': args.full_recent_files,
                'old_regression': args.old_regression_files,
            },
        )
    candidate = loader_ab.load_reference_candidate(
        run_name=args.run_name,
        protocol_arm=args.protocol_arm,
        rank_budget_ratio=args.rank_budget_ratio,
        opp_budget_ratio=args.opp_budget_ratio,
        danger_budget_ratio=args.danger_budget_ratio,
    )
    config = loader_ab.make_loader_config(
        num_workers=args.num_workers,
        file_batch_size=args.file_batch_size,
        prefetch_factor=args.prefetch_factor,
        val_file_batch_size=args.val_file_batch_size,
        val_prefetch_factor=args.val_prefetch_factor,
        batch_size=args.batch_size,
    )
    result = loader_ab.run_loader_config(
        suite_name=args.suite_name,
        round_name=args.round_name,
        config=config,
        grouped=grouped,
        eval_splits=eval_splits,
        candidate=candidate,
        seed=args.seed,
        step_scale=args.step_scale,
        val_every_steps=0,
        monitor_val_batches=0,
        full_recent_files=args.full_recent_files,
        old_regression_files=args.old_regression_files,
        phase_name='phase_a',
    )
    summary_dir = loader_ab.LOADER_AB_ROOT / args.suite_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'{args.round_name}.summary.json'
    payload = {
        'suite_name': args.suite_name,
        'round_name': args.round_name,
        'mode': 'pure_train_only',
        'pure_train_only': pure_train_only,
        'affinity_label': args.affinity_label,
        'config': {
            'name': config.name,
            'num_workers': config.num_workers,
            'file_batch_size': config.file_batch_size,
            'prefetch_factor': config.prefetch_factor,
            'val_file_batch_size': config.val_file_batch_size,
            'val_prefetch_factor': config.val_prefetch_factor,
            'batch_size': config.batch_size,
        },
        'reference_candidate': {
            'run_name': args.run_name,
            'arm_name': candidate.arm_name,
            'protocol_arm': candidate.meta.get('protocol_arm'),
            'rank_budget_ratio': candidate.meta.get('rank_budget_ratio'),
            'opp_budget_ratio': candidate.meta.get('opp_budget_ratio'),
            'danger_budget_ratio': candidate.meta.get('danger_budget_ratio'),
        },
        'benchmark_settings': {
            'seed': args.seed,
            'step_scale': args.step_scale,
            'phase_name': 'phase_a',
            'val_every_steps': 0,
            'monitor_val_batches': 0,
            'full_recent_files': args.full_recent_files,
            'old_regression_files': args.old_regression_files,
            'pure_train_only': pure_train_only,
        },
        'result': result,
    }
    fidelity.atomic_write_json(summary_path, payload)
    print(json.dumps(fidelity.normalize_payload(payload), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

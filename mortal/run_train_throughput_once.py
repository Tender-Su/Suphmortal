from __future__ import annotations

import argparse
import json
import os

import run_stage05_ab as ab
import run_stage05_loader_ab as loader_ab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--suite-name', required=True)
    parser.add_argument('--round-name', required=True)
    parser.add_argument('--protocol-arm', default=loader_ab.DEFAULT_PROTOCOL_ARM)
    parser.add_argument('--rank-budget-ratio', type=float, default=loader_ab.DEFAULT_RANK_BUDGET_RATIO)
    parser.add_argument('--opp-budget-ratio', type=float, default=loader_ab.DEFAULT_OPP_BUDGET_RATIO)
    parser.add_argument('--danger-budget-ratio', type=float, default=loader_ab.DEFAULT_DANGER_BUDGET_RATIO)
    parser.add_argument('--seed', type=int, default=20260326)
    parser.add_argument('--step-scale', type=float, default=1.0)
    parser.add_argument('--phase-name', default='phase_a')
    parser.add_argument('--full-recent-files', type=int, default=0)
    parser.add_argument('--old-regression-files', type=int, default=0)
    parser.add_argument('--val-every-steps', type=int, default=0)
    parser.add_argument('--monitor-val-batches', type=int, default=0)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--file-batch-size', type=int, required=True)
    parser.add_argument('--prefetch-factor', type=int, required=True)
    parser.add_argument('--val-file-batch-size', type=int, default=loader_ab.DEFAULT_VAL_FILE_BATCH_SIZE)
    parser.add_argument('--val-prefetch-factor', type=int, default=loader_ab.DEFAULT_VAL_PREFETCH_FACTOR)
    parser.add_argument('--batch-size', type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    pure_train_only = (
        args.val_every_steps <= 0
        and args.monitor_val_batches <= 0
        and args.full_recent_files <= 0
        and args.old_regression_files <= 0
    )
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
    benchmark_inputs_signature = loader_ab.loader_benchmark_inputs_signature(
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
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
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        benchmark_inputs_signature=benchmark_inputs_signature,
        candidate=candidate,
        seed=args.seed,
        step_scale=args.step_scale,
        val_every_steps=args.val_every_steps,
        monitor_val_batches=args.monitor_val_batches,
        full_recent_files=args.full_recent_files,
        old_regression_files=args.old_regression_files,
        phase_name=args.phase_name,
    )
    payload = {
        'suite_name': args.suite_name,
        'round_name': args.round_name,
        'mode': 'pure_train_only',
        'pure_train_only': pure_train_only,
        'affinity': os.environ.get('MORTAL_CPU_AFFINITY', ''),
        'config': result['config'],
        'candidate': result['candidate'],
        'phase_name': args.phase_name,
        'step_scale': args.step_scale,
        'val_every_steps': args.val_every_steps,
        'monitor_val_batches': args.monitor_val_batches,
        'full_recent_files': args.full_recent_files,
        'old_regression_files': args.old_regression_files,
        'total_runtime_sec': result['total_runtime_sec'],
        'total_steps': result['total_steps'],
        'total_steps_per_sec': result['total_steps_per_sec'],
        'stable': result['stable'],
        'total_retry_count': result['total_retry_count'],
        'log_path': result['phase_timings'][args.phase_name]['log_path'],
        'config_path': result['phase_timings'][args.phase_name]['config_path'],
    }
    summary_path = loader_ab.LOADER_AB_ROOT / args.suite_name / f'{args.round_name}.summary.json'
    loader_ab.fidelity.atomic_write_json(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

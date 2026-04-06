from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import run_sl_fidelity as fidelity
import sl_current_defaults as sl_defaults


REPO_ROOT = Path(__file__).resolve().parent.parent
MORTAL_DIR = REPO_ROOT / 'mortal'
PAIRWISE_ROOT = REPO_ROOT / 'logs' / 'sl_ab'


def resolve_pairwise_dir(run_name: str) -> Path:
    matches = sorted(
        PAIRWISE_ROOT.glob(f'{run_name}_p1_pairwise_*'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f'cannot find pairwise run dir for {run_name}')
    return matches[0]


def kill_process_tree(pid: int) -> None:
    subprocess.run(
        ['taskkill', '/PID', str(pid), '/T', '/F'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cleanup_run_lock(run_dir: Path) -> None:
    lock_path = run_dir / 'run.lock.json'
    if not lock_path.exists():
        return
    payload = fidelity.load_json(lock_path)
    if fidelity.lock_belongs_to_running_process(payload):
        return
    lock_path.unlink(missing_ok=True)


def stop_owned_pairwise_controller(run_dir: Path) -> int | None:
    lock_path = run_dir / 'run.lock.json'
    if not lock_path.exists():
        return None
    payload = fidelity.load_json(lock_path)
    if not fidelity.lock_belongs_to_running_process(payload):
        return None
    pid = int(payload.get('pid', 0) or 0)
    if pid <= 0:
        return None
    kill_process_tree(pid)
    time.sleep(3)
    return pid


def mark_paused(run_dir: Path, *, reason: str) -> None:
    state_path = run_dir / 'state.json'
    state = fidelity.load_json(state_path)
    state['status'] = 'paused_for_loader_ab'
    state['pause_reason'] = reason
    fidelity.atomic_write_json(state_path, state)
    fidelity.update_results_doc(run_dir, state)


def wait_for_arm_completion(run_name: str, arm_name: str, poll_seconds: int) -> tuple[Path, Path]:
    while True:
        try:
            pairwise_dir = resolve_pairwise_dir(run_name)
        except FileNotFoundError:
            time.sleep(poll_seconds)
            continue
        arm_result_path = pairwise_dir / arm_name / 'arm_result.json'
        if arm_result_path.exists():
            return pairwise_dir, arm_result_path
        time.sleep(poll_seconds)


def run_loader_ab(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        'run_sl_loader_ab.py',
        '--run-name',
        args.run_name,
        '--protocol-arm',
        args.protocol_arm,
        '--rank-budget-ratio',
        str(args.rank_budget_ratio),
        '--opp-budget-ratio',
        str(args.opp_budget_ratio),
        '--danger-budget-ratio',
        str(args.danger_budget_ratio),
        '--seed',
        str(args.seed),
        '--step-scale',
        str(args.step_scale),
        '--batch-size',
        str(args.batch_size),
        '--phase-name',
        args.phase_name,
        '--training-finalists',
        str(args.training_finalists),
        '--val-every-steps',
        str(args.val_every_steps),
        '--monitor-val-batches',
        str(args.monitor_val_batches),
        '--full-recent-files',
        str(args.full_recent_files),
        '--old-regression-files',
        str(args.old_regression_files),
    ]
    if args.suite_name:
        cmd.extend(['--suite-name', args.suite_name])
    proc = subprocess.run(cmd, cwd=MORTAL_DIR, check=False)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--arm-name', required=True)
    parser.add_argument('--poll-seconds', type=int, default=20)
    parser.add_argument('--protocol-arm', default=sl_defaults.CURRENT_PRIMARY_PROTOCOL_ARM)
    parser.add_argument('--rank-budget-ratio', type=float, default=0.15)
    parser.add_argument('--opp-budget-ratio', type=float, default=0.03)
    parser.add_argument('--danger-budget-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=20260326)
    parser.add_argument('--step-scale', type=float, default=0.25)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--phase-name', default='phase_a')
    parser.add_argument('--training-finalists', type=int, default=2)
    parser.add_argument('--val-every-steps', type=int, default=750)
    parser.add_argument('--monitor-val-batches', type=int, default=64)
    parser.add_argument('--full-recent-files', type=int, default=64)
    parser.add_argument('--old-regression-files', type=int, default=32)
    parser.add_argument('--suite-name', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = fidelity.FIDELITY_ROOT / args.run_name
    pairwise_dir, arm_result_path = wait_for_arm_completion(
        args.run_name,
        args.arm_name,
        args.poll_seconds,
    )

    lock_path = run_dir / 'run.lock.json'
    pid = stop_owned_pairwise_controller(run_dir) if lock_path.exists() else None
    cleanup_run_lock(run_dir)
    mark_paused(
        run_dir,
        reason=(
            f'paused after {args.arm_name} completed so loader throughput AB could start; '
            f'pairwise controller pid={pid}'
        ),
    )
    exit_code = run_loader_ab(args)
    print(
        json.dumps(
            {
                'run_name': args.run_name,
                'pairwise_dir': str(pairwise_dir),
                'arm_name': args.arm_name,
                'arm_result_path': str(arm_result_path),
                'pairwise_pid': pid,
                'loader_ab_exit_code': exit_code,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()

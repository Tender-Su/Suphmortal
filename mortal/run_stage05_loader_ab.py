from __future__ import annotations

import argparse
import json
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_stage05_ab as ab
import run_stage05_fidelity as fidelity
import run_stage05_p1_only as p1_only
import stage05_current_defaults as stage05_defaults


REPO_ROOT = Path(__file__).resolve().parent.parent
LOADER_AB_ROOT = REPO_ROOT / 'logs' / 'stage05_loader_ab'
LOADER_AB_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_PROTOCOL_ARM = stage05_defaults.CURRENT_PRIMARY_PROTOCOL_ARM
DEFAULT_RANK_BUDGET_RATIO = 0.15
DEFAULT_OPP_BUDGET_RATIO = 0.03
DEFAULT_DANGER_BUDGET_RATIO = 0.10
DEFAULT_VAL_FILE_BATCH_SIZE = stage05_defaults.DEFAULT_VAL_FILE_BATCH_SIZE
DEFAULT_VAL_PREFETCH_FACTOR = stage05_defaults.DEFAULT_VAL_PREFETCH_FACTOR
RETRY_MARKER = '=== transient training failure detected:'
LOADER_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class LoaderConfig:
    name: str
    num_workers: int
    file_batch_size: int
    prefetch_factor: int
    val_file_batch_size: int
    val_prefetch_factor: int
    batch_size: int = 1024


BASELINE_NUM_WORKERS = 4
BASELINE_FILE_BATCH_SIZE = 10
BASELINE_PREFETCH_FACTOR = 3


def config_name(
    *,
    num_workers: int,
    file_batch_size: int,
    prefetch_factor: int,
    val_file_batch_size: int,
    val_prefetch_factor: int,
    batch_size: int,
) -> str:
    return (
        f'nw{num_workers}_fb{file_batch_size}_pf{prefetch_factor}'
        f'_vfb{val_file_batch_size}_vpf{val_prefetch_factor}'
        f'_bs{batch_size}'
    )


def make_loader_config(
    *,
    num_workers: int,
    file_batch_size: int,
    prefetch_factor: int,
    val_file_batch_size: int,
    val_prefetch_factor: int,
    batch_size: int = 1024,
) -> LoaderConfig:
    return LoaderConfig(
        name=config_name(
            num_workers=num_workers,
            file_batch_size=file_batch_size,
            prefetch_factor=prefetch_factor,
            val_file_batch_size=val_file_batch_size,
            val_prefetch_factor=val_prefetch_factor,
            batch_size=batch_size,
        ),
        num_workers=num_workers,
        file_batch_size=file_batch_size,
        prefetch_factor=prefetch_factor,
        val_file_batch_size=val_file_batch_size,
        val_prefetch_factor=val_prefetch_factor,
        batch_size=batch_size,
    )


def coarse_train_configs(*, batch_size: int = 1024) -> list[LoaderConfig]:
    # Search along a coupled manifold around the current observed
    # winner `nw4_fb10_pf3`, plus a few nearby correlated contenders instead
    # of treating num_workers / file_batch / prefetch as independent axes.
    return [
        make_loader_config(
            num_workers=4,
            file_batch_size=8,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=4,
            file_batch_size=9,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=4,
            file_batch_size=10,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=4,
            file_batch_size=10,
            prefetch_factor=4,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=4,
            file_batch_size=11,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=6,
            file_batch_size=7,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=6,
            file_batch_size=8,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
        make_loader_config(
            num_workers=3,
            file_batch_size=13,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=batch_size,
        ),
    ]


def validation_configs(best: LoaderConfig) -> list[LoaderConfig]:
    candidates = [
        best,
        make_loader_config(
            num_workers=best.num_workers,
            file_batch_size=best.file_batch_size,
            prefetch_factor=best.prefetch_factor,
            val_file_batch_size=7,
            val_prefetch_factor=4,
            batch_size=best.batch_size,
        ),
        make_loader_config(
            num_workers=best.num_workers,
            file_batch_size=best.file_batch_size,
            prefetch_factor=best.prefetch_factor,
            val_file_batch_size=7,
            val_prefetch_factor=5,
            batch_size=best.batch_size,
        ),
        make_loader_config(
            num_workers=best.num_workers,
            file_batch_size=best.file_batch_size,
            prefetch_factor=best.prefetch_factor,
            val_file_batch_size=7,
            val_prefetch_factor=6,
            batch_size=best.batch_size,
        ),
        make_loader_config(
            num_workers=best.num_workers,
            file_batch_size=best.file_batch_size,
            prefetch_factor=best.prefetch_factor,
            val_file_batch_size=8,
            val_prefetch_factor=5,
            batch_size=best.batch_size,
        ),
        make_loader_config(
            num_workers=best.num_workers,
            file_batch_size=best.file_batch_size,
            prefetch_factor=best.prefetch_factor,
            val_file_batch_size=8,
            val_prefetch_factor=6,
            batch_size=best.batch_size,
        ),
        make_loader_config(
            num_workers=best.num_workers,
            file_batch_size=best.file_batch_size,
            prefetch_factor=best.prefetch_factor,
            val_file_batch_size=9,
            val_prefetch_factor=5,
            batch_size=best.batch_size,
        ),
    ]
    return dedupe_loader_configs(candidates)


def dedupe_loader_configs(configs: list[LoaderConfig]) -> list[LoaderConfig]:
    deduped: dict[str, LoaderConfig] = {}
    for config in configs:
        deduped[config.name] = config
    return list(deduped.values())


def top_stable_results(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    stable_results = [result for result in results if result.get('stable')]
    return sorted(
        stable_results,
        key=lambda result: (
            -float(result.get('total_steps_per_sec', 0.0)),
            float(result.get('total_runtime_sec', float('inf'))),
        ),
    )[: max(0, limit)]


def choose_best_stable(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    stable_results = top_stable_results(results, 1)
    if not stable_results:
        return None
    return stable_results[0]


def scaled_phase_steps(step_scale: float) -> dict[str, int]:
    return {
        phase_name: max(1, int(round(steps * step_scale)))
        for phase_name, steps in ab.BASE_SCREENING['phase_steps'].items()
    }


def count_retry_markers(log_path: str | Path) -> int:
    path = Path(log_path)
    if not path.exists():
        return 0
    return path.read_text(encoding='utf-8', errors='ignore').count(RETRY_MARKER)


@contextmanager
def patched_base_screening(overrides: dict[str, Any]):
    original = deepcopy(ab.BASE_SCREENING)
    ab.BASE_SCREENING.update(overrides)
    try:
        yield
    finally:
        ab.BASE_SCREENING.clear()
        ab.BASE_SCREENING.update(original)


def load_reference_candidate(
    *,
    run_name: str,
    protocol_arm: str,
    rank_budget_ratio: float,
    opp_budget_ratio: float,
    danger_budget_ratio: float,
) -> fidelity.CandidateSpec:
    run_dir = fidelity.FIDELITY_ROOT / run_name
    state = fidelity.load_json(run_dir / 'state.json')
    calibration = (state.get('p1') or {}).get('calibration')
    if calibration is None:
        raise ValueError(f'missing p1.calibration in {run_dir / "state.json"}')
    protocols = p1_only.build_protocol_candidates([protocol_arm])
    if len(protocols) != 1:
        raise ValueError(f'expected 1 protocol for {protocol_arm}, got {len(protocols)}')
    return fidelity.make_p1_budget_candidate(
        protocols[0],
        calibration=calibration,
        rank_budget_ratio=rank_budget_ratio,
        opp_budget_ratio=opp_budget_ratio,
        danger_budget_ratio=danger_budget_ratio,
        stage='loader_ab',
        family='rank+opp+danger',
        source_arm='loader_ab_reference',
    )


def loader_benchmark_inputs_signature(
    *,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
) -> str:
    return fidelity.stable_payload_digest(
        {
            'schema_version': LOADER_CACHE_SCHEMA_VERSION,
            'base_cfg': base_cfg,
            'grouped': grouped,
            'eval_splits': eval_splits,
        }
    )


def loader_cache_signature(
    *,
    suite_name: str,
    round_name: str,
    config: LoaderConfig,
    benchmark_inputs_signature: str,
    candidate: fidelity.CandidateSpec,
    seed: int,
    step_scale: float,
    val_every_steps: int,
    monitor_val_batches: int,
    full_recent_files: int,
    old_regression_files: int,
    phase_name: str,
) -> str:
    return fidelity.stable_payload_digest(
        {
            'schema_version': LOADER_CACHE_SCHEMA_VERSION,
            'suite_name': suite_name,
            'round_name': round_name,
            'config': asdict(config),
            'benchmark_inputs_signature': benchmark_inputs_signature,
            'candidate_arm_name': candidate.arm_name,
            'candidate_cfg_overrides': candidate.cfg_overrides,
            'seed': seed,
            'step_scale': step_scale,
            'phase_name': phase_name,
            'val_every_steps': val_every_steps,
            'monitor_val_batches': monitor_val_batches,
            'full_recent_files': full_recent_files,
            'old_regression_files': old_regression_files,
        }
    )


def run_loader_config(
    *,
    suite_name: str,
    round_name: str,
    config: LoaderConfig,
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    benchmark_inputs_signature: str,
    candidate: fidelity.CandidateSpec,
    seed: int,
    step_scale: float,
    val_every_steps: int,
    monitor_val_batches: int,
    full_recent_files: int,
    old_regression_files: int,
    phase_name: str,
) -> dict[str, Any]:
    suite_dir = LOADER_AB_ROOT / suite_name
    round_result_dir = suite_dir / round_name
    result_path = round_result_dir / f'{config.name}.json'
    benchmark_ab_name = f'{suite_name}__{round_name}'
    benchmark_arm_name = f'{candidate.arm_name}__{config.name}'
    signature = loader_cache_signature(
        suite_name=suite_name,
        round_name=round_name,
        config=config,
        benchmark_inputs_signature=benchmark_inputs_signature,
        candidate=candidate,
        seed=seed,
        step_scale=step_scale,
        phase_name=phase_name,
        val_every_steps=val_every_steps,
        monitor_val_batches=monitor_val_batches,
        full_recent_files=full_recent_files,
        old_regression_files=old_regression_files,
    )
    if result_path.exists():
        cached = fidelity.load_json(result_path)
        if cached.get('signature') == signature:
            return cached

    phase_steps = scaled_phase_steps(step_scale)
    benchmark_steps = int(phase_steps[phase_name])
    screening_overrides = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'file_batch_size': config.file_batch_size,
        'val_file_batch_size': config.val_file_batch_size,
        'prefetch_factor': config.prefetch_factor,
        'val_prefetch_factor': config.val_prefetch_factor,
        'val_every_steps': val_every_steps,
        'monitor_val_batches': monitor_val_batches,
        'eval_files': {
            'full_recent': full_recent_files,
            'old_regression': old_regression_files,
        },
        'phase_steps': phase_steps,
        'log_every': min(ab.BASE_SCREENING['log_every'], max(250, benchmark_steps // 3)),
        'save_every': benchmark_steps,
    }

    round_result_dir.mkdir(parents=True, exist_ok=True)
    candidate_base_cfg = ab.merge_dict(base_cfg, deepcopy(candidate.cfg_overrides))
    phase_results: dict[str, Any] = {}
    phase_timings: dict[str, Any] = {}
    total_runtime_sec = 0.0
    total_steps = 0
    success = False
    error = None
    tb = None

    previous_ab_root = ab.AB_ROOT
    ab.AB_ROOT = LOADER_AB_ROOT
    ab.AB_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with patched_base_screening(screening_overrides):
            arm_root = ab.AB_ROOT / benchmark_ab_name / benchmark_arm_name
            if arm_root.exists():
                fidelity.remove_tree_with_retries(arm_root)
            scheduler_type = ab.SCHEDULER_PROFILES[candidate.scheduler_profile][phase_name]
            started = time.perf_counter()
            phase_result = ab.run_phase(
                candidate_base_cfg,
                grouped,
                ab_name=benchmark_ab_name,
                arm_name=benchmark_arm_name,
                phase_name=phase_name,
                scheduler_type=scheduler_type,
                weight_profile=candidate.weight_profile,
                window_profile=candidate.window_profile,
                seed=seed,
                eval_splits=eval_splits,
                init_state_file=None,
                step_scale=step_scale,
            )
            runtime_sec = time.perf_counter() - started
            retry_count = count_retry_markers(phase_result['log_path'])
            steps = int(phase_result['latest']['optimizer_steps'])
            phase_results[phase_name] = phase_result
            phase_timings[phase_name] = {
                'runtime_sec': runtime_sec,
                'steps': steps,
                'steps_per_sec': 0.0 if runtime_sec <= 0 else steps / runtime_sec,
                'retry_count': retry_count,
                'log_path': phase_result['log_path'],
                'config_path': phase_result['config_path'],
            }
            total_runtime_sec += runtime_sec
            total_steps += steps
        success = True
    except Exception as exc:
        error = str(exc)
        tb = traceback.format_exc()
    finally:
        ab.AB_ROOT = previous_ab_root

    total_retry_count = sum(int(phase['retry_count']) for phase in phase_timings.values())
    stable = success and total_retry_count == 0
    payload = {
        'signature': signature,
        'benchmark_inputs_signature': benchmark_inputs_signature,
        'suite_name': suite_name,
        'round_name': round_name,
        'config': asdict(config),
        'candidate': {
            'arm_name': candidate.arm_name,
            'protocol_arm': candidate.meta.get('protocol_arm'),
            'rank_budget_ratio': candidate.meta.get('rank_budget_ratio'),
            'opp_budget_ratio': candidate.meta.get('opp_budget_ratio'),
            'danger_budget_ratio': candidate.meta.get('danger_budget_ratio'),
        },
        'seed': seed,
        'step_scale': step_scale,
        'benchmark_phase': phase_name,
        'success': success,
        'stable': stable,
        'total_retry_count': total_retry_count,
        'total_runtime_sec': total_runtime_sec,
        'total_steps': total_steps,
        'total_steps_per_sec': 0.0 if total_runtime_sec <= 0 else total_steps / total_runtime_sec,
        'phase_order': [phase_name],
        'phase_timings': phase_timings,
        'phase_results': phase_results,
        'error': error,
        'traceback': tb,
    }
    fidelity.atomic_write_json(result_path, payload)
    return payload


def run_round(
    *,
    suite_name: str,
    round_name: str,
    configs: list[LoaderConfig],
    base_cfg: dict[str, Any],
    grouped: dict[str, list[str]],
    eval_splits: dict[str, list[str]],
    benchmark_inputs_signature: str,
    candidate: fidelity.CandidateSpec,
    seed: int,
    step_scale: float,
    val_every_steps: int,
    monitor_val_batches: int,
    full_recent_files: int,
    old_regression_files: int,
    summary_payload: dict[str, Any],
    summary_path: Path,
    phase_name: str,
) -> dict[str, Any]:
    results = []
    for config in configs:
        result = run_loader_config(
            suite_name=suite_name,
            round_name=round_name,
            config=config,
            base_cfg=base_cfg,
            grouped=grouped,
            eval_splits=eval_splits,
            benchmark_inputs_signature=benchmark_inputs_signature,
            candidate=candidate,
            seed=seed,
            step_scale=step_scale,
            val_every_steps=val_every_steps,
            monitor_val_batches=monitor_val_batches,
            full_recent_files=full_recent_files,
            old_regression_files=old_regression_files,
            phase_name=phase_name,
        )
        results.append(result)
        summary_payload['rounds'][round_name] = {
            'best': None,
            'results': results,
        }
        fidelity.atomic_write_json(summary_path, summary_payload)
    best = choose_best_stable(results)
    round_payload = {
        'best': None if best is None else best['config']['name'],
        'results': results,
    }
    summary_payload['rounds'][round_name] = round_payload
    if best is not None:
        summary_payload['best_so_far'] = best
    fidelity.atomic_write_json(summary_path, summary_payload)
    return round_payload


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(
        grouped,
        args.seed + 77,
        {
            'full_recent': args.full_recent_files,
            'old_regression': args.old_regression_files,
        },
    )
    benchmark_inputs_signature = loader_benchmark_inputs_signature(
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
    )
    candidate = load_reference_candidate(
        run_name=args.run_name,
        protocol_arm=args.protocol_arm,
        rank_budget_ratio=args.rank_budget_ratio,
        opp_budget_ratio=args.opp_budget_ratio,
        danger_budget_ratio=args.danger_budget_ratio,
    )

    suite_name = args.suite_name or (
        f'{args.run_name}_loader_ab'
        f'_r{int(round(args.rank_budget_ratio * 100)):03d}'
        f'_o{int(round(args.opp_budget_ratio * 100)):03d}'
        f'_d{int(round(args.danger_budget_ratio * 100)):03d}'
    )
    summary_path = LOADER_AB_ROOT / suite_name / 'summary.json'
    summary_payload: dict[str, Any] = {
        'suite_name': suite_name,
        'reference_candidate': {
            'arm_name': candidate.arm_name,
            'protocol_arm': candidate.meta.get('protocol_arm'),
            'rank_budget_ratio': candidate.meta.get('rank_budget_ratio'),
            'opp_budget_ratio': candidate.meta.get('opp_budget_ratio'),
            'danger_budget_ratio': candidate.meta.get('danger_budget_ratio'),
        },
        'benchmark_settings': {
            'seed': args.seed,
            'step_scale': args.step_scale,
            'benchmark_phase': args.phase_name,
            'val_every_steps': args.val_every_steps,
            'monitor_val_batches': args.monitor_val_batches,
            'full_recent_files': args.full_recent_files,
            'old_regression_files': args.old_regression_files,
            'benchmark_inputs_signature': benchmark_inputs_signature,
        },
        'rounds': {},
        'best_so_far': None,
    }
    fidelity.atomic_write_json(summary_path, summary_payload)

    train_scan = run_round(
        suite_name=suite_name,
        round_name='train_scan',
        configs=coarse_train_configs(batch_size=args.batch_size),
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
        summary_payload=summary_payload,
        summary_path=summary_path,
        phase_name=args.phase_name,
    )
    train_finalists = top_stable_results(train_scan['results'], args.training_finalists)
    if not train_finalists:
        summary_payload['overall_best'] = None
        fidelity.atomic_write_json(summary_path, summary_payload)
        return summary_payload

    validation_pool: list[LoaderConfig] = []
    for finalist in train_finalists:
        validation_pool.extend(validation_configs(LoaderConfig(**finalist['config'])))
    validation_pool = dedupe_loader_configs(validation_pool)

    validate = run_round(
        suite_name=suite_name,
        round_name='validation_followup',
        configs=validation_pool,
        base_cfg=base_cfg,
        grouped=grouped,
        eval_splits=eval_splits,
        benchmark_inputs_signature=benchmark_inputs_signature,
        candidate=candidate,
        seed=args.seed + 1000,
        step_scale=args.step_scale,
        val_every_steps=args.val_every_steps,
        monitor_val_batches=args.monitor_val_batches,
        full_recent_files=args.full_recent_files,
        old_regression_files=args.old_regression_files,
        summary_payload=summary_payload,
        summary_path=summary_path,
        phase_name=args.phase_name,
    )
    overall_best = choose_best_stable(validate['results']) or train_finalists[0]
    summary_payload['overall_best'] = overall_best
    fidelity.atomic_write_json(summary_path, summary_payload)
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--suite-name', default='')
    parser.add_argument('--protocol-arm', default=DEFAULT_PROTOCOL_ARM)
    parser.add_argument('--rank-budget-ratio', type=float, default=DEFAULT_RANK_BUDGET_RATIO)
    parser.add_argument('--opp-budget-ratio', type=float, default=DEFAULT_OPP_BUDGET_RATIO)
    parser.add_argument('--danger-budget-ratio', type=float, default=DEFAULT_DANGER_BUDGET_RATIO)
    parser.add_argument('--seed', type=int, default=20260326)
    parser.add_argument('--step-scale', type=float, default=0.25)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--phase-name', default='phase_a')
    parser.add_argument('--training-finalists', type=int, default=2)
    parser.add_argument('--val-every-steps', type=int, default=750)
    parser.add_argument('--monitor-val-batches', type=int, default=64)
    parser.add_argument('--full-recent-files', type=int, default=64)
    parser.add_argument('--old-regression-files', type=int, default=32)
    return parser.parse_args()


def main() -> None:
    summary = run_suite(parse_args())
    print(json.dumps(fidelity.normalize_payload(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

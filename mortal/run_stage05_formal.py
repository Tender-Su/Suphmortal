from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import run_stage05_ab as ab
import stage05_current_defaults as stage05_defaults

CURRENT_PRIMARY_PROTOCOL_ARM = stage05_defaults.CURRENT_PRIMARY_PROTOCOL_ARM
CURRENT_STAGE1_TOP_PROTOCOL_ARMS = stage05_defaults.CURRENT_STAGE1_TOP_PROTOCOL_ARMS
# Compatibility alias for older callers that still expect the historical name.
CURRENT_STAGE1_TOP4 = CURRENT_STAGE1_TOP_PROTOCOL_ARMS


FORMAL_DEFAULTS = {
    'batch_size': 1024,
    'num_workers': 4,
    'file_batch_size': 10,
    'val_file_batch_size': stage05_defaults.DEFAULT_VAL_FILE_BATCH_SIZE,
    'prefetch_factor': 3,
    'val_prefetch_factor': stage05_defaults.DEFAULT_VAL_PREFETCH_FACTOR,
    'force_safe_training': False,
    'log_every': 1000,
    'save_every': 2000,
    'val_every_steps': 20000,
    'monitor_val_batches': 512,
    'full_val_every_checks': 1,
    'old_regression_every_checks': 0,
    'max_epochs': 99,
    'phase_steps': {
        'phase_a': 6000,
        'phase_b': 4000,
        'phase_c': 2000,
    },
    'phase_train_pool': {
        'phase_a': 0,
        'phase_b': 0,
        'phase_c': 0,
    },
    'eval_files': {
        'full_recent': 512,
        'old_regression': 256,
    },
    'seed': 20260312,
}


def resolve_config_path(value: str, *, config_dir: Path | None = None) -> Path:
    if config_dir is None:
        config_dir = ab.BASE_CFG_PATH.parent
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def stage05_publish_plan(
    base_cfg: dict,
    payload: dict,
    *,
    config_dir: Path | None = None,
) -> list[tuple[Path, Path]]:
    if config_dir is None:
        config_dir = ab.BASE_CFG_PATH.parent

    supervised_cfg = base_cfg.get('supervised', {})
    if not isinstance(supervised_cfg, dict):
        return []

    candidates = payload.get('candidates', {})
    winner = payload.get('winner')
    if winner not in candidates:
        raise KeyError(f'missing selected Stage 0.5 winner in payload: {winner!r}')

    plan: dict[Path, tuple[int, Path]] = {}
    secondary_target_priority = 10
    latest_target_priority = 20
    canonical_target_priority = 30

    def candidate_path(name: str) -> Path:
        candidate = candidates.get(name)
        if not isinstance(candidate, dict) or not candidate.get('path'):
            raise KeyError(f'missing Stage 0.5 candidate path for {name!r}')
        return Path(candidate['path']).resolve()

    def add_target(destination: str, source: Path, *, priority: int) -> None:
        if not destination:
            return
        dest_path = resolve_config_path(destination, config_dir=config_dir)
        existing = plan.get(dest_path)
        if existing is None or priority >= existing[0]:
            plan[dest_path] = (priority, source)

    winner_source = candidate_path(winner)
    add_target(
        supervised_cfg.get('state_file', ''),
        candidate_path('latest'),
        priority=latest_target_priority,
    )
    add_target(
        supervised_cfg.get('best_acc_state_file', ''),
        candidate_path('best_acc'),
        priority=secondary_target_priority,
    )
    add_target(
        supervised_cfg.get('best_rank_state_file', ''),
        candidate_path('best_rank'),
        priority=secondary_target_priority,
    )

    best_state_file = supervised_cfg.get('best_state_file', '')
    best_loss_state_file = supervised_cfg.get('best_loss_state_file', best_state_file)
    # Keep both supervised seed aliases on the formal winner so Stage 1 fallback
    # paths do not silently boot from an unselected checkpoint.
    for canonical_seed_file in dict.fromkeys((best_state_file, best_loss_state_file)):
        add_target(
            canonical_seed_file,
            winner_source,
            priority=canonical_target_priority,
        )

    return [
        (destination, source)
        for destination, (_, source) in plan.items()
    ]


def publish_stage05_canonical_checkpoints(
    base_cfg: dict,
    payload: dict,
    *,
    config_dir: Path | None = None,
    protocol_arm: str | None = None,
) -> list[dict[str, str]]:
    if protocol_arm is not None and protocol_arm != CURRENT_PRIMARY_PROTOCOL_ARM:
        return []

    published = []
    for destination, source in stage05_publish_plan(base_cfg, payload, config_dir=config_dir):
        if not source.exists():
            raise FileNotFoundError(f'Stage 0.5 publish source does not exist: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copy2(source, destination)
        published.append({
            'source': str(source),
            'destination': str(destination),
        })
    return published


def build_protocol_arm_map() -> dict[str, tuple[str, str, str, str]]:
    arms: dict[str, tuple[str, str, str, str]] = {}
    for scheduler_prefix, scheduler_profile in ab.SCHEDULER_PREFIXES:
        for curriculum_prefix, curriculum_profile in ab.CURRICULUM_PREFIXES:
            for weight_prefix, weight_profile in ab.WEIGHT_PREFIXES:
                for window_prefix, window_profile in ab.WINDOW_PREFIXES:
                    arm_name = (
                        f'{scheduler_prefix}_{curriculum_prefix}{weight_prefix}{window_prefix}_'
                        f'{scheduler_profile}_{curriculum_profile}_{weight_profile}_{window_profile}'
                    )
                    arms[arm_name] = (
                        scheduler_profile,
                        curriculum_profile,
                        weight_profile,
                        window_profile,
                    )
    return arms


PROTOCOL_ARM_MAP = build_protocol_arm_map()


def apply_formal_defaults() -> None:
    ab.BASE_SCREENING.clear()
    ab.BASE_SCREENING.update(deepcopy(FORMAL_DEFAULTS))


def finalize_formal_result(
    base_cfg: dict,
    result: dict,
    *,
    protocol_arm: str | None = None,
    config_dir: Path | None = None,
) -> dict:
    result['selected_protocol_arm'] = protocol_arm
    result['current_primary_protocol_arm'] = CURRENT_PRIMARY_PROTOCOL_ARM
    result['current_stage1_top4'] = list(CURRENT_STAGE1_TOP4)
    result['current_stage1_top_protocol_arms'] = list(CURRENT_STAGE1_TOP_PROTOCOL_ARMS)
    result['published_canonical_checkpoints'] = publish_stage05_canonical_checkpoints(
        base_cfg,
        result,
        config_dir=config_dir,
        protocol_arm=protocol_arm,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab-name')
    parser.add_argument(
        '--protocol-arm',
        default=CURRENT_PRIMARY_PROTOCOL_ARM,
        choices=sorted(PROTOCOL_ARM_MAP),
        help='Stage 0.5 formal protocol arm. Default is the current primary candidate, not the old historical placeholder.',
    )
    parser.add_argument('--list-protocol-arms', action='store_true')
    parser.add_argument('--step-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=FORMAL_DEFAULTS['seed'])
    args = parser.parse_args()

    if args.list_protocol_arms:
        print(json.dumps(
            {
                'current_primary_protocol_arm': CURRENT_PRIMARY_PROTOCOL_ARM,
                'current_stage1_top4': list(CURRENT_STAGE1_TOP4),
                'current_stage1_top_protocol_arms': list(CURRENT_STAGE1_TOP_PROTOCOL_ARMS),
                'all_protocol_arms': sorted(PROTOCOL_ARM_MAP),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    apply_formal_defaults()
    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    scheduler_profile, curriculum_profile, weight_profile, window_profile = PROTOCOL_ARM_MAP[args.protocol_arm]
    ab_name = args.ab_name or f's05_formal_{args.protocol_arm}'
    result = ab.run_ab6_checkpoint(
        base_cfg,
        grouped,
        seed=args.seed,
        scheduler_profile=scheduler_profile,
        curriculum_profile=curriculum_profile,
        weight_profile=weight_profile,
        window_profile=window_profile,
        step_scale=args.step_scale,
        ab_name=ab_name,
    )
    result = finalize_formal_result(
        base_cfg,
        result,
        protocol_arm=args.protocol_arm,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

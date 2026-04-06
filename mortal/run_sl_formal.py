from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import run_sl_ab as ab
import sl_current_defaults as sl_defaults

CURRENT_PRIMARY_PROTOCOL_ARM = sl_defaults.CURRENT_PRIMARY_PROTOCOL_ARM
CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS = sl_defaults.CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS
CURRENT_SUPERVISED_TOP4 = CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS

SL_FIDELITY_ROOT = ab.REPO_ROOT / 'logs' / 'sl_fidelity'
FORMAL_CONFIG_SNAPSHOT_SCHEMA_VERSION = 1
CONFIG_PATH_KEYS = {
    'init_state_file',
    'state_file',
    'latest_state_file',
    'best_loss_state_file',
    'best_acc_state_file',
    'best_state_file',
    'tensorboard_dir',
    'log_dir',
    'file_index',
    'buffer_dir',
    'drain_dir',
    'dir',
    'tactics',
}
CONFIG_GLOB_KEYS = {
    'globs',
    'train_globs',
    'val_globs',
}


FORMAL_DEFAULTS = {
    'batch_size': 1024,
    'num_workers': 4,
    'file_batch_size': 10,
    'val_file_batch_size': sl_defaults.DEFAULT_VAL_FILE_BATCH_SIZE,
    'prefetch_factor': 3,
    'val_prefetch_factor': sl_defaults.DEFAULT_VAL_PREFETCH_FACTOR,
    'force_safe_training': False,
    'log_every': 1000,
    'save_every': 2000,
    'val_every_steps': 20000,
    'monitor_val_batches': 512,
    'full_val_every_checks': 1,
    'old_regression_every_checks': 0,
    'max_epochs': 99,
    'phase_steps': {
        'phase_a': 9000,
        'phase_b': 6000,
        'phase_c': 3000,
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

FORMAL_SHORTLIST_CHECKPOINT_TYPES = (
    'best_loss',
    'best_acc',
    'best_rank',
)


def resolve_config_path(value: str, *, config_dir: Path | None = None) -> Path:
    if config_dir is None:
        config_dir = ab.BASE_CFG_PATH.parent
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def resolve_snapshot_paths(node, *, config_dir: Path):
    if isinstance(node, dict):
        resolved = {}
        for key, value in node.items():
            if isinstance(value, dict):
                resolved[key] = resolve_snapshot_paths(value, config_dir=config_dir)
            elif key in CONFIG_PATH_KEYS and isinstance(value, str):
                resolved[key] = str(resolve_config_path(value, config_dir=config_dir))
            elif key in CONFIG_GLOB_KEYS and isinstance(value, list):
                resolved[key] = [
                    str(resolve_config_path(item, config_dir=config_dir))
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved
    return node


def build_formal_config_snapshot(
    base_cfg: dict,
    *,
    config_path: Path | None = None,
) -> dict:
    frozen_config_path = Path(config_path or ab.BASE_CFG_PATH).resolve()
    base_cfg_snapshot = deepcopy(base_cfg)
    base_1v3_cfg = base_cfg_snapshot.get('1v3', {})
    if not isinstance(base_1v3_cfg, dict):
        base_1v3_cfg = {}
    return {
        'schema_version': FORMAL_CONFIG_SNAPSHOT_SCHEMA_VERSION,
        'config_path': str(frozen_config_path),
        'config_dir': str(frozen_config_path.parent),
        'base_cfg': base_cfg_snapshot,
        'base_1v3_cfg': resolve_snapshot_paths(
            deepcopy(base_1v3_cfg),
            config_dir=frozen_config_path.parent,
        ),
    }


def sl_publish_plan(
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
        raise KeyError(f'missing selected supervised winner in payload: {winner!r}')

    plan: dict[Path, tuple[int, Path]] = {}
    secondary_target_priority = 10
    winner_alias_priority = 30

    def candidate_path(name: str) -> Path:
        candidate = candidates.get(name)
        if not isinstance(candidate, dict) or not candidate.get('path'):
            raise KeyError(f'missing supervised candidate path for {name!r}')
        return Path(candidate['path']).resolve()

    def add_target(destination: str, source: Path, *, priority: int) -> None:
        if not destination:
            return
        dest_path = resolve_config_path(destination, config_dir=config_dir)
        existing = plan.get(dest_path)
        if existing is None or priority >= existing[0]:
            plan[dest_path] = (priority, source)

    winner_source = candidate_path(winner)
    # `latest` is intentionally discarded once formal enters the 1v3 handoff
    # flow. Keep the canonical exact-resume alias on the selected winner so the
    # final supervised package stays self-consistent.
    add_target(
        supervised_cfg.get('state_file', ''),
        winner_source,
        priority=winner_alias_priority,
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
    # Keep both supervised seed aliases on the formal winner so downstream
    # consumers do not silently boot from an unselected checkpoint.
    for canonical_seed_file in dict.fromkeys((best_state_file, best_loss_state_file)):
        add_target(
            canonical_seed_file,
            winner_source,
            priority=winner_alias_priority,
        )

    return [
        (destination, source)
        for destination, (_, source) in plan.items()
    ]


def sl_publish_targets(
    base_cfg: dict,
    payload: dict,
    *,
    config_dir: Path | None = None,
    protocol_arm: str | None = None,
    primary_protocol_arm: str | None = CURRENT_PRIMARY_PROTOCOL_ARM,
) -> list[Path]:
    normalized_protocol_arm = str(protocol_arm or '').strip() or None
    normalized_primary_protocol_arm = (
        str(primary_protocol_arm or '').strip()
        or str(CURRENT_PRIMARY_PROTOCOL_ARM or '').strip()
        or None
    )
    if (
        normalized_protocol_arm is not None
        and normalized_primary_protocol_arm is not None
        and normalized_protocol_arm != normalized_primary_protocol_arm
    ):
        return []
    return [
        destination
        for destination, _ in sl_publish_plan(base_cfg, payload, config_dir=config_dir)
    ]


def publish_sl_canonical_checkpoints(
    base_cfg: dict,
    payload: dict,
    *,
    config_dir: Path | None = None,
    protocol_arm: str | None = None,
    primary_protocol_arm: str | None = CURRENT_PRIMARY_PROTOCOL_ARM,
) -> list[dict[str, str]]:
    published = []
    allowed_targets = set(
        sl_publish_targets(
            base_cfg,
            payload,
            config_dir=config_dir,
            protocol_arm=protocol_arm,
            primary_protocol_arm=primary_protocol_arm,
        )
    )
    for destination, source in sl_publish_plan(base_cfg, payload, config_dir=config_dir):
        if destination not in allowed_targets:
            continue
        if not source.exists():
            raise FileNotFoundError(f'supervised canonical-alias source does not exist: {source}')
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


def build_formal_shortlist_candidates(result: dict) -> dict[str, dict]:
    candidates = result.get('candidates')
    if not isinstance(candidates, dict):
        raise RuntimeError('formal result is missing checkpoint candidates')
    shortlist: dict[str, dict] = {}
    missing: list[str] = []
    for checkpoint_type in FORMAL_SHORTLIST_CHECKPOINT_TYPES:
        candidate = candidates.get(checkpoint_type)
        if not isinstance(candidate, dict) or not candidate.get('path'):
            missing.append(checkpoint_type)
            continue
        shortlist[checkpoint_type] = candidate
    if missing:
        raise RuntimeError(
            'formal checkpoint pack is missing required checkpoint types: '
            + ', '.join(sorted(missing))
        )
    return shortlist


def resolve_formal_checkpoint_pack_winner(
    *,
    offline_winner: str | None,
    candidates: dict[str, dict],
) -> str:
    if offline_winner in candidates:
        return str(offline_winner)
    if 'best_loss' in candidates:
        return 'best_loss'
    raise RuntimeError(
        'formal result cannot resolve a retained checkpoint-pack winner after latest discard: '
        f'{offline_winner!r}'
    )


def finalize_formal_result(
    base_cfg: dict,
    result: dict,
    *,
    protocol_arm: str | None = None,
    config_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    offline_winner = str(result.get('winner', '') or '').strip() or None
    result['offline_checkpoint_winner'] = offline_winner
    result['candidates'] = build_formal_shortlist_candidates(result)
    checkpoint_pack_winner = resolve_formal_checkpoint_pack_winner(
        offline_winner=offline_winner,
        candidates=result['candidates'],
    )
    result['checkpoint_pack_winner'] = checkpoint_pack_winner
    config_snapshot = build_formal_config_snapshot(
        base_cfg,
        config_path=config_path,
    )
    result['shortlist_checkpoint_types'] = list(FORMAL_SHORTLIST_CHECKPOINT_TYPES)
    result['checkpoint_pack_types'] = list(FORMAL_SHORTLIST_CHECKPOINT_TYPES)
    result['latest_discarded'] = True
    result['publish_pending'] = True
    result['canonical_alias_sync_pending'] = True
    result['config_snapshot'] = config_snapshot
    result['pending_canonical_alias_targets'] = [
        str(destination)
        for destination in sl_publish_targets(
            config_snapshot['base_cfg'],
            {
                'winner': checkpoint_pack_winner,
                'candidates': result['candidates'],
            },
            config_dir=Path(str(config_snapshot['config_dir'])),
            protocol_arm=protocol_arm,
            primary_protocol_arm=CURRENT_PRIMARY_PROTOCOL_ARM,
        )
    ]
    result['selected_protocol_arm'] = protocol_arm
    result['current_primary_protocol_arm'] = CURRENT_PRIMARY_PROTOCOL_ARM
    result['current_supervised_top4'] = list(CURRENT_SUPERVISED_TOP4)
    result['current_supervised_top_protocol_arms'] = list(CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS)
    return result


def load_sl_state(path: Path) -> dict | None:
    try:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def iter_sl_state_paths() -> list[Path]:
    if not SL_FIDELITY_ROOT.exists():
        return []
    state_paths = [
        path
        for path in SL_FIDELITY_ROOT.glob('*/state.json')
        if path.is_file()
    ]
    state_paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return state_paths


def resolve_pending_canonical_alias_targets(state: dict) -> list[str]:
    formal_state = state.get('formal', {})
    if not isinstance(formal_state, dict):
        return []
    result = formal_state.get('result', {})
    if not isinstance(result, dict):
        return []
    pending_targets = result.get('pending_canonical_alias_targets')
    if isinstance(pending_targets, list):
        return [str(target) for target in pending_targets if str(target)]

    candidates = result.get('candidates')
    if not isinstance(candidates, dict):
        return []
    winner = str(
        result.get('checkpoint_pack_winner')
        or result.get('offline_checkpoint_winner')
        or result.get('winner')
        or ''
    ).strip()
    if winner not in candidates:
        return []

    snapshot = result.get('config_snapshot')
    if isinstance(snapshot, dict):
        frozen_base_cfg = snapshot.get('base_cfg')
        config_dir_value = snapshot.get('config_dir')
        if isinstance(frozen_base_cfg, dict) and config_dir_value:
            return [
                str(destination)
                for destination in sl_publish_targets(
                    frozen_base_cfg,
                    {'winner': winner, 'candidates': candidates},
                    config_dir=Path(str(config_dir_value)),
                    protocol_arm=result.get('selected_protocol_arm'),
                    primary_protocol_arm=result.get('current_primary_protocol_arm'),
                )
            ]

    return [
        str(destination)
        for destination in sl_publish_targets(
            ab.build_base_config(),
            {'winner': winner, 'candidates': candidates},
            config_dir=ab.BASE_CFG_PATH.parent,
            protocol_arm=result.get('selected_protocol_arm'),
            primary_protocol_arm=result.get('current_primary_protocol_arm'),
        )
    ]


def resolve_published_canonical_alias_targets(state: dict) -> list[str]:
    formal_1v3_state = state.get('formal_1v3')
    if not isinstance(formal_1v3_state, dict):
        return []
    published = formal_1v3_state.get('published_canonical_checkpoints')
    if not isinstance(published, list):
        return []
    targets: list[str] = []
    for entry in published:
        if not isinstance(entry, dict):
            continue
        destination = str(entry.get('destination') or '').strip()
        if destination:
            targets.append(destination)
    return targets


def sl_canonical_handoff_complete(state: dict) -> bool:
    pending_targets = resolve_pending_canonical_alias_targets(state)
    if not pending_targets:
        return True
    published_targets = {
        Path(target).resolve()
        for target in resolve_published_canonical_alias_targets(state)
    }
    return all(Path(target).resolve() in published_targets for target in pending_targets)


def find_pending_sl_handoff_for_init_state(init_state_file: str) -> dict | None:
    resolved_init_state = Path(init_state_file).resolve()
    for state_path in iter_sl_state_paths():
        state = load_sl_state(state_path)
        if state is None:
            continue
        pending_targets = resolve_pending_canonical_alias_targets(state)
        if not pending_targets:
            continue
        if not any(Path(target).resolve() == resolved_init_state for target in pending_targets):
            continue
        final_conclusion = state.get('final_conclusion', {})
        if not isinstance(final_conclusion, dict):
            continue
        if sl_canonical_handoff_complete(state):
            return None
        return {
            'run_name': state_path.parent.name,
            'state_path': str(state_path),
            'formal_status': str(final_conclusion.get('formal_status') or ''),
            'formal_1v3_status': str(final_conclusion.get('formal_1v3_status') or ''),
            'init_state_file': str(resolved_init_state),
        }
    return None


def ensure_supervised_canonical_handoff_ready(init_state_file: str) -> None:
    if not init_state_file:
        return
    blocker = find_pending_sl_handoff_for_init_state(init_state_file)
    if blocker is None:
        return
    raise RuntimeError(
        'canonical supervised init checkpoint is blocked by a pending supervised formal_1v3 handoff: '
        f'init_state_file={blocker["init_state_file"]}, '
        f'run={blocker["run_name"]}, '
        f'formal_status={blocker["formal_status"]}, '
        f'formal_1v3_status={blocker["formal_1v3_status"]}, '
        f'state={blocker["state_path"]}. '
        'Finish formal_1v3 before using the canonical supervised checkpoint, or pass a non-canonical explicit init checkpoint.'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab-name')
    parser.add_argument(
        '--protocol-arm',
        default=CURRENT_PRIMARY_PROTOCOL_ARM,
        choices=sorted(PROTOCOL_ARM_MAP),
        help='Supervised formal protocol arm. Default is the current primary candidate, not the old historical placeholder.',
    )
    parser.add_argument('--list-protocol-arms', action='store_true')
    parser.add_argument('--step-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=FORMAL_DEFAULTS['seed'])
    args = parser.parse_args()

    if args.list_protocol_arms:
        print(json.dumps(
            {
                'current_primary_protocol_arm': CURRENT_PRIMARY_PROTOCOL_ARM,
                'current_supervised_top4': list(CURRENT_SUPERVISED_TOP4),
                'current_supervised_top_protocol_arms': list(CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS),
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
    ab_name = args.ab_name or f'sl_formal_{args.protocol_arm}'
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

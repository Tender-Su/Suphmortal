from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from mortal._repo import MORTAL_ROOT, REPO_ROOT
from mortal.core.cpu_affinity import AFFINITY_ENV_VAR
from mortal.supervised.sl_selection import (
    LOSS_EPSILON,
    action_quality_score,
    refresh_scenario_quality_score,
    refresh_selection_quality_score,
    scenario_quality_score,
    selection_tiebreak_key,
)
from mortal.core.toml_utils import load_toml_file, write_toml_file


MORTAL_DIR = MORTAL_ROOT
BASE_CFG_PATH = MORTAL_DIR / 'config.toml'
BASE_INDEX_PATH = MORTAL_DIR / 'checkpoints' / 'file_index_supervised_json.pth'
AB_ROOT = REPO_ROOT / 'logs' / 'sl_ab'
AB_ROOT.mkdir(parents=True, exist_ok=True)
PHASE_PLAN_SCHEMA_VERSION = 1

PROVENANCE_IGNORED_SUPERVISED_KEYS = {
    'state_file',
    'best_state_file',
    'best_loss_state_file',
    'best_acc_state_file',
    'best_rank_state_file',
    'tensorboard_dir',
    'file_index',
    'init_state_file',
    'run_provenance',
    'num_workers',
    'file_batch_size',
    'val_file_batch_size',
    'prefetch_factor',
    'val_prefetch_factor',
    'rayon_num_threads',
    'worker_torch_num_threads',
    'worker_torch_num_interop_threads',
    'force_safe_training',
}


BASE_SCREENING = {
    'batch_size': 1024,
    'num_workers': 4,
    'file_batch_size': 10,
    'val_file_batch_size': 8,
    'prefetch_factor': 3,
    'val_prefetch_factor': 5,
    'force_safe_training': False,
    'log_every': 1000,
    'save_every': 4000,
    'val_every_steps': 4000,
    'monitor_val_batches': 128,
    'full_val_every_checks': 1,
    'old_regression_every_checks': 1,
    'max_epochs': 99,
    'phase_steps': {
        'phase_a': 6000,
        'phase_b': 4000,
        'phase_c': 2000,
    },
    'phase_train_pool': {
        'phase_a': 180000,
        'phase_b': 120000,
        'phase_c': 80000,
    },
    'eval_files': {
        'full_recent': 128,
        'old_regression': 64,
    },
    'seed': 20260312,
}

WINDOWS = {
    'monitor_recent': ('202601', '202601'),
    'full_recent': ('202501', '202512'),
    'old_regression': ('202201', '202212'),
    'broad_all': ('200901', '202412'),
    'early': ('200901', '202012'),
    'mid': ('202101', '202212'),
    'recent_24': ('202301', '202412'),
    'recent_12': ('202401', '202412'),
    'recent_6': ('202407', '202412'),
}


WEIGHT_PROFILES = {
    'mild': {
        'phase_a': ([0.40, 0.30, 0.30], ['recent', 'mid', 'early']),
        'phase_b': ([0.75, 0.25], ['recent', 'replay']),
        'phase_c': ([0.90, 0.10], ['recent', 'replay']),
    },
    'strong': {
        'phase_a': ([0.60, 0.25, 0.15], ['recent', 'mid', 'early']),
        'phase_b': ([0.90, 0.10], ['recent', 'replay']),
        'phase_c': ([0.98, 0.02], ['recent', 'replay']),
    },
    'two_stage': {
        'phase_a': ([0.50, 0.30, 0.20], ['recent', 'mid', 'early']),
        'phase_b': ([0.85, 0.15], ['recent', 'replay']),
        'phase_c': ([0.95, 0.05], ['recent', 'replay']),
    },
}


WINDOW_PROFILES = {
    '24m_12m': {'phase_b': 'recent_24', 'phase_c': 'recent_12'},
    '12m_6m': {'phase_b': 'recent_12', 'phase_c': 'recent_6'},
    '6m_6m': {'phase_b': 'recent_6', 'phase_c': 'recent_6'},
}


SCHEDULER_PROFILES = {
    'plateau': {'phase_a': 'plateau', 'phase_b': 'plateau', 'phase_c': 'plateau'},
    'cosine': {'phase_a': 'cosine', 'phase_b': 'cosine', 'phase_c': 'cosine'},
    'phasewise': {'phase_a': 'cosine', 'phase_b': 'cosine', 'phase_c': 'plateau'},
}


CURRICULUM_PROFILES = {
    'broad_to_recent': ['phase_a', 'phase_b', 'phase_c'],
    'recent_broad_recent': ['phase_b', 'phase_a', 'phase_c'],
}


SCHEDULER_PREFIXES = (
    ('P', 'plateau'),
    ('C', 'cosine'),
    ('W', 'phasewise'),
)

CURRICULUM_PREFIXES = (
    ('A', 'broad_to_recent'),
    ('B', 'recent_broad_recent'),
)

WEIGHT_PREFIXES = (
    ('1', 'mild'),
    ('2', 'strong'),
    ('3', 'two_stage'),
)

WINDOW_PREFIXES = (
    ('x', '24m_12m'),
    ('y', '12m_6m'),
    ('z', '6m_6m'),
)


PHASE_SEED_OFFSETS = {
    'phase_a': 101,
    'phase_b': 202,
    'phase_c': 303,
}


BUCKET_SEED_OFFSETS = {
    'recent': 11,
    'mid': 23,
    'early': 37,
    'replay': 53,
}


TRANSIENT_TRAINING_FAILURE_MARKERS = (
    'error code: <1455>',
    "Couldn't open shared file mapping",
    "Couldn't open shared event",
    'WinError 1455',
    'paging file is too small',
    'CUDA error: out of memory',
    'torch.OutOfMemoryError',
    'Pin memory thread exited unexpectedly',
    'CUBLAS_STATUS_INTERNAL_ERROR',
    'cublasLtMatmul',
    'CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED',
    'CUDNN_STATUS_EXECUTION_FAILED',
)


def month_key(file_path: str) -> str:
    parts = Path(file_path).parts
    if len(parts) < 3:
        raise ValueError(f'cannot parse month from path: {file_path}')
    month = parts[-2]
    if not re.fullmatch(r'\d{6}', month):
        raise ValueError(f'cannot parse month from path: {file_path}')
    return month


def yyyymm_int(value: str) -> int:
    return int(value)


def load_all_files() -> list[str]:
    index = torch.load(BASE_INDEX_PATH, weights_only=True)
    files = list(dict.fromkeys(index['train_files'] + index['val_files']))
    return files


def group_files_by_month(files: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for file_path in files:
        grouped.setdefault(month_key(file_path), []).append(file_path)
    for bucket in grouped.values():
        bucket.sort()
    return dict(sorted(grouped.items()))


def select_range(grouped: dict[str, list[str]], start: str, end: str) -> list[str]:
    start_i = yyyymm_int(start)
    end_i = yyyymm_int(end)
    out: list[str] = []
    for month, files in grouped.items():
        month_i = yyyymm_int(month)
        if start_i <= month_i <= end_i:
            out.extend(files)
    return out


def select_range_excluding(
    grouped: dict[str, list[str]],
    start: str,
    end: str,
    *,
    exclude_ranges: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    start_i = yyyymm_int(start)
    end_i = yyyymm_int(end)
    normalized_excludes = tuple(
        (yyyymm_int(exclude_start), yyyymm_int(exclude_end))
        for exclude_start, exclude_end in exclude_ranges
    )
    out: list[str] = []
    for month, files in grouped.items():
        month_i = yyyymm_int(month)
        if month_i < start_i or month_i > end_i:
            continue
        if any(exclude_start_i <= month_i <= exclude_end_i for exclude_start_i, exclude_end_i in normalized_excludes):
            continue
        out.extend(files)
    return out


def sample_files(files: list[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or len(files) <= limit:
        return list(files)
    rng = random.Random(seed)
    return sorted(rng.sample(files, limit))


def ordered_files(files: list[str], seed: int) -> list[str]:
    ordered = list(files)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def expand_weighted_pool(
    bucket_files: dict[str, list[str]],
    weights: list[float],
    bucket_names: list[str],
    target_size: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    allocated = 0
    for idx, (weight, name) in enumerate(zip(weights, bucket_names, strict=True)):
        files = bucket_files[name]
        if not files:
            continue
        if idx == len(weights) - 1:
            target_count = max(target_size - allocated, 0)
        else:
            target_count = int(round(target_size * weight))
        allocated += target_count
        whole, rem = divmod(target_count, len(files))
        if whole > 0:
            out.extend(files * whole)
        if rem > 0:
            out.extend(files[:rem])
    rng.shuffle(out)
    return out


def phase_seed(base_seed: int, phase_name: str) -> int:
    return base_seed + PHASE_SEED_OFFSETS[phase_name]


def merge_dict(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def checkpoint_paths(
    exp_dir: Path,
    *,
    storage_root: Path | None = None,
) -> dict[str, Path]:
    artifact_root = storage_root or exp_dir
    ckpt_dir = artifact_root / 'checkpoints'
    tb_dir = artifact_root / 'tb'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    best_loss_state_file = ckpt_dir / 'best_loss.pth'
    best_acc_state_file = ckpt_dir / 'best_action_score.pth'
    best_rank_state_file = ckpt_dir / 'best_rank.pth'
    return {
        'state_file': ckpt_dir / 'latest.pth',
        'best_state_file': best_loss_state_file,
        'best_loss_state_file': best_loss_state_file,
        'best_acc_state_file': best_acc_state_file,
        'best_rank_state_file': best_rank_state_file,
        'tensorboard_dir': tb_dir,
        'file_index': artifact_root / 'file_index.pth',
        'manifest_file': artifact_root / 'phase_manifest.json',
    }


def phase_storage_root_override(phase_name: str) -> Path | None:
    env_name = f'MORTAL_SL_AB_{phase_name.upper()}_STORAGE_ROOT'
    value = os.environ.get(env_name, '').strip()
    return Path(value).resolve() if value else None


def load_state_summary(state_path: Path) -> dict:
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    optimizer = state.get('optimizer', {})
    param_groups = optimizer.get('param_groups', [])
    lr = param_groups[0]['lr'] if param_groups else None
    last_full_recent_metrics = dict(state.get('last_full_recent_metrics') or {})
    if last_full_recent_metrics:
        refresh_scenario_quality_score(last_full_recent_metrics)
        refresh_selection_quality_score(last_full_recent_metrics)
    return {
        'path': str(state_path),
        'checkpoint_id': state.get('checkpoint_id'),
        'run_provenance': state.get('run_provenance'),
        'steps': state.get('steps'),
        'optimizer_steps': state.get('optimizer_steps'),
        'epoch': state.get('epoch'),
        'best_monitor_loss': state.get('best_val_loss'),
        'best_monitor_action_acc': state.get('best_val_action_acc'),
        'best_monitor_action_score': state.get('best_val_action_score', state.get('best_val_action_acc')),
        'best_monitor_rank_acc': state.get('best_val_rank_acc'),
        'best_full_recent_loss': state.get('best_full_recent_loss', state.get('best_val_loss')),
        'best_full_recent_macro_action_acc': state.get('best_full_recent_action_acc', state.get('best_val_action_acc')),
        'best_full_recent_action_score': state.get('best_full_recent_action_score', state.get('best_full_recent_action_acc', state.get('best_val_action_acc'))),
        'best_full_recent_rank_acc': state.get('best_full_recent_rank_acc', state.get('best_val_rank_acc')),
        'last_monitor_recent_metrics': state.get('last_monitor_recent_metrics'),
        'last_full_recent_metrics': last_full_recent_metrics or None,
        'last_old_regression_metrics': state.get('last_old_regression_metrics'),
        'validation_checks': state.get('validation_checks'),
        'full_validation_checks': state.get('full_validation_checks'),
        'old_regression_checks': state.get('old_regression_checks'),
        'lr': lr,
    }


def load_state_summary_with_fallback(state_path: Path, *fallback_paths: Path) -> dict:
    for candidate_path in (state_path, *fallback_paths):
        if candidate_path.exists():
            return load_state_summary(candidate_path)
    raise FileNotFoundError(state_path)


def checkpoint_is_complete_for_config(state: dict, supervised_cfg: dict) -> bool:
    steps = int(state.get('steps') or 0)
    max_steps = int(supervised_cfg.get('max_steps') or 0)
    if max_steps > 0 and steps >= max_steps:
        return True

    patience_checks = int(
        supervised_cfg.get(
            'early_stopping_patience_checks',
            supervised_cfg.get('early_stopping_patience', 0),
        ) or 0
    )
    if patience_checks <= 0:
        return False

    validation_checks = int(state.get('validation_checks') or 0)
    min_validation_checks = int(supervised_cfg.get('min_validation_checks') or 0)
    patience_counter = int(state.get('patience_counter') or 0)
    num_lr_reductions = int(state.get('num_lr_reductions') or 0)
    min_lr_reductions = int(supervised_cfg.get('early_stopping_min_lr_reductions') or 0)
    return (
        validation_checks >= min_validation_checks
        and patience_counter >= patience_checks
        and num_lr_reductions >= min_lr_reductions
    )
def stable_digest(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def semantic_config_digest(cfg: dict) -> str:
    semantic_cfg = deepcopy(cfg)
    supervised_cfg = semantic_cfg.get('supervised')
    if isinstance(supervised_cfg, dict):
        for key in PROVENANCE_IGNORED_SUPERVISED_KEYS:
            supervised_cfg.pop(key, None)
    return stable_digest(semantic_cfg)


def checkpoint_identity(state_path: str | Path) -> dict:
    checkpoint_path = Path(state_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'missing phase handoff checkpoint: {checkpoint_path}')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_id = state.get('checkpoint_id')
    provenance = state.get('run_provenance')
    if (
        not checkpoint_id
        or not isinstance(provenance, dict)
        or not provenance.get('plan_id')
    ):
        raise RuntimeError(
            f'checkpoint {checkpoint_path} has no immutable provenance. '
            'Legacy checkpoints cannot be reused as an automatic phase handoff; '
            'start a new run or perform an explicit one-time migration.'
        )
    return {
        'path': str(checkpoint_path),
        'checkpoint_id': str(checkpoint_id),
        'plan_id': provenance.get('plan_id'),
        'steps': int(state.get('steps') or 0),
        'timestamp': state.get('timestamp'),
    }


def build_phase_plan(
    *,
    ab_name: str,
    arm_name: str,
    phase_name: str,
    scheduler_type: str,
    weight_profile: str,
    window_profile: str,
    seed: int,
    step_scale: float,
    max_steps: int,
    train_files: list[str],
    eval_splits: dict[str, list[str]],
    cfg: dict,
    init_state_file: str | None,
) -> dict:
    parent = checkpoint_identity(init_state_file) if init_state_file else None
    plan = {
        'schema_version': PHASE_PLAN_SCHEMA_VERSION,
        'ab_name': ab_name,
        'arm_name': arm_name,
        'phase_name': phase_name,
        'scheduler_type': scheduler_type,
        'weight_profile': weight_profile,
        'window_profile': window_profile,
        'training_seed': int(seed),
        'file_order_seed': int(phase_seed(seed, phase_name)),
        'step_scale': float(step_scale),
        'max_steps': int(max_steps),
        'train_files_digest': stable_digest(list(train_files)),
        'monitor_recent_files_digest': stable_digest(
            list(eval_splits['monitor_recent_files'])
        ),
        'full_recent_files_digest': stable_digest(
            list(eval_splits['full_recent_files'])
        ),
        'old_regression_files_digest': stable_digest(
            list(eval_splits['old_regression_files'])
        ),
        'semantic_config_digest': semantic_config_digest(cfg),
        'parent_checkpoint_id': parent['checkpoint_id'] if parent else '',
        'parent_plan_id': parent['plan_id'] if parent else '',
    }
    plan['plan_id'] = stable_digest(plan)
    return plan


def load_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
            newline='\n',
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def record_phase_handoff(manifest_path: Path, handoff: dict) -> None:
    manifest = load_json_file(manifest_path)
    manifest['handoff'] = handoff
    manifest['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    atomic_write_json(manifest_path, manifest)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_existing_phase_artifacts(
    ckpts: dict[str, Path],
    expected_plan: dict,
) -> dict | None:
    manifest_path = ckpts['manifest_file']
    latest_path = ckpts['state_file']
    manifest = load_json_file(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        actual_plan_id = (manifest.get('plan') or {}).get('plan_id')
        if actual_plan_id != expected_plan['plan_id']:
            raise RuntimeError(
                f'phase artifact plan mismatch under {manifest_path.parent}: '
                f'expected plan_id={expected_plan["plan_id"]}, '
                f'actual plan_id={actual_plan_id}. '
                'Use a new run name or an empty storage root; existing artifacts '
                'will not be overwritten.'
            )
    if not latest_path.exists():
        orphaned = [
            path
            for key, path in ckpts.items()
            if key.endswith('_state_file') and path.exists()
        ]
        if orphaned:
            raise RuntimeError(
                f'phase artifacts under {latest_path.parent.parent} contain named '
                f'checkpoints but no latest checkpoint: {orphaned}'
            )
        return None
    if manifest is None:
        raise RuntimeError(
            f'checkpoint {latest_path} predates phase provenance manifests. '
            'Refusing unsafe automatic reuse.'
        )
    state = torch.load(latest_path, map_location='cpu', weights_only=False)
    actual_plan = state.get('run_provenance')
    if actual_plan != expected_plan:
        actual_plan_id = (
            actual_plan.get('plan_id')
            if isinstance(actual_plan, dict)
            else None
        )
        raise RuntimeError(
            f'checkpoint provenance mismatch at {latest_path}: '
            f'expected plan_id={expected_plan["plan_id"]}, '
            f'actual plan_id={actual_plan_id}.'
        )
    return state


def score_summary(summary: dict) -> tuple[float, float, float, float]:
    full_metrics = summary.get('last_full_recent_metrics') or {}
    old_metrics = summary.get('last_old_regression_metrics') or {}
    full_loss = full_metrics.get('loss', summary.get('best_full_recent_loss', math.inf))
    action_score = full_metrics.get('action_quality_score', summary.get('best_full_recent_action_score', action_quality_score(full_metrics or summary)))
    rank_acc = full_metrics.get('rank_acc', summary.get('best_full_recent_rank_acc', 0.0))
    old_loss = old_metrics.get('loss', 0.0)
    return (full_loss, -action_score, -rank_acc, old_loss)


def full_recent_loss(summary: dict) -> float:
    full_metrics = summary.get('last_full_recent_metrics') or {}
    return full_metrics.get('loss', summary.get('best_full_recent_loss', math.inf))


def action_priority(summary: dict) -> tuple:
    full_metrics = summary.get('last_full_recent_metrics') or {}
    old_metrics = summary.get('last_old_regression_metrics') or {}
    return selection_tiebreak_key(
        full_metrics or summary,
        recent_loss=full_recent_loss(summary),
        old_regression_loss=old_metrics.get('loss', math.inf),
    )


def select_winner_by_policy(results: dict[str, dict]) -> tuple[str, dict]:
    best_loss = min(full_recent_loss(result['final']['best_loss']) for result in results.values())
    eligible = {
        name: result
        for name, result in results.items()
        if full_recent_loss(result['final']['best_loss']) <= best_loss + LOSS_EPSILON
    }
    winner = max(
        eligible.items(),
        key=lambda item: action_priority(item[1]['final']['best_loss']),
    )[0]
    return winner, {
        'loss_epsilon': LOSS_EPSILON,
        'best_loss': best_loss,
        'eligible': sorted(eligible),
        'eligible_action_scores': {
            name: action_quality_score((result['final']['best_loss'].get('last_full_recent_metrics') or result['final']['best_loss']))
            for name, result in eligible.items()
        },
        'eligible_scenario_scores': {
            name: scenario_quality_score((result['final']['best_loss'].get('last_full_recent_metrics') or result['final']['best_loss']))
            for name, result in eligible.items()
        },
    }


def select_checkpoint_candidate(candidates: dict[str, dict]) -> tuple[str, dict]:
    best_loss = min(full_recent_loss(candidate) for candidate in candidates.values())
    eligible = {
        name: candidate
        for name, candidate in candidates.items()
        if full_recent_loss(candidate) <= best_loss + LOSS_EPSILON
    }
    winner = max(
        eligible.items(),
        key=lambda item: action_priority(item[1]),
    )[0]
    return winner, {
        'loss_epsilon': LOSS_EPSILON,
        'best_loss': best_loss,
        'eligible': sorted(eligible),
        'eligible_action_scores': {
            name: action_quality_score(candidate.get('last_full_recent_metrics') or candidate)
            for name, candidate in eligible.items()
        },
        'eligible_scenario_scores': {
            name: scenario_quality_score(candidate.get('last_full_recent_metrics') or candidate)
            for name, candidate in eligible.items()
        },
    }


def build_base_config() -> dict:
    return load_toml_file(BASE_CFG_PATH)


def write_toml(path: Path, data: dict) -> None:
    temp_path = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        write_toml_file(temp_path, data)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_index(path: Path, *, train_files: list[str], monitor_recent_files: list[str], full_recent_files: list[str], old_regression_files: list[str], meta: dict) -> None:
    payload = {
        'train_files': train_files,
        'monitor_recent_files': monitor_recent_files,
        'full_recent_files': full_recent_files,
        'old_regression_files': old_regression_files,
        'meta': meta,
    }
    atomic_torch_save(payload, path)


def read_log_tail(path: Path, max_bytes: int = 131072, *, start_offset: int = 0) -> str:
    if not path.exists():
        return ''
    with path.open('rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        seek_pos = max(min(max(start_offset, 0), size), max(0, size - max_bytes))
        f.seek(seek_pos)
        return f.read().decode('utf-8', errors='ignore')


def transient_training_failure_marker(log_path: Path, *, start_offset: int = 0) -> str | None:
    tail = read_log_tail(log_path, start_offset=start_offset)
    tail_lower = tail.lower()
    for marker in TRANSIENT_TRAINING_FAILURE_MARKERS:
        if marker.lower() in tail_lower:
            return marker
    return None


def run_training(cfg_path: Path, log_path: Path) -> None:
    env = os.environ.copy()
    env['MORTAL_CFG'] = str(cfg_path)
    env.setdefault('MORTAL_TQDM_DISABLE', '1')
    affinity = os.environ.get(AFFINITY_ENV_VAR)
    if affinity is not None:
        env[AFFINITY_ENV_VAR] = affinity
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        attempt_log_start = log_path.stat().st_size if log_path.exists() else 0
        mode = 'a' if log_path.exists() else 'w'
        with log_path.open(mode, encoding='utf-8', newline='\n') as f:
            f.write(f'\n=== train_supervised attempt {attempt} @ {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
            f.flush()
            proc = subprocess.run(
                [sys.executable, '-m', 'mortal.supervised.train_supervised'],
                cwd=MORTAL_DIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode == 0:
            return
        marker = transient_training_failure_marker(log_path, start_offset=attempt_log_start)
        if marker is None:
            raise RuntimeError(f'train_supervised.py failed, see {log_path}')
        sleep_seconds = min(30, 5 + attempt)
        with log_path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(
                f'=== transient training failure detected: {marker}; '
                f'retrying with same config after {sleep_seconds}s ===\n'
            )
        time.sleep(sleep_seconds)


def build_eval_splits(grouped: dict[str, list[str]], seed: int, limits: dict[str, int]) -> dict[str, list[str]]:
    return {
        'monitor_recent_files': select_range(grouped, *WINDOWS['monitor_recent']),
        'full_recent_files': sample_files(select_range(grouped, *WINDOWS['full_recent']), limits['full_recent'], seed + 17),
        'old_regression_files': sample_files(select_range(grouped, *WINDOWS['old_regression']), limits['old_regression'], seed + 29),
    }


def phase_train_files(
    grouped: dict[str, list[str]],
    phase_name: str,
    *,
    weight_profile: str,
    window_profile: str,
    pool_size: int,
    seed: int,
) -> list[str]:
    weights, bucket_names = WEIGHT_PROFILES[weight_profile][phase_name]
    mid_train_files = select_range_excluding(
        grouped,
        *WINDOWS['mid'],
        exclude_ranges=(WINDOWS['old_regression'],),
    )
    if phase_name == 'phase_a':
        buckets = {
            'recent': select_range(grouped, *WINDOWS['recent_24']),
            'mid': mid_train_files,
            'early': select_range(grouped, *WINDOWS['early']),
        }
    else:
        recent_window = WINDOW_PROFILES[window_profile][phase_name]
        buckets = {
            'recent': select_range(grouped, *WINDOWS[recent_window]),
            'replay': select_range(grouped, *WINDOWS['early']) + mid_train_files,
        }
    ordered_buckets = {
        name: ordered_files(files, seed + BUCKET_SEED_OFFSETS.get(name, 0))
        for name, files in buckets.items()
    }
    target_pool_size = pool_size
    if target_pool_size <= 0:
        target_pool_size = sum(len(ordered_buckets[name]) for name in bucket_names)
    return expand_weighted_pool(ordered_buckets, weights, bucket_names, target_pool_size, seed)


def make_phase_overrides(
    ckpts: dict[str, Path],
    *,
    seed: int,
    phase_name: str,
    max_steps: int,
    scheduler_type: str,
    init_state_file: str | None,
    allow_early_stopping: bool,
) -> dict:
    warm_up_steps = min(
        2000 if phase_name == 'phase_a' else 1000,
        max(1, max_steps // 4),
    )
    scheduler_overrides = {
        'type': scheduler_type,
        'warm_up_steps': warm_up_steps,
        'init': 1e-8,
        'factor': 0.5,
        'patience': 2,
        'threshold': 0.0005,
        'cooldown': 0,
        'min_lr': 1e-6,
        'final': 1e-5,
        'max_steps': max_steps,
    }
    supervised = {
        'state_file': str(ckpts['state_file']),
        'best_state_file': str(ckpts['best_state_file']),
        'best_loss_state_file': str(ckpts['best_loss_state_file']),
        'best_acc_state_file': str(ckpts['best_acc_state_file']),
        'best_rank_state_file': str(ckpts['best_rank_state_file']),
        'tensorboard_dir': str(ckpts['tensorboard_dir']),
        'file_index': str(ckpts['file_index']),
        'batch_size': BASE_SCREENING['batch_size'],
        'save_every': BASE_SCREENING['save_every'],
        'num_workers': BASE_SCREENING['num_workers'],
        'file_batch_size': BASE_SCREENING['file_batch_size'],
        'val_file_batch_size': BASE_SCREENING.get('val_file_batch_size', BASE_SCREENING['file_batch_size']),
        'prefetch_factor': BASE_SCREENING['prefetch_factor'],
        'val_prefetch_factor': BASE_SCREENING.get('val_prefetch_factor', BASE_SCREENING['prefetch_factor']),
        'log_every': BASE_SCREENING['log_every'],
        'max_epochs': BASE_SCREENING['max_epochs'],
        'max_steps': max_steps,
        'min_epochs': 1,
        'val_every_steps': BASE_SCREENING['val_every_steps'],
        'monitor_val_batches': BASE_SCREENING['monitor_val_batches'],
        'full_val_every_checks': BASE_SCREENING['full_val_every_checks'],
        'old_regression_every_checks': BASE_SCREENING['old_regression_every_checks'],
        'force_safe_training': BASE_SCREENING.get('force_safe_training', False),
        'min_validation_checks': 2,
        'early_stopping_patience': 8 if allow_early_stopping else 0,
        'early_stopping_patience_checks': 8 if allow_early_stopping else 0,
        'early_stopping_min_delta': 0.0005 if allow_early_stopping else 0.0,
        'early_stopping_min_lr_reductions': 2 if scheduler_type == 'plateau' else 0,
        'seed': seed,
        'scheduler': scheduler_overrides,
    }
    if init_state_file:
        supervised['init_state_file'] = init_state_file
    return {'supervised': supervised}


def run_phase(
    base_cfg: dict,
    grouped: dict[str, list[str]],
    *,
    ab_name: str,
    arm_name: str,
    phase_name: str,
    scheduler_type: str,
    weight_profile: str,
    window_profile: str,
    seed: int,
    eval_splits: dict[str, list[str]],
    init_state_file: str | None,
    step_scale: float,
    storage_root: Path | None = None,
    allow_early_stopping: bool = True,
) -> dict:
    exp_dir = AB_ROOT / ab_name / arm_name / phase_name
    ckpts = checkpoint_paths(
        exp_dir,
        storage_root=storage_root,
    )
    max_steps = max(1, int(round(BASE_SCREENING['phase_steps'][phase_name] * step_scale)))
    pool_size = BASE_SCREENING['phase_train_pool'][phase_name]
    train_files = phase_train_files(
        grouped,
        phase_name,
        weight_profile=weight_profile,
        window_profile=window_profile,
        pool_size=pool_size,
        seed=phase_seed(seed, phase_name),
    )
    cfg = merge_dict(
        base_cfg,
        make_phase_overrides(
            ckpts,
            seed=seed,
            phase_name=phase_name,
            max_steps=max_steps,
            scheduler_type=scheduler_type,
            init_state_file=init_state_file,
            allow_early_stopping=allow_early_stopping,
        ),
    )
    cfg_path = exp_dir / 'config.toml'
    log_path = exp_dir / 'train.log'
    phase_plan = build_phase_plan(
        ab_name=ab_name,
        arm_name=arm_name,
        phase_name=phase_name,
        scheduler_type=scheduler_type,
        weight_profile=weight_profile,
        window_profile=window_profile,
        seed=seed,
        step_scale=step_scale,
        max_steps=max_steps,
        train_files=train_files,
        eval_splits=eval_splits,
        cfg=cfg,
        init_state_file=init_state_file,
    )
    cfg['supervised']['run_provenance'] = phase_plan
    existing_state = validate_existing_phase_artifacts(ckpts, phase_plan)
    write_index(
        ckpts['file_index'],
        train_files=train_files,
        monitor_recent_files=eval_splits['monitor_recent_files'],
        full_recent_files=eval_splits['full_recent_files'],
        old_regression_files=eval_splits['old_regression_files'],
        meta={
            'ab_name': ab_name,
            'arm_name': arm_name,
            'phase_name': phase_name,
            'scheduler_type': scheduler_type,
            'weight_profile': weight_profile,
            'window_profile': window_profile,
            'train_files': len(train_files),
            'plan_id': phase_plan['plan_id'],
        },
    )
    write_toml(cfg_path, cfg)
    manifest = {
        'schema_version': PHASE_PLAN_SCHEMA_VERSION,
        'plan': phase_plan,
        'status': 'running',
        'config_path': str(cfg_path.resolve()),
        'file_index_path': str(ckpts['file_index'].resolve()),
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    atomic_write_json(ckpts['manifest_file'], manifest)
    reused_completed_checkpoint = (
        existing_state is not None
        and checkpoint_is_complete_for_config(existing_state, cfg['supervised'])
    )
    if reused_completed_checkpoint:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(
                f'\n=== existing completed {phase_name} checkpoint detected; '
                f'skipping training @ {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n'
            )
    else:
        run_training(cfg_path, log_path)
    summaries = {
        'latest': load_state_summary(ckpts['state_file']),
        'best_loss': load_state_summary_with_fallback(ckpts['best_loss_state_file'], ckpts['state_file']),
        'best_acc': load_state_summary_with_fallback(ckpts['best_acc_state_file'], ckpts['state_file']),
        'best_rank': load_state_summary_with_fallback(ckpts['best_rank_state_file'], ckpts['state_file']),
        'artifact_root': str((storage_root or exp_dir).resolve()),
        'artifacts_retained': True,
        'paths': {name: str(path) for name, path in ckpts.items()},
        'log_path': str(log_path),
        'config_path': str(cfg_path),
        'reused_completed_checkpoint': reused_completed_checkpoint,
        'phase_plan': phase_plan,
    }
    latest_state = torch.load(
        ckpts['state_file'],
        map_location='cpu',
        weights_only=False,
    )
    phase_complete = checkpoint_is_complete_for_config(
        latest_state,
        cfg['supervised'],
    )
    manifest.update({
        'status': 'completed' if phase_complete else 'incomplete',
        'reused_completed_checkpoint': reused_completed_checkpoint,
        'latest_checkpoint_id': summaries['latest']['checkpoint_id'],
        'latest_steps': summaries['latest']['steps'],
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    atomic_write_json(ckpts['manifest_file'], manifest)
    if not phase_complete:
        raise RuntimeError(
            f'{phase_name} exited without satisfying its completion policy; '
            f'see {log_path}'
        )
    return summaries


def run_arm(
    base_cfg: dict,
    grouped: dict[str, list[str]],
    *,
    ab_name: str,
    arm_name: str,
    scheduler_profile: str,
    curriculum_profile: str,
    weight_profile: str,
    window_profile: str,
    seed: int,
    eval_splits: dict[str, list[str]],
    step_scale: float,
    allow_early_stopping: bool = True,
) -> dict:
    phase_order = CURRICULUM_PROFILES[curriculum_profile]
    final_phase_name = phase_order[-1]
    init_state_file = None
    phase_results = {}
    for phase_name in phase_order:
        scheduler_type = SCHEDULER_PROFILES[scheduler_profile][phase_name]
        storage_root_override = phase_storage_root_override(phase_name)
        phase_result = run_phase(
            base_cfg,
            grouped,
            ab_name=ab_name,
            arm_name=arm_name,
            phase_name=phase_name,
            scheduler_type=scheduler_type,
            weight_profile=weight_profile,
            window_profile=window_profile,
            seed=seed,
            eval_splits=eval_splits,
            init_state_file=init_state_file,
            step_scale=step_scale,
            storage_root=storage_root_override,
            allow_early_stopping=allow_early_stopping,
        )
        handoff_candidates = {
            'best_loss': phase_result['best_loss'],
            'best_acc': phase_result['best_acc'],
            'best_rank': phase_result['best_rank'],
        }
        handoff_winner, handoff_selection = select_checkpoint_candidate(
            handoff_candidates
        )
        handoff_summary = handoff_candidates[handoff_winner]
        phase_result['handoff'] = {
            'checkpoint_type': handoff_winner,
            'checkpoint_id': handoff_summary['checkpoint_id'],
            'path': handoff_summary['path'],
            'selection': handoff_selection,
        }
        record_phase_handoff(
            Path(phase_result['paths']['manifest_file']),
            phase_result['handoff'],
        )
        phase_results[phase_name] = phase_result
        init_state_file = handoff_summary['path']

    final_best_loss = phase_results[final_phase_name]['best_loss']
    final_best_acc = phase_results[final_phase_name]['best_acc']
    final_best_rank = phase_results[final_phase_name]['best_rank']
    final_latest = phase_results[final_phase_name]['latest']
    return {
        'scheduler_profile': scheduler_profile,
        'curriculum_profile': curriculum_profile,
        'weight_profile': weight_profile,
        'window_profile': window_profile,
        'phase_order': phase_order,
        'phase_results': phase_results,
        'final': {
            'best_loss': final_best_loss,
            'best_acc': final_best_acc,
            'best_rank': final_best_rank,
            'latest': final_latest,
        },
        'score': score_summary(final_best_loss),
    }


def save_results(ab_name: str, results: dict) -> Path:
    out_path = AB_ROOT / ab_name / 'summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out_path


def run_ab1(
    base_cfg: dict,
    grouped: dict[str, list[str]],
    seed: int,
    step_scale: float,
    *,
    ab_name: str = 'sl_ab1_scheduler',
) -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    for arm_name, scheduler_profile in [('A_plateau', 'plateau'), ('B_cosine', 'cosine'), ('C_phasewise', 'phasewise')]:
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name=ab_name,
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile='broad_to_recent',
            weight_profile='two_stage',
            window_profile='24m_12m',
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results(ab_name, payload)
    return payload


def run_ab2(base_cfg: dict, grouped: dict[str, list[str]], seed: int, scheduler_profile: str, step_scale: float) -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    for arm_name, curriculum_profile in [('A_broad_to_recent', 'broad_to_recent'), ('B_recent_broad_recent', 'recent_broad_recent')]:
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name='sl_ab2_curriculum',
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile=curriculum_profile,
            weight_profile='two_stage',
            window_profile='24m_12m',
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results('sl_ab2_curriculum', payload)
    return payload


def run_ab3(base_cfg: dict, grouped: dict[str, list[str]], seed: int, scheduler_profile: str, curriculum_profile: str, step_scale: float) -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    for arm_name, weight_profile in [('A_mild', 'mild'), ('B_strong', 'strong'), ('C_two_stage', 'two_stage')]:
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name='sl_ab3_weights',
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile=curriculum_profile,
            weight_profile=weight_profile,
            window_profile='24m_12m',
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results('sl_ab3_weights', payload)
    return payload


def run_ab4(base_cfg: dict, grouped: dict[str, list[str]], seed: int, scheduler_profile: str, curriculum_profile: str, weight_profile: str, step_scale: float) -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    for arm_name, window_profile in [('A_24m_12m', '24m_12m'), ('B_12m_6m', '12m_6m'), ('C_6m_6m', '6m_6m')]:
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name='sl_ab4_windows',
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile=curriculum_profile,
            weight_profile=weight_profile,
            window_profile=window_profile,
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results('sl_ab4_windows', payload)
    return payload


def run_ab23_joint(base_cfg: dict, grouped: dict[str, list[str]], seed: int, scheduler_profile: str, window_profile: str, step_scale: float) -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    arms = [
        ('A1_broad_mild', 'broad_to_recent', 'mild'),
        ('A2_broad_strong', 'broad_to_recent', 'strong'),
        ('A3_broad_two_stage', 'broad_to_recent', 'two_stage'),
        ('B1_recent_mild', 'recent_broad_recent', 'mild'),
        ('B2_recent_strong', 'recent_broad_recent', 'strong'),
        ('B3_recent_two_stage', 'recent_broad_recent', 'two_stage'),
    ]
    for idx, (arm_name, curriculum_profile, weight_profile) in enumerate(arms):
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name='sl_ab23_joint',
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile=curriculum_profile,
            weight_profile=weight_profile,
            window_profile=window_profile,
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results('sl_ab23_joint', payload)
    return payload


def run_ab234_joint(base_cfg: dict, grouped: dict[str, list[str]], seed: int, scheduler_profile: str, step_scale: float) -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    arms = []
    for curriculum_prefix, curriculum_profile in [('A', 'broad_to_recent'), ('B', 'recent_broad_recent')]:
        for weight_prefix, weight_profile in [('1', 'mild'), ('2', 'strong'), ('3', 'two_stage')]:
            for window_prefix, window_profile in [('x', '24m_12m'), ('y', '12m_6m')]:
                arm_name = f'{curriculum_prefix}{weight_prefix}{window_prefix}_{curriculum_profile}_{weight_profile}_{window_profile}'
                arms.append((arm_name, curriculum_profile, weight_profile, window_profile))

    for idx, (arm_name, curriculum_profile, weight_profile, window_profile) in enumerate(arms):
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name='sl_ab234_joint',
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile=curriculum_profile,
            weight_profile=weight_profile,
            window_profile=window_profile,
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results('sl_ab234_joint', payload)
    return payload


def run_ab1234_joint(base_cfg: dict, grouped: dict[str, list[str]], seed: int, step_scale: float, ab_name: str = 'sl_ab1234_joint') -> dict:
    results = {}
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    arms = []
    for scheduler_prefix, scheduler_profile in SCHEDULER_PREFIXES:
        for curriculum_prefix, curriculum_profile in CURRICULUM_PREFIXES:
            for weight_prefix, weight_profile in WEIGHT_PREFIXES:
                for window_prefix, window_profile in WINDOW_PREFIXES:
                    arm_name = (
                        f'{scheduler_prefix}_{curriculum_prefix}{weight_prefix}{window_prefix}_'
                        f'{scheduler_profile}_{curriculum_profile}_{weight_profile}_{window_profile}'
                    )
                    arms.append((
                        arm_name,
                        scheduler_profile,
                        curriculum_profile,
                        weight_profile,
                        window_profile,
                    ))

    for arm_name, scheduler_profile, curriculum_profile, weight_profile, window_profile in arms:
        results[arm_name] = run_arm(
            base_cfg,
            grouped,
            ab_name=ab_name,
            arm_name=arm_name,
            scheduler_profile=scheduler_profile,
            curriculum_profile=curriculum_profile,
            weight_profile=weight_profile,
            window_profile=window_profile,
            seed=seed,
            eval_splits=eval_splits,
            step_scale=step_scale,
        )
    winner, selection = select_winner_by_policy(results)
    payload = {'winner': winner, 'selection': selection, 'results': results}
    save_results(ab_name, payload)
    return payload


def run_ab5_quality_signal(grouped: dict[str, list[str]]) -> dict:
    sample_files = load_all_files()[:100000]
    room_codes = {}
    for file_path in sample_files:
        match = re.search(r'gm-([0-9a-f]{4})-', Path(file_path).name)
        room_codes[match.group(1) if match else 'UNKNOWN'] = room_codes.get(match.group(1) if match else 'UNKNOWN', 0) + 1
    unique_codes = sorted(room_codes)
    conclusion = {
        'room_codes_sampled': room_codes,
        'unique_room_codes': unique_codes,
        'supported': len(unique_codes) > 1,
        'conclusion': (
            'blocked_no_quality_signal'
            if len(unique_codes) == 1
            else 'room_quality_signal_exists'
        ),
    }
    save_results('sl_ab5_quality_signal', conclusion)
    return conclusion


def run_ab6_checkpoint(base_cfg: dict, grouped: dict[str, list[str]], seed: int, scheduler_profile: str, curriculum_profile: str, weight_profile: str, window_profile: str, step_scale: float, ab_name: str = 'sl_ab6_checkpoint') -> dict:
    eval_splits = build_eval_splits(grouped, seed, BASE_SCREENING['eval_files'])
    result = run_arm(
        base_cfg,
        grouped,
        ab_name=ab_name,
        arm_name='checkpoint_compare',
        scheduler_profile=scheduler_profile,
        curriculum_profile=curriculum_profile,
        weight_profile=weight_profile,
        window_profile=window_profile,
        seed=seed,
        eval_splits=eval_splits,
        step_scale=step_scale,
        allow_early_stopping=False,
    )
    final = result['final']
    candidates = {
        'best_loss': final['best_loss'],
        'best_acc': final['best_acc'],
        'best_rank': final['best_rank'],
        'latest': final['latest'],
    }
    winner, selection = select_checkpoint_candidate(candidates)
    payload = {'winner': winner, 'selection': selection, 'candidates': candidates, 'result': result}
    save_results(ab_name, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab', choices=['ab1', 'ab2', 'ab3', 'ab4', 'ab23', 'ab234', 'ab1234', 'ab5', 'ab6', 'all'], default='all')
    parser.add_argument('--scheduler-profile', default='phasewise')
    parser.add_argument('--curriculum-profile', default='broad_to_recent')
    parser.add_argument('--weight-profile', default='two_stage')
    parser.add_argument('--window-profile', default='24m_12m')
    parser.add_argument('--seed', type=int, default=BASE_SCREENING['seed'])
    parser.add_argument('--step-scale', type=float, default=1.0)
    parser.add_argument('--monitor-val-batches', type=int, default=0)
    parser.add_argument('--full-recent-files', type=int, default=0)
    parser.add_argument('--old-regression-files', type=int, default=0)
    parser.add_argument('--ab-name', default='')
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--file-batch-size', type=int, default=-1)
    parser.add_argument('--val-file-batch-size', type=int, default=-1)
    parser.add_argument('--prefetch-factor', type=int, default=-1)
    parser.add_argument('--val-prefetch-factor', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--force-safe-training', action='store_true')
    args = parser.parse_args()

    if args.monitor_val_batches > 0:
        BASE_SCREENING['monitor_val_batches'] = args.monitor_val_batches
    if args.full_recent_files > 0:
        BASE_SCREENING['eval_files']['full_recent'] = args.full_recent_files
    if args.old_regression_files > 0:
        BASE_SCREENING['eval_files']['old_regression'] = args.old_regression_files
    if args.num_workers >= 0:
        BASE_SCREENING['num_workers'] = args.num_workers
    if args.file_batch_size > 0:
        BASE_SCREENING['file_batch_size'] = args.file_batch_size
    if args.val_file_batch_size > 0:
        BASE_SCREENING['val_file_batch_size'] = args.val_file_batch_size
    if args.prefetch_factor > 0:
        BASE_SCREENING['prefetch_factor'] = args.prefetch_factor
    if args.val_prefetch_factor > 0:
        BASE_SCREENING['val_prefetch_factor'] = args.val_prefetch_factor
    if args.batch_size > 0:
        BASE_SCREENING['batch_size'] = args.batch_size
    if args.force_safe_training:
        BASE_SCREENING['force_safe_training'] = True

    base_cfg = build_base_config()
    grouped = group_files_by_month(load_all_files())

    if args.ab == 'ab1':
        ab_name = args.ab_name or 'sl_ab1_scheduler'
        print(json.dumps(run_ab1(base_cfg, grouped, args.seed, args.step_scale, ab_name=ab_name), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab2':
        print(json.dumps(run_ab2(base_cfg, grouped, args.seed, args.scheduler_profile, args.step_scale), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab3':
        print(json.dumps(run_ab3(base_cfg, grouped, args.seed, args.scheduler_profile, args.curriculum_profile, args.step_scale), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab4':
        print(json.dumps(run_ab4(base_cfg, grouped, args.seed, args.scheduler_profile, args.curriculum_profile, args.weight_profile, args.step_scale), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab23':
        print(json.dumps(run_ab23_joint(base_cfg, grouped, args.seed, args.scheduler_profile, args.window_profile, args.step_scale), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab234':
        print(json.dumps(run_ab234_joint(base_cfg, grouped, args.seed, args.scheduler_profile, args.step_scale), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab1234':
        ab_name = args.ab_name or 'sl_ab1234_joint'
        print(json.dumps(run_ab1234_joint(base_cfg, grouped, args.seed, args.step_scale, ab_name=ab_name), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab5':
        print(json.dumps(run_ab5_quality_signal(grouped), ensure_ascii=False, indent=2))
        return
    if args.ab == 'ab6':
        ab_name = args.ab_name or 'sl_ab6_checkpoint'
        print(json.dumps(run_ab6_checkpoint(base_cfg, grouped, args.seed, args.scheduler_profile, args.curriculum_profile, args.weight_profile, args.window_profile, args.step_scale, ab_name=ab_name), ensure_ascii=False, indent=2))
        return

    ab1 = run_ab1(base_cfg, grouped, args.seed, args.step_scale)
    ab1234 = run_ab1234_joint(base_cfg, grouped, args.seed + 1000, args.step_scale)
    winner_result = ab1234['results'][ab1234['winner']]
    ab5 = run_ab5_quality_signal(grouped)
    ab6 = run_ab6_checkpoint(
        base_cfg,
        grouped,
        args.seed + 2000,
        winner_result['scheduler_profile'],
        winner_result['curriculum_profile'],
        winner_result['weight_profile'],
        winner_result['window_profile'],
        args.step_scale,
    )
    payload = {
        'winner_scheduler': winner_result['scheduler_profile'],
        'winner_curriculum': winner_result['curriculum_profile'],
        'winner_weight': winner_result['weight_profile'],
        'winner_window': winner_result['window_profile'],
        'ab1': ab1,
        'ab1234': ab1234,
        'ab5': ab5,
        'ab6': ab6,
    }
    out_path = save_results('sl_all', payload)
    print(json.dumps({'summary_path': str(out_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

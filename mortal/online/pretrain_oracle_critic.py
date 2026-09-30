import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from glob import glob
from functools import partial, wraps
from os import path

import numpy as np
import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from libriichi.consts import obs_shape
from mortal._repo import MORTAL_ROOT, REPO_ROOT
from mortal.config import config
from mortal.core.checkpoint_utils import (
    BRAIN_IS_ORACLE_KEY,
    FIRST_CONV_KEY,
    load_brain_state_strict,
    load_brain_state_with_input_bridge,
)
from mortal.core.common import parameter_count, tqdm
from mortal.core.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    initial_adaptive_curriculum_state,
    normalize_adaptive_curriculum_state,
    observe_adaptive_curriculum,
    validate_adaptive_observation,
)
from mortal.core.oracle_checkpoint_selection import checkpoint_selection, baseline_checkpoint_roles
from mortal.core.lr_scheduler import build_lr_scheduler, normalize_scheduler_type
from mortal.core.evidence_contract import validation_input_contract, require_finalist_decision, sha256_file, native_module_file
from mortal.core.model import Brain, HLGaussValueHead, OracleDualTowerBrain, ValueHead
from mortal.core.process_resources import configure_windows_high_qos
from mortal.data.dataloader import (
    normalize_value_target_mode,
    resolve_rayon_num_threads,
    worker_init_fn,
)
from mortal.data.current_policy_manifest import load_current_policy_manifest
from mortal.data.oracle_value import (
    ORACLE_STATE_FOLDING_VERSION,
    ORACLE_IMPUTATION_VERSION,
    ORACLE_TARGET_CLOCK_VERSION,
    ORACLE_STREAM_SHARDING_VERSION,
    OracleTerminalValueDataset,
    deterministic_game_id,
    normalize_state_fold_backend,
)
from mortal.supervised.convergence import (
    ConvergenceConfig,
    initial_convergence_state,
    normalize_convergence_state,
    observe_convergence,
)


ORACLE_EVAL_INPUT_MODES = ('true', 'zero', 'shuffled')
ORACLE_CONVERGENCE_METRICS = ('primary_loss', 'loss')
ORACLE_EXTERNAL_PAUSE_ENV = 'MORTAL_ORACLE_PAUSE_FILE'
ORACLE_EXTERNAL_PAUSE_EXIT_CODE = 75


class OracleEvaluationPaused(InterruptedError):
    """Validation stopped after a pre-evaluation checkpoint, or before any update."""


LEGACY_PATHS_FOR_SCRIPT_IMPORTS = {
    MORTAL_ROOT,
    MORTAL_ROOT / 'core',
    MORTAL_ROOT / 'data',
    MORTAL_ROOT / 'supervised',
    MORTAL_ROOT / 'online',
    MORTAL_ROOT / 'eval',
    MORTAL_ROOT / 'research',
    REPO_ROOT / 'scripts',
}


def sanitize_sys_path_for_spawn():
    """Keep Windows DataLoader workers from importing legacy script modules as packages."""
    repo_text = str(REPO_ROOT)
    filtered = [
        item for item in sys.path
        if item and item not in {str(p) for p in LEGACY_PATHS_FOR_SCRIPT_IMPORTS}
    ]
    sys.path[:] = [repo_text, *[item for item in filtered if item != repo_text]]


def oracle_pretrain_cfg():
    cfg = config.get('oracle_critic_pretrain', {})
    return cfg if isinstance(cfg, dict) else {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pretrain Oracle critic on non-GRP rank returns without actor imitation.'
    )
    parser.add_argument('--current-policy-manifest', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--val-every-steps', type=int, default=None)
    parser.add_argument('--dependency-val-every-steps', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--max-train-files', type=int, default=None)
    parser.add_argument('--max-val-files', type=int, default=None)
    parser.add_argument('--max-test-files', type=int, default=None)
    parser.add_argument('--val-batches', type=int, default=None)
    parser.add_argument('--test-batches', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--init-state-file', type=str, default=None)
    parser.add_argument('--return-mode', type=str, default=None)
    parser.add_argument('--discount-gamma', type=float, default=None)
    parser.add_argument('--train-scope', type=str, default=None)
    parser.add_argument('--critic-arch', type=str, default=None)
    parser.add_argument('--teacher-state-file', type=str, default=None)
    parser.add_argument('--teacher-loss-weight', type=float, default=None)
    parser.add_argument('--target-loss-weight', type=float, default=None)
    parser.add_argument('--target-output-weights', type=float, nargs=4, default=None)
    parser.add_argument('--target-output-weights-initial', type=float, nargs=4, default=None)
    parser.add_argument('--target-output-weight-ramp-start-steps', type=int, default=None)
    parser.add_argument('--target-output-weight-ramp-end-steps', type=int, default=None)
    parser.add_argument('--value-loss-mode', type=str, default=None)
    parser.add_argument('--value-head-hidden', type=int, default=None)
    parser.add_argument('--value-num-bins', type=int, default=None)
    parser.add_argument('--value-target-min', type=float, default=None)
    parser.add_argument('--value-target-max', type=float, default=None)
    parser.add_argument('--value-sigma-to-bin-ratio', type=float, default=None)
    parser.add_argument('--value-padding-sigma', type=float, default=None)
    parser.add_argument('--tail-blocks', type=int, default=None)
    parser.add_argument('--encoder-lr-scale', type=float, default=None)
    parser.add_argument('--visible-lr-scale', type=float, default=None)
    parser.add_argument('--oracle-lr-scale', type=float, default=None)
    parser.add_argument('--fusion-lr-scale', type=float, default=None)
    parser.add_argument('--value-lr-scale', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--oracle-fusion-init', type=float, default=None)
    parser.add_argument('--oracle-fusion-mode', type=str, default=None)
    parser.add_argument('--oracle-fusion-hidden', type=int, default=None)
    parser.add_argument('--oracle-tower-init', type=str, default=None)
    parser.add_argument('--oracle-first-conv-init', type=str, default=None)
    parser.add_argument('--oracle-input-init-scale', type=float, default=None)
    parser.add_argument('--oracle-hand-init-scale', type=float, default=None)
    parser.add_argument(
        '--exact-zero-sum',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument('--amp-init-scale', type=float, default=None)
    parser.add_argument('--amp-growth-interval', type=int, default=None)
    parser.add_argument('--scheduler-peak', type=float, default=None)
    parser.add_argument('--scheduler-final', type=float, default=None)
    parser.add_argument('--scheduler-warm-up-steps', '--warm-up-steps', dest='scheduler_warm_up_steps', type=int, default=None)
    parser.add_argument('--scheduler-horizon-steps', type=int, default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-checkpoint', type=str, default=None)
    parser.add_argument('--eval-split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--finalist-decision', default='')
    parser.add_argument(
        '--eval-input-modes',
        nargs='+',
        choices=ORACLE_EVAL_INPUT_MODES,
        default=None,
        help='Oracle inputs to compare during eval-only qualification.',
    )
    parser.add_argument('--fresh', action='store_true')
    return parser.parse_args()


def ensure_parent_dir_for_file(file_path):
    if file_path:
        os.makedirs(path.dirname(path.abspath(file_path)), exist_ok=True)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def resolve_output_path(value):
    text = str(value)
    if path.isabs(text):
        return text
    if text.startswith('./checkpoints/') or text == './checkpoints':
        return str((MORTAL_ROOT / text[2:]).resolve())
    return str((REPO_ROOT / text).resolve())


def resolve_external_pause_file(cfg):
    value = os.environ.get(ORACLE_EXTERNAL_PAUSE_ENV) or cfg.get(
        'external_pause_file',
        '',
    )
    text = str(value or '').strip()
    return resolve_output_path(text) if text else ''


def external_pause_requested(file_path):
    return bool(file_path and path.isfile(file_path))


def external_pause_due(file_path, steps, max_steps):
    return steps < max_steps and external_pause_requested(file_path)


def resolve_cli_path(value):
    if not value:
        return ''
    return resolve_output_path(value)


def artifact_path(cfg, key, default_template, run_name):
    if key in cfg:
        return str(cfg[key])
    return resolve_output_path(default_template.format(run_name=run_name))


def display_path(value):
    try:
        return str(path.relpath(value, REPO_ROOT))
    except ValueError:
        return str(value)


def resolve_init_state_file(cfg):
    init_state_file = str(cfg.get('init_state_file', '') or '').strip()
    if init_state_file:
        return init_state_file
    online_cfg = config.get('online', {})
    if isinstance(online_cfg, dict):
        init_state_file = str(online_cfg.get('init_state_file', '') or '').strip()
        if init_state_file:
            return init_state_file
    supervised_cfg = config.get('supervised', {})
    if not isinstance(supervised_cfg, dict):
        return ''
    return str(
        supervised_cfg.get('best_loss_state_file', '')
        or supervised_cfg.get('best_state_file', '')
        or ''
    ).strip()


def load_file_index(file_index):
    if file_index and path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        file_list = index.get('file_list')
        if isinstance(file_list, list):
            return file_list
    return None


def file_list_fingerprint(file_list):
    digest = hashlib.sha256()
    for filename in file_list:
        digest.update(str(filename).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def capped_split_count(total, *, ratio, minimum, maximum):
    count = max(int(minimum), int(total * float(ratio)))
    if int(maximum) > 0:
        count = min(count, int(maximum))
    return max(count, 0)


def current_policy_data(cfg):
    """Resolve and verify the opt-in fixed rollout ledger once at startup."""
    manifest = cfg.get('current_policy_manifest')
    if not manifest:
        return None
    cached = cfg.get('_current_policy_data')
    if cached is not None:
        return cached
    required = {'target_mode': 'all_players', 'return_mode': 'score_rank_mc',
                'discount_gamma': 1.0, 'exact_zero_sum': True,
                'critic_arch': 'dual_tower', 'train_scope': 'all'}
    for name, value in required.items():
        if cfg.get(name) != value:
            raise ValueError(f'current-policy calibration requires {name}={value!r}')
    if config['env']['pts'] != [2, 1, 0, -3] or config['control']['version'] != 4:
        raise ValueError('current-policy calibration requires v4 and pts=[2,1,0,-3]')
    if cfg.get('player_names', ['trainee']) != ['trainee']:
        raise ValueError('current-policy calibration requires player_names=["trainee"]')
    if normalize_eval_input_modes(cfg.get('eval_input_modes', ['true'])) != ('true',):
        raise ValueError('current-policy calibration uses only the existing true/imputed Oracle input')
    cfg.setdefault('eval_input_modes', ['true'])
    for name in ('max_train_files', 'max_val_files', 'max_test_files', 'val_batches', 'test_batches'):
        if int(cfg.get(name, 0) or 0) != 0:
            raise ValueError(f'current-policy calibration disallows {name} truncation')
    # Explicitly override the legacy bounded-validation default, retaining all groups.
    cfg.setdefault('val_batches', 0)
    cfg.setdefault('test_batches', 0)
    for name in ('train_file_index', 'dev_file_index', 'test_file_index', 'file_index', 'globs'):
        if cfg.get(name):
            raise ValueError('current_policy_manifest is the only file/split authority; remove ' + name)
    if int(cfg.get('val_state_fold_count', 1) or 1) != 1:
        raise ValueError('current-policy dev requires all trainee states')
    if int(cfg.get('val_game_id_modulus', 1) or 1) != 1:
        raise ValueError('select complete dev groups in the manifest, not evaluator subsets')
    data = load_current_policy_manifest(manifest)
    import libriichi
    source_files = (
        'mortal/online/pretrain_oracle_critic.py', 'mortal/data/oracle_value.py',
        'mortal/data/dataloader.py', 'mortal/data/current_policy_manifest.py',
        'mortal/core/model.py', 'mortal/core/checkpoint_utils.py', 'mortal/config.py',
        'mortal/core/evidence_contract.py', 'mortal/core/adaptive_curriculum.py',
        'mortal/core/oracle_checkpoint_selection.py',
    )
    data['contract']['training_runtime'] = {
        'native_sha256': sha256_file(native_module_file(libriichi)),
        'source_sha256': {name: sha256_file(REPO_ROOT / name) for name in source_files},
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
    }
    cfg['_current_policy_data'] = data
    return data


def build_file_splits(cfg):
    policy_data = current_policy_data(cfg)
    if policy_data is not None:
        return tuple(list(policy_data['splits'][name]) for name in ('train', 'dev', 'test'))
    explicit_indexes = {
        split: str(cfg.get(f'{split}_file_index', '') or '').strip()
        for split in ('train', 'dev', 'test')
    }
    if any(explicit_indexes.values()):
        if not explicit_indexes['train'] or not explicit_indexes['dev']:
            raise ValueError(
                'explicit Oracle critic splits require train_file_index and '
                'dev_file_index; test_file_index may be empty'
            )
        splits = {}
        for split, file_index in explicit_indexes.items():
            if not file_index:
                splits[split] = []
                continue
            file_list = load_file_index(file_index)
            if file_list is None:
                raise FileNotFoundError(
                    f'oracle_critic_pretrain.{split}_file_index is missing or invalid: '
                    f'{file_index}'
                )
            splits[split] = file_list
        max_train_files = int(cfg.get('max_train_files', 0) or 0)
        max_val_files = int(cfg.get('max_val_files', 0) or 0)
        max_test_files = int(cfg.get('max_test_files', 0) or 0)
        if max_train_files > 0:
            splits['train'] = splits['train'][:max_train_files]
        if max_val_files > 0:
            splits['dev'] = splits['dev'][:max_val_files]
        if max_test_files > 0:
            splits['test'] = splits['test'][:max_test_files]
        if not splits['train'] or not splits['dev']:
            raise ValueError('explicit Oracle critic train/dev splits must be non-empty')
        return splits['train'], splits['dev'], splits['test']

    file_index = str(cfg.get('file_index', config['dataset'].get('file_index', '')) or '')
    file_list = load_file_index(file_index)
    if file_list is None:
        file_list = []
        globs = cfg.get('globs', config['dataset']['globs'])
        for pattern in globs:
            file_list.extend(glob(pattern, recursive=True))
        file_list.sort(reverse=True)
        if file_index:
            ensure_parent_dir_for_file(file_index)
            torch.save({'file_list': file_list}, file_index)
    if not file_list:
        raise FileNotFoundError('oracle_critic_pretrain found no training files')

    seed = int(cfg.get('split_seed', cfg.get('seed', 20260416)) or 0)
    rng = random.Random(seed)
    shuffled = list(file_list)
    rng.shuffle(shuffled)

    val_count = capped_split_count(
        len(shuffled),
        ratio=float(cfg.get('val_ratio', 0.02) or 0.0),
        minimum=int(cfg.get('min_val_files', 64) or 0),
        maximum=int(cfg.get('max_val_files', 0) or 0),
    )
    test_count = capped_split_count(
        len(shuffled),
        ratio=float(cfg.get('test_ratio', 0.0) or 0.0),
        minimum=int(cfg.get('min_test_files', 0) or 0),
        maximum=int(cfg.get('max_test_files', 0) or 0),
    )
    if val_count <= 0:
        raise ValueError('oracle critic pretrain requires a non-empty dev split')
    max_holdout = max(len(shuffled) - 1, 1)
    if val_count + test_count > max_holdout:
        overflow = val_count + test_count - max_holdout
        test_count = max(test_count - overflow, 0)
    if val_count + test_count > max_holdout:
        val_count = max(max_holdout - test_count, 1)

    val_files = shuffled[:val_count]
    test_files = shuffled[val_count:val_count + test_count]
    train_files = shuffled[val_count + test_count:] or shuffled

    max_train_files = int(cfg.get('max_train_files', 0) or 0)
    if max_train_files > 0:
        train_files = train_files[:max_train_files]
    return train_files, val_files, test_files


def build_file_lists(cfg):
    """Backward-compatible two-way split for small external probes."""
    train_files, val_files, _test_files = build_file_splits(cfg)
    return train_files, val_files


def summarize_file_splits(cfg, train_files, val_files, test_files):
    return {
        'seed': int(cfg.get('split_seed', cfg.get('seed', 20260416)) or 0),
        'train': {
            'files': len(train_files),
            'sha256': file_list_fingerprint(train_files),
        },
        'dev': {
            'files': len(val_files),
            'sha256': file_list_fingerprint(val_files),
        },
        'test': {
            'files': len(test_files),
            'sha256': file_list_fingerprint(test_files),
        },
    }


def data_stream_signature(cfg):
    signature = {
        'worker_sharding': ORACLE_STREAM_SHARDING_VERSION,
        'batch_size': int(cfg.get('batch_size', config['control'].get('batch_size', 512)) or 512),
        'num_workers': int(cfg.get('num_workers', config['dataset'].get('num_workers', 0)) or 0),
        'file_batch_size': int(cfg.get('file_batch_size', config['dataset']['file_batch_size']) or 1),
        'reserve_ratio': float(
            cfg.get('reserve_ratio', config['dataset'].get('reserve_ratio', 0.0)) or 0.0
        ),
        'num_epochs': int(cfg.get('num_epochs', config['dataset'].get('num_epochs', 1)) or 1),
        'enable_augmentation': bool(
            cfg.get('enable_augmentation', config['dataset'].get('enable_augmentation', False))
        ),
        'augmented_first': bool(
            cfg.get('augmented_first', config['dataset'].get('augmented_first', False))
        ),
        'shuffle_seed': int(cfg.get('data_shuffle_seed', cfg.get('seed', 20260416)) or 0),
        'state_fold_count': int(cfg.get('state_fold_count', 1) or 1),
        'state_fold_seed': int(cfg.get('state_fold_seed', cfg.get('seed', 20260416)) or 0),
        'state_fold_backend': normalize_state_fold_backend(
            cfg.get('state_fold_backend', 'python_permutation')
        ),
        'native_state_folding_version': ORACLE_STATE_FOLDING_VERSION,
    }
    policy_data = current_policy_data(cfg)
    if policy_data is not None:
        signature['current_policy'] = copy.deepcopy(policy_data['contract'])
    return signature


def initial_data_progress(signature):
    return {
        'signature': copy.deepcopy(signature),
        'cycle': 0,
        'resume_cursors': {},
        'samples_consumed': 0,
        'batches_consumed': 0,
    }


def advance_data_progress_cycle(data_progress):
    data_progress['cycle'] = int(data_progress.get('cycle', 0)) + 1
    data_progress['resume_cursors'] = {}
    return data_progress


def update_data_progress(data_progress, progress):
    if progress is None:
        return
    progress = torch.as_tensor(progress).detach().cpu().reshape(-1, 3)
    data_progress['samples_consumed'] = int(data_progress.get('samples_consumed', 0)) + progress.shape[0]
    data_progress['batches_consumed'] = int(data_progress.get('batches_consumed', 0)) + 1
    cursors = data_progress.setdefault('resume_cursors', {})
    for worker_id in progress[:, 0].unique(sorted=True).tolist():
        worker_rows = progress[progress[:, 0] == worker_id]
        tokens = sorted((int(row[1]), int(row[2])) for row in worker_rows.tolist())
        safe_cursor = tokens[0]
        previous = tuple(cursors.get(int(worker_id), (-1, 0)))
        if safe_cursor >= previous:
            cursors[int(worker_id)] = [safe_cursor[0], safe_cursor[1]]


def validate_resume_data_stream(state, signature):
    saved = state.get('data_progress')
    if saved is None:
        return None
    if saved.get('signature') != signature:
        raise ValueError(
            'oracle critic resume data stream mismatch; batch, worker, file-buffer, '
            'augmentation, and shuffle settings must stay fixed'
        )
    return copy.deepcopy(saved)


def make_dataset(file_list, cfg, *, train, stream_state=None):
    stream_state = stream_state or {}
    policy_data = current_policy_data(cfg)
    num_workers = int(
        cfg.get(
            'num_workers' if train else 'val_num_workers',
            cfg.get('num_workers', config['dataset'].get('num_workers', 0)) if train else 0,
        )
        or 0
    )
    file_batch_size = int(
        cfg.get(
            'file_batch_size' if train else 'val_file_batch_size',
            cfg.get('file_batch_size', config['dataset']['file_batch_size']),
        )
        or 1
    )
    rayon_num_threads = resolve_rayon_num_threads(
        num_workers,
        file_batch_size,
        int(cfg.get('rayon_num_threads', config['dataset'].get('rayon_num_threads', 0)) or 0),
    )
    return OracleTerminalValueDataset(
        version=config['control']['version'],
        file_list=file_list,
        pts=config['env']['pts'],
        file_batch_size=file_batch_size,
        reserve_ratio=(
            float(cfg.get('reserve_ratio', config['dataset'].get('reserve_ratio', 0.0)) or 0.0)
            if train else 0.0
        ),
        player_names=['trainee'] if policy_data is not None else None,
        excludes=None,
        num_epochs=int(cfg.get('num_epochs', config['dataset'].get('num_epochs', 1)) or 1),
        enable_augmentation=(
            bool(cfg.get('enable_augmentation', config['dataset'].get('enable_augmentation', False)))
            if train else False
        ),
        augmented_first=bool(cfg.get('augmented_first', config['dataset'].get('augmented_first', False))),
        shuffle_files=bool(train),
        value_target_mode=str(cfg.get('target_mode', 'all_players')),
        return_mode=str(cfg.get('return_mode', 'score_rank_mc')),
        discount_gamma=float(cfg.get('discount_gamma', config.get('policy', {}).get('gae_gamma', 1.0)) or 1.0),
        worker_torch_num_threads=int(
            cfg.get('worker_torch_num_threads', config['dataset'].get('worker_torch_num_threads', 1))
            or 1
        ),
        worker_torch_num_interop_threads=int(
            cfg.get(
                'worker_torch_num_interop_threads',
                config['dataset'].get('worker_torch_num_interop_threads', 1),
            )
            or 1
        ),
        rayon_num_threads=rayon_num_threads,
        shuffle_seed=(int(cfg.get('data_shuffle_seed', cfg.get('seed', 20260416))) if train else None),
        stream_cycle=int(stream_state.get('cycle', 0)),
        resume_cursors=stream_state.get('resume_cursors', {}),
        emit_progress=bool(train),
        emit_game_id=not train,
        game_id_by_source=None if policy_data is None else policy_data['group_ids'],
        state_fold_count=int(
            cfg.get(
                'state_fold_count' if train else 'val_state_fold_count',
                1,
            )
            or 1
        ),
        state_fold_seed=int(cfg.get('state_fold_seed', cfg.get('seed', 20260416)) or 0),
        state_fold_backend=cfg.get('state_fold_backend', 'python_permutation'),
        oracle_imputation_seed=(
            int(cfg.get('train_oracle_imputation_seed', cfg.get('seed', 20260416))) if train
            else int(cfg.get('val_oracle_imputation_seed', 20260905))
        ),
    )


def oracle_worker_init_fn(worker_id, *, windows_high_qos=False):
    configure_windows_high_qos(windows_high_qos)
    worker_init_fn(worker_id)


def make_loader(dataset, cfg, *, train):
    num_workers = int(
        cfg.get(
            'num_workers' if train else 'val_num_workers',
            cfg.get('num_workers', config['dataset'].get('num_workers', 0)) if train else 0,
        )
        or 0
    )
    batch_size = int(cfg.get('batch_size', config['control'].get('batch_size', 512)) or 512)
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': bool(train),
        'worker_init_fn': (
            partial(oracle_worker_init_fn, windows_high_qos=bool(cfg.get('windows_high_qos', False)))
            if num_workers > 0 else None
        ),
    }
    if num_workers > 0:
        kwargs['prefetch_factor'] = int(
            cfg.get(
                'prefetch_factor' if train else 'val_prefetch_factor',
                config['dataset'].get('prefetch_factor', 2),
            )
            or 2
        )
        kwargs['persistent_workers'] = bool(cfg.get('persistent_workers', True))
    return DataLoader(**kwargs)


def evaluation_files(file_list, cfg, *, game_id_modulus=1, game_id_remainders=(),
                     max_batches=0, input_modes=('true',)):
    """Skip unselected files only for complete, batch-independent validation.

    Validation datasets use the source filename as the game-ID key, including
    cache payloads containing multiple games. Filtering preserves ordered sample
    tensors, but can change floating-point rounding through batch composition.
    Bounded batch evaluation and shuffled-input diagnostics retain their batches.
    """
    if not cfg.get('eval_prefilter_games', False):
        return file_list
    modulus, remainders = normalize_game_id_subset(game_id_modulus, game_id_remainders)
    if max_batches > 0 or 'shuffled' in normalize_eval_input_modes(input_modes):
        return file_list
    if len(remainders) == modulus:
        return file_list
    policy_data = current_policy_data(cfg)
    selected = [filename for filename in file_list
                if (policy_data['group_ids'][path.realpath(filename)] if policy_data is not None
                    else deterministic_game_id(filename)) % modulus in remainders]
    logging.info('evaluation file subset: %s/%s files before feature encoding',
                 len(selected), len(file_list))
    return selected


def evaluation_thread_scope(function):
    """Opt-in CPU thread limit, restored even when validation pauses or fails."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        threads = int(config.get('oracle_critic_pretrain', {}).get('eval_torch_num_threads', 0) or 0)
        previous = torch.get_num_threads()
        if threads < 0:
            raise ValueError('eval_torch_num_threads must be non-negative')
        if threads:
            torch.set_num_threads(threads)
        try:
            return function(*args, **kwargs)
        finally:
            if threads:
                torch.set_num_threads(previous)
    return wrapped


def shutdown_data_loader_iterator(loader, iterator):
    shutdown_workers = getattr(iterator, '_shutdown_workers', None)
    if not callable(shutdown_workers):
        return False
    shutdown_workers()
    if getattr(loader, '_iterator', None) is iterator:
        loader._iterator = None
    return True


def in_training_eval_config(cfg):
    eval_cfg = dict(cfg)
    eval_cfg['val_num_workers'] = int(
        cfg.get('in_training_val_num_workers', 0) or 0
    )
    return eval_cfg


def normalize_train_scope(value):
    scope = str(value or 'oracle_input_value').strip().lower()
    if scope in ('all', 'full', 'all_params'):
        return 'all'
    if scope in (
        'oracle_tail',
        'tail',
        'tail_blocks',
        'oracle_input_tail_value',
        'oracle_input_tail',
    ):
        return 'oracle_tail'
    if scope in (
        'oracle_input_value',
        'oracle_input_and_value',
        'bridge',
        'adapter',
        'head_only',
        'value_head',
    ):
        return 'oracle_input_value'
    raise ValueError(
        f"unsupported oracle critic train_scope={value!r}; "
        "expected 'oracle_input_value', 'oracle_tail', or 'all'"
    )


def normalize_critic_arch(value):
    arch = str(value or 'single_tower').strip().lower()
    if arch in ('single', 'single_tower', 'bridge', 'resnet'):
        return 'single_tower'
    if arch in ('dual', 'dual_tower', 'two_tower', 'visible_oracle_tower'):
        return 'dual_tower'
    raise ValueError(
        f"unsupported oracle critic arch={value!r}; "
        "expected 'single_tower' or 'dual_tower'"
    )


def normalize_oracle_first_conv_init(value):
    mode = str(value or 'legacy_mean_scaled').strip().lower()
    if mode in ('legacy', 'mean', 'mean_scaled', 'legacy_mean_scaled'):
        return 'legacy_mean_scaled'
    if mode in ('random', 'native_random', 'keep_random'):
        return 'random'
    if mode in ('hand', 'hand_aligned', 'opponent_hand_aligned'):
        return 'hand_aligned'
    raise ValueError(
        f'unsupported oracle_first_conv_init={value!r}; expected '
        "'legacy_mean_scaled', 'random', or 'hand_aligned'"
    )


def normalize_oracle_tower_init(value):
    mode = str(value or 'visible_transfer').strip().lower()
    if mode in ('visible', 'visible_transfer', 'transfer', 'copy_visible'):
        return 'visible_transfer'
    if mode in ('random', 'native_random', 'keep_random'):
        return 'random'
    raise ValueError(
        f'unsupported oracle_tower_init={value!r}; expected '
        "'visible_transfer' or 'random'"
    )


def normalize_oracle_fusion_mode(value):
    mode = str(value or 'linear').strip().lower()
    if mode in ('linear', 'late_linear'):
        return 'linear'
    if mode in ('film', 'conditioned', 'feature_wise_linear_modulation'):
        return 'film'
    if mode in ('residual', 'residual_mlp', 'joint_residual_mlp'):
        return 'residual_mlp'
    raise ValueError(
        f'unsupported oracle_fusion_mode={value!r}; expected '
        "'linear', 'film', or 'residual_mlp'"
    )


def normalize_value_loss_mode(value):
    mode = str(value or 'mse').strip().lower().replace('-', '_')
    if mode in ('mse', 'scalar', 'scalar_mse'):
        return 'mse'
    if mode in ('hl_gauss', 'hlgauss', 'histogram', 'histogram_gaussian'):
        return 'hl_gauss'
    raise ValueError(
        f'unsupported value_loss_mode={value!r}; expected '
        "'mse' or 'hl_gauss'"
    )


def normalize_target_output_weights(value, *, num_outputs=4):
    if value is None:
        return None
    weights = tuple(float(weight) for weight in value)
    if len(weights) != int(num_outputs):
        raise ValueError(
            f'target_output_weights must contain {num_outputs} values, got {len(weights)}'
        )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError('target_output_weights must be finite and non-negative')
    total = sum(weights)
    if total <= 0.0:
        raise ValueError('target_output_weights must contain at least one positive value')
    scale = float(num_outputs) / total
    return tuple(weight * scale for weight in weights)


def target_output_weights_at_step(cfg, steps, *, num_outputs=4):
    final_weights = normalize_target_output_weights(
        cfg.get('target_output_weights'),
        num_outputs=num_outputs,
    )
    if 'target_output_weights_initial' not in cfg:
        return final_weights
    initial_weights = normalize_target_output_weights(
        cfg.get('target_output_weights_initial'),
        num_outputs=num_outputs,
    )
    if initial_weights is None or final_weights is None:
        raise ValueError(
            'scheduled target output weights require both '
            'target_output_weights_initial and target_output_weights'
        )
    start = int(cfg.get('target_output_weight_ramp_start_steps', 0) or 0)
    end = int(cfg.get('target_output_weight_ramp_end_steps', start) or 0)
    if start < 0 or end < start:
        raise ValueError(
            'target output weight ramp requires 0 <= start_steps <= end_steps'
        )
    step = int(steps)
    if step < start:
        return initial_weights
    if end == start or step >= end:
        return final_weights
    ratio = (step - start) / float(end - start)
    return tuple(
        initial + ratio * (final - initial)
        for initial, final in zip(initial_weights, final_weights)
    )


def output_weighted_mse(pred, target, normalized_output_weights=None):
    if normalized_output_weights is None:
        return nn.functional.mse_loss(pred, target)
    squared_error = (pred.float() - target.float()).square()
    weights = torch.as_tensor(
        normalized_output_weights,
        dtype=squared_error.dtype,
        device=squared_error.device,
    )
    if weights.ndim != 1 or weights.shape[0] != squared_error.shape[-1]:
        raise ValueError(
            f'output weights shape={tuple(weights.shape)} does not match '
            f'prediction shape={tuple(pred.shape)}'
        )
    return (squared_error * weights).mean()


def float_config_value(cfg, key, default):
    value = cfg.get(key, default)
    return float(default if value is None else value)


def is_dual_tower_brain(brain):
    return isinstance(brain, OracleDualTowerBrain)


def first_conv_module(brain):
    if is_dual_tower_brain(brain):
        raise TypeError('dual-tower Oracle critic has no concatenated first conv')
    module = brain.encoder.net[0]
    if not isinstance(module, nn.Conv1d):
        raise TypeError(f'expected oracle_brain.encoder.net[0] to be Conv1d, got {type(module)!r}')
    return module


def tail_encoder_modules(brain, *, tail_blocks):
    if is_dual_tower_brain(brain):
        raise TypeError('dual-tower Oracle critic does not support tail-block scope')
    net = brain.encoder.net
    blocks = [
        module for module in net
        if module.__class__.__name__ == 'ResBlock'
    ]
    if not blocks:
        raise TypeError('expected oracle_brain.encoder.net to contain ResBlock modules')

    tail_blocks = min(max(int(tail_blocks), 0), len(blocks))
    modules = list(blocks[-tail_blocks:]) if tail_blocks > 0 else []
    first_tail_idx = 1 + len(blocks)
    modules.extend(net[first_tail_idx:])
    return modules


def configure_trainable_parameters(oracle_brain, value_net, *, scope, version, tail_blocks=8):
    scope = normalize_train_scope(scope)
    if is_dual_tower_brain(oracle_brain) and scope != 'all':
        raise ValueError("dual-tower Oracle critic requires train_scope='all'")
    for param in oracle_brain.parameters():
        param.requires_grad_(scope == 'all')
    for param in value_net.parameters():
        param.requires_grad_(True)

    first_conv_mask = None
    normalized_tail_blocks = 0
    if scope in ('oracle_input_value', 'oracle_tail'):
        first_conv = first_conv_module(oracle_brain)
        first_conv.weight.requires_grad_(True)
        visible_channels = obs_shape(version)[0]
        first_conv_mask = torch.zeros_like(first_conv.weight)
        first_conv_mask[:, visible_channels:, :] = 1.0
    if scope == 'oracle_tail':
        normalized_tail_blocks = int(tail_blocks)
        for module in tail_encoder_modules(oracle_brain, tail_blocks=normalized_tail_blocks):
            module.requires_grad_(True)

    trainable = sum(param.numel() for param in oracle_brain.parameters() if param.requires_grad)
    trainable += sum(param.numel() for param in value_net.parameters() if param.requires_grad)
    return {
        'scope': scope,
        'tail_blocks': normalized_tail_blocks,
        'first_conv_mask': first_conv_mask,
        'trainable_params': int(trainable),
    }


def split_decay_params(model, *, prefix='', exclude_params=()):
    excluded = {id(param) for param in exclude_params}
    decay_params = []
    no_decay_params = []
    params_dict = {}
    to_decay = set()
    for mod_name, mod in model.named_modules():
        for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
            if not param.requires_grad or id(param) in excluded:
                continue
            full_name = f'{prefix}{name}'
            params_dict[full_name] = param
            if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                to_decay.add(full_name)
    decay_params.extend(params_dict[name] for name in sorted(to_decay))
    no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    return decay_params, no_decay_params


def optimizer_param_groups(
    oracle_brain,
    value_net,
    *,
    scope,
    encoder_lr_scale=1.0,
    visible_lr_scale=None,
    oracle_lr_scale=None,
    fusion_lr_scale=None,
    value_lr_scale=1.0,
    weight_decay=None,
):
    scope = normalize_train_scope(scope)
    weight_decay = float(
        config['optim'].get('weight_decay', 0.0)
        if weight_decay is None else weight_decay
    )
    value_decay, value_no_decay = split_decay_params(value_net, prefix='value_net.')
    groups = [
        {
            'name': 'value_decay',
            'params': value_decay,
            'weight_decay': weight_decay,
            'lr': float(value_lr_scale),
        },
        {
            'name': 'value_no_decay',
            'params': value_no_decay,
            'lr': float(value_lr_scale),
        },
    ]
    split_dual_groups = is_dual_tower_brain(oracle_brain) and any(
        scale is not None
        for scale in (visible_lr_scale, oracle_lr_scale, fusion_lr_scale)
    )
    if scope == 'all' and split_dual_groups:
        modules = (
            ('visible_encoder', oracle_brain.visible_encoder, visible_lr_scale),
            ('oracle_encoder', oracle_brain.oracle_encoder, oracle_lr_scale),
            ('fusion', oracle_brain.fusion, fusion_lr_scale),
        )
        insert_at = 0
        for name, module, scale in modules:
            resolved_scale = float(encoder_lr_scale if scale is None else scale)
            decay, no_decay = split_decay_params(module, prefix=f'oracle_brain.{name}.')
            if decay:
                groups.insert(insert_at, {
                    'name': f'{name}_decay',
                    'params': decay,
                    'weight_decay': weight_decay,
                    'lr': resolved_scale,
                })
                insert_at += 1
            if no_decay:
                groups.insert(insert_at, {
                    'name': f'{name}_no_decay',
                    'params': no_decay,
                    'lr': resolved_scale,
                })
                insert_at += 1
    elif scope == 'all':
        brain_decay, brain_no_decay = split_decay_params(oracle_brain, prefix='oracle_brain.')
        groups.insert(
            0,
            {
                'name': 'oracle_encoder_decay',
                'params': brain_decay,
                'weight_decay': weight_decay,
                'lr': float(encoder_lr_scale),
            },
        )
        groups.insert(
            1,
            {'name': 'oracle_encoder_no_decay', 'params': brain_no_decay, 'lr': float(encoder_lr_scale)},
        )
    else:
        first_conv_weight = first_conv_module(oracle_brain).weight
        groups.insert(0, {'name': 'oracle_input', 'params': [first_conv_weight]})
        if scope == 'oracle_tail':
            brain_decay, brain_no_decay = split_decay_params(
                oracle_brain,
                prefix='oracle_brain.',
                exclude_params=(first_conv_weight,),
            )
            groups.insert(
                1,
                {
                    'name': 'oracle_encoder_decay',
                    'params': brain_decay,
                    'weight_decay': weight_decay,
                    'lr': float(encoder_lr_scale),
                },
            )
            groups.insert(
                2,
                {'name': 'oracle_encoder_no_decay', 'params': brain_no_decay, 'lr': float(encoder_lr_scale)},
            )
    return [group for group in groups if group['params']]


def normalize_optimizer_type(value):
    value = str(value or 'adamw').strip().lower().replace('-', '_')
    aliases = {
        'adam': 'adamw',
        'schedulefree': 'schedule_free_adamw',
        'adamw_schedule_free': 'schedule_free_adamw',
        'adamw_schedulefree': 'schedule_free_adamw',
    }
    value = aliases.get(value, value)
    if value not in {'adamw', 'schedule_free_adamw'}:
        raise ValueError(f'unsupported Oracle critic optimizer type: {value!r}')
    return value


def resolved_optimizer_config(cfg, scheduler_cfg):
    raw = cfg.get('optimizer', {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('oracle_critic_pretrain.optimizer must be a table')
    optimizer_type = normalize_optimizer_type(raw.get('type', 'adamw'))
    betas = tuple(raw.get('betas', config['optim'].get('betas', (0.9, 0.999))))
    if len(betas) != 2:
        raise ValueError('Oracle critic optimizer betas must contain two values')
    resolved = {
        'type': optimizer_type,
        'betas': [float(betas[0]), float(betas[1])],
        'eps': float(raw.get('eps', config['optim'].get('eps', 1e-8))),
    }
    if optimizer_type == 'schedule_free_adamw':
        resolved.update({
            'lr': float(raw.get('lr', scheduler_cfg.get('peak', 0.0))),
            'warmup_steps': int(
                raw.get('warmup_steps', scheduler_cfg.get('warm_up_steps', 0)) or 0
            ),
            'r': float(raw.get('r', 0.0)),
            'weight_lr_power': float(raw.get('weight_lr_power', 2.0)),
            'foreach': bool(raw.get('foreach', True)),
        })
        if float(raw.get('inner_momentum', 0.0)) != 0.0:
            raise ValueError(
                'schedulefree==1.4.1 does not support inner_momentum'
            )
        if resolved['lr'] <= 0:
            raise ValueError('schedule-free AdamW requires a positive optimizer.lr')
    return resolved


def build_optimizer(param_groups, cfg, scheduler_cfg):
    optimizer_cfg = resolved_optimizer_config(cfg, scheduler_cfg)
    if optimizer_cfg['type'] == 'adamw':
        return optim.AdamW(
            param_groups,
            lr=1.0,
            weight_decay=0.0,
            betas=tuple(optimizer_cfg['betas']),
            eps=optimizer_cfg['eps'],
        )

    if normalize_scheduler_type(scheduler_cfg.get('type', 'optimizer')) != 'optimizer':
        raise ValueError(
            'schedule-free AdamW must use scheduler.type="optimizer"; '
            'its warmup and iterate averaging are owned by the optimizer'
        )
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError as exc:
        raise RuntimeError(
            'schedule-free AdamW requires schedulefree==1.4.1 in the training environment'
        ) from exc
    base_lr = optimizer_cfg['lr']
    scaled_groups = []
    for group in param_groups:
        scaled = dict(group)
        scaled['lr'] = float(group.get('lr', 1.0)) * base_lr
        scaled_groups.append(scaled)
    return AdamWScheduleFree(
        scaled_groups,
        lr=base_lr,
        betas=tuple(optimizer_cfg['betas']),
        eps=optimizer_cfg['eps'],
        weight_decay=0.0,
        warmup_steps=optimizer_cfg['warmup_steps'],
        r=optimizer_cfg['r'],
        weight_lr_power=optimizer_cfg['weight_lr_power'],
        foreach=optimizer_cfg['foreach'],
    )


def optimizer_train_mode(optimizer):
    train = getattr(optimizer, 'train', None)
    if callable(train):
        train()


def optimizer_eval_mode(optimizer):
    evaluate = getattr(optimizer, 'eval', None)
    if callable(evaluate):
        evaluate()


def optimizer_requires_eval_checkpoint(optimizer):
    return callable(getattr(optimizer, 'eval', None))


def named_lrs(optimizer, scheduler):
    lrs = scheduler.get_last_lr()
    names = [group.get('name', f'group{idx}') for idx, group in enumerate(optimizer.param_groups)]
    return dict(zip(names, lrs))


def lr_summary(optimizer, scheduler):
    lrs = named_lrs(optimizer, scheduler)
    main_lr = max(lrs.values()) if lrs else 0.0
    encoder_lrs = [lr for name, lr in lrs.items() if name.startswith('oracle_encoder')]
    encoder_lr = min(encoder_lrs) if encoder_lrs else main_lr
    return main_lr, encoder_lr


def component_lrs(optimizer, scheduler):
    lrs = named_lrs(optimizer, scheduler)
    fallback = max(lrs.values()) if lrs else 0.0

    def resolve(prefix):
        values = [lr for name, lr in lrs.items() if name.startswith(prefix)]
        return values[0] if values else fallback

    return {
        'visible': resolve('visible_encoder'),
        'oracle': resolve('oracle_encoder'),
        'fusion': resolve('fusion'),
        'value': resolve('value'),
    }


def build_models(
    device,
    *,
    critic_arch='single_tower',
    oracle_fusion_init=0.5,
    oracle_fusion_mode='linear',
    oracle_fusion_hidden=512,
    exact_zero_sum=False,
    value_loss_mode='mse',
    value_head_hidden=256,
    value_num_bins=100,
    value_target_min=-6.0,
    value_target_max=6.0,
    value_sigma_to_bin_ratio=2.0,
    value_padding_sigma=3.0,
):
    version = config['control']['version']
    critic_arch = normalize_critic_arch(critic_arch)
    if critic_arch == 'dual_tower':
        oracle_brain = OracleDualTowerBrain(
            version=version,
            **config['resnet'],
            Norm='GN',
            oracle_fusion_init=oracle_fusion_init,
            oracle_fusion_mode=normalize_oracle_fusion_mode(oracle_fusion_mode),
            oracle_fusion_hidden=int(oracle_fusion_hidden),
        ).to(device)
    else:
        oracle_brain = Brain(version=version, is_oracle=True, **config['resnet'], Norm='GN').to(device)
    value_loss_mode = normalize_value_loss_mode(value_loss_mode)
    if value_loss_mode == 'hl_gauss':
        value_net = HLGaussValueHead(
            num_players=4,
            hidden_size=int(value_head_hidden),
            num_bins=int(value_num_bins),
            target_min=float(value_target_min),
            target_max=float(value_target_max),
            sigma_to_bin_ratio=float(value_sigma_to_bin_ratio),
            padding_sigma=float(value_padding_sigma),
            zero_sum=exact_zero_sum,
        ).to(device)
    else:
        value_net = ValueHead(
            num_players=4,
            hidden_size=int(value_head_hidden),
            zero_sum=exact_zero_sum,
        ).to(device)
    return oracle_brain, value_net


def load_resnet_from_brain_encoder(target_resnet, source_state, *, source_prefix='encoder.'):
    target_state = target_resnet.state_dict()
    loaded_keys = []
    skipped_keys = []
    sliced_input_keys = []
    with torch.no_grad():
        for key, target_tensor in target_state.items():
            source_key = f'{source_prefix}{key}'
            source_tensor = source_state.get(source_key)
            if source_tensor is None:
                skipped_keys.append(key)
                continue
            if source_tensor.shape == target_tensor.shape:
                target_tensor.copy_(source_tensor)
                loaded_keys.append(key)
                continue
            if (
                key == 'net.0.weight'
                and source_tensor.ndim == 3
                and target_tensor.ndim == 3
                and source_tensor.shape[0] == target_tensor.shape[0]
                and source_tensor.shape[2] == target_tensor.shape[2]
                and source_tensor.shape[1] >= target_tensor.shape[1]
            ):
                target_tensor.copy_(source_tensor[:, :target_tensor.shape[1], :])
                loaded_keys.append(key)
                sliced_input_keys.append(key)
                continue
            skipped_keys.append(key)
    target_resnet.load_state_dict(target_state)
    return {
        'loaded_keys': loaded_keys,
        'skipped_keys': skipped_keys,
        'sliced_input_keys': sliced_input_keys,
    }


def init_dual_tower_from_visible_state(
    oracle_brain,
    source_state,
    *,
    oracle_tower_init='visible_transfer',
    oracle_first_conv_init='legacy_mean_scaled',
    oracle_hand_init_scale=1.0 / 3.0,
):
    if not is_dual_tower_brain(oracle_brain):
        return None

    oracle_tower_init = normalize_oracle_tower_init(oracle_tower_init)
    oracle_first_conv_init = normalize_oracle_first_conv_init(
        oracle_first_conv_init
    )
    oracle_hand_init_scale = float(oracle_hand_init_scale)
    visible_info = load_resnet_from_brain_encoder(oracle_brain.visible_encoder, source_state)
    oracle_state = oracle_brain.oracle_encoder.state_dict()
    visible_first = source_state.get(FIRST_CONV_KEY)
    visible_channels = obs_shape(oracle_brain.version)[0]
    loaded_oracle_keys = []
    skipped_oracle_keys = []
    first_conv_init = 'missing'
    with torch.no_grad():
        for key, target_tensor in oracle_state.items():
            if key == 'net.0.weight':
                if torch.is_tensor(visible_first):
                    out_channels = min(target_tensor.shape[0], visible_first.shape[0])
                    kernel = min(target_tensor.shape[2], visible_first.shape[2])
                    source_channels = visible_first.shape[1]
                    target_channels = target_tensor.shape[1]
                    oracle_start = visible_channels
                    oracle_end = oracle_start + target_channels
                    if source_channels >= oracle_end:
                        target_tensor.zero_()
                        target_tensor[:out_channels, :, :kernel].copy_(
                            visible_first[:out_channels, oracle_start:oracle_end, :kernel]
                        )
                        first_conv_init = 'single_oracle_slice'
                        loaded_oracle_keys.append(key)
                    elif source_channels == target_channels:
                        target_tensor.zero_()
                        target_tensor[:out_channels, :, :kernel].copy_(
                            visible_first[:out_channels, :, :kernel]
                        )
                        first_conv_init = 'channel_aligned_copy'
                        loaded_oracle_keys.append(key)
                    elif oracle_first_conv_init == 'legacy_mean_scaled':
                        target_tensor.zero_()
                        source_mean = visible_first[:out_channels, :, :kernel].mean(dim=1, keepdim=True)
                        target_tensor[:out_channels, :, :kernel].copy_(
                            source_mean.expand(-1, target_channels, -1)
                        )
                        target_tensor.mul_(0.25)
                        first_conv_init = 'visible_channel_mean_scaled'
                        loaded_oracle_keys.append(key)
                    elif oracle_first_conv_init == 'hand_aligned':
                        target_tensor.zero_()
                        opponent_stride = 15 if oracle_brain.version == 1 else 17
                        hand_channels = min(7, source_channels)
                        for opponent_id in range(3):
                            target_start = opponent_id * opponent_stride
                            target_end = min(target_start + hand_channels, target_channels)
                            copied_channels = target_end - target_start
                            if copied_channels <= 0:
                                continue
                            target_tensor[
                                :out_channels,
                                target_start:target_end,
                                :kernel,
                            ].copy_(
                                visible_first[
                                    :out_channels,
                                    :copied_channels,
                                    :kernel,
                                ] * oracle_hand_init_scale
                            )
                        first_conv_init = 'opponent_hand_aligned'
                        loaded_oracle_keys.append(key)
                    else:
                        first_conv_init = 'native_random'
                        skipped_oracle_keys.append(key)
                else:
                    skipped_oracle_keys.append(key)
                continue
            source_tensor = source_state.get(f'encoder.{key}')
            if (
                oracle_tower_init == 'visible_transfer'
                and source_tensor is not None
                and source_tensor.shape == target_tensor.shape
            ):
                target_tensor.copy_(source_tensor)
                loaded_oracle_keys.append(key)
            else:
                skipped_oracle_keys.append(key)
        oracle_brain.oracle_encoder.load_state_dict(oracle_state)
    return {
        'visible_encoder': visible_info,
        'oracle_encoder': {
            'loaded_keys': loaded_oracle_keys,
            'skipped_keys': skipped_oracle_keys,
            'tower_init': oracle_tower_init,
            'first_conv_init': first_conv_init,
        },
    }


def maybe_init_from_checkpoint(
    oracle_brain,
    value_net,
    init_state_file,
    device,
    *,
    strict_oracle_checkpoint=False,
    oracle_input_init_scale=0.02,
    oracle_tower_init='visible_transfer',
    oracle_first_conv_init='legacy_mean_scaled',
    oracle_hand_init_scale=1.0 / 3.0,
):
    if not init_state_file:
        return {'source': '', 'loaded': False}
    if not path.exists(init_state_file):
        raise FileNotFoundError(f'oracle_critic_pretrain.init_state_file does not exist: {init_state_file}')

    state = torch.load(init_state_file, weights_only=False, map_location=device)
    loaded = {
        'source': init_state_file,
        'source_sha256': sha256_file(init_state_file),
        'source_steps': int(state.get('steps', 0)),
        'source_rank_points': state.get('config', {}).get('env', {}).get('pts'),
        'initialization_mode': 'weights_only_new_optimizer_and_validation_baseline',
        'loaded': True,
        'oracle_brain': None,
        'value_net': False,
        'strict_oracle_checkpoint': bool(strict_oracle_checkpoint),
    }
    if state.get('oracle_brain') is not None:
        source_oracle_state = state['oracle_brain']
        if strict_oracle_checkpoint:
            loaded['oracle_brain'] = load_brain_state_strict(
                oracle_brain,
                source_oracle_state,
                checkpoint_name='oracle_critic_pretrain.init_state_file',
            )
        else:
            try:
                oracle_brain.load_state_dict(source_oracle_state)
                loaded['oracle_brain'] = {
                    'loaded_keys': tuple(source_oracle_state.keys()),
                    'skipped_keys': (),
                    'expanded_input_keys': (),
                    'extra_input_init_scale': 0.0,
                    'direct': True,
                }
            except RuntimeError:
                if is_dual_tower_brain(oracle_brain):
                    source_visible_state = state.get('mortal') or source_oracle_state
                    loaded['oracle_brain'] = init_dual_tower_from_visible_state(
                        oracle_brain,
                        source_visible_state,
                        oracle_tower_init=oracle_tower_init,
                        oracle_first_conv_init=oracle_first_conv_init,
                        oracle_hand_init_scale=oracle_hand_init_scale,
                    )
                else:
                    loaded['oracle_brain'] = load_brain_state_with_input_bridge(
                        oracle_brain,
                        source_oracle_state,
                        extra_input_init_scale=oracle_input_init_scale,
                    )
    elif state.get('mortal') is not None:
        if strict_oracle_checkpoint:
            raise ValueError(
                'strict oracle critic init checkpoint must contain oracle_brain; '
                f'got visible-only mortal checkpoint: {init_state_file}'
            )
        if is_dual_tower_brain(oracle_brain):
            loaded['oracle_brain'] = init_dual_tower_from_visible_state(
                oracle_brain,
                state['mortal'],
                oracle_tower_init=oracle_tower_init,
                oracle_first_conv_init=oracle_first_conv_init,
                oracle_hand_init_scale=oracle_hand_init_scale,
            )
        else:
            loaded['oracle_brain'] = load_brain_state_with_input_bridge(
                oracle_brain,
                state['mortal'],
                extra_input_init_scale=oracle_input_init_scale,
            )
    if state.get('value_net') is not None:
        value_net.load_state_dict(state['value_net'])
        loaded['value_net'] = True
    return loaded


def summarize_init_info(init_info):
    if not isinstance(init_info, dict):
        return init_info
    summary = dict(init_info)
    bridge_info = summary.get('oracle_brain')
    if isinstance(bridge_info, dict):
        if 'visible_encoder' in bridge_info or 'oracle_encoder' in bridge_info:
            visible_info = bridge_info.get('visible_encoder', {})
            oracle_info = bridge_info.get('oracle_encoder', {})
            summary['oracle_brain'] = {
                'visible_loaded_keys': len(visible_info.get('loaded_keys', ())),
                'visible_skipped_keys': len(visible_info.get('skipped_keys', ())),
                'oracle_loaded_keys': len(oracle_info.get('loaded_keys', ())),
                'oracle_skipped_keys': len(oracle_info.get('skipped_keys', ())),
                'oracle_tower_init': oracle_info.get('tower_init'),
                'oracle_first_conv_init': oracle_info.get('first_conv_init'),
                'direct': False,
            }
            return summary
        summary['oracle_brain'] = {
            'loaded_keys': len(bridge_info.get('loaded_keys', ())),
            'skipped_keys': len(bridge_info.get('skipped_keys', ())),
            'expanded_input_keys': tuple(bridge_info.get('expanded_input_keys', ())),
            'extra_input_init_scale': bridge_info.get('extra_input_init_scale'),
            'direct': bool(bridge_info.get('direct', False)),
            'strict': bool(bridge_info.get('strict', False)),
        }
    return summary


def model_forward(
    oracle_brain,
    value_net,
    obs,
    invisible_obs,
    *,
    enable_amp,
    device_type,
    return_logits=False,
):
    with torch.autocast(device_type, enabled=enable_amp):
        phi = oracle_brain(obs, invisible_obs=invisible_obs)
        if isinstance(value_net, HLGaussValueHead):
            logits = value_net.logits(phi)
            pred = value_net.values_from_logits(logits)
        else:
            logits = None
            pred = value_net(phi)
    return (pred, logits) if return_logits else pred


def normalize_eval_input_modes(value):
    if value is None:
        return ORACLE_EVAL_INPUT_MODES
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    normalized = []
    for item in value:
        mode = str(item).strip().lower()
        if mode not in ORACLE_EVAL_INPUT_MODES:
            raise ValueError(
                f'unsupported Oracle eval input mode {item!r}; '
                f'expected one of {ORACLE_EVAL_INPUT_MODES}'
            )
        if mode not in normalized:
            normalized.append(mode)
    if not normalized:
        raise ValueError('Oracle eval input modes must not be empty')
    return tuple(normalized)


def normalize_game_id_subset(modulus, remainders):
    modulus = int(modulus or 1)
    if modulus <= 0:
        raise ValueError('game id modulus must be positive')
    if remainders is None:
        remainders = []
    if isinstance(remainders, (str, int)):
        remainders = str(remainders).replace(',', ' ').split()
    normalized = tuple(sorted({int(value) for value in remainders}))
    if not normalized:
        normalized = tuple(range(modulus))
    if normalized[0] < 0 or normalized[-1] >= modulus:
        raise ValueError(
            f'game id remainders must be in [0, {modulus}), got {normalized}'
        )
    return modulus, normalized


def grad_scaler_step_succeeded(scale_before, scale_after):
    return float(scale_after) >= float(scale_before)


def summarize_oracle_dependency(val_by_input):
    true_metrics = val_by_input.get('true')
    if true_metrics is None:
        return {}

    summary = {}
    for baseline in ('shuffled', 'zero'):
        baseline_metrics = val_by_input.get(baseline)
        if baseline_metrics is None:
            continue
        loss_improvement = float(baseline_metrics['loss']) - float(true_metrics['loss'])
        summary[f'true_vs_{baseline}'] = {
            'loss_improvement': loss_improvement,
            'relative_loss_improvement': (
                loss_improvement / float(baseline_metrics['loss'])
                if float(baseline_metrics['loss']) != 0.0
                else 0.0
            ),
            'corr_gain': float(true_metrics['corr']) - float(baseline_metrics['corr']),
            'explained_variance_gain': (
                float(true_metrics['explained_variance'])
                - float(baseline_metrics['explained_variance'])
            ),
        }
    return summary


def transform_eval_invisible_obs(invisible_obs, mode):
    mode = normalize_eval_input_modes((mode,))[0]
    if mode == 'true':
        return invisible_obs
    if mode == 'zero':
        return torch.zeros_like(invisible_obs)
    if invisible_obs.ndim < 2:
        raise ValueError(
            f'shuffled Oracle input expects at least 2 dims, got {tuple(invisible_obs.shape)}'
        )
    if invisible_obs.shape[0] > 1:
        return torch.roll(invisible_obs, shifts=max(invisible_obs.shape[0] // 2, 1), dims=0)
    return torch.roll(invisible_obs, shifts=max(invisible_obs.shape[1] // 2, 1), dims=1)


def load_teacher_models(teacher_state_file, device):
    if not teacher_state_file:
        return None, None
    if not path.exists(teacher_state_file):
        raise FileNotFoundError(f'oracle critic teacher_state_file does not exist: {teacher_state_file}')
    state = torch.load(teacher_state_file, weights_only=False, map_location=device)
    teacher_brain = Brain(version=config['control']['version'], is_oracle=True, **config['resnet'], Norm='GN').to(device)
    teacher_value_net = ValueHead(num_players=4).to(device)
    teacher_brain.load_state_dict(state['oracle_brain'])
    teacher_value_net.load_state_dict(state['value_net'])
    teacher_brain.eval()
    teacher_value_net.eval()
    for param in teacher_brain.parameters():
        param.requires_grad_(False)
    for param in teacher_value_net.parameters():
        param.requires_grad_(False)
    return teacher_brain, teacher_value_net


def batch_metrics(
    pred,
    target,
    *,
    objective_loss=None,
    target_objective=None,
    target_mse=None,
    teacher_mse=None,
    zero_sum_loss=None,
    game_id=None,
):
    err = pred.detach().float() - target.detach().float()
    loss_sum = err.square().sum().item()
    abs_sum = err.abs().sum().item()
    count = err.numel()
    zero_sum_abs = pred.detach().float().sum(dim=-1).abs().sum().item()
    sample_count = pred.shape[0]
    pred_cpu = pred.detach().float().cpu()
    target_cpu = target.detach().float().cpu()
    result = {
        'loss_sum': float(loss_sum),
        'abs_sum': float(abs_sum),
        'count': int(count),
        'zero_sum_abs': float(zero_sum_abs),
        'sample_count': int(sample_count),
        'pred': pred_cpu,
        'target': target_cpu,
    }
    if game_id is not None:
        result['game_id'] = torch.as_tensor(game_id).detach().cpu().reshape(-1)
    for name, value in (
        ('objective_loss', objective_loss),
        ('target_objective', target_objective),
        ('target_mse', target_mse),
        ('teacher_mse', teacher_mse),
        ('zero_sum_loss', zero_sum_loss),
    ):
        if value is not None:
            result[f'{name}_sum'] = float(value.detach().float().item())
            result[f'{name}_batches'] = 1
    return result


def binned_calibration_metrics(pred, target, *, num_bins=20):
    pred = torch.as_tensor(pred).detach().double().cpu().reshape(-1)
    target = torch.as_tensor(target).detach().double().cpu().reshape(-1)
    if pred.shape != target.shape or pred.numel() == 0:
        raise ValueError('calibration prediction and target must be non-empty and aligned')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('calibration prediction and target must be finite')
    num_bins = min(int(num_bins), int(pred.numel()))
    if num_bins <= 0:
        raise ValueError('calibration num_bins must be positive')

    order = torch.argsort(pred)
    weighted_abs = 0.0
    weighted_square = 0.0
    max_abs = 0.0
    for indices in torch.tensor_split(order, num_bins):
        gap = float((pred[indices].mean() - target[indices].mean()).item())
        weight = int(indices.numel())
        abs_gap = abs(gap)
        weighted_abs += weight * abs_gap
        weighted_square += weight * gap * gap
        max_abs = max(max_abs, abs_gap)
    count = float(pred.numel())
    return {
        'binned_calibration_bins': num_bins,
        'binned_calibration_mae': weighted_abs / count,
        'binned_calibration_rmse': math.sqrt(weighted_square / count),
        'binned_calibration_max_abs': max_abs,
    }


def summarize_regression_slices(pred, target):
    pred = torch.as_tensor(pred).detach().double().cpu().reshape(-1)
    target = torch.as_tensor(target).detach().double().cpu().reshape(-1)
    if pred.shape != target.shape or pred.numel() == 0:
        raise ValueError('slice prediction and target must be non-empty and aligned')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('slice prediction and target must be finite')

    masks = {
        'all': torch.ones_like(target, dtype=torch.bool),
        'exact_zero': target == 0.0,
        'nonzero': target != 0.0,
        'abs_ge_2': target.abs() >= 2.0,
        'abs_ge_4': target.abs() >= 4.0,
    }
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum().item())
        if count == 0:
            continue
        error = pred[mask] - target[mask]
        result[name] = {
            'count': count,
            'fraction': count / float(target.numel()),
            'loss': float(error.square().mean().item()),
            'mae': float(error.abs().mean().item()),
            'bias': float(error.mean().item()),
            'pred_mean': float(pred[mask].mean().item()),
            'target_mean': float(target[mask].mean().item()),
        }
    return result


def _cluster_sum_records(values, game_id, *, mask=None):
    values = torch.as_tensor(values).detach().double().cpu().reshape(-1)
    game_id = torch.as_tensor(game_id).detach().cpu().reshape(-1)
    if values.shape != game_id.shape:
        raise ValueError('cluster values and game ids must have the same shape')
    if mask is not None:
        mask = torch.as_tensor(mask).detach().cpu().bool().reshape(-1)
        if mask.shape != values.shape:
            raise ValueError('cluster mask and values must have the same shape')
        values = values[mask]
        game_id = game_id[mask]
    if values.numel() == 0:
        return []
    unique_games, inverse = torch.unique(game_id, sorted=True, return_inverse=True)
    value_sum = torch.zeros(unique_games.numel(), dtype=torch.float64).scatter_add_(
        0,
        inverse,
        values,
    )
    value_count = torch.zeros(unique_games.numel(), dtype=torch.int64).scatter_add_(
        0,
        inverse,
        torch.ones_like(inverse, dtype=torch.int64),
    )
    return [
        [int(gid), float(total), int(count)]
        for gid, total, count in zip(unique_games, value_sum, value_count)
    ]


def finalize_metrics(parts, *, include_cluster_records=False):
    total_loss = sum(item['loss_sum'] for item in parts)
    total_abs = sum(item['abs_sum'] for item in parts)
    total_count = sum(item['count'] for item in parts)
    total_zero = sum(item['zero_sum_abs'] for item in parts)
    total_samples = sum(item['sample_count'] for item in parts)
    if total_count <= 0:
        return {'loss': math.inf, 'mae': math.inf, 'corr': 0.0, 'zero_sum_mae': math.inf}

    pred_by_output = torch.cat([item['pred'] for item in parts], dim=0)
    target_by_output = torch.cat([item['target'] for item in parts], dim=0)
    pred = pred_by_output.reshape(-1)
    target = target_by_output.reshape(-1)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = pred_centered.norm() * target_centered.norm()
    corr = float((pred_centered @ target_centered / denom).item()) if denom.item() > 0 else 0.0
    error = pred - target
    scale_denom = pred.square().sum()
    calibration_scale = (
        float(((pred * target).sum() / scale_denom).item())
        if scale_denom.item() > 0
        else 1.0
    )
    calibrated_error = calibration_scale * pred - target
    target_variance = target.var(unbiased=False)
    explained_variance = (
        1.0 - float(error.var(unbiased=False).item() / target_variance.item())
        if target_variance.item() > 0
        else 0.0
    )
    metrics = {
        'loss': total_loss / total_count,
        'rmse': math.sqrt(total_loss / total_count),
        'mae': total_abs / total_count,
        'corr': corr,
        'explained_variance': explained_variance,
        'bias': float(error.mean().item()),
        'calibration_scale': calibration_scale,
        'calibrated_loss': float(calibrated_error.square().mean().item()),
        'pred_mean': float(pred.mean().item()),
        'pred_std': float(pred.std(unbiased=False).item()),
        'target_mean': float(target.mean().item()),
        'target_std': float(target.std(unbiased=False).item()),
        'zero_baseline_loss': float(target.square().mean().item()),
        'zero_sum_mae': total_zero / max(total_samples, 1),
        'num_batches': len(parts),
        'num_samples': total_samples,
        'num_values': total_count,
    }
    metrics.update(binned_calibration_metrics(pred, target))
    metrics['slices'] = summarize_regression_slices(pred, target)
    game_ids = [item.get('game_id') for item in parts]
    if game_ids and all(item is not None for item in game_ids):
        game_id = torch.cat(game_ids, dim=0)
        row_loss = (pred_by_output - target_by_output).double().square().mean(dim=-1)
        unique_games, inverse = torch.unique(game_id, sorted=True, return_inverse=True)
        game_loss_sum = torch.zeros(
            unique_games.shape[0], dtype=row_loss.dtype
        ).scatter_add_(0, inverse, row_loss)
        game_sample_count = torch.zeros(
            unique_games.shape[0], dtype=row_loss.dtype
        ).scatter_add_(0, inverse, torch.ones_like(row_loss))
        game_loss = game_loss_sum / game_sample_count
        game_count = int(unique_games.numel())
        state_loss = float(metrics['loss'])
        cluster_residual = game_loss_sum - state_loss * game_sample_count
        if game_count > 1:
            cluster_se = math.sqrt(
                game_count / (game_count - 1)
                * float(cluster_residual.square().sum().item())
                / float(row_loss.numel() ** 2)
            )
            game_balanced_se = float(
                game_loss.std(unbiased=True).item() / math.sqrt(game_count)
            )
        else:
            cluster_se = 0.0
            game_balanced_se = 0.0
        metrics.update({
            'num_games': game_count,
            'loss_cluster_se': cluster_se,
            'loss_ci95_low': state_loss - 1.96 * cluster_se,
            'loss_ci95_high': state_loss + 1.96 * cluster_se,
            'game_balanced_loss': float(game_loss.mean().item()),
            'game_balanced_loss_se': game_balanced_se,
        })
        if include_cluster_records:
            primary_error = (
                pred_by_output[:, 0].double() - target_by_output[:, 0].double()
            )
            primary_square = primary_error.square()
            primary_target = target_by_output[:, 0].double()
            records = {
                'primary_loss': _cluster_sum_records(primary_square, game_id),
                'all_players_loss': _cluster_sum_records(row_loss, game_id),
                'p0_mae': _cluster_sum_records(primary_error.abs(), game_id),
            }
            for slice_name, mask in (
                ('exact_zero_loss', primary_target == 0.0),
                ('nonzero_loss', primary_target != 0.0),
                ('abs_ge_2_loss', primary_target.abs() >= 2.0),
                ('abs_ge_4_loss', primary_target.abs() >= 4.0),
            ):
                records[slice_name] = _cluster_sum_records(
                    primary_square,
                    game_id,
                    mask=mask,
                )
            metrics['_adaptive_cluster_records'] = records
    output_metrics = {}
    for output_idx in range(pred_by_output.shape[1]):
        output_pred = pred_by_output[:, output_idx]
        output_target = target_by_output[:, output_idx]
        output_error = output_pred - output_target
        output_scale_denom = output_pred.square().sum()
        output_calibration_scale = (
            float(
                (
                    (output_pred * output_target).sum()
                    / output_scale_denom
                ).item()
            )
            if output_scale_denom.item() > 0
            else 1.0
        )
        output_calibrated_error = (
            output_calibration_scale * output_pred - output_target
        )
        output_pred_centered = output_pred - output_pred.mean()
        output_target_centered = output_target - output_target.mean()
        output_denom = output_pred_centered.norm() * output_target_centered.norm()
        output_corr = (
            float((output_pred_centered @ output_target_centered / output_denom).item())
            if output_denom.item() > 0
            else 0.0
        )
        output_target_variance = output_target.var(unbiased=False)
        output_explained_variance = (
            1.0
            - float(
                output_error.var(unbiased=False).item()
                / output_target_variance.item()
            )
            if output_target_variance.item() > 0
            else 0.0
        )
        output_metrics[f'relative_player_{output_idx}'] = {
            'loss': float(output_error.square().mean().item()),
            'mae': float(output_error.abs().mean().item()),
            'corr': output_corr,
            'explained_variance': output_explained_variance,
            'bias': float(output_error.mean().item()),
            'calibration_scale': output_calibration_scale,
            'calibrated_loss': float(
                output_calibrated_error.square().mean().item()
            ),
        }
        output_metrics[f'relative_player_{output_idx}'].update(
            binned_calibration_metrics(output_pred, output_target)
        )
        output_metrics[f'relative_player_{output_idx}']['slices'] = (
            summarize_regression_slices(output_pred, output_target)
        )
    metrics['outputs'] = output_metrics
    for name in (
        'objective_loss',
        'target_objective',
        'target_mse',
        'teacher_mse',
        'zero_sum_loss',
    ):
        batches = sum(item.get(f'{name}_batches', 0) for item in parts)
        if batches > 0:
            metrics[name] = sum(item.get(f'{name}_sum', 0.0) for item in parts) / batches
    return metrics


@torch.inference_mode()
@evaluation_thread_scope
def evaluate_modes(
    oracle_brain,
    value_net,
    loader,
    device,
    *,
    enable_amp,
    max_batches,
    input_modes=ORACLE_EVAL_INPUT_MODES,
    log_every_batches=0,
    label='eval',
    game_id_modulus=1,
    game_id_remainders=(),
    include_cluster_records=False,
):
    input_modes = normalize_eval_input_modes(input_modes)
    game_id_modulus, game_id_remainders = normalize_game_id_subset(
        game_id_modulus,
        game_id_remainders,
    )
    use_game_subset = len(game_id_remainders) != game_id_modulus
    oracle_brain.eval()
    value_net.eval()
    parts_by_mode = {mode: [] for mode in input_modes}
    started_at = time.monotonic()
    samples_seen = 0
    for batch_idx, batch in enumerate(loader):
        if external_pause_requested(resolve_external_pause_file(config.get('oracle_critic_pretrain', {}))):
            raise OracleEvaluationPaused('Oracle evaluation paused at a batch boundary')
        if max_batches > 0 and batch_idx >= max_batches:
            break
        obs, invisible_obs, target, _player_id = batch[:4]
        game_id = batch[4] if len(batch) > 4 else None
        if use_game_subset:
            if game_id is None:
                raise ValueError('game id subset evaluation requires game ids in each batch')
            folded = torch.remainder(torch.as_tensor(game_id).reshape(-1), game_id_modulus)
            keep = torch.zeros_like(folded, dtype=torch.bool)
            for remainder in game_id_remainders:
                keep |= folded == remainder
            if not bool(keep.any()):
                continue
            obs = obs[keep]
            invisible_obs = invisible_obs[keep]
            target = target[keep]
            game_id = torch.as_tensor(game_id).reshape(-1)[keep]
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        target = target.to(dtype=torch.float32, device=device, non_blocking=True)
        samples_seen += int(obs.shape[0])
        for mode in input_modes:
            pred = model_forward(
                oracle_brain,
                value_net,
                obs,
                transform_eval_invisible_obs(invisible_obs, mode),
                enable_amp=enable_amp,
                device_type=device.type,
            )
            parts_by_mode[mode].append(batch_metrics(pred, target, game_id=game_id))
        completed_batches = batch_idx + 1
        if log_every_batches > 0 and completed_batches % log_every_batches == 0:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            logging.info(
                '%s progress batches=%s samples=%s elapsed=%.1fs batches_per_s=%.3f',
                label,
                completed_batches,
                samples_seen,
                elapsed,
                completed_batches / elapsed,
            )
    oracle_brain.train()
    value_net.train()
    results = {
        mode: finalize_metrics(parts, include_cluster_records=include_cluster_records)
        for mode, parts in parts_by_mode.items()
    }
    if getattr(getattr(loader, 'dataset', None), 'cluster_unit', None) == 'full_seed_key_four_seat_group':
        for metrics in results.values():
            metrics['ci_cluster_unit'] = 'full_seed_key_four_seat_group'
            metrics['ci_estimand'] = 'state_weighted_mean_with_seed_group_cluster_se'
            # Keep legacy keys for controllers; name their changed unit explicitly.
            metrics['num_seed_groups'] = metrics.get('num_games', 0)
            metrics['num_games_key_unit'] = 'seed_groups'
            metrics['seed_group_balanced_loss'] = metrics.get('game_balanced_loss')
            metrics['seed_group_balanced_loss_se'] = metrics.get('game_balanced_loss_se')
    return results


def evaluate(
    oracle_brain,
    value_net,
    loader,
    device,
    *,
    enable_amp,
    max_batches,
    log_every_batches=0,
    label='eval',
    game_id_modulus=1,
    game_id_remainders=(),
    include_cluster_records=False,
):
    return evaluate_modes(
        oracle_brain,
        value_net,
        loader,
        device,
        enable_amp=enable_amp,
        max_batches=max_batches,
        input_modes=('true',),
        log_every_batches=log_every_batches,
        label=label,
        game_id_modulus=game_id_modulus,
        game_id_remainders=game_id_remainders,
        include_cluster_records=include_cluster_records,
    )['true']


def resolve_convergence_config(cfg):
    raw = cfg.get('convergence', {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('oracle_critic_pretrain.convergence must be a table')
    if not bool(raw.get('enabled', False)):
        return None
    convergence_config = ConvergenceConfig.from_mapping(raw)
    if convergence_config.metric not in ORACLE_CONVERGENCE_METRICS:
        raise ValueError(
            'oracle critic convergence.metric must be one of '
            f'{ORACLE_CONVERGENCE_METRICS}, got {convergence_config.metric!r}'
        )
    return convergence_config


def resolve_adaptive_curriculum_config(cfg):
    raw = cfg.get('adaptive_curriculum', {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('oracle_critic_pretrain.adaptive_curriculum must be a table')
    if not bool(raw.get('enabled', False)):
        return None
    adaptive = AdaptiveCurriculumConfig.from_mapping(raw)
    if (adaptive.selection_protocol == 'primary_with_diagnostics' and
            (adaptive.primary.name != 'primary_loss' or adaptive.primary.direction != 'lower')):
        raise ValueError('Oracle primary_with_diagnostics requires lower primary_loss (p0 MSE)')
    return adaptive


def convergence_contract(convergence_config):
    if convergence_config is None:
        return None
    return {
        'core_optimizer_steps': convergence_config.core_optimizer_steps,
        'tail_lr_levels': list(convergence_config.tail_lr_levels),
        'smoothing_checks': convergence_config.smoothing_checks,
        'improvement_delta': convergence_config.improvement_delta,
        'reduce_patience_steps': convergence_config.reduce_patience_steps,
        'stop_patience_steps': convergence_config.stop_patience_steps,
        'min_level_steps': convergence_config.min_level_steps,
        'metric': convergence_config.metric,
    }


def adaptive_curriculum_contract(adaptive_config):
    if adaptive_config is None:
        return None
    return {
        **({'selection_protocol': adaptive_config.selection_protocol}
           if adaptive_config.selection_protocol != 'legacy_guardrails' else {}),
        'phase_name': adaptive_config.phase_name,
        'primary': {
            'name': adaptive_config.primary.name,
            'direction': adaptive_config.primary.direction,
            'meaningful_delta': adaptive_config.primary.meaningful_delta,
        },
        'final_phase': adaptive_config.final_phase,
        'gate_every_steps': adaptive_config.gate_every_steps,
        'required_futile_gates': adaptive_config.required_futile_gates,
        'max_unresolved_gates': adaptive_config.max_unresolved_gates,
        'min_paired_games': adaptive_config.min_paired_games,
        'confidence_z': adaptive_config.confidence_z,
        'primary_noninferiority_margin': (
            adaptive_config.primary_noninferiority_margin
        ),
        'guardrails': [
            {
                'name': spec.name,
                'direction': spec.direction,
                'meaningful_delta': spec.meaningful_delta,
                'noninferiority_margin': spec.noninferiority_margin,
            }
            for spec in adaptive_config.guardrails
        ],
        'lr_levels': list(adaptive_config.lr_levels),
    }


def validate_convergence_schedule(
    convergence_config,
    *,
    scheduler_cfg,
    scheduler_horizon_steps,
    max_steps,
    optimizer_cfg=None,
):
    scheduler_type = normalize_scheduler_type(scheduler_cfg.get('type', 'cosine'))
    if convergence_config is None:
        if scheduler_type == 'cosine' and scheduler_horizon_steps < max_steps:
            raise ValueError(
                'oracle critic scheduler_horizon_steps must be >= max_steps '
                'for cosine runs when convergence tails are disabled'
            )
        return
    if scheduler_type not in {'cosine', 'optimizer'}:
        raise ValueError(
            'Oracle critic convergence tails require cosine scheduling or '
            'Schedule-Free optimizer-owned LR control'
        )
    if convergence_config.core_optimizer_steps != scheduler_horizon_steps:
        raise ValueError(
            'oracle critic convergence.core_optimizer_steps must exactly match '
            f'the scheduler horizon: {convergence_config.core_optimizer_steps} '
            f'!= {scheduler_horizon_steps}'
        )
    if scheduler_type == 'cosine':
        scheduler_final = float(scheduler_cfg.get('final', 0.0) or 0.0)
        if not math.isclose(
            convergence_config.tail_lr_levels[0],
            scheduler_final,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                'the first Oracle critic convergence tail lr must match '
                f'scheduler.final: {convergence_config.tail_lr_levels[0]} '
                f'!= {scheduler_final}'
            )
    else:
        optimizer_cfg = optimizer_cfg or {}
        if optimizer_cfg.get('type') != 'schedule_free_adamw':
            raise ValueError(
                'optimizer-owned convergence tails require Schedule-Free AdamW'
            )
        optimizer_lr = float(optimizer_cfg['lr'])
        if convergence_config.tail_lr_levels[0] >= optimizer_lr:
            raise ValueError(
                'the first Schedule-Free convergence tail lr must be below '
                f'optimizer.lr: {convergence_config.tail_lr_levels[0]} '
                f'>= {optimizer_lr}'
            )
    if max_steps <= convergence_config.core_optimizer_steps:
        raise ValueError(
            'oracle critic convergence max_steps must leave room after the '
            f'cosine core: {max_steps} <= '
            f'{convergence_config.core_optimizer_steps}'
        )


def scheduler_contract(scheduler_cfg):
    explicit_type = 'type' in scheduler_cfg
    scheduler_type = normalize_scheduler_type(scheduler_cfg.get('type', 'cosine'))
    keys_by_type = {
        'cosine': ('init', 'peak', 'final', 'warm_up_steps', 'max_steps'),
        'constant': ('init', 'peak', 'warm_up_steps'),
        'wsd': (
            'init',
            'peak',
            'final',
            'warm_up_steps',
            'stable_steps',
            'decay_steps',
            'decay_style',
        ),
        'plateau': (
            'init',
            'peak',
            'warm_up_steps',
            'factor',
            'patience_steps',
            'threshold',
            'min_lr',
            'metric',
        ),
        'optimizer': (),
    }
    contract = {
        key: scheduler_cfg.get(key)
        for key in keys_by_type[scheduler_type]
    }
    if explicit_type:
        contract = {'type': scheduler_type, **contract}
    return contract


def training_contract(
    cfg,
    scheduler_cfg,
    convergence_config=None,
    adaptive_config=None,
):
    raw_target_loss_weight = cfg.get('target_loss_weight', 1.0)
    target_loss_weight = (
        1.0 if raw_target_loss_weight is None else float(raw_target_loss_weight)
    )
    contract = {
        'critic_arch': normalize_critic_arch(cfg.get('critic_arch', 'single_tower')),
        'train_scope': normalize_train_scope(cfg.get('train_scope', 'oracle_input_value')),
        'target_mode': str(cfg.get('target_mode', 'all_players')),
        'return_mode': str(cfg.get('return_mode', 'score_rank_mc')),
        'discount_gamma': float(cfg.get('discount_gamma', 1.0)),
        'centered_rank_points': (
            np.asarray(config['env']['pts'], dtype=np.float64)
            - np.mean(config['env']['pts'])
        ).tolist(),
        'target_clock_version': ORACLE_TARGET_CLOCK_VERSION,
        'oracle_imputation_version': ORACLE_IMPUTATION_VERSION,
        'val_oracle_imputation_seed': int(cfg.get('val_oracle_imputation_seed', 20260905)),
        'train_oracle_imputation_seed': int(cfg.get('train_oracle_imputation_seed', cfg.get('seed', 20260416))),
        'train_imputation_schedule': 'seed_plus_1000003_stream_pass_v1',
        'exact_zero_sum': bool(cfg.get('exact_zero_sum', False)),
        'oracle_fusion_init': float(cfg.get('oracle_fusion_init', 0.5)),
        'oracle_fusion_mode': normalize_oracle_fusion_mode(
            cfg.get('oracle_fusion_mode', 'linear')
        ),
        'oracle_fusion_hidden': int(cfg.get('oracle_fusion_hidden', 512) or 512),
        'oracle_tower_init': normalize_oracle_tower_init(
            cfg.get('oracle_tower_init', 'visible_transfer')
        ),
        'oracle_first_conv_init': normalize_oracle_first_conv_init(
            cfg.get('oracle_first_conv_init', 'legacy_mean_scaled')
        ),
        'oracle_input_init_scale': float(cfg.get('oracle_input_init_scale', 0.02)),
        'oracle_hand_init_scale': float(cfg.get('oracle_hand_init_scale', 1.0 / 3.0)),
        'encoder_lr_scale': float_config_value(cfg, 'encoder_lr_scale', 1.0),
        'visible_lr_scale': float_config_value(cfg, 'visible_lr_scale', 1.0),
        'oracle_lr_scale': float_config_value(cfg, 'oracle_lr_scale', 1.0),
        'fusion_lr_scale': float_config_value(cfg, 'fusion_lr_scale', 1.0),
        'value_lr_scale': float_config_value(cfg, 'value_lr_scale', 1.0),
        'weight_decay': float(
            cfg.get('weight_decay', config['optim'].get('weight_decay', 0.0)) or 0.0
        ),
        'zero_sum_weight': float(
            cfg.get('zero_sum_weight', config.get('value', {}).get('zero_sum_weight', 0.0))
            or 0.0
        ),
        'teacher_state_file': str(cfg.get('teacher_state_file', '') or ''),
        'teacher_loss_weight': float(cfg.get('teacher_loss_weight', 0.0) or 0.0),
        'target_loss_weight': target_loss_weight,
        'val_state_fold_count': int(cfg.get('val_state_fold_count', 1) or 1),
        'release_train_loader_for_eval': bool(
            cfg.get('release_train_loader_for_eval', True)
        ),
        'scheduler': scheduler_contract(scheduler_cfg),
    }
    if 'optimizer' in cfg:
        contract['optimizer'] = resolved_optimizer_config(cfg, scheduler_cfg)
    if 'val_game_id_modulus' in cfg or 'val_game_id_remainders' in cfg:
        val_modulus, val_remainders = normalize_game_id_subset(
            cfg.get('val_game_id_modulus', 1),
            cfg.get('val_game_id_remainders', ()),
        )
        contract['validation_subset'] = {
            'game_id_modulus': val_modulus,
            'game_id_remainders': list(val_remainders),
        }
    if 'amp_init_scale' in cfg or 'amp_growth_interval' in cfg:
        contract['amp'] = {
            'init_scale': float(cfg.get('amp_init_scale', 65536.0) or 65536.0),
            'growth_interval': int(cfg.get('amp_growth_interval', 2000) or 2000),
        }
    if 'value_loss_mode' in cfg or 'value_head_hidden' in cfg:
        value_loss_mode = normalize_value_loss_mode(cfg.get('value_loss_mode', 'mse'))
        contract['value_loss_mode'] = value_loss_mode
        contract['value_head_hidden'] = int(cfg.get('value_head_hidden', 256) or 256)
        if value_loss_mode == 'hl_gauss':
            contract['value_distribution'] = {
                'num_bins': int(cfg.get('value_num_bins', 100) or 100),
                'target_min': float(cfg.get('value_target_min', -6.0)),
                'target_max': float(cfg.get('value_target_max', 6.0)),
                'sigma_to_bin_ratio': float(
                    cfg.get('value_sigma_to_bin_ratio', 2.0)
                ),
                'padding_sigma': float(cfg.get('value_padding_sigma', 3.0)),
            }
    normalized_output_weights = normalize_target_output_weights(
        cfg.get('target_output_weights')
    )
    if normalized_output_weights is not None:
        contract['target_output_weights'] = list(normalized_output_weights)
    if 'target_output_weights_initial' in cfg:
        initial_output_weights = normalize_target_output_weights(
            cfg.get('target_output_weights_initial')
        )
        target_output_weights_at_step(cfg, 0)
        contract['target_output_weight_schedule'] = {
            'initial': list(initial_output_weights),
            'ramp_start_steps': int(
                cfg.get('target_output_weight_ramp_start_steps', 0) or 0
            ),
            'ramp_end_steps': int(
                cfg.get(
                    'target_output_weight_ramp_end_steps',
                    cfg.get('target_output_weight_ramp_start_steps', 0),
                ) or 0
            ),
        }
    normalized_convergence = convergence_contract(convergence_config)
    if normalized_convergence is not None:
        contract['convergence'] = normalized_convergence
    policy_data = current_policy_data(cfg)
    if policy_data is not None:
        contract['current_policy'] = copy.deepcopy(policy_data['contract'])
    normalized_adaptive = adaptive_curriculum_contract(adaptive_config)
    if normalized_adaptive is not None:
        contract['adaptive_curriculum'] = normalized_adaptive
    return contract


def checkpoint_payload(
    *,
    oracle_brain,
    value_net,
    optimizer,
    scheduler,
    scaler,
    steps,
    best_val_loss,
    best_primary_loss,
    init_info,
    cfg,
    train_info,
    split_info,
    data_progress,
    training_contract_info,
    convergence_state,
    adaptive_curriculum_state=None,
    best_observed_primary_loss=None,
):
    if optimizer_requires_eval_checkpoint(optimizer) and any(
        bool(group.get('train_mode', False)) for group in optimizer.param_groups
    ):
        raise RuntimeError(
            'schedule-free optimizer must be in eval mode while checkpointing'
        )
    train_info_payload = {
        key: value
        for key, value in train_info.items()
        if key != 'first_conv_mask'
    }
    return {
        'timestamp': time.time(),
        'steps': int(steps),
        'oracle_brain': oracle_brain.state_dict(),
        'value_net': value_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_loss': float(best_val_loss),
        'best_primary_loss': float(best_primary_loss),
        **({'best_observed_primary_loss': float(best_observed_primary_loss)}
           if best_observed_primary_loss is not None else {}),
        'config': copy.deepcopy(config),
        'oracle_critic_pretrain': {key: value for key, value in cfg.items() if key != '_current_policy_data'},
        'init_info': init_info,
        'train_info': train_info_payload,
        'file_splits': copy.deepcopy(split_info),
        'data_progress': copy.deepcopy(data_progress),
        'training_contract': copy.deepcopy(training_contract_info),
        'convergence_state': copy.deepcopy(convergence_state),
        'adaptive_curriculum_state': copy.deepcopy(adaptive_curriculum_state),
        BRAIN_IS_ORACLE_KEY: True,
        'resume_supported': True,
        'format': 'oracle_critic_pretrain_v1',
    }


def validate_resume_file_splits(state, split_info):
    saved = state.get('file_splits')
    if saved is None:
        return False
    if saved != split_info:
        raise ValueError(
            'oracle critic resume file split mismatch; refusing to mix training, '
            'dev, or test files across one run'
        )
    return True


def validate_resume_training_contract(
    state,
    expected,
    *,
    convergence_config=None,
):
    saved = state.get('training_contract')
    if saved == expected:
        return False
    if convergence_config is not None and 'convergence' in expected:
        pre_convergence_contract = copy.deepcopy(expected)
        pre_convergence_contract.pop('convergence')
        if (
            saved == pre_convergence_contract
            and int(state.get('steps', 0)) < convergence_config.core_optimizer_steps
        ):
            return True
    raise ValueError(
        'oracle critic resume training contract mismatch; model, optimizer, '
        'target, validation, convergence, and scheduler semantics must stay fixed'
    )


def save_checkpoint(file_path, payload):
    ensure_parent_dir_for_file(file_path)
    temporary = f'{file_path}.{os.getpid()}.tmp'
    try:
        torch.save(payload, temporary)
        os.replace(temporary, file_path)
    finally:
        if path.exists(temporary):
            os.remove(temporary)


def maybe_load_best_val_loss(best_state_file, current_best_val_loss, device):
    if not path.exists(best_state_file):
        return current_best_val_loss
    try:
        best_state = torch.load(best_state_file, weights_only=False, map_location=device)
    except Exception as exc:
        logging.warning('failed to read best oracle critic checkpoint %s: %s', best_state_file, exc)
        return current_best_val_loss
    best_val_loss = best_state.get('best_val_loss')
    if best_val_loss is None:
        return current_best_val_loss
    return min(float(current_best_val_loss), float(best_val_loss))


def maybe_load_best_primary_loss(
    best_primary_state_file,
    current_best_primary_loss,
    device,
):
    if not path.exists(best_primary_state_file):
        return current_best_primary_loss
    try:
        best_state = torch.load(
            best_primary_state_file,
            weights_only=False,
            map_location=device,
        )
    except Exception as exc:
        logging.warning(
            'failed to read best-primary oracle critic checkpoint %s: %s',
            best_primary_state_file,
            exc,
        )
        return current_best_primary_loss
    best_primary_loss = best_state.get('best_primary_loss')
    if best_primary_loss is None:
        val = best_state.get('best_val_metrics', {})
        best_primary_loss = (
            val.get('outputs', {})
            .get('relative_player_0', {})
            .get('loss')
        )
    if best_primary_loss is None:
        return current_best_primary_loss
    return min(float(current_best_primary_loss), float(best_primary_loss))


def maybe_load_protocol_best_loss(checkpoint_file, current_loss, metric_key,
                                  training_contract_info, split_info):
    """Reconcile crash-time companion saves without importing foreign minima."""
    if not path.exists(checkpoint_file):
        return current_loss
    saved = torch.load(checkpoint_file, weights_only=False, map_location='cpu')
    validate_resume_training_contract(saved, training_contract_info)
    if not validate_resume_file_splits(saved, split_info):
        raise ValueError('selection checkpoint is missing file split provenance')
    value = float(saved[metric_key])
    if not math.isfinite(value):
        raise ValueError('selection checkpoint loss must be finite')
    if metric_key != 'best_observed_primary_loss':
        if (saved.get('adaptive_curriculum_state') or {}).get('best_step') != saved.get('steps'):
            raise ValueError('accepted selection checkpoint must be an accepted step')
    return min(current_loss, value)


def train():
    sanitize_sys_path_for_spawn()
    args = parse_args()
    cfg = dict(oracle_pretrain_cfg())
    for key in (
        'max_steps',
        'val_every_steps',
        'dependency_val_every_steps',
        'save_every',
        'max_train_files',
        'max_val_files',
        'max_test_files',
        'val_batches',
        'test_batches',
        'num_workers',
        'batch_size',
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.current_policy_manifest:
        cfg['current_policy_manifest'] = resolve_cli_path(args.current_policy_manifest)
    if args.device:
        cfg['device'] = args.device
    if args.run_name:
        cfg['run_name'] = args.run_name
    if args.init_state_file:
        cfg['init_state_file'] = resolve_cli_path(args.init_state_file)
    if args.return_mode:
        cfg['return_mode'] = args.return_mode
    if args.discount_gamma is not None:
        cfg['discount_gamma'] = args.discount_gamma
    if args.critic_arch:
        cfg['critic_arch'] = args.critic_arch
    if args.teacher_state_file:
        cfg['teacher_state_file'] = resolve_cli_path(args.teacher_state_file)
    if args.teacher_loss_weight is not None:
        cfg['teacher_loss_weight'] = args.teacher_loss_weight
    if args.target_loss_weight is not None:
        cfg['target_loss_weight'] = args.target_loss_weight
    if args.target_output_weights is not None:
        cfg['target_output_weights'] = args.target_output_weights
    if args.target_output_weights_initial is not None:
        cfg['target_output_weights_initial'] = args.target_output_weights_initial
    if args.value_loss_mode is not None:
        cfg['value_loss_mode'] = args.value_loss_mode
    if args.train_scope:
        cfg['train_scope'] = args.train_scope
    if args.tail_blocks is not None:
        cfg['tail_blocks'] = args.tail_blocks
    if args.encoder_lr_scale is not None:
        cfg['encoder_lr_scale'] = args.encoder_lr_scale
    for key in (
        'visible_lr_scale',
        'oracle_lr_scale',
        'fusion_lr_scale',
        'value_lr_scale',
        'weight_decay',
        'oracle_fusion_init',
        'oracle_fusion_hidden',
        'oracle_input_init_scale',
        'oracle_hand_init_scale',
        'amp_init_scale',
        'amp_growth_interval',
        'target_output_weight_ramp_start_steps',
        'target_output_weight_ramp_end_steps',
        'value_head_hidden',
        'value_num_bins',
        'value_target_min',
        'value_target_max',
        'value_sigma_to_bin_ratio',
        'value_padding_sigma',
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.oracle_first_conv_init is not None:
        cfg['oracle_first_conv_init'] = args.oracle_first_conv_init
    if args.oracle_fusion_mode is not None:
        cfg['oracle_fusion_mode'] = args.oracle_fusion_mode
    if args.oracle_tower_init is not None:
        cfg['oracle_tower_init'] = args.oracle_tower_init
    if args.exact_zero_sum is not None:
        cfg['exact_zero_sum'] = args.exact_zero_sum
    scheduler_overrides = {}
    if args.scheduler_peak is not None:
        scheduler_overrides['peak'] = args.scheduler_peak
    if args.scheduler_final is not None:
        scheduler_overrides['final'] = args.scheduler_final
    if args.scheduler_warm_up_steps is not None:
        scheduler_overrides['warm_up_steps'] = args.scheduler_warm_up_steps
    if args.scheduler_horizon_steps is not None:
        cfg['scheduler_horizon_steps'] = args.scheduler_horizon_steps
    if scheduler_overrides:
        scheduler_cfg_from_cli = dict(cfg.get('scheduler', {}) if isinstance(cfg.get('scheduler', {}), dict) else {})
        scheduler_cfg_from_cli.update(scheduler_overrides)
        cfg['scheduler'] = scheduler_cfg_from_cli

    external_pause_file = resolve_external_pause_file(cfg)
    if external_pause_requested(external_pause_file):
        logging.info(
            '[EXTERNAL_PAUSE] request already present before initialization: %s',
            external_pause_file,
        )
        return ORACLE_EXTERNAL_PAUSE_EXIT_CODE

    configure_windows_high_qos(bool(cfg.get('windows_high_qos', False)))
    seed = int(cfg.get('seed', 20260416) or 0)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    train_threads = int(cfg.get('train_torch_num_threads', 0) or 0)
    if train_threads < 0:
        raise ValueError('train_torch_num_threads must be non-negative')
    if train_threads:
        torch.set_num_threads(train_threads)

    device = torch.device(str(cfg.get('device', config['control'].get('device', 'cuda:0'))))
    cuda_memory_fraction = cfg.get('cuda_memory_fraction')
    if cuda_memory_fraction is not None:
        cuda_memory_fraction = float(cuda_memory_fraction)
        if not 0 < cuda_memory_fraction <= 1:
            raise ValueError('cuda_memory_fraction must be in (0, 1]')
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)
    enable_amp = bool(cfg.get('enable_amp', config['control'].get('enable_amp', False)))
    eval_enable_amp = bool(cfg.get('eval_enable_amp', enable_amp))
    allow_tf32 = bool(cfg.get('allow_tf32', config['control'].get('allow_tf32', True)))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    torch.backends.cudnn.benchmark = bool(
        cfg.get('enable_cudnn_benchmark', config['control'].get('enable_cudnn_benchmark', True))
    )

    run_name = str(cfg.get('run_name', 'oracle_critic_terminal_rank')).strip()
    if args.run_name:
        for key in (
            'state_file',
            'best_state_file',
            'best_primary_state_file',
            'best_observed_primary_state_file',
            'tensorboard_dir',
            'metrics_file',
        ):
            cfg.pop(key, None)
    state_file = artifact_path(
        cfg,
        'state_file',
        './checkpoints/oracle_critic/{run_name}_latest.pth',
        run_name,
    )
    best_state_file = artifact_path(
        cfg,
        'best_state_file',
        './checkpoints/oracle_critic/{run_name}_best.pth',
        run_name,
    )
    best_primary_state_file = artifact_path(
        cfg,
        'best_primary_state_file',
        './checkpoints/oracle_critic/{run_name}_best_primary.pth',
        run_name,
    )
    best_observed_primary_state_file = artifact_path(
        cfg, 'best_observed_primary_state_file',
        './checkpoints/oracle_critic/{run_name}_best_observed_primary.pth', run_name,
    )
    adaptive_best_state_file = artifact_path(
        cfg,
        'adaptive_best_state_file',
        './checkpoints/oracle_critic/{run_name}_adaptive_best.pth',
        run_name,
    )
    tensorboard_dir = artifact_path(
        cfg,
        'tensorboard_dir',
        './tb_log_oracle_critic/{run_name}',
        run_name,
    )
    metrics_file = artifact_path(
        cfg,
        'metrics_file',
        './logs/oracle_critic/{run_name}_metrics.jsonl',
        run_name,
    )
    logging.info(
        (
            'oracle critic artifacts: state=%s best=%s best_primary=%s '
            'adaptive_best=%s metrics=%s tb=%s'
        ),
        display_path(state_file),
        display_path(best_state_file),
        display_path(best_primary_state_file),
        display_path(adaptive_best_state_file),
        display_path(metrics_file),
        display_path(tensorboard_dir),
    )
    ensure_dir(tensorboard_dir)
    ensure_parent_dir_for_file(metrics_file)
    if args.fresh and path.exists(metrics_file):
        os.remove(metrics_file)

    train_files, val_files, test_files = build_file_splits(cfg)
    split_info = summarize_file_splits(cfg, train_files, val_files, test_files)
    logging.info(
        'oracle critic pretrain files: train=%s dev=%s test=%s',
        len(train_files),
        len(val_files),
        len(test_files),
    )
    logging.info(
        'oracle critic split fingerprints: train=%s dev=%s test=%s',
        split_info['train']['sha256'][:12],
        split_info['dev']['sha256'][:12],
        split_info['test']['sha256'][:12],
    )

    critic_arch = normalize_critic_arch(cfg.get('critic_arch', 'single_tower'))
    exact_zero_sum = bool(cfg.get('exact_zero_sum', False))
    if (
        exact_zero_sum
        and normalize_value_target_mode(cfg.get('target_mode', 'all_players')) != 'all_players'
    ):
        raise ValueError('exact_zero_sum requires target_mode=all_players')
    oracle_brain, value_net = build_models(
        device,
        critic_arch=critic_arch,
        oracle_fusion_init=float(cfg.get('oracle_fusion_init', 0.5)),
        oracle_fusion_mode=cfg.get('oracle_fusion_mode', 'linear'),
        oracle_fusion_hidden=int(cfg.get('oracle_fusion_hidden', 512) or 512),
        exact_zero_sum=exact_zero_sum,
        value_loss_mode=cfg.get('value_loss_mode', 'mse'),
        value_head_hidden=int(cfg.get('value_head_hidden', 256) or 256),
        value_num_bins=int(cfg.get('value_num_bins', 100) or 100),
        value_target_min=float(cfg.get('value_target_min', -6.0)),
        value_target_max=float(cfg.get('value_target_max', 6.0)),
        value_sigma_to_bin_ratio=float(
            cfg.get('value_sigma_to_bin_ratio', 2.0)
        ),
        value_padding_sigma=float(cfg.get('value_padding_sigma', 3.0)),
    )
    logging.info('oracle_brain params: %s', f'{parameter_count(oracle_brain):,}')
    logging.info('value_net params: %s', f'{parameter_count(value_net):,}')
    val_batches = int(cfg.get('val_batches', 256) or 0)
    test_batches = int(cfg.get('test_batches', val_batches) or 0)
    eval_log_every_batches = int(cfg.get('eval_log_every_batches', 32) or 0)
    if args.eval_only:
        eval_checkpoint = resolve_cli_path(args.eval_checkpoint) or state_file
        if args.eval_split == 'test':
            require_finalist_decision(args.finalist_decision, [eval_checkpoint])
        if not path.exists(eval_checkpoint):
            raise FileNotFoundError(f'oracle critic eval checkpoint does not exist: {eval_checkpoint}')
        state = torch.load(eval_checkpoint, weights_only=False, map_location=device)
        oracle_brain.load_state_dict(state['oracle_brain'])
        value_net.load_state_dict(state['value_net'])
        steps = int(state.get('steps', 0))
        eval_files = val_files if args.eval_split == 'dev' else test_files
        if not eval_files:
            raise ValueError(
                f'oracle critic eval split {args.eval_split!r} is empty; '
                'configure test_ratio/min_test_files before evaluating test'
            )
        eval_batches = val_batches if args.eval_split == 'dev' else test_batches
        val_loader = make_loader(make_dataset(eval_files, cfg, train=False), cfg, train=False)
        eval_input_modes = normalize_eval_input_modes(
            args.eval_input_modes
            if args.eval_input_modes is not None
            else cfg.get('eval_input_modes')
        )
        val_by_input = evaluate_modes(
            oracle_brain,
            value_net,
            val_loader,
            device,
            enable_amp=eval_enable_amp,
            max_batches=eval_batches,
            input_modes=eval_input_modes,
            log_every_batches=eval_log_every_batches,
            label=f'{args.eval_split}/{"+".join(eval_input_modes)}',
        )
        val_metrics = val_by_input.get('true', next(iter(val_by_input.values())))
        result = {
            'checkpoint': eval_checkpoint,
            'steps': steps,
            'split': args.eval_split,
            'split_info': split_info[args.eval_split],
            'val': val_metrics,
            'val_by_input': val_by_input,
            'oracle_dependency': summarize_oracle_dependency(val_by_input),
        }
        for mode, mode_metrics in val_by_input.items():
            logging.info(
                (
                    'eval-only checkpoint=%s step=%s input=%s val_loss=%.6f '
                    'val_mae=%.6f val_corr=%.4f val_ev=%.4f val_zero_sum_mae=%.6f '
                    'samples=%s games=%s loss_ci95=[%.6f,%.6f]'
                ),
                display_path(eval_checkpoint),
                steps,
                mode,
                mode_metrics['loss'],
                mode_metrics['mae'],
                mode_metrics['corr'],
                mode_metrics['explained_variance'],
                mode_metrics['zero_sum_mae'],
                mode_metrics['num_samples'],
                mode_metrics.get('num_games', 0),
                mode_metrics.get('loss_ci95_low', mode_metrics['loss']),
                mode_metrics.get('loss_ci95_high', mode_metrics['loss']),
            )
        for comparison, dependency_metrics in result['oracle_dependency'].items():
            logging.info(
                (
                    'eval-only checkpoint=%s step=%s dependency=%s '
                    'loss_improvement=%.6f relative=%.4f corr_gain=%.4f ev_gain=%.4f'
                ),
                display_path(eval_checkpoint),
                steps,
                comparison,
                dependency_metrics['loss_improvement'],
                dependency_metrics['relative_loss_improvement'],
                dependency_metrics['corr_gain'],
                dependency_metrics['explained_variance_gain'],
            )
        print(json.dumps(result, sort_keys=True))
        return

    train_scope = normalize_train_scope(cfg.get('train_scope', 'oracle_input_value'))
    train_info = configure_trainable_parameters(
        oracle_brain,
        value_net,
        scope=train_scope,
        version=config['control']['version'],
        tail_blocks=int(cfg.get('tail_blocks', 8) or 0),
    )
    logging.info(
        (
            'oracle critic arch=%s train scope=%s tail_blocks=%s '
            'encoder_lr_scale=%.4g visible_lr_scale=%s oracle_lr_scale=%s '
            'fusion_lr_scale=%s value_lr_scale=%.4g weight_decay=%.4g '
            'exact_zero_sum=%s trainable_params=%s'
        ),
        critic_arch,
        train_info['scope'],
        train_info.get('tail_blocks', 0),
        float_config_value(cfg, 'encoder_lr_scale', 1.0),
        cfg.get('visible_lr_scale', 'shared'),
        cfg.get('oracle_lr_scale', 'shared'),
        cfg.get('fusion_lr_scale', 'shared'),
        float_config_value(cfg, 'value_lr_scale', 1.0),
        float(cfg.get('weight_decay', config['optim'].get('weight_decay', 0.0)) or 0.0),
        exact_zero_sum,
        f"{train_info['trainable_params']:,}",
    )

    scheduler_cfg = dict(config['optim'].get('scheduler', {}))
    scheduler_cfg.update(cfg.get('scheduler', {}) if isinstance(cfg.get('scheduler', {}), dict) else {})
    scheduler_type = normalize_scheduler_type(scheduler_cfg.get('type', 'cosine'))
    max_steps = int(cfg.get('max_steps', scheduler_cfg.get('max_steps', 100000)) or 100000)
    scheduler_horizon_steps = int(cfg.get('scheduler_horizon_steps', max_steps) or max_steps)
    convergence_config = resolve_convergence_config(cfg)
    adaptive_config = resolve_adaptive_curriculum_config(cfg)
    if current_policy_data(cfg) is not None:
        for name in ('max_steps', 'val_every_steps', 'save_every'):
            if name not in cfg or int(cfg[name]) <= 0:
                raise ValueError('current-policy training needs an explicit positive ' + name)
        if adaptive_config is None or adaptive_config.selection_protocol != 'primary_with_diagnostics':
            raise ValueError('current-policy calibration requires primary_with_diagnostics selection')
    primary_with_diagnostics = (adaptive_config is not None and
                                adaptive_config.selection_protocol == 'primary_with_diagnostics')
    if primary_with_diagnostics:
        selection_paths = [path.normcase(path.realpath(item)) for item in
                           (state_file, best_state_file, best_primary_state_file,
                            adaptive_best_state_file, best_observed_primary_state_file)]
        if len(set(selection_paths)) != len(selection_paths):
            raise ValueError('selection checkpoint roles require distinct artifact paths')
    if convergence_config is not None and adaptive_config is not None:
        raise ValueError(
            'oracle critic convergence and adaptive_curriculum controllers '
            'cannot be enabled together'
        )
    optimizer_cfg = resolved_optimizer_config(cfg, scheduler_cfg)
    validate_convergence_schedule(
        convergence_config,
        scheduler_cfg=scheduler_cfg,
        optimizer_cfg=optimizer_cfg,
        scheduler_horizon_steps=scheduler_horizon_steps,
        max_steps=max_steps,
    )
    if scheduler_type == 'cosine':
        scheduler_cfg['max_steps'] = max(
            scheduler_horizon_steps,
            int(scheduler_cfg.get('warm_up_steps', 0) or 0),
        )
    optimizer = build_optimizer(
        optimizer_param_groups(
            oracle_brain,
            value_net,
            scope=train_scope,
            encoder_lr_scale=float_config_value(cfg, 'encoder_lr_scale', 1.0),
            visible_lr_scale=cfg.get('visible_lr_scale'),
            oracle_lr_scale=cfg.get('oracle_lr_scale'),
            fusion_lr_scale=cfg.get('fusion_lr_scale'),
            value_lr_scale=float_config_value(cfg, 'value_lr_scale', 1.0),
            weight_decay=cfg.get('weight_decay'),
        ),
        cfg,
        scheduler_cfg,
    )
    training_contract_info = training_contract(
        cfg,
        scheduler_cfg,
        convergence_config,
        adaptive_config,
    )
    import libriichi
    policy_data = current_policy_data(cfg)
    validation_contract = validation_input_contract(
        val_files,
        native_file=native_module_file(libriichi),
        verified_file_sha256=None if policy_data is None else policy_data['file_sha256'],
        settings={
            'oracle_imputation_version': ORACLE_IMPUTATION_VERSION,
            'imputation_seed': int(cfg.get('val_oracle_imputation_seed', 20260905)),
            'target_clock_version': ORACLE_TARGET_CLOCK_VERSION,
            'target': {key: training_contract_info[key] for key in (
                'return_mode', 'discount_gamma', 'centered_rank_points', 'target_mode',
            )},
            **({'current_policy': training_contract_info['current_policy']}
               if 'current_policy' in training_contract_info else {}),
            'validation_stream': {
                key: value for key, value in cfg.items()
                if key.startswith('val_') or key in {
                    'max_val_files', 'batch_size', 'state_fold_backend', 'state_fold_seed',
                }
            },
        },
    )
    training_contract_info['validation_input_fingerprint'] = validation_contract['fingerprint']
    # Save the large file ledger once; every checkpoint carries its fingerprint.
    validation_manifest = path.join(path.dirname(state_file), 'validation_inputs.json')
    ensure_parent_dir_for_file(validation_manifest)
    if path.exists(validation_manifest):
        with open(validation_manifest, encoding='utf-8') as stream:
            existing_validation_contract = json.load(stream)
        if existing_validation_contract['fingerprint'] != validation_contract['fingerprint']:
            raise ValueError('validation input contract changed; use a new run and rebase all best metrics')
    else:
        with open(validation_manifest, 'x', encoding='utf-8') as stream:
            json.dump(validation_contract, stream, indent=2, sort_keys=True)
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg)
    scaler = GradScaler(
        device.type,
        enabled=enable_amp,
        init_scale=float(cfg.get('amp_init_scale', 65536.0) or 65536.0),
        growth_interval=int(cfg.get('amp_growth_interval', 2000) or 2000),
    )
    logging.info(
        'oracle critic AMP enabled=%s init_scale=%.6g growth_interval=%s',
        enable_amp,
        scaler.get_scale(),
        int(cfg.get('amp_growth_interval', 2000) or 2000),
    )
    logging.info(
        'oracle critic optimizer=%s scheduler=%s contract=%s',
        resolved_optimizer_config(cfg, scheduler_cfg)['type'],
        scheduler_type,
        scheduler_contract(scheduler_cfg),
    )
    if convergence_config is not None:
        logging.info(
            (
                'oracle critic convergence metric=%s core_steps=%s '
                'tail_lrs=%s smoothing_checks=%s improvement_delta=%.6g '
                'reduce_patience_steps=%s stop_patience_steps=%s '
                'min_level_steps=%s hard_cap=%s'
            ),
            convergence_config.metric,
            convergence_config.core_optimizer_steps,
            convergence_config.tail_lr_levels,
            convergence_config.smoothing_checks,
            convergence_config.improvement_delta,
            convergence_config.reduce_patience_steps,
            convergence_config.stop_patience_steps,
            convergence_config.min_level_steps,
            max_steps,
        )
    if adaptive_config is not None:
        if val_every_steps := int(cfg.get('val_every_steps', 2000) or 2000):
            if adaptive_config.gate_every_steps % val_every_steps != 0:
                raise ValueError(
                    'adaptive gate_every_steps must be divisible by '
                    'val_every_steps'
                )
        if adaptive_config.final_phase:
            if scheduler_type != 'optimizer':
                raise ValueError(
                    'final Oracle adaptive phase requires optimizer-owned LR control'
                )
            if optimizer_cfg.get('type') != 'schedule_free_adamw':
                raise ValueError(
                    'final Oracle adaptive phase requires Schedule-Free AdamW'
                )
            optimizer_lr = float(optimizer_cfg['lr'])
            if adaptive_config.lr_levels[0] > optimizer_lr:
                raise ValueError(
                    'first adaptive lr level must not exceed optimizer.lr'
                )
        logging.info(
            (
                'oracle critic adaptive curriculum phase=%s final=%s '
                'gate_every=%s required_futile=%s primary=%s delta=%.6g '
                'guardrails=%s lr_levels=%s hard_cap=%s'
            ),
            adaptive_config.phase_name,
            adaptive_config.final_phase,
            adaptive_config.gate_every_steps,
            adaptive_config.required_futile_gates,
            adaptive_config.primary.name,
            adaptive_config.primary.meaningful_delta,
            [spec.name for spec in adaptive_config.guardrails],
            adaptive_config.lr_levels,
            max_steps,
        )
    teacher_brain, teacher_value_net = load_teacher_models(
        str(cfg.get('teacher_state_file', '') or '').strip(),
        device,
    )
    teacher_loss_weight = float(cfg.get('teacher_loss_weight', 0.0) or 0.0)
    raw_target_loss_weight = cfg.get('target_loss_weight', 1.0)
    target_loss_weight = 1.0 if raw_target_loss_weight is None else float(raw_target_loss_weight)
    initial_output_weights = target_output_weights_at_step(cfg, 0)
    final_output_weights = normalize_target_output_weights(
        cfg.get('target_output_weights')
    )
    scheduled_output_weights = 'target_output_weights_initial' in cfg
    output_weight_ramp_start = int(
        cfg.get('target_output_weight_ramp_start_steps', 0) or 0
    )
    output_weight_ramp_end = int(
        cfg.get(
            'target_output_weight_ramp_end_steps',
            output_weight_ramp_start,
        ) or 0
    )
    initial_output_weight_tensor = (
        torch.tensor(initial_output_weights, dtype=torch.float32, device=device)
        if initial_output_weights is not None
        else None
    )
    final_output_weight_tensor = (
        torch.tensor(final_output_weights, dtype=torch.float32, device=device)
        if final_output_weights is not None
        else None
    )
    if initial_output_weights is not None:
        logging.info(
            'oracle critic target output weights at step 0=%s final=%s ramp=[%s,%s]',
            tuple(round(weight, 6) for weight in initial_output_weights),
            tuple(
                round(weight, 6)
                for weight in final_output_weights
            ),
            output_weight_ramp_start,
            output_weight_ramp_end,
        )
    if teacher_brain is not None:
        logging.info(
            'oracle critic teacher distill enabled: source=%s teacher_weight=%.4g target_weight=%.4g',
            display_path(str(cfg.get('teacher_state_file'))),
            teacher_loss_weight,
            target_loss_weight,
        )

    steps = 0
    best_val_loss = math.inf
    best_primary_loss = math.inf
    best_observed_primary_loss = math.inf if primary_with_diagnostics else None
    init_info = {'source': '', 'loaded': False}
    stream_signature = data_stream_signature(cfg)
    data_progress = initial_data_progress(stream_signature)
    convergence_state = (
        initial_convergence_state(convergence_config)
        if convergence_config is not None
        else None
    )
    adaptive_curriculum_state = (
        initial_adaptive_curriculum_state(adaptive_config)
        if adaptive_config is not None
        else None
    )

    save_every = int(cfg.get('save_every', 1000) or 1000)
    log_every = int(cfg.get('log_every', 100) or 100)
    val_every_steps = int(cfg.get('val_every_steps', 2000) or 2000)
    dependency_val_every_steps = int(cfg.get('dependency_val_every_steps', 0) or 0)
    eval_input_modes = normalize_eval_input_modes(cfg.get('eval_input_modes'))
    val_game_id_modulus, val_game_id_remainders = normalize_game_id_subset(
        cfg.get('val_game_id_modulus', 1),
        cfg.get('val_game_id_remainders', ()),
    )
    logging.info(
        'oracle critic in-training validation game_id_modulus=%s remainders=%s',
        val_game_id_modulus,
        val_game_id_remainders,
    )
    max_grad_norm = float(cfg.get('max_grad_norm', config['optim'].get('max_grad_norm', 0.0)) or 0.0)
    zero_sum_weight = float(cfg.get('zero_sum_weight', config.get('value', {}).get('zero_sum_weight', 0.0)) or 0.0)

    if path.exists(state_file) and not args.fresh:
        state = torch.load(state_file, weights_only=False, map_location=device)
        validate_resume_file_splits(state, split_info)
        migrated_convergence_contract = validate_resume_training_contract(
            state,
            training_contract_info,
            convergence_config=convergence_config,
        )
        resumed_data_progress = validate_resume_data_stream(state, stream_signature)
        if resumed_data_progress is not None:
            data_progress = resumed_data_progress
        oracle_brain.load_state_dict(state['oracle_brain'])
        value_net.load_state_dict(state['value_net'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        steps = int(state.get('steps', 0))
        best_val_loss = float(state.get('best_val_loss', math.inf))
        best_primary_loss = float(state.get('best_primary_loss', math.inf))
        if primary_with_diagnostics:
            best_val_loss = maybe_load_protocol_best_loss(
                best_state_file, best_val_loss, 'best_val_loss', training_contract_info, split_info)
            best_primary_loss = maybe_load_protocol_best_loss(
                best_primary_state_file, best_primary_loss, 'best_primary_loss',
                training_contract_info, split_info)
            best_observed_primary_loss = maybe_load_protocol_best_loss(
                best_observed_primary_state_file,
                float(state.get('best_observed_primary_loss', math.inf)),
                'best_observed_primary_loss', training_contract_info, split_info)
        else:
            best_val_loss = maybe_load_best_val_loss(best_state_file, best_val_loss, device)
            best_primary_loss = maybe_load_best_primary_loss(
                best_primary_state_file, best_primary_loss, device)
        init_info = state.get('init_info', init_info)
        if convergence_config is not None:
            convergence_state = normalize_convergence_state(
                None if migrated_convergence_contract else state.get('convergence_state'),
                convergence_config,
            )
            if migrated_convergence_contract:
                logging.info(
                    'adopted convergence contract for pre-core checkpoint at step=%s',
                    steps,
                )
            if convergence_state['tail_started']:
                scheduler.set_tail_lr(
                    convergence_config.tail_lr_levels[
                        convergence_state['level_index']
                    ]
                )
        if adaptive_config is not None:
            adaptive_curriculum_state = normalize_adaptive_curriculum_state(
                state.get('adaptive_curriculum_state'),
                adaptive_config,
            )
            if adaptive_config.final_phase:
                scheduler.set_tail_lr(
                    adaptive_config.lr_levels[
                        adaptive_curriculum_state['lr_level_index']
                    ]
                )
        logging.info('resumed oracle critic pretrain from %s at step=%s', state_file, steps)
    else:
        init_state_file = resolve_init_state_file(cfg)
        init_info = maybe_init_from_checkpoint(
            oracle_brain,
            value_net,
            init_state_file,
            device,
            strict_oracle_checkpoint=bool(cfg.get('strict_init_checkpoint', False)),
            oracle_input_init_scale=float(cfg.get('oracle_input_init_scale', 0.02)),
            oracle_tower_init=cfg.get(
                'oracle_tower_init',
                'visible_transfer',
            ),
            oracle_first_conv_init=cfg.get(
                'oracle_first_conv_init',
                'legacy_mean_scaled',
            ),
            oracle_hand_init_scale=float(
                cfg.get('oracle_hand_init_scale', 1.0 / 3.0)
            ),
        )
        logging.info('oracle critic init: %s', summarize_init_info(init_info))

    if adaptive_config is not None and adaptive_curriculum_state.get('best_step') is None:
        # A warm start is a candidate baseline, not 50k unmeasured updates.
        optimizer_eval_mode(optimizer)
        baseline_files = evaluation_files(
            val_files, cfg, game_id_modulus=val_game_id_modulus,
            game_id_remainders=val_game_id_remainders, max_batches=val_batches,
        )
        baseline_loader = make_loader(make_dataset(baseline_files, cfg, train=False), cfg, train=False)
        baseline_metrics = evaluate(
            oracle_brain, value_net, baseline_loader, device,
            enable_amp=eval_enable_amp, max_batches=val_batches,
            game_id_modulus=val_game_id_modulus, game_id_remainders=val_game_id_remainders,
            include_cluster_records=True,
        )
        baseline_primary = baseline_metrics['outputs']['relative_player_0']
        baseline_values = {
            'primary_loss': float(baseline_primary['loss']),
            'all_players_loss': float(baseline_metrics['loss']),
            'p0_mae': float(baseline_primary['mae']),
            **{f'{name}_loss': float(item['loss']) for name, item in baseline_primary['slices'].items()},
        }
        baseline_decision = observe_adaptive_curriculum(
            adaptive_curriculum_state, adaptive_config, optimizer_steps=steps,
            metrics=baseline_values, cluster_records=baseline_metrics.pop('_adaptive_cluster_records'),
        )
        adaptive_curriculum_state = baseline_decision.state
        best_val_loss = float(baseline_metrics['loss'])
        best_primary_loss = float(baseline_primary['loss'])
        if primary_with_diagnostics:
            best_observed_primary_loss = best_primary_loss
        baseline_payload = checkpoint_payload(
            oracle_brain=oracle_brain, value_net=value_net, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, steps=steps, best_val_loss=best_val_loss,
            best_primary_loss=best_primary_loss,
            best_observed_primary_loss=best_observed_primary_loss, init_info=init_info, cfg=cfg,
            train_info=train_info, split_info=split_info, data_progress=data_progress,
            training_contract_info=training_contract_info, convergence_state=convergence_state,
            adaptive_curriculum_state=adaptive_curriculum_state,
        )
        baseline_payload['baseline_val_metrics'] = baseline_metrics
        baseline_paths = {'latest': state_file, 'adaptive_best': adaptive_best_state_file,
                          'best': best_state_file, 'best_primary': best_primary_state_file,
                          'best_observed_primary': best_observed_primary_state_file}
        for role in baseline_checkpoint_roles(primary_with_diagnostics):
            save_checkpoint(baseline_paths[role], baseline_payload)
        logging.info('saved fixed-input no-update anchor at step=%s primary_loss=%.6f', steps, best_primary_loss)
        del baseline_payload, baseline_loader

    if adaptive_config is not None and adaptive_config.final_phase:
        scheduler.set_tail_lr(
            adaptive_config.lr_levels[
                int(adaptive_curriculum_state['lr_level_index'])
            ]
        )

    oracle_brain.train()
    value_net.train()
    optimizer_train_mode(optimizer)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(tensorboard_dir)

    stop_training = bool(
        convergence_state is not None and convergence_state.get('converged', False)
    )
    if (
        adaptive_curriculum_state is not None
        and adaptive_curriculum_state.get('completed', False)
    ):
        stop_training = True
    if stop_training:
        logging.info(
            'oracle critic checkpoint is already complete at step=%s; skipping training',
            steps,
        )
    train_loader = (
        make_loader(
            make_dataset(train_files, cfg, train=True, stream_state=data_progress),
            cfg,
            train=True,
        )
        if steps < max_steps and not stop_training
        else None
    )
    training_eval_cfg = in_training_eval_config(cfg)
    logging.info(
        'oracle critic in-training eval workers=%s',
        training_eval_cfg['val_num_workers'],
    )
    stats = []
    skipped_optimizer_steps = 0
    paused_for_external_request = False
    show_progress = bool(cfg.get('progress_bar', sys.stderr.isatty()))
    pb = tqdm(total=max(save_every, 1), desc='ORACLE', disable=not show_progress)
    while steps < max_steps and not stop_training and not paused_for_external_request:
        restart_same_cycle = False
        train_iterator = iter(train_loader)
        for batch in train_iterator:
            if steps >= max_steps:
                break
            if len(batch) == 5:
                obs, invisible_obs, target, _player_id, stream_progress = batch
            else:
                obs, invisible_obs, target, _player_id = batch
                stream_progress = None
            update_data_progress(data_progress, stream_progress)
            steps += 1
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
            target = target.to(dtype=torch.float32, device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if initial_output_weight_tensor is None:
                target_output_weights = None
            elif not scheduled_output_weights or steps >= output_weight_ramp_end:
                target_output_weights = final_output_weight_tensor
            elif steps < output_weight_ramp_start:
                target_output_weights = initial_output_weight_tensor
            elif output_weight_ramp_end == output_weight_ramp_start:
                target_output_weights = final_output_weight_tensor
            else:
                ramp_ratio = (
                    (steps - output_weight_ramp_start)
                    / float(output_weight_ramp_end - output_weight_ramp_start)
                )
                target_output_weights = torch.lerp(
                    initial_output_weight_tensor,
                    final_output_weight_tensor,
                    ramp_ratio,
                )
            pred, value_logits = model_forward(
                oracle_brain,
                value_net,
                obs,
                invisible_obs,
                enable_amp=enable_amp,
                device_type=device.type,
                return_logits=True,
            )
            teacher_target = None
            if teacher_brain is not None:
                with torch.inference_mode():
                    teacher_target = model_forward(
                        teacher_brain,
                        teacher_value_net,
                        obs,
                        invisible_obs,
                        enable_amp=enable_amp,
                        device_type=device.type,
                    ).detach()
            with torch.autocast(device.type, enabled=enable_amp):
                target_mse = output_weighted_mse(
                    pred,
                    target,
                    target_output_weights,
                )
                target_objective = (
                    value_net.cross_entropy(
                        value_logits,
                        target,
                        target_output_weights,
                    )
                    if isinstance(value_net, HLGaussValueHead)
                    else target_mse
                )
                teacher_mse = (
                    output_weighted_mse(
                        pred,
                        teacher_target,
                        target_output_weights,
                    )
                    if teacher_target is not None
                    else None
                )
                zero_sum_loss = pred.sum(dim=-1).square().mean()
                loss = pred.new_tensor(0.0)
                if target_loss_weight > 0:
                    loss = loss + target_loss_weight * target_objective
                if teacher_mse is not None and teacher_loss_weight > 0:
                    loss = loss + teacher_loss_weight * teacher_mse
                if zero_sum_weight > 0:
                    loss = loss + zero_sum_weight * zero_sum_loss

            scaler.scale(loss).backward()
            if train_info['first_conv_mask'] is not None:
                first_conv_grad = first_conv_module(oracle_brain).weight.grad
                if first_conv_grad is not None:
                    first_conv_grad.mul_(train_info['first_conv_mask'])
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    tuple(param for param in oracle_brain.parameters() if param.requires_grad)
                    + tuple(param for param in value_net.parameters() if param.requires_grad),
                    max_grad_norm,
                )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not grad_scaler_step_succeeded(scale_before, scaler.get_scale()):
                steps -= 1
                skipped_optimizer_steps += 1
                if skipped_optimizer_steps <= 3 or skipped_optimizer_steps & (skipped_optimizer_steps - 1) == 0:
                    logging.warning(
                        'skipped non-finite optimizer update count=%s amp_scale=%.6g',
                        skipped_optimizer_steps,
                        scaler.get_scale(),
                    )
                continue
            scheduler.step()

            stats.append(
                batch_metrics(
                    pred,
                    target,
                    objective_loss=loss,
                    target_objective=target_objective,
                    target_mse=target_mse,
                    teacher_mse=teacher_mse,
                    zero_sum_loss=zero_sum_loss,
                )
            )
            pb.update(1)

            if steps % log_every == 0:
                metrics = finalize_metrics(stats)
                group_lrs = component_lrs(optimizer, scheduler)
                logging.info(
                    (
                        'step=%s train_loss=%.6f primary_loss=%.6f '
                        'objective_loss=%.6f target_objective=%.6f train_mae=%.6f '
                        'corr=%.4f zero_sum_mae=%.6f visible_lr=%.3g oracle_lr=%.3g '
                        'fusion_lr=%.3g value_lr=%.3g p0_weight=%.4f'
                    ),
                    steps,
                    metrics['loss'],
                    metrics['outputs']['relative_player_0']['loss'],
                    metrics.get('objective_loss', metrics['loss']),
                    metrics.get('target_objective', metrics['loss']),
                    metrics['mae'],
                    metrics['corr'],
                    metrics['zero_sum_mae'],
                    group_lrs['visible'],
                    group_lrs['oracle'],
                    group_lrs['fusion'],
                    group_lrs['value'],
                    target_output_weights[0] if target_output_weights is not None else 1.0,
                )
                if 'teacher_mse' in metrics:
                    logging.info(
                        'step=%s distill_target_mse=%.6f teacher_mse=%.6f zero_sum_loss=%.6f',
                        steps,
                        metrics.get('target_mse', metrics['loss']),
                        metrics['teacher_mse'],
                        metrics.get('zero_sum_loss', 0.0),
                    )
                writer.add_scalar('train/loss', metrics['loss'], steps)
                writer.add_scalar(
                    'train/primary_loss',
                    metrics['outputs']['relative_player_0']['loss'],
                    steps,
                )
                for output_name, output_metrics in metrics['outputs'].items():
                    writer.add_scalar(
                        f'train_outputs/{output_name}_loss',
                        output_metrics['loss'],
                        steps,
                    )
                for slice_name, slice_metrics in metrics['outputs'][
                    'relative_player_0'
                ]['slices'].items():
                    writer.add_scalar(
                        f'train_primary_slices/{slice_name}_loss',
                        slice_metrics['loss'],
                        steps,
                    )
                writer.add_scalar('train/objective_loss', metrics.get('objective_loss', metrics['loss']), steps)
                writer.add_scalar(
                    'train/target_objective',
                    metrics.get('target_objective', metrics['loss']),
                    steps,
                )
                writer.add_scalar('train/target_mse', metrics.get('target_mse', metrics['loss']), steps)
                writer.add_scalar(
                    'train/p0_output_weight',
                    target_output_weights[0] if target_output_weights is not None else 1.0,
                    steps,
                )
                if 'teacher_mse' in metrics:
                    writer.add_scalar('train/teacher_mse', metrics['teacher_mse'], steps)
                writer.add_scalar('train/zero_sum_loss', metrics.get('zero_sum_loss', 0.0), steps)
                writer.add_scalar('train/mae', metrics['mae'], steps)
                writer.add_scalar('train/corr', metrics['corr'], steps)
                writer.add_scalar('train/zero_sum_mae', metrics['zero_sum_mae'], steps)
                for component, component_lr in group_lrs.items():
                    writer.add_scalar(f'lr/{component}', component_lr, steps)
                stats.clear()

            pause_checkpoint_due = external_pause_due(
                external_pause_file,
                steps,
                max_steps,
            )
            validation_due = steps % val_every_steps == 0 or steps >= max_steps
            checkpoint_due = steps % save_every == 0 or steps >= max_steps
            convergence_observed = False
            adaptive_observed = False
            scheduler_observed = False
            released_for_eval = False
            if validation_due or checkpoint_due or pause_checkpoint_due:
                optimizer_eval_mode(optimizer)
            if validation_due:
                adaptive_gate_due = bool(
                    adaptive_config is not None
                    and steps % adaptive_config.gate_every_steps == 0
                )
                if checkpoint_due:
                    payload = checkpoint_payload(
                        oracle_brain=oracle_brain,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_val_loss=best_val_loss,
                        best_primary_loss=best_primary_loss,
                        best_observed_primary_loss=best_observed_primary_loss,
                        init_info=init_info,
                        cfg=cfg,
                        train_info=train_info,
                        split_info=split_info,
                        data_progress=data_progress,
                        training_contract_info=training_contract_info,
                        convergence_state=convergence_state,
                        adaptive_curriculum_state=adaptive_curriculum_state,
                    )
                    save_checkpoint(state_file, payload)
                    logging.info(
                        'saved pre-eval oracle critic checkpoint to %s',
                        state_file,
                    )
                if (
                    bool(cfg.get('release_train_loader_for_eval', True))
                    and train_loader.num_workers > 0
                ):
                    logging.info(
                        'releasing %s train DataLoader workers before evaluation',
                        train_loader.num_workers,
                    )
                    released_for_eval = shutdown_data_loader_iterator(
                        train_loader,
                        train_iterator,
                    )
                    gc.collect()
                run_dependency_val = (
                    steps >= max_steps
                    or (
                        dependency_val_every_steps > 0
                        and steps % dependency_val_every_steps == 0
                    )
                )
                selected_val_files = evaluation_files(
                    val_files, training_eval_cfg,
                    game_id_modulus=val_game_id_modulus,
                    game_id_remainders=val_game_id_remainders,
                    max_batches=val_batches,
                    input_modes=eval_input_modes if run_dependency_val else ('true',),
                )
                val_loader = make_loader(
                    make_dataset(selected_val_files, training_eval_cfg, train=False),
                    training_eval_cfg,
                    train=False,
                )
                if run_dependency_val:
                    val_by_input = evaluate_modes(
                        oracle_brain,
                        value_net,
                        val_loader,
                        device,
                        enable_amp=eval_enable_amp,
                        max_batches=val_batches,
                        input_modes=eval_input_modes,
                        log_every_batches=eval_log_every_batches,
                        label=f'dev@{steps}/{"+".join(eval_input_modes)}',
                        game_id_modulus=val_game_id_modulus,
                        game_id_remainders=val_game_id_remainders,
                        include_cluster_records=adaptive_gate_due or primary_with_diagnostics,
                    )
                    val_metrics = val_by_input.get('true', next(iter(val_by_input.values())))
                    oracle_dependency = summarize_oracle_dependency(val_by_input)
                    for mode, mode_metrics in val_by_input.items():
                        logging.info(
                            (
                                'step=%s input=%s val_loss=%.6f val_mae=%.6f '
                                'val_corr=%.4f val_ev=%.4f val_zero_sum_mae=%.6f '
                                'samples=%s games=%s loss_ci95=[%.6f,%.6f]'
                            ),
                            steps,
                            mode,
                            mode_metrics['loss'],
                            mode_metrics['mae'],
                            mode_metrics['corr'],
                            mode_metrics['explained_variance'],
                            mode_metrics['zero_sum_mae'],
                            mode_metrics['num_samples'],
                            mode_metrics.get('num_games', 0),
                            mode_metrics.get('loss_ci95_low', mode_metrics['loss']),
                            mode_metrics.get('loss_ci95_high', mode_metrics['loss']),
                        )
                    for comparison, dependency_metrics in oracle_dependency.items():
                        logging.info(
                            (
                                'step=%s dependency=%s loss_improvement=%.6f '
                                'relative=%.4f corr_gain=%.4f ev_gain=%.4f'
                            ),
                            steps,
                            comparison,
                            dependency_metrics['loss_improvement'],
                            dependency_metrics['relative_loss_improvement'],
                            dependency_metrics['corr_gain'],
                            dependency_metrics['explained_variance_gain'],
                        )
                else:
                    val_by_input = None
                    oracle_dependency = None
                    val_metrics = evaluate(
                        oracle_brain,
                        value_net,
                        val_loader,
                        device,
                        enable_amp=eval_enable_amp,
                        max_batches=val_batches,
                        log_every_batches=eval_log_every_batches,
                        label=f'dev@{steps}/true',
                        game_id_modulus=val_game_id_modulus,
                        game_id_remainders=val_game_id_remainders,
                        include_cluster_records=adaptive_gate_due or primary_with_diagnostics,
                    )
                    logging.info(
                        (
                            'step=%s val_loss=%.6f val_mae=%.6f val_corr=%.4f '
                            'val_zero_sum_mae=%.6f samples=%s games=%s '
                            'loss_ci95=[%.6f,%.6f]'
                        ),
                        steps,
                        val_metrics['loss'],
                        val_metrics['mae'],
                        val_metrics['corr'],
                        val_metrics['zero_sum_mae'],
                        val_metrics['num_samples'],
                        val_metrics.get('num_games', 0),
                        val_metrics.get('loss_ci95_low', val_metrics['loss']),
                        val_metrics.get('loss_ci95_high', val_metrics['loss']),
                    )
                writer.add_scalar('val/loss', val_metrics['loss'], steps)
                primary_val_loss = float(
                    val_metrics['outputs']['relative_player_0']['loss']
                )
                writer.add_scalar('val/primary_loss', primary_val_loss, steps)
                writer.add_scalar('val/mae', val_metrics['mae'], steps)
                writer.add_scalar('val/corr', val_metrics['corr'], steps)
                writer.add_scalar('val/zero_sum_mae', val_metrics['zero_sum_mae'], steps)
                primary_slices = val_metrics['outputs']['relative_player_0']['slices']
                adaptive_cluster_records = val_metrics.pop(
                    '_adaptive_cluster_records',
                    None,
                )
                if val_by_input is not None:
                    for mode_metrics in val_by_input.values():
                        if mode_metrics is not val_metrics:
                            mode_metrics.pop('_adaptive_cluster_records', None)
                logging.info(
                    (
                        'step=%s p0_slices exact_zero=%.6f nonzero=%.6f '
                        'abs_ge_2=%.6f abs_ge_4=%.6f'
                    ),
                    steps,
                    primary_slices.get('exact_zero', {}).get('loss', math.nan),
                    primary_slices.get('nonzero', {}).get('loss', math.nan),
                    primary_slices.get('abs_ge_2', {}).get('loss', math.nan),
                    primary_slices.get('abs_ge_4', {}).get('loss', math.nan),
                )
                for slice_name, slice_metrics in primary_slices.items():
                    writer.add_scalar(
                        f'val_primary_slices/{slice_name}_loss',
                        slice_metrics['loss'],
                        steps,
                    )
                scheduler_decision = None
                observe_scheduler = getattr(scheduler, 'observe', None)
                if callable(observe_scheduler):
                    scheduler_metric = str(
                        scheduler_cfg.get('metric', 'primary_loss')
                    )
                    scheduler_metric_values = {
                        'loss': float(val_metrics['loss']),
                        'primary_loss': primary_val_loss,
                    }
                    if scheduler_metric not in scheduler_metric_values:
                        raise ValueError(
                            'plateau scheduler metric must be loss or primary_loss'
                        )
                    scheduler_decision = observe_scheduler(
                        scheduler_metric_values[scheduler_metric],
                        steps,
                    )
                    scheduler_observed = True
                    logging.info(
                        '[LR_MONITOR] step=%s metric=%s value=%.6f action=%s lr=%.6g',
                        steps,
                        scheduler_metric,
                        scheduler_metric_values[scheduler_metric],
                        scheduler_decision['action'],
                        scheduler_decision['lr'],
                    )
                    writer.add_scalar(
                        'lr_monitor/base_lr', scheduler_decision['lr'], steps
                    )
                if val_by_input is not None:
                    for mode, mode_metrics in val_by_input.items():
                        writer.add_scalar(f'val_by_input/{mode}_loss', mode_metrics['loss'], steps)
                        writer.add_scalar(f'val_by_input/{mode}_corr', mode_metrics['corr'], steps)
                        writer.add_scalar(
                            f'val_by_input/{mode}_explained_variance',
                            mode_metrics['explained_variance'],
                            steps,
                        )
                convergence_decision = None
                if convergence_config is not None:
                    tail_was_started = bool(convergence_state['tail_started'])
                    convergence_metric_values = {
                        'loss': float(val_metrics['loss']),
                        'primary_loss': primary_val_loss,
                    }
                    convergence_decision = observe_convergence(
                        convergence_state,
                        convergence_config,
                        optimizer_steps=steps,
                        metric_value=convergence_metric_values[
                            convergence_config.metric
                        ],
                    )
                    convergence_state = convergence_decision.state
                    convergence_observed = True
                    tail_just_started = (
                        not tail_was_started
                        and bool(convergence_state['tail_started'])
                    )
                    if tail_just_started or convergence_decision.action == 'reduce_lr':
                        previous_lrs = component_lrs(optimizer, scheduler)
                        scheduler.set_tail_lr(convergence_decision.target_lr)
                        logging.info(
                            '[CONVERGENCE] %s; component_lrs %s -> %s',
                            convergence_decision.reason,
                            previous_lrs,
                            component_lrs(optimizer, scheduler),
                        )
                    else:
                        logging.info(
                            '[CONVERGENCE] action=%s smoothed=%s %s',
                            convergence_decision.action,
                            (
                                'NA'
                                if convergence_decision.smoothed_metric is None
                                else f'{convergence_decision.smoothed_metric:.6f}'
                            ),
                            convergence_decision.reason,
                        )
                    writer.add_scalar(
                        'convergence/tail_level',
                        convergence_state['level_index'],
                        steps,
                    )
                    writer.add_scalar(
                        'convergence/tail_started',
                        int(convergence_state['tail_started']),
                        steps,
                    )
                    if convergence_decision.smoothed_metric is not None:
                        writer.add_scalar(
                            f'convergence/smoothed_{convergence_config.metric}',
                            convergence_decision.smoothed_metric,
                            steps,
                        )
                    if convergence_decision.action == 'stop':
                        stop_training = True
                adaptive_decision = None
                if adaptive_gate_due or primary_with_diagnostics:
                    if adaptive_cluster_records is None:
                        raise RuntimeError(
                            'adaptive Oracle gate requires per-game cluster records'
                        )
                    adaptive_metric_values = {
                        'primary_loss': primary_val_loss,
                        'all_players_loss': float(val_metrics['loss']),
                        'p0_mae': float(
                            val_metrics['outputs']['relative_player_0']['mae']
                        ),
                        **{
                            f'{slice_name}_loss': float(slice_metrics['loss'])
                            for slice_name, slice_metrics in primary_slices.items()
                        },
                    }
                    validate_adaptive_observation(
                        adaptive_config, adaptive_metric_values, adaptive_cluster_records,
                        adaptive_curriculum_state if primary_with_diagnostics else None,
                    )
                if adaptive_gate_due:
                    adaptive_decision = observe_adaptive_curriculum(
                        adaptive_curriculum_state,
                        adaptive_config,
                        optimizer_steps=steps,
                        metrics=adaptive_metric_values,
                        cluster_records=adaptive_cluster_records,
                    )
                    adaptive_curriculum_state = adaptive_decision.state
                    adaptive_observed = True
                    if adaptive_decision.action == 'reduce_lr':
                        previous_lrs = component_lrs(optimizer, scheduler)
                        scheduler.set_tail_lr(adaptive_decision.target_lr)
                        logging.info(
                            '[ADAPTIVE_CURRICULUM] %s; component_lrs %s -> %s',
                            adaptive_decision.reason,
                            previous_lrs,
                            component_lrs(optimizer, scheduler),
                        )
                    else:
                        logging.info(
                            '[ADAPTIVE_CURRICULUM] phase=%s gate=%s action=%s '
                            'futile=%s/%s %s',
                            adaptive_config.phase_name,
                            adaptive_curriculum_state['gate_index'],
                            adaptive_decision.action,
                            adaptive_curriculum_state['consecutive_futile_gates'],
                            adaptive_config.required_futile_gates,
                            adaptive_decision.reason,
                        )
                    writer.add_scalar(
                        'adaptive_curriculum/consecutive_futile_gates',
                        adaptive_curriculum_state['consecutive_futile_gates'],
                        steps,
                    )
                    writer.add_scalar(
                        'adaptive_curriculum/lr_level',
                        adaptive_curriculum_state['lr_level_index'],
                        steps,
                    )
                    if adaptive_decision.action in {'transition', 'stop', 'inconclusive'}:
                        stop_training = True
                with open(metrics_file, 'a', encoding='utf-8') as f:
                    payload = {'steps': steps, 'val': val_metrics}
                    if scheduler_decision is not None:
                        payload['lr_monitor'] = scheduler_decision
                    if val_by_input is not None:
                        payload['val_by_input'] = val_by_input
                        payload['oracle_dependency'] = oracle_dependency
                    if convergence_state is not None:
                        payload['convergence'] = copy.deepcopy(convergence_state)
                    if adaptive_curriculum_state is not None:
                        payload['adaptive_curriculum'] = copy.deepcopy(
                            adaptive_curriculum_state
                        )
                    f.write(json.dumps(payload, sort_keys=True) + '\n')
                selected = checkpoint_selection(
                    primary_loss=primary_val_loss, all_players_loss=float(val_metrics['loss']),
                    best_primary_loss=best_primary_loss, best_val_loss=best_val_loss,
                    best_observed_primary_loss=best_observed_primary_loss,
                    primary_with_diagnostics=primary_with_diagnostics,
                    adaptive_active=adaptive_config is not None,
                    gate_observed=adaptive_decision is not None,
                    accepted=adaptive_curriculum_state is not None and
                             adaptive_curriculum_state['best_step'] == steps,
                )
                improved_best_val = selected['best']
                improved_best_primary = selected['best_primary']
                if selected['best_observed_primary']:
                    best_observed_primary_loss = primary_val_loss
                if improved_best_val:
                    best_val_loss = val_metrics['loss']
                if improved_best_primary:
                    best_primary_loss = primary_val_loss
                if improved_best_val:
                    payload = checkpoint_payload(
                        oracle_brain=oracle_brain,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_val_loss=best_val_loss,
                        best_primary_loss=best_primary_loss,
                        best_observed_primary_loss=best_observed_primary_loss,
                        init_info=init_info,
                        cfg=cfg,
                        train_info=train_info,
                        split_info=split_info,
                        data_progress=data_progress,
                        training_contract_info=training_contract_info,
                        convergence_state=convergence_state,
                        adaptive_curriculum_state=adaptive_curriculum_state,
                    )
                    save_checkpoint(best_state_file, payload)
                    logging.info('saved best oracle critic checkpoint to %s', best_state_file)
                if improved_best_primary or selected['best_observed_primary']:
                    payload = checkpoint_payload(
                        oracle_brain=oracle_brain,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_val_loss=best_val_loss,
                        best_primary_loss=best_primary_loss,
                        best_observed_primary_loss=best_observed_primary_loss,
                        init_info=init_info,
                        cfg=cfg,
                        train_info=train_info,
                        split_info=split_info,
                        data_progress=data_progress,
                        training_contract_info=training_contract_info,
                        convergence_state=convergence_state,
                        adaptive_curriculum_state=adaptive_curriculum_state,
                    )
                    if selected['best_observed_primary']:
                        save_checkpoint(best_observed_primary_state_file, payload)
                        logging.info('saved observed-primary candidate to %s', best_observed_primary_state_file)
                    if improved_best_primary:
                        save_checkpoint(best_primary_state_file, payload)
                        logging.info(
                            'saved best-primary oracle critic checkpoint to %s',
                            best_primary_state_file,
                        )
                if (
                    adaptive_decision is not None
                    and adaptive_decision.action in {
                        'continue',
                        'update_best',
                        'reduce_lr',
                    }
                    and adaptive_curriculum_state['best_step'] == steps
                ):
                    payload = checkpoint_payload(
                        oracle_brain=oracle_brain,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_val_loss=best_val_loss,
                        best_primary_loss=best_primary_loss,
                        best_observed_primary_loss=best_observed_primary_loss,
                        init_info=init_info,
                        cfg=cfg,
                        train_info=train_info,
                        split_info=split_info,
                        data_progress=data_progress,
                        training_contract_info=training_contract_info,
                        convergence_state=convergence_state,
                        adaptive_curriculum_state=adaptive_curriculum_state,
                    )
                    save_checkpoint(adaptive_best_state_file, payload)
                    logging.info(
                        'saved adaptive phase-best oracle critic checkpoint to %s',
                        adaptive_best_state_file,
                    )

            if (
                checkpoint_due
                or convergence_observed
                or adaptive_observed
                or scheduler_observed
            ):
                payload = checkpoint_payload(
                    oracle_brain=oracle_brain,
                    value_net=value_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    steps=steps,
                    best_val_loss=best_val_loss,
                    best_primary_loss=best_primary_loss,
                    best_observed_primary_loss=best_observed_primary_loss,
                    init_info=init_info,
                    cfg=cfg,
                    train_info=train_info,
                    split_info=split_info,
                    data_progress=data_progress,
                    training_contract_info=training_contract_info,
                    convergence_state=convergence_state,
                    adaptive_curriculum_state=adaptive_curriculum_state,
                )
                save_checkpoint(state_file, payload)
                pb.close()
                if not stop_training:
                    pb = tqdm(
                        total=max(save_every, 1),
                        desc='ORACLE',
                        disable=not show_progress,
                    )
            if pause_checkpoint_due:
                if not checkpoint_due:
                    payload = checkpoint_payload(
                        oracle_brain=oracle_brain,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_val_loss=best_val_loss,
                        best_primary_loss=best_primary_loss,
                        best_observed_primary_loss=best_observed_primary_loss,
                        init_info=init_info,
                        cfg=cfg,
                        train_info=train_info,
                        split_info=split_info,
                        data_progress=data_progress,
                        training_contract_info=training_contract_info,
                        convergence_state=convergence_state,
                        adaptive_curriculum_state=adaptive_curriculum_state,
                    )
                    save_checkpoint(state_file, payload)
                shutdown_data_loader_iterator(train_loader, train_iterator)
                writer.flush()
                paused_for_external_request = True
                logging.info(
                    '[EXTERNAL_PAUSE] saved exact checkpoint at step=%s to %s',
                    steps,
                    state_file,
                )
            if steps < max_steps and not stop_training and not paused_for_external_request:
                optimizer_train_mode(optimizer)
            if paused_for_external_request:
                break
            if stop_training:
                stop_reason = (
                    adaptive_curriculum_state.get('last_reason', '')
                    if adaptive_curriculum_state is not None
                    and adaptive_curriculum_state.get('completed', False)
                    else convergence_state.get('last_reason', '')
                    if convergence_state is not None
                    else ''
                )
                logging.info(
                    'oracle critic controlled stop at step=%s: %s',
                    steps,
                    stop_reason,
                )
                break
            if released_for_eval:
                restart_same_cycle = steps < max_steps
                break

        if steps < max_steps and not stop_training and not paused_for_external_request:
            if restart_same_cycle:
                logging.info(
                    'restarting oracle critic train loader in cycle=%s after evaluation',
                    data_progress['cycle'],
                )
            else:
                advance_data_progress_cycle(data_progress)
                logging.info(
                    'oracle critic data stream advanced to cycle=%s fold=%s/%s',
                    data_progress['cycle'],
                    int(data_progress['cycle']) % int(cfg.get('state_fold_count', 1) or 1),
                    int(cfg.get('state_fold_count', 1) or 1),
                )
            train_loader = make_loader(
                make_dataset(train_files, cfg, train=True, stream_state=data_progress),
                cfg,
                train=True,
            )

    pb.close()
    if (
        convergence_state is not None
        and steps >= max_steps
        and not convergence_state.get('converged', False)
    ):
        logging.warning(
            'oracle critic reached hard cap step=%s before convergence; '
            'inspect fixed-dev trend before extending max_steps',
            steps,
        )
    if (
        adaptive_curriculum_state is not None
        and steps >= max_steps
        and not adaptive_curriculum_state.get('completed', False)
    ):
        logging.warning(
            'oracle critic reached adaptive hard cap step=%s before a '
            'transition/stop decision; inspect paired gates before extending',
            steps,
        )

    if (
        not paused_for_external_request
        and test_files
        and bool(cfg.get('final_test_enabled', False))
    ):
        seen_paths = set()
        for checkpoint_role, checkpoint_file in (
            ('latest', state_file),
            ('best_dev', best_state_file),
            ('best_primary', best_primary_state_file),
        ):
            resolved_checkpoint = path.abspath(checkpoint_file)
            if resolved_checkpoint in seen_paths or not path.exists(checkpoint_file):
                continue
            seen_paths.add(resolved_checkpoint)
            require_finalist_decision(
                args.finalist_decision or cfg.get('finalist_decision', ''), [checkpoint_file],
            )
            test_state = torch.load(checkpoint_file, weights_only=False, map_location=device)
            oracle_brain.load_state_dict(test_state['oracle_brain'])
            value_net.load_state_dict(test_state['value_net'])
            test_loader = make_loader(
                make_dataset(test_files, training_eval_cfg, train=False),
                training_eval_cfg,
                train=False,
            )
            test_by_input = evaluate_modes(
                oracle_brain,
                value_net,
                test_loader,
                device,
                enable_amp=eval_enable_amp,
                max_batches=test_batches,
                input_modes=eval_input_modes,
                log_every_batches=eval_log_every_batches,
                label=f'test/{checkpoint_role}/{"+".join(eval_input_modes)}',
            )
            test_metrics = test_by_input.get('true', next(iter(test_by_input.values())))
            test_dependency = summarize_oracle_dependency(test_by_input)
            logging.info(
                (
                    'final test role=%s step=%s loss=%.6f corr=%.4f ev=%.4f '
                    'samples=%s games=%s loss_ci95=[%.6f,%.6f]'
                ),
                checkpoint_role,
                int(test_state.get('steps', 0)),
                test_metrics['loss'],
                test_metrics['corr'],
                test_metrics['explained_variance'],
                test_metrics['num_samples'],
                test_metrics.get('num_games', 0),
                test_metrics.get('loss_ci95_low', test_metrics['loss']),
                test_metrics.get('loss_ci95_high', test_metrics['loss']),
            )
            with open(metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'steps': int(test_state.get('steps', 0)),
                    'checkpoint_role': checkpoint_role,
                    'test': test_metrics,
                    'test_by_input': test_by_input,
                    'oracle_dependency': test_dependency,
                    'test_split': split_info['test'],
                }, sort_keys=True) + '\n')

    writer.close()
    pb.close()
    if paused_for_external_request:
        return ORACLE_EXTERNAL_PAUSE_EXIT_CODE
    return 0


if __name__ == '__main__':
    import mortal.core.prelude

    try:
        raise SystemExit(train())
    except OracleEvaluationPaused as exc:
        logging.info('[EXTERNAL_PAUSE] %s; incomplete validation was not used for selection', exc)
        raise SystemExit(ORACLE_EXTERNAL_PAUSE_EXIT_CODE) from None

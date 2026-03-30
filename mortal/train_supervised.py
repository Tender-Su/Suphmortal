import inspect
import logging
import time
from os import path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def resolve_effective_config_section(config, config_section):
    if not isinstance(config, dict):
        return {}
    section_cfg = config.get(config_section)
    if isinstance(section_cfg, dict):
        return section_cfg
    if config_section == 'stage1':
        fallback_cfg = config.get('supervised')
        if isinstance(fallback_cfg, dict):
            return fallback_cfg
    return {}


def resolve_effective_aux_cfg(config, config_section):
    if not isinstance(config, dict):
        return {}
    base_aux_cfg = config.get('aux', {})
    if not isinstance(base_aux_cfg, dict):
        base_aux_cfg = {}
    aux_cfg = dict(base_aux_cfg)
    section_cfg = resolve_effective_config_section(config, config_section)
    section_aux_cfg = section_cfg.get('aux', {}) if isinstance(section_cfg, dict) else {}
    if isinstance(section_aux_cfg, dict):
        aux_cfg.update(section_aux_cfg)
    return aux_cfg


def checkpoint_optional_head_flags_for_state(state, *, config_section):
    saved_config = state.get('config')
    saved_aux_cfg = resolve_effective_aux_cfg(saved_config, config_section)

    checkpoint_opponent_enabled = state.get('opponent_aux_net') is not None
    if 'opponent_state_weight' in saved_aux_cfg:
        checkpoint_opponent_enabled = float(saved_aux_cfg.get('opponent_state_weight', 0.0) or 0.0) > 0

    checkpoint_danger_enabled = state.get('danger_aux_net') is not None
    if 'danger_enabled' in saved_aux_cfg or 'danger_weight' in saved_aux_cfg:
        checkpoint_danger_enabled = bool(
            saved_aux_cfg.get('danger_enabled', False)
            or float(saved_aux_cfg.get('danger_weight', 0.0) or 0.0) > 0
        )

    return {
        'opponent_aux_net': checkpoint_opponent_enabled,
        'danger_aux_net': checkpoint_danger_enabled,
    }

def resume_optimizer_steps_from_state(state, *, opt_step_every=1, default=0):
    if not isinstance(state, dict):
        return default
    if 'optimizer_steps' in state:
        return state['optimizer_steps']
    if 'steps' in state:
        legacy_steps = int(state['steps'])
        opt_step_every = int(opt_step_every)
        if opt_step_every <= 0:
            raise ValueError('opt_step_every must be positive')
        # Legacy checkpoints only stored micro-batch `steps`. They were saved only
        # after a completed optimizer step or an epoch-end flush, so reconstruct
        # optimizer updates with ceil division to preserve schedule state.
        return (legacy_steps + opt_step_every - 1) // opt_step_every
    return default


TRANSIENT_VALIDATION_FAILURE_MARKERS = (
    'error code: <1455>',
    "Couldn't open shared file mapping",
    'WinError 1455',
    'paging file is too small',
)


def is_retryable_validation_error(exc):
    error_parts = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_parts.append(str(current))
        error_parts.append(repr(current))
        current = current.__cause__ or current.__context__
    error_text = '\n'.join(error_parts)
    return any(marker in error_text for marker in TRANSIENT_VALIDATION_FAILURE_MARKERS)


def run_with_validation_retries(
    fn,
    *,
    device_type,
    context,
    sleep_fn=time.sleep,
    empty_cache_fn=None,
):
    if empty_cache_fn is None:
        empty_cache_fn = torch.cuda.empty_cache

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_retryable_validation_error(exc):
                raise
            attempt += 1
            wait_seconds = min(30.0, max(1.0, float(attempt)))
            if device_type == 'cuda':
                empty_cache_fn()
            logging.exception(
                '%s hit a transient loader/resource failure; retry same '
                'validation settings forever (attempt=%s, sleep=%.1fs): %s',
                context,
                attempt,
                wait_seconds,
                exc,
            )
            sleep_fn(wait_seconds)


def gradient_probe_rms(grad_tensor):
    if grad_tensor is None:
        return 0.0
    vector = grad_tensor.detach().float().reshape(-1)
    if vector.numel() <= 0:
        return 0.0
    return float(vector.square().mean().sqrt().item())


def gradient_probe_cosine(lhs, rhs):
    if lhs is None or rhs is None:
        return 0.0
    left = lhs.detach().float().reshape(-1)
    right = rhs.detach().float().reshape(-1)
    if left.numel() <= 0 or right.numel() <= 0:
        return 0.0
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float((left @ right).item() / (left_norm * right_norm))


def gradient_probe_combo_factor(lhs, rhs):
    if lhs is None or rhs is None:
        return 1.0
    left = lhs.detach().float().reshape(-1)
    right = rhs.detach().float().reshape(-1)
    if left.numel() <= 0 or right.numel() <= 0:
        return 1.0
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    denom = left_norm + right_norm
    if denom <= 1e-12:
        return 1.0
    combo_norm = float((left + right).norm().item())
    return combo_norm / denom


def plan_post_optimizer_step_actions(*, steps, save_every, val_every_steps, max_steps):
    periodic_save_due = save_every > 0 and steps % save_every == 0
    budget_reached = max_steps > 0 and steps >= max_steps
    monitor_validation_due = val_every_steps > 0 and steps % val_every_steps == 0

    validation_reason = None
    if budget_reached:
        validation_reason = 'max_steps'
    elif monitor_validation_due:
        validation_reason = 'monitor_val'

    return {
        'save_periodic': periodic_save_due,
        'save_budget_checkpoint': budget_reached and not periodic_save_due,
        'release_train_loader': budget_reached,
        'validation_reason': validation_reason,
        'stop_due_to_budget': budget_reached,
    }


def plan_budget_stop_final_actions(
    *,
    stop_due_to_budget,
    ran_full_val,
    has_full_recent_files,
    has_old_regression_files,
    old_regression_every_checks,
):
    run_full_validation = bool(stop_due_to_budget) and should_run_fallback_full_validation(
        ran_full_val=ran_full_val,
        has_full_recent_files=has_full_recent_files,
    )
    run_old_regression_validation = bool(stop_due_to_budget) and should_run_old_regression_after_full_validation(
        old_regression_every_checks=old_regression_every_checks,
        ran_full_val=run_full_validation,
        has_old_regression_files=has_old_regression_files,
    )
    return {
        'run_full_validation': run_full_validation,
        'run_old_regression_validation': run_old_regression_validation,
        'resave_latest_state': run_full_validation or run_old_regression_validation,
    }


def loader_uses_oracle(*, training, use_oracle, validation_use_oracle):
    if not use_oracle:
        return False
    return True if training else bool(validation_use_oracle)


def ensure_init_state_file_exists(init_state_file, *, cfg_prefix):
    if init_state_file and not path.exists(init_state_file):
        raise FileNotFoundError(f'{cfg_prefix}.init_state_file does not exist: {init_state_file}')


def normalize_numpy_bool_scalars(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, tuple):
        return tuple(normalize_numpy_bool_scalars(item) for item in value)
    if isinstance(value, list):
        return [normalize_numpy_bool_scalars(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_numpy_bool_scalars(item) for key, item in value.items()}
    return value


def safe_default_collate(batch):
    return default_collate([normalize_numpy_bool_scalars(item) for item in batch])


def paths_match(lhs, rhs):
    if not lhs or not rhs:
        return False
    return path.abspath(lhs) == path.abspath(rhs)


def resolve_stage2_handoff_state_file(*, cfg_prefix, supervised_cfg, control_cfg):
    if cfg_prefix != 'stage1':
        return ''
    if not isinstance(supervised_cfg, dict):
        return ''
    if not bool(supervised_cfg.get('publish_stage2_handoff', True)):
        return ''
    if not isinstance(control_cfg, dict):
        return ''
    return str(control_cfg.get('state_file', '') or '')


def should_enable_normal_export(
    *,
    export_normal_checkpoints,
    best_loss_normal_state_file,
    best_acc_normal_state_file,
    best_rank_normal_state_file,
    stage2_handoff_state_file,
):
    return bool(
        export_normal_checkpoints
        or best_loss_normal_state_file
        or best_acc_normal_state_file
        or best_rank_normal_state_file
        or stage2_handoff_state_file
    )


def should_run_full_validation_this_check(*, full_val_every_checks, validation_checks, has_full_recent_files):
    if not has_full_recent_files:
        return False
    if full_val_every_checks <= 0:
        return False
    return validation_checks % full_val_every_checks == 0


def should_run_fallback_full_validation(*, ran_full_val, has_full_recent_files):
    return bool(has_full_recent_files) and not ran_full_val


def should_run_old_regression_validation_this_check(
    *,
    old_regression_every_checks,
    validation_checks,
    has_old_regression_files,
):
    if not has_old_regression_files:
        return False
    if old_regression_every_checks <= 0:
        return False
    return validation_checks % old_regression_every_checks == 0


def should_run_old_regression_after_full_validation(
    *,
    old_regression_every_checks,
    ran_full_val,
    has_old_regression_files,
):
    return bool(has_old_regression_files) and ran_full_val and old_regression_every_checks <= 0


def make_closeable_batch_iter(loader, *, enable_cuda_prefetch, prefetcher_factory):
    if enable_cuda_prefetch:
        return prefetcher_factory(loader), True
    return iter(loader), False


def batch_includes_oracle(batch_len, *, enable_danger_aux):
    base_len = 5 + (4 if enable_danger_aux else 0)
    valid_without_oracle = {base_len, base_len + 2}
    valid_with_oracle = {length + 1 for length in valid_without_oracle}
    if batch_len in valid_without_oracle:
        return False
    if batch_len in valid_with_oracle:
        return True
    raise ValueError(
        'unexpected batch length: '
        f'expected one of {sorted(valid_without_oracle | valid_with_oracle)}, got {batch_len}'
    )


def backfill_missing_oracle_obs(obs, oracle_obs, *, require_oracle, oracle_channels):
    if oracle_obs is not None or not require_oracle:
        return oracle_obs
    return obs.new_zeros((obs.shape[0], int(oracle_channels), obs.shape[-1]))


def resolve_turn_weighting_cfg(
    raw_cfg,
    *,
    default_early_factor,
    default_mid_factor,
    default_late_factor,
    default_early_max_turn=6,
    default_late_min_turn=13,
):
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    early_max_turn = raw_cfg.get('early_max_turn', default_early_max_turn)
    late_min_turn = raw_cfg.get('late_min_turn', default_late_min_turn)
    early_factor = raw_cfg.get('early_factor', default_early_factor)
    mid_factor = raw_cfg.get('mid_factor', default_mid_factor)
    late_factor = raw_cfg.get('late_factor', default_late_factor)
    if early_max_turn is None:
        early_max_turn = default_early_max_turn
    if late_min_turn is None:
        late_min_turn = default_late_min_turn
    if early_factor is None:
        early_factor = default_early_factor
    if mid_factor is None:
        mid_factor = default_mid_factor
    if late_factor is None:
        late_factor = default_late_factor
    early_max_turn = int(early_max_turn)
    late_min_turn = int(late_min_turn)
    early_max_turn = max(0, early_max_turn)
    late_min_turn = max(early_max_turn + 1, late_min_turn)
    return {
        'early_factor': max(float(early_factor), 0.0),
        'mid_factor': max(float(mid_factor), 0.0),
        'late_factor': max(float(late_factor), 0.0),
        'early_max_turn': early_max_turn,
        'late_min_turn': late_min_turn,
    }


def compute_turn_bucket_weights(
    at_turn,
    *,
    early_factor,
    mid_factor,
    late_factor,
    early_max_turn=6,
    late_min_turn=13,
):
    if not torch.is_tensor(at_turn):
        at_turn = torch.as_tensor(at_turn, dtype=torch.int64)
    else:
        at_turn = at_turn.to(dtype=torch.int64)
    weights = torch.full(
        at_turn.shape,
        float(mid_factor),
        dtype=torch.float32,
        device=at_turn.device,
    )
    weights = torch.where(
        at_turn <= int(early_max_turn),
        torch.full_like(weights, float(early_factor)),
        weights,
    )
    weights = torch.where(
        at_turn >= int(late_min_turn),
        torch.full_like(weights, float(late_factor)),
        weights,
    )
    return weights


def init_exact_action_metric_dict(*, device):
    return {
        'nll_sum': torch.zeros((), dtype=torch.float64, device=device),
        'top1_correct': torch.zeros((), dtype=torch.int64, device=device),
        'top3_correct': torch.zeros((), dtype=torch.int64, device=device),
        'count': torch.zeros((), dtype=torch.int64, device=device),
    }


def compute_exact_action_metric_stats(
    probs,
    actions,
    *,
    start,
    end,
    topk_size=3,
    normalize_within_slice=False,
):
    probs = probs.detach().to(torch.float32)
    actions = actions.to(device=probs.device, dtype=torch.int64)
    result = init_exact_action_metric_dict(device=probs.device)
    target = (actions >= start) & (actions < end)
    result['count'] = target.to(torch.int64).sum()
    if int(result['count'].item()) <= 0:
        return result

    target_probs = probs[target]
    target_actions = actions[target]
    if normalize_within_slice:
        target_probs = target_probs[:, start:end]
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target_actions = target_actions - start

    predicted_actions = target_probs.argmax(-1)
    chosen_probs = target_probs.gather(1, target_actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
    topk = target_probs.topk(k=min(topk_size, target_probs.shape[-1]), dim=-1).indices

    result['nll_sum'] = -chosen_probs.log().sum().to(torch.float64)
    result['top1_correct'] = predicted_actions.eq(target_actions).to(torch.int64).sum()
    result['top3_correct'] = topk.eq(target_actions.unsqueeze(-1)).any(-1).to(torch.int64).sum()
    return result


def train(
    config_section='supervised',
    *,
    stage_label='Supervised Pretraining (Stage 0.5)',
    checkpoint_label='supervised',
    export_normal_checkpoints=False,
):
    import copy
    import gc
    import gzip
    import json
    import logging
    import math
    import random
    import sys
    import os
    import torch
    import time
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from pathlib import Path
    from torch import nn, optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    import prelude
    from common import filtered_trimmed_lines, parameter_count, tqdm
    from checkpoint_utils import (
        load_brain_state_with_input_bridge,
        make_normal_checkpoint_from_oracle_checkpoint,
    )
    from config import config
    from dataloader import SupervisedFileDatasetsIter, resolve_rayon_num_threads, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from libriichi.consts import ACTION_SPACE, obs_shape, oracle_obs_shape
    from model import AuxNet, Brain, CategoricalPolicy, DangerAuxNet, OpponentStateAuxNet
    from stage05_selection import (
        ACTION_SCORE_WEIGHTS,
        SCENARIO_SCORE_WEIGHTS,
        action_quality_score,
        refresh_scenario_quality_score,
        refresh_selection_quality_score,
    )

    version = config['control']['version']
    oracle_channels = oracle_obs_shape(version)[0]
    if config_section not in config:
        if config_section != 'stage1' or 'supervised' not in config:
            raise KeyError(f'missing config section: {config_section}')

        supervised_fallback = copy.deepcopy(config['supervised'])
        fallback_state_dir = Path(supervised_fallback['state_file']).resolve().parent
        fallback_tb_dir = Path(supervised_fallback['tensorboard_dir']).resolve().parent
        fallback_init_state = config.get('oracle', {}).get(
            'init_state_file',
            supervised_fallback.get('best_loss_state_file', supervised_fallback['best_state_file']),
        )
        synthetic_stage1 = {
            **supervised_fallback,
            'state_file': str(fallback_state_dir / 'stage1_latest_oracle.pth'),
            'best_state_file': str(fallback_state_dir / 'stage1_best_loss_oracle.pth'),
            'best_loss_state_file': str(fallback_state_dir / 'stage1_best_loss_oracle.pth'),
            'best_acc_state_file': str(fallback_state_dir / 'stage1_best_acc_oracle.pth'),
            'best_rank_state_file': str(fallback_state_dir / 'stage1_best_rank_oracle.pth'),
            'best_loss_normal_state_file': str(fallback_state_dir / 'stage1_best_loss_normal.pth'),
            'best_acc_normal_state_file': str(fallback_state_dir / 'stage1_best_acc_normal.pth'),
            'best_rank_normal_state_file': str(fallback_state_dir / 'stage1_best_rank_normal.pth'),
            'tensorboard_dir': str(fallback_tb_dir / 'tb_log_stage1'),
            'file_index': str(fallback_state_dir / 'file_index_stage1_json.pth'),
            'init_state_file': fallback_init_state,
            'enable_oracle': True,
            'validation_use_oracle': False,
            'oracle_dropout': {
                'enabled': True,
                'schedule': 'linear',
                'gamma_start': float(config.get('oracle', {}).get('gamma_start', 1.0)),
                'gamma_end': float(config.get('oracle', {}).get('gamma_end', 0.0)),
                'hold_steps': 0,
                'decay_steps': int(
                    supervised_fallback.get(
                        'max_steps',
                        config.get('optim', {}).get('scheduler', {}).get('max_steps', 0),
                    ) or 0
                ),
            },
        }
        config[config_section] = synthetic_stage1
        logging.warning(
            'missing [stage1] in config; using synthesized Stage 1 section derived from [supervised]. '
            'Add an explicit [stage1] section to config.toml to freeze real Stage 1 semantics.'
        )
    supervised_cfg = config[config_section]
    cfg_prefix = config_section

    batch_size = supervised_cfg.get('batch_size', config['control']['batch_size'])
    opt_step_every = config['control']['opt_step_every']
    log_every = supervised_cfg['log_every']
    save_every = supervised_cfg.get('save_every', config['control']['save_every'])
    max_epochs = supervised_cfg['max_epochs']
    min_epochs = supervised_cfg.get('min_epochs', 1)
    early_stopping_patience = supervised_cfg.get('early_stopping_patience', 0)
    early_stopping_patience_checks = supervised_cfg.get('early_stopping_patience_checks', early_stopping_patience)
    early_stopping_min_delta = supervised_cfg.get('early_stopping_min_delta', 0.0)
    early_stopping_min_lr_reductions = supervised_cfg.get('early_stopping_min_lr_reductions', 0)
    min_validation_checks = supervised_cfg.get('min_validation_checks', min_epochs)
    val_every_steps = supervised_cfg.get('val_every_steps', 0)
    monitor_val_batches = supervised_cfg.get('monitor_val_batches', 0)
    full_val_every_checks = supervised_cfg.get('full_val_every_checks', 0)
    old_regression_every_checks = supervised_cfg.get('old_regression_every_checks', 0)
    max_steps = supervised_cfg.get('max_steps', 0)
    force_safe_training = supervised_cfg.get('force_safe_training', False)
    gradient_calibration_cfg = supervised_cfg.get('gradient_calibration', {})
    if not isinstance(gradient_calibration_cfg, dict):
        gradient_calibration_cfg = {}
    gradient_calibration_enabled = bool(gradient_calibration_cfg.get('enabled', False))
    gradient_calibration_split = str(gradient_calibration_cfg.get('split', 'full_recent'))
    gradient_calibration_max_batches = int(gradient_calibration_cfg.get('max_batches', 0) or 0)
    if save_every < 0:
        raise ValueError(f'{cfg_prefix}.save_every must be non-negative')
    if save_every > 0 and save_every % opt_step_every != 0:
        raise ValueError(f'{cfg_prefix}.save_every must be divisible by control.opt_step_every')
    if val_every_steps < 0:
        raise ValueError(f'{cfg_prefix}.val_every_steps must be non-negative')
    if val_every_steps > 0 and val_every_steps % opt_step_every != 0:
        raise ValueError(f'{cfg_prefix}.val_every_steps must be divisible by control.opt_step_every')
    if monitor_val_batches < 0:
        raise ValueError(f'{cfg_prefix}.monitor_val_batches must be non-negative')
    if full_val_every_checks < 0:
        raise ValueError(f'{cfg_prefix}.full_val_every_checks must be non-negative')
    if gradient_calibration_split not in {'monitor_recent', 'full_recent'}:
        raise ValueError(
            f'{cfg_prefix}.gradient_calibration.split must be monitor_recent or full_recent'
        )
    if gradient_calibration_max_batches < 0:
        raise ValueError(f'{cfg_prefix}.gradient_calibration.max_batches must be non-negative')

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']
    allow_tf32 = config['control'].get('allow_tf32', True)
    enable_cuda_prefetch = config['control'].get('enable_cuda_prefetch', True) and device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')

    file_batch_size = supervised_cfg.get('file_batch_size', config['dataset']['file_batch_size'])
    val_file_batch_size = supervised_cfg.get('val_file_batch_size', file_batch_size)
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = supervised_cfg.get('num_workers', config['dataset']['num_workers'])
    prefetch_factor = supervised_cfg.get('prefetch_factor', 2)
    val_prefetch_factor = supervised_cfg.get('val_prefetch_factor', prefetch_factor)
    explicit_rayon_threads = supervised_cfg.get('rayon_num_threads', config['dataset'].get('rayon_num_threads', 0))
    rayon_num_threads = resolve_rayon_num_threads(num_workers, file_batch_size, explicit_rayon_threads)
    worker_torch_num_threads = supervised_cfg.get('worker_torch_num_threads', config['dataset'].get('worker_torch_num_threads', 1))
    worker_torch_num_interop_threads = supervised_cfg.get(
        'worker_torch_num_interop_threads',
        config['dataset'].get('worker_torch_num_interop_threads', 1),
    )
    train_in_order = supervised_cfg.get('train_in_order', False)
    val_in_order = supervised_cfg.get('val_in_order', True)
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']

    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    aux_cfg = resolve_effective_aux_cfg(config, cfg_prefix)
    aux_weight = aux_cfg['next_rank_weight']
    opponent_state_weight = aux_cfg.get('opponent_state_weight', 0.0)
    opponent_shanten_weight = aux_cfg.get('opponent_shanten_weight', 1.0)
    opponent_tenpai_weight = aux_cfg.get('opponent_tenpai_weight', 1.0)
    enable_opponent_state_aux = opponent_state_weight > 0
    emit_opponent_state_metric_labels = bool(aux_cfg.get('emit_opponent_state_labels', True))
    danger_enabled = aux_cfg.get('danger_enabled', False)
    danger_weight = aux_cfg.get('danger_weight', 0.0)
    danger_any_weight = aux_cfg.get('danger_any_weight', 0.0)
    danger_value_weight = aux_cfg.get('danger_value_weight', 0.0)
    danger_player_weight = aux_cfg.get('danger_player_weight', 0.0)
    danger_focal_gamma = aux_cfg.get('danger_focal_gamma', 0.0)
    danger_ramp_steps = aux_cfg.get('danger_ramp_steps', 0)
    danger_value_cap = float(aux_cfg.get('danger_value_cap', 96000.0))
    danger_value_cap_log = math.log1p(max(danger_value_cap, 1.0))
    enable_danger_aux = danger_enabled or danger_weight > 0
    rank_aux_cfg = supervised_cfg.get('rank_aux', {})
    if not isinstance(rank_aux_cfg, dict):
        rank_aux_cfg = {}
    rank_aux_base_weight = rank_aux_cfg.get('base_weight', aux_weight)
    rank_aux_south_factor = rank_aux_cfg.get('south_factor', 1.0)
    rank_aux_all_last_factor = rank_aux_cfg.get('all_last_factor', 1.0)
    rank_aux_gap_focus_points = rank_aux_cfg.get('gap_focus_points', 0.0)
    rank_aux_gap_close_bonus = rank_aux_cfg.get('gap_close_bonus', 0.0)
    rank_aux_max_weight = rank_aux_cfg.get('max_weight', 0.0)
    rank_turn_weighting = resolve_turn_weighting_cfg(
        rank_aux_cfg.get('turn_weighting', {}),
        default_early_factor=1.0,
        default_mid_factor=1.05,
        default_late_factor=1.15,
        default_early_max_turn=4,
        default_late_min_turn=12,
    )
    opponent_turn_weighting = resolve_turn_weighting_cfg(
        aux_cfg.get('opponent_turn_weighting', {}),
        default_early_factor=0.20,
        default_mid_factor=1.0,
        default_late_factor=1.60,
        default_early_max_turn=4,
        default_late_min_turn=12,
    )
    danger_turn_weighting = resolve_turn_weighting_cfg(
        aux_cfg.get('danger_turn_weighting', {}),
        default_early_factor=0.05,
        default_mid_factor=1.0,
        default_late_factor=2.5,
        default_early_max_turn=4,
        default_late_min_turn=12,
    )
    context_meta_specs = {
        'at_turn': 0,
        'round_stage': 1,
        'is_dealer': 2,
        'is_all_last': 3,
        'self_rank': 4,
        'opp_riichi_count': 5,
        'up_gap_100': 6,
        'down_gap_100': 7,
    }
    val_ratio = supervised_cfg['val_ratio']
    min_val_files = supervised_cfg['min_val_files']
    max_train_files = supervised_cfg.get('max_train_files', 0)
    max_val_files = supervised_cfg.get('max_val_files', 0)
    seed = supervised_cfg['seed']
    use_oracle = supervised_cfg.get('enable_oracle', False)
    validation_use_oracle = supervised_cfg.get(
        'validation_use_oracle',
        False if cfg_prefix == 'stage1' else use_oracle,
    )
    oracle_dropout_cfg = supervised_cfg.get('oracle_dropout', {})
    oracle_dropout_enabled = bool(use_oracle and oracle_dropout_cfg.get('enabled', False))
    oracle_dropout_schedule = oracle_dropout_cfg.get('schedule', 'none')
    oracle_dropout_gamma_start = float(oracle_dropout_cfg.get('gamma_start', 1.0))
    oracle_dropout_gamma_end = float(oracle_dropout_cfg.get('gamma_end', 0.0))
    oracle_dropout_hold_steps = int(oracle_dropout_cfg.get('hold_steps', 0))
    oracle_dropout_decay_steps = int(oracle_dropout_cfg.get('decay_steps', 0))

    state_file = supervised_cfg['state_file']
    best_state_file = supervised_cfg['best_state_file']
    best_loss_state_file = supervised_cfg.get('best_loss_state_file', best_state_file)
    best_acc_state_file = supervised_cfg.get('best_acc_state_file', best_state_file)
    best_rank_state_file = supervised_cfg.get('best_rank_state_file', best_state_file)
    best_loss_normal_state_file = supervised_cfg.get('best_loss_normal_state_file', '')
    best_acc_normal_state_file = supervised_cfg.get('best_acc_normal_state_file', '')
    best_rank_normal_state_file = supervised_cfg.get('best_rank_normal_state_file', '')
    init_state_file = supervised_cfg.get('init_state_file', '')
    tensorboard_dir = supervised_cfg['tensorboard_dir']
    file_index = supervised_cfg['file_index']
    stage2_handoff_state_file = resolve_stage2_handoff_state_file(
        cfg_prefix=cfg_prefix,
        supervised_cfg=supervised_cfg,
        control_cfg=config.get('control', {}),
    )
    scheduler_cfg = supervised_cfg.get('scheduler', {})
    peak_lr = supervised_cfg.get('lr', config['optim']['scheduler']['peak'])
    warm_up_steps = scheduler_cfg.get('warm_up_steps', config['optim']['scheduler'].get('warm_up_steps', 0))
    warmup_init = scheduler_cfg.get('init', config['optim']['scheduler'].get('init', 1e-8))
    scheduler_type = scheduler_cfg.get('type', 'plateau')
    normal_export_enabled = should_enable_normal_export(
        export_normal_checkpoints=export_normal_checkpoints,
        best_loss_normal_state_file=best_loss_normal_state_file,
        best_acc_normal_state_file=best_acc_normal_state_file,
        best_rank_normal_state_file=best_rank_normal_state_file,
        stage2_handoff_state_file=stage2_handoff_state_file,
    )
    if normal_export_enabled and not use_oracle:
        logging.info('normal-export requested on non-oracle run; normal export will mirror original weights')
    if oracle_dropout_enabled and oracle_dropout_schedule not in {'linear', 'cosine', 'none'}:
        raise ValueError(
            f'unsupported {cfg_prefix}.oracle_dropout.schedule: {oracle_dropout_schedule}'
        )
    if oracle_dropout_enabled and oracle_dropout_decay_steps < 0:
        raise ValueError(f'{cfg_prefix}.oracle_dropout.decay_steps must be non-negative')

    action_group_specs = {
        'discard': (0, 37),
        'riichi': (37, 38),
        'chi': (38, 41),
        'pon': (41, 42),
        'kan': (42, 43),
        'agari': (43, 44),
        'ryukyoku': (44, 45),
        'pass': (45, 46),
    }
    action_group_names = tuple(action_group_specs.keys())
    num_action_groups = len(action_group_names)
    action_to_group = torch.empty(ACTION_SPACE, dtype=torch.int64, device=device)
    for idx, (start, end) in enumerate(action_group_specs.values()):
        action_to_group[start:end] = idx

    random.seed(seed)
    torch.manual_seed(seed)

    mortal = Brain(version=version, is_oracle=use_oracle, **config['resnet'], Norm='GN').to(device)
    normal_export_brain = (
        Brain(version=version, is_oracle=False, **config['resnet'], Norm='GN')
        if normal_export_enabled else None
    )
    policy_net = CategoricalPolicy().to(device)
    aux_net = AuxNet(dims=(4,)).to(device)
    opponent_aux_net = OpponentStateAuxNet().to(device) if enable_opponent_state_aux else None
    danger_aux_net = DangerAuxNet().to(device) if enable_danger_aux else None
    named_models = (
        ('mortal', mortal),
        ('policy_net', policy_net),
        ('aux_net', aux_net),
    ) + ((('opponent_aux_net', opponent_aux_net),) if opponent_aux_net is not None else ()) + (
        (('danger_aux_net', danger_aux_net),) if danger_aux_net is not None else ()
    )
    all_models = tuple(model for _, model in named_models)
    if enable_compile:
        for model in all_models:
            model.compile()

    logging.info(f'=== {stage_label} ===')
    logging.info(f'config_section: {cfg_prefix}')
    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    if use_oracle:
        logging.info(f'oracle obs shape: {oracle_obs_shape(version)}')
    logging.info(f'use_oracle: {use_oracle}')
    logging.info(f'validation_use_oracle: {validation_use_oracle}')
    logging.info(f'normal_export_enabled: {normal_export_enabled}')
    logging.info(f'oracle_dropout_enabled: {oracle_dropout_enabled}')
    logging.info(f'oracle_dropout_schedule: {oracle_dropout_schedule}')
    logging.info(f'oracle_dropout_gamma_start: {oracle_dropout_gamma_start}')
    logging.info(f'oracle_dropout_gamma_end: {oracle_dropout_gamma_end}')
    logging.info(f'oracle_dropout_hold_steps: {oracle_dropout_hold_steps}')
    logging.info(f'oracle_dropout_decay_steps: {oracle_dropout_decay_steps}')
    logging.info(f'batch_size: {batch_size}')
    logging.info(f'num_workers: {num_workers}')
    logging.info(f'rayon_num_threads: {rayon_num_threads}')
    logging.info(f'worker_torch_num_threads: {worker_torch_num_threads}')
    logging.info(f'worker_torch_num_interop_threads: {worker_torch_num_interop_threads}')
    logging.info(f'file_batch_size: {file_batch_size}')
    logging.info(f'val_file_batch_size: {val_file_batch_size}')
    logging.info(f'val_prefetch_factor: {val_prefetch_factor}')
    logging.info(f'train_in_order: {train_in_order}')
    logging.info(f'val_in_order: {val_in_order}')
    logging.info(f'save_every: {save_every}')
    logging.info(f'val_every_steps: {val_every_steps}')
    logging.info(f'monitor_val_batches: {monitor_val_batches}')
    logging.info(
        'full_val_every_checks: %s',
        'disabled' if full_val_every_checks <= 0 else full_val_every_checks,
    )
    logging.info(f'old_regression_every_checks: {old_regression_every_checks}')
    logging.info(f'force_safe_training: {force_safe_training}')
    logging.info(f'gradient_calibration_enabled: {gradient_calibration_enabled}')
    logging.info(f'gradient_calibration_split: {gradient_calibration_split}')
    logging.info(f'gradient_calibration_max_batches: {gradient_calibration_max_batches}')
    logging.info(f'min_validation_checks: {min_validation_checks}')
    logging.info(f'early_stopping_patience_checks: {early_stopping_patience_checks}')
    logging.info(f'allow_tf32: {allow_tf32}')
    logging.info(f'enable_cuda_prefetch: {enable_cuda_prefetch}')
    logging.info(f'scheduler_type: {scheduler_type}')
    logging.info(f'peak_lr: {peak_lr:.3e}')
    logging.info(f'warm_up_steps: {warm_up_steps}')
    logging.info(f'max_steps: {max_steps}')
    logging.info(f'action_score_weights: {ACTION_SCORE_WEIGHTS}')
    logging.info(f'opponent_state_weight: {opponent_state_weight}')
    logging.info(f'opponent_shanten_weight: {opponent_shanten_weight}')
    logging.info(f'opponent_tenpai_weight: {opponent_tenpai_weight}')
    logging.info(f'emit_opponent_state_metric_labels: {emit_opponent_state_metric_labels}')
    logging.info(f'danger_enabled: {enable_danger_aux}')
    logging.info(f'danger_weight: {danger_weight}')
    logging.info(f'danger_any_weight: {danger_any_weight}')
    logging.info(f'danger_value_weight: {danger_value_weight}')
    logging.info(f'danger_player_weight: {danger_player_weight}')
    logging.info(f'danger_focal_gamma: {danger_focal_gamma}')
    logging.info(f'danger_ramp_steps: {danger_ramp_steps}')
    logging.info(f'danger_value_cap: {danger_value_cap}')
    logging.info(f'rank_aux_base_weight: {rank_aux_base_weight}')
    logging.info(f'rank_aux_south_factor: {rank_aux_south_factor}')
    logging.info(f'rank_aux_all_last_factor: {rank_aux_all_last_factor}')
    logging.info(f'rank_aux_gap_focus_points: {rank_aux_gap_focus_points}')
    logging.info(f'rank_aux_gap_close_bonus: {rank_aux_gap_close_bonus}')
    logging.info(f'rank_aux_max_weight: {rank_aux_max_weight}')
    logging.info(
        'rank_turn_weighting: early<=%s -> %.3f, mid -> %.3f, late>=%s -> %.3f',
        rank_turn_weighting['early_max_turn'],
        rank_turn_weighting['early_factor'],
        rank_turn_weighting['mid_factor'],
        rank_turn_weighting['late_min_turn'],
        rank_turn_weighting['late_factor'],
    )
    logging.info(
        'opponent_turn_weighting: early<=%s -> %.3f, mid -> %.3f, late>=%s -> %.3f',
        opponent_turn_weighting['early_max_turn'],
        opponent_turn_weighting['early_factor'],
        opponent_turn_weighting['mid_factor'],
        opponent_turn_weighting['late_min_turn'],
        opponent_turn_weighting['late_factor'],
    )
    logging.info(
        'danger_turn_weighting: early<=%s -> %.3f, mid -> %.3f, late>=%s -> %.3f',
        danger_turn_weighting['early_max_turn'],
        danger_turn_weighting['early_factor'],
        danger_turn_weighting['mid_factor'],
        danger_turn_weighting['late_min_turn'],
        danger_turn_weighting['late_factor'],
    )
    logging.info(f'scenario_score_weights: {SCENARIO_SCORE_WEIGHTS}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy_net):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')
    if opponent_aux_net is not None:
        logging.info(f'opponent_aux params: {parameter_count(opponent_aux_net):,}')
    if danger_aux_net is not None:
        logging.info(f'danger_aux params: {parameter_count(danger_aux_net):,}')

    def collect_optimizer_param_specs():
        decay_specs = []
        no_decay_specs = []
        for model_name, model in named_models:
            params_dict = {}
            to_decay = set()
            for mod_name, mod in model.named_modules():
                for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                    qualified_name = f'{model_name}.{name}' if name else model_name
                    params_dict[qualified_name] = param
                    if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                        to_decay.add(qualified_name)
            decay_specs.extend((name, params_dict[name]) for name in sorted(to_decay))
            no_decay_specs.extend(
                (name, params_dict[name]) for name in sorted(params_dict.keys() - to_decay)
            )
        return decay_specs, no_decay_specs

    decay_specs, no_decay_specs = collect_optimizer_param_specs()
    optimizer_param_groups = (
        tuple(name for name, _ in decay_specs),
        tuple(name for name, _ in no_decay_specs),
    )
    param_groups = [
        {'params': [param for _, param in decay_specs], 'weight_decay': weight_decay},
        {'params': [param for _, param in no_decay_specs]},
    ]
    if scheduler_type == 'plateau':
        optimizer = optim.AdamW(
            param_groups,
            lr=warmup_init if warm_up_steps > 0 else peak_lr,
            weight_decay=0,
            betas=betas,
            eps=eps,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_cfg.get('factor', scheduler_cfg.get('plateau_factor', 0.5)),
            patience=scheduler_cfg.get('patience', scheduler_cfg.get('plateau_patience', 1)),
            threshold=scheduler_cfg.get('threshold', scheduler_cfg.get('plateau_threshold', 0.0005)),
            cooldown=scheduler_cfg.get('cooldown', scheduler_cfg.get('plateau_cooldown', 0)),
            min_lr=scheduler_cfg.get('min_lr', 1e-6),
        )
    elif scheduler_type == 'cosine':
        cosine_total_steps = scheduler_cfg.get(
            'max_steps',
            scheduler_cfg.get('cosine_total_steps', max_steps if max_steps > 0 else config['optim']['scheduler'].get('max_steps', 0)),
        )
        if cosine_total_steps <= 0:
            raise ValueError(
                f'{cfg_prefix}.scheduler.max_steps/cosine_total_steps must be positive when '
                'scheduler.type=cosine'
            )
        cosine_final_lr = scheduler_cfg.get('final', scheduler_cfg.get('cosine_final_lr', config['optim']['scheduler'].get('final', 1e-5)))
        optimizer = optim.AdamW(
            param_groups,
            lr=1,
            weight_decay=0,
            betas=betas,
            eps=eps,
        )
        scheduler = LinearWarmUpCosineAnnealingLR(
            optimizer,
            peak=peak_lr,
            final=cosine_final_lr,
            warm_up_steps=warm_up_steps,
            max_steps=cosine_total_steps,
            init=warmup_init,
        )
    else:
        raise ValueError(f'unsupported {cfg_prefix}.scheduler.type: {scheduler_type}')
    scaler = GradScaler(device.type, enabled=enable_amp)
    def current_oracle_dropout_gamma():
        if not use_oracle:
            return 0.0
        if not oracle_dropout_enabled or oracle_dropout_schedule == 'none':
            return 1.0
        if optimizer_steps <= oracle_dropout_hold_steps:
            return oracle_dropout_gamma_start
        if oracle_dropout_decay_steps <= 0:
            return oracle_dropout_gamma_end
        progress = min(
            max(optimizer_steps - oracle_dropout_hold_steps, 0) / oracle_dropout_decay_steps,
            1.0,
        )
        if oracle_dropout_schedule == 'linear':
            return oracle_dropout_gamma_start + (
                oracle_dropout_gamma_end - oracle_dropout_gamma_start
            ) * progress
        if oracle_dropout_schedule == 'cosine':
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return oracle_dropout_gamma_end + (
                oracle_dropout_gamma_start - oracle_dropout_gamma_end
            ) * cosine
        return oracle_dropout_gamma_start

    def prepare_oracle_obs(oracle_obs, *, obs, training):
        if training:
            oracle_obs = backfill_missing_oracle_obs(
                obs,
                oracle_obs,
                require_oracle=use_oracle,
                oracle_channels=oracle_channels,
            )
            if oracle_obs is None:
                return None, 0.0
            gamma = current_oracle_dropout_gamma()
            if gamma <= 0:
                return torch.zeros_like(oracle_obs), gamma
            if gamma >= 1:
                return oracle_obs, gamma
            keep_mask = torch.rand_like(oracle_obs).le(gamma)
            return oracle_obs * keep_mask.to(dtype=oracle_obs.dtype), gamma
        if validation_use_oracle:
            oracle_obs = backfill_missing_oracle_obs(
                obs,
                oracle_obs,
                require_oracle=use_oracle,
                oracle_channels=oracle_channels,
            )
            return oracle_obs, 1.0
        oracle_obs = backfill_missing_oracle_obs(
            obs,
            oracle_obs,
            require_oracle=use_oracle,
            oracle_channels=oracle_channels,
        )
        if oracle_obs is None:
            return None, 0.0
        return torch.zeros_like(oracle_obs), 0.0

    def export_normal_state(state, export_path, *, label):
        if not export_path or normal_export_brain is None:
            return
        export_state = make_normal_checkpoint_from_oracle_checkpoint(state, normal_export_brain)
        export_state.pop('optimizer', None)
        export_state.pop('optimizer_param_groups', None)
        export_state.pop('scheduler', None)
        export_state.pop('scaler', None)
        export_state['resume_supported'] = False
        export_state['exported_from_oracle'] = bool(use_oracle)
        export_state['checkpoint_label'] = checkpoint_label
        export_config = copy.deepcopy(export_state.get('config', config))
        if isinstance(export_config.get(cfg_prefix), dict):
            export_config[cfg_prefix]['enable_oracle'] = False
            export_config[cfg_prefix]['validation_use_oracle'] = False
        export_state['config'] = export_config
        torch.save(export_state, export_path)
        logging.info(f'saved {label} to {export_path}')

    def load_optional_head_states(state):
        mortal.load_state_dict(state['mortal'])
        policy_net.load_state_dict(state['policy_net'])
        if state.get('aux_net') is not None:
            aux_net.load_state_dict(state['aux_net'])
        if opponent_aux_net is not None and state.get('opponent_aux_net') is not None:
            opponent_aux_net.load_state_dict(state['opponent_aux_net'])
        if danger_aux_net is not None and state.get('danger_aux_net') is not None:
            danger_aux_net.load_state_dict(state['danger_aux_net'])

    def get_checkpoint_optional_head_flags(state):
        return checkpoint_optional_head_flags_for_state(state, config_section=cfg_prefix)

    def validate_exact_resume_heads(state):
        checkpoint_flags = get_checkpoint_optional_head_flags(state)
        current_flags = {
            'opponent_aux_net': enable_opponent_state_aux,
            'danger_aux_net': enable_danger_aux,
        }
        mismatches = [
            f'{head}: checkpoint={checkpoint_flags[head]} current={current_flags[head]}'
            for head in current_flags
            if checkpoint_flags[head] != current_flags[head]
        ]
        missing_enabled_weights = [
            head for head, enabled in current_flags.items() if enabled and state.get(head) is None
        ]
        if mismatches or missing_enabled_weights:
            details = mismatches[:]
            if missing_enabled_weights:
                details.append(
                    'missing weights for enabled heads: '
                    + ', '.join(sorted(missing_enabled_weights))
                )
            raise RuntimeError(
                f'{cfg_prefix}.state_file requires an exact auxiliary-head match, but '
                + '; '.join(details)
                + f'. Use {cfg_prefix}.init_state_file for weights-only initialization, '
                'or remove the stale resume checkpoint and start a fresh run.'
            )

    optimizer_state_aliases = {
        'danger_aux_net.net.weight': (
            'danger_aux_net.any_net.weight',
            'danger_aux_net.value_net.weight',
            'danger_aux_net.player_net.weight',
        ),
    }

    def merge_optimizer_state_entries(state_entries):
        merged = {}
        for key in state_entries[0]:
            values = [entry[key] for entry in state_entries]
            first = values[0]
            if torch.is_tensor(first):
                merged[key] = (
                    first.clone()
                    if first.ndim == 0 else torch.cat([value.clone() for value in values], dim=0)
                )
            else:
                merged[key] = copy.deepcopy(first)
        return merged

    def load_optimizer_state_compat(saved_optimizer_state, saved_optimizer_param_groups=None):
        if saved_optimizer_state is None:
            return False
        try:
            optimizer.load_state_dict(saved_optimizer_state)
            return True
        except ValueError:
            if not saved_optimizer_param_groups:
                return False
            current_optimizer_state = optimizer.state_dict()
            saved_groups = saved_optimizer_state.get('param_groups', ())
            current_groups = current_optimizer_state.get('param_groups', ())
            if len(saved_groups) != len(current_groups):
                return False
            if len(saved_optimizer_param_groups) != len(saved_groups):
                return False
            if len(optimizer_param_groups) != len(current_groups):
                return False

            merged_groups = []
            merged_state = {}
            saved_state_entries = saved_optimizer_state.get('state', {})
            matched_params = 0

            for saved_group, current_group, saved_names, current_names in zip(
                saved_groups,
                current_groups,
                saved_optimizer_param_groups,
                optimizer_param_groups,
            ):
                if len(saved_group['params']) != len(saved_names):
                    return False
                if len(current_group['params']) != len(current_names):
                    return False
                saved_name_to_param_id = {
                    name: param_id for name, param_id in zip(saved_names, saved_group['params'])
                }

                for current_name, current_param_id in zip(current_names, current_group['params']):
                    saved_param_id = saved_name_to_param_id.get(current_name)
                    if saved_param_id is not None and saved_param_id in saved_state_entries:
                        merged_state[current_param_id] = copy.deepcopy(
                            saved_state_entries[saved_param_id]
                        )
                        matched_params += 1
                        continue

                    alias_names = optimizer_state_aliases.get(current_name)
                    if not alias_names:
                        continue

                    alias_states = []
                    for alias_name in alias_names:
                        alias_param_id = saved_name_to_param_id.get(alias_name)
                        if alias_param_id is None or alias_param_id not in saved_state_entries:
                            alias_states = []
                            break
                        alias_states.append(saved_state_entries[alias_param_id])
                    if alias_states:
                        merged_state[current_param_id] = merge_optimizer_state_entries(alias_states)
                        matched_params += 1

                merged_group = copy.deepcopy(saved_group)
                merged_group['params'] = current_group['params']
                merged_groups.append(merged_group)

            if matched_params <= 0:
                return False

            optimizer.load_state_dict({
                'state': merged_state,
                'param_groups': merged_groups,
            })
            logging.warning(
                'loaded optimizer state with parameter-name remap; '
                'checkpoint parameter ordering/names differ from the current model'
            )
            return True

    def load_scheduler_and_scaler_states(state):
        if state.get('scheduler') is not None:
            scheduler.load_state_dict(state['scheduler'])
        if state.get('scaler') is not None:
            scaler.load_state_dict(state['scaler'])

    def init_group_metric_dict():
        return {
            'correct': torch.zeros(num_action_groups, dtype=torch.int64, device=device),
            'count': torch.zeros(num_action_groups, dtype=torch.int64, device=device),
        }

    def init_discard_metric_dict():
        return {
            'nll_sum': torch.zeros((), dtype=torch.float64, device=device),
            'top1_correct': torch.zeros((), dtype=torch.int64, device=device),
            'top3_correct': torch.zeros((), dtype=torch.int64, device=device),
            'count': torch.zeros((), dtype=torch.int64, device=device),
        }

    decision_metric_names = (
        'discard_top1',
        'riichi_decision',
        'chi_decision',
        'pon_decision',
        'kan_decision',
        'agari_decision',
        'ryukyoku_decision',
        'call_decision',
        'pass_decision',
    )
    num_decision_metrics = len(decision_metric_names)
    decision_metric_to_idx = {name: idx for idx, name in enumerate(decision_metric_names)}
    sliced_decision_names = (
        'riichi_decision',
        'chi_decision',
        'pon_decision',
        'kan_decision',
        'agari_decision',
    )
    decision_slice_names = (
        'turn_early',
        'turn_mid',
        'turn_late',
        'round_east',
        'round_southplus',
        'role_dealer',
        'role_nondealer',
        'all_last_yes',
        'all_last_no',
        'pressure_threat',
        'pressure_calm',
        'pressure_single_threat',
        'pressure_multi_threat',
        'gap_close_2k',
        'gap_close_4k',
        'gap_up_close_2k',
        'gap_up_close_4k',
        'gap_down_close_2k',
        'gap_down_close_4k',
        'all_last_gap_close_4k',
        'rank_1',
        'rank_2',
        'rank_3',
        'rank_4',
        'all_last_target_keep_first',
        'all_last_target_chase_first',
        'all_last_target_keep_above_fourth',
        'all_last_target_escape_fourth',
        'opp_any_tenpai',
        'opp_multi_tenpai',
        'opp_any_near_tenpai',
        'opp_multi_near_tenpai',
    )
    discard_slice_names = (
        'all_last_yes',
        'pressure_threat',
        'pressure_single_threat',
        'pressure_multi_threat',
        'gap_close_2k',
        'gap_close_4k',
        'gap_up_close_2k',
        'gap_down_close_2k',
        'all_last_gap_close_4k',
        'all_last_target_keep_first',
        'all_last_target_chase_first',
        'all_last_target_keep_above_fourth',
        'all_last_target_escape_fourth',
        'opp_any_tenpai',
        'opp_multi_tenpai',
        'opp_any_near_tenpai',
        'opp_multi_near_tenpai',
        'push_fold_core',
        'push_fold_extreme',
    )

    def init_decision_metric_dict():
        return {
            'correct': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'count': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'tp': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'tn': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'pos_count': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'neg_count': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'pred_pos_count': torch.zeros(num_decision_metrics, dtype=torch.int64, device=device),
            'pos_loss_sum': torch.zeros(num_decision_metrics, dtype=torch.float64, device=device),
            'neg_loss_sum': torch.zeros(num_decision_metrics, dtype=torch.float64, device=device),
        }

    def init_binary_metric_dict():
        return {
            'correct': torch.zeros((), dtype=torch.int64, device=device),
            'count': torch.zeros((), dtype=torch.int64, device=device),
            'tp': torch.zeros((), dtype=torch.int64, device=device),
            'tn': torch.zeros((), dtype=torch.int64, device=device),
            'pos_count': torch.zeros((), dtype=torch.int64, device=device),
            'neg_count': torch.zeros((), dtype=torch.int64, device=device),
            'pred_pos_count': torch.zeros((), dtype=torch.int64, device=device),
            'pos_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'neg_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
        }

    def init_sliced_decision_metric_dict():
        return {
            f'{decision}_{slice_name}': init_binary_metric_dict()
            for decision in sliced_decision_names
            for slice_name in decision_slice_names
        }

    def init_sliced_discard_metric_dict():
        return {
            slice_name: init_discard_metric_dict()
            for slice_name in discard_slice_names
        }

    def init_opponent_metric_dict():
        return {
            'loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'shanten_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'tenpai_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'count': torch.zeros(3, dtype=torch.int64, device=device),
            'shanten_correct': torch.zeros(3, dtype=torch.int64, device=device),
            'tenpai_correct': torch.zeros(3, dtype=torch.int64, device=device),
        }

    def init_danger_loss_dict():
        return {
            'loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'any_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'value_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'player_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
        }

    def init_danger_metric_dict():
        return {
            **init_danger_loss_dict(),
            'any_stats': init_binary_metric_dict(),
            'player_stats': init_binary_metric_dict(),
            'value_pos_count': torch.zeros((), dtype=torch.int64, device=device),
            'value_abs_err_sum': torch.zeros((), dtype=torch.float64, device=device),
            'value_sq_err_sum': torch.zeros((), dtype=torch.float64, device=device),
        }

    empty_opponent_metric_stats = init_opponent_metric_dict()
    empty_danger_loss_stats = init_danger_loss_dict()
    empty_danger_metric_stats = init_danger_metric_dict()

    danger_mix_weights = [
        max(float(danger_any_weight), 0.0),
        max(float(danger_value_weight), 0.0),
        max(float(danger_player_weight), 0.0),
    ]
    if sum(danger_mix_weights) <= 0:
        danger_mix_weights = [0.09042179466099699, 0.8180402859274302, 0.09153791941157279]
    danger_mix_weights = torch.tensor(danger_mix_weights, dtype=torch.float32, device=device)
    danger_mix_weights = danger_mix_weights / danger_mix_weights.sum()

    def balanced_bce_per_sample_with_logits(logits, targets, eligible):
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            reduction='none',
        )
        if danger_focal_gamma > 0:
            probs = logits.sigmoid()
            focal_factor = torch.where(targets, 1 - probs, probs)
            loss = loss * focal_factor.pow(float(danger_focal_gamma))

        reduce_dims = tuple(range(1, loss.ndim))
        pos_mask = eligible & targets
        neg_mask = eligible & ~targets
        pos_weight = pos_mask.to(dtype=loss.dtype)
        neg_weight = neg_mask.to(dtype=loss.dtype)
        pos_count = pos_weight.sum(dim=reduce_dims)
        neg_count = neg_weight.sum(dim=reduce_dims)
        pos_present = (pos_count > 0).to(dtype=loss.dtype)
        neg_present = (neg_count > 0).to(dtype=loss.dtype)
        present_count = pos_present + neg_present
        pos_mean = (loss * pos_weight).sum(dim=reduce_dims) / pos_count.clamp_min(1.0)
        neg_mean = (loss * neg_weight).sum(dim=reduce_dims) / neg_count.clamp_min(1.0)
        per_sample = (pos_mean * pos_present + neg_mean * neg_present) / present_count.clamp_min(1.0)
        eligible_count = eligible.to(dtype=loss.dtype).sum(dim=reduce_dims)
        return torch.where(eligible_count > 0, per_sample, torch.zeros_like(per_sample))

    def update_binary_metric(result, eligible, target_positive, pred_positive, positive_prob):
        target_positive = target_positive & eligible
        target_negative = eligible & ~target_positive
        pred_positive = pred_positive & eligible
        pred_negative = eligible & ~pred_positive

        tp = (pred_positive & target_positive).to(torch.int64).sum()
        tn = (pred_negative & target_negative).to(torch.int64).sum()
        pos_count = target_positive.to(torch.int64).sum()
        neg_count = target_negative.to(torch.int64).sum()
        eligible_count = pos_count + neg_count

        result['count'] = eligible_count
        result['correct'] = tp + tn
        result['tp'] = tp
        result['tn'] = tn
        result['pos_count'] = pos_count
        result['neg_count'] = neg_count
        result['pred_pos_count'] = pred_positive.to(torch.int64).sum()

        positive_prob = positive_prob.clamp(1e-6, 1 - 1e-6)
        result['pos_loss_sum'] = -positive_prob[target_positive].log().sum().to(torch.float64)
        result['neg_loss_sum'] = -torch.log1p(-positive_prob[target_negative]).sum().to(torch.float64)

    def update_decision_metric(result, name, eligible, target_positive, pred_positive, positive_prob):
        idx = decision_metric_to_idx[name]
        stat = init_binary_metric_dict()
        update_binary_metric(stat, eligible, target_positive, pred_positive, positive_prob)
        for key in ('correct', 'count', 'tp', 'tn', 'pos_count', 'neg_count', 'pred_pos_count'):
            result[key][idx] = stat[key]
        for key in ('pos_loss_sum', 'neg_loss_sum'):
            result[key][idx] = stat[key]

    def decode_context_meta(context_meta):
        up_gap_points = context_meta[:, context_meta_specs['up_gap_100']].to(torch.int64) * 100
        down_gap_points = context_meta[:, context_meta_specs['down_gap_100']].to(torch.int64) * 100
        sentinel = torch.iinfo(torch.int64).max
        encoded_missing_gap = 65535 * 100
        up_gap_points = torch.where(
            up_gap_points == encoded_missing_gap,
            torch.full_like(up_gap_points, sentinel),
            up_gap_points,
        )
        down_gap_points = torch.where(
            down_gap_points == encoded_missing_gap,
            torch.full_like(down_gap_points, sentinel),
            down_gap_points,
        )
        nearest_gap_points = torch.minimum(up_gap_points, down_gap_points)
        return {
            'at_turn': context_meta[:, context_meta_specs['at_turn']].to(torch.int64),
            'round_stage': context_meta[:, context_meta_specs['round_stage']].to(torch.int64),
            'is_dealer': context_meta[:, context_meta_specs['is_dealer']].to(torch.bool),
            'is_all_last': context_meta[:, context_meta_specs['is_all_last']].to(torch.bool),
            'self_rank': context_meta[:, context_meta_specs['self_rank']].to(torch.int64),
            'opp_riichi_count': context_meta[:, context_meta_specs['opp_riichi_count']].to(torch.int64),
            'up_gap_points': up_gap_points,
            'down_gap_points': down_gap_points,
            'nearest_gap_points': nearest_gap_points,
        }

    def build_decision_slice_masks(context, opponent_shanten=None, opponent_tenpai=None):
        up_gap_close_2k = context['up_gap_points'] <= 2000
        up_gap_close_4k = context['up_gap_points'] <= 4000
        down_gap_close_2k = context['down_gap_points'] <= 2000
        down_gap_close_4k = context['down_gap_points'] <= 4000
        nearest_gap_close_2k = context['nearest_gap_points'] <= 2000
        nearest_gap_close_4k = context['nearest_gap_points'] <= 4000
        pressure_single_threat = context['opp_riichi_count'] == 1
        pressure_multi_threat = context['opp_riichi_count'] >= 2

        if opponent_tenpai is None:
            opp_any_tenpai = torch.zeros_like(context['is_all_last'])
            opp_multi_tenpai = torch.zeros_like(context['is_all_last'])
        else:
            opponent_tenpai = opponent_tenpai.to(torch.bool)
            opp_any_tenpai = opponent_tenpai.any(dim=1)
            opp_multi_tenpai = opponent_tenpai.sum(dim=1) >= 2

        if opponent_shanten is None:
            opp_any_near_tenpai = torch.zeros_like(context['is_all_last'])
            opp_multi_near_tenpai = torch.zeros_like(context['is_all_last'])
        else:
            near_tenpai = opponent_shanten <= 1
            opp_any_near_tenpai = near_tenpai.any(dim=1)
            opp_multi_near_tenpai = near_tenpai.sum(dim=1) >= 2

        return {
            'turn_early': context['at_turn'] <= 4,
            'turn_mid': (context['at_turn'] >= 5) & (context['at_turn'] <= 11),
            'turn_late': context['at_turn'] >= 12,
            'round_east': context['round_stage'] == 0,
            'round_southplus': context['round_stage'] == 1,
            'role_dealer': context['is_dealer'],
            'role_nondealer': ~context['is_dealer'],
            'all_last_yes': context['is_all_last'],
            'all_last_no': ~context['is_all_last'],
            'pressure_threat': context['opp_riichi_count'] >= 1,
            'pressure_calm': context['opp_riichi_count'] == 0,
            'pressure_single_threat': pressure_single_threat,
            'pressure_multi_threat': pressure_multi_threat,
            'gap_close_2k': nearest_gap_close_2k,
            'gap_close_4k': nearest_gap_close_4k,
            'gap_up_close_2k': up_gap_close_2k,
            'gap_up_close_4k': up_gap_close_4k,
            'gap_down_close_2k': down_gap_close_2k,
            'gap_down_close_4k': down_gap_close_4k,
            'all_last_gap_close_4k': context['is_all_last'] & nearest_gap_close_4k,
            'rank_1': context['self_rank'] == 0,
            'rank_2': context['self_rank'] == 1,
            'rank_3': context['self_rank'] == 2,
            'rank_4': context['self_rank'] == 3,
            'all_last_target_keep_first': context['is_all_last'] & (context['self_rank'] == 0) & down_gap_close_4k,
            'all_last_target_chase_first': context['is_all_last'] & (context['self_rank'] == 1) & up_gap_close_4k,
            'all_last_target_keep_above_fourth': context['is_all_last'] & (context['self_rank'] == 2) & down_gap_close_4k,
            'all_last_target_escape_fourth': context['is_all_last'] & (context['self_rank'] == 3) & up_gap_close_4k,
            'opp_any_tenpai': opp_any_tenpai,
            'opp_multi_tenpai': opp_multi_tenpai,
            'opp_any_near_tenpai': opp_any_near_tenpai,
            'opp_multi_near_tenpai': opp_multi_near_tenpai,
        }

    def build_discard_slice_masks(context, opponent_shanten=None, opponent_tenpai=None):
        slice_masks = build_decision_slice_masks(
            context,
            opponent_shanten=opponent_shanten,
            opponent_tenpai=opponent_tenpai,
        )
        threat_mask = slice_masks['pressure_threat'] | slice_masks['opp_any_tenpai']
        extreme_threat_mask = slice_masks['pressure_multi_threat'] | slice_masks['opp_multi_tenpai']
        micro_gap_mask = slice_masks['gap_close_2k'] | slice_masks['gap_up_close_2k'] | slice_masks['gap_down_close_2k']
        slice_masks.update({
            'push_fold_core': threat_mask & (slice_masks['all_last_yes'] | slice_masks['gap_close_4k']),
            'push_fold_extreme': extreme_threat_mask & (slice_masks['all_last_gap_close_4k'] | micro_gap_mask),
        })
        return {name: slice_masks[name] for name in discard_slice_names}

    def compute_context_turn_weights(context_meta, weighting_cfg):
        return compute_turn_bucket_weights(
            context_meta[:, context_meta_specs['at_turn']],
            early_factor=weighting_cfg['early_factor'],
            mid_factor=weighting_cfg['mid_factor'],
            late_factor=weighting_cfg['late_factor'],
            early_max_turn=weighting_cfg['early_max_turn'],
            late_min_turn=weighting_cfg['late_min_turn'],
        ).to(device=device, non_blocking=True)

    def compute_rank_aux_sample_weights(context_meta):
        weights = torch.full(
            (context_meta.shape[0],),
            float(rank_aux_base_weight),
            dtype=torch.float32,
            device=device,
        )
        weights.mul_(compute_context_turn_weights(context_meta, rank_turn_weighting))
        if rank_aux_south_factor > 0 and rank_aux_south_factor != 1.0:
            weights[context_meta[:, context_meta_specs['round_stage']] == 1] *= float(
                rank_aux_south_factor
            )
        if rank_aux_all_last_factor > 0 and rank_aux_all_last_factor != 1.0:
            weights[context_meta[:, context_meta_specs['is_all_last']].to(torch.bool)] *= float(
                rank_aux_all_last_factor
            )
        if rank_aux_gap_focus_points > 0 and rank_aux_gap_close_bonus > 0:
            nearest_gap = torch.minimum(
                context_meta[:, context_meta_specs['up_gap_100']],
                context_meta[:, context_meta_specs['down_gap_100']],
            ).to(torch.float32)
            nearest_gap.mul_(100.0)
            closeness = (1.0 - nearest_gap / float(rank_aux_gap_focus_points)).clamp_(0.0, 1.0)
            weights.mul_(1.0 + float(rank_aux_gap_close_bonus) * closeness)
        if rank_aux_max_weight > 0:
            weights.clamp_(max=float(rank_aux_max_weight))
        return weights

    def compute_decision_stats(probs, pred_actions, actions, masks):
        result = init_decision_metric_dict()
        probs = probs.detach().to(torch.float32)

        def in_group(action_tensor, start, end):
            return (action_tensor >= start) & (action_tensor < end)

        discard_target = in_group(actions, 0, 37)
        update_decision_metric(
            result,
            'discard_top1',
            discard_target,
            discard_target,
            pred_actions == actions,
            probs.gather(1, actions.unsqueeze(-1)).squeeze(-1),
        )

        group_ranges = {
            'riichi_decision': (37, 38),
            'chi_decision': (38, 41),
            'pon_decision': (41, 42),
            'kan_decision': (42, 43),
            'agari_decision': (43, 44),
            'ryukyoku_decision': (44, 45),
            'call_decision': (38, 43),
        }

        for name, (start, end) in group_ranges.items():
            eligible = masks[:, start:end].any(-1)
            target_positive = in_group(actions, start, end)
            pred_positive = in_group(pred_actions, start, end)
            update_decision_metric(
                result,
                name,
                eligible,
                target_positive,
                pred_positive,
                probs[:, start:end].sum(-1),
            )

        pass_eligible = masks[:, 45] & masks[:, 37:45].any(-1)
        pass_target = actions == 45
        pass_pred = pred_actions == 45
        update_decision_metric(
            result,
            'pass_decision',
            pass_eligible,
            pass_target,
            pass_pred,
            probs[:, 45],
        )

        return result

    def compute_sliced_decision_stats(
        probs,
        pred_actions,
        actions,
        masks,
        context_meta,
        opponent_shanten=None,
        opponent_tenpai=None,
    ):
        result = init_sliced_decision_metric_dict()
        probs = probs.detach().to(torch.float32)
        context = decode_context_meta(context_meta)
        slice_masks = build_decision_slice_masks(
            context,
            opponent_shanten=opponent_shanten,
            opponent_tenpai=opponent_tenpai,
        )

        def in_group(action_tensor, start, end):
            return (action_tensor >= start) & (action_tensor < end)

        group_ranges = {
            'riichi_decision': (37, 38),
            'chi_decision': (38, 41),
            'pon_decision': (41, 42),
            'kan_decision': (42, 43),
            'agari_decision': (43, 44),
        }

        for name, (start, end) in group_ranges.items():
            eligible = masks[:, start:end].any(-1)
            target_positive = in_group(actions, start, end)
            pred_positive = in_group(pred_actions, start, end)
            positive_prob = probs[:, start:end].sum(-1)
            for slice_name, slice_mask in slice_masks.items():
                stat_key = f'{name}_{slice_name}'
                update_binary_metric(
                    result[stat_key],
                    eligible & slice_mask,
                    target_positive,
                    pred_positive,
                    positive_prob,
                )
        return result

    def compute_sliced_discard_stats(
        probs,
        pred_actions,
        actions,
        context_meta,
        opponent_shanten=None,
        opponent_tenpai=None,
    ):
        result = init_sliced_discard_metric_dict()
        probs = probs.detach().to(torch.float32)
        context = decode_context_meta(context_meta)
        slice_masks = build_discard_slice_masks(
            context,
            opponent_shanten=opponent_shanten,
            opponent_tenpai=opponent_tenpai,
        )

        discard_target = (actions >= 0) & (actions < 37)
        if not discard_target.any():
            return result

        for slice_name, slice_mask in slice_masks.items():
            eligible = discard_target & slice_mask
            count = eligible.to(torch.int64).sum()
            if int(count.item()) <= 0:
                continue
            slice_probs = probs[eligible]
            slice_actions = actions[eligible]
            target_probs = slice_probs.gather(1, slice_actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
            topk = slice_probs.topk(k=min(3, slice_probs.shape[-1]), dim=-1).indices
            result[slice_name]['count'] = count
            result[slice_name]['nll_sum'] = -target_probs.log().sum().to(torch.float64)
            result[slice_name]['top1_correct'] = pred_actions[eligible].eq(slice_actions).to(torch.int64).sum()
            result[slice_name]['top3_correct'] = topk.eq(slice_actions.unsqueeze(-1)).any(-1).to(torch.int64).sum()
        return result

    def compute_exact_action_stats(
        probs,
        actions,
        *,
        start,
        end,
        topk_size=3,
        normalize_within_slice=False,
    ):
        return compute_exact_action_metric_stats(
            probs,
            actions,
            start=start,
            end=end,
            topk_size=topk_size,
            normalize_within_slice=normalize_within_slice,
        )

    def compute_discard_stats(probs, pred_actions, actions):
        return compute_exact_action_stats(
            probs,
            actions,
            start=0,
            end=37,
            topk_size=3,
        )

    def compute_chi_exact_stats(probs, pred_actions, actions):
        return compute_exact_action_stats(
            probs,
            actions,
            start=38,
            end=41,
            topk_size=3,
            normalize_within_slice=True,
        )

    def compute_opponent_metrics(
        shanten_logits,
        tenpai_logits,
        opponent_shanten,
        opponent_tenpai,
        sample_weights,
    ):
        result = init_opponent_metric_dict()
        batch_size = opponent_shanten.shape[0]
        if batch_size <= 0:
            return torch.zeros((), dtype=torch.float32, device=device), result

        shanten_losses = []
        tenpai_losses = []
        for idx in range(3):
            shanten_losses.append(
                nn.functional.cross_entropy(
                    shanten_logits[idx],
                    opponent_shanten[:, idx],
                    reduction='none',
                )
            )
            tenpai_losses.append(
                nn.functional.cross_entropy(
                    tenpai_logits[idx],
                    opponent_tenpai[:, idx],
                    reduction='none',
                )
            )
            result['count'][idx] = batch_size
            result['shanten_correct'][idx] = (
                shanten_logits[idx].argmax(-1) == opponent_shanten[:, idx]
            ).to(torch.int64).sum()
            result['tenpai_correct'][idx] = (
                tenpai_logits[idx].argmax(-1) == opponent_tenpai[:, idx]
            ).to(torch.int64).sum()

        shanten_loss_vec = torch.stack(shanten_losses).mean(dim=0)
        tenpai_loss_vec = torch.stack(tenpai_losses).mean(dim=0)
        raw_loss_vec = opponent_shanten_weight * shanten_loss_vec + opponent_tenpai_weight * tenpai_loss_vec
        weighted_loss = opponent_state_weight * (raw_loss_vec * sample_weights).mean()

        result['loss_sum'] = weighted_loss.detach().to(torch.float64) * batch_size
        result['shanten_loss_sum'] = shanten_loss_vec.detach().to(torch.float64).sum()
        result['tenpai_loss_sum'] = tenpai_loss_vec.detach().to(torch.float64).sum()
        return weighted_loss, result

    def compute_danger_metrics(
        any_logits,
        value_pred,
        player_logits,
        masks,
        danger_valid,
        danger_any,
        danger_value,
        danger_player_mask,
        sample_weights,
        *,
        compute_stats=True,
    ):
        result = init_danger_metric_dict() if compute_stats else init_danger_loss_dict()
        batch_size = danger_any.shape[0]
        if batch_size <= 0:
            return torch.zeros((), dtype=torch.float32, device=device), result

        eligible = danger_valid.unsqueeze(-1) & masks[:, :37]
        any_target = danger_any
        any_loss_vec = balanced_bce_per_sample_with_logits(any_logits, any_target, eligible)

        eligible_player = eligible.unsqueeze(-1).expand_as(danger_player_mask)
        player_target = danger_player_mask
        player_loss_vec = balanced_bce_per_sample_with_logits(player_logits, player_target, eligible_player)

        value_positive = eligible & any_target
        positive_value_target = (
            torch.log1p(danger_value.clamp(min=0.0, max=danger_value_cap)) / danger_value_cap_log
        )
        value_loss_map = nn.functional.smooth_l1_loss(
            value_pred.sigmoid(),
            positive_value_target,
            reduction='none',
        )
        value_positive_weight = value_positive.to(dtype=value_loss_map.dtype)
        value_positive_count = value_positive_weight.sum(dim=1)
        value_loss_vec = (value_loss_map * value_positive_weight).sum(dim=1) / value_positive_count.clamp_min(1.0)
        value_loss_vec = torch.where(value_positive_count > 0, value_loss_vec, torch.zeros_like(value_loss_vec))

        raw_loss_vec = (
            danger_mix_weights[0] * any_loss_vec
            + danger_mix_weights[1] * value_loss_vec
            + danger_mix_weights[2] * player_loss_vec
        )
        if danger_ramp_steps > 0:
            ramp = min(float(optimizer_steps) / max(float(danger_ramp_steps), 1.0), 1.0)
        else:
            ramp = 1.0
        overall = max(float(danger_weight), 0.0)
        weighted_loss = (raw_loss_vec * sample_weights).mean() * overall * ramp

        result['loss_sum'] = weighted_loss.detach().to(torch.float64) * batch_size
        result['any_loss_sum'] = any_loss_vec.detach().to(torch.float64).sum()
        result['value_loss_sum'] = value_loss_vec.detach().to(torch.float64).sum()
        result['player_loss_sum'] = player_loss_vec.detach().to(torch.float64).sum()
        if not compute_stats:
            return weighted_loss, result

        any_prob = any_logits.sigmoid()
        any_pred = any_prob >= 0.5
        update_binary_metric(result['any_stats'], eligible, any_target, any_pred, any_prob)

        player_prob = player_logits.sigmoid()
        player_pred = player_prob >= 0.5
        update_binary_metric(result['player_stats'], eligible_player, player_target, player_pred, player_prob)

        result['value_pos_count'] = value_positive.sum(dtype=torch.int64)
        positive_value_pred = value_pred[value_positive].sigmoid()
        positive_value = danger_value[value_positive]
        if positive_value_pred.numel() > 0:
            point_error = torch.expm1(positive_value_pred * danger_value_cap_log) - positive_value
            result['value_abs_err_sum'] = point_error.abs().sum().to(torch.float64)
            result['value_sq_err_sum'] = point_error.square().sum().to(torch.float64)
        return weighted_loss, result

    def compute_action_group_stats(pred_actions, actions):
        group_indices = action_to_group[actions]
        count = torch.bincount(group_indices, minlength=num_action_groups)
        correct = torch.bincount(group_indices[pred_actions == actions], minlength=num_action_groups)
        return {
            'correct': correct,
            'count': count,
        }

    def merge_group_stats(target, source):
        target['correct'] += source['correct']
        target['count'] += source['count']

    def merge_discard_stats(target, source):
        target['nll_sum'] += source['nll_sum']
        target['top1_correct'] += source['top1_correct']
        target['top3_correct'] += source['top3_correct']
        target['count'] += source['count']

    def merge_binary_metric(target, source):
        for key in ('correct', 'count', 'tp', 'tn', 'pos_count', 'neg_count', 'pred_pos_count'):
            target[key] += source[key]
        for key in ('pos_loss_sum', 'neg_loss_sum'):
            target[key] += source[key]

    def merge_decision_stats(target, source):
        target['correct'] += source['correct']
        target['count'] += source['count']
        target['tp'] += source['tp']
        target['tn'] += source['tn']
        target['pos_count'] += source['pos_count']
        target['neg_count'] += source['neg_count']
        target['pred_pos_count'] += source['pred_pos_count']
        target['pos_loss_sum'] += source['pos_loss_sum']
        target['neg_loss_sum'] += source['neg_loss_sum']

    def merge_sliced_decision_stats(target, source):
        for name in target:
            merge_binary_metric(target[name], source[name])

    def merge_sliced_discard_stats(target, source):
        for name in target:
            merge_discard_stats(target[name], source[name])

    def merge_opponent_stats(target, source):
        target['loss_sum'] += source['loss_sum']
        target['shanten_loss_sum'] += source['shanten_loss_sum']
        target['tenpai_loss_sum'] += source['tenpai_loss_sum']
        target['count'] += source['count']
        target['shanten_correct'] += source['shanten_correct']
        target['tenpai_correct'] += source['tenpai_correct']

    def merge_danger_stats(target, source):
        target['loss_sum'] += source['loss_sum']
        target['any_loss_sum'] += source['any_loss_sum']
        target['value_loss_sum'] += source['value_loss_sum']
        target['player_loss_sum'] += source['player_loss_sum']
        merge_binary_metric(target['any_stats'], source['any_stats'])
        merge_binary_metric(target['player_stats'], source['player_stats'])
        target['value_pos_count'] += source['value_pos_count']
        target['value_abs_err_sum'] += source['value_abs_err_sum']
        target['value_sq_err_sum'] += source['value_sq_err_sum']

    def finalize_binary_metric(prefix, stat, output):
        count = int(stat['count'].item())
        if count <= 0:
            return
        pos_count = int(stat['pos_count'].item())
        neg_count = int(stat['neg_count'].item())
        output[f'{prefix}_count'] = count
        output[f'{prefix}_pos_count'] = pos_count
        output[f'{prefix}_neg_count'] = neg_count
        output[f'{prefix}_acc'] = stat['correct'].item() / count
        output[f'{prefix}_pred_rate'] = stat['pred_pos_count'].item() / count
        output[f'{prefix}_target_rate'] = stat['pos_count'].item() / count
        balanced_acc_terms = []
        balanced_bce_terms = []
        if pos_count > 0:
            pos_recall = stat['tp'].item() / pos_count
            pos_bce = stat['pos_loss_sum'].item() / pos_count
            output[f'{prefix}_pos_recall'] = pos_recall
            output[f'{prefix}_pos_bce'] = pos_bce
            balanced_acc_terms.append(pos_recall)
            balanced_bce_terms.append(pos_bce)
        if neg_count > 0:
            neg_recall = stat['tn'].item() / neg_count
            neg_bce = stat['neg_loss_sum'].item() / neg_count
            output[f'{prefix}_neg_recall'] = neg_recall
            output[f'{prefix}_neg_bce'] = neg_bce
            balanced_acc_terms.append(neg_recall)
            balanced_bce_terms.append(neg_bce)
        if balanced_acc_terms:
            output[f'{prefix}_balanced_acc'] = sum(balanced_acc_terms) / len(balanced_acc_terms)
        if balanced_bce_terms:
            output[f'{prefix}_balanced_bce'] = sum(balanced_bce_terms) / len(balanced_bce_terms)

    def log_group_acc(prefix, group_stats, step_or_epoch):
        correct = group_stats['correct'].detach().cpu()
        count = group_stats['count'].detach().cpu()
        for idx, name in enumerate(action_group_names):
            if count[idx].item() <= 0:
                continue
            writer.add_scalar(f'{prefix}/{name}_acc', correct[idx].item() / count[idx].item(), step_or_epoch)

    def log_decision_acc(prefix, decision_stats, step_or_epoch):
        if decision_stats is None:
            return
        correct = decision_stats['correct'].detach().cpu()
        count = decision_stats['count'].detach().cpu()
        for idx, name in enumerate(decision_metric_names):
            if count[idx].item() <= 0:
                continue
            writer.add_scalar(f'{prefix}/{name}_acc', correct[idx].item() / count[idx].item(), step_or_epoch)

    def log_opponent_acc(prefix, opponent_stats, step_or_epoch):
        count = opponent_stats['count'].detach().cpu()
        shanten_correct = opponent_stats['shanten_correct'].detach().cpu()
        tenpai_correct = opponent_stats['tenpai_correct'].detach().cpu()
        for idx in range(3):
            if count[idx].item() <= 0:
                continue
            writer.add_scalar(f'{prefix}/opp{idx + 1}_shanten_acc', shanten_correct[idx].item() / count[idx].item(), step_or_epoch)
            writer.add_scalar(f'{prefix}/opp{idx + 1}_tenpai_acc', tenpai_correct[idx].item() / count[idx].item(), step_or_epoch)

    def log_danger_metrics(prefix, danger_stats, step_or_epoch):
        if danger_stats is None:
            return
        finalized = {}
        finalize_binary_metric('danger_any', danger_stats['any_stats'], finalized)
        finalize_binary_metric('danger_player', danger_stats['player_stats'], finalized)
        count = int(danger_stats['value_pos_count'].item())
        if count > 0:
            finalized['danger_value_mae'] = danger_stats['value_abs_err_sum'].item() / count
            finalized['danger_value_rmse'] = math.sqrt(danger_stats['value_sq_err_sum'].item() / count)
        for metric_name, value in finalized.items():
            writer.add_scalar(f'{prefix}/{metric_name}', value, step_or_epoch)

    def log_sliced_decision_metrics(prefix, sliced_stats, step_or_epoch):
        if sliced_stats is None:
            return
        for name, stat in sliced_stats.items():
            finalized = {}
            finalize_binary_metric(name, stat, finalized)
            for metric_name, value in finalized.items():
                writer.add_scalar(f'{prefix}/{metric_name}', value, step_or_epoch)

    def log_sliced_discard_metrics(prefix, sliced_stats, step_or_epoch):
        if sliced_stats is None:
            return
        for name, stat in sliced_stats.items():
            count = int(stat['count'].item())
            if count <= 0:
                continue
            writer.add_scalar(f'{prefix}/{name}_nll', stat['nll_sum'].item() / count, step_or_epoch)
            writer.add_scalar(f'{prefix}/{name}_top1_acc', stat['top1_correct'].item() / count, step_or_epoch)
            writer.add_scalar(f'{prefix}/{name}_top3_acc', stat['top3_correct'].item() / count, step_or_epoch)

    def init_metric_dict(*, include_detailed_metrics=True, include_sliced_metrics=True):
        return {
            'loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'policy_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'aux_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'rank_aux_raw_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'rank_aux_weight_sum': torch.zeros((), dtype=torch.float64, device=device),
            'opponent_turn_weight_sum': torch.zeros((), dtype=torch.float64, device=device),
            'opponent_aux_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'danger_turn_weight_sum': torch.zeros((), dtype=torch.float64, device=device),
            'danger_aux_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'danger_any_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'danger_value_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'danger_player_loss_sum': torch.zeros((), dtype=torch.float64, device=device),
            'action_correct': torch.zeros((), dtype=torch.int64, device=device),
            'rank_correct': torch.zeros((), dtype=torch.int64, device=device),
            'count': 0,
            'discard_stats': init_discard_metric_dict() if include_detailed_metrics else None,
            'chi_exact_stats': init_discard_metric_dict() if include_detailed_metrics else None,
            'group_stats': init_group_metric_dict(),
            'decision_stats': init_decision_metric_dict() if include_detailed_metrics else None,
            'sliced_decision_stats': init_sliced_decision_metric_dict() if include_sliced_metrics else None,
            'sliced_discard_stats': init_sliced_discard_metric_dict() if include_sliced_metrics else None,
            'opponent_stats': init_opponent_metric_dict(),
            'danger_stats': init_danger_metric_dict() if include_detailed_metrics else None,
        }

    def merge_metrics(target, source):
        target['loss_sum'] += source['loss_sum']
        target['policy_loss_sum'] += source['policy_loss_sum']
        target['aux_loss_sum'] += source['aux_loss_sum']
        target['rank_aux_raw_loss_sum'] += source['rank_aux_raw_loss_sum']
        target['rank_aux_weight_sum'] += source['rank_aux_weight_sum']
        target['opponent_turn_weight_sum'] += source['opponent_turn_weight_sum']
        target['opponent_aux_loss_sum'] += source['opponent_aux_loss_sum']
        target['danger_turn_weight_sum'] += source['danger_turn_weight_sum']
        target['danger_aux_loss_sum'] += source['danger_aux_loss_sum']
        target['danger_any_loss_sum'] += source['danger_any_loss_sum']
        target['danger_value_loss_sum'] += source['danger_value_loss_sum']
        target['danger_player_loss_sum'] += source['danger_player_loss_sum']
        target['action_correct'] += source['action_correct']
        target['rank_correct'] += source['rank_correct']
        target['count'] += source['count']
        if target['discard_stats'] is not None and source['discard_stats'] is not None:
            merge_discard_stats(target['discard_stats'], source['discard_stats'])
        if target['chi_exact_stats'] is not None and source['chi_exact_stats'] is not None:
            merge_discard_stats(target['chi_exact_stats'], source['chi_exact_stats'])
        merge_group_stats(target['group_stats'], source['group_stats'])
        if target['decision_stats'] is not None and source['decision_stats'] is not None:
            merge_decision_stats(target['decision_stats'], source['decision_stats'])
        if target['sliced_decision_stats'] is not None and source['sliced_decision_stats'] is not None:
            merge_sliced_decision_stats(target['sliced_decision_stats'], source['sliced_decision_stats'])
        if target['sliced_discard_stats'] is not None and source['sliced_discard_stats'] is not None:
            merge_sliced_discard_stats(target['sliced_discard_stats'], source['sliced_discard_stats'])
        merge_opponent_stats(target['opponent_stats'], source['opponent_stats'])
        if target['danger_stats'] is not None and source['danger_stats'] is not None:
            merge_danger_stats(target['danger_stats'], source['danger_stats'])

    def finalize_metrics(metrics):
        denom = max(metrics['count'], 1)
        group_correct = metrics['group_stats']['correct'].detach().cpu()
        group_count = metrics['group_stats']['count'].detach().cpu()
        group_acc_values = [
            group_correct[idx].item() / group_count[idx].item()
            for idx in range(num_action_groups)
            if group_count[idx].item() > 0
        ]
        macro_action_acc = sum(group_acc_values) / max(len(group_acc_values), 1)
        finalized = {
            'loss': metrics['loss_sum'].item() / denom,
            'policy_loss': metrics['policy_loss_sum'].item() / denom,
            'aux_loss': metrics['aux_loss_sum'].item() / denom,
            'rank_aux_raw_loss': metrics['rank_aux_raw_loss_sum'].item() / denom,
            'rank_aux_weight_mean': metrics['rank_aux_weight_sum'].item() / denom,
            'opponent_turn_weight_mean': metrics['opponent_turn_weight_sum'].item() / denom,
            'opponent_aux_loss': metrics['opponent_aux_loss_sum'].item() / denom,
            'danger_turn_weight_mean': metrics['danger_turn_weight_sum'].item() / denom,
            'danger_aux_loss': metrics['danger_aux_loss_sum'].item() / denom,
            'danger_any_loss': metrics['danger_any_loss_sum'].item() / denom,
            'danger_value_loss': metrics['danger_value_loss_sum'].item() / denom,
            'danger_player_loss': metrics['danger_player_loss_sum'].item() / denom,
            'action_acc': metrics['action_correct'].item() / denom,
            'rank_acc': metrics['rank_correct'].item() / denom,
            'macro_action_acc': macro_action_acc,
        }
        discard_stats = metrics['discard_stats']
        if discard_stats is not None:
            discard_count = int(discard_stats['count'].item())
            if discard_count > 0:
                finalized['discard_count'] = discard_count
                finalized['discard_nll'] = discard_stats['nll_sum'].item() / discard_count
                finalized['discard_top1_acc'] = discard_stats['top1_correct'].item() / discard_count
                finalized['discard_top3_acc'] = discard_stats['top3_correct'].item() / discard_count
        chi_exact_stats = metrics['chi_exact_stats']
        if chi_exact_stats is not None:
            chi_exact_count = int(chi_exact_stats['count'].item())
            if chi_exact_count > 0:
                finalized['chi_exact_count'] = chi_exact_count
                finalized['chi_exact_nll'] = chi_exact_stats['nll_sum'].item() / chi_exact_count
                finalized['chi_exact_top1_acc'] = chi_exact_stats['top1_correct'].item() / chi_exact_count
                finalized['chi_exact_top3_acc'] = chi_exact_stats['top3_correct'].item() / chi_exact_count
        if metrics['decision_stats'] is not None:
            decision_correct = metrics['decision_stats']['correct'].detach().cpu()
            decision_count = metrics['decision_stats']['count'].detach().cpu()
            decision_tp = metrics['decision_stats']['tp'].detach().cpu()
            decision_tn = metrics['decision_stats']['tn'].detach().cpu()
            decision_pos_count = metrics['decision_stats']['pos_count'].detach().cpu()
            decision_neg_count = metrics['decision_stats']['neg_count'].detach().cpu()
            decision_pos_loss_sum = metrics['decision_stats']['pos_loss_sum'].detach().cpu()
            decision_neg_loss_sum = metrics['decision_stats']['neg_loss_sum'].detach().cpu()
            for idx, name in enumerate(decision_metric_names):
                if decision_count[idx].item() <= 0:
                    continue
                pos_count = decision_pos_count[idx].item()
                neg_count = decision_neg_count[idx].item()
                finalized[f'{name}_count'] = decision_count[idx].item()
                finalized[f'{name}_pos_count'] = pos_count
                finalized[f'{name}_neg_count'] = neg_count
                finalized[f'{name}_acc'] = decision_correct[idx].item() / decision_count[idx].item()
                if pos_count > 0:
                    finalized[f'{name}_pos_recall'] = decision_tp[idx].item() / pos_count
                    finalized[f'{name}_pos_bce'] = decision_pos_loss_sum[idx].item() / pos_count
                if neg_count > 0:
                    finalized[f'{name}_neg_recall'] = decision_tn[idx].item() / neg_count
                    finalized[f'{name}_neg_bce'] = decision_neg_loss_sum[idx].item() / neg_count
                balanced_acc_terms = []
                balanced_bce_terms = []
                if pos_count > 0:
                    balanced_acc_terms.append(finalized[f'{name}_pos_recall'])
                    balanced_bce_terms.append(finalized[f'{name}_pos_bce'])
                if neg_count > 0:
                    balanced_acc_terms.append(finalized[f'{name}_neg_recall'])
                    balanced_bce_terms.append(finalized[f'{name}_neg_bce'])
                if balanced_acc_terms:
                    finalized[f'{name}_balanced_acc'] = sum(balanced_acc_terms) / len(balanced_acc_terms)
                if balanced_bce_terms:
                    finalized[f'{name}_balanced_bce'] = sum(balanced_bce_terms) / len(balanced_bce_terms)
        opponent_stats = metrics['opponent_stats']
        opponent_count = opponent_stats['count'].detach().cpu()
        opponent_shanten_correct = opponent_stats['shanten_correct'].detach().cpu()
        opponent_tenpai_correct = opponent_stats['tenpai_correct'].detach().cpu()
        shanten_accs = []
        tenpai_accs = []
        for idx in range(3):
            count = opponent_count[idx].item()
            if count <= 0:
                continue
            shanten_acc = opponent_shanten_correct[idx].item() / count
            tenpai_acc = opponent_tenpai_correct[idx].item() / count
            finalized[f'opp{idx + 1}_shanten_acc'] = shanten_acc
            finalized[f'opp{idx + 1}_tenpai_acc'] = tenpai_acc
            shanten_accs.append(shanten_acc)
            tenpai_accs.append(tenpai_acc)
        if shanten_accs:
            finalized['opponent_shanten_macro_acc'] = sum(shanten_accs) / len(shanten_accs)
            finalized['opponent_shanten_loss'] = opponent_stats['shanten_loss_sum'].item() / denom
        if tenpai_accs:
            finalized['opponent_tenpai_macro_acc'] = sum(tenpai_accs) / len(tenpai_accs)
            finalized['opponent_tenpai_loss'] = opponent_stats['tenpai_loss_sum'].item() / denom
        if metrics['danger_stats'] is not None:
            danger_summary = {}
            finalize_binary_metric('danger_any', metrics['danger_stats']['any_stats'], danger_summary)
            finalize_binary_metric('danger_player', metrics['danger_stats']['player_stats'], danger_summary)
            value_count = int(metrics['danger_stats']['value_pos_count'].item())
            if value_count > 0:
                danger_summary['danger_value_mae'] = metrics['danger_stats']['value_abs_err_sum'].item() / value_count
                danger_summary['danger_value_rmse'] = math.sqrt(
                    metrics['danger_stats']['value_sq_err_sum'].item() / value_count
                )
            finalized.update(danger_summary)
        if metrics['sliced_decision_stats'] is not None:
            sliced_summary = {}
            for name, stat in metrics['sliced_decision_stats'].items():
                finalize_binary_metric(name, stat, sliced_summary)
            finalized.update(sliced_summary)
        if metrics['sliced_discard_stats'] is not None:
            for name, stat in metrics['sliced_discard_stats'].items():
                count = int(stat['count'].item())
                if count <= 0:
                    continue
                finalized[f'discard_{name}_count'] = count
                finalized[f'discard_{name}_nll'] = stat['nll_sum'].item() / count
                finalized[f'discard_{name}_top1_acc'] = stat['top1_correct'].item() / count
                finalized[f'discard_{name}_top3_acc'] = stat['top3_correct'].item() / count
        if metrics['discard_stats'] is not None and metrics['decision_stats'] is not None:
            finalized['action_quality_score'] = action_quality_score(finalized)
        if metrics['sliced_decision_stats'] is not None:
            refresh_scenario_quality_score(finalized)
        if metrics['discard_stats'] is not None and metrics['decision_stats'] is not None:
            refresh_selection_quality_score(finalized)
        return finalized

    def build_file_lists():
        player_names_set = set()
        for filename in config['dataset'].get('player_names_files', []):
            with open(filename, encoding='utf-8') as f:
                player_names_set.update(filtered_trimmed_lines(f))

        if path.exists(file_index):
            index = torch.load(file_index, weights_only=True)
            if 'train_files' in index:
                train_files = index['train_files']
                fallback_val_files = index.get('val_files', [])
                monitor_recent_files = index.get('monitor_recent_files', fallback_val_files)
                full_recent_files = index.get('full_recent_files', fallback_val_files)
                old_regression_files = index.get('old_regression_files', [])
                return train_files, monitor_recent_files, full_recent_files, old_regression_files

            logging.info(f'building {checkpoint_label} file index...')
        file_list = []
        for pat in config['dataset']['globs']:
            file_list.extend(glob(pat, recursive=True))

        if player_names_set:
            filtered = []
            for filename in tqdm(file_list, unit='file'):
                with gzip.open(filename, 'rt', encoding='utf-8') as f:
                    start = json.loads(next(f))
                    if not set(start['names']).isdisjoint(player_names_set):
                        filtered.append(filename)
            file_list = filtered

        file_list.sort()
        rng = random.Random(seed)
        rng.shuffle(file_list)

        if len(file_list) < 2:
            raise RuntimeError('not enough files to split train/val')

        val_count = max(int(len(file_list) * val_ratio), min_val_files)
        val_count = min(val_count, len(file_list) - 1)

        val_files = file_list[:val_count]
        train_files = file_list[val_count:]

        if max_train_files > 0:
            train_files = train_files[:max_train_files]
        if max_val_files > 0:
            val_files = val_files[:max_val_files]

        torch.save(
            {
                'train_files': train_files,
                'val_files': val_files,
                'monitor_recent_files': val_files,
                'full_recent_files': val_files,
                'old_regression_files': [],
                'seed': seed,
                'val_ratio': val_ratio,
            },
            file_index,
        )
        return train_files, val_files, val_files, []

    train_files, monitor_recent_files, full_recent_files, old_regression_files = build_file_lists()
    logging.info(f'train files: {len(train_files):,}')
    logging.info(f'monitor recent files: {len(monitor_recent_files):,}')
    logging.info(f'full recent files: {len(full_recent_files):,}')
    logging.info(f'old regression files: {len(old_regression_files):,}')

    def build_loader(file_list, *, training, shuffle_files, safe_train=False):
        phase_file_batch_size = file_batch_size if training else val_file_batch_size
        phase_num_workers = num_workers
        phase_prefetch_factor = prefetch_factor if training else val_prefetch_factor
        phase_in_order = train_in_order if training else val_in_order
        phase_pin_memory = True
        phase_persistent_workers = training
        phase_rayon_num_threads = rayon_num_threads

        if training and safe_train:
            phase_file_batch_size = min(phase_file_batch_size, 4)
            phase_num_workers = min(max(phase_num_workers, 1), 2)
            phase_prefetch_factor = 1
            phase_pin_memory = False
            phase_persistent_workers = False
            phase_rayon_num_threads = min(max(phase_rayon_num_threads, 1), 2)

        if phase_rayon_num_threads > 0:
            os.environ['RAYON_NUM_THREADS'] = str(phase_rayon_num_threads)
        phase_emit_opponent_state_labels = enable_opponent_state_aux or (
            not training and emit_opponent_state_metric_labels
        )
        phase_use_oracle = loader_uses_oracle(
            training=training,
            use_oracle=use_oracle,
            validation_use_oracle=validation_use_oracle,
        )
        dataset = SupervisedFileDatasetsIter(
            version=version,
            file_list=list(file_list),
            oracle=phase_use_oracle,
            file_batch_size=phase_file_batch_size,
            reserve_ratio=reserve_ratio if training else 0.0,
            player_names=None,
            num_epochs=1,
            enable_augmentation=enable_augmentation if training else False,
            augmented_first=augmented_first if training else False,
            shuffle_files=shuffle_files,
            worker_torch_num_threads=worker_torch_num_threads,
            worker_torch_num_interop_threads=worker_torch_num_interop_threads,
            rayon_num_threads=phase_rayon_num_threads,
            emit_opponent_state_labels=phase_emit_opponent_state_labels,
            track_danger_labels=enable_danger_aux,
        )
        kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'drop_last': False,
            'num_workers': phase_num_workers,
            'pin_memory': phase_pin_memory,
            'worker_init_fn': worker_init_fn,
            'collate_fn': safe_default_collate,
        }
        if phase_num_workers > 0:
            kwargs['prefetch_factor'] = phase_prefetch_factor
            kwargs['persistent_workers'] = phase_persistent_workers
            if 'in_order' in inspect.signature(DataLoader.__init__).parameters:
                kwargs['in_order'] = phase_in_order
        return DataLoader(**kwargs)

    def move_batch_to_device(batch):
        (
            obs,
            oracle_obs,
            actions,
            masks,
            player_rank,
            context_meta,
            opponent_shanten,
            opponent_tenpai,
            danger_valid,
            danger_any,
            danger_value,
            danger_player_mask,
        ) = unpack_batch(batch)
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        if not torch.is_tensor(masks):
            masks = torch.as_tensor(masks)
        if not torch.is_tensor(player_rank):
            player_rank = torch.as_tensor(player_rank)
        if not torch.is_tensor(context_meta):
            context_meta = torch.as_tensor(context_meta)
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
        player_rank = player_rank.to(dtype=torch.int64, device=device, non_blocking=True)
        context_meta = context_meta.to(dtype=torch.int64, device=device, non_blocking=True)
        if oracle_obs is not None:
            if not torch.is_tensor(oracle_obs):
                oracle_obs = torch.as_tensor(oracle_obs)
            oracle_obs = oracle_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        if opponent_shanten is not None:
            if not torch.is_tensor(opponent_shanten):
                opponent_shanten = torch.as_tensor(opponent_shanten)
            opponent_shanten = opponent_shanten.to(dtype=torch.int64, device=device, non_blocking=True)
        if opponent_tenpai is not None:
            if not torch.is_tensor(opponent_tenpai):
                opponent_tenpai = torch.as_tensor(opponent_tenpai)
            opponent_tenpai = opponent_tenpai.to(dtype=torch.int64, device=device, non_blocking=True)
        if danger_valid is not None:
            if not torch.is_tensor(danger_valid):
                danger_valid = torch.as_tensor(danger_valid)
            danger_valid = danger_valid.to(dtype=torch.bool, device=device, non_blocking=True)
        if danger_any is not None:
            if not torch.is_tensor(danger_any):
                danger_any = torch.as_tensor(danger_any)
            danger_any = danger_any.to(dtype=torch.bool, device=device, non_blocking=True)
        if danger_value is not None:
            if not torch.is_tensor(danger_value):
                danger_value = torch.as_tensor(danger_value)
            danger_value = danger_value.to(dtype=torch.float32, device=device, non_blocking=True)
        if danger_player_mask is not None:
            if not torch.is_tensor(danger_player_mask):
                danger_player_mask = torch.as_tensor(danger_player_mask)
            danger_player_mask = danger_player_mask.to(dtype=torch.bool, device=device, non_blocking=True)
        return (
            obs,
            oracle_obs,
            actions,
            masks,
            player_rank,
            context_meta,
            opponent_shanten,
            opponent_tenpai,
            danger_valid,
            danger_any,
            danger_value,
            danger_player_mask,
        )

    def record_cuda_stream(batch):
        if device.type != 'cuda':
            return
        current_stream = torch.cuda.current_stream(device)

        def record(value):
            if torch.is_tensor(value):
                if value.device.type == 'cuda':
                    value.record_stream(current_stream)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    record(item)

        record(batch)

    class CudaPrefetcher:
        def __init__(self, loader):
            self.base_loader = loader
            self.loader = iter(loader)
            self.stream = torch.cuda.Stream(device=device)
            self.next_batch = None
            self.started = False
            self.closed = False

        def preload(self):
            if self.closed:
                self.next_batch = None
                return
            try:
                batch = next(self.loader)
            except StopIteration:
                self.next_batch = None
                return
            with torch.cuda.stream(self.stream):
                self.next_batch = move_batch_to_device(batch)

        def __iter__(self):
            return self

        def __next__(self):
            if not self.started:
                self.started = True
                self.preload()
            if self.next_batch is None:
                self.close()
                raise StopIteration
            torch.cuda.current_stream(device).wait_stream(self.stream)
            batch = self.next_batch
            record_cuda_stream(batch)
            self.preload()
            return batch

        def close(self):
            if self.closed:
                return
            self.closed = True
            self.next_batch = None
            loader_iter = self.loader
            self.loader = None
            if loader_iter is not None:
                shutdown = getattr(loader_iter, '_shutdown_workers', None)
                if callable(shutdown):
                    shutdown()

    def close_batch_iter(batch_iter):
        if batch_iter is None:
            return
        close = getattr(batch_iter, 'close', None)
        if callable(close):
            close()
            return
        shutdown = getattr(batch_iter, '_shutdown_workers', None)
        if callable(shutdown):
            shutdown()

    def build_batch_iter(file_list, *, training, shuffle_files, safe_train=False):
        loader = build_loader(
            file_list,
            training=training,
            shuffle_files=shuffle_files,
            safe_train=safe_train,
        )
        return make_closeable_batch_iter(
            loader,
            enable_cuda_prefetch=enable_cuda_prefetch,
            prefetcher_factory=CudaPrefetcher,
        )

    steps = 0
    optimizer_steps = 0
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_action_acc = 0.0
    best_val_action_score = float('-inf')
    best_val_rank_acc = 0.0
    best_full_recent_loss = float('inf')
    best_full_recent_action_acc = 0.0
    best_full_recent_action_score = float('-inf')
    best_full_recent_rank_acc = 0.0
    patience_counter = 0
    num_lr_reductions = 0
    validation_checks = 0
    full_validation_checks = 0
    old_regression_checks = 0
    last_monitor_recent_metrics = None
    last_full_recent_metrics = None
    last_old_regression_metrics = None

    def reset_resume_validation_history():
        nonlocal best_val_loss
        nonlocal best_val_action_acc
        nonlocal best_val_action_score
        nonlocal best_val_rank_acc
        nonlocal best_full_recent_loss
        nonlocal best_full_recent_action_acc
        nonlocal best_full_recent_action_score
        nonlocal best_full_recent_rank_acc
        nonlocal patience_counter
        nonlocal num_lr_reductions
        nonlocal validation_checks
        nonlocal full_validation_checks
        nonlocal old_regression_checks
        nonlocal last_monitor_recent_metrics
        nonlocal last_full_recent_metrics
        nonlocal last_old_regression_metrics

        best_val_loss = float('inf')
        best_val_action_acc = 0.0
        best_val_action_score = float('-inf')
        best_val_rank_acc = 0.0
        best_full_recent_loss = float('inf')
        best_full_recent_action_acc = 0.0
        best_full_recent_action_score = float('-inf')
        best_full_recent_rank_acc = 0.0
        patience_counter = 0
        num_lr_reductions = 0
        validation_checks = 0
        full_validation_checks = 0
        old_regression_checks = 0
        last_monitor_recent_metrics = None
        last_full_recent_metrics = None
        last_old_regression_metrics = None

    if path.exists(state_file):
        state = torch.load(state_file, weights_only=False, map_location=device)
        validate_exact_resume_heads(state)
        load_optional_head_states(state)
        optimizer_loaded = load_optimizer_state_compat(
            state.get('optimizer'),
            state.get('optimizer_param_groups'),
        )
        loaded_epoch_complete = state.get('epoch_complete', True)
        if not optimizer_loaded:
            raise RuntimeError(
                f'{cfg_prefix}.state_file is for exact resume only, but the saved '
                'optimizer/scheduler/scaler state is missing or incompatible with '
                f'the current model configuration. Use {cfg_prefix}.init_state_file '
                'for weights-only initialization, or remove the stale resume '
                'checkpoint and start a fresh run.'
            )
        load_scheduler_and_scaler_states(state)
        steps = state['steps']
        optimizer_steps = resume_optimizer_steps_from_state(
            state,
            opt_step_every=opt_step_every,
            default=optimizer_steps,
        )
        epoch_complete = loaded_epoch_complete
        start_epoch = state['epoch'] + 1 if epoch_complete else state['epoch']
        best_val_loss = state.get('best_val_loss', best_val_loss)
        best_val_action_acc = state.get('best_val_action_acc', best_val_action_acc)
        best_val_action_score = state.get('best_val_action_score', best_val_action_score)
        best_val_rank_acc = state.get('best_val_rank_acc', best_val_rank_acc)
        best_full_recent_loss = state.get('best_full_recent_loss', best_full_recent_loss)
        best_full_recent_action_acc = state.get(
            'best_full_recent_action_acc',
            best_full_recent_action_acc,
        )
        best_full_recent_action_score = state.get(
            'best_full_recent_action_score',
            best_full_recent_action_score,
        )
        best_full_recent_rank_acc = state.get(
            'best_full_recent_rank_acc',
            best_full_recent_rank_acc,
        )
        patience_counter = state.get('patience_counter', 0)
        num_lr_reductions = state.get('num_lr_reductions', 0)
        validation_checks = state.get('validation_checks', 0)
        full_validation_checks = state.get('full_validation_checks', 0)
        old_regression_checks = state.get('old_regression_checks', 0)
        last_monitor_recent_metrics = state.get('last_monitor_recent_metrics')
        last_full_recent_metrics = state.get('last_full_recent_metrics')
        last_old_regression_metrics = state.get('last_old_regression_metrics')
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(
            f'loaded {checkpoint_label} checkpoint: {timestamp}; '
            f'epoch={state["epoch"] + 1} epoch_complete={epoch_complete} '
            f'start_epoch={start_epoch + 1} '
            f'validation_checks={validation_checks}'
        )
    elif init_state_file:
        ensure_init_state_file_exists(init_state_file, cfg_prefix=cfg_prefix)
        state = torch.load(init_state_file, weights_only=False, map_location=device)
        bridge_info = load_brain_state_with_input_bridge(mortal, state['mortal'])
        policy_net.load_state_dict(state['policy_net'])
        if state.get('aux_net') is not None:
            aux_net.load_state_dict(state['aux_net'])
        if opponent_aux_net is not None and state.get('opponent_aux_net') is not None:
            opponent_aux_net.load_state_dict(state['opponent_aux_net'])
        if danger_aux_net is not None and state.get('danger_aux_net') is not None:
            danger_aux_net.load_state_dict(state['danger_aux_net'])
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(
            f'initialized {checkpoint_label} weights from checkpoint: {init_state_file} ({timestamp}); '
            f'brain bridge loaded={len(bridge_info["loaded_keys"])} '
            f'skipped={len(bridge_info["skipped_keys"])}'
        )

    optimizer.zero_grad(set_to_none=True)
    writer = SummaryWriter(tensorboard_dir)

    def update_warmup_lr():
        nonlocal optimizer_steps
        if scheduler_type != 'plateau':
            return
        if warm_up_steps <= 0 or optimizer_steps > warm_up_steps:
            return
        progress = optimizer_steps / max(warm_up_steps, 1)
        lr = warmup_init + (peak_lr - warmup_init) * progress
        for group in optimizer.param_groups:
            group['lr'] = lr

    def unpack_batch(batch):
        idx = 0
        obs = batch[idx]
        idx += 1
        if batch_includes_oracle(len(batch), enable_danger_aux=enable_danger_aux):
            oracle_obs = batch[idx]
            idx += 1
        else:
            oracle_obs = None
        actions = batch[idx]
        masks = batch[idx + 1]
        player_rank = batch[idx + 2]
        context_meta = batch[idx + 3]
        idx += 4
        remaining = len(batch) - idx
        expected_without_opponent = 4 if enable_danger_aux else 0
        if remaining == expected_without_opponent + 2:
            batch_has_opponent_state_labels = True
        elif remaining == expected_without_opponent:
            batch_has_opponent_state_labels = False
        else:
            raise ValueError(
                f'unexpected {checkpoint_label} batch length: '
                f'expected remaining {expected_without_opponent} or {expected_without_opponent + 2}, '
                f'got {remaining}'
            )
        if batch_has_opponent_state_labels:
            opponent_shanten = batch[idx]
            opponent_tenpai = batch[idx + 1]
            idx += 2
        else:
            opponent_shanten = None
            opponent_tenpai = None
        if enable_danger_aux:
            danger_valid = batch[idx]
            danger_any = batch[idx + 1]
            danger_value = batch[idx + 2]
            danger_player_mask = batch[idx + 3]
            idx += 4
        else:
            danger_valid = None
            danger_any = None
            danger_value = None
            danger_player_mask = None
        if idx != len(batch):
            raise ValueError(
                f'unexpected {checkpoint_label} batch length: expected {idx}, got {len(batch)}'
            )
        return (
            obs,
            oracle_obs,
            actions,
            masks,
            player_rank,
            context_meta,
            opponent_shanten,
            opponent_tenpai,
            danger_valid,
            danger_any,
            danger_value,
            danger_player_mask,
        )

    def forward_loss(
        batch,
        *,
        batch_on_device=False,
        compute_detailed_metrics=True,
        compute_sliced_metrics=True,
        training=False,
    ):
        if batch_on_device:
            (
                obs,
                oracle_obs,
                actions,
                masks,
                player_rank,
                context_meta,
                opponent_shanten,
                opponent_tenpai,
                danger_valid,
                danger_any,
                danger_value,
                danger_player_mask,
            ) = batch
        else:
            (
                obs,
                oracle_obs,
                actions,
                masks,
                player_rank,
                context_meta,
                opponent_shanten,
                opponent_tenpai,
                danger_valid,
                danger_any,
                danger_value,
                danger_player_mask,
            ) = move_batch_to_device(batch)
        oracle_input, oracle_gamma = prepare_oracle_obs(oracle_obs, obs=obs, training=training)

        with torch.autocast(device.type, enabled=enable_amp):
            phi = mortal(obs, invisible_obs=oracle_input) if oracle_input is not None else mortal(obs)
            policy_logits = policy_net.logits(phi, masks)
            policy_loss = nn.functional.cross_entropy(policy_logits, actions)
            probs = (
                policy_logits.softmax(-1)
                if compute_detailed_metrics or compute_sliced_metrics else None
            )
            rank_logits = aux_net(phi)[0]
            rank_aux_weights = compute_rank_aux_sample_weights(context_meta)
            raw_aux_loss_vec = nn.functional.cross_entropy(rank_logits, player_rank, reduction='none')
            raw_aux_loss = raw_aux_loss_vec.mean()
            aux_loss = (raw_aux_loss_vec * rank_aux_weights).mean()
            opponent_turn_weights = torch.zeros((obs.shape[0],), dtype=torch.float32, device=device)
            opponent_aux_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
            opponent_metric_stats = (
                init_opponent_metric_dict()
                if opponent_aux_net is not None else empty_opponent_metric_stats
            )
            danger_turn_weights = torch.zeros((obs.shape[0],), dtype=torch.float32, device=device)
            danger_aux_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
            if danger_aux_net is not None:
                danger_metric_stats = (
                    init_danger_metric_dict() if compute_detailed_metrics else init_danger_loss_dict()
                )
            else:
                danger_metric_stats = (
                    empty_danger_metric_stats if compute_detailed_metrics else empty_danger_loss_stats
                )
            if opponent_aux_net is not None:
                if opponent_shanten is None or opponent_tenpai is None:
                    raise RuntimeError('opponent-state auxiliary head enabled but labels are missing')
                shanten_logits, tenpai_logits = opponent_aux_net(phi)
                opponent_turn_weights = compute_context_turn_weights(context_meta, opponent_turn_weighting)
                opponent_aux_loss, opponent_metric_stats = compute_opponent_metrics(
                    shanten_logits,
                    tenpai_logits,
                    opponent_shanten,
                    opponent_tenpai,
                    opponent_turn_weights,
                )
            if danger_aux_net is not None:
                if danger_valid is None or danger_any is None or danger_value is None or danger_player_mask is None:
                    raise RuntimeError('danger auxiliary head enabled but labels are missing')
                danger_any_logits, danger_value_pred, danger_player_logits = danger_aux_net(phi)
                danger_turn_weights = compute_context_turn_weights(context_meta, danger_turn_weighting)
                danger_aux_loss, danger_metric_stats = compute_danger_metrics(
                    danger_any_logits,
                    danger_value_pred,
                    danger_player_logits,
                    masks,
                    danger_valid,
                    danger_any,
                    danger_value,
                    danger_player_mask,
                    danger_turn_weights,
                    compute_stats=compute_detailed_metrics,
                )
            total_loss = policy_loss + aux_loss + opponent_aux_loss + danger_aux_loss

        with torch.inference_mode():
            pred_actions = policy_logits.argmax(-1)
            rank_preds = rank_logits.argmax(-1)
            action_matches = pred_actions == actions
            rank_matches = rank_preds == player_rank
            batch_size = obs.shape[0]
            batch_metrics = {
                'loss_sum': total_loss.detach().to(torch.float64) * batch_size,
                'policy_loss_sum': policy_loss.detach().to(torch.float64) * batch_size,
                'aux_loss_sum': aux_loss.detach().to(torch.float64) * batch_size,
                'rank_aux_raw_loss_sum': raw_aux_loss.detach().to(torch.float64) * batch_size,
                'rank_aux_weight_sum': rank_aux_weights.detach().to(torch.float64).sum(),
                'opponent_turn_weight_sum': opponent_turn_weights.detach().to(torch.float64).sum(),
                'opponent_aux_loss_sum': opponent_aux_loss.detach().to(torch.float64) * batch_size,
                'danger_turn_weight_sum': danger_turn_weights.detach().to(torch.float64).sum(),
                'danger_aux_loss_sum': danger_aux_loss.detach().to(torch.float64) * batch_size,
                'danger_any_loss_sum': danger_metric_stats['any_loss_sum'],
                'danger_value_loss_sum': danger_metric_stats['value_loss_sum'],
                'danger_player_loss_sum': danger_metric_stats['player_loss_sum'],
                'action_correct': action_matches.to(torch.int64).sum(),
                'rank_correct': rank_matches.to(torch.int64).sum(),
                'count': batch_size,
                'discard_stats': compute_discard_stats(probs, pred_actions, actions) if compute_detailed_metrics else None,
                'chi_exact_stats': compute_chi_exact_stats(probs, pred_actions, actions) if compute_detailed_metrics else None,
                'group_stats': compute_action_group_stats(pred_actions, actions),
                'decision_stats': compute_decision_stats(probs, pred_actions, actions, masks) if compute_detailed_metrics else None,
                'sliced_decision_stats': (
                    compute_sliced_decision_stats(
                        probs,
                        pred_actions,
                        actions,
                        masks,
                        context_meta,
                        opponent_shanten=opponent_shanten,
                        opponent_tenpai=opponent_tenpai,
                    )
                    if compute_sliced_metrics else None
                ),
                'sliced_discard_stats': (
                    compute_sliced_discard_stats(
                        probs,
                        pred_actions,
                        actions,
                        context_meta,
                        opponent_shanten=opponent_shanten,
                        opponent_tenpai=opponent_tenpai,
                    )
                    if compute_sliced_metrics else None
                ),
                'opponent_stats': opponent_metric_stats,
                'danger_stats': danger_metric_stats if compute_detailed_metrics else None,
            }

        return total_loss, batch_metrics, oracle_gamma

    def evaluate(file_list, log_step, *, desc, scalar_prefix, max_batches=0):
        if not file_list:
            logging.info(f'[{scalar_prefix}] skipped: empty file list')
            return None, 0
        mortal.eval()
        policy_net.eval()
        aux_net.eval()
        if opponent_aux_net is not None:
            opponent_aux_net.eval()
        if danger_aux_net is not None:
            danger_aux_net.eval()

        def run_eval_loop(totals):
            val_loader = None
            val_batches_on_device = None
            batch_count = 0
            try:
                val_loader, val_batches_on_device = build_batch_iter(
                    file_list,
                    training=False,
                    shuffle_files=False,
                )
                with torch.inference_mode():
                    for batch in tqdm(val_loader, desc=desc, unit='batch'):
                        _, batch_metrics, _ = forward_loss(
                            batch,
                            batch_on_device=val_batches_on_device,
                            compute_detailed_metrics=True,
                            compute_sliced_metrics=True,
                            training=False,
                        )
                        merge_metrics(totals, batch_metrics)
                        batch_count += 1
                        if max_batches > 0 and batch_count >= max_batches:
                            break
            finally:
                close_batch_iter(val_loader)
                del val_loader
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            return batch_count

        def run_eval_attempt():
            totals = init_metric_dict(include_detailed_metrics=True, include_sliced_metrics=True)
            batch_count = run_eval_loop(totals)
            return totals, batch_count

        totals, batch_count = run_with_validation_retries(
            run_eval_attempt,
            device_type=device.type,
            context='validation',
        )

        metrics = finalize_metrics(totals)
        writer.add_scalar(f'{scalar_prefix}/loss', metrics['loss'], log_step)
        writer.add_scalar(f'{scalar_prefix}/policy_loss', metrics['policy_loss'], log_step)
        writer.add_scalar(f'{scalar_prefix}/aux_loss', metrics['aux_loss'], log_step)
        writer.add_scalar(f'{scalar_prefix}/rank_aux_raw_loss', metrics['rank_aux_raw_loss'], log_step)
        writer.add_scalar(f'{scalar_prefix}/rank_aux_weight_mean', metrics['rank_aux_weight_mean'], log_step)
        writer.add_scalar(f'{scalar_prefix}/opponent_turn_weight_mean', metrics['opponent_turn_weight_mean'], log_step)
        writer.add_scalar(f'{scalar_prefix}/danger_turn_weight_mean', metrics['danger_turn_weight_mean'], log_step)
        writer.add_scalar(f'{scalar_prefix}/action_quality_score', metrics['action_quality_score'], log_step)
        writer.add_scalar(f'{scalar_prefix}/scenario_quality_score', metrics['scenario_quality_score'], log_step)
        if 'selection_quality_score' in metrics:
            writer.add_scalar(f'{scalar_prefix}/selection_quality_score', metrics['selection_quality_score'], log_step)
        writer.add_scalar(f'{scalar_prefix}/action_acc', metrics['action_acc'], log_step)
        writer.add_scalar(f'{scalar_prefix}/macro_action_acc', metrics['macro_action_acc'], log_step)
        writer.add_scalar(f'{scalar_prefix}/rank_acc', metrics['rank_acc'], log_step)
        if 'discard_nll' in metrics:
            writer.add_scalar(f'{scalar_prefix}/discard_nll', metrics['discard_nll'], log_step)
        if 'chi_exact_nll' in metrics:
            writer.add_scalar(f'{scalar_prefix}/chi_exact_nll', metrics['chi_exact_nll'], log_step)
        if 'discard_top3_acc' in metrics:
            writer.add_scalar(f'{scalar_prefix}/discard_top3_acc', metrics['discard_top3_acc'], log_step)
        if 'opponent_aux_loss' in metrics:
            writer.add_scalar(f'{scalar_prefix}/opponent_aux_loss', metrics['opponent_aux_loss'], log_step)
        if 'danger_aux_loss' in metrics:
            writer.add_scalar(f'{scalar_prefix}/danger_aux_loss', metrics['danger_aux_loss'], log_step)
            writer.add_scalar(f'{scalar_prefix}/danger_any_loss', metrics['danger_any_loss'], log_step)
            writer.add_scalar(f'{scalar_prefix}/danger_value_loss', metrics['danger_value_loss'], log_step)
            writer.add_scalar(f'{scalar_prefix}/danger_player_loss', metrics['danger_player_loss'], log_step)
        if 'opponent_shanten_macro_acc' in metrics:
            writer.add_scalar(f'{scalar_prefix}/opponent_shanten_macro_acc', metrics['opponent_shanten_macro_acc'], log_step)
        if 'opponent_tenpai_macro_acc' in metrics:
            writer.add_scalar(f'{scalar_prefix}/opponent_tenpai_macro_acc', metrics['opponent_tenpai_macro_acc'], log_step)
        for name in decision_metric_names:
            for suffix in ('balanced_acc', 'balanced_bce', 'pred_rate', 'target_rate'):
                key = f'{name}_{suffix}'
                if key in metrics:
                    writer.add_scalar(f'{scalar_prefix}/{key}', metrics[key], log_step)
        log_group_acc(f'{scalar_prefix}_acc', totals['group_stats'], log_step)
        log_decision_acc(f'{scalar_prefix}_decision', totals['decision_stats'], log_step)
        log_sliced_decision_metrics(f'{scalar_prefix}_decision_slice', totals['sliced_decision_stats'], log_step)
        log_sliced_discard_metrics(f'{scalar_prefix}_discard_slice', totals['sliced_discard_stats'], log_step)
        log_opponent_acc(f'{scalar_prefix}_opp', totals['opponent_stats'], log_step)
        log_danger_metrics(f'{scalar_prefix}_danger', totals['danger_stats'], log_step)
        writer.flush()
        return metrics, batch_count

    def run_gradient_calibration_probe(file_list, log_step, *, desc, scalar_prefix):
        if not gradient_calibration_enabled or gradient_calibration_max_batches <= 0:
            return {}
        if not file_list:
            return {}

        mortal.eval()
        policy_net.eval()
        aux_net.eval()
        if opponent_aux_net is not None:
            opponent_aux_net.eval()
        if danger_aux_net is not None:
            danger_aux_net.eval()

        def run_probe_attempt():
            totals = {
                'policy_phi_grad_rms_sum': 0.0,
                'rank_phi_grad_rms_sum': 0.0,
                'opponent_phi_grad_rms_sum': 0.0,
                'danger_phi_grad_rms_sum': 0.0,
                'rank_policy_phi_grad_cos_sum': 0.0,
                'opponent_policy_phi_grad_cos_sum': 0.0,
                'danger_policy_phi_grad_cos_sum': 0.0,
                'rank_opponent_phi_grad_cos_sum': 0.0,
                'rank_danger_phi_grad_cos_sum': 0.0,
                'opp_danger_phi_grad_cos_sum': 0.0,
                'opp_danger_phi_combo_factor_sum': 0.0,
                'count': 0,
            }
            val_loader = None
            val_batches_on_device = None
            try:
                val_loader, val_batches_on_device = build_batch_iter(
                    file_list,
                    training=False,
                    shuffle_files=False,
                )
                for batch in tqdm(val_loader, desc=desc, unit='batch'):
                    (
                        obs,
                        oracle_obs,
                        actions,
                        masks,
                        player_rank,
                        context_meta,
                        opponent_shanten,
                        opponent_tenpai,
                        danger_valid,
                        danger_any,
                        danger_value,
                        danger_player_mask,
                    ) = batch if val_batches_on_device else move_batch_to_device(batch)

                    oracle_input, _ = prepare_oracle_obs(oracle_obs, obs=obs, training=False)
                    with torch.enable_grad():
                        with torch.autocast(device.type, enabled=enable_amp):
                            phi = mortal(obs, invisible_obs=oracle_input) if oracle_input is not None else mortal(obs)
                            policy_logits = policy_net.logits(phi, masks)
                            policy_loss = nn.functional.cross_entropy(policy_logits, actions)

                            rank_logits = aux_net(phi)[0]
                            rank_aux_weights = compute_rank_aux_sample_weights(context_meta)
                            raw_aux_loss_vec = nn.functional.cross_entropy(
                                rank_logits,
                                player_rank,
                                reduction='none',
                            )
                            rank_aux_loss = (raw_aux_loss_vec * rank_aux_weights).mean()

                            opponent_aux_loss = None
                            if opponent_aux_net is not None:
                                shanten_logits, tenpai_logits = opponent_aux_net(phi)
                                opponent_turn_weights = compute_context_turn_weights(
                                    context_meta,
                                    opponent_turn_weighting,
                                )
                                opponent_aux_loss, _ = compute_opponent_metrics(
                                    shanten_logits,
                                    tenpai_logits,
                                    opponent_shanten,
                                    opponent_tenpai,
                                    opponent_turn_weights,
                                )

                            danger_aux_loss = None
                            if danger_aux_net is not None:
                                danger_any_logits, danger_value_pred, danger_player_logits = danger_aux_net(phi)
                                danger_turn_weights = compute_context_turn_weights(
                                    context_meta,
                                    danger_turn_weighting,
                                )
                                danger_aux_loss, _ = compute_danger_metrics(
                                    danger_any_logits,
                                    danger_value_pred,
                                    danger_player_logits,
                                    masks,
                                    danger_valid,
                                    danger_any,
                                    danger_value,
                                    danger_player_mask,
                                    danger_turn_weights,
                                    compute_stats=False,
                                )

                        active_losses = [
                            ('policy', policy_loss),
                            ('rank', rank_aux_loss),
                        ]
                        if opponent_aux_loss is not None:
                            active_losses.append(('opponent', opponent_aux_loss))
                        if danger_aux_loss is not None:
                            active_losses.append(('danger', danger_aux_loss))

                        grad_vectors = {}
                        for idx, (name, loss_value) in enumerate(active_losses):
                            retain_graph = idx < len(active_losses) - 1
                            grad_vectors[name] = torch.autograd.grad(
                                loss_value,
                                phi,
                                retain_graph=retain_graph,
                                allow_unused=False,
                            )[0].detach()

                    policy_grad = grad_vectors.get('policy')
                    rank_grad = grad_vectors.get('rank')
                    opponent_grad = grad_vectors.get('opponent')
                    danger_grad = grad_vectors.get('danger')

                    totals['policy_phi_grad_rms_sum'] += gradient_probe_rms(policy_grad)
                    totals['rank_phi_grad_rms_sum'] += gradient_probe_rms(rank_grad)
                    totals['opponent_phi_grad_rms_sum'] += gradient_probe_rms(opponent_grad)
                    totals['danger_phi_grad_rms_sum'] += gradient_probe_rms(danger_grad)
                    totals['rank_policy_phi_grad_cos_sum'] += gradient_probe_cosine(rank_grad, policy_grad)
                    totals['opponent_policy_phi_grad_cos_sum'] += gradient_probe_cosine(opponent_grad, policy_grad)
                    totals['danger_policy_phi_grad_cos_sum'] += gradient_probe_cosine(danger_grad, policy_grad)
                    totals['rank_opponent_phi_grad_cos_sum'] += gradient_probe_cosine(rank_grad, opponent_grad)
                    totals['rank_danger_phi_grad_cos_sum'] += gradient_probe_cosine(rank_grad, danger_grad)
                    totals['opp_danger_phi_grad_cos_sum'] += gradient_probe_cosine(opponent_grad, danger_grad)
                    totals['opp_danger_phi_combo_factor_sum'] += gradient_probe_combo_factor(
                        opponent_grad,
                        danger_grad,
                    )
                    totals['count'] += 1

                    del grad_vectors
                    if totals['count'] >= gradient_calibration_max_batches:
                        break
            finally:
                close_batch_iter(val_loader)
                del val_loader
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            return totals

        totals = run_with_validation_retries(
            run_probe_attempt,
            device_type=device.type,
            context='gradient calibration validation',
        )

        if totals['count'] <= 0:
            return {}
        denom = totals['count']
        metrics = {
            'grad_probe_batches': denom,
            'policy_phi_grad_rms': totals['policy_phi_grad_rms_sum'] / denom,
            'rank_phi_grad_rms': totals['rank_phi_grad_rms_sum'] / denom,
            'opponent_phi_grad_rms': totals['opponent_phi_grad_rms_sum'] / denom,
            'danger_phi_grad_rms': totals['danger_phi_grad_rms_sum'] / denom,
            'rank_policy_phi_grad_cos': totals['rank_policy_phi_grad_cos_sum'] / denom,
            'opponent_policy_phi_grad_cos': totals['opponent_policy_phi_grad_cos_sum'] / denom,
            'danger_policy_phi_grad_cos': totals['danger_policy_phi_grad_cos_sum'] / denom,
            'rank_opponent_phi_grad_cos': totals['rank_opponent_phi_grad_cos_sum'] / denom,
            'rank_danger_phi_grad_cos': totals['rank_danger_phi_grad_cos_sum'] / denom,
            'opp_danger_phi_grad_cos': totals['opp_danger_phi_grad_cos_sum'] / denom,
            'opp_danger_phi_combo_factor': totals['opp_danger_phi_combo_factor_sum'] / denom,
        }
        for key, value in metrics.items():
            writer.add_scalar(f'{scalar_prefix}/{key}', value, log_step)
        writer.flush()
        return metrics

    def run_full_validation(check_index):
        nonlocal last_full_recent_metrics
        nonlocal full_validation_checks
        full_validation_checks += 1
        metrics, batch_count = evaluate(
            full_recent_files,
            steps,
            desc=f'FULL VAL {full_validation_checks}',
            scalar_prefix='full_recent',
            max_batches=0,
        )
        if metrics is None:
            return None
        if gradient_calibration_split == 'full_recent':
            metrics.update(
                run_gradient_calibration_probe(
                    full_recent_files,
                    steps,
                    desc=f'GRAD CAL FULL {full_validation_checks}',
                    scalar_prefix='full_recent_grad',
                )
            )
        last_full_recent_metrics = metrics
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f'[FULL RECENT] full_check={full_validation_checks} '
            f'monitor_check={check_index} '
            f'step={steps:,} batches={batch_count:,} '
            f'loss={metrics["loss"]:.4f} '
            f'policy={metrics["policy_loss"]:.4f} '
            f'aux={metrics["aux_loss"]:.4f} '
            f'rank_aux_raw={metrics["rank_aux_raw_loss"]:.4f} '
            f'rank_w={metrics["rank_aux_weight_mean"]:.4f} '
            f'opp={metrics["opponent_aux_loss"]:.4f} '
            f'opp_w={metrics["opponent_turn_weight_mean"]:.4f} '
            f'danger={metrics["danger_aux_loss"]:.4f} '
            f'danger_w={metrics["danger_turn_weight_mean"]:.4f} '
            f'action_score={metrics["action_quality_score"]:.4f} '
            f'scenario_score={metrics["scenario_quality_score"]:.4f} '
            f'selection_score={metrics.get("selection_quality_score", float("nan")):.4f} '
            f'action_acc={metrics["action_acc"]:.4f} '
            f'macro_action_acc={metrics["macro_action_acc"]:.4f} '
            f'rank_acc={metrics["rank_acc"]:.4f} '
            f'lr={current_lr:.3e}'
        )
        return metrics

    def run_old_regression_validation(check_index):
        nonlocal last_old_regression_metrics
        nonlocal old_regression_checks
        if not old_regression_files:
            return None
        old_regression_checks += 1
        metrics, batch_count = evaluate(
            old_regression_files,
            steps,
            desc=f'OLD REG {old_regression_checks}',
            scalar_prefix='old_regression',
            max_batches=0,
        )
        if metrics is None:
            return None
        last_old_regression_metrics = metrics
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f'[OLD REGRESSION] check={old_regression_checks} '
            f'monitor_check={check_index} '
            f'step={steps:,} batches={batch_count:,} '
            f'loss={metrics["loss"]:.4f} '
            f'policy={metrics["policy_loss"]:.4f} '
            f'aux={metrics["aux_loss"]:.4f} '
            f'rank_aux_raw={metrics["rank_aux_raw_loss"]:.4f} '
            f'rank_w={metrics["rank_aux_weight_mean"]:.4f} '
            f'opp={metrics["opponent_aux_loss"]:.4f} '
            f'opp_w={metrics["opponent_turn_weight_mean"]:.4f} '
            f'danger={metrics["danger_aux_loss"]:.4f} '
            f'danger_w={metrics["danger_turn_weight_mean"]:.4f} '
            f'action_score={metrics["action_quality_score"]:.4f} '
            f'scenario_score={metrics["scenario_quality_score"]:.4f} '
            f'selection_score={metrics.get("selection_quality_score", float("nan")):.4f} '
            f'action_acc={metrics["action_acc"]:.4f} '
            f'macro_action_acc={metrics["macro_action_acc"]:.4f} '
            f'rank_acc={metrics["rank_acc"]:.4f} '
            f'lr={current_lr:.3e}'
        )
        return metrics

    def run_monitor_validation(epoch, *, reason):
        nonlocal best_val_loss
        nonlocal best_val_action_acc
        nonlocal best_val_action_score
        nonlocal best_val_rank_acc
        nonlocal best_full_recent_loss
        nonlocal best_full_recent_action_acc
        nonlocal best_full_recent_action_score
        nonlocal best_full_recent_rank_acc
        nonlocal last_monitor_recent_metrics
        nonlocal patience_counter
        nonlocal num_lr_reductions
        nonlocal validation_checks

        validation_checks += 1
        metrics, batch_count = evaluate(
            monitor_recent_files,
            steps,
            desc=f'MONITOR VAL {validation_checks}',
            scalar_prefix='monitor_recent',
            max_batches=monitor_val_batches,
        )
        if metrics is None:
            return False, False
        if gradient_calibration_split == 'monitor_recent':
            metrics.update(
                run_gradient_calibration_probe(
                    monitor_recent_files,
                    steps,
                    desc=f'GRAD CAL MON {validation_checks}',
                    scalar_prefix='monitor_recent_grad',
                )
            )
        last_monitor_recent_metrics = metrics

        prev_lr = optimizer.param_groups[0]['lr']
        current_lr = prev_lr
        if scheduler_type == 'plateau' and optimizer_steps >= warm_up_steps:
            scheduler.step(metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < prev_lr - 1e-12:
                num_lr_reductions += 1
                logging.info(f'lr reduced: {prev_lr:.3e} -> {current_lr:.3e} (count={num_lr_reductions})')
        elif scheduler_type == 'plateau':
            logging.info(
                f'warmup active: optimizer_steps={optimizer_steps}/{warm_up_steps}, '
                f'skip plateau step'
            )

        logging.info(
            f'[MONITOR RECENT] check={validation_checks} '
            f'step={steps:,} batches={batch_count:,} '
            f'loss={metrics["loss"]:.4f} '
            f'policy={metrics["policy_loss"]:.4f} '
            f'aux={metrics["aux_loss"]:.4f} '
            f'rank_aux_raw={metrics["rank_aux_raw_loss"]:.4f} '
            f'rank_w={metrics["rank_aux_weight_mean"]:.4f} '
            f'opp={metrics["opponent_aux_loss"]:.4f} '
            f'opp_w={metrics["opponent_turn_weight_mean"]:.4f} '
            f'danger={metrics["danger_aux_loss"]:.4f} '
            f'danger_w={metrics["danger_turn_weight_mean"]:.4f} '
            f'action_score={metrics["action_quality_score"]:.4f} '
            f'scenario_score={metrics["scenario_quality_score"]:.4f} '
            f'selection_score={metrics.get("selection_quality_score", float("nan")):.4f} '
            f'action_acc={metrics["action_acc"]:.4f} '
            f'macro_action_acc={metrics["macro_action_acc"]:.4f} '
            f'rank_acc={metrics["rank_acc"]:.4f} '
            f'lr={current_lr:.3e}'
        )

        improved_loss = metrics['loss'] < best_val_loss - early_stopping_min_delta
        improved_acc = metrics['action_acc'] > best_val_action_acc
        improved_action_score = metrics['action_quality_score'] > best_val_action_score
        improved_rank_acc = metrics['rank_acc'] > best_val_rank_acc

        if improved_loss:
            best_val_loss = metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(
                f'no monitor-loss improvement: best_val_loss={best_val_loss:.4f} '
                f'patience={patience_counter}/{early_stopping_patience_checks}'
            )

        if improved_acc:
            best_val_action_acc = metrics['action_acc']
        if improved_action_score:
            best_val_action_score = metrics['action_quality_score']

        if improved_rank_acc:
            best_val_rank_acc = metrics['rank_acc']

        ran_full_val = False
        selection_metrics = metrics
        if should_run_full_validation_this_check(
            full_val_every_checks=full_val_every_checks,
            validation_checks=validation_checks,
            has_full_recent_files=bool(full_recent_files),
        ):
            full_metrics = run_full_validation(validation_checks)
            ran_full_val = True
            if full_metrics is not None:
                selection_metrics = full_metrics
        elif not full_recent_files:
            ran_full_val = True

        if should_run_old_regression_validation_this_check(
            old_regression_every_checks=old_regression_every_checks,
            validation_checks=validation_checks,
            has_old_regression_files=bool(old_regression_files),
        ):
            run_old_regression_validation(validation_checks)
        elif should_run_old_regression_after_full_validation(
            old_regression_every_checks=old_regression_every_checks,
            ran_full_val=ran_full_val,
            has_old_regression_files=bool(old_regression_files),
        ):
            run_old_regression_validation(validation_checks)

        improved_selection_loss = selection_metrics['loss'] < best_full_recent_loss - early_stopping_min_delta
        improved_selection_acc = selection_metrics['macro_action_acc'] > best_full_recent_action_acc
        improved_selection_action_score = selection_metrics['action_quality_score'] > best_full_recent_action_score
        improved_selection_rank_acc = selection_metrics['rank_acc'] > best_full_recent_rank_acc
        best_checkpoint_targets = []

        if improved_selection_loss:
            best_full_recent_loss = selection_metrics['loss']
            best_checkpoint_targets.append(
                (best_loss_state_file, f'new best-loss {checkpoint_label} checkpoint')
            )

        if improved_selection_acc:
            best_full_recent_action_acc = selection_metrics['macro_action_acc']
        if improved_selection_action_score:
            best_full_recent_action_score = selection_metrics['action_quality_score']
            best_checkpoint_targets.append(
                (best_acc_state_file, f'new best-action-score {checkpoint_label} checkpoint')
            )

        if improved_selection_rank_acc:
            best_full_recent_rank_acc = selection_metrics['rank_acc']
            best_checkpoint_targets.append(
                (best_rank_state_file, f'new best-rank-acc {checkpoint_label} checkpoint')
            )

        state = save_latest_state(epoch, epoch_complete=False, reason=reason)
        saved_best_paths = set()
        for checkpoint_path, label in best_checkpoint_targets:
            checkpoint_key = str(checkpoint_path)
            if checkpoint_key in saved_best_paths:
                continue
            saved_best_paths.add(checkpoint_key)
            save_named_state(state, checkpoint_path, label=label)

        mortal.train()
        policy_net.train()
        aux_net.train()
        if opponent_aux_net is not None:
            opponent_aux_net.train()
        if danger_aux_net is not None:
            danger_aux_net.train()

        should_stop = (
            early_stopping_patience_checks > 0
            and validation_checks >= min_validation_checks
            and patience_counter >= early_stopping_patience_checks
            and num_lr_reductions >= early_stopping_min_lr_reductions
        )
        if should_stop:
            logging.info(
                f'early stopping triggered at monitor check {validation_checks}; '
                f'best_val_loss={best_val_loss:.4f} '
                f'best_val_action_acc={best_val_action_acc:.4f} '
                f'best_val_action_score={best_val_action_score:.4f} '
                f'best_val_rank_acc={best_val_rank_acc:.4f}'
            )
        return should_stop, ran_full_val

    def build_state(epoch, *, epoch_complete):
        return {
            'mortal': mortal.state_dict(),
            'policy_net': policy_net.state_dict(),
            'aux_net': aux_net.state_dict(),
            'opponent_aux_net': opponent_aux_net.state_dict() if opponent_aux_net is not None else None,
            'danger_aux_net': danger_aux_net.state_dict() if danger_aux_net is not None else None,
            'optimizer': optimizer.state_dict(),
            'optimizer_param_groups': optimizer_param_groups,
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'optimizer_steps': optimizer_steps,
            'epoch': epoch,
            'epoch_complete': epoch_complete,
            'timestamp': datetime.now().timestamp(),
            'best_val_loss': best_val_loss,
            'best_val_action_acc': best_val_action_acc,
            'best_val_action_score': best_val_action_score,
            'best_val_rank_acc': best_val_rank_acc,
            'best_full_recent_loss': best_full_recent_loss,
            'best_full_recent_action_acc': best_full_recent_action_acc,
            'best_full_recent_action_score': best_full_recent_action_score,
            'best_full_recent_rank_acc': best_full_recent_rank_acc,
            'patience_counter': patience_counter,
            'num_lr_reductions': num_lr_reductions,
            'validation_checks': validation_checks,
            'full_validation_checks': full_validation_checks,
            'old_regression_checks': old_regression_checks,
            'last_monitor_recent_metrics': last_monitor_recent_metrics,
            'last_full_recent_metrics': last_full_recent_metrics,
            'last_old_regression_metrics': last_old_regression_metrics,
            'config_section': cfg_prefix,
            'stage_label': stage_label,
            'checkpoint_label': checkpoint_label,
            'config': config,
        }

    def save_latest_state(epoch, *, epoch_complete, reason):
        state = build_state(epoch, epoch_complete=epoch_complete)
        torch.save(state, state_file)
        logging.info(
            f'saved latest {checkpoint_label} checkpoint to {state_file} '
            f'({reason}, step={steps:,}, optimizer_steps={optimizer_steps:,}, epoch={epoch + 1})'
        )
        return state

    def save_named_state(state, checkpoint_path, *, label):
        torch.save(state, checkpoint_path)
        logging.info(f'saved {label} to {checkpoint_path}')
        if checkpoint_path == best_loss_state_file and best_loss_normal_state_file:
            export_normal_state(
                state,
                best_loss_normal_state_file,
                label=f'{label} (visible-only export)',
            )
        if (
            checkpoint_path == best_loss_state_file
            and stage2_handoff_state_file
            and not paths_match(stage2_handoff_state_file, best_loss_normal_state_file)
        ):
            export_normal_state(
                state,
                stage2_handoff_state_file,
                label='Stage 2 handoff checkpoint',
            )
        elif checkpoint_path == best_acc_state_file and best_acc_normal_state_file:
            export_normal_state(
                state,
                best_acc_normal_state_file,
                label=f'{label} (visible-only export)',
            )
        elif checkpoint_path == best_rank_state_file and best_rank_normal_state_file:
            export_normal_state(
                state,
                best_rank_normal_state_file,
                label=f'{label} (visible-only export)',
            )

    if (
        steps > 0
        and val_every_steps > 0
        and validation_checks == 0
        and best_val_loss == float('inf')
    ):
        logging.info(
            f'no validation history found in resumed checkpoint at step={steps:,}; '
            f'run resume-baseline monitor validation before continuing'
        )
        should_stop, _ = run_monitor_validation(start_epoch, reason='resume_baseline')
        if should_stop:
            return

    train_loader = None
    train_batches_on_device = False

    def release_train_loader():
        nonlocal train_loader, train_batches_on_device
        if train_loader is None:
            return
        close_batch_iter(train_loader)
        train_loader = None
        train_batches_on_device = False
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    def handle_post_optimizer_step(epoch):
        nonlocal should_stop, ran_full_val, stop_due_to_budget

        post_step_actions = plan_post_optimizer_step_actions(
            steps=steps,
            save_every=save_every,
            val_every_steps=val_every_steps,
            max_steps=max_steps,
        )
        if post_step_actions['save_periodic']:
            save_latest_state(epoch, epoch_complete=False, reason='periodic')
        if post_step_actions['save_budget_checkpoint']:
            save_latest_state(epoch, epoch_complete=False, reason='max_steps')
        if post_step_actions['release_train_loader']:
            release_train_loader()
        if post_step_actions['validation_reason'] is not None:
            should_stop, ran_full_val = run_monitor_validation(
                epoch,
                reason=post_step_actions['validation_reason'],
            )
        if post_step_actions['stop_due_to_budget']:
            stop_due_to_budget = True
        return should_stop or stop_due_to_budget

    for epoch in range(start_epoch, max_epochs):
        mortal.train()
        policy_net.train()
        aux_net.train()
        if opponent_aux_net is not None:
            opponent_aux_net.train()
        if danger_aux_net is not None:
            danger_aux_net.train()

        train_loader, train_batches_on_device = build_batch_iter(
            train_files,
            training=True,
            shuffle_files=True,
            safe_train=force_safe_training,
        )
        running = init_metric_dict(include_detailed_metrics=False, include_sliced_metrics=False)
        running['batches'] = 0
        accum = 0
        should_stop = False
        ran_full_val = False
        stop_due_to_budget = False

        for batch in tqdm(train_loader, desc=f'TRAIN E{epoch + 1}', unit='batch'):
            total_loss, batch_metrics, oracle_gamma = forward_loss(
                batch,
                batch_on_device=train_batches_on_device,
                compute_detailed_metrics=False,
                compute_sliced_metrics=False,
                training=True,
            )
            scaler.scale(total_loss / opt_step_every).backward()

            merge_metrics(running, batch_metrics)
            running['batches'] += 1

            steps += 1
            accum += 1
            if accum % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(group['params'] for group in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if scheduler_type == 'cosine':
                    scheduler.step()
                else:
                    update_warmup_lr()
                if handle_post_optimizer_step(epoch):
                    break

            if running['batches'] % log_every == 0:
                metrics = finalize_metrics(running)
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f'epoch={epoch + 1} step={steps:,} '
                    f'loss={metrics["loss"]:.4f} '
                    f'policy={metrics["policy_loss"]:.4f} '
                    f'aux={metrics["aux_loss"]:.4f} '
                    f'rank_aux_raw={metrics["rank_aux_raw_loss"]:.4f} '
                    f'rank_w={metrics["rank_aux_weight_mean"]:.4f} '
                    f'opp={metrics["opponent_aux_loss"]:.4f} '
                    f'opp_w={metrics["opponent_turn_weight_mean"]:.4f} '
                    f'danger={metrics["danger_aux_loss"]:.4f} '
                    f'danger_w={metrics["danger_turn_weight_mean"]:.4f} '
                    f'action_acc={metrics["action_acc"]:.4f} '
                    f'macro_action_acc={metrics["macro_action_acc"]:.4f} '
                    f'rank_acc={metrics["rank_acc"]:.4f} '
                    f'oracle_gamma={oracle_gamma:.4f} '
                    f'lr={current_lr:.3e}'
                )
                writer.add_scalar('train/loss', metrics['loss'], steps)
                writer.add_scalar('train/policy_loss', metrics['policy_loss'], steps)
                writer.add_scalar('train/aux_loss', metrics['aux_loss'], steps)
                writer.add_scalar('train/rank_aux_raw_loss', metrics['rank_aux_raw_loss'], steps)
                writer.add_scalar('train/rank_aux_weight_mean', metrics['rank_aux_weight_mean'], steps)
                writer.add_scalar('train/opponent_turn_weight_mean', metrics['opponent_turn_weight_mean'], steps)
                writer.add_scalar('train/danger_turn_weight_mean', metrics['danger_turn_weight_mean'], steps)
                writer.add_scalar('train/action_acc', metrics['action_acc'], steps)
                writer.add_scalar('train/macro_action_acc', metrics['macro_action_acc'], steps)
                writer.add_scalar('train/rank_acc', metrics['rank_acc'], steps)
                if 'action_quality_score' in metrics:
                    writer.add_scalar('train/action_quality_score', metrics['action_quality_score'], steps)
                if 'scenario_quality_score' in metrics:
                    writer.add_scalar('train/scenario_quality_score', metrics['scenario_quality_score'], steps)
                if 'selection_quality_score' in metrics:
                    writer.add_scalar('train/selection_quality_score', metrics['selection_quality_score'], steps)
                if 'discard_nll' in metrics:
                    writer.add_scalar('train/discard_nll', metrics['discard_nll'], steps)
                if 'chi_exact_nll' in metrics:
                    writer.add_scalar('train/chi_exact_nll', metrics['chi_exact_nll'], steps)
                if 'discard_top3_acc' in metrics:
                    writer.add_scalar('train/discard_top3_acc', metrics['discard_top3_acc'], steps)
                if 'opponent_aux_loss' in metrics:
                    writer.add_scalar('train/opponent_aux_loss', metrics['opponent_aux_loss'], steps)
                if 'danger_aux_loss' in metrics:
                    writer.add_scalar('train/danger_aux_loss', metrics['danger_aux_loss'], steps)
                    writer.add_scalar('train/danger_any_loss', metrics['danger_any_loss'], steps)
                    writer.add_scalar('train/danger_value_loss', metrics['danger_value_loss'], steps)
                    writer.add_scalar('train/danger_player_loss', metrics['danger_player_loss'], steps)
                if 'opponent_shanten_macro_acc' in metrics:
                    writer.add_scalar('train/opponent_shanten_macro_acc', metrics['opponent_shanten_macro_acc'], steps)
                if 'opponent_tenpai_macro_acc' in metrics:
                    writer.add_scalar('train/opponent_tenpai_macro_acc', metrics['opponent_tenpai_macro_acc'], steps)
                if use_oracle:
                    writer.add_scalar('train/oracle_gamma', oracle_gamma, steps)
                writer.add_scalar('train/lr', current_lr, steps)
                log_group_acc('train_acc', running['group_stats'], steps)
                log_decision_acc('train_decision', running['decision_stats'], steps)
                log_opponent_acc('train_opp', running['opponent_stats'], steps)
                log_danger_metrics('train_danger', running['danger_stats'], steps)
                writer.flush()

        if accum % opt_step_every != 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                params = chain.from_iterable(group['params'] for group in optimizer.param_groups)
                clip_grad_norm_(params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if scheduler_type == 'cosine':
                scheduler.step()
            else:
                update_warmup_lr()
            handle_post_optimizer_step(epoch)

        release_train_loader()

        budget_stop_actions = plan_budget_stop_final_actions(
            stop_due_to_budget=stop_due_to_budget,
            ran_full_val=ran_full_val,
            has_full_recent_files=bool(full_recent_files),
            has_old_regression_files=bool(old_regression_files),
            old_regression_every_checks=old_regression_every_checks,
        )
        if budget_stop_actions['run_full_validation']:
            run_full_validation(validation_checks)
        if budget_stop_actions['run_old_regression_validation']:
            run_old_regression_validation(validation_checks)
        if budget_stop_actions['resave_latest_state']:
            save_latest_state(epoch, epoch_complete=False, reason='max_steps_post_validation')

        if should_stop or stop_due_to_budget:
            break

        if val_every_steps <= 0 or steps % val_every_steps != 0:
            should_stop, ran_full_val = run_monitor_validation(epoch, reason='epoch_end_monitor')
            if should_stop:
                break

        if should_run_fallback_full_validation(
            ran_full_val=ran_full_val,
            has_full_recent_files=bool(full_recent_files),
        ):
            run_full_validation(validation_checks)

        save_latest_state(epoch, epoch_complete=True, reason='epoch_end')

        gc.collect()


def main():
    train()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

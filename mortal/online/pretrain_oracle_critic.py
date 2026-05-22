import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from glob import glob
from os import path

import numpy as np
import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mortal.core.prelude
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
from mortal.core.lr_scheduler import LinearWarmUpCosineAnnealingLR
from mortal.core.model import Brain, OracleDualTowerBrain, ValueHead
from mortal.data.dataloader import resolve_rayon_num_threads, worker_init_fn
from mortal.data.oracle_value import OracleTerminalValueDataset


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
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--val-every-steps', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--max-train-files', type=int, default=None)
    parser.add_argument('--max-val-files', type=int, default=None)
    parser.add_argument('--val-batches', type=int, default=None)
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
    parser.add_argument('--tail-blocks', type=int, default=None)
    parser.add_argument('--encoder-lr-scale', type=float, default=None)
    parser.add_argument('--scheduler-peak', type=float, default=None)
    parser.add_argument('--scheduler-final', type=float, default=None)
    parser.add_argument('--scheduler-warm-up-steps', '--warm-up-steps', dest='scheduler_warm_up_steps', type=int, default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-checkpoint', type=str, default=None)
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


def build_file_lists(cfg):
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

    seed = int(cfg.get('seed', 20260416) or 0)
    rng = random.Random(seed)
    shuffled = list(file_list)
    rng.shuffle(shuffled)

    val_ratio = float(cfg.get('val_ratio', 0.02) or 0.0)
    min_val_files = int(cfg.get('min_val_files', 64) or 0)
    max_val_files = int(cfg.get('max_val_files', 0) or 0)
    val_count = max(min_val_files, int(len(shuffled) * val_ratio))
    if max_val_files > 0:
        val_count = min(val_count, max_val_files)
    val_count = max(1, min(val_count, max(len(shuffled) - 1, 1)))

    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:] or shuffled

    max_train_files = int(cfg.get('max_train_files', 0) or 0)
    if max_train_files > 0:
        train_files = train_files[:max_train_files]
    max_val_files = int(cfg.get('max_val_files', 0) or 0)
    if max_val_files > 0:
        val_files = val_files[:max_val_files]
    return train_files, val_files


def make_dataset(file_list, cfg, *, train):
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
        reserve_ratio=float(config['dataset'].get('reserve_ratio', 0.0) or 0.0) if train else 0.0,
        player_names=None,
        excludes=None,
        num_epochs=int(cfg.get('num_epochs', config['dataset'].get('num_epochs', 1)) or 1),
        enable_augmentation=bool(config['dataset'].get('enable_augmentation', False)) if train else False,
        augmented_first=bool(config['dataset'].get('augmented_first', False)),
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
    )


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
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
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


def optimizer_param_groups(oracle_brain, value_net, *, scope, encoder_lr_scale=1.0):
    scope = normalize_train_scope(scope)
    weight_decay = float(config['optim'].get('weight_decay', 0.0))
    value_decay, value_no_decay = split_decay_params(value_net, prefix='value_net.')
    groups = [
        {'name': 'value_decay', 'params': value_decay, 'weight_decay': weight_decay},
        {'name': 'value_no_decay', 'params': value_no_decay},
    ]
    if scope == 'all':
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


def build_models(device, *, critic_arch='single_tower'):
    version = config['control']['version']
    critic_arch = normalize_critic_arch(critic_arch)
    if critic_arch == 'dual_tower':
        oracle_brain = OracleDualTowerBrain(version=version, **config['resnet'], Norm='GN').to(device)
    else:
        oracle_brain = Brain(version=version, is_oracle=True, **config['resnet'], Norm='GN').to(device)
    value_net = ValueHead(num_players=4).to(device)
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


def init_dual_tower_from_visible_state(oracle_brain, source_state):
    if not is_dual_tower_brain(oracle_brain):
        return None

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
                    target_tensor.zero_()
                    out_channels = min(target_tensor.shape[0], visible_first.shape[0])
                    kernel = min(target_tensor.shape[2], visible_first.shape[2])
                    source_channels = visible_first.shape[1]
                    target_channels = target_tensor.shape[1]
                    oracle_start = visible_channels
                    oracle_end = oracle_start + target_channels
                    if source_channels >= oracle_end:
                        target_tensor[:out_channels, :, :kernel].copy_(
                            visible_first[:out_channels, oracle_start:oracle_end, :kernel]
                        )
                        first_conv_init = 'single_oracle_slice'
                    elif source_channels == target_channels:
                        target_tensor[:out_channels, :, :kernel].copy_(
                            visible_first[:out_channels, :, :kernel]
                        )
                        first_conv_init = 'channel_aligned_copy'
                    else:
                        source_mean = visible_first[:out_channels, :, :kernel].mean(dim=1, keepdim=True)
                        target_tensor[:out_channels, :, :kernel].copy_(
                            source_mean.expand(-1, target_channels, -1)
                        )
                        target_tensor.mul_(0.25)
                        first_conv_init = 'visible_channel_mean_scaled'
                    loaded_oracle_keys.append(key)
                else:
                    skipped_oracle_keys.append(key)
                continue
            source_tensor = source_state.get(f'encoder.{key}')
            if source_tensor is not None and source_tensor.shape == target_tensor.shape:
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
            'first_conv_init': first_conv_init,
        },
    }


def maybe_init_from_checkpoint(oracle_brain, value_net, init_state_file, device, *, strict_oracle_checkpoint=False):
    if not init_state_file:
        return {'source': '', 'loaded': False}
    if not path.exists(init_state_file):
        raise FileNotFoundError(f'oracle_critic_pretrain.init_state_file does not exist: {init_state_file}')

    state = torch.load(init_state_file, weights_only=False, map_location=device)
    loaded = {
        'source': init_state_file,
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
                    loaded['oracle_brain'] = init_dual_tower_from_visible_state(oracle_brain, source_visible_state)
                else:
                    loaded['oracle_brain'] = load_brain_state_with_input_bridge(oracle_brain, source_oracle_state)
    elif state.get('mortal') is not None:
        if strict_oracle_checkpoint:
            raise ValueError(
                'strict oracle critic init checkpoint must contain oracle_brain; '
                f'got visible-only mortal checkpoint: {init_state_file}'
            )
        if is_dual_tower_brain(oracle_brain):
            loaded['oracle_brain'] = init_dual_tower_from_visible_state(oracle_brain, state['mortal'])
        else:
            loaded['oracle_brain'] = load_brain_state_with_input_bridge(oracle_brain, state['mortal'])
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


def model_forward(oracle_brain, value_net, obs, invisible_obs, *, enable_amp, device_type):
    with torch.autocast(device_type, enabled=enable_amp):
        phi = oracle_brain(obs, invisible_obs=invisible_obs)
        return value_net(phi)


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
    target_mse=None,
    teacher_mse=None,
    zero_sum_loss=None,
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
    for name, value in (
        ('objective_loss', objective_loss),
        ('target_mse', target_mse),
        ('teacher_mse', teacher_mse),
        ('zero_sum_loss', zero_sum_loss),
    ):
        if value is not None:
            result[f'{name}_sum'] = float(value.detach().float().item())
            result[f'{name}_batches'] = 1
    return result


def finalize_metrics(parts):
    total_loss = sum(item['loss_sum'] for item in parts)
    total_abs = sum(item['abs_sum'] for item in parts)
    total_count = sum(item['count'] for item in parts)
    total_zero = sum(item['zero_sum_abs'] for item in parts)
    total_samples = sum(item['sample_count'] for item in parts)
    if total_count <= 0:
        return {'loss': math.inf, 'mae': math.inf, 'corr': 0.0, 'zero_sum_mae': math.inf}

    pred = torch.cat([item['pred'].reshape(-1) for item in parts])
    target = torch.cat([item['target'].reshape(-1) for item in parts])
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = pred_centered.norm() * target_centered.norm()
    corr = float((pred_centered @ target_centered / denom).item()) if denom.item() > 0 else 0.0
    metrics = {
        'loss': total_loss / total_count,
        'mae': total_abs / total_count,
        'corr': corr,
        'zero_sum_mae': total_zero / max(total_samples, 1),
    }
    for name in ('objective_loss', 'target_mse', 'teacher_mse', 'zero_sum_loss'):
        batches = sum(item.get(f'{name}_batches', 0) for item in parts)
        if batches > 0:
            metrics[name] = sum(item.get(f'{name}_sum', 0.0) for item in parts) / batches
    return metrics


@torch.inference_mode()
def evaluate(oracle_brain, value_net, loader, device, *, enable_amp, max_batches):
    oracle_brain.eval()
    value_net.eval()
    parts = []
    for batch_idx, (obs, invisible_obs, target, _player_id) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        target = target.to(dtype=torch.float32, device=device, non_blocking=True)
        pred = model_forward(
            oracle_brain,
            value_net,
            obs,
            invisible_obs,
            enable_amp=enable_amp,
            device_type=device.type,
        )
        parts.append(batch_metrics(pred, target))
    oracle_brain.train()
    value_net.train()
    return finalize_metrics(parts)


def checkpoint_payload(
    *,
    oracle_brain,
    value_net,
    optimizer,
    scheduler,
    scaler,
    steps,
    best_val_loss,
    init_info,
    cfg,
    train_info,
):
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
        'config': copy.deepcopy(config),
        'oracle_critic_pretrain': dict(cfg),
        'init_info': init_info,
        'train_info': train_info_payload,
        BRAIN_IS_ORACLE_KEY: True,
        'resume_supported': True,
        'format': 'oracle_critic_pretrain_v1',
    }


def save_checkpoint(file_path, payload):
    ensure_parent_dir_for_file(file_path)
    torch.save(payload, file_path)


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


def train():
    sanitize_sys_path_for_spawn()
    args = parse_args()
    cfg = dict(oracle_pretrain_cfg())
    for key in (
        'max_steps',
        'val_every_steps',
        'save_every',
        'max_train_files',
        'max_val_files',
        'val_batches',
        'num_workers',
        'batch_size',
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
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
    if args.train_scope:
        cfg['train_scope'] = args.train_scope
    if args.tail_blocks is not None:
        cfg['tail_blocks'] = args.tail_blocks
    if args.encoder_lr_scale is not None:
        cfg['encoder_lr_scale'] = args.encoder_lr_scale
    scheduler_overrides = {}
    if args.scheduler_peak is not None:
        scheduler_overrides['peak'] = args.scheduler_peak
    if args.scheduler_final is not None:
        scheduler_overrides['final'] = args.scheduler_final
    if args.scheduler_warm_up_steps is not None:
        scheduler_overrides['warm_up_steps'] = args.scheduler_warm_up_steps
    if scheduler_overrides:
        scheduler_cfg_from_cli = dict(cfg.get('scheduler', {}) if isinstance(cfg.get('scheduler', {}), dict) else {})
        scheduler_cfg_from_cli.update(scheduler_overrides)
        cfg['scheduler'] = scheduler_cfg_from_cli

    seed = int(cfg.get('seed', 20260416) or 0)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    device = torch.device(str(cfg.get('device', config['control'].get('device', 'cuda:0'))))
    enable_amp = bool(cfg.get('enable_amp', config['control'].get('enable_amp', False)))
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
        for key in ('state_file', 'best_state_file', 'tensorboard_dir', 'metrics_file'):
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
        'oracle critic artifacts: state=%s best=%s metrics=%s tb=%s',
        display_path(state_file),
        display_path(best_state_file),
        display_path(metrics_file),
        display_path(tensorboard_dir),
    )
    ensure_dir(tensorboard_dir)
    ensure_parent_dir_for_file(metrics_file)
    if args.fresh and path.exists(metrics_file):
        os.remove(metrics_file)

    train_files, val_files = build_file_lists(cfg)
    logging.info('oracle critic pretrain files: train=%s val=%s', len(train_files), len(val_files))

    critic_arch = normalize_critic_arch(cfg.get('critic_arch', 'single_tower'))
    oracle_brain, value_net = build_models(device, critic_arch=critic_arch)
    logging.info('oracle_brain params: %s', f'{parameter_count(oracle_brain):,}')
    logging.info('value_net params: %s', f'{parameter_count(value_net):,}')
    val_batches = int(cfg.get('val_batches', 256) or 0)
    if args.eval_only:
        eval_checkpoint = resolve_cli_path(args.eval_checkpoint) or state_file
        if not path.exists(eval_checkpoint):
            raise FileNotFoundError(f'oracle critic eval checkpoint does not exist: {eval_checkpoint}')
        state = torch.load(eval_checkpoint, weights_only=False, map_location=device)
        oracle_brain.load_state_dict(state['oracle_brain'])
        value_net.load_state_dict(state['value_net'])
        steps = int(state.get('steps', 0))
        val_loader = make_loader(make_dataset(val_files, cfg, train=False), cfg, train=False)
        val_metrics = evaluate(
            oracle_brain,
            value_net,
            val_loader,
            device,
            enable_amp=enable_amp,
            max_batches=val_batches,
        )
        result = {
            'checkpoint': eval_checkpoint,
            'steps': steps,
            'val': val_metrics,
        }
        logging.info(
            'eval-only checkpoint=%s step=%s val_loss=%.6f val_mae=%.6f val_corr=%.4f val_zero_sum_mae=%.6f',
            display_path(eval_checkpoint),
            steps,
            val_metrics['loss'],
            val_metrics['mae'],
            val_metrics['corr'],
            val_metrics['zero_sum_mae'],
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
        'oracle critic arch=%s train scope=%s tail_blocks=%s encoder_lr_scale=%.4g trainable_params=%s',
        critic_arch,
        train_info['scope'],
        train_info.get('tail_blocks', 0),
        float(cfg.get('encoder_lr_scale', 1.0) or 1.0),
        f"{train_info['trainable_params']:,}",
    )

    optimizer = optim.AdamW(
        optimizer_param_groups(
            oracle_brain,
            value_net,
            scope=train_scope,
            encoder_lr_scale=float(cfg.get('encoder_lr_scale', 1.0) or 1.0),
        ),
        lr=1.0,
        weight_decay=0.0,
        betas=tuple(config['optim'].get('betas', (0.9, 0.999))),
        eps=float(config['optim'].get('eps', 1e-8)),
    )
    scheduler_cfg = dict(config['optim'].get('scheduler', {}))
    scheduler_cfg.update(cfg.get('scheduler', {}) if isinstance(cfg.get('scheduler', {}), dict) else {})
    max_steps = int(cfg.get('max_steps', scheduler_cfg.get('max_steps', 100000)) or 100000)
    scheduler_cfg['max_steps'] = max(max_steps, int(scheduler_cfg.get('warm_up_steps', 0) or 0))
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **scheduler_cfg)
    scaler = GradScaler(device.type, enabled=enable_amp)
    teacher_brain, teacher_value_net = load_teacher_models(
        str(cfg.get('teacher_state_file', '') or '').strip(),
        device,
    )
    teacher_loss_weight = float(cfg.get('teacher_loss_weight', 0.0) or 0.0)
    raw_target_loss_weight = cfg.get('target_loss_weight', 1.0)
    target_loss_weight = 1.0 if raw_target_loss_weight is None else float(raw_target_loss_weight)
    if teacher_brain is not None:
        logging.info(
            'oracle critic teacher distill enabled: source=%s teacher_weight=%.4g target_weight=%.4g',
            display_path(str(cfg.get('teacher_state_file'))),
            teacher_loss_weight,
            target_loss_weight,
        )

    steps = 0
    best_val_loss = math.inf
    init_info = {'source': '', 'loaded': False}

    save_every = int(cfg.get('save_every', 1000) or 1000)
    log_every = int(cfg.get('log_every', 100) or 100)
    val_every_steps = int(cfg.get('val_every_steps', 2000) or 2000)
    max_grad_norm = float(cfg.get('max_grad_norm', config['optim'].get('max_grad_norm', 0.0)) or 0.0)
    zero_sum_weight = float(cfg.get('zero_sum_weight', config.get('value', {}).get('zero_sum_weight', 0.0)) or 0.0)

    if path.exists(state_file) and not args.fresh:
        state = torch.load(state_file, weights_only=False, map_location=device)
        oracle_brain.load_state_dict(state['oracle_brain'])
        value_net.load_state_dict(state['value_net'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        steps = int(state.get('steps', 0))
        best_val_loss = float(state.get('best_val_loss', math.inf))
        best_val_loss = maybe_load_best_val_loss(best_state_file, best_val_loss, device)
        init_info = state.get('init_info', init_info)
        logging.info('resumed oracle critic pretrain from %s at step=%s', state_file, steps)
    else:
        init_state_file = resolve_init_state_file(cfg)
        init_info = maybe_init_from_checkpoint(
            oracle_brain,
            value_net,
            init_state_file,
            device,
            strict_oracle_checkpoint=bool(cfg.get('strict_init_checkpoint', False)),
        )
        logging.info('oracle critic init: %s', summarize_init_info(init_info))

    oracle_brain.train()
    value_net.train()
    writer = SummaryWriter(tensorboard_dir)

    train_loader = make_loader(make_dataset(train_files, cfg, train=True), cfg, train=True)
    stats = []
    pb = tqdm(total=max(save_every, 1), desc='ORACLE')
    while steps < max_steps:
        for obs, invisible_obs, target, _player_id in train_loader:
            if steps >= max_steps:
                break
            steps += 1
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
            target = target.to(dtype=torch.float32, device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model_forward(
                oracle_brain,
                value_net,
                obs,
                invisible_obs,
                enable_amp=enable_amp,
                device_type=device.type,
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
                target_mse = nn.functional.mse_loss(pred, target)
                teacher_mse = (
                    nn.functional.mse_loss(pred, teacher_target)
                    if teacher_target is not None
                    else None
                )
                zero_sum_loss = pred.sum(dim=-1).square().mean()
                loss = pred.new_tensor(0.0)
                if target_loss_weight > 0:
                    loss = loss + target_loss_weight * target_mse
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
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            stats.append(
                batch_metrics(
                    pred,
                    target,
                    objective_loss=loss,
                    target_mse=target_mse,
                    teacher_mse=teacher_mse,
                    zero_sum_loss=zero_sum_loss,
                )
            )
            pb.update(1)

            if steps % log_every == 0:
                metrics = finalize_metrics(stats)
                lr, encoder_lr = lr_summary(optimizer, scheduler)
                logging.info(
                    (
                        'step=%s train_loss=%.6f objective_loss=%.6f train_mae=%.6f '
                        'corr=%.4f zero_sum_mae=%.6f lr=%.3g encoder_lr=%.3g'
                    ),
                    steps,
                    metrics['loss'],
                    metrics.get('objective_loss', metrics['loss']),
                    metrics['mae'],
                    metrics['corr'],
                    metrics['zero_sum_mae'],
                    lr,
                    encoder_lr,
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
                writer.add_scalar('train/objective_loss', metrics.get('objective_loss', metrics['loss']), steps)
                writer.add_scalar('train/target_mse', metrics.get('target_mse', metrics['loss']), steps)
                if 'teacher_mse' in metrics:
                    writer.add_scalar('train/teacher_mse', metrics['teacher_mse'], steps)
                writer.add_scalar('train/zero_sum_loss', metrics.get('zero_sum_loss', 0.0), steps)
                writer.add_scalar('train/mae', metrics['mae'], steps)
                writer.add_scalar('train/corr', metrics['corr'], steps)
                writer.add_scalar('train/zero_sum_mae', metrics['zero_sum_mae'], steps)
                writer.add_scalar('lr', lr, steps)
                writer.add_scalar('encoder_lr', encoder_lr, steps)
                stats.clear()

            if steps % val_every_steps == 0 or steps >= max_steps:
                val_loader = make_loader(make_dataset(val_files, cfg, train=False), cfg, train=False)
                val_metrics = evaluate(
                    oracle_brain,
                    value_net,
                    val_loader,
                    device,
                    enable_amp=enable_amp,
                    max_batches=val_batches,
                )
                logging.info(
                    'step=%s val_loss=%.6f val_mae=%.6f val_corr=%.4f val_zero_sum_mae=%.6f',
                    steps,
                    val_metrics['loss'],
                    val_metrics['mae'],
                    val_metrics['corr'],
                    val_metrics['zero_sum_mae'],
                )
                writer.add_scalar('val/loss', val_metrics['loss'], steps)
                writer.add_scalar('val/mae', val_metrics['mae'], steps)
                writer.add_scalar('val/corr', val_metrics['corr'], steps)
                writer.add_scalar('val/zero_sum_mae', val_metrics['zero_sum_mae'], steps)
                with open(metrics_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'steps': steps, 'val': val_metrics}, sort_keys=True) + '\n')
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    payload = checkpoint_payload(
                        oracle_brain=oracle_brain,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        steps=steps,
                        best_val_loss=best_val_loss,
                        init_info=init_info,
                        cfg=cfg,
                        train_info=train_info,
                    )
                    save_checkpoint(best_state_file, payload)
                    logging.info('saved best oracle critic checkpoint to %s', best_state_file)

            if steps % save_every == 0 or steps >= max_steps:
                payload = checkpoint_payload(
                    oracle_brain=oracle_brain,
                    value_net=value_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    steps=steps,
                    best_val_loss=best_val_loss,
                    init_info=init_info,
                    cfg=cfg,
                    train_info=train_info,
                )
                save_checkpoint(state_file, payload)
                pb.close()
                pb = tqdm(total=max(save_every, 1), desc='ORACLE')

        if steps < max_steps:
            train_loader = make_loader(make_dataset(train_files, cfg, train=True), cfg, train=True)

    writer.close()
    pb.close()


if __name__ == '__main__':
    train()

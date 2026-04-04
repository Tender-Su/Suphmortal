import prelude

import argparse
import json
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from config import config
from engine import MortalEngine, coerce_batch_inputs
from libriichi.arena import OneVsThree
from model import Brain, CategoricalPolicy

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

DEFAULT_1V3_GPU_SEED_COUNTS = {
    'NVIDIA GeForce RTX 5070 Ti': 1024,
    'NVIDIA GeForce RTX 4060 Laptop GPU': 640,
}

DEFAULT_1V3_GPU_SHARD_COUNTS = {
    'NVIDIA GeForce RTX 5070 Ti': 4,
    'NVIDIA GeForce RTX 4060 Laptop GPU': 3,
}

DEFAULT_1V3_GPU_BROKER_SHARD_COUNTS = {
    'NVIDIA GeForce RTX 5070 Ti': 3,
}

DEFAULT_1V3_EXECUTION_MODE = 'process'
DEFAULT_1V3_BROKER_BATCH_WINDOW_MS = 0.0


def _coerce_positive_int(value, *, field_name):
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f'{field_name} must be positive, got {parsed}')
    return parsed


def _resolve_cfg_seed_count(override_cfg, *, source_name):
    if not isinstance(override_cfg, dict):
        return None
    if 'seed_count' in override_cfg:
        return _coerce_positive_int(override_cfg['seed_count'], field_name=f'{source_name}.seed_count')
    if 'games_per_iter' in override_cfg:
        games_per_iter = _coerce_positive_int(
            override_cfg['games_per_iter'],
            field_name=f'{source_name}.games_per_iter',
        )
        if games_per_iter % 4 != 0:
            raise ValueError(f'{source_name}.games_per_iter must be divisible by 4, got {games_per_iter}')
        return games_per_iter // 4
    return None


def _resolve_cfg_shard_count(override_cfg, *, source_name):
    if not isinstance(override_cfg, dict):
        return None
    if 'shard_count' not in override_cfg:
        return None
    return _coerce_positive_int(override_cfg['shard_count'], field_name=f'{source_name}.shard_count')


def resolve_gpu_name(cfg):
    challenger_device = str(cfg.get('challenger', {}).get('device', 'cpu'))
    if not challenger_device.startswith('cuda'):
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.device(challenger_device))


def resolve_seed_count(cfg):
    if (env_override := os.environ.get('MORTAL_1V3_SEED_COUNT')):
        return (
            _coerce_positive_int(env_override, field_name='MORTAL_1V3_SEED_COUNT'),
            f'env MORTAL_1V3_SEED_COUNT={env_override}',
        )

    computer_name = os.environ.get('COMPUTERNAME', '')
    machine_overrides = cfg.get('machine_overrides', {})
    if computer_name:
        machine_seed_count = _resolve_cfg_seed_count(
            machine_overrides.get(computer_name),
            source_name=f'[1v3.machine_overrides.{computer_name}]',
        )
        if machine_seed_count is not None:
            return machine_seed_count, f'config machine override for {computer_name}'

    gpu_name = resolve_gpu_name(cfg)
    gpu_overrides = cfg.get('gpu_overrides', {})
    if gpu_name:
        gpu_seed_count = _resolve_cfg_seed_count(
            gpu_overrides.get(gpu_name),
            source_name=f'[1v3.gpu_overrides.{gpu_name}]',
        )
        if gpu_seed_count is not None:
            return gpu_seed_count, f'config gpu override for {gpu_name}'

        default_seed_count = DEFAULT_1V3_GPU_SEED_COUNTS.get(gpu_name)
        if default_seed_count is not None:
            return default_seed_count, f'built-in gpu default for {gpu_name}'

    seed_count = _resolve_cfg_seed_count(cfg, source_name='[1v3]')
    if seed_count is not None:
        return seed_count, 'config [1v3]'

    games_per_iter = _coerce_positive_int(cfg['games_per_iter'], field_name='[1v3].games_per_iter')
    if games_per_iter % 4 != 0:
        raise ValueError(f'[1v3].games_per_iter must be divisible by 4, got {games_per_iter}')
    return games_per_iter // 4, 'config [1v3].games_per_iter / 4'


def resolve_shard_count(cfg):
    if (env_override := os.environ.get('MORTAL_1V3_SHARD_COUNT')):
        return (
            _coerce_positive_int(env_override, field_name='MORTAL_1V3_SHARD_COUNT'),
            f'env MORTAL_1V3_SHARD_COUNT={env_override}',
        )

    computer_name = os.environ.get('COMPUTERNAME', '')
    machine_overrides = cfg.get('machine_overrides', {})
    if computer_name:
        machine_shard_count = _resolve_cfg_shard_count(
            machine_overrides.get(computer_name),
            source_name=f'[1v3.machine_overrides.{computer_name}]',
        )
        if machine_shard_count is not None:
            return machine_shard_count, f'config machine override for {computer_name}'

    gpu_name = resolve_gpu_name(cfg)
    gpu_overrides = cfg.get('gpu_overrides', {})
    if gpu_name:
        gpu_shard_count = _resolve_cfg_shard_count(
            gpu_overrides.get(gpu_name),
            source_name=f'[1v3.gpu_overrides.{gpu_name}]',
        )
        if gpu_shard_count is not None:
            return gpu_shard_count, f'config gpu override for {gpu_name}'

        default_shard_count = DEFAULT_1V3_GPU_SHARD_COUNTS.get(gpu_name)
        if default_shard_count is not None:
            return default_shard_count, f'built-in gpu default for {gpu_name}'

    shard_count = _resolve_cfg_shard_count(cfg, source_name='[1v3]')
    if shard_count is not None:
        return shard_count, 'config [1v3]'

    return 1, 'implicit default shard_count=1'


def _resolve_cfg_broker_shard_count(override_cfg, *, source_name):
    if not isinstance(override_cfg, dict):
        return None
    if 'broker_shard_count' not in override_cfg:
        return None
    return _coerce_positive_int(override_cfg['broker_shard_count'], field_name=f'{source_name}.broker_shard_count')


def resolve_broker_shard_count(cfg, base_shard_count):
    if (env_override := os.environ.get('MORTAL_1V3_BROKER_SHARD_COUNT')):
        return (
            _coerce_positive_int(env_override, field_name='MORTAL_1V3_BROKER_SHARD_COUNT'),
            f'env MORTAL_1V3_BROKER_SHARD_COUNT={env_override}',
        )

    computer_name = os.environ.get('COMPUTERNAME', '')
    machine_overrides = cfg.get('machine_overrides', {})
    if computer_name:
        machine_broker_shard_count = _resolve_cfg_broker_shard_count(
            machine_overrides.get(computer_name),
            source_name=f'[1v3.machine_overrides.{computer_name}]',
        )
        if machine_broker_shard_count is not None:
            return machine_broker_shard_count, f'config machine broker override for {computer_name}'

    gpu_name = resolve_gpu_name(cfg)
    gpu_overrides = cfg.get('gpu_overrides', {})
    if gpu_name:
        gpu_broker_shard_count = _resolve_cfg_broker_shard_count(
            gpu_overrides.get(gpu_name),
            source_name=f'[1v3.gpu_overrides.{gpu_name}]',
        )
        if gpu_broker_shard_count is not None:
            return gpu_broker_shard_count, f'config gpu broker override for {gpu_name}'

        default_broker_shard_count = DEFAULT_1V3_GPU_BROKER_SHARD_COUNTS.get(gpu_name)
        if default_broker_shard_count is not None:
            return default_broker_shard_count, f'built-in broker gpu default for {gpu_name}'

    if 'broker_shard_count' in cfg:
        return (
            _coerce_positive_int(cfg['broker_shard_count'], field_name='[1v3].broker_shard_count'),
            'config [1v3].broker_shard_count',
        )

    return base_shard_count, 'inherit base shard_count'


def plan_shards(total_seed_count, requested_shard_count):
    total_seed_count = _coerce_positive_int(total_seed_count, field_name='total_seed_count')
    requested_shard_count = _coerce_positive_int(requested_shard_count, field_name='requested_shard_count')
    shard_count = min(total_seed_count, requested_shard_count)
    base, remainder = divmod(total_seed_count, shard_count)
    return [base + (1 if index < remainder else 0) for index in range(shard_count)]


def resolve_requested_execution_mode(cfg):
    requested_mode = str(
        os.environ.get('MORTAL_1V3_EXECUTION_MODE', cfg.get('execution_mode', DEFAULT_1V3_EXECUTION_MODE))
    ).strip().lower()
    if requested_mode not in {'auto', 'process', 'broker'}:
        raise ValueError(
            'invalid 1v3 execution mode: '
            f'{requested_mode!r}; expected auto/process/broker'
        )
    return requested_mode


def resolve_execution_mode(cfg, effective_shard_count, broker_effective_shard_count=None):
    base_effective_shard_count = _coerce_positive_int(effective_shard_count, field_name='effective_shard_count')
    if broker_effective_shard_count is None:
        broker_effective_shard_count = base_effective_shard_count
    else:
        broker_effective_shard_count = _coerce_positive_int(
            broker_effective_shard_count,
            field_name='broker_effective_shard_count',
        )
    requested_mode = resolve_requested_execution_mode(cfg)

    if requested_mode == 'process':
        if base_effective_shard_count <= 1:
            return 'process', 'config/env process (single-shard runs use the direct path)'
        return 'process', 'config/env process'

    if requested_mode == 'broker':
        if broker_effective_shard_count <= 1:
            return 'process', 'config/env broker requested, but broker shard_count resolved to a single-shard direct path'
        broker_supported, broker_reason = can_use_local_broker(cfg, broker_effective_shard_count)
        if not broker_supported:
            raise ValueError(f'1v3 broker mode is unavailable: {broker_reason}')
        return 'broker', f'config/env broker ({broker_reason})'

    if broker_effective_shard_count > 1:
        broker_supported, broker_reason = can_use_local_broker(cfg, broker_effective_shard_count)
        if broker_supported:
            return 'broker', f'auto -> broker ({broker_reason})'
    else:
        broker_reason = 'broker shard_count resolved to a single-shard direct path'

    if base_effective_shard_count <= 1:
        if broker_effective_shard_count > 1:
            return 'process', f'auto -> process ({broker_reason}; single-shard runs use the direct path)'
        return 'process', 'auto -> process (single-shard runs use the direct path)'
    return 'process', f'auto -> process ({broker_reason})'


def can_use_local_broker(cfg, effective_shard_count):
    if effective_shard_count <= 1:
        return False, 'requires shard_count > 1'
    if not torch.cuda.is_available():
        return False, 'CUDA is unavailable'
    challenger_device = normalize_cuda_device(cfg.get('challenger', {}).get('device', 'cpu'))
    champion_device = normalize_cuda_device(cfg.get('champion', {}).get('device', 'cpu'))
    if challenger_device is None:
        return False, f"challenger device {cfg.get('challenger', {}).get('device', 'cpu')!r} is not CUDA"
    if champion_device is None:
        return False, f"champion device {cfg.get('champion', {}).get('device', 'cpu')!r} is not CUDA"
    if challenger_device != champion_device:
        return False, 'challenger and champion must share the same CUDA device'
    return True, f'single-device {challenger_device}'


def normalize_cuda_device(device_value):
    device = torch.device(str(device_value))
    if device.type != 'cuda':
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return f'cuda:{device_index}'


def resolve_broker_batch_window_ms(cfg):
    if (env_override := os.environ.get('MORTAL_1V3_BROKER_BATCH_WINDOW_MS')):
        return float(env_override)
    return float(cfg.get('broker_batch_window_ms', DEFAULT_1V3_BROKER_BATCH_WINDOW_MS))


class BrokerClientEngine:
    engine_type = 'mortal'

    def __init__(self, *, broker, engine_role, metadata):
        self._broker = broker
        self._engine_role = engine_role
        self.name = metadata['name']
        self.is_oracle = bool(metadata['is_oracle'])
        self.version = int(metadata['version'])
        self.enable_quick_eval = bool(metadata['enable_quick_eval'])
        self.enable_rule_based_agari_guard = bool(metadata['enable_rule_based_agari_guard'])
        self.enable_metadata = bool(metadata['enable_metadata'])

    def react_batch(self, obs, masks, invisible_obs):
        obs, masks, invisible_obs = coerce_batch_inputs(obs, masks, invisible_obs)
        return self._broker.submit(
            engine_role=self._engine_role,
            obs=obs,
            masks=masks,
            invisible_obs=invisible_obs,
            action_only=False,
        )

    def react_batch_action_only(self, obs, masks, invisible_obs):
        obs, masks, invisible_obs = coerce_batch_inputs(obs, masks, invisible_obs)
        return self._broker.submit(
            engine_role=self._engine_role,
            obs=obs,
            masks=masks,
            invisible_obs=invisible_obs,
            action_only=True,
        )


class InferenceBroker:
    def __init__(self, *, engine_map, engine_metadata, batch_window_ms):
        self._engine_map = engine_map
        self._engine_metadata = engine_metadata
        self._batch_window_s = max(0.0, float(batch_window_ms)) / 1000.0
        self._role_streams = {}
        self._close_lock = threading.Lock()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_requests = {}
        self._stats = {
            'request_count': 0,
            'request_wait_s': 0.0,
            'roles': {},
        }
        self._role_workers = {}
        for engine_role in engine_map:
            engine = engine_map[engine_role]
            if getattr(getattr(engine, 'device', None), 'type', None) == 'cuda':
                self._role_streams[engine_role] = torch.cuda.Stream(device=engine.device)
            self._pending_requests[engine_role] = 0
            self._stats['roles'][engine_role] = {
                'batch_count': 0,
                'batch_wait_s': 0.0,
                'batch_items': 0,
                'batch_max_items': 0,
                'infer_s': 0.0,
            }
            stop_token = object()
            request_queue = queue.Queue()
            thread = threading.Thread(
                target=self._run_role_loop,
                args=(engine_role, request_queue, stop_token),
                name=f'mahjongai-1v3-broker-{engine_role}',
                daemon=True,
            )
            thread.start()
            self._role_workers[engine_role] = {
                'queue': request_queue,
                'stop_token': stop_token,
                'thread': thread,
            }

    @classmethod
    def from_cfg(cls, cfg):
        enable_metadata = resolve_enable_metadata(cfg)
        challenger_bundle = load_mortal_engine_bundle(cfg['challenger'], enable_metadata=enable_metadata)
        champion_bundle = load_mortal_engine_bundle(cfg['champion'], enable_metadata=enable_metadata)
        return cls(
            engine_map={
                'challenger': challenger_bundle['engine'],
                'champion': champion_bundle['engine'],
            },
            engine_metadata={
                'challenger': challenger_bundle['metadata'],
                'champion': champion_bundle['metadata'],
            },
            batch_window_ms=resolve_broker_batch_window_ms(cfg),
        )

    def make_eval_context(self):
        return {
            'engine_chal': BrokerClientEngine(
                broker=self,
                engine_role='challenger',
                metadata=self._engine_metadata['challenger'],
            ),
            'engine_cham': BrokerClientEngine(
                broker=self,
                engine_role='champion',
                metadata=self._engine_metadata['champion'],
            ),
        }

    def submit(self, *, engine_role, obs, masks, invisible_obs, action_only):
        submit_started = time.perf_counter()
        request = {
            'engine_role': str(engine_role),
            'obs': obs,
            'masks': masks,
            'invisible_obs': invisible_obs,
            'action_only': bool(action_only),
            'event': threading.Event(),
            'response': None,
            'error': None,
        }
        with self._close_lock:
            if self._closed:
                raise RuntimeError('1v3 inference broker is closed')
            with self._pending_lock:
                self._pending_requests[engine_role] += 1
            self._role_workers[engine_role]['queue'].put(request)
        try:
            request['event'].wait()
            submit_elapsed = time.perf_counter() - submit_started
            with self._stats_lock:
                self._stats['request_count'] += 1
                self._stats['request_wait_s'] += submit_elapsed
            if request['error'] is not None:
                raise RuntimeError(f'1v3 broker inference failed for {engine_role}: {request["error"]}')
            return request['response']
        finally:
            with self._pending_lock:
                self._pending_requests[engine_role] -= 1

    def close(self):
        workers = None
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            workers = tuple(self._role_workers.values())
            for worker in workers:
                worker['queue'].put(worker['stop_token'])
        for worker in workers:
            worker['thread'].join(timeout=10)

    def snapshot_stats(self):
        with self._stats_lock:
            return dict(self._stats)

    def _run_role_loop(self, engine_role, request_queue, stop_token):
        while True:
            first_item = request_queue.get()
            if first_item is stop_token:
                return
            batch_items = [first_item]
            self._drain_ready_requests(request_queue, stop_token, batch_items)
            wait_started = time.perf_counter()
            if self._batch_window_s > 0 and len(batch_items) == 1 and self._should_wait_for_more(engine_role):
                deadline = wait_started + self._batch_window_s
                while True:
                    timeout = deadline - time.perf_counter()
                    if timeout <= 0:
                        break
                    try:
                        extra_item = request_queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if extra_item is stop_token:
                        request_queue.put(stop_token)
                        break
                    batch_items.append(extra_item)
                    self._drain_ready_requests(request_queue, stop_token, batch_items)
                    break
            batch_wait_s = time.perf_counter() - wait_started
            with self._stats_lock:
                role_stats = self._stats['roles'][engine_role]
                role_stats['batch_count'] += 1
                role_stats['batch_wait_s'] += batch_wait_s
                role_stats['batch_items'] += len(batch_items)
                role_stats['batch_max_items'] = max(role_stats['batch_max_items'], len(batch_items))

            try:
                self._process_role_batch(engine_role, batch_items)
            except BaseException as exc:
                error_text = f'{exc}\n{traceback.format_exc()}'
                for item in batch_items:
                    item['error'] = error_text
                    item['event'].set()

    def _drain_ready_requests(self, request_queue, stop_token, batch_items):
        while True:
            try:
                extra_item = request_queue.get_nowait()
            except queue.Empty:
                return
            if extra_item is stop_token:
                request_queue.put(stop_token)
                return
            batch_items.append(extra_item)

    def _process_role_batch(self, engine_role, items):
        engine = self._engine_map[engine_role]
        action_only = bool(items[0]['action_only'])
        if len(items) == 1:
            obs = items[0]['obs']
            masks = items[0]['masks']
            invisible_obs = items[0]['invisible_obs']
        else:
            obs = np.ascontiguousarray(np.concatenate([item['obs'] for item in items], axis=0), dtype=np.float32)
            masks = np.ascontiguousarray(np.concatenate([item['masks'] for item in items], axis=0), dtype=np.bool_)
            if items[0]['invisible_obs'] is None:
                invisible_obs = None
            else:
                invisible_obs = np.ascontiguousarray(
                    np.concatenate([item['invisible_obs'] for item in items], axis=0),
                    dtype=np.float32,
                )

        infer_started = time.perf_counter()
        role_stream = self._role_streams.get(engine_role)
        if role_stream is None:
            if action_only:
                actions, alt_actions = engine.react_batch_action_only(obs, masks, invisible_obs)
            else:
                actions, q_out, masks_out, is_greedy = engine.react_batch(obs, masks, invisible_obs)
        else:
            with torch.cuda.stream(role_stream):
                if action_only:
                    actions, alt_actions = engine.react_batch_action_only(obs, masks, invisible_obs)
                else:
                    actions, q_out, masks_out, is_greedy = engine.react_batch(obs, masks, invisible_obs)
        infer_elapsed = time.perf_counter() - infer_started
        with self._stats_lock:
            self._stats['roles'][engine_role]['infer_s'] += infer_elapsed
        start = 0
        for item in items:
            batch_size = int(item['obs'].shape[0])
            stop = start + batch_size
            if action_only:
                item['response'] = (
                    actions[start:stop],
                    alt_actions[start:stop],
                )
            else:
                item['response'] = (
                    actions[start:stop],
                    q_out[start:stop],
                    masks_out[start:stop],
                    is_greedy[start:stop],
                )
            item['event'].set()
            start = stop

    def _should_wait_for_more(self, engine_role):
        with self._pending_lock:
            return self._pending_requests[engine_role] > 1


def load_mortal_engine_bundle(engine_cfg, *, enable_metadata=True):
    state = torch.load(engine_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
    saved_cfg = state['config']
    version = saved_cfg['control'].get('version', 1)
    conv_channels = saved_cfg['resnet']['conv_channels']
    num_blocks = saved_cfg['resnet']['num_blocks']
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels, Norm='GN').eval()
    dqn = CategoricalPolicy().eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['policy_net'])
    if engine_cfg['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine = MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(engine_cfg['device']),
        enable_amp=engine_cfg['enable_amp'],
        enable_rule_based_agari_guard=engine_cfg['enable_rule_based_agari_guard'],
        enable_metadata=bool(engine_cfg.get('enable_metadata', enable_metadata)),
        name=engine_cfg['name'],
    )
    return {
        'engine': engine,
        'metadata': {
            'name': engine_cfg['name'],
            'is_oracle': False,
            'version': version,
            'enable_quick_eval': bool(engine.enable_quick_eval),
            'enable_rule_based_agari_guard': bool(engine.enable_rule_based_agari_guard),
            'enable_metadata': bool(engine.enable_metadata),
        },
    }


def load_mortal_engine(engine_cfg, *, enable_metadata=True):
    return load_mortal_engine_bundle(engine_cfg, enable_metadata=enable_metadata)['engine']


def summarize_rankings(rankings):
    rankings = np.array(rankings, dtype=np.int64)
    avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
    avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
    return rankings, float(avg_rank), float(avg_pt)


def resolve_enable_metadata(cfg):
    if 'enable_metadata' in cfg:
        return bool(cfg['enable_metadata'])
    return bool(cfg.get('log_dir'))


def load_eval_engines(cfg, *, broker=None, enable_metadata=None):
    if enable_metadata is None:
        enable_metadata = resolve_enable_metadata(cfg)
    if broker is not None:
        return broker.make_eval_context()
    engine_cham = load_mortal_engine(cfg['champion'], enable_metadata=enable_metadata)
    engine_chal = load_mortal_engine(cfg['challenger'], enable_metadata=enable_metadata)
    return {
        'engine_cham': engine_cham,
        'engine_chal': engine_chal,
    }


def run_eval_once(*, cfg, seed_start, seed_key, seed_count, log_dir, disable_progress_bar, eval_context=None):
    if eval_context is None:
        eval_context = load_eval_engines(cfg)
    engine_cham = eval_context['engine_cham']
    engine_chal = eval_context['engine_chal']
    env = OneVsThree(
        disable_progress_bar=disable_progress_bar,
        log_dir=log_dir,
    )
    rankings = env.py_vs_py(
        challenger=engine_chal,
        champion=engine_cham,
        seed_start=(seed_start, seed_key),
        seed_count=seed_count,
    )
    return rankings


def build_worker_command(*, seed_start, seed_key, seed_count, result_json, log_dir, shard_index):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        '--worker',
        '--seed-start',
        str(seed_start),
        '--seed-key',
        str(seed_key),
        '--seed-count',
        str(seed_count),
        '--result-json',
        str(result_json),
        '--shard-index',
        str(shard_index),
        '--disable-progress-bar',
    ]
    if log_dir is not None:
        command.extend(['--log-dir', str(log_dir)])
    return command


def normalize_child_env():
    env = os.environ.copy()
    config_override = env.get('MORTAL_CFG')
    if config_override:
        env['MORTAL_CFG'] = str(Path(config_override).resolve())
    return env


def build_worker_loop_command(*, shard_index, disable_progress_bar):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        '--worker-loop',
        '--shard-index',
        str(shard_index),
    ]
    if disable_progress_bar:
        command.append('--disable-progress-bar')
    return command


def read_worker_loop_message(worker_entry):
    line = worker_entry['process'].stdout.readline()
    if line == '':
        exit_code = worker_entry['process'].poll()
        log_excerpt = ''
        if worker_entry['log_path'].exists():
            log_excerpt = worker_entry['log_path'].read_text(encoding='utf-8', errors='replace')
        raise RuntimeError(
            '1v3 shard worker closed stdout unexpectedly: '
            f"shard={worker_entry['shard_index']} exit_code={exit_code} log={worker_entry['log_path']}\n{log_excerpt}"
        )
    return json.loads(line)


def run_worker_loop(args):
    cfg = config['1v3']
    eval_context = load_eval_engines(cfg)
    print(
        json.dumps({
            'kind': 'ready',
            'shard_index': args.shard_index,
        }),
        flush=True,
    )
    for raw_line in sys.stdin:
        task = json.loads(raw_line)
        if str(task.get('kind', '')) == 'stop':
            break
        if str(task.get('kind', '')) != 'run':
            continue
        try:
            rankings = run_eval_once(
                cfg=cfg,
                seed_start=int(task['seed_start']),
                seed_key=int(task['seed_key']),
                seed_count=int(task['seed_count']),
                log_dir=task.get('log_dir'),
                disable_progress_bar=args.disable_progress_bar,
                eval_context=eval_context,
            )
            rankings, avg_rank, avg_pt = summarize_rankings(rankings)
            print(
                json.dumps({
                    'kind': 'result',
                    'iter_index': int(task['iter_index']),
                    'shard_index': int(args.shard_index),
                    'seed_start': int(task['seed_start']),
                    'seed_count': int(task['seed_count']),
                    'rankings': rankings.tolist(),
                    'avg_rank': avg_rank,
                    'avg_pt': avg_pt,
                }),
                flush=True,
            )
        except BaseException as exc:
            print(
                json.dumps({
                    'kind': 'task_error',
                    'iter_index': int(task['iter_index']),
                    'shard_index': int(args.shard_index),
                    'seed_start': int(task['seed_start']),
                    'seed_count': int(task['seed_count']),
                    'error': str(exc),
                    'traceback': traceback.format_exc(),
                }),
                flush=True,
            )
            break


def run_threaded_worker_loop(*, cfg, shard_index, disable_progress_bar, task_queue, result_queue, broker):
    try:
        eval_context = load_eval_engines(cfg, broker=broker)
        result_queue.put({
            'kind': 'ready',
            'shard_index': shard_index,
        })
    except BaseException as exc:
        result_queue.put({
            'kind': 'startup_error',
            'shard_index': shard_index,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
        return

    while True:
        task = task_queue.get()
        if task is None or str(task.get('kind', '')) == 'stop':
            return
        if str(task.get('kind', '')) != 'run':
            continue
        try:
            rankings = run_eval_once(
                cfg=cfg,
                seed_start=int(task['seed_start']),
                seed_key=int(task['seed_key']),
                seed_count=int(task['seed_count']),
                log_dir=task.get('log_dir'),
                disable_progress_bar=disable_progress_bar,
                eval_context=eval_context,
            )
            rankings, avg_rank, avg_pt = summarize_rankings(rankings)
            result_queue.put({
                'kind': 'result',
                'iter_index': int(task['iter_index']),
                'shard_index': shard_index,
                'seed_start': int(task['seed_start']),
                'seed_count': int(task['seed_count']),
                'rankings': rankings.tolist(),
                'avg_rank': avg_rank,
                'avg_pt': avg_pt,
            })
        except BaseException as exc:
            result_queue.put({
                'kind': 'task_error',
                'iter_index': int(task['iter_index']),
                'shard_index': shard_index,
                'seed_start': int(task['seed_start']),
                'seed_count': int(task['seed_count']),
                'error': str(exc),
                'traceback': traceback.format_exc(),
            })
            return


def _stop_process_shard_workers(shard_worker_pool):
    if shard_worker_pool is None:
        return
    for entry in shard_worker_pool['workers']:
        try:
            if entry['process'].poll() is None and entry['process'].stdin:
                entry['process'].stdin.write(json.dumps({'kind': 'stop'}) + '\n')
                entry['process'].stdin.flush()
        except Exception:
            continue
    for entry in shard_worker_pool['workers']:
        process = entry['process']
        if process.stdin:
            try:
                process.stdin.close()
            except Exception:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
    for entry in shard_worker_pool['workers']:
        process = entry['process']
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass
        try:
            entry['log_handle'].close()
        except Exception:
            pass


def _stop_threaded_shard_workers(shard_worker_pool):
    if shard_worker_pool is None:
        return True
    for entry in shard_worker_pool['workers']:
        entry['task_queue'].put({'kind': 'stop'})
    all_stopped = True
    for entry in shard_worker_pool['workers']:
        entry['thread'].join(timeout=10)
        if entry['thread'].is_alive():
            all_stopped = False
    return all_stopped


def _close_broker_after_thread_workers_stop(broker, workers):
    try:
        for entry in workers:
            entry['thread'].join()
    finally:
        broker.close()


def stop_persistent_shard_workers(shard_worker_pool):
    if shard_worker_pool is None:
        return
    if shard_worker_pool.get('mode') == 'broker':
        broker = shard_worker_pool.get('broker')
        workers_stopped = _stop_threaded_shard_workers(shard_worker_pool)
        if broker is None:
            return
        if workers_stopped:
            broker.close()
            return
        reaper = threading.Thread(
            target=_close_broker_after_thread_workers_stop,
            args=(broker, tuple(shard_worker_pool['workers'])),
            name='mahjongai-1v3-broker-close-reaper',
            daemon=True,
        )
        shard_worker_pool['broker_close_reaper'] = reaper
        reaper.start()
        return

    try:
        _stop_process_shard_workers(shard_worker_pool)
    finally:
        broker = shard_worker_pool.get('broker')
        if broker is not None:
            broker.close()


def _start_process_shard_workers(*, cfg, shard_count, disable_progress_bar):
    runtime_root = Path(tempfile.mkdtemp(prefix='mahjongai_1v3_worker_pool_'))
    child_env = normalize_child_env()
    workers = []
    for shard_index in range(shard_count):
        log_path = runtime_root / f'shard_{shard_index:02d}.log'
        log_handle = log_path.open('w', encoding='utf-8', newline='\n')
        process = subprocess.Popen(
            build_worker_loop_command(
                shard_index=shard_index,
                disable_progress_bar=disable_progress_bar,
            ),
            cwd=str(SCRIPT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            env=child_env,
        )
        workers.append({
            'shard_index': shard_index,
            'process': process,
            'log_path': log_path,
            'log_handle': log_handle,
        })

    try:
        for entry in workers:
            message = read_worker_loop_message(entry)
            if str(message.get('kind', '')) != 'ready':
                raise RuntimeError(
                    'unexpected shard worker startup message: '
                    f"shard={entry['shard_index']} message={message}"
                )
    except BaseException:
        stop_persistent_shard_workers({
            'workers': workers,
        })
        raise

    return {
        'mode': 'process',
        'workers': workers,
        'runtime_root': runtime_root,
    }


def _start_threaded_shard_workers(*, cfg, shard_count, disable_progress_bar, broker):
    runtime_root = Path(tempfile.mkdtemp(prefix='mahjongai_1v3_broker_pool_'))
    result_queue = queue.Queue()
    workers = []
    for shard_index in range(shard_count):
        task_queue = queue.Queue()
        thread = threading.Thread(
            target=run_threaded_worker_loop,
            kwargs={
                'cfg': cfg,
                'shard_index': shard_index,
                'disable_progress_bar': disable_progress_bar,
                'task_queue': task_queue,
                'result_queue': result_queue,
                'broker': broker,
            },
            name=f'mahjongai-1v3-shard-{shard_index:02d}',
            daemon=True,
        )
        thread.start()
        workers.append({
            'shard_index': shard_index,
            'thread': thread,
            'task_queue': task_queue,
        })

    try:
        for _ in range(shard_count):
            message = result_queue.get(timeout=60)
            kind = str(message.get('kind', ''))
            if kind != 'ready':
                raise RuntimeError(
                    'unexpected threaded shard worker startup message: '
                    f'{message}'
                )
    except BaseException:
        _stop_threaded_shard_workers({
            'workers': workers,
        })
        raise

    return {
        'mode': 'broker',
        'workers': workers,
        'runtime_root': runtime_root,
        'result_queue': result_queue,
        'broker': broker,
    }


def start_persistent_shard_workers(*, cfg, shard_count, disable_progress_bar, execution_mode='process', broker=None):
    if execution_mode == 'broker':
        if broker is None:
            raise ValueError('broker execution mode requires an InferenceBroker instance')
        return _start_threaded_shard_workers(
            cfg=cfg,
            shard_count=shard_count,
            disable_progress_bar=disable_progress_bar,
            broker=broker,
        )
    return _start_process_shard_workers(
        cfg=cfg,
        shard_count=shard_count,
        disable_progress_bar=disable_progress_bar,
    )


def _run_process_sharded_iteration_with_workers(
    *,
    iter_index,
    seed_start,
    seed_key,
    shard_seed_counts,
    log_dir,
    shard_worker_pool,
):
    runtime_root = Path(tempfile.mkdtemp(prefix=f'mahjongai_1v3_iter_{iter_index:04d}_'))
    workers = shard_worker_pool['workers']
    if len(workers) != len(shard_seed_counts):
        raise ValueError(
            f'shard worker count mismatch: workers={len(workers)} shard_seed_counts={len(shard_seed_counts)}'
        )

    seed_cursor = seed_start
    for worker_entry, shard_seed_count in zip(workers, shard_seed_counts, strict=True):
        child_log_dir = None
        if log_dir is not None:
            child_log_dir = str(Path(log_dir) / f'iter_{iter_index:04d}' / f'shard_{worker_entry["shard_index"]:02d}')
        worker_entry['process'].stdin.write(json.dumps({
            'kind': 'run',
            'iter_index': iter_index,
            'seed_start': seed_cursor,
            'seed_key': seed_key,
            'seed_count': shard_seed_count,
            'log_dir': child_log_dir,
        }) + '\n')
        worker_entry['process'].stdin.flush()
        seed_cursor += shard_seed_count

    shard_payloads: dict[int, dict] = {}
    for worker_entry in workers:
        message = read_worker_loop_message(worker_entry)
        kind = str(message.get('kind', ''))
        if kind != 'result' and kind != 'task_error':
            raise RuntimeError(f'unexpected shard worker message: {message}')
        if int(message.get('iter_index', -1)) != iter_index:
            raise RuntimeError(f'unexpected shard worker iter index: {message}')
        shard_index = int(message['shard_index'])
        if kind == 'task_error':
            log_path = runtime_root / f'shard_{shard_index:02d}.log'
            log_path.write_text(str(message.get('traceback') or message.get('error') or ''), encoding='utf-8')
            raise RuntimeError(
                '1v3 shard failed: '
                f"iter={iter_index} shard={shard_index} seed_start={message['seed_start']} "
                f"seed_count={message['seed_count']} log={log_path}"
            )
        shard_payloads[shard_index] = {
            'rankings': list(message['rankings']),
            'avg_rank': float(message['avg_rank']),
            'avg_pt': float(message['avg_pt']),
        }
        result_path = runtime_root / f'shard_{shard_index:02d}.json'
        result_path.write_text(json.dumps(shard_payloads[shard_index], ensure_ascii=False, indent=2), encoding='utf-8')

    total_rankings = np.zeros(4, dtype=np.int64)
    for shard_index in range(len(workers)):
        total_rankings += np.array(shard_payloads[shard_index]['rankings'], dtype=np.int64)
    rankings, avg_rank, avg_pt = summarize_rankings(total_rankings)
    return rankings, avg_rank, avg_pt, runtime_root


def _run_threaded_sharded_iteration_with_workers(
    *,
    iter_index,
    seed_start,
    seed_key,
    shard_seed_counts,
    log_dir,
    shard_worker_pool,
):
    runtime_root = Path(tempfile.mkdtemp(prefix=f'mahjongai_1v3_iter_{iter_index:04d}_'))
    workers = shard_worker_pool['workers']
    result_queue = shard_worker_pool['result_queue']
    if len(workers) != len(shard_seed_counts):
        raise ValueError(
            f'shard worker count mismatch: workers={len(workers)} shard_seed_counts={len(shard_seed_counts)}'
        )

    seed_cursor = seed_start
    for worker_entry, shard_seed_count in zip(workers, shard_seed_counts, strict=True):
        child_log_dir = None
        if log_dir is not None:
            child_log_dir = str(Path(log_dir) / f'iter_{iter_index:04d}' / f'shard_{worker_entry["shard_index"]:02d}')
        worker_entry['task_queue'].put({
            'kind': 'run',
            'iter_index': iter_index,
            'seed_start': seed_cursor,
            'seed_key': seed_key,
            'seed_count': shard_seed_count,
            'log_dir': child_log_dir,
        })
        seed_cursor += shard_seed_count

    shard_payloads = {}
    for _ in workers:
        message = result_queue.get()
        kind = str(message.get('kind', ''))
        if kind != 'result' and kind != 'task_error':
            raise RuntimeError(f'unexpected threaded shard worker message: {message}')
        if int(message.get('iter_index', -1)) != iter_index:
            raise RuntimeError(f'unexpected threaded shard worker iter index: {message}')
        shard_index = int(message['shard_index'])
        if kind == 'task_error':
            log_path = runtime_root / f'shard_{shard_index:02d}.log'
            log_path.write_text(str(message.get('traceback') or message.get('error') or ''), encoding='utf-8')
            raise RuntimeError(
                '1v3 broker shard failed: '
                f"iter={iter_index} shard={shard_index} seed_start={message['seed_start']} "
                f"seed_count={message['seed_count']} log={log_path}"
            )
        shard_payloads[shard_index] = {
            'rankings': list(message['rankings']),
            'avg_rank': float(message['avg_rank']),
            'avg_pt': float(message['avg_pt']),
        }
        result_path = runtime_root / f'shard_{shard_index:02d}.json'
        result_path.write_text(json.dumps(shard_payloads[shard_index], ensure_ascii=False, indent=2), encoding='utf-8')

    total_rankings = np.zeros(4, dtype=np.int64)
    for shard_index in range(len(workers)):
        total_rankings += np.array(shard_payloads[shard_index]['rankings'], dtype=np.int64)
    rankings, avg_rank, avg_pt = summarize_rankings(total_rankings)
    return rankings, avg_rank, avg_pt, runtime_root


def run_sharded_iteration_with_workers(
    *,
    iter_index,
    seed_start,
    seed_key,
    shard_seed_counts,
    log_dir,
    shard_worker_pool,
):
    if shard_worker_pool.get('mode') == 'broker':
        return _run_threaded_sharded_iteration_with_workers(
            iter_index=iter_index,
            seed_start=seed_start,
            seed_key=seed_key,
            shard_seed_counts=shard_seed_counts,
            log_dir=log_dir,
            shard_worker_pool=shard_worker_pool,
        )
    return _run_process_sharded_iteration_with_workers(
        iter_index=iter_index,
        seed_start=seed_start,
        seed_key=seed_key,
        shard_seed_counts=shard_seed_counts,
        log_dir=log_dir,
        shard_worker_pool=shard_worker_pool,
    )


def run_sharded_iteration(*, cfg, iter_index, seed_start, seed_key, seed_count, shard_seed_counts, log_dir):
    runtime_root = Path(tempfile.mkdtemp(prefix=f'mahjongai_1v3_iter_{iter_index:04d}_'))
    child_env = normalize_child_env()
    processes = []
    seed_cursor = seed_start

    try:
        for shard_index, shard_seed_count in enumerate(shard_seed_counts):
            result_json = runtime_root / f'shard_{shard_index:02d}.json'
            log_path = runtime_root / f'shard_{shard_index:02d}.log'
            child_log_dir = None
            if log_dir is not None:
                child_log_dir = Path(log_dir) / f'iter_{iter_index:04d}' / f'shard_{shard_index:02d}'
            command = build_worker_command(
                seed_start=seed_cursor,
                seed_key=seed_key,
                seed_count=shard_seed_count,
                result_json=result_json,
                log_dir=child_log_dir,
                shard_index=shard_index,
            )
            log_handle = log_path.open('w', encoding='utf-8', newline='\n')
            process = subprocess.Popen(
                command,
                cwd=str(SCRIPT_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            log_handle.close()
            processes.append(
                {
                    'process': process,
                    'result_json': result_json,
                    'log_path': log_path,
                    'seed_start': seed_cursor,
                    'seed_count': shard_seed_count,
                    'shard_index': shard_index,
                }
            )
            seed_cursor += shard_seed_count

        shard_payloads = []
        for entry in processes:
            return_code = entry['process'].wait()
            if return_code != 0:
                raise RuntimeError(
                    '1v3 shard failed: '
                    f"iter={iter_index} shard={entry['shard_index']} seed_start={entry['seed_start']} "
                    f"seed_count={entry['seed_count']} log={entry['log_path']}"
                )
            with entry['result_json'].open(encoding='utf-8') as f:
                shard_payloads.append(json.load(f))

        total_rankings = np.zeros(4, dtype=np.int64)
        for payload in shard_payloads:
            total_rankings += np.array(payload['rankings'], dtype=np.int64)
        rankings, avg_rank, avg_pt = summarize_rankings(total_rankings)
        return rankings, avg_rank, avg_pt, runtime_root
    except BaseException:
        for entry in processes:
            process = entry['process']
            if process.poll() is None:
                process.terminate()
        for entry in processes:
            process = entry['process']
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise


def run_worker(args):
    cfg = config['1v3']
    rankings = run_eval_once(
        cfg=cfg,
        seed_start=args.seed_start,
        seed_key=args.seed_key,
        seed_count=args.seed_count,
        log_dir=args.log_dir,
        disable_progress_bar=args.disable_progress_bar,
    )
    rankings, avg_rank, avg_pt = summarize_rankings(rankings)
    payload = {
        'seed_start': args.seed_start,
        'seed_key': args.seed_key,
        'seed_count': args.seed_count,
        'shard_index': args.shard_index,
        'rankings': rankings.tolist(),
        'avg_rank': avg_rank,
        'avg_pt': avg_pt,
    }
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def run_main(args):
    cfg = config['1v3']
    seeds_per_iter, seed_count_source = resolve_seed_count(cfg)
    games_per_iter = seeds_per_iter * 4
    shard_count, shard_count_source = resolve_shard_count(cfg)
    requested_execution_mode = resolve_requested_execution_mode(cfg)
    shard_seed_counts = plan_shards(seeds_per_iter, shard_count)
    effective_shard_count = len(shard_seed_counts)
    broker_shard_source = shard_count_source
    broker_shard_seed_counts = shard_seed_counts
    broker_effective_shard_count = effective_shard_count
    if requested_execution_mode in {'auto', 'broker'}:
        broker_shard_count, broker_shard_source = resolve_broker_shard_count(cfg, shard_count)
        broker_shard_seed_counts = plan_shards(seeds_per_iter, broker_shard_count)
        broker_effective_shard_count = len(broker_shard_seed_counts)
    execution_mode, execution_mode_source = resolve_execution_mode(
        cfg,
        effective_shard_count,
        broker_effective_shard_count=broker_effective_shard_count,
    )
    if execution_mode == 'broker':
        shard_seed_counts = broker_shard_seed_counts
        effective_shard_count = broker_effective_shard_count
        shard_count_source = broker_shard_source
    iters = _coerce_positive_int(cfg['iters'], field_name='[1v3].iters')
    log_dir = cfg.get('log_dir') or None
    disable_progress_bar = bool(cfg.get('disable_progress_bar', False))

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    print(f'1v3 seed_count={seeds_per_iter} ({games_per_iter} games/iter) from {seed_count_source}')
    print(f'1v3 shard_count={effective_shard_count} from {shard_count_source}; shard seeds={shard_seed_counts}')
    print(f'1v3 execution_mode={execution_mode} from {execution_mode_source}')
    if log_dir is None:
        print('1v3 log_dir disabled')

    single_shard_eval_context = None
    shard_worker_pool = None
    broker = None
    if effective_shard_count == 1:
        single_shard_eval_context = load_eval_engines(cfg)
    else:
        try:
            if execution_mode == 'broker':
                broker = InferenceBroker.from_cfg(cfg)
            shard_worker_pool = start_persistent_shard_workers(
                cfg=cfg,
                shard_count=effective_shard_count,
                disable_progress_bar=disable_progress_bar,
                execution_mode=execution_mode,
                broker=broker,
            )
        except BaseException:
            if broker is not None:
                broker.close()
            raise

    try:
        seed_start = 10000
        for iter_index, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
            print('-' * 50)
            print('#', iter_index)
            if effective_shard_count == 1:
                iteration_log_dir = log_dir
                rankings = run_eval_once(
                    cfg=cfg,
                    seed_start=seed,
                    seed_key=key,
                    seed_count=seeds_per_iter,
                    log_dir=iteration_log_dir,
                    disable_progress_bar=disable_progress_bar,
                    eval_context=single_shard_eval_context,
                )
                rankings, avg_rank, avg_pt = summarize_rankings(rankings)
            else:
                rankings, avg_rank, avg_pt, runtime_root = run_sharded_iteration_with_workers(
                    iter_index=iter_index,
                    seed_start=seed,
                    seed_key=key,
                    shard_seed_counts=shard_seed_counts,
                    log_dir=log_dir,
                    shard_worker_pool=shard_worker_pool,
                )
                print(f'shard runtime: {runtime_root}')
            print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')
    finally:
        stop_persistent_shard_workers(shard_worker_pool)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--worker-loop', action='store_true')
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--seed-key', type=int)
    parser.add_argument('--seed-count', type=int)
    parser.add_argument('--result-json')
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--disable-progress-bar', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_loop:
        run_worker_loop(args)
        return
    if args.worker:
        required = {
            'seed_start': args.seed_start,
            'seed_key': args.seed_key,
            'seed_count': args.seed_count,
            'result_json': args.result_json,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f'missing worker args: {", ".join(missing)}')
        run_worker(args)
        return
    run_main(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

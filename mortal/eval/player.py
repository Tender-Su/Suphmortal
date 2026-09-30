import torch
import numpy as np
import os
import shutil
import secrets
import logging
from os import path
from mortal.core.model import Brain, DQN,CategoricalPolicy
from mortal.eval.engine import MortalEngine
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from mortal.config import config
from mortal.core.checkpoint_utils import checkpoint_brain_is_oracle_structure, load_brain_state_with_input_bridge
from mortal.eval.search_runtime import build_search_runtime_bundle_from_state
from mortal.core.repro import (
    effective_cudnn_benchmark,
    resolve_baseline_pool_seed,
    resolve_train_key,
    resolve_train_seed_start,
)


def _normalized_checkpoint_path(value):
    if value is None:
        return ''
    checkpoint = str(value).strip()
    return path.abspath(checkpoint) if checkpoint else ''


def build_train_baseline_pool_entries(baseline_cfg):
    state_file = _normalized_checkpoint_path(baseline_cfg.get('state_file'))
    explicit_pool_requested = any(
        key in baseline_cfg
        for key in ('champion_state_file', 'anchor_state_file', 'history_state_files')
    )
    champion_state_file = _normalized_checkpoint_path(
        baseline_cfg.get('champion_state_file') or state_file
    )
    anchor_state_file = _normalized_checkpoint_path(baseline_cfg.get('anchor_state_file'))
    history_state_files = []
    for item in baseline_cfg.get('history_state_files', []) or []:
        checkpoint = _normalized_checkpoint_path(item)
        if checkpoint:
            history_state_files.append(checkpoint)

    # Backward-compatible fallback: if no pool is configured, keep the legacy
    # single frozen baseline behavior.
    if not explicit_pool_requested:
        if not state_file:
            raise ValueError('baseline.train.state_file is required')
        return [{
            'state_file': state_file,
            'weight': 1.0,
            'labels': ('legacy',),
        }]

    champion_prob = max(float(baseline_cfg.get('champion_prob', 0.5) or 0.0), 0.0)
    anchor_prob = max(float(baseline_cfg.get('anchor_prob', 0.25) or 0.0), 0.0)
    history_prob = max(float(baseline_cfg.get('history_prob', 0.25) or 0.0), 0.0)

    merged_entries = {}

    def add_entry(checkpoint_path, weight, label):
        if not checkpoint_path or weight <= 0.0:
            return
        entry = merged_entries.setdefault(
            checkpoint_path,
            {
                'state_file': checkpoint_path,
                'weight': 0.0,
                'labels': set(),
            },
        )
        entry['weight'] += float(weight)
        entry['labels'].add(label)

    add_entry(champion_state_file, champion_prob, 'champion')
    add_entry(anchor_state_file, anchor_prob, 'anchor')

    unique_history_state_files = [
        checkpoint
        for checkpoint in dict.fromkeys(history_state_files)
        if checkpoint
    ]
    if unique_history_state_files and history_prob > 0.0:
        history_share = history_prob / len(unique_history_state_files)
        for checkpoint in unique_history_state_files:
            add_entry(checkpoint, history_share, 'history')

    if not merged_entries:
        fallback = champion_state_file or anchor_state_file or state_file
        if not fallback:
            raise ValueError(
                'baseline.train needs state_file or champion/anchor/history pool paths'
            )
        return [{
            'state_file': fallback,
            'weight': 1.0,
            'labels': ('fallback',),
        }]

    total_weight = sum(entry['weight'] for entry in merged_entries.values())
    if total_weight <= 0.0:
        normalized_entries = list(merged_entries.values())
        for entry in normalized_entries:
            entry['weight'] = 0.0
        normalized_entries[0]['weight'] = 1.0
    else:
        normalized_entries = list(merged_entries.values())
        for entry in normalized_entries:
            entry['weight'] /= total_weight

    normalized_entries.sort(key=lambda item: item['state_file'])
    for entry in normalized_entries:
        entry['labels'] = tuple(sorted(entry['labels']))
    return normalized_entries


def _load_baseline_engine_from_file(*, baseline_cfg, baseline_device, state_file, name):
    state = torch.load(state_file, weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']
    stable_mortal = Brain(
        version=version,
        conv_channels=conv_channels,
        num_blocks=num_blocks,
        is_oracle=checkpoint_brain_is_oracle_structure(state),
        Norm="GN",
    ).eval()
    stable_dqn = CategoricalPolicy().eval()
    load_brain_state_with_input_bridge(stable_mortal, state['mortal'])
    stable_dqn.load_state_dict(state['policy_net'])

    if baseline_cfg['enable_compile']:
        stable_mortal.compile()
        stable_dqn.compile()
    baseline_search_runtime = build_search_runtime_bundle_from_state(
        state,
        device=baseline_device,
        enable_compile=baseline_cfg['enable_compile'],
    )

    return MortalEngine(
        stable_mortal,
        stable_dqn,
        is_oracle=False,
        version=version,
        device=baseline_device,
        enable_amp=baseline_device.type == 'cuda',
        enable_rule_based_agari_guard=True,
        name=name,
        oracle_guiding_keep_prob=0.0,
        search_runtime_bundle=baseline_search_runtime,
    )


def _baseline_pool_signature(entries):
    return tuple(
        (
            entry['state_file'],
            os.stat(entry['state_file']).st_mtime_ns,
            os.stat(entry['state_file']).st_size,
            round(float(entry['weight']), 8),
            entry['labels'],
        )
        for entry in entries
    )

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        baseline_device = torch.device(baseline_cfg['device'])
        baseline_state_file = _normalized_checkpoint_path(baseline_cfg['state_file'])
        self.baseline_engine = _load_baseline_engine_from_file(
            baseline_cfg=baseline_cfg,
            baseline_device=baseline_device,
            state_file=baseline_state_file,
            name='baseline_test',
        )
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])

    def test_play(
        self,
        seed_count,
        mortal,
        dqn,
        device,
        *,
        search_runtime_bundle=None,
        oracle_input_mode='zero',
        oracle_guiding_keep_prob=1.0,
    ):
        torch.backends.cudnn.benchmark = False
        runtime_is_oracle = bool(getattr(mortal, 'is_oracle', False) and oracle_input_mode != 'zero')
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = runtime_is_oracle,
            version = self.chal_version,
            device = device,
            enable_amp = device.type == 'cuda',
            name = 'mortal',
            oracle_guiding_keep_prob = oracle_guiding_keep_prob,
            oracle_input_mode = oracle_input_mode,
            search_runtime_bundle = search_runtime_bundle,
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = effective_cudnn_benchmark(config)
        return stat

class TrainPlayer:
    def __init__(self):
        self.baseline_cfg = config['baseline']['train']
        self.baseline_device = torch.device(self.baseline_cfg['device'])
        self._baseline_pool = []
        self._baseline_pool_signature = None
        self.reload_baseline_pool_each_session = bool(
            self.baseline_cfg.get('reload_each_session', True)
        )
        pool_seed = resolve_baseline_pool_seed(config, self.baseline_cfg)
        self.baseline_rng = np.random.default_rng(pool_seed)
        self._ensure_baseline_pool_ready(force=True)

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        resolved_train_key = resolve_train_key(config)
        self.train_key = resolved_train_key if resolved_train_key is not None else secrets.randbits(64)
        self.train_seed = resolve_train_seed_start(config)
        self.explore_rate = cfg['explore_rate']
        self.seed_count = cfg['games'] // 4
        self.repeats = cfg['repeats']

        self.repeat_counter = 0

    def _ensure_baseline_pool_ready(self, *, force=False):
        entries = build_train_baseline_pool_entries(self.baseline_cfg)
        signature = _baseline_pool_signature(entries)
        if (
            not force
            and self._baseline_pool_signature == signature
            and self._baseline_pool
        ):
            return

        baseline_pool = []
        for idx, entry in enumerate(entries):
            labels = '+'.join(entry['labels'])
            engine = _load_baseline_engine_from_file(
                baseline_cfg=self.baseline_cfg,
                baseline_device=self.baseline_device,
                state_file=entry['state_file'],
                name=f'baseline_train_{idx}',
            )
            baseline_pool.append({
                'engine': engine,
                'state_file': entry['state_file'],
                'weight': float(entry['weight']),
                'labels': entry['labels'],
                'description': labels,
            })

        self._baseline_pool = baseline_pool
        self._baseline_pool_signature = signature
        logging.info(
            'loaded baseline.train opponent pool: %s',
            ', '.join(
                f"{path.basename(entry['state_file'])}:{entry['weight']:.2f}"
                f"[{entry['description']}]"
                for entry in baseline_pool
            ),
        )

    def _sample_baseline_engine(self):
        if self.reload_baseline_pool_each_session or not self._baseline_pool:
            self._ensure_baseline_pool_ready()
        weights = np.asarray(
            [entry['weight'] for entry in self._baseline_pool],
            dtype=np.float64,
        )
        sampled_index = int(self.baseline_rng.choice(len(self._baseline_pool), p=weights))
        return self._baseline_pool[sampled_index]

    def train_play(
        self,
        mortal,
        dqn,
        device,
        *,
        actor_oracle_enabled=False,
        actor_oracle_keep_prob=1.0,
        actor_oracle_input_mode='true',
        search_runtime_bundle=None,
    ):
        torch.backends.cudnn.benchmark = False
        runtime_is_oracle = bool(
            actor_oracle_enabled
            and actor_oracle_keep_prob > 0.0
            and actor_oracle_input_mode != 'zero'
        )
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = runtime_is_oracle,
            version = self.chal_version,
            explore_rate = self.explore_rate,
            device = device,
            enable_amp = device.type == 'cuda',
            name = 'trainee',
            oracle_guiding_keep_prob = actor_oracle_keep_prob,
            oracle_input_mode = actor_oracle_input_mode,
            search_runtime_bundle = search_runtime_bundle,
        )
        baseline_entry = self._sample_baseline_engine()
        logging.info(
            'train baseline sampled: %s [%s] (weight=%.2f)',
            path.basename(baseline_entry['state_file']),
            baseline_entry['description'],
            baseline_entry['weight'],
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = baseline_entry['engine'],
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), sorted(os.listdir(self.log_dir))))

        torch.backends.cudnn.benchmark = effective_cudnn_benchmark(config)
        return rankings, file_list

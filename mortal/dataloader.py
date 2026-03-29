import os
import random
from itertools import repeat

import numpy as np
import torch
from torch.utils.data import IterableDataset

from config import config
from cpu_affinity import maybe_configure_process_affinity
from libriichi.dataset import GameplayLoader
from model import GRP
from reward_calculator import RewardCalculator


def resolve_rayon_num_threads(num_workers, file_batch_size, explicit_threads=0):
    if explicit_threads and explicit_threads > 0:
        return int(explicit_threads)

    env_threads = os.environ.get('RAYON_NUM_THREADS')
    if env_threads:
        try:
            parsed = int(env_threads)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    cpu_count = os.cpu_count() or 1
    if num_workers <= 0:
        return max(1, min(file_batch_size, max(cpu_count - 2, 1)))
    return max(1, min(file_batch_size, max(cpu_count // (num_workers + 1), 1)))


def danger_labels_enabled():
    aux_cfg = config.get('aux', {})
    return bool(aux_cfg.get('danger_enabled', False)) or aux_cfg.get('danger_weight', 0.0) > 0


def iter_loaded_gameplay_batches(loader, file_list):
    if file_list and str(file_list[0]).endswith('.pt'):
        for cache_file in file_list:
            payload = torch.load(cache_file, weights_only=False)
            raw_logs = payload['logs'] if isinstance(payload, dict) else payload
            yield from loader.load_logs(raw_logs)
        return
    yield from loader.load_log_files(file_list)


def extend_buffer_from_columns(buffer, *columns):
    buffer.extend(zip(*columns))

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        shared_stats=None,
        oracle = False,
        file_batch_size = 20, 
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        worker_torch_num_threads = 1,
        worker_torch_num_interop_threads = 1,
        rayon_num_threads = 0,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.worker_torch_num_threads = worker_torch_num_threads
        self.worker_torch_num_interop_threads = worker_torch_num_interop_threads
        self.rayon_num_threads = rayon_num_threads
        self.iterator = None
        self.shared_stats = shared_stats 
        self.track_opponent_states = False

    def build_iter(self):
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(
            self.grp,
            self.pts,
            shared_stats=self.shared_stats  
        )

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        random.shuffle(self.file_list)
        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
            track_opponent_states = self.track_opponent_states,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        for gameplay_batch in iter_loaded_gameplay_batches(self.loader, file_list):
            for game in gameplay_batch:
                # per move
                obs = game.take_obs_batch()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs_batch()
                actions = game.take_actions_batch()
                masks = game.take_masks_batch()
                at_kyoku = game.take_at_kyoku_batch()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                advantage = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(advantage) >= at_kyoku[-1] + 1 

                # player's final rank (0-3) for AuxNet label
                player_rank = rank_by_player[player_id]
                sample_advantage = np.take(
                    np.asarray(advantage),
                    np.asarray(at_kyoku, dtype=np.int64),
                )

                columns = [obs]
                if self.oracle:
                    columns.append(invisible_obs)
                columns.extend((
                    actions,
                    masks,
                    sample_advantage,
                    repeat(player_rank, game_size),
                ))
                extend_buffer_from_columns(self.buffer, *columns)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


class SupervisedFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        oracle=False,
        file_batch_size=20,
        reserve_ratio=0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        shuffle_files=True,
        worker_torch_num_threads=1,
        worker_torch_num_interop_threads=1,
        rayon_num_threads=0,
        emit_opponent_state_labels=None,
        track_danger_labels=None,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.shuffle_files = shuffle_files
        self.worker_torch_num_threads = worker_torch_num_threads
        self.worker_torch_num_interop_threads = worker_torch_num_interop_threads
        self.rayon_num_threads = rayon_num_threads
        self.iterator = None
        if emit_opponent_state_labels is None:
            emit_opponent_state_labels = config['aux'].get('opponent_state_weight', 0.0) > 0
        if track_danger_labels is None:
            track_danger_labels = danger_labels_enabled()
        self.emit_opponent_state_labels = bool(emit_opponent_state_labels)
        self.track_danger_labels = bool(track_danger_labels)
        self.track_opponent_states = self.emit_opponent_state_labels

    def build_iter(self):
        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        if self.shuffle_files:
            random.shuffle(self.file_list)
        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
            track_opponent_states=self.track_opponent_states,
            track_danger_labels=self.track_danger_labels,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            if self.shuffle_files:
                random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        if self.shuffle_files:
            random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        for gameplay_batch in iter_loaded_gameplay_batches(self.loader, file_list):
            for game in gameplay_batch:
                obs = game.take_obs_batch()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs_batch()
                actions = game.take_actions_batch()
                masks = game.take_masks_batch()
                context_meta = game.take_context_meta_batch()

                grp = game.take_grp()
                player_id = game.take_player_id()
                rank_by_player = grp.take_rank_by_player()
                player_rank = rank_by_player[player_id]
                if self.emit_opponent_state_labels:
                    opponent_shanten = game.take_opponent_shanten_batch()
                    opponent_tenpai = game.take_opponent_tenpai_batch()
                if self.track_danger_labels:
                    danger_valid = game.take_danger_valid_batch()
                    danger_any = game.take_danger_any_batch()
                    danger_value = game.take_danger_value_batch()
                    danger_player_mask = game.take_danger_player_mask_batch()
                game_size = len(obs)
                columns = [obs]
                if self.oracle:
                    columns.append(invisible_obs)
                columns.extend((
                    actions,
                    masks,
                    repeat(player_rank, game_size),
                    context_meta,
                ))
                if self.emit_opponent_state_labels:
                    columns.extend((
                        opponent_shanten,
                        opponent_tenpai,
                    ))
                if self.track_danger_labels:
                    columns.extend((
                        danger_valid,
                        danger_any,
                        danger_value,
                        danger_player_mask,
                    ))
                extend_buffer_from_columns(self.buffer, *columns)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    maybe_configure_process_affinity(log=False, context='dataloader worker')
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    rayon_num_threads = int(getattr(dataset, 'rayon_num_threads', 0))
    if rayon_num_threads > 0:
        os.environ['RAYON_NUM_THREADS'] = str(rayon_num_threads)
    torch_num_threads = max(int(getattr(dataset, 'worker_torch_num_threads', 1)), 1)
    torch.set_num_threads(torch_num_threads)
    torch_num_interop_threads = int(getattr(dataset, 'worker_torch_num_interop_threads', 1))
    if torch_num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(torch_num_interop_threads)
        except RuntimeError:
            pass
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]

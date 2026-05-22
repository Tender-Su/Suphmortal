import random
from itertools import repeat

import numpy as np
import torch
from torch.utils.data import IterableDataset

from libriichi.dataset import GameplayLoader
from mortal.data.dataloader import (
    iter_loaded_gameplay_batches,
    normalize_value_target_mode,
    select_value_targets,
)


def centered_rank_points(pts):
    values = np.asarray(pts, dtype=np.float32)
    if values.shape != (4,):
        raise ValueError(f'pts must contain exactly 4 rank values, got shape={values.shape}')
    return values - values.mean(dtype=np.float32)


def rank_array(rank_by_player):
    if isinstance(rank_by_player, (bytes, bytearray, memoryview)):
        values = np.frombuffer(rank_by_player, dtype=np.uint8).astype(np.int64)
    else:
        values = np.asarray(rank_by_player, dtype=np.int64)
    return values


def terminal_rank_values_by_player(rank_by_player, pts):
    rank_by_player = rank_array(rank_by_player)
    if rank_by_player.shape != (4,):
        raise ValueError(
            f'rank_by_player must contain exactly 4 ranks, got shape={rank_by_player.shape}'
        )
    if np.any(rank_by_player < 0) or np.any(rank_by_player > 3):
        raise ValueError(f'rank_by_player contains invalid rank ids: {rank_by_player!r}')
    return centered_rank_points(pts)[rank_by_player].astype(np.float32, copy=False)


def terminal_rank_value_target(rank_by_player, player_id, pts, value_target_mode='all_players'):
    values_by_player = terminal_rank_values_by_player(rank_by_player, pts)[None, :]
    return select_value_targets(values_by_player, int(player_id), value_target_mode)[0]


def rank_by_score(scores_by_player):
    scores = np.asarray(scores_by_player, dtype=np.float32)
    if scores.shape[-1] != 4:
        raise ValueError(f'scores_by_player must end with 4 scores, got shape={scores.shape}')

    flat_scores = scores.reshape(-1, 4)
    flat_ranks = np.empty(flat_scores.shape, dtype=np.int64)
    for row_idx, row in enumerate(flat_scores):
        player_by_rank = sorted(range(4), key=lambda player_id: (-float(row[player_id]), player_id))
        for rank_id, player_id in enumerate(player_by_rank):
            flat_ranks[row_idx, player_id] = rank_id
    return flat_ranks.reshape(scores.shape)


def current_rank_values_by_kyoku(grp_feature, pts):
    feature = np.asarray(grp_feature, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[1] < 7:
        raise ValueError(f'grp_feature must have shape (kyoku, >=7), got shape={feature.shape}')
    current_ranks = rank_by_score(feature[:, 3:7])
    return centered_rank_points(pts)[current_ranks].astype(np.float32, copy=False)


def terminal_rank_delta_values_by_kyoku(grp_feature, rank_by_player, pts):
    terminal_values = terminal_rank_values_by_player(rank_by_player, pts)[None, :]
    current_values = current_rank_values_by_kyoku(grp_feature, pts)
    return (terminal_values - current_values).astype(np.float32, copy=False)


def score_rank_delta_rewards_by_kyoku(grp_feature, rank_by_player, pts):
    current_values = current_rank_values_by_kyoku(grp_feature, pts)
    if current_values.shape[0] == 0:
        return current_values

    next_values = np.empty_like(current_values)
    if current_values.shape[0] > 1:
        next_values[:-1] = current_values[1:]
    next_values[-1] = terminal_rank_values_by_player(rank_by_player, pts)
    return (next_values - current_values).astype(np.float32, copy=False)


def expand_kyoku_rewards_to_steps(kyoku_rewards, at_kyoku):
    rewards = np.asarray(kyoku_rewards, dtype=np.float32)
    kyoku_ids = np.asarray(at_kyoku, dtype=np.int64)
    if rewards.ndim != 2 or rewards.shape[1] != 4:
        raise ValueError(f'kyoku_rewards must have shape (kyoku, 4), got shape={rewards.shape}')
    if kyoku_ids.ndim != 1:
        raise ValueError(f'at_kyoku must be 1-D, got shape={kyoku_ids.shape}')

    step_rewards = np.zeros((kyoku_ids.shape[0], 4), dtype=np.float32)
    for idx, kyoku_id in enumerate(kyoku_ids):
        if kyoku_id < 0 or kyoku_id >= rewards.shape[0]:
            raise ValueError(f'at_kyoku out of range for kyoku_rewards shape={rewards.shape}')
        next_kyoku_id = rewards.shape[0] if idx == kyoku_ids.shape[0] - 1 else kyoku_ids[idx + 1]
        if next_kyoku_id < kyoku_id:
            raise ValueError(f'at_kyoku must be non-decreasing, got transition {kyoku_id}->{next_kyoku_id}')
        if next_kyoku_id != kyoku_id:
            step_rewards[idx] = rewards[kyoku_id:next_kyoku_id].sum(axis=0)
    return step_rewards


def discounted_returns_from_step_rewards(step_rewards, gamma):
    rewards = np.asarray(step_rewards, dtype=np.float32)
    if rewards.ndim != 2 or rewards.shape[1] != 4:
        raise ValueError(f'step_rewards must have shape (steps, 4), got shape={rewards.shape}')

    gamma = float(gamma)
    returns = np.empty_like(rewards)
    running = np.zeros((4,), dtype=np.float32)
    for idx in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def normalize_oracle_return_mode(value):
    mode = str(value or 'score_rank_mc').strip().lower()
    if mode in ('terminal', 'terminal_rank', 'rank_terminal'):
        return 'terminal_rank'
    if mode in ('delta', 'rank_delta', 'terminal_rank_delta', 'future_rank_delta'):
        return 'rank_delta'
    if mode in (
        'mc',
        'monte_carlo',
        'score_rank_mc',
        'score_rank_return',
        'discounted_rank_delta',
        'discounted_score_rank_delta',
    ):
        return 'score_rank_mc'
    raise ValueError(
        f"unsupported oracle return_mode={value!r}; "
        "expected 'score_rank_mc', 'rank_delta', or 'terminal_rank'"
    )


def oracle_values_by_kyoku(grp_feature, rank_by_player, pts, return_mode='rank_delta'):
    mode = normalize_oracle_return_mode(return_mode)
    feature = np.asarray(grp_feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError(f'grp_feature must be 2-D, got shape={feature.shape}')
    if mode == 'terminal_rank':
        terminal_values = terminal_rank_values_by_player(rank_by_player, pts)
        return np.repeat(terminal_values[None, :], feature.shape[0], axis=0)
    return terminal_rank_delta_values_by_kyoku(feature, rank_by_player, pts)


def oracle_step_value_targets(grp_feature, rank_by_player, pts, at_kyoku, return_mode='score_rank_mc', gamma=1.0):
    mode = normalize_oracle_return_mode(return_mode)
    feature = np.asarray(grp_feature, dtype=np.float32)
    kyoku_ids = np.asarray(at_kyoku, dtype=np.int64)
    if feature.ndim != 2:
        raise ValueError(f'grp_feature must be 2-D, got shape={feature.shape}')
    if kyoku_ids.ndim != 1:
        raise ValueError(f'at_kyoku must be 1-D, got shape={kyoku_ids.shape}')
    if kyoku_ids.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if np.any(kyoku_ids < 0) or np.any(kyoku_ids >= feature.shape[0]):
        raise ValueError(f'at_kyoku out of range for grp_feature shape={feature.shape}')

    if mode in ('terminal_rank', 'rank_delta'):
        values_by_kyoku = oracle_values_by_kyoku(feature, rank_by_player, pts, mode)
        return np.take(values_by_kyoku, kyoku_ids, axis=0).astype(np.float32, copy=False)

    kyoku_rewards = score_rank_delta_rewards_by_kyoku(feature, rank_by_player, pts)
    step_rewards = expand_kyoku_rewards_to_steps(kyoku_rewards, kyoku_ids)
    return discounted_returns_from_step_rewards(step_rewards, gamma)


class OracleTerminalValueDataset(IterableDataset):
    """Yields obs, oracle obs, and non-GRP rank-return value targets."""

    def __init__(
        self,
        *,
        version,
        file_list,
        pts,
        file_batch_size=10,
        reserve_ratio=0.0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        shuffle_files=True,
        value_target_mode='all_players',
        return_mode='score_rank_mc',
        discount_gamma=1.0,
        worker_torch_num_threads=1,
        worker_torch_num_interop_threads=1,
        rayon_num_threads=0,
    ):
        super().__init__()
        self.version = int(version)
        self.file_list = list(file_list)
        self.pts = list(pts)
        self.file_batch_size = int(file_batch_size)
        self.reserve_ratio = float(reserve_ratio)
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = int(num_epochs)
        self.enable_augmentation = bool(enable_augmentation)
        self.augmented_first = bool(augmented_first)
        self.shuffle_files = bool(shuffle_files)
        self.value_target_mode = normalize_value_target_mode(value_target_mode)
        self.return_mode = normalize_oracle_return_mode(return_mode)
        self.discount_gamma = float(discount_gamma)
        self.worker_torch_num_threads = int(worker_torch_num_threads)
        self.worker_torch_num_interop_threads = int(worker_torch_num_interop_threads)
        self.rayon_num_threads = int(rayon_num_threads)

    def __iter__(self):
        return self.build_iter()

    def build_iter(self):
        for _ in range(max(self.num_epochs, 1)):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        file_list = list(self.file_list)
        if self.shuffle_files:
            random.shuffle(file_list)

        loader = GameplayLoader(
            version=self.version,
            oracle=True,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=bool(augmented),
            track_opponent_states=False,
            track_danger_labels=False,
            track_regret_labels=False,
        )
        buffer = []
        file_batch_size = max(int(self.file_batch_size), 1)
        for start_idx in range(0, len(file_list), file_batch_size):
            old_buffer_size = len(buffer)
            self.populate_buffer(
                loader,
                file_list[start_idx:start_idx + file_batch_size],
                buffer,
            )
            new_rows = len(buffer) - old_buffer_size
            reserved_size = int(new_rows * self.reserve_ratio)
            if reserved_size > len(buffer):
                continue
            if self.shuffle_files:
                random.shuffle(buffer)
            yield from buffer[reserved_size:]
            del buffer[reserved_size:]
        if self.shuffle_files:
            random.shuffle(buffer)
        yield from buffer

    def populate_buffer(self, loader, file_list, buffer):
        for _source_name, gameplay_batch in iter_loaded_gameplay_batches(loader, file_list):
            for game in gameplay_batch:
                obs = np.asarray(game.take_obs_batch(), dtype=np.float32)
                invisible_obs = np.asarray(game.take_invisible_obs_batch(), dtype=np.float32)
                at_kyoku = np.asarray(game.take_at_kyoku_batch(), dtype=np.int64)
                game_size = int(obs.shape[0])
                if game_size == 0:
                    continue
                if at_kyoku.shape != (game_size,):
                    raise ValueError(
                        f'at_kyoku length mismatch: obs={game_size}, at_kyoku_shape={at_kyoku.shape}'
                    )

                grp = game.take_grp()
                player_id = int(game.take_player_id())
                grp_feature = np.asarray(grp.take_feature(), dtype=np.float32)
                rank_by_player = grp.take_rank_by_player()
                target_by_player = oracle_step_value_targets(
                    grp_feature,
                    rank_by_player,
                    self.pts,
                    at_kyoku,
                    self.return_mode,
                    self.discount_gamma,
                )
                targets = select_value_targets(
                    target_by_player,
                    player_id,
                    self.value_target_mode,
                ).astype(np.float32, copy=False)
                buffer.extend(zip(obs, invisible_obs, targets, repeat(player_id, game_size)))

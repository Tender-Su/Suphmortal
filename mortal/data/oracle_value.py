import hashlib
import random
from itertools import repeat
from os import path

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from libriichi.dataset import GameplayLoader
from mortal.data.dataloader import (
    iter_loaded_gameplay_batches,
    normalize_value_target_mode,
    select_value_targets,
)


ORACLE_STREAM_SHARDING_VERSION = 'global_permutation_stride_v2'
ORACLE_STATE_FOLDING_VERSION = 'native_hash_full_decision_clock_v2'
ORACLE_IMPUTATION_VERSION = 'canonical_events_fnv1a_mix64_chacha8_v1'
ORACLE_TARGET_CLOCK_VERSION = 'full_player_decisions_v1'


def normalize_state_fold_backend(value):
    backend = str(value or 'python_permutation').strip().lower()
    if backend in ('python', 'python_permutation', 'post_decode'):
        return 'python_permutation'
    if backend in ('native', 'native_hash', 'rust', ORACLE_STATE_FOLDING_VERSION):
        return 'native_hash'
    raise ValueError(
        f'unsupported state_fold_backend={value!r}; '
        "expected 'python_permutation' or 'native_hash'"
    )


def deterministic_game_id(source_name):
    source_name = path.basename(str(source_name).replace('\\', '/'))
    digest = hashlib.blake2b(source_name.encode('utf-8'), digest_size=8)
    return int.from_bytes(digest.digest(), byteorder='little', signed=False) & ((1 << 63) - 1)


def deterministic_state_fold_indices(
    size,
    *,
    fold_count,
    fold_index,
    seed,
    source_name,
    game_index,
    player_id,
):
    size = int(size)
    fold_count = int(fold_count)
    fold_index = int(fold_index)
    if fold_count <= 1:
        return np.arange(size, dtype=np.int64)
    if fold_index < 0 or fold_index >= fold_count:
        raise ValueError(f'fold_index must be in [0, {fold_count}), got {fold_index}')

    digest = hashlib.blake2b(digest_size=8)
    for value in (seed, source_name, game_index, player_id):
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\0')
    permutation_seed = int.from_bytes(digest.digest(), byteorder='little', signed=False)
    permutation = np.random.default_rng(permutation_seed).permutation(size)
    return permutation[fold_index::fold_count]


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


def full_decision_clock(game, at_kyoku, *, require_metadata):
    full_getter = getattr(game, 'take_full_at_kyoku_batch', None)
    index_getter = getattr(game, 'take_sample_indices_batch', None)
    if full_getter is None or index_getter is None:
        if require_metadata:
            raise RuntimeError(
                'native folding requires full decision clock metadata; rebuild libriichi '
                'and start a new run instead of resuming legacy folded targets'
            )
        return at_kyoku, np.arange(len(at_kyoku), dtype=np.int64)
    full = np.asarray(full_getter(), dtype=np.int64)
    indices = np.asarray(index_getter(), dtype=np.int64)
    if (
        full.ndim != 1 or indices.shape != at_kyoku.shape
        or np.any(indices < 0) or np.any(indices >= len(full))
        or np.any(np.diff(indices) <= 0) or np.any(np.diff(full) < 0)
        or not np.array_equal(full[indices], at_kyoku)
    ):
        raise ValueError('invalid full decision clock metadata')
    return full, indices


class OracleTerminalValueDataset(IterableDataset):
    """Yields obs, oracle obs, and non-GRP rank-return value targets."""

    shards_files_in_iter = True

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
        shuffle_seed=None,
        stream_cycle=0,
        resume_cursors=None,
        emit_progress=False,
        emit_game_id=False,
        state_fold_count=1,
        state_fold_seed=0,
        state_fold_backend='python_permutation',
        oracle_imputation_seed=None,
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
        self.shuffle_seed = None if shuffle_seed is None else int(shuffle_seed)
        self.stream_cycle = int(stream_cycle)
        self.resume_cursors = {
            int(worker_id): (int(cursor[0]), int(cursor[1]))
            for worker_id, cursor in (resume_cursors or {}).items()
        }
        self.emit_progress = bool(emit_progress)
        self.emit_game_id = bool(emit_game_id)
        if self.emit_progress and self.emit_game_id:
            raise ValueError('emit_progress and emit_game_id are mutually exclusive')
        self.state_fold_count = int(state_fold_count)
        self.state_fold_seed = int(state_fold_seed)
        self.state_fold_backend = normalize_state_fold_backend(state_fold_backend)
        self.oracle_imputation_seed = oracle_imputation_seed
        if oracle_imputation_seed is not None:
            self.oracle_imputation_seed = int(oracle_imputation_seed)
            if not 0 <= self.oracle_imputation_seed < 2**64:
                raise ValueError('oracle_imputation_seed must be an unsigned 64-bit integer')
        if self.state_fold_count <= 0:
            raise ValueError('state_fold_count must be positive')

    def __iter__(self):
        return self.build_iter()

    def build_iter(self):
        pass_count = 2 if self.enable_augmentation else 1
        passes_per_cycle = max(self.num_epochs, 1) * pass_count
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        resume_pass, resume_file_offset = self.resume_cursors.get(worker_id, (-1, 0))
        local_pass = 0
        for _ in range(max(self.num_epochs, 1)):
            global_pass = self.stream_cycle * passes_per_cycle + local_pass
            if global_pass >= resume_pass:
                start_offset = resume_file_offset if global_pass == resume_pass else 0
                yield from self.load_files(
                    self.augmented_first,
                    stream_pass=global_pass,
                    start_file_offset=start_offset,
                )
            local_pass += 1
            if self.enable_augmentation:
                global_pass = self.stream_cycle * passes_per_cycle + local_pass
                if global_pass >= resume_pass:
                    start_offset = resume_file_offset if global_pass == resume_pass else 0
                    yield from self.load_files(
                        not self.augmented_first,
                        stream_pass=global_pass,
                        start_file_offset=start_offset,
                    )
                local_pass += 1

    def load_files(self, augmented, *, stream_pass=0, start_file_offset=0):
        file_list = list(self.file_list)
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        num_workers = 1 if worker_info is None else int(worker_info.num_workers)
        buffer_rng = None
        if self.shuffle_files:
            if self.shuffle_seed is None:
                if worker_info is None:
                    random.shuffle(file_list)
                else:
                    shared_seed = int(worker_info.seed) - worker_id
                    file_rng = random.Random(
                        shared_seed + 1_000_003 * int(stream_pass)
                    )
                    file_rng.shuffle(file_list)
                    buffer_rng = random.Random(
                        shared_seed
                        + 1_000_003 * int(stream_pass)
                        + 9_176 * worker_id
                    )
            else:
                file_rng = random.Random(
                    self.shuffle_seed
                    + 1_000_003 * int(stream_pass)
                )
                file_rng.shuffle(file_list)
                buffer_rng = random.Random(
                    self.shuffle_seed
                    + 1_000_003 * int(stream_pass)
                    + 9_176 * worker_id
                )

        # IterableDataset workers otherwise each consume the complete file list.
        # Shard one shared permutation so every pass covers each file exactly once.
        file_list = file_list[worker_id::num_workers]

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
        state_fold_index = int(stream_pass) % self.state_fold_count
        if self.oracle_imputation_seed is not None:
            imputation_setter = getattr(loader, 'set_oracle_imputation_seed', None)
            if imputation_setter is None:
                raise RuntimeError('deterministic Oracle inputs require the rebuilt libriichi')
            imputation_seed = self.oracle_imputation_seed
            if self.shuffle_files:
                imputation_seed = (imputation_seed + 1_000_003 * int(stream_pass)) % 2**64
            imputation_setter(imputation_seed)
        if self.state_fold_backend == 'native_hash' and self.state_fold_count > 1:
            setter = getattr(loader, 'set_sample_fold', None)
            if setter is None:
                raise RuntimeError(
                    'native state folding requires a rebuilt libriichi with set_sample_fold'
                )
            setter(self.state_fold_count, state_fold_index, self.state_fold_seed)
        buffer = []
        file_batch_size = max(int(self.file_batch_size), 1)
        start_file_offset = max(int(start_file_offset), 0)
        start_file_offset -= start_file_offset % file_batch_size
        for start_idx in range(start_file_offset, len(file_list), file_batch_size):
            old_buffer_size = len(buffer)
            progress_token = (
                np.asarray((worker_id, int(stream_pass), start_idx), dtype=np.int64)
                if self.emit_progress else None
            )
            self.populate_buffer(
                loader,
                file_list[start_idx:start_idx + file_batch_size],
                buffer,
                progress_token=progress_token,
                state_fold_index=state_fold_index,
            )
            new_rows = len(buffer) - old_buffer_size
            reserved_size = int(new_rows * self.reserve_ratio)
            if reserved_size > len(buffer):
                continue
            if self.shuffle_files:
                (buffer_rng.shuffle if buffer_rng is not None else random.shuffle)(buffer)
            emitted_rows = buffer[reserved_size:]
            del buffer[reserved_size:]
            if self.emit_progress and emitted_rows:
                retained_offsets = [int(row[-1][2]) for row in buffer]
                safe_offset = min([start_idx, *retained_offsets])
                safe_progress = np.asarray(
                    (worker_id, int(stream_pass), safe_offset),
                    dtype=np.int64,
                )
                for row in emitted_rows:
                    yield (*row[:-1], safe_progress)
            else:
                yield from emitted_rows
            del emitted_rows
        if self.shuffle_files:
            (buffer_rng.shuffle if buffer_rng is not None else random.shuffle)(buffer)
        if self.emit_progress and buffer:
            safe_offset = min(int(row[-1][2]) for row in buffer)
            safe_progress = np.asarray(
                (worker_id, int(stream_pass), safe_offset),
                dtype=np.int64,
            )
            for row in buffer:
                yield (*row[:-1], safe_progress)
        else:
            yield from buffer

    def populate_buffer(
        self,
        loader,
        file_list,
        buffer,
        *,
        progress_token=None,
        state_fold_index=0,
    ):
        for source_name, gameplay_batch in iter_loaded_gameplay_batches(
            loader,
            file_list,
            bulk_event_cache=(
                self.state_fold_backend == 'native_hash'
                and not self.emit_game_id
            ),
        ):
            game_id = np.int64(deterministic_game_id(source_name))
            for game_index, game in enumerate(gameplay_batch):
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

                full_at_kyoku, retained_indices = full_decision_clock(
                    game, at_kyoku,
                    require_metadata=(
                        self.state_fold_backend == 'native_hash' and self.state_fold_count > 1
                    ),
                )
                grp = game.take_grp()
                player_id = int(game.take_player_id())
                grp_feature = np.asarray(grp.take_feature(), dtype=np.float32)
                rank_by_player = grp.take_rank_by_player()
                target_by_player = oracle_step_value_targets(
                    grp_feature,
                    rank_by_player,
                    self.pts,
                    full_at_kyoku,
                    self.return_mode,
                    self.discount_gamma,
                )[retained_indices]
                targets = select_value_targets(
                    target_by_player,
                    player_id,
                    self.value_target_mode,
                ).astype(np.float32, copy=False)
                fold_count = (
                    1 if self.state_fold_backend == 'native_hash'
                    else self.state_fold_count
                )
                indices = deterministic_state_fold_indices(
                    game_size,
                    fold_count=fold_count,
                    fold_index=state_fold_index,
                    seed=self.state_fold_seed,
                    source_name=source_name,
                    game_index=game_index,
                    player_id=player_id,
                )
                if indices.shape[0] != game_size:
                    obs = obs[indices]
                    invisible_obs = invisible_obs[indices]
                    targets = targets[indices]
                    game_size = int(indices.shape[0])
                rows = zip(obs, invisible_obs, targets, repeat(player_id, game_size))
                if progress_token is not None:
                    buffer.extend(
                        (*row, progress_token)
                        for row in rows
                    )
                elif self.emit_game_id:
                    buffer.extend(
                        (*row, game_id)
                        for row in rows
                    )
                else:
                    buffer.extend(rows)

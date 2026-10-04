import hashlib
import os
import random
import re
import logging
from itertools import repeat

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mortal.config import config
from mortal.core.cpu_affinity import maybe_configure_process_affinity
from libriichi.dataset import GameplayLoader
from mortal.core.model import GRP
from mortal.data.reward_calculator import RewardCalculator


_REPLAY_PARAM_VERSION_RE = re.compile(r'^pv(\d+)_sid\d+_')


def _is_invalid_game_log_error(exc):
    return 'empty or invalid game log' in str(exc).lower()


def replay_param_version_from_path(filename):
    match = _REPLAY_PARAM_VERSION_RE.match(os.path.basename(str(filename)))
    if match is None:
        return None
    return int(match.group(1))


def stable_source_game_id(filename):
    digest = hashlib.blake2b(
        str(filename).encode('utf-8'),
        digest_size=8,
        person=b'mortal-sl',
    ).digest()
    return int.from_bytes(digest, byteorder='little') & ((1 << 63) - 1)


def rotate_values_to_relative_order(values_by_player, player_id):
    order = [
        (int(player_id) + offset) % values_by_player.shape[-1]
        for offset in range(values_by_player.shape[-1])
    ]
    return np.take(values_by_player, order, axis=-1)


def normalize_value_target_mode(value):
    mode = str(value or 'current_player').strip().lower()
    if mode in ('current', 'current_player', 'self', 'one'):
        return 'current_player'
    if mode in ('all', 'all_players', 'four_player', '4p'):
        return 'all_players'
    raise ValueError(
        f"unsupported value_target_mode={value!r}; expected 'current_player' or 'all_players'"
    )


def normalize_value_reward_source(value):
    mode = str(value or 'grp').strip().lower()
    if mode in ('grp', 'global_reward_predictor', 'reward_calculator'):
        return 'grp'
    if mode in (
        'score_rank',
        'score_rank_mc',
        'score_rank_delta',
        'score_rank_return',
        'non_grp',
        'exact_score_rank',
    ):
        return 'score_rank'
    raise ValueError(
        f"unsupported value_reward_source={value!r}; expected 'grp' or 'score_rank'"
    )


def select_value_targets(values_by_player, player_id, value_target_mode):
    rotated = rotate_values_to_relative_order(values_by_player, player_id)
    mode = normalize_value_target_mode(value_target_mode)
    if mode == 'all_players':
        return rotated
    return rotated[:, :1]


def _raw_logs_from_payload(payload):
    raw_logs = payload['logs'] if isinstance(payload, dict) else payload
    if isinstance(raw_logs, dict):
        return list(raw_logs.values())
    return list(raw_logs)


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


def regret_labels_enabled():
    aux_cfg = config.get('aux', {})
    return (
        float(aux_cfg.get('tile_efficiency_weight', 0.0) or 0.0) > 0
        or float(aux_cfg.get('furo_regret_weight', 0.0) or 0.0) > 0
        or float(aux_cfg.get('hand_value_regret_weight', 0.0) or 0.0) > 0
    )


def iter_loaded_gameplay_batches(loader, file_list, *, bulk_event_cache=False):
    if not file_list:
        return
    if file_list and str(file_list[0]).endswith('.pt'):
        for cache_file in file_list:
            payload = torch.load(cache_file, weights_only=False)
            raw_logs = _raw_logs_from_payload(payload)
            for gameplay_batch in loader.load_logs(raw_logs):
                yield str(cache_file), gameplay_batch
        return
    event_cache_files = [
        filename for filename in file_list
        if str(filename).endswith('.events.zst')
    ]
    if bulk_event_cache and len(event_cache_files) == len(file_list):
        for index, gameplay_batch in enumerate(loader.load_log_files(file_list)):
            yield f'{file_list[0]}#{index}', gameplay_batch
        return
    if not event_cache_files:
        try:
            loaded_batches = loader.load_log_files(file_list)
        except RuntimeError as exc:
            if not _is_invalid_game_log_error(exc):
                raise
            logging.warning(
                'bulk log load hit an empty or invalid game log; falling back to per-file load'
            )
        else:
            if len(loaded_batches) == len(file_list):
                for filename, gameplay_batch in zip(file_list, loaded_batches):
                    yield str(filename), gameplay_batch
                return
    for filename in file_list:
        try:
            loaded_batches = loader.load_log_files([filename])
        except RuntimeError as exc:
            if not _is_invalid_game_log_error(exc):
                raise
            logging.warning('skipping empty or invalid game log: %s', filename)
            continue
        for gameplay_batch in loaded_batches:
            yield str(filename), gameplay_batch


def extend_buffer_from_columns(buffer, *columns):
    buffer.extend(zip(*columns))


def score_rank_delta_pt_all_players(grp_feature, rank_by_player, pts):
    from mortal.data.oracle_value import score_rank_delta_rewards_by_kyoku

    return score_rank_delta_rewards_by_kyoku(grp_feature, rank_by_player, pts)


def value_rewards_all_players(reward_source, reward_calc, grp_feature, rank_by_player, pts):
    source = normalize_value_reward_source(reward_source)
    if source == 'score_rank':
        return score_rank_delta_pt_all_players(grp_feature, rank_by_player, pts)
    if reward_calc is None:
        raise RuntimeError('GRP value reward source requires an initialized RewardCalculator')
    return reward_calc.calc_delta_pt_all_players(grp_feature, rank_by_player)


class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
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
        emit_opponent_state_labels = False,
        track_danger_labels = False,
        track_regret_labels = False,
        emit_value_targets = False,
        value_target_mode = 'current_player',
        value_reward_source = 'grp',
        emit_context_meta = False,
        emit_replay_param_version = False,
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
        self.emit_opponent_state_labels = bool(emit_opponent_state_labels)
        self.track_danger_labels = bool(track_danger_labels)
        self.track_regret_labels = bool(track_regret_labels)
        self.track_opponent_states = self.emit_opponent_state_labels
        self.emit_value_targets = bool(emit_value_targets)
        self.value_target_mode = normalize_value_target_mode(value_target_mode)
        self.value_reward_source = normalize_value_reward_source(value_reward_source)
        self.emit_context_meta = bool(emit_context_meta)
        self.emit_replay_param_version = bool(emit_replay_param_version)

    def build_iter(self):
        self.reward_calc = None
        if not self.emit_value_targets or self.value_reward_source == 'grp':
            self.grp = GRP(**config['grp']['network'])
            grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
            self.grp.load_state_dict(grp_state['model'])
            label_smoothing = config.get('grp', {}).get('label_smoothing', 0.0)
            self.reward_calc = RewardCalculator(
                self.grp,
                self.pts,
                label_smoothing=label_smoothing,
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
            track_danger_labels = self.track_danger_labels,
            track_regret_labels = self.track_regret_labels,
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
        for source_name, gameplay_batch in iter_loaded_gameplay_batches(self.loader, file_list):
            replay_param_version = replay_param_version_from_path(source_name)
            for game in gameplay_batch:
                # per move
                obs = game.take_obs_batch()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs_batch()
                actions = game.take_actions_batch()
                masks = game.take_masks_batch()
                at_kyoku = game.take_at_kyoku_batch()
                context_meta = np.array(game.take_context_meta_batch()) if self.emit_context_meta else None

                if self.emit_opponent_state_labels:
                    opponent_shanten = game.take_opponent_shanten_batch()
                    opponent_tenpai = game.take_opponent_tenpai_batch()
                if self.track_danger_labels:
                    danger_valid = game.take_danger_valid_batch()
                    danger_any = game.take_danger_any_batch()
                    danger_value = game.take_danger_value_batch()
                    danger_player_mask = game.take_danger_player_mask_batch()
                if self.track_regret_labels:
                    tile_eff_valid = game.take_tile_eff_valid_batch()
                    tile_eff_shanten_delta = game.take_tile_eff_shanten_delta_batch()
                    furo_valid = game.take_furo_valid_batch()
                    furo_label = game.take_furo_label_batch()
                    hand_value_valid = game.take_hand_value_valid_batch()
                    hand_value_points = game.take_hand_value_points_batch()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_value_target = None
                if self.emit_value_targets:
                    kyoku_value_target_abs = value_rewards_all_players(
                        self.value_reward_source,
                        self.reward_calc,
                        grp_feature,
                        rank_by_player,
                        self.pts,
                    )
                    kyoku_value_target = select_value_targets(
                        kyoku_value_target_abs,
                        player_id,
                        self.value_target_mode,
                    )
                    advantage = kyoku_value_target[:, 0]
                else:
                    advantage = self.reward_calc.calc_delta_pt(
                        player_id,
                        grp_feature,
                        rank_by_player,
                    )
                assert len(advantage) >= at_kyoku[-1] + 1

                # player's final rank (0-3) for AuxNet label
                player_rank = rank_by_player[player_id]
                sample_advantage = np.take(
                    np.asarray(advantage),
                    np.asarray(at_kyoku, dtype=np.int64),
                )
                sample_value_target = None
                if kyoku_value_target is not None:
                    sample_value_target = np.take(
                        kyoku_value_target,
                        np.asarray(at_kyoku, dtype=np.int64),
                        axis=0,
                    )

                columns = [obs]
                if self.oracle:
                    columns.append(invisible_obs)
                columns.extend((
                    actions,
                    masks,
                    sample_advantage,
                ))
                if self.emit_value_targets:
                    columns.append(sample_value_target)
                columns.append(repeat(player_rank, game_size))
                if self.emit_context_meta:
                    columns.append(context_meta)
                if self.emit_replay_param_version:
                    columns.append(repeat(-1 if replay_param_version is None else replay_param_version, game_size))
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
                if self.track_regret_labels:
                    columns.extend((
                        tile_eff_valid,
                        tile_eff_shanten_delta,
                        furo_valid,
                        furo_label,
                        hand_value_valid,
                        hand_value_points,
                    ))
                extend_buffer_from_columns(self.buffer, *columns)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

    def iter_game_trajectories(self, file_list, *, oracle_imputation_seed=None):
        """Yield complete game trajectory dicts in temporal step order (no shuffle).
        Used for step-level GAE preprocessing in the main training process.

        An optional fixed imputation seed makes unobserved-wall completion
        reproducible for diagnostics. It does not enable seed reconstruction or
        change the default online input protocol.
        """
        if oracle_imputation_seed is not None:
            if (isinstance(oracle_imputation_seed, bool)
                    or not isinstance(oracle_imputation_seed, int)
                    or not 0 <= oracle_imputation_seed < 2**64):
                raise ValueError('oracle_imputation_seed must be an unsigned 64-bit integer')
        reward_calc = None
        if self.value_reward_source == 'grp':
            grp = GRP(**config['grp']['network'])
            grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')
            grp.load_state_dict(grp_state['model'])
            label_smoothing = config.get('grp', {}).get('label_smoothing', 0.0)
            reward_calc = RewardCalculator(grp, self.pts, label_smoothing=label_smoothing)
        loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=False,
            track_opponent_states=self.track_opponent_states,
            track_danger_labels=self.track_danger_labels,
            track_regret_labels=self.track_regret_labels,
        )
        if oracle_imputation_seed is not None:
            setter = getattr(loader, 'set_oracle_imputation_seed', None)
            if setter is None:
                raise RuntimeError('native loader lacks fixed Oracle imputation support')
            setter(oracle_imputation_seed)
        for source_name, gameplay_batch in iter_loaded_gameplay_batches(loader, file_list):
            for game in gameplay_batch:
                obs = np.array(game.take_obs_batch())
                invisible_obs = np.array(game.take_invisible_obs_batch()) if self.oracle else None
                actions = np.array(game.take_actions_batch())
                masks = np.array(game.take_masks_batch())
                at_kyoku = np.array(game.take_at_kyoku_batch(), dtype=np.int64)
                context_meta = np.array(game.take_context_meta_batch()) if self.emit_context_meta else None
                opp_shanten = np.array(game.take_opponent_shanten_batch()) if self.track_opponent_states else None
                opp_tenpai  = np.array(game.take_opponent_tenpai_batch())  if self.track_opponent_states else None
                danger_valid       = np.array(game.take_danger_valid_batch())       if self.track_danger_labels else None
                danger_any         = np.array(game.take_danger_any_batch())         if self.track_danger_labels else None
                danger_value       = np.array(game.take_danger_value_batch())       if self.track_danger_labels else None
                danger_player_mask = np.array(game.take_danger_player_mask_batch()) if self.track_danger_labels else None
                tile_eff_valid        = np.array(game.take_tile_eff_valid_batch())        if self.track_regret_labels else None
                tile_eff_shanten_delta= np.array(game.take_tile_eff_shanten_delta_batch())if self.track_regret_labels else None
                furo_valid            = np.array(game.take_furo_valid_batch())            if self.track_regret_labels else None
                furo_label            = np.array(game.take_furo_label_batch())            if self.track_regret_labels else None
                hand_value_valid      = np.array(game.take_hand_value_valid_batch())      if self.track_regret_labels else None
                hand_value_points     = np.array(game.take_hand_value_points_batch())     if self.track_regret_labels else None
                grp_feat = game.take_grp()
                player_id = game.take_player_id()
                grp_feature = grp_feat.take_feature()
                rank_by_player = grp_feat.take_rank_by_player()
                kyoku_value_target_abs = np.asarray(
                    value_rewards_all_players(
                        self.value_reward_source,
                        reward_calc,
                        grp_feature,
                        rank_by_player,
                        self.pts,
                    ),
                    dtype=np.float32,
                )
                kyoku_value_target = select_value_targets(
                    kyoku_value_target_abs,
                    player_id,
                    self.value_target_mode,
                )
                yield {
                    'obs': obs, 'invisible_obs': invisible_obs,
                    'actions': actions, 'masks': masks,
                    'at_kyoku': at_kyoku,
                    'player_id': int(player_id),
                    'decision_indices': np.arange(len(obs), dtype=np.int64),
                    'kyoku_advantage': kyoku_value_target[:, 0],
                    'kyoku_value_target': kyoku_value_target,
                    'player_rank': int(rank_by_player[player_id]),
                    'context_meta': context_meta,
                    'replay_param_version': replay_param_version_from_path(source_name),
                    'opp_shanten': opp_shanten, 'opp_tenpai': opp_tenpai,
                    'danger_valid': danger_valid, 'danger_any': danger_any,
                    'danger_value': danger_value, 'danger_player_mask': danger_player_mask,
                    'tile_eff_valid': tile_eff_valid, 'tile_eff_shanten_delta': tile_eff_shanten_delta,
                    'furo_valid': furo_valid, 'furo_label': furo_label,
                    'hand_value_valid': hand_value_valid, 'hand_value_points': hand_value_points,
                }


class SupervisedFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
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
        emit_game_id=False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
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
        self.emit_game_id = bool(emit_game_id)
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
            oracle=False,
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
        for source_name, gameplay_batch in iter_loaded_gameplay_batches(self.loader, file_list):
            game_id = stable_source_game_id(source_name)
            for game in gameplay_batch:
                obs = game.take_obs_batch()
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
                if self.emit_game_id:
                    columns.append(repeat(game_id, game_size))
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
    if bool(getattr(dataset, 'shards_files_in_iter', False)):
        return
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]

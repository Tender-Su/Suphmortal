import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mortal.online import train_online
from mortal.data.oracle_value import (
    OracleTerminalValueDataset,
    centered_rank_points,
    current_rank_values_by_kyoku,
    discounted_returns_from_step_rewards,
    expand_kyoku_rewards_to_steps,
    oracle_values_by_kyoku,
    oracle_step_value_targets,
    rank_array,
    rank_by_score,
    score_rank_delta_rewards_by_kyoku,
    terminal_rank_delta_values_by_kyoku,
    terminal_rank_value_target,
    terminal_rank_values_by_player,
    deterministic_state_fold_indices,
    deterministic_game_id,
    normalize_state_fold_backend,
)
from mortal.data.dataloader import (
    FileDatasetsIter,
    SupervisedFileDatasetsIter,
    iter_loaded_gameplay_batches,
    normalize_value_reward_source,
    replay_param_version_from_path,
    rotate_values_to_relative_order,
    select_value_targets,
    stable_source_game_id,
    worker_init_fn,
)


class SupervisedFileDatasetsIterTests(unittest.TestCase):
    def test_event_cache_bulk_loading_uses_one_native_call(self):
        loader = Mock()
        loader.load_log_files.return_value = [['game-a'], ['game-b']]
        files = ['a.events.zst', 'b.events.zst']

        batches = list(iter_loaded_gameplay_batches(
            loader,
            files,
            bulk_event_cache=True,
        ))

        loader.load_log_files.assert_called_once_with(files)
        self.assertEqual(
            [('a.events.zst#0', ['game-a']), ('a.events.zst#1', ['game-b'])],
            batches,
        )

    def test_event_cache_default_preserves_per_cache_source_names(self):
        loader = Mock()
        loader.load_log_files.side_effect = [[['game-a']], [['game-b']]]
        files = ['a.events.zst', 'b.events.zst']

        batches = list(iter_loaded_gameplay_batches(loader, files))

        self.assertEqual(
            [('a.events.zst', ['game-a']), ('b.events.zst', ['game-b'])],
            batches,
        )

    def test_rotate_values_to_relative_order_puts_self_first(self):
        values = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        rotated = rotate_values_to_relative_order(values, 2)
        np.testing.assert_array_equal(
            rotated,
            np.array([[30.0, 40.0, 10.0, 20.0]], dtype=np.float32),
        )

    def test_select_value_targets_can_reduce_to_current_player_only(self):
        values = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        selected = select_value_targets(values, 2, 'current_player')
        np.testing.assert_array_equal(
            selected,
            np.array([[30.0]], dtype=np.float32),
        )

    def test_replay_param_version_from_path_parses_online_prefix(self):
        self.assertEqual(
            123,
            replay_param_version_from_path(r'C:\tmp\pv123_sid45_game_0001.json.gz'),
        )
        self.assertIsNone(
            replay_param_version_from_path(r'C:\tmp\game_0001.json.gz'),
        )

    def test_danger_only_mode_does_not_request_opponent_label_emission(self):
        with patch('mortal.data.dataloader.GameplayLoader') as loader_cls:
            dataset = SupervisedFileDatasetsIter(
                version=4,
                file_list=[],
                emit_opponent_state_labels=False,
                track_danger_labels=True,
                shuffle_files=False,
            )
            list(dataset.load_files(False))

        kwargs = loader_cls.call_args.kwargs
        self.assertFalse(kwargs['track_opponent_states'])
        self.assertTrue(kwargs['track_danger_labels'])

    def test_opponent_aux_mode_still_requests_opponent_labels(self):
        with patch('mortal.data.dataloader.GameplayLoader') as loader_cls:
            dataset = SupervisedFileDatasetsIter(
                version=4,
                file_list=[],
                emit_opponent_state_labels=True,
                track_danger_labels=False,
                shuffle_files=False,
            )
            list(dataset.load_files(False))

        kwargs = loader_cls.call_args.kwargs
        self.assertTrue(kwargs['track_opponent_states'])
        self.assertFalse(kwargs['track_danger_labels'])

    def test_supervised_validation_can_emit_stable_source_game_id(self):
        dataset = SupervisedFileDatasetsIter(
            version=4,
            file_list=['dummy.json'],
            emit_opponent_state_labels=False,
            track_danger_labels=False,
            emit_game_id=True,
            shuffle_files=False,
        )
        dataset.buffer = []
        dataset.loader = object()

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['dummy.json'])

        self.assertEqual(2, len(dataset.buffer))
        self.assertTrue(all(len(row) == 6 for row in dataset.buffer))
        self.assertTrue(all(
            int(row[-1]) == stable_source_game_id('dummy.json')
            for row in dataset.buffer
        ))


class _FakeGrp:
    def take_feature(self):
        return np.array(
            [
                [0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5],
                [1.0, 0.0, 0.0, 3.0, 2.0, 4.0, 1.0],
            ],
            dtype=np.float32,
        )

    def take_rank_by_player(self):
        return np.array([0, 1, 2, 3], dtype=np.int64)


class _FakeGame:
    def take_obs_batch(self):
        return np.zeros((2, 3, 34), dtype=np.float32)

    def take_actions_batch(self):
        return np.array([0, 1], dtype=np.int64)

    def take_masks_batch(self):
        return np.ones((2, 46), dtype=np.bool_)

    def take_at_kyoku_batch(self):
        return np.array([0, 1], dtype=np.int64)

    def take_context_meta_batch(self):
        return np.array(
            [
                [3, 0, 1, 0, 1, 0, 20, 40],
                [12, 1, 0, 1, 2, 1, 10, 65535],
            ],
            dtype=np.int64,
        )

    def take_grp(self):
        return _FakeGrp()

    def take_player_id(self):
        return 2


class _FakeOracleGame(_FakeGame):
    def take_full_at_kyoku_batch(self):
        return np.array([0, 1], dtype=np.int64)

    def take_sample_indices_batch(self):
        return np.array([0, 1], dtype=np.int64)

    def take_invisible_obs_batch(self):
        return np.ones((2, 2, 34), dtype=np.float32)


class OracleTerminalValueDatasetTests(unittest.TestCase):
    def test_centered_rank_points_make_default_pts_zero_sum(self):
        np.testing.assert_array_equal(
            centered_rank_points([6.0, 4.0, 2.0, 0.0]),
            np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float32),
        )

    def test_terminal_rank_values_map_rank_ids_to_centered_points(self):
        values = terminal_rank_values_by_player(
            np.array([2, 0, 3, 1], dtype=np.int64),
            [6.0, 4.0, 2.0, 0.0],
        )
        np.testing.assert_array_equal(
            values,
            np.array([-1.0, 3.0, -3.0, 1.0], dtype=np.float32),
        )

    def test_rank_array_accepts_py03_bytes_for_u8_array(self):
        np.testing.assert_array_equal(
            rank_array(bytes([2, 0, 3, 1])),
            np.array([2, 0, 3, 1], dtype=np.int64),
        )

    def test_rank_by_score_matches_repo_tie_breaking(self):
        np.testing.assert_array_equal(
            rank_by_score(
                np.array(
                    [
                        [25000, 25000, 30000, 20000],
                        [32000, 18000, 18000, 32000],
                    ],
                    dtype=np.float32,
                )
            ),
            np.array(
                [
                    [1, 2, 0, 3],
                    [0, 2, 3, 1],
                ],
                dtype=np.int64,
            ),
        )

    def test_rank_delta_values_subtract_current_score_rank_baseline(self):
        grp_feature = np.array(
            [
                [0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5],
                [1.0, 0.0, 0.0, 3.0, 2.0, 4.0, 1.0],
            ],
            dtype=np.float32,
        )
        rank_by_player = np.array([2, 0, 3, 1], dtype=np.int64)

        np.testing.assert_array_equal(
            current_rank_values_by_kyoku(grp_feature, [6.0, 4.0, 2.0, 0.0]),
            np.array(
                [
                    [3.0, 1.0, -1.0, -3.0],
                    [1.0, -1.0, 3.0, -3.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            terminal_rank_delta_values_by_kyoku(
                grp_feature,
                rank_by_player,
                [6.0, 4.0, 2.0, 0.0],
            ),
            np.array(
                [
                    [-4.0, 2.0, -2.0, 4.0],
                    [-2.0, 4.0, -6.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_oracle_values_can_keep_terminal_rank_baseline_for_ablation(self):
        values = oracle_values_by_kyoku(
            np.array(
                [
                    [0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5],
                    [1.0, 0.0, 0.0, 3.0, 2.0, 4.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array([2, 0, 3, 1], dtype=np.int64),
            [6.0, 4.0, 2.0, 0.0],
            'terminal_rank',
        )
        np.testing.assert_array_equal(
            values,
            np.array(
                [
                    [-1.0, 3.0, -3.0, 1.0],
                    [-1.0, 3.0, -3.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_score_rank_mc_targets_match_online_sparse_reward_contract(self):
        grp_feature = np.array(
            [
                [0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5],
                [1.0, 0.0, 0.0, 3.0, 2.0, 4.0, 1.0],
                [2.0, 0.0, 0.0, 1.0, 5.0, 3.0, 1.0],
            ],
            dtype=np.float32,
        )
        rank_by_player = np.array([2, 0, 3, 1], dtype=np.int64)
        at_kyoku = np.array([0, 0, 1, 1, 2], dtype=np.int64)

        kyoku_rewards = score_rank_delta_rewards_by_kyoku(
            grp_feature,
            rank_by_player,
            [6.0, 4.0, 2.0, 0.0],
        )
        np.testing.assert_array_equal(
            kyoku_rewards,
            np.array(
                [
                    [-2.0, -2.0, 4.0, 0.0],
                    [-2.0, 4.0, -2.0, 0.0],
                    [0.0, 0.0, -4.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )

        step_rewards = expand_kyoku_rewards_to_steps(kyoku_rewards, at_kyoku)
        np.testing.assert_array_equal(
            step_rewards,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-2.0, -2.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-2.0, 4.0, -2.0, 0.0],
                    [0.0, 0.0, -4.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            discounted_returns_from_step_rewards(step_rewards, gamma=1.0),
            oracle_step_value_targets(
                grp_feature,
                rank_by_player,
                [6.0, 4.0, 2.0, 0.0],
                at_kyoku,
                'score_rank_mc',
                gamma=1.0,
            ),
        )
        np.testing.assert_array_equal(
            oracle_step_value_targets(
                grp_feature,
                rank_by_player,
                [6.0, 4.0, 2.0, 0.0],
                at_kyoku,
                'score_rank_mc',
                gamma=1.0,
            ),
            np.array(
                [
                    [-4.0, 2.0, -2.0, 4.0],
                    [-4.0, 2.0, -2.0, 4.0],
                    [-2.0, 4.0, -6.0, 4.0],
                    [-2.0, 4.0, -6.0, 4.0],
                    [0.0, 0.0, -4.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            train_online.compute_gae_advantages(
                kyoku_rewards[:, 0],
                at_kyoku,
                np.zeros((at_kyoku.shape[0],), dtype=np.float32),
                gamma=1.0,
                lam=1.0,
            ),
            oracle_step_value_targets(
                grp_feature,
                rank_by_player,
                [6.0, 4.0, 2.0, 0.0],
                at_kyoku,
                'score_rank_mc',
                gamma=1.0,
            )[:, 0],
        )

    def test_score_rank_mc_keeps_rewards_for_kyoku_without_player_samples(self):
        grp_feature = np.array(
            [
                [0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5],
                [1.0, 0.0, 0.0, 3.0, 2.0, 4.0, 1.0],
                [2.0, 0.0, 0.0, 1.0, 5.0, 3.0, 1.0],
            ],
            dtype=np.float32,
        )
        rank_by_player = np.array([2, 0, 3, 1], dtype=np.int64)
        at_kyoku = np.array([0, 0, 2], dtype=np.int64)
        kyoku_rewards = score_rank_delta_rewards_by_kyoku(
            grp_feature,
            rank_by_player,
            [6.0, 4.0, 2.0, 0.0],
        )

        np.testing.assert_array_equal(
            expand_kyoku_rewards_to_steps(kyoku_rewards, at_kyoku),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-4.0, 2.0, 2.0, 0.0],
                    [0.0, 0.0, -4.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            oracle_step_value_targets(
                grp_feature,
                rank_by_player,
                [6.0, 4.0, 2.0, 0.0],
                at_kyoku,
                'score_rank_mc',
                gamma=1.0,
            ),
            np.array(
                [
                    [-4.0, 2.0, -2.0, 4.0],
                    [-4.0, 2.0, -2.0, 4.0],
                    [0.0, 0.0, -4.0, 4.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_terminal_rank_target_rotates_to_current_player_order(self):
        target = terminal_rank_value_target(
            np.array([2, 0, 3, 1], dtype=np.int64),
            2,
            [6.0, 4.0, 2.0, 0.0],
            'all_players',
        )
        np.testing.assert_array_equal(
            target,
            np.array([-3.0, 1.0, -1.0, 3.0], dtype=np.float32),
        )

    def test_oracle_dataset_emits_score_rank_mc_targets_without_reward_calculator(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=1,
            shuffle_files=False,
        )
        buffer = []

        with patch(
            'mortal.data.oracle_value.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeOracleGame()])]),
        ):
            dataset.populate_buffer(object(), ['dummy.json'], buffer)

        self.assertEqual(2, len(buffer))
        obs, invisible_obs, target, player_id = buffer[0]
        self.assertEqual((3, 34), obs.shape)
        self.assertEqual((2, 34), invisible_obs.shape)
        self.assertEqual(2, int(player_id))
        np.testing.assert_array_equal(
            target,
            np.zeros((4,), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            buffer[1][2],
            np.array([-4.0, 0.0, 2.0, 2.0], dtype=np.float32),
        )

    def test_oracle_validation_dataset_emits_stable_game_id(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=1,
            shuffle_files=False,
            emit_game_id=True,
        )
        buffer = []

        with patch(
            'mortal.data.oracle_value.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeOracleGame()])]),
        ):
            dataset.populate_buffer(object(), ['dummy.json'], buffer)

        self.assertEqual(deterministic_game_id('dummy.json'), int(buffer[0][-1]))
        self.assertEqual(int(buffer[0][-1]), int(buffer[1][-1]))
        self.assertNotEqual(
            deterministic_game_id('dummy.json'),
            deterministic_game_id('other.json'),
        )

    def test_oracle_validation_dataset_does_not_shuffle_files_or_buffer(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=1,
            shuffle_files=False,
        )
        dataset.populate_buffer = Mock(
            side_effect=lambda _loader, files, buffer, **_kwargs: buffer.extend(files)
        )

        with patch(
            'mortal.data.oracle_value.random.shuffle',
            side_effect=AssertionError('validation must not shuffle'),
        ):
            rows = list(dataset.load_files(augmented=False))

        self.assertEqual(['dummy.json'], rows)

    def test_oracle_train_dataset_shuffles_files_and_buffer(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=1,
            shuffle_files=True,
        )
        dataset.populate_buffer = Mock(
            side_effect=lambda _loader, files, buffer, **_kwargs: buffer.extend(files)
        )

        with patch('mortal.data.oracle_value.random.shuffle') as shuffle:
            rows = list(dataset.load_files(augmented=False))

        self.assertEqual(['dummy.json'], rows)
        self.assertGreaterEqual(shuffle.call_count, 2)

    def test_native_state_fold_is_configured_before_feature_loading(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=1,
            shuffle_files=False,
            state_fold_count=8,
            state_fold_seed=99,
            state_fold_backend='native_hash',
        )
        loader = Mock()
        dataset.populate_buffer = Mock(
            side_effect=lambda _loader, files, buffer, **_kwargs: buffer.extend(files)
        )

        with patch('mortal.data.oracle_value.GameplayLoader', return_value=loader):
            rows = list(dataset.load_files(False, stream_pass=3))

        self.assertEqual(['dummy.json'], rows)
        loader.set_sample_fold.assert_called_once_with(8, 3, 99)

    def test_native_state_fold_does_not_filter_decoded_rows_twice(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=1,
            shuffle_files=False,
            state_fold_count=64,
            state_fold_backend='native_hash',
        )
        buffer = []

        with patch(
            'mortal.data.oracle_value.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeOracleGame()])]),
        ):
            dataset.populate_buffer(object(), ['dummy.json'], buffer, state_fold_index=37)

        self.assertEqual(2, len(buffer))

    def test_state_fold_backend_aliases_are_explicit(self):
        self.assertEqual('python_permutation', normalize_state_fold_backend('python'))
        self.assertEqual('native_hash', normalize_state_fold_backend('native'))
        with self.assertRaisesRegex(ValueError, 'unsupported state_fold_backend'):
            normalize_state_fold_backend('magic')

    def test_oracle_stream_resume_replays_current_chunk_then_continues_deterministically(self):
        def make_dataset(resume_cursors=None):
            dataset = OracleTerminalValueDataset(
                version=4,
                file_list=[f'{idx}.json' for idx in range(6)],
                pts=[6.0, 4.0, 2.0, 0.0],
                file_batch_size=1,
                shuffle_files=True,
                shuffle_seed=123,
                resume_cursors=resume_cursors,
                emit_progress=True,
            )

            def populate(_loader, files, buffer, *, progress_token=None, **_kwargs):
                buffer.extend((filename, None, None, None, progress_token) for filename in files)

            dataset.populate_buffer = Mock(side_effect=populate)
            return dataset

        full_rows = list(make_dataset().load_files(False, stream_pass=0))
        resumed_rows = list(
            make_dataset({0: (0, 2)}).load_files(
                False,
                stream_pass=0,
                start_file_offset=2,
            )
        )

        self.assertEqual([row[0] for row in full_rows[2:]], [row[0] for row in resumed_rows])
        self.assertTrue(all(tuple(row[-1]) >= (0, 0, 2) for row in resumed_rows))

    def test_oracle_workers_consume_disjoint_complete_file_shards(self):
        def rows_for_worker(worker_id, resume_offset=0):
            dataset = OracleTerminalValueDataset(
                version=4,
                file_list=[f'{idx}.json' for idx in range(12)],
                pts=[6.0, 4.0, 2.0, 0.0],
                file_batch_size=1,
                shuffle_files=True,
                shuffle_seed=123,
                emit_progress=True,
            )

            def populate(_loader, files, buffer, *, progress_token=None, **_kwargs):
                buffer.extend((filename, None, None, None, progress_token) for filename in files)

            dataset.populate_buffer = Mock(side_effect=populate)
            worker_info = Mock(id=worker_id, num_workers=2, seed=10_000 + worker_id)
            with patch('mortal.data.oracle_value.get_worker_info', return_value=worker_info):
                return list(dataset.load_files(
                    False,
                    stream_pass=0,
                    start_file_offset=resume_offset,
                ))

        worker_0 = rows_for_worker(0)
        worker_1 = rows_for_worker(1)
        files_0 = [row[0] for row in worker_0]
        files_1 = [row[0] for row in worker_1]

        self.assertFalse(set(files_0) & set(files_1))
        self.assertEqual({f'{idx}.json' for idx in range(12)}, set(files_0) | set(files_1))
        self.assertEqual(files_0[2:], [row[0] for row in rows_for_worker(0, resume_offset=2)])
        self.assertEqual(files_1[2:], [row[0] for row in rows_for_worker(1, resume_offset=2)])
        self.assertTrue(all(int(row[-1][0]) == 0 for row in worker_0))
        self.assertTrue(all(int(row[-1][0]) == 1 for row in worker_1))

    def test_worker_init_does_not_pre_shard_oracle_file_list(self):
        dataset = OracleTerminalValueDataset(
            version=4,
            file_list=[f'{idx}.json' for idx in range(8)],
            pts=[6.0, 4.0, 2.0, 0.0],
        )
        worker_info = Mock(id=1, num_workers=2, dataset=dataset)

        with (
            patch('mortal.data.dataloader.maybe_configure_process_affinity'),
            patch('mortal.data.dataloader.torch.utils.data.get_worker_info', return_value=worker_info),
            patch('mortal.data.dataloader.torch.set_num_threads'),
            patch('mortal.data.dataloader.torch.set_num_interop_threads'),
        ):
            worker_init_fn()

        self.assertEqual([f'{idx}.json' for idx in range(8)], dataset.file_list)

    def test_worker_init_still_shards_legacy_file_dataset(self):
        dataset = SimpleNamespace(
            file_list=[f'{idx}.json' for idx in range(8)],
            rayon_num_threads=0,
            worker_torch_num_threads=1,
            worker_torch_num_interop_threads=1,
        )
        worker_info = Mock(id=1, num_workers=2, dataset=dataset)

        with (
            patch('mortal.data.dataloader.maybe_configure_process_affinity'),
            patch('mortal.data.dataloader.torch.utils.data.get_worker_info', return_value=worker_info),
            patch('mortal.data.dataloader.torch.set_num_threads'),
            patch('mortal.data.dataloader.torch.set_num_interop_threads'),
        ):
            worker_init_fn()

        self.assertEqual([f'{idx}.json' for idx in range(4, 8)], dataset.file_list)

    def test_oracle_state_folds_are_deterministic_disjoint_and_complete(self):
        kwargs = {
            'size': 23,
            'fold_count': 4,
            'seed': 99,
            'source_name': 'game.json',
            'game_index': 0,
            'player_id': 2,
        }
        folds = [
            deterministic_state_fold_indices(fold_index=fold_index, **kwargs)
            for fold_index in range(4)
        ]

        self.assertEqual(
            folds[0].tolist(),
            deterministic_state_fold_indices(fold_index=0, **kwargs).tolist(),
        )
        self.assertEqual(list(range(23)), sorted(np.concatenate(folds).tolist()))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(set(folds[left]) & set(folds[right]))


class FileDatasetsIterRewardTests(unittest.TestCase):
    def test_without_value_targets_uses_selected_player_delta_pt(self):
        dataset = FileDatasetsIter(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            emit_value_targets=False,
        )
        dataset.buffer = []
        dataset.loader = object()
        dataset.reward_calc = Mock()
        dataset.reward_calc.calc_delta_pt.return_value = np.array([0.5, -0.25], dtype=np.float32)
        dataset.reward_calc.calc_delta_pt_all_players.side_effect = AssertionError('should not be called')

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['dummy.json'])

        dataset.reward_calc.calc_delta_pt.assert_called_once()
        dataset.reward_calc.calc_delta_pt_all_players.assert_not_called()
        self.assertEqual(2, len(dataset.buffer))
        self.assertAlmostEqual(0.5, float(dataset.buffer[0][3]))
        self.assertAlmostEqual(-0.25, float(dataset.buffer[1][3]))

    def test_with_value_targets_uses_all_player_delta_pt(self):
        dataset = FileDatasetsIter(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            emit_value_targets=True,
            value_target_mode='all_players',
        )
        dataset.buffer = []
        dataset.loader = object()
        dataset.reward_calc = Mock()
        dataset.reward_calc.calc_delta_pt.side_effect = AssertionError('should not be called')
        dataset.reward_calc.calc_delta_pt_all_players.return_value = np.array(
            [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
            dtype=np.float32,
        )

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['dummy.json'])

        dataset.reward_calc.calc_delta_pt.assert_not_called()
        dataset.reward_calc.calc_delta_pt_all_players.assert_called_once()
        self.assertEqual(2, len(dataset.buffer))
        self.assertEqual((4,), dataset.buffer[0][4].shape)

    def test_with_value_targets_can_use_current_player_only(self):
        dataset = FileDatasetsIter(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            emit_value_targets=True,
            value_target_mode='current_player',
        )
        dataset.buffer = []
        dataset.loader = object()
        dataset.reward_calc = Mock()
        dataset.reward_calc.calc_delta_pt.side_effect = AssertionError('should not be called')
        dataset.reward_calc.calc_delta_pt_all_players.return_value = np.array(
            [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
            dtype=np.float32,
        )

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['dummy.json'])

        self.assertEqual(2, len(dataset.buffer))
        self.assertEqual((1,), dataset.buffer[0][4].shape)
        self.assertAlmostEqual(3.0, float(dataset.buffer[0][4][0]))

    def test_with_value_targets_can_use_score_rank_reward_source(self):
        dataset = FileDatasetsIter(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            emit_value_targets=True,
            value_target_mode='all_players',
            value_reward_source='score_rank',
        )
        dataset.buffer = []
        dataset.loader = object()
        dataset.reward_calc = Mock()
        dataset.reward_calc.calc_delta_pt_all_players.side_effect = AssertionError('should not use GRP')

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['dummy.json'])

        dataset.reward_calc.calc_delta_pt_all_players.assert_not_called()
        self.assertEqual(2, len(dataset.buffer))
        np.testing.assert_array_equal(
            dataset.buffer[0][3],
            np.float32(4.0),
        )
        np.testing.assert_array_equal(
            dataset.buffer[0][4],
            np.array([4.0, 0.0, -2.0, -2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            dataset.buffer[1][4],
            np.array([-4.0, 0.0, 2.0, 2.0], dtype=np.float32),
        )

    def test_value_reward_source_normalizer_accepts_score_rank_aliases(self):
        self.assertEqual('score_rank', normalize_value_reward_source('score_rank_mc'))
        self.assertEqual('grp', normalize_value_reward_source('global_reward_predictor'))

    def test_emit_context_meta_appends_context_batch_after_player_rank(self):
        dataset = FileDatasetsIter(
            version=4,
            file_list=['dummy.json'],
            pts=[6.0, 4.0, 2.0, 0.0],
            emit_context_meta=True,
        )
        dataset.buffer = []
        dataset.loader = object()
        dataset.reward_calc = Mock()
        dataset.reward_calc.calc_delta_pt.return_value = np.array([0.5, -0.25], dtype=np.float32)

        with patch(
            'mortal.data.dataloader.iter_loaded_gameplay_batches',
            return_value=iter([('dummy.json', [_FakeGame()])]),
        ):
            dataset.populate_buffer(['dummy.json'])

        self.assertEqual(2, len(dataset.buffer))
        first_row = dataset.buffer[0]
        self.assertEqual(2, int(first_row[4]))
        np.testing.assert_array_equal(
            first_row[5],
            np.array([3, 0, 1, 0, 1, 0, 20, 40], dtype=np.int64),
        )


if __name__ == '__main__':
    unittest.main()

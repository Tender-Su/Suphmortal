import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_online


def make_config(*, online, version=4, next_rank_weight=0.0):
    return {
        'control': {
            'online': online,
            'version': version,
        },
        'online': {},
        'supervised': {},
        'resnet': {
            'channels': 192,
            'num_blocks': 40,
        },
        'aux': {
            'next_rank_weight': next_rank_weight,
        },
    }


def make_optimizer_state(*, group_sizes):
    return {
        'state': {},
        'param_groups': [
            {'params': list(range(size))}
            for size in group_sizes
        ],
    }


class DummyOptimizer:
    def __init__(self, group_sizes):
        self.param_groups = [
            {'params': [object() for _ in range(size)]}
            for size in group_sizes
        ]


class TrainOnlineCheckpointTests(unittest.TestCase):
    def test_resolve_online_init_state_file_prefers_online_override(self):
        config = make_config(online=True)
        config['online']['init_state_file'] = './checkpoints/custom_supervised_winner.pth'
        config['supervised']['best_loss_state_file'] = './checkpoints/sl_canonical.pth'

        self.assertEqual(
            './checkpoints/custom_supervised_winner.pth',
            train_online.resolve_online_init_state_file(config),
        )

    def test_resolve_online_init_state_file_falls_back_to_supervised_best_loss(self):
        config = make_config(online=True)
        config['supervised']['best_loss_state_file'] = './checkpoints/sl_canonical.pth'

        self.assertEqual(
            './checkpoints/sl_canonical.pth',
            train_online.resolve_online_init_state_file(config),
        )

    def test_resolve_online_init_state_file_returns_empty_when_missing(self):
        self.assertEqual('', train_online.resolve_online_init_state_file(make_config(online=True)))

    def test_ensure_online_init_state_file_ready_checks_canonical_handoff(self):
        with patch(
            'run_sl_formal.ensure_supervised_canonical_handoff_ready',
            side_effect=RuntimeError('pending formal_1v3 handoff'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'pending formal_1v3 handoff'):
                train_online.ensure_online_init_state_file_ready('./checkpoints/sl_canonical.pth')

    def test_ensure_online_init_state_file_ready_requires_existing_file_after_handoff_check(self):
        with patch('run_sl_formal.ensure_supervised_canonical_handoff_ready'):
            with self.assertRaisesRegex(FileNotFoundError, r'online\.init_state_file does not exist'):
                train_online.ensure_online_init_state_file_ready(r'X:\missing\sl_canonical.pth')

    def test_checkpoint_supports_online_resume_requires_online_training_state(self):
        state = {
            'config': make_config(online=True),
            'optimizer': make_optimizer_state(group_sizes=[2, 1]),
            'scheduler': {'state': {}},
            'scaler': {'scale': 1.0},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        self.assertTrue(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=True),
                optimizer=DummyOptimizer([2, 1]),
            )
        )

    def test_checkpoint_supports_online_resume_accepts_compatible_offline_training_state(self):
        state = {
            'config': make_config(online=False, next_rank_weight=0.25),
            'optimizer': make_optimizer_state(group_sizes=[2, 1, 1]),
            'scheduler': {'state': {}},
            'scaler': {'scale': 1.0},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        self.assertTrue(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=False, next_rank_weight=0.25),
                optimizer=DummyOptimizer([2, 1, 1]),
            )
        )

    def test_checkpoint_supports_online_resume_accepts_online_flag_mismatch_when_layout_matches(self):
        state = {
            'config': make_config(online=True, next_rank_weight=0.25),
            'optimizer': make_optimizer_state(group_sizes=[2, 1, 1]),
            'scheduler': {'state': {}},
            'scaler': {'scale': 1.0},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        self.assertTrue(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=False, next_rank_weight=0.25),
                optimizer=DummyOptimizer([2, 1, 1]),
            )
        )

    def test_checkpoint_supports_online_resume_rejects_version_mismatch(self):
        state = {
            'config': make_config(online=False),
            'optimizer': make_optimizer_state(group_sizes=[2, 1]),
            'scheduler': {'state': {}},
            'scaler': {'scale': 1.0},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        self.assertFalse(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=False, version=3),
                optimizer=DummyOptimizer([2, 1]),
            )
        )

    def test_checkpoint_supports_online_resume_rejects_param_group_layout_mismatch(self):
        state = {
            'config': make_config(online=False, next_rank_weight=0.25),
            'optimizer': make_optimizer_state(group_sizes=[2, 1, 1]),
            'scheduler': {'state': {}},
            'scaler': {'scale': 1.0},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        self.assertFalse(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=False, next_rank_weight=0.0),
                optimizer=DummyOptimizer([2, 1]),
            )
        )

    def test_checkpoint_supports_online_resume_rejects_non_resumable_handoff_exports(self):
        state = {
            'resume_supported': False,
            'config': make_config(online=False),
            'steps': 400000,
        }

        self.assertFalse(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=False),
                optimizer=DummyOptimizer([2, 1]),
            )
        )

    def test_checkpoint_supports_online_resume_rejects_incomplete_online_state(self):
        state = {
            'config': make_config(online=True),
            'optimizer': make_optimizer_state(group_sizes=[2, 1]),
            'scheduler': {'state': {}},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        self.assertFalse(
            train_online.checkpoint_supports_online_resume(
                state,
                current_config=make_config(online=True),
                optimizer=DummyOptimizer([2, 1]),
            )
        )


if __name__ == '__main__':
    unittest.main()

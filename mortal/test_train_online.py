import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_online


def make_config(*, online, version=4, next_rank_weight=0.0,
                value_enabled=False, oracle_critic=False,
                tile_eff_weight=0.0, furo_regret_weight=0.0,
                hand_value_regret_weight=0.0,
                exp_reward_enabled=False,
                opponent_state_weight=0.0, danger_enabled=False, danger_weight=0.0):
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
            'tile_efficiency_weight': tile_eff_weight,
            'furo_regret_weight': furo_regret_weight,
            'hand_value_regret_weight': hand_value_regret_weight,
            'opponent_state_weight': opponent_state_weight,
            'danger_enabled': danger_enabled,
            'danger_weight': danger_weight,
        },
        'value': {
            'enabled': value_enabled,
            'oracle_critic': oracle_critic,
        },
        'expected_reward': {
            'enabled': exp_reward_enabled,
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

    # --- New tests for added features ---

    def test_signature_rejects_value_head_mismatch(self):
        """Enabling value head must change the model signature."""
        state = {
            'config': make_config(online=True),
            'optimizer': make_optimizer_state(group_sizes=[2, 1]),
            'scheduler': {'state': {}},
            'scaler': {'scale': 1.0},
            'best_perf': {'avg_rank': 3.0, 'avg_pt': -10.0},
            'steps': 123,
        }

        sig_without = train_online.online_resume_model_signature(make_config(online=True))
        sig_with = train_online.online_resume_model_signature(
            make_config(online=True, value_enabled=True, oracle_critic=True)
        )
        self.assertNotEqual(sig_without, sig_with)

    def test_signature_rejects_regret_head_mismatch(self):
        """Enabling regret heads must change the model signature."""
        sig_base = train_online.online_resume_model_signature(make_config(online=True))
        sig_tile_eff = train_online.online_resume_model_signature(
            make_config(online=True, tile_eff_weight=0.1)
        )
        sig_furo = train_online.online_resume_model_signature(
            make_config(online=True, furo_regret_weight=0.1)
        )
        self.assertNotEqual(sig_base, sig_tile_eff)
        self.assertNotEqual(sig_base, sig_furo)

    def test_signature_rejects_expected_reward_mismatch(self):
        """Enabling expected reward net must change the model signature."""
        sig_base = train_online.online_resume_model_signature(make_config(online=True))
        sig_exp = train_online.online_resume_model_signature(
            make_config(online=True, exp_reward_enabled=True)
        )
        self.assertNotEqual(sig_base, sig_exp)

    def test_signature_rejects_opp_danger_mismatch(self):
        """Enabling opp/danger heads must change the model signature."""
        sig_base = train_online.online_resume_model_signature(make_config(online=True))
        sig_opp = train_online.online_resume_model_signature(
            make_config(online=True, opponent_state_weight=0.03)
        )
        sig_danger = train_online.online_resume_model_signature(
            make_config(online=True, danger_enabled=True, danger_weight=0.05)
        )
        self.assertNotEqual(sig_base, sig_opp)
        self.assertNotEqual(sig_base, sig_danger)

    def test_signature_stable_with_same_config(self):
        """Same config should produce same signature."""
        cfg = make_config(
            online=True, next_rank_weight=0.2,
            value_enabled=True, oracle_critic=True,
            tile_eff_weight=0.1, furo_regret_weight=0.05,
            exp_reward_enabled=True,
        )
        sig1 = train_online.online_resume_model_signature(cfg)
        sig2 = train_online.online_resume_model_signature(cfg)
        self.assertEqual(sig1, sig2)


class LabelSmoothingTests(unittest.TestCase):
    def test_label_smoothing_sums_to_one(self):
        """Smoothed final ranking should sum to 1.0."""
        import torch
        from reward_calculator import RewardCalculator

        # Create a minimal dummy GRP
        from model import GRP
        grp = GRP(hidden_size=8, num_layers=1, dtype='float32')
        rc = RewardCalculator(grp=grp, label_smoothing=0.1)

        rank_by_player = [2, 0, 3, 1]  # player 0 is 3rd, player 1 is 1st, etc.
        # Build a minimal grp_feature (1 kyoku)
        grp_feature = [[0, 0, 0, 2.5, 2.5, 2.5, 2.5]]
        rank_prob = rc.calc_rank_prob(0, grp_feature, rank_by_player)
        # Last row is the smoothed final ranking
        final = rank_prob[-1]
        self.assertAlmostEqual(final.sum().item(), 1.0, places=5)
        # Correct rank (rank_by_player[0]=2, so 3rd place) should have highest probability
        self.assertGreater(final[2].item(), final[0].item())
        self.assertGreater(final[2].item(), final[1].item())
        self.assertGreater(final[2].item(), final[3].item())

    def test_no_smoothing_is_one_hot(self):
        """Without smoothing, final ranking should be one-hot."""
        import torch
        from reward_calculator import RewardCalculator
        from model import GRP

        grp = GRP(hidden_size=8, num_layers=1, dtype='float32')
        rc = RewardCalculator(grp=grp, label_smoothing=0.0)

        rank_by_player = [1, 0, 3, 2]
        grp_feature = [[0, 0, 0, 2.5, 2.5, 2.5, 2.5]]
        rank_prob = rc.calc_rank_prob(0, grp_feature, rank_by_player)
        final = rank_prob[-1]
        self.assertAlmostEqual(final[1].item(), 1.0, places=5)
        self.assertAlmostEqual(final[0].item(), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()

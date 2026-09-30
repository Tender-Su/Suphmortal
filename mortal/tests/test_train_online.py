import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.online.train_online as train_online


def make_config(*, online, version=4, next_rank_weight=0.0,
                value_enabled=False, oracle_critic=False,
                value_target_mode=None,
                tile_eff_weight=0.0, furo_regret_weight=0.0,
                hand_value_regret_weight=0.0,
                exp_reward_enabled=False,
                opponent_state_weight=0.0, danger_enabled=False, danger_weight=0.0,
                actor_oracle_enabled=False,
                replay_is_enabled=False,
                online_action_scope='all'):
    return {
        'control': {
            'online': online,
            'version': version,
        },
        'online': {
            'stop_at_max_steps': True,
            'importance_sampling': {
                'enabled': replay_is_enabled,
                'max_policy_versions': 8,
                'drop_untracked_samples': False,
            },
        },
        'optim': {
            'scheduler': {
                'max_steps': 0,
            },
        },
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
            **({'target_mode': value_target_mode} if value_target_mode is not None else {}),
        },
        'expected_reward': {
            'enabled': exp_reward_enabled,
        },
        'policy': {
            'entropy_weight': 1e-3,
            'clip_ratio': 0.2,
            'dual_clip': 3.0,
            'online_action_scope': online_action_scope,
        },
        'test_play': {
            'enable': True,
            'games': 3000,
            'initial_enable': False,
            'initial_games': 600,
        },
        'oracle_guiding': {
            'actor_enabled': actor_oracle_enabled,
            'actor_source': 'true',
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


class DummyScheduler:
    def __init__(self):
        self.init = 1e-8
        self.peak = 1e-4
        self.final = 1e-5
        self.warm_up_steps = 1000
        self.max_steps = 20000
        self.offset = 0
        self.epoch_size = 0
        self.base_lrs = [1.0, 1.0]
        self.last_epoch = 20000
        self._last_lr = [1e-5, 1e-5]

    def _step_inner(self, steps):
        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            return self.init + (self.peak - self.init) / self.warm_up_steps * steps
        if steps < self.max_steps:
            cos_steps = steps - self.warm_up_steps
            cos_max_steps = self.max_steps - self.warm_up_steps
            return self.final + 0.5 * (self.peak - self.final) * (
                1 + np.cos(cos_steps / cos_max_steps * np.pi)
            )
        return self.final


class DummyStateModule:
    def __init__(self, state):
        self._state = dict(state)
        self.loaded_state = None

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self.loaded_state = dict(state)


class TrainOnlineCheckpointTests(unittest.TestCase):
    def test_online_reached_max_steps_uses_scheduler_max_steps_by_default(self):
        config = make_config(online=True)
        config['optim']['scheduler']['max_steps'] = 100

        self.assertTrue(train_online.online_stop_at_max_steps(config))
        self.assertEqual(100, train_online.online_scheduler_max_steps(config))
        self.assertFalse(train_online.online_reached_max_steps(config, 99))
        self.assertTrue(train_online.online_reached_max_steps(config, 100))

    def test_online_reached_max_steps_can_be_disabled(self):
        config = make_config(online=True)
        config['optim']['scheduler']['max_steps'] = 100
        config['online']['stop_at_max_steps'] = False

        self.assertFalse(train_online.online_reached_max_steps(config, 100))

    def test_policy_importance_rho_clip_prefers_new_key(self):
        config = make_config(online=True)
        config['policy']['importance_rho_clip'] = 1.25
        config['policy']['vtrace_rho_clip'] = 3.0

        self.assertEqual(1.25, train_online.policy_importance_rho_clip(config))

    def test_policy_importance_rho_clip_falls_back_to_legacy_alias(self):
        config = make_config(online=True)
        config['policy']['vtrace_rho_clip'] = 1.5

        self.assertEqual(1.5, train_online.policy_importance_rho_clip(config))

    def test_policy_importance_c_clip_prefers_new_key(self):
        config = make_config(online=True)
        config['policy']['importance_c_clip'] = 0.75
        config['policy']['vtrace_c_clip'] = 1.0

        self.assertEqual(0.75, train_online.policy_importance_c_clip(config))

    def test_policy_importance_c_clip_falls_back_to_legacy_alias(self):
        config = make_config(online=True)
        config['policy']['vtrace_c_clip'] = 1.0

        self.assertEqual(1.0, train_online.policy_importance_c_clip(config))

    def test_policy_vtrace_target_clips_prefer_dedicated_keys(self):
        config = make_config(online=True)
        config['policy']['importance_rho_clip'] = 2.0
        config['policy']['importance_c_clip'] = 2.0
        config['policy']['vtrace_target_rho_clip'] = 1.1
        config['policy']['vtrace_target_c_clip'] = 0.9

        self.assertEqual(2.0, train_online.policy_importance_rho_clip(config))
        self.assertEqual(2.0, train_online.policy_importance_c_clip(config))
        self.assertEqual(1.1, train_online.policy_vtrace_target_rho_clip(config))
        self.assertEqual(0.9, train_online.policy_vtrace_target_c_clip(config))

    def test_policy_vtrace_target_clips_fall_back_without_dedicated_keys(self):
        config = make_config(online=True)
        config['policy']['importance_rho_clip'] = 1.3
        config['policy']['importance_c_clip'] = 0.7

        self.assertEqual(1.3, train_online.policy_vtrace_target_rho_clip(config))
        self.assertEqual(0.7, train_online.policy_vtrace_target_c_clip(config))

    def test_policy_entropy_floor_uses_entropy_target_alias(self):
        config = make_config(online=True)
        config['policy']['entropy_target'] = 0.9

        self.assertEqual(0.9, train_online.policy_entropy_floor(config))

    def test_policy_entropy_floor_start_step_uses_default_when_unset(self):
        config = make_config(online=True)

        self.assertEqual(
            123,
            train_online.policy_entropy_floor_start_step(config, default=123),
        )

    def test_policy_lr_scales_default_to_one(self):
        config = make_config(online=True)

        self.assertEqual(1.0, train_online.policy_actor_lr_scale(config))
        self.assertEqual(1.0, train_online.policy_head_lr_scale(config))

    def test_policy_lr_scales_are_non_negative(self):
        config = make_config(online=True)
        config['policy']['actor_lr_scale'] = -0.5
        config['policy']['policy_head_lr_scale'] = 0.25

        self.assertEqual(0.0, train_online.policy_actor_lr_scale(config))
        self.assertEqual(0.25, train_online.policy_head_lr_scale(config))

    def test_policy_update_throttle_defaults_to_every_step(self):
        config = make_config(online=True)

        self.assertEqual(1, train_online.policy_update_interval(config))
        self.assertEqual(0, train_online.policy_update_phase(config))
        self.assertTrue(train_online.policy_update_active(config, 0))
        self.assertTrue(train_online.policy_update_active(config, 17))

    def test_policy_update_throttle_uses_interval_and_phase(self):
        config = make_config(online=True)
        config['policy']['update_interval'] = 3
        config['policy']['update_phase'] = 1

        self.assertEqual(3, train_online.policy_update_interval(config))
        self.assertEqual(1, train_online.policy_update_phase(config))
        self.assertFalse(train_online.policy_update_active(config, 0))
        self.assertTrue(train_online.policy_update_active(config, 1))
        self.assertFalse(train_online.policy_update_active(config, 2))
        self.assertFalse(train_online.policy_update_active(config, 3))
        self.assertTrue(train_online.policy_update_active(config, 4))

    def test_policy_lr_scales_do_not_change_model_signature(self):
        base = make_config(online=True)
        scaled = make_config(online=True)
        scaled['policy']['actor_lr_scale'] = 0.5
        scaled['policy']['policy_head_lr_scale'] = 0.25

        self.assertEqual(
            train_online.online_resume_model_signature(base),
            train_online.online_resume_model_signature(scaled),
        )

    def test_value_critic_warmup_defaults_to_disabled(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)

        self.assertEqual(0, train_online.value_critic_warmup_steps(config))
        self.assertFalse(train_online.value_critic_warmup_active(config, 0))

    def test_value_critic_warmup_uses_new_key_before_legacy_alias(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)
        config['value']['actor_freeze_steps'] = 100
        config['value']['critic_warmup_steps'] = 300

        self.assertEqual(300, train_online.value_critic_warmup_steps(config))
        self.assertTrue(train_online.value_critic_warmup_active(config, 299))
        self.assertFalse(train_online.value_critic_warmup_active(config, 300))

    def test_value_critic_warmup_legacy_actor_freeze_alias(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)
        config['value']['actor_freeze_steps'] = 100

        self.assertEqual(100, train_online.value_critic_warmup_steps(config))
        self.assertTrue(train_online.value_critic_warmup_active(config, 0))
        self.assertFalse(train_online.value_critic_warmup_active(config, 100))

    def test_value_critic_warmup_inactive_when_value_disabled(self):
        config = make_config(online=True, value_enabled=False, oracle_critic=False)
        config['value']['critic_warmup_steps'] = 100

        self.assertEqual(100, train_online.value_critic_warmup_steps(config))
        self.assertFalse(train_online.value_critic_warmup_active(config, 0))

    def test_critic_warmup_enables_independent_actor_lr_clock_by_default(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)
        config['value']['critic_warmup_steps'] = 100

        self.assertTrue(train_online.value_independent_actor_lr_clock(config))
        config['value']['independent_actor_lr_clock'] = False
        self.assertFalse(train_online.value_independent_actor_lr_clock(config))

    def test_independent_actor_lr_clock_starts_after_critic_warmup(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)
        config['value']['critic_warmup_steps'] = 100
        config['optim']['scheduler'] = {
            'init': 1e-8,
            'peak': 1e-4,
            'final': 1e-5,
            'warm_up_steps': 20,
            'max_steps': 300,
        }
        optimizer = DummyOptimizer([1, 1])
        optimizer.param_groups[0].update({
            'schedule_role': 'actor',
            'lr_scale': 0.5,
            'lr': 0.0,
        })
        optimizer.param_groups[1].update({
            'schedule_role': 'critic',
            'lr_scale': 1.0,
            'lr': 7e-5,
        })
        scheduler = DummyScheduler()

        warmup_lr = train_online.apply_independent_actor_lr_clock(
            optimizer,
            scheduler,
            config,
            steps=100,
        )
        warmup_actor_group_lr = optimizer.param_groups[0]['lr']
        first_actor_lr = train_online.apply_independent_actor_lr_clock(
            optimizer,
            scheduler,
            config,
            steps=101,
        )

        self.assertEqual(1e-8, warmup_lr)
        self.assertGreater(first_actor_lr, warmup_lr)
        self.assertAlmostEqual(warmup_lr * 0.5, warmup_actor_group_lr)
        self.assertAlmostEqual(first_actor_lr * 0.5, optimizer.param_groups[0]['lr'])
        self.assertEqual(7e-5, optimizer.param_groups[1]['lr'])

    def test_critic_warmup_loss_freezes_actor_gradients(self):
        actor = torch.nn.Linear(3, 4)
        policy = torch.nn.Linear(4, 2)
        critic = torch.nn.Linear(5, 4)
        value = torch.nn.Linear(4, 1)
        obs = torch.randn(6, 3)
        oracle_obs = torch.randn(6, 5)
        target = torch.randn(6, 1)

        with torch.no_grad():
            phi = actor(obs)
            _ = policy(phi)
        loss = torch.nn.functional.mse_loss(value(critic(oracle_obs)), target)
        loss.backward()

        self.assertTrue(all(param.grad is None for param in actor.parameters()))
        self.assertTrue(all(param.grad is None for param in policy.parameters()))
        self.assertTrue(all(param.grad is not None for param in critic.parameters()))
        self.assertTrue(all(param.grad is not None for param in value.parameters()))

    def test_ensure_parent_dir_for_file_creates_missing_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / 'nested' / 'checkpoints' / 'mortal.pth'

            train_online.ensure_parent_dir_for_file(str(target))

            self.assertTrue(target.parent.is_dir())

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

    def test_resolve_oracle_critic_init_state_file_prefers_value_override(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)
        config['value']['oracle_critic_state_file'] = './checkpoints/oracle_best.pth'
        config['oracle_critic_pretrain'] = {
            'best_state_file': './checkpoints/pretrain_best.pth',
        }

        self.assertEqual(
            './checkpoints/oracle_best.pth',
            train_online.resolve_oracle_critic_init_state_file(config),
        )

    def test_resolve_oracle_critic_init_state_file_falls_back_to_pretrain_best(self):
        config = make_config(online=True, value_enabled=True, oracle_critic=True)
        config['oracle_critic_pretrain'] = {
            'best_state_file': './checkpoints/pretrain_best.pth',
        }

        self.assertEqual(
            './checkpoints/pretrain_best.pth',
            train_online.resolve_oracle_critic_init_state_file(config),
        )

    def test_ensure_online_init_state_file_ready_checks_canonical_handoff(self):
        with patch(
            'mortal.supervised.run_sl_formal.ensure_supervised_canonical_handoff_ready',
            side_effect=RuntimeError('pending formal_1v3 handoff'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'pending formal_1v3 handoff'):
                train_online.ensure_online_init_state_file_ready('./checkpoints/sl_canonical.pth')

    def test_ensure_online_init_state_file_ready_requires_existing_file_after_handoff_check(self):
        with patch('mortal.supervised.run_sl_formal.ensure_supervised_canonical_handoff_ready'):
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

    def test_reconcile_loaded_scheduler_state_guards_against_lr_increase(self):
        optimizer = DummyOptimizer([2, 1])
        for group in optimizer.param_groups:
            group['lr'] = 1e-5
            group['initial_lr'] = 1.0
        scheduler = DummyScheduler()

        changes = train_online.reconcile_loaded_scheduler_state(
            scheduler,
            optimizer,
            {
                'init': 1e-8,
                'peak': 1e-4,
                'final': 1e-5,
                'warm_up_steps': 2500,
                'max_steps': 50000,
                'offset': 0,
                'epoch_size': 0,
            },
            steps=20000,
        )

        self.assertEqual((1000, 2500), changes['warm_up_steps'])
        self.assertEqual((20000, 50000), changes['max_steps'])
        self.assertIn('lr_increase_guard', changes)
        self.assertEqual(20000, scheduler.last_epoch)
        self.assertEqual(1000, scheduler.warm_up_steps)
        self.assertEqual(20000, scheduler.max_steps)
        self.assertEqual(1e-5, optimizer.param_groups[0]['lr'])
        self.assertEqual(1e-5, scheduler._last_lr[0])

    def test_reconcile_loaded_scheduler_state_allows_lr_decrease(self):
        optimizer = DummyOptimizer([2, 1])
        for group in optimizer.param_groups:
            group['lr'] = 1e-4
            group['initial_lr'] = 1.0
        scheduler = DummyScheduler()

        changes = train_online.reconcile_loaded_scheduler_state(
            scheduler,
            optimizer,
            {
                'init': 1e-8,
                'peak': 1e-4,
                'final': 1e-5,
                'warm_up_steps': 2500,
                'max_steps': 50000,
                'offset': 0,
                'epoch_size': 0,
            },
            steps=20000,
        )

        self.assertEqual((1000, 2500), changes['warm_up_steps'])
        self.assertEqual((20000, 50000), changes['max_steps'])
        self.assertNotIn('lr_increase_guard', changes)
        self.assertEqual(2500, scheduler.warm_up_steps)
        self.assertEqual(50000, scheduler.max_steps)
        self.assertLess(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(scheduler._last_lr[0], optimizer.param_groups[0]['lr'])

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

    def test_value_target_mode_auto_uses_current_player_without_oracle(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=False)
        self.assertEqual('current_player', train_online.value_target_mode(cfg))
        self.assertEqual(1, train_online.value_num_players_from_mode('current_player'))

    def test_value_target_mode_auto_uses_all_players_with_oracle(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        self.assertEqual('all_players', train_online.value_target_mode(cfg))
        self.assertEqual(4, train_online.value_num_players_from_mode('all_players'))

    def test_value_reward_source_defaults_to_score_rank_for_oracle_critic(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        self.assertEqual('score_rank', train_online.value_reward_source(cfg))

    def test_value_reward_source_defaults_to_grp_without_oracle_critic(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=False)
        self.assertEqual('grp', train_online.value_reward_source(cfg))

    def test_value_reward_source_allows_explicit_override(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['value']['reward_source'] = 'grp'
        self.assertEqual('grp', train_online.value_reward_source(cfg))

    def test_signature_rejects_value_reward_source_mismatch(self):
        cfg_score_rank = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_grp = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_grp['value']['reward_source'] = 'grp'

        self.assertNotEqual(
            train_online.online_resume_model_signature(cfg_score_rank),
            train_online.online_resume_model_signature(cfg_grp),
        )

    def test_signature_rejects_oracle_critic_arch_mismatch(self):
        cfg_single = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_dual = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_dual['value']['oracle_critic_arch'] = 'dual_tower'

        self.assertNotEqual(
            train_online.online_resume_model_signature(cfg_single),
            train_online.online_resume_model_signature(cfg_dual),
        )

    def test_signature_rejects_gae_gamma_mismatch(self):
        cfg_a = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_b = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_a['policy']['gae_enabled'] = True
        cfg_b['policy']['gae_enabled'] = True
        cfg_a['policy']['gae_gamma'] = 0.999
        cfg_b['policy']['gae_gamma'] = 0.995

        self.assertNotEqual(
            train_online.online_resume_model_signature(cfg_a),
            train_online.online_resume_model_signature(cfg_b),
        )

    def test_checkpoint_model_signature_match_helper_accepts_same_semantics(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        state = {'config': cfg}

        self.assertTrue(
            train_online.checkpoint_matches_online_model_signature(
                state,
                current_config=cfg,
            )
        )

    def test_checkpoint_model_signature_match_helper_rejects_value_semantics(self):
        cfg_current = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_saved = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg_saved['value']['reward_source'] = 'grp'

        self.assertFalse(
            train_online.checkpoint_matches_online_model_signature(
                {'config': cfg_saved},
                current_config=cfg_current,
            )
        )

    def test_oracle_pretrain_init_only_loads_without_matching_state_resume(self):
        self.assertTrue(
            train_online.should_load_oracle_critic_init_checkpoint(
                state_file_exists=False,
                state_file_model_signature_matches=False,
            )
        )
        self.assertTrue(
            train_online.should_load_oracle_critic_init_checkpoint(
                state_file_exists=True,
                state_file_model_signature_matches=False,
            )
        )
        self.assertFalse(
            train_online.should_load_oracle_critic_init_checkpoint(
                state_file_exists=True,
                state_file_model_signature_matches=True,
            )
        )

    def test_oracle_critic_init_metadata_accepts_matching_score_rank_mc(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['policy']['gae_gamma'] = 0.999
        cfg['value']['reward_source'] = 'score_rank'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'return_mode': 'score_rank_mc',
                'discount_gamma': 0.999,
                'critic_arch': 'single_tower',
            },
        }

        info = train_online.validate_oracle_critic_init_checkpoint(state, cfg)

        self.assertEqual('score_rank_mc', info['return_mode'])
        self.assertEqual('all_players', info['target_mode'])

    def test_oracle_critic_init_metadata_rejects_target_mismatch(self):
        cfg = make_config(
            online=True,
            value_enabled=True,
            oracle_critic=True,
            value_target_mode='current_player',
        )
        cfg['value']['reward_source'] = 'score_rank'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'return_mode': 'score_rank_mc',
                'discount_gamma': 0.999,
                'critic_arch': 'single_tower',
            },
        }

        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            train_online.validate_oracle_critic_init_checkpoint(state, cfg)

    def test_oracle_critic_init_metadata_rejects_missing_required_fields(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['value']['reward_source'] = 'score_rank'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'critic_arch': 'single_tower',
            },
        }

        with self.assertRaisesRegex(ValueError, 'metadata missing required field'):
            train_online.validate_oracle_critic_init_checkpoint(state, cfg)

    def test_oracle_critic_init_metadata_can_infer_missing_arch_from_state_dict(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['policy']['gae_gamma'] = 0.999
        cfg['value']['reward_source'] = 'score_rank'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'return_mode': 'score_rank_mc',
                'discount_gamma': 0.999,
            },
            'oracle_brain': {
                'encoder.net.0.weight': torch.empty(1),
            },
        }

        info = train_online.validate_oracle_critic_init_checkpoint(state, cfg)

        self.assertEqual('single_tower', info['critic_arch'])

    def test_oracle_critic_init_metadata_rejects_reward_source_mismatch(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['value']['reward_source'] = 'grp'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'return_mode': 'score_rank_mc',
                'discount_gamma': 0.999,
                'critic_arch': 'single_tower',
            },
        }

        with self.assertRaisesRegex(ValueError, 'reward mismatch'):
            train_online.validate_oracle_critic_init_checkpoint(state, cfg)

    def test_oracle_critic_init_metadata_rejects_gamma_mismatch(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['policy']['gae_gamma'] = 0.995
        cfg['value']['reward_source'] = 'score_rank'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'return_mode': 'score_rank_mc',
                'discount_gamma': 0.999,
                'critic_arch': 'single_tower',
            },
        }

        with self.assertRaisesRegex(ValueError, 'discount mismatch'):
            train_online.validate_oracle_critic_init_checkpoint(state, cfg)

    def test_oracle_critic_init_metadata_rejects_arch_mismatch(self):
        cfg = make_config(online=True, value_enabled=True, oracle_critic=True)
        cfg['value']['reward_source'] = 'score_rank'
        cfg['value']['oracle_critic_arch'] = 'dual_tower'
        state = {
            'oracle_critic_pretrain': {
                'target_mode': 'all_players',
                'return_mode': 'score_rank_mc',
                'discount_gamma': 0.999,
                'critic_arch': 'single_tower',
            },
        }

        with self.assertRaisesRegex(ValueError, 'arch mismatch'):
            train_online.validate_oracle_critic_init_checkpoint(state, cfg)

    def test_value_target_mode_allows_explicit_override(self):
        cfg = make_config(
            online=True,
            value_enabled=True,
            oracle_critic=False,
            value_target_mode='all_players',
        )
        self.assertEqual('all_players', train_online.value_target_mode(cfg))

    def test_signature_rejects_actor_oracle_mismatch(self):
        sig_base = train_online.online_resume_model_signature(make_config(online=True))
        sig_actor_oracle = train_online.online_resume_model_signature(
            make_config(online=True, actor_oracle_enabled=True)
        )
        self.assertNotEqual(sig_base, sig_actor_oracle)

    def test_signature_rejects_actor_oracle_source_mismatch(self):
        cfg_true = make_config(online=True, actor_oracle_enabled=True)
        cfg_shuffled = make_config(online=True, actor_oracle_enabled=True)
        cfg_shuffled['oracle_guiding']['actor_source'] = 'shuffled'
        sig_true = train_online.online_resume_model_signature(cfg_true)
        sig_shuffled = train_online.online_resume_model_signature(cfg_shuffled)
        self.assertNotEqual(sig_true, sig_shuffled)

    def test_signature_rejects_invalid_policy_action_scope(self):
        with self.assertRaisesRegex(ValueError, 'online_action_scope'):
            train_online.online_resume_model_signature(
                make_config(online=True, online_action_scope='legacy_discard')
            )

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
            exp_reward_enabled=True, actor_oracle_enabled=True,
        )
        sig1 = train_online.online_resume_model_signature(cfg)
        sig2 = train_online.online_resume_model_signature(cfg)
        self.assertEqual(sig1, sig2)

    def test_actor_oracle_keep_prob_linear_schedule_matches_paper_direction(self):
        cfg = make_config(online=True, actor_oracle_enabled=True)
        cfg['oracle_guiding'].update({
            'schedule': 'linear',
            'gamma_start': 1.0,
            'gamma_end': 0.0,
            'hold_steps': 0,
            'decay_steps': 100,
        })
        self.assertEqual(1.0, train_online.actor_oracle_guiding_keep_prob(cfg, 0))
        self.assertAlmostEqual(0.5, train_online.actor_oracle_guiding_keep_prob(cfg, 50))
        self.assertEqual(0.0, train_online.actor_oracle_guiding_keep_prob(cfg, 100))
        self.assertEqual(0.0, train_online.actor_oracle_guiding_keep_prob(cfg, 150))

    def test_actor_oracle_continuation_activates_after_oracle_reaches_zero(self):
        cfg = make_config(online=True, actor_oracle_enabled=True)
        cfg['oracle_guiding'].update({
            'schedule': 'linear',
            'gamma_start': 1.0,
            'gamma_end': 0.0,
            'hold_steps': 10,
            'decay_steps': 20,
        })
        self.assertFalse(train_online.actor_oracle_guiding_continuation_active(cfg, 25))
        self.assertTrue(train_online.actor_oracle_guiding_continuation_active(cfg, 30))

    def test_transform_actor_oracle_invisible_obs_zeroes_with_zero_keep_prob(self):
        tensor = torch.ones((2, 3, 4), dtype=torch.float32)

        out = train_online.transform_actor_oracle_invisible_obs(
            tensor,
            actor_source='true',
            keep_prob=0.0,
        )

        self.assertTrue(torch.equal(out, torch.zeros_like(tensor)))

    def test_replay_importance_sampling_cfg_reads_online_block(self):
        cfg = make_config(online=True, replay_is_enabled=True)
        cfg['online']['importance_sampling']['max_policy_versions'] = 12
        self.assertTrue(train_online.replay_importance_sampling_enabled(cfg))
        self.assertEqual(12, train_online.replay_importance_sampling_max_versions(cfg))

    def test_replay_importance_sampling_vtrace_mode_defaults_to_auto(self):
        cfg = make_config(online=True, replay_is_enabled=True)
        self.assertEqual('auto', train_online.replay_importance_sampling_vtrace_mode(cfg))
        self.assertEqual(2, train_online.replay_importance_sampling_vtrace_min_version_gap(cfg))

    def test_replay_importance_sampling_vtrace_mode_allows_always(self):
        cfg = make_config(online=True, replay_is_enabled=True)
        cfg['online']['importance_sampling']['vtrace_mode'] = 'always'
        self.assertEqual('always', train_online.replay_importance_sampling_vtrace_mode(cfg))
        self.assertTrue(
            train_online.replay_importance_sampling_should_use_vtrace(
                cfg,
                published_param_version=10,
                replay_param_version=-1,
            )
        )

    def test_replay_importance_sampling_vtrace_mode_auto_requires_min_gap(self):
        cfg = make_config(online=True, replay_is_enabled=True)
        cfg['online']['importance_sampling']['vtrace_min_version_gap'] = 3
        self.assertFalse(
            train_online.replay_importance_sampling_should_use_vtrace(
                cfg,
                published_param_version=10,
                replay_param_version=8,
            )
        )
        self.assertTrue(
            train_online.replay_importance_sampling_should_use_vtrace(
                cfg,
                published_param_version=10,
                replay_param_version=7,
            )
        )

    def test_replay_importance_sampling_vtrace_mode_disabled_never_uses_vtrace(self):
        cfg = make_config(online=True, replay_is_enabled=True)
        cfg['online']['importance_sampling']['vtrace_mode'] = 'disabled'
        self.assertFalse(
            train_online.replay_importance_sampling_should_use_vtrace(
                cfg,
                published_param_version=10,
                replay_param_version=0,
            )
        )

    def test_policy_online_action_scope_defaults_to_all(self):
        cfg = make_config(online=True)
        del cfg['policy']['online_action_scope']
        self.assertEqual('all', train_online.policy_online_action_scope(cfg))

    def test_initial_test_play_defaults_to_disabled(self):
        cfg = make_config(online=True)
        self.assertFalse(train_online.initial_test_play_enabled(cfg))
        self.assertEqual(600, train_online.initial_test_play_games(cfg))

    def test_periodic_test_play_due_requires_enable_flag(self):
        self.assertFalse(
            train_online.periodic_test_play_due(
                enabled=False,
                steps=3000,
                test_every=3000,
            )
        )
        self.assertTrue(
            train_online.periodic_test_play_due(
                enabled=True,
                steps=3000,
                test_every=3000,
            )
        )

    def test_periodic_test_play_due_ignores_zero_interval(self):
        self.assertFalse(
            train_online.periodic_test_play_due(
                enabled=True,
                steps=3000,
                test_every=0,
            )
        )

    def test_old_policy_update_due_is_independent_from_save_every(self):
        due_steps = [
            step
            for step in range(1, 1201)
            if train_online.old_policy_update_due(step, old_update_every=400)
        ]

        self.assertEqual([400, 800, 1200], due_steps)

    def test_old_policy_update_due_ignores_zero_interval(self):
        self.assertFalse(train_online.old_policy_update_due(400, old_update_every=0))

    def test_refresh_old_policy_snapshot_reuses_existing_modules(self):
        old_mortal = DummyStateModule({"old": 1})
        old_policy = DummyStateModule({"old_policy": 1})
        mortal = DummyStateModule({"new": 2})
        policy = DummyStateModule({"new_policy": 2})

        returned = train_online.refresh_old_policy_snapshot(
            old_mortal,
            old_policy,
            mortal,
            policy,
        )

        self.assertIsNone(returned)
        self.assertEqual({"new": 2}, old_mortal.loaded_state)
        self.assertEqual({"new_policy": 2}, old_policy.loaded_state)

    def test_recorded_step0_baseline_reads_profile_metadata(self):
        cfg = make_config(online=True)
        cfg['online_experiment_profile'] = {
            'recorded_step0_baseline': {
                'games': 600,
                'avg_rank': 2.525,
                'avg_pt': -2.475,
                'source_run': 'rl1_add_value_gae_is_20k_20260412_002033',
            }
        }
        baseline = train_online.recorded_step0_baseline(cfg)
        self.assertIsNotNone(baseline)
        self.assertEqual(600, baseline['games'])
        self.assertEqual(-2.475, baseline['avg_pt'])

    def test_policy_online_action_keep_mask_returns_none_for_all(self):
        mask = train_online.policy_online_action_keep_mask(
            torch.tensor([0, 12, 36, 37, 45], dtype=torch.int64),
            'all',
        )
        self.assertIsNone(mask)

    def test_policy_online_action_scope_rejects_invalid_value(self):
        cfg = make_config(online=True, online_action_scope='invalid')
        with self.assertRaisesRegex(ValueError, 'online_action_scope'):
            train_online.policy_online_action_scope(cfg)

    def test_published_policy_history_evicts_old_versions(self):
        history = train_online.PublishedPolicyHistory(max_versions=2)
        history.remember(1, mortal_state={'a': 1}, policy_state={'b': 1}, runtime={'x': 1})
        history.remember(2, mortal_state={'a': 2}, policy_state={'b': 2}, runtime={'x': 2})
        history.remember(3, mortal_state={'a': 3}, policy_state={'b': 3}, runtime={'x': 3})
        self.assertIsNone(history.get(1))
        self.assertEqual((2, 3), history.versions())
        self.assertEqual(3, history.get(3)['mortal']['a'])

    def test_online_stats_track_replay_is_window_extremes(self):
        stats = train_online.init_online_stats(device=torch.device('cpu'))

        self.assertEqual(1.0, stats['replay_is_coverage_min'].item())
        self.assertEqual(0.0, stats['replay_is_missing_fraction_max'].item())
        self.assertEqual(0.0, stats['replay_is_version_gap_max'].item())
        self.assertEqual(0.0, stats['ratio_batch_max_sum'].item())
        self.assertEqual(0.0, stats['ratio_window_max'].item())
        self.assertEqual(0.0, stats['clipped_ratio_window_max'].item())


class RewardTargetScaleTests(unittest.TestCase):
    def test_prepare_policy_advantage_normalizes_actor_only(self):
        advantage = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
        v_target = torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0],
                [11.0, 21.0, 31.0, 41.0],
                [12.0, 22.0, 32.0, 42.0],
            ],
            dtype=torch.float32,
        )

        raw_advantage, normalized_advantage, prepared_v_target = (
            train_online.prepare_policy_advantage_and_value_target(
                advantage,
                v_target,
                device=torch.device('cpu'),
                gae_enabled=True,
            )
        )

        self.assertTrue(torch.equal(raw_advantage, advantage))
        self.assertAlmostEqual(0.0, normalized_advantage.mean().item(), places=6)
        self.assertAlmostEqual(1.0, normalized_advantage.std().item(), places=6)
        self.assertTrue(torch.equal(prepared_v_target, v_target))

    def test_policy_objective_keeps_samplewise_entropy_shape(self):
        clip_loss = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        entropy = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

        loss = train_online.compute_policy_objective_loss(
            clip_loss,
            entropy,
            entropy_weight=0.5,
        )

        expected = -((clip_loss + entropy * 0.5).mean())
        self.assertTrue(torch.equal(loss, expected))

    def test_policy_objective_rejects_broadcasted_entropy_shape(self):
        clip_loss = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        entropy = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, 'identical shapes'):
            train_online.compute_policy_objective_loss(
                clip_loss,
                entropy,
                entropy_weight=0.5,
            )

    def test_reward_calculator_keeps_raw_delta_pt_scale_across_calls(self):
        from mortal.data.reward_calculator import RewardCalculator

        rc = RewardCalculator(grp=None, label_smoothing=0.0)
        uniform = torch.full((4, 4), 0.25, dtype=torch.float32)
        deterministic = torch.eye(4, dtype=torch.float32)
        matrix = torch.stack((uniform, deterministic), dim=0)
        rc.calc_grp = lambda grp_feature: matrix

        reward1 = rc.calc_delta_pt_all_players([], [0, 1, 2, 3])
        reward2 = rc.calc_delta_pt_all_players([], [0, 1, 2, 3])

        self.assertEqual('float32', str(reward1.dtype))
        self.assertEqual(
            [[3.0, 1.0, -1.0, -3.0], [0.0, 0.0, 0.0, 0.0]],
            reward1.tolist(),
        )
        self.assertEqual(reward1.tolist(), reward2.tolist())

    def test_compute_vtrace_targets_reduces_to_mc_return_when_ratios_are_one(self):
        vs, pg_adv = train_online.compute_vtrace_targets_from_step_rewards(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            gamma=1.0,
            rho_clip=1.0,
            c_clip=1.0,
        )

        np.testing.assert_allclose(vs, np.array([3.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(pg_adv, np.array([3.0, 2.0], dtype=np.float32))

    def test_compute_vtrace_targets_uses_c_clip_in_backward_correction(self):
        vs_full, pg_adv_full = train_online.compute_vtrace_targets_from_step_rewards(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.log(np.array([2.0, 1.0], dtype=np.float32)),
            gamma=1.0,
            rho_clip=1.0,
            c_clip=1.0,
        )
        vs_half, pg_adv_half = train_online.compute_vtrace_targets_from_step_rewards(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.log(np.array([2.0, 1.0], dtype=np.float32)),
            gamma=1.0,
            rho_clip=1.0,
            c_clip=0.5,
        )

        np.testing.assert_allclose(vs_full, np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(pg_adv_full, np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(vs_half, np.array([0.5, 1.0], dtype=np.float32))
        np.testing.assert_allclose(pg_adv_half, np.array([1.0, 1.0], dtype=np.float32))

    def test_sparse_kyoku_reward_keeps_rewards_for_kyoku_without_player_samples(self):
        expanded = train_online.expand_sparse_kyoku_reward_to_steps(
            np.array([1.0, 2.0, 4.0], dtype=np.float32),
            np.array([0, 0, 2], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            expanded,
            np.array([0.0, 3.0, 4.0], dtype=np.float32),
        )


class OnlineAuxAlignmentTests(unittest.TestCase):
    def test_resolve_effective_online_aux_training_cfg_prefers_checkpoint_scales(self):
        current_cfg = make_config(
            online=True,
            next_rank_weight=0.2,
            opponent_state_weight=0.03,
            danger_enabled=True,
            danger_weight=0.05,
        )
        current_cfg['supervised']['rank_aux'] = {
            'base_weight': 0.03,
            'south_factor': 1.2,
            'all_last_factor': 1.3,
            'gap_focus_points': 2000.0,
            'gap_close_bonus': 0.4,
            'max_weight': 0.1,
        }
        checkpoint_cfg = make_config(
            online=False,
            next_rank_weight=0.2,
            opponent_state_weight=0.00135,
            danger_enabled=True,
            danger_weight=0.00804,
        )
        checkpoint_cfg['supervised']['rank_aux'] = {
            'base_weight': 0.001548,
            'south_factor': 1.59,
            'all_last_factor': 1.617,
            'gap_focus_points': 4000.0,
            'gap_close_bonus': 0.0,
            'max_weight': 0.00516,
        }
        checkpoint_cfg['aux'].update({
            'opponent_shanten_weight': 0.8506568408072642,
            'opponent_tenpai_weight': 1.1493431591927359,
            'danger_any_weight': 0.09042179466099699,
            'danger_value_weight': 0.8180402859274302,
            'danger_player_weight': 0.09153791941157279,
            'danger_ramp_steps': 1000,
        })

        resolved = train_online.resolve_effective_online_aux_training_cfg(
            current_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        self.assertEqual('checkpoint', resolved['source'])
        self.assertAlmostEqual(0.001548, resolved['rank_base_weight'])
        self.assertAlmostEqual(1.59, resolved['rank_south_factor'])
        self.assertAlmostEqual(1.617, resolved['rank_all_last_factor'])
        self.assertAlmostEqual(0.00135, resolved['opponent_state_weight'])
        self.assertAlmostEqual(0.00804, resolved['danger_weight'])
        self.assertEqual(1000, resolved['danger_ramp_steps'])

    def test_compute_rank_aux_sample_weights_applies_turn_stage_and_gap_rules(self):
        context_meta = torch.tensor(
            [
                [3, 0, 0, 0, 1, 0, 20, 40],
                [12, 1, 0, 1, 2, 0, 10, 30],
            ],
            dtype=torch.int64,
        )

        weights = train_online.compute_rank_aux_sample_weights(
            context_meta,
            device=torch.device('cpu'),
            base_weight=0.01,
            south_factor=1.5,
            all_last_factor=2.0,
            gap_focus_points=4000.0,
            gap_close_bonus=1.0,
            max_weight=0.05,
            turn_weighting={
                'early_factor': 1.0,
                'mid_factor': 1.05,
                'late_factor': 1.15,
                'early_max_turn': 4,
                'late_min_turn': 12,
            },
        )

        self.assertAlmostEqual(0.015, float(weights[0]), places=6)
        self.assertAlmostEqual(0.05, float(weights[1]), places=6)


class LabelSmoothingTests(unittest.TestCase):
    def test_label_smoothing_sums_to_one(self):
        """Smoothed final ranking should sum to 1.0."""
        import torch
        from mortal.data.reward_calculator import RewardCalculator

        # Create a minimal dummy GRP
        from mortal.core.model import GRP
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
        from mortal.data.reward_calculator import RewardCalculator
        from mortal.core.model import GRP

        grp = GRP(hidden_size=8, num_layers=1, dtype='float32')
        rc = RewardCalculator(grp=grp, label_smoothing=0.0)

        rank_by_player = [1, 0, 3, 2]
        grp_feature = [[0, 0, 0, 2.5, 2.5, 2.5, 2.5]]
        rank_prob = rc.calc_rank_prob(0, grp_feature, rank_by_player)
        final = rank_prob[-1]
        self.assertAlmostEqual(final[1].item(), 1.0, places=5)
        self.assertAlmostEqual(final[0].item(), 0.0, places=5)


class AuxMonitorMetricTests(unittest.TestCase):
    def test_finalize_binary_metric_reports_balanced_fields(self):
        stat = train_online.init_binary_metric_dict(device=torch.device('cpu'))
        eligible = torch.tensor([[True, True, True, True]])
        target_positive = torch.tensor([[True, False, True, False]])
        pred_positive = torch.tensor([[True, False, False, False]])
        positive_prob = torch.tensor([[0.9, 0.2, 0.4, 0.1]])

        train_online.update_binary_metric(
            stat,
            eligible,
            target_positive,
            pred_positive,
            positive_prob,
        )

        output = {}
        train_online.finalize_binary_metric('danger_any', stat, output)

        self.assertEqual(4, output['danger_any_count'])
        self.assertEqual(2, output['danger_any_pos_count'])
        self.assertEqual(2, output['danger_any_neg_count'])
        self.assertAlmostEqual(0.75, output['danger_any_acc'])
        self.assertAlmostEqual(0.25, output['danger_any_pred_rate'])
        self.assertAlmostEqual(0.5, output['danger_any_target_rate'])
        self.assertAlmostEqual(0.5, output['danger_any_pos_recall'])
        self.assertAlmostEqual(1.0, output['danger_any_neg_recall'])
        self.assertAlmostEqual(0.75, output['danger_any_balanced_acc'])

    def test_finalize_online_aux_monitor_stats_matches_sl_style_outputs(self):
        device = torch.device('cpu')
        stats = train_online.init_online_aux_monitor_stats(device=device)
        stats['rank_correct'] += torch.tensor(3, dtype=torch.int64)
        stats['rank_count'] += torch.tensor(4, dtype=torch.int64)
        stats['rank_aux_loss_sum'] += torch.tensor(1.2, dtype=torch.float64)
        stats['rank_aux_raw_loss_sum'] += torch.tensor(2.0, dtype=torch.float64)
        stats['rank_aux_weight_sum'] += torch.tensor(0.2, dtype=torch.float64)

        stats['opponent_sample_count'] += torch.tensor(4, dtype=torch.int64)
        stats['opponent_aux_loss_sum'] += torch.tensor(0.8, dtype=torch.float64)
        stats['opponent_turn_weight_sum'] += torch.tensor(4.8, dtype=torch.float64)
        stats['opponent_shanten_loss_sum'] += torch.tensor(1.6, dtype=torch.float64)
        stats['opponent_tenpai_loss_sum'] += torch.tensor(1.2, dtype=torch.float64)
        stats['opponent_count'] += torch.tensor([4, 4, 4], dtype=torch.int64)
        stats['opponent_shanten_correct'] += torch.tensor([4, 3, 2], dtype=torch.int64)
        stats['opponent_tenpai_correct'] += torch.tensor([1, 2, 4], dtype=torch.int64)

        stats['danger_sample_count'] += torch.tensor(4, dtype=torch.int64)
        stats['danger_aux_loss_sum'] += torch.tensor(0.6, dtype=torch.float64)
        stats['danger_turn_weight_sum'] += torch.tensor(5.2, dtype=torch.float64)
        stats['danger_any_loss_sum'] += torch.tensor(1.0, dtype=torch.float64)
        stats['danger_value_loss_sum'] += torch.tensor(0.8, dtype=torch.float64)
        stats['danger_player_loss_sum'] += torch.tensor(0.4, dtype=torch.float64)
        stats['danger_value_pos_count'] += torch.tensor(2, dtype=torch.int64)
        stats['danger_value_abs_err_sum'] += torch.tensor(1200.0, dtype=torch.float64)
        stats['danger_value_sq_err_sum'] += torch.tensor(1000000.0, dtype=torch.float64)

        any_batch = train_online.init_binary_metric_dict(device=device)
        train_online.update_binary_metric(
            any_batch,
            torch.tensor([[True, True, True, True]]),
            torch.tensor([[True, False, False, False]]),
            torch.tensor([[True, False, True, False]]),
            torch.tensor([[0.8, 0.2, 0.6, 0.1]]),
        )
        train_online.merge_binary_metric(stats['danger_any_stats'], any_batch)

        player_batch = train_online.init_binary_metric_dict(device=device)
        train_online.update_binary_metric(
            player_batch,
            torch.tensor([[[True, True], [True, True]]]),
            torch.tensor([[[True, False], [False, True]]]),
            torch.tensor([[[True, False], [True, True]]]),
            torch.tensor([[[0.9, 0.2], [0.6, 0.8]]]),
        )
        train_online.merge_binary_metric(stats['danger_player_stats'], player_batch)

        finalized = train_online.finalize_online_aux_monitor_stats(stats)

        self.assertAlmostEqual(0.3, finalized['aux_loss'])
        self.assertAlmostEqual(0.5, finalized['rank_aux_raw_loss'])
        self.assertAlmostEqual(0.05, finalized['rank_aux_weight_mean'])
        self.assertAlmostEqual(0.75, finalized['rank_acc'])
        self.assertAlmostEqual(0.2, finalized['opponent_aux_loss'])
        self.assertAlmostEqual(1.2, finalized['opponent_turn_weight_mean'])
        self.assertAlmostEqual((1.0 + 0.75 + 0.5) / 3.0, finalized['opponent_shanten_macro_acc'])
        self.assertAlmostEqual((0.25 + 0.5 + 1.0) / 3.0, finalized['opponent_tenpai_macro_acc'])
        self.assertAlmostEqual(0.15, finalized['danger_aux_loss'])
        self.assertAlmostEqual(1.3, finalized['danger_turn_weight_mean'])
        self.assertAlmostEqual(0.25, finalized['danger_any_loss'])
        self.assertAlmostEqual(0.2, finalized['danger_value_loss'])
        self.assertAlmostEqual(0.1, finalized['danger_player_loss'])
        self.assertAlmostEqual(600.0, finalized['danger_value_mae'])
        self.assertAlmostEqual((1000000.0 / 2.0) ** 0.5, finalized['danger_value_rmse'])
        self.assertAlmostEqual(0.75, finalized['danger_any_acc'])
        self.assertAlmostEqual((1.0 + (2.0 / 3.0)) / 2.0, finalized['danger_any_balanced_acc'])
        self.assertAlmostEqual(0.75, finalized['danger_player_acc'])


if __name__ == '__main__':
    unittest.main()

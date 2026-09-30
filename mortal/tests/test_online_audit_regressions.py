import copy
import unittest

import torch

from mortal.online.train_online import (
    PublishedPolicyHistory,
    checkpoint_matches_online_model_signature,
    prepare_policy_advantage_and_value_target,
    tracked_replay_versions_mask,
    validate_oracle_critic_init_checkpoint,
)


def oracle_config(points):
    return {
        'env': {'pts': points},
        'value': {'enabled': True, 'oracle_critic': True, 'target_mode': 'all_players', 'reward_source': 'score_rank'},
        'policy': {'gae_gamma': 0.999},
    }


def oracle_state(points):
    return {
        'config': oracle_config(points),
        'oracle_critic_pretrain': {
            'target_mode': 'all_players', 'return_mode': 'score_rank_mc',
            'discount_gamma': 0.999, 'critic_arch': 'single_tower',
        },
    }


class OnlineAuditRegressions(unittest.TestCase):
    def test_rejects_same_shape_critic_with_different_reward(self):
        with self.assertRaisesRegex(ValueError, 'rank reward mismatch'):
            validate_oracle_critic_init_checkpoint(oracle_state([90, 45, 0, -135]), oracle_config([6, 4, 2, 0]))

    def test_reward_offset_preserves_delta_targets_but_scale_does_not(self):
        config = oracle_config([6, 4, 2, 0])
        validate_oracle_critic_init_checkpoint(oracle_state([3, 1, -1, -3]), config)
        with self.assertRaisesRegex(ValueError, 'rank reward mismatch'):
            validate_oracle_critic_init_checkpoint(oracle_state([12, 8, 4, 0]), config)

    def test_real_config_requires_checkpoint_reward_provenance(self):
        state = oracle_state([6, 4, 2, 0])
        del state['config']
        with self.assertRaisesRegex(ValueError, 'rank reward mismatch'):
            validate_oracle_critic_init_checkpoint(state, oracle_config([6, 4, 2, 0]))

    def test_reward_change_cannot_be_treated_as_exact_online_resume(self):
        config = oracle_config([6, 4, 2, 0])
        saved = copy.deepcopy(config)
        saved['env']['pts'] = [90, 45, 0, -135]
        self.assertFalse(checkpoint_matches_online_model_signature({'config': saved}, current_config=config))
        self.assertTrue(checkpoint_matches_online_model_signature({'config': config}, current_config=config))

    def test_single_survivor_has_finite_advantage_and_unchanged_value_labels(self):
        target = torch.tensor([[3., 1., -1., -3.]])
        raw, normalized, value = prepare_policy_advantage_and_value_target(torch.tensor([2.]), target, device='cpu', gae_enabled=True)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(normalized.item(), 0.)
        self.assertEqual(raw.item(), 2.)
        torch.testing.assert_close(value, target)

    def test_evicted_and_unversioned_replay_are_not_tracked(self):
        history = PublishedPolicyHistory(1)
        for version in (4, 5):
            history.remember(version, mortal_state={}, policy_state={}, runtime={})
        mask = tracked_replay_versions_mask(torch.tensor([-1, 4, 5, 5]), history)
        self.assertEqual(mask.tolist(), [False, False, True, True])
        self.assertFalse(tracked_replay_versions_mask(torch.tensor([4, -1]), history).any())


if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from mortal.eval.search_runtime import BeliefSampler, LocalSearchPlanner, SearchConfig


class BeliefSamplerTests(unittest.TestCase):
    def test_sample_without_aux_heads_returns_zero_risk(self):
        cfg = SearchConfig(enabled=True, belief_samples=4)
        sampler = BeliefSampler(cfg)
        masks = torch.zeros((2, 46), dtype=torch.bool)
        masks[:, :3] = True

        belief = sampler.sample(masks=masks)

        self.assertEqual((4, 2, 37), tuple(belief.discard_risk_samples.shape))
        self.assertTrue(torch.allclose(belief.discard_risk_mean, torch.zeros_like(belief.discard_risk_mean)))
        self.assertTrue(torch.allclose(belief.global_pressure, torch.zeros_like(belief.global_pressure)))


class LocalSearchPlannerTests(unittest.TestCase):
    def test_high_danger_state_reorders_to_safe_discard(self):
        cfg = SearchConfig(
            enabled=True,
            belief_samples=4,
            min_policy_entropy=0.0,
            min_margin=1.0,
            min_danger_prob=0.0,
            planner_blend=1.0,
            risk_weight=2.0,
            variance_weight=0.0,
            score_temperature=1.0,
            top_k=2,
        )
        planner = LocalSearchPlanner(cfg)
        policy_logits = torch.full((1, 46), float("-inf"))
        policy_logits[0, 0] = 2.0
        policy_logits[0, 1] = 1.9
        masks = torch.zeros((1, 46), dtype=torch.bool)
        masks[0, 0] = True
        masks[0, 1] = True
        policy_probs = torch.softmax(policy_logits, dim=-1)

        danger_any_logits = torch.full((1, 37), -8.0)
        danger_any_logits[0, 0] = 8.0
        danger_any_logits[0, 1] = -8.0
        danger_value_pred = torch.zeros((1, 37))
        danger_value_pred[0, 0] = 6.0
        danger_player_logits = torch.full((1, 37, 3), -8.0)
        danger_player_logits[0, 0, :] = 8.0

        plan = planner.plan(
            policy_logits=policy_logits,
            policy_probs=policy_probs,
            masks=masks,
            danger_outputs=(danger_any_logits, danger_value_pred, danger_player_logits),
        )

        self.assertTrue(bool(plan.active_mask.item()))
        self.assertGreater(float(plan.final_probs[0, 1]), float(plan.final_probs[0, 0]))
        self.assertEqual(int(plan.final_probs.argmax(-1).item()), 1)

    def test_low_gap_teacher_marks_hard_state(self):
        cfg = SearchConfig(
            enabled=True,
            belief_samples=2,
            min_policy_entropy=0.0,
            min_margin=1.0,
            hard_entropy_min=0.0,
            hard_margin_max=1.0,
            planner_blend=1.0,
            top_k=2,
        )
        planner = LocalSearchPlanner(cfg)
        policy_logits = torch.full((1, 46), float("-inf"))
        policy_logits[0, 0] = 0.0
        policy_logits[0, 1] = 0.0
        masks = torch.zeros((1, 46), dtype=torch.bool)
        masks[0, 0] = True
        masks[0, 1] = True
        policy_probs = torch.softmax(policy_logits, dim=-1)

        plan = planner.plan(
            policy_logits=policy_logits,
            policy_probs=policy_probs,
            masks=masks,
        )

        self.assertTrue(bool(plan.active_mask.item()))
        self.assertTrue(bool(plan.hard_mask.item()))


if __name__ == '__main__':
    unittest.main()

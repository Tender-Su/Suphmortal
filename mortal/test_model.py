import unittest
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ACTION_SPACE, CategoricalPolicy, DANGER_DISCARD_DIM, DANGER_PLAYER_DIM, DangerAuxNet


class DangerAuxNetTests(unittest.TestCase):
    def test_forward_matches_legacy_head_by_head_projection(self):
        torch.manual_seed(1234)
        net = DangerAuxNet()
        x = torch.randn(7, 1024)

        any_logits, value, player_logits = net(x)
        fused_weight = net.net.weight

        expected_any = F.linear(x, fused_weight[:DANGER_DISCARD_DIM])
        expected_value = F.linear(x, fused_weight[DANGER_DISCARD_DIM:DANGER_DISCARD_DIM * 2])
        expected_player = F.linear(x, fused_weight[DANGER_DISCARD_DIM * 2:]).view(
            -1,
            DANGER_DISCARD_DIM,
            DANGER_PLAYER_DIM,
        )

        self.assertTrue(torch.allclose(any_logits, expected_any, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(value, expected_value, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(player_logits, expected_player, atol=1e-6, rtol=1e-6))

    def test_legacy_state_dict_round_trip_remains_supported(self):
        torch.manual_seed(4321)
        net = DangerAuxNet()
        legacy_state = net.state_dict()

        restored = DangerAuxNet()
        restored.load_state_dict(legacy_state)

        self.assertTrue(torch.equal(restored.net.weight, net.net.weight))


class CategoricalPolicyTests(unittest.TestCase):
    def test_forward_matches_softmax_of_logits(self):
        torch.manual_seed(2468)
        net = CategoricalPolicy()
        phi = torch.randn(6, 1024)
        mask = torch.rand(6, ACTION_SPACE) > 0.35
        mask[:, 0] = True

        probs = net(phi, mask)
        logits = net.logits(phi, mask)

        self.assertTrue(torch.allclose(probs, logits.softmax(-1), atol=1e-6, rtol=1e-6))

    def test_cross_entropy_matches_legacy_probability_loss(self):
        torch.manual_seed(1357)
        net = CategoricalPolicy()
        phi = torch.randn(8, 1024)
        actions = torch.randint(0, ACTION_SPACE, (8,))
        mask = torch.rand(8, ACTION_SPACE) > 0.4
        mask[torch.arange(actions.shape[0]), actions] = True

        probs = net(phi, mask)
        logits = net.logits(phi, mask)
        legacy_loss = -probs.gather(1, actions.unsqueeze(-1)).squeeze(-1).log().mean()
        refactored_loss = F.cross_entropy(logits, actions)

        self.assertTrue(torch.allclose(refactored_loss, legacy_loss, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()

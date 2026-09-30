"""Runner tests: real Torch autograd/AdamW, eval restoration, GN/BN state."""
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from mortal.online.critic_calibration import actor_forward_context, restore_actor_training_mode


@unittest.skipIf(torch is None, 'PyTorch required; run on the pinned Git runner')
class CriticCalibrationTorchTests(unittest.TestCase):
    def exercise(self, norm, oracle):
        torch.manual_seed(917)
        actor = torch.nn.Sequential(torch.nn.Linear(4, 4), norm, torch.nn.Tanh())
        policy = torch.nn.Sequential(torch.nn.Dropout(0.8), torch.nn.Linear(4, 2))
        critic = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh()) if oracle else None
        value = torch.nn.Linear(4, 1)
        modules = [actor, policy, value] + ([critic] if oracle else [])
        optimizer = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=0.03, weight_decay=0.1)
        x = torch.randn(8, 4)
        restore_actor_training_mode(actor, policy, critic_only=True)
        frozen_state = [{k: v.clone() for k, v in m.state_dict().items()} for m in (actor, policy)]
        value_before = value.weight.detach().clone()
        critic_before = critic[0].weight.detach().clone() if oracle else None
        with torch.no_grad():
            baseline = policy(actor(x)).clone()
        for _ in range(3):
            # Simulate evaluation followed by the actual restoration helper.
            actor.eval()
            policy.eval()
            restore_actor_training_mode(actor, policy, critic_only=True)
            with actor_forward_context(actor, policy, frozen=True, critic_only=True):
                phi = actor(x)
                logits = policy(phi)
            self.assertFalse(phi.requires_grad)
            self.assertFalse(logits.requires_grad)
            self.assertFalse(actor.training)
            self.assertFalse(policy.training)
            features = critic(x) if oracle else phi
            value(features).square().mean().backward()
            self.assertTrue(value.training)
            if oracle:
                self.assertTrue(critic.training)
                self.assertIsNotNone(critic[0].weight.grad)
            self.assertTrue(all(p.grad is None for m in (actor, policy) for p in m.parameters()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for model, before in zip((actor, policy), frozen_state):
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, before[key]), key)
        with torch.no_grad():
            self.assertTrue(torch.equal(policy(actor(x)), baseline))
        self.assertFalse(torch.equal(value.weight, value_before))
        if oracle:
            self.assertFalse(torch.equal(critic[0].weight, critic_before))

    def test_gn_visible_head(self):
        self.exercise(torch.nn.GroupNorm(2, 4), False)

    def test_bn_buffers_visible_head(self):
        self.exercise(torch.nn.BatchNorm1d(4), False)

    def test_gn_oracle_critic(self):
        self.exercise(torch.nn.GroupNorm(2, 4), True)

    def test_bn_buffers_oracle_critic(self):
        self.exercise(torch.nn.BatchNorm1d(4), True)

    def test_normal_alternating_retains_shared_trunk_gradients(self):
        actor = torch.nn.Linear(4, 4)
        policy = torch.nn.Linear(4, 2)
        value = torch.nn.Linear(4, 1)
        restore_actor_training_mode(actor, policy, critic_only=False)
        with actor_forward_context(actor, policy, frozen=False, critic_only=False):
            phi = actor(torch.randn(8, 4))
            policy(phi)  # Policy loss deliberately inactive this step.
        value(phi).square().mean().backward()
        self.assertTrue(actor.training)
        self.assertTrue(policy.training)
        self.assertIsNotNone(actor.weight.grad)
        self.assertIsNone(policy.weight.grad)

    def test_legacy_warmup_retains_existing_training_mode(self):
        actor, policy = torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)
        with actor_forward_context(actor, policy, frozen=True, critic_only=False):
            phi = actor(torch.randn(8, 4))
        self.assertTrue(actor.training)
        self.assertFalse(phi.requires_grad)


try:
    if torch is None:
        raise ModuleNotFoundError('torch')
    from mortal.core.model import Brain, CategoricalPolicy, ValueHead
    from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE
    native_models_available = True
except ModuleNotFoundError:
    native_models_available = False


@unittest.skipUnless(native_models_available, 'PyTorch + libriichi required on pinned runner')
class NativeCriticCalibrationTests(unittest.TestCase):
    def test_actual_models_visible_and_oracle_gn_bn(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for norm in ('GN', 'BN'):
                for use_oracle in (False, True):
                    with self.subTest(norm=norm, oracle=use_oracle):
                        self.exercise(norm, use_oracle)
        finally:
            torch.set_num_threads(previous_threads)

    def exercise(self, norm, use_oracle):
        torch.manual_seed(307)
        actor = Brain(version=4, conv_channels=32, num_blocks=1, Norm=norm)
        policy = CategoricalPolicy()
        value = ValueHead(num_players=4, hidden_size=16)
        critic = Brain(version=4, conv_channels=32, num_blocks=1, is_oracle=True, Norm=norm) if use_oracle else None
        restore_actor_training_mode(actor, policy, critic_only=True)
        state_before = [{k: v.clone() for k, v in m.state_dict().items()} for m in (actor, policy)]
        value_before = {k: v.clone() for k, v in value.state_dict().items()}
        critic_before = {k: v.clone() for k, v in critic.named_parameters()} if use_oracle else None
        obs = torch.randn(2, *obs_shape(4))
        invisible = torch.randn(2, *oracle_obs_shape(4))
        mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)
        optimizer = torch.optim.AdamW([p for m in (actor, policy, value, critic) if m is not None for p in m.parameters()], lr=0.001)
        with torch.no_grad():
            baseline = policy.logits(actor(obs), mask).clone()
        with actor_forward_context(actor, policy, frozen=True, critic_only=True):
            phi = actor(obs)
            logits = policy.logits(phi, mask)
        self.assertFalse(logits.requires_grad)
        critic_phi = critic(obs, invisible_obs=invisible) if use_oracle else phi
        value(critic_phi).square().mean().backward()
        self.assertTrue(all(p.grad is None for m in (actor, policy) for p in m.parameters()))
        optimizer.step()
        actor.eval()
        policy.eval()
        restore_actor_training_mode(actor, policy, critic_only=True)
        for model, before in zip((actor, policy), state_before):
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline, policy.logits(actor(obs), mask)))
        self.assertTrue(any(not torch.equal(v, value_before[k]) for k, v in value.state_dict().items()))
        if use_oracle:
            self.assertTrue(any(not torch.equal(v, critic_before[k]) for k, v in critic.named_parameters()))


if __name__ == '__main__':
    unittest.main()

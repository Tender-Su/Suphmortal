"""
Measure IS ratio variance under simulated policy drift.

Creates random-init models, applies gradient steps of varying magnitude,
and reports ratio statistics. This helps decide vtrace_rho_clip.

Usage:
    python measure_ratio_variance.py
"""

import sys
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, '.')
from config import config
from model import Brain, CategoricalPolicy
from libriichi.consts import obs_shape, ACTION_SPACE


def measure_ratio_stats(num_samples=2048, num_gradient_steps_list=None):
    if num_gradient_steps_list is None:
        num_gradient_steps_list = [1, 5, 10, 50, 100, 200, 400]

    version = config['control']['version']
    obs_channels = obs_shape(version)[0]
    device = torch.device('cpu')

    # Use a smaller model for CPU-feasible measurement.
    # Ratio statistics depend primarily on the policy head shape and logit_thres,
    # not the backbone size, so this gives representative results.
    conv_channels = 64
    num_blocks = 4
    logit_thres = config['policy'].get('logit_thres', 0.0)

    # Create model
    mortal = Brain(conv_channels=conv_channels, num_blocks=num_blocks, version=version, Norm="GN").to(device)
    policy_net = CategoricalPolicy().to(device)
    mortal.eval()
    policy_net.eval()

    # Generate random observations and masks
    torch.manual_seed(42)
    obs = torch.randn(num_samples, obs_channels, 34, device=device)
    # Random masks: at least 2 valid actions per sample
    masks = torch.zeros(num_samples, ACTION_SPACE, dtype=torch.bool, device=device)
    for i in range(num_samples):
        # Enable 3-8 random actions
        n_valid = torch.randint(3, 9, (1,)).item()
        valid_indices = torch.randperm(ACTION_SPACE)[:n_valid]
        masks[i, valid_indices] = True

    # Compute old policy log probs
    with torch.no_grad():
        phi = mortal(obs)
        old_logits = policy_net.logits(phi, masks)
        if logit_thres > 0:
            old_logits = old_logits.clamp(-logit_thres, logit_thres).masked_fill(~masks, -torch.inf)
        old_dist = Categorical(logits=old_logits)
        actions = old_dist.sample()
        old_log_probs = old_dist.log_prob(actions)

    print(f"{'Steps':>6s}  {'Mean':>8s}  {'Std':>8s}  {'Var':>10s}  {'Max':>8s}  {'P99':>8s}  {'P95':>8s}  {'>1.5':>6s}  {'>2.0':>6s}  {'>3.0':>6s}")
    print("-" * 100)

    for n_steps in num_gradient_steps_list:
        # Clone model and apply random gradient perturbation
        mortal_new = Brain(conv_channels=conv_channels, num_blocks=num_blocks, version=version, Norm="GN").to(device)
        policy_new = CategoricalPolicy().to(device)
        mortal_new.load_state_dict(mortal.state_dict())
        policy_new.load_state_dict(policy_net.state_dict())

        # Simulate gradient steps by adding noise proportional to param scale
        mortal_new.train()
        policy_new.train()
        lr = 1e-4  # typical learning rate
        for _ in range(n_steps):
            for p in list(mortal_new.parameters()) + list(policy_new.parameters()):
                if p.requires_grad:
                    # Simulate gradient as random noise with magnitude ~1
                    noise = torch.randn_like(p) * p.data.abs().mean().clamp(min=1e-6)
                    p.data.add_(noise, alpha=-lr)

        mortal_new.eval()
        policy_new.eval()

        # Compute new log probs
        with torch.no_grad():
            new_phi = mortal_new(obs)
            new_logits = policy_new.logits(new_phi, masks)
            if logit_thres > 0:
                new_logits = new_logits.clamp(-logit_thres, logit_thres).masked_fill(~masks, -torch.inf)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()

        mean = ratio.mean().item()
        std = ratio.std().item()
        var = ratio.var().item()
        max_val = ratio.max().item()
        p99 = ratio.quantile(0.99).item()
        p95 = ratio.quantile(0.95).item()
        gt15 = (ratio > 1.5).float().mean().item() * 100
        gt20 = (ratio > 2.0).float().mean().item() * 100
        gt30 = (ratio > 3.0).float().mean().item() * 100

        print(f"{n_steps:>6d}  {mean:>8.4f}  {std:>8.4f}  {var:>10.6f}  {max_val:>8.4f}  {p99:>8.4f}  {p95:>8.4f}  {gt15:>5.1f}%  {gt20:>5.1f}%  {gt30:>5.1f}%")

    # Also measure: what does the ratio look like with actual PPO gradient?
    # Simulate a mini-PPO step with the advantage = random normal
    print("\n--- Simulating actual PPO gradient (not random noise) ---\n")
    print(f"{'Steps':>6s}  {'Mean':>8s}  {'Std':>8s}  {'Var':>10s}  {'Max':>8s}  {'P99':>8s}  {'P95':>8s}  {'>1.5':>6s}  {'>2.0':>6s}  {'>3.0':>6s}")
    print("-" * 100)

    for n_steps in [1, 5, 10, 50, 100, 200, 400]:
        mortal_ppo = Brain(conv_channels=conv_channels, num_blocks=num_blocks, version=version, Norm="GN").to(device)
        policy_ppo = CategoricalPolicy().to(device)
        mortal_ppo.load_state_dict(mortal.state_dict())
        policy_ppo.load_state_dict(policy_net.state_dict())
        mortal_ppo.train()
        policy_ppo.train()

        optimizer = torch.optim.Adam(
            list(mortal_ppo.parameters()) + list(policy_ppo.parameters()),
            lr=1e-4,
        )

        # Use a subset for PPO steps
        ppo_bs = min(512, num_samples)
        advantage = torch.randn(ppo_bs, 1, device=device)

        for step_i in range(n_steps):
            optimizer.zero_grad()
            phi_ppo = mortal_ppo(obs[:ppo_bs])
            logits_ppo = policy_ppo.logits(phi_ppo, masks[:ppo_bs])
            if logit_thres > 0:
                logits_ppo = logits_ppo.clamp(-logit_thres, logit_thres).masked_fill(~masks[:ppo_bs], -torch.inf)
            dist_ppo = Categorical(logits=logits_ppo)
            log_probs_ppo = dist_ppo.log_prob(actions[:ppo_bs])
            # Simplified PPO loss
            ratio_ppo = (log_probs_ppo - old_log_probs[:ppo_bs].detach()).exp()
            loss = -(ratio_ppo * advantage.squeeze()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(mortal_ppo.parameters()) + list(policy_ppo.parameters()),
                1.0
            )
            optimizer.step()

        mortal_ppo.eval()
        policy_ppo.eval()

        with torch.no_grad():
            new_phi = mortal_ppo(obs)
            new_logits = policy_ppo.logits(new_phi, masks)
            if logit_thres > 0:
                new_logits = new_logits.clamp(-logit_thres, logit_thres).masked_fill(~masks, -torch.inf)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()
        mean = ratio.mean().item()
        std = ratio.std().item()
        var = ratio.var().item()
        max_val = ratio.max().item()
        p99 = ratio.quantile(0.99).item()
        p95 = ratio.quantile(0.95).item()
        gt15 = (ratio > 1.5).float().mean().item() * 100
        gt20 = (ratio > 2.0).float().mean().item() * 100
        gt30 = (ratio > 3.0).float().mean().item() * 100

        print(f"{n_steps:>6d}  {mean:>8.4f}  {std:>8.4f}  {var:>10.6f}  {max_val:>8.4f}  {p99:>8.4f}  {p95:>8.4f}  {gt15:>5.1f}%  {gt20:>5.1f}%  {gt30:>5.1f}%")


if __name__ == '__main__':
    measure_ratio_stats()

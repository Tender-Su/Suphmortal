"""CPU-only contract probes on existing training/evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mortal.core.adaptive_curriculum import AdaptiveCurriculumConfig, observe_adaptive_curriculum
from mortal.core.artifacts import atomic_write_json
from mortal.core.model import OracleDualTowerBrain, ValueHead
from mortal.data.dataloader import FileDatasetsIter
from mortal.data.oracle_value import discounted_returns_from_step_rewards, expand_kyoku_rewards_to_steps
from mortal.online.train_online import compute_gae_advantages, validate_oracle_critic_init_checkpoint


def load_checkpoint_models(state):
    cfg = state['config']
    pretrain = state['oracle_critic_pretrain']
    if pretrain['critic_arch'] != 'dual_tower' or pretrain.get('value_loss_mode', 'mse') != 'mse':
        raise ValueError('this audit probe expects the current dual-tower MSE critic')
    brain = OracleDualTowerBrain(
        version=cfg['control']['version'], **cfg['resnet'], Norm='GN',
        oracle_fusion_mode=pretrain.get('oracle_fusion_mode', 'linear'),
        oracle_fusion_hidden=pretrain.get('oracle_fusion_hidden', 512),
    ).eval()
    value = ValueHead(
        num_players=4, hidden_size=pretrain.get('value_head_hidden', 256),
        zero_sum=pretrain.get('exact_zero_sum', False),
    ).eval()
    brain.load_state_dict(state['oracle_brain'], strict=True)
    value.load_state_dict(state['value_net'], strict=True)
    return brain, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--critic-checkpoint', required=True)
    parser.add_argument('--game-log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = Path(args.critic_checkpoint)
    before = path.stat()
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    pretrain = state['oracle_critic_pretrain']
    online_cfg = {
        'env': cfg['env'],
        'policy': {'gae_gamma': pretrain['discount_gamma']},
        'value': {
            'enabled': True, 'oracle_critic': True,
            'target_mode': pretrain['target_mode'], 'reward_source': 'score_rank',
            'oracle_critic_arch': pretrain['critic_arch'],
            'exact_zero_sum': pretrain.get('exact_zero_sum', False),
        },
    }
    contract = validate_oracle_critic_init_checkpoint(state, online_cfg)
    brain, value = load_checkpoint_models(state)
    dataset = FileDatasetsIter(
        version=cfg['control']['version'], file_list=[args.game_log],
        pts=cfg['env']['pts'], oracle=True, player_names=['sl_canonical'],
        emit_opponent_state_labels=False, track_danger_labels=False,
        track_regret_labels=False, value_target_mode='all_players',
        value_reward_source='score_rank',
    )
    trajectory = next(dataset.iter_game_trajectories([args.game_log]))
    obs = torch.from_numpy(trajectory['obs'][:4]).float()
    invisible = torch.from_numpy(trajectory['invisible_obs'][:4]).float()
    with torch.inference_mode():
        pred = value(brain(obs, invisible_obs=invisible))
    assert pred.shape == (4, 4) and torch.isfinite(pred).all()
    rewards = trajectory['kyoku_value_target']
    steps = trajectory['at_kyoku']
    gamma = pretrain['discount_gamma']
    mc = discounted_returns_from_step_rewards(expand_kyoku_rewards_to_steps(rewards, steps), gamma)
    online_mc = np.stack([
        compute_gae_advantages(rewards[:, i], steps, np.zeros(len(steps)), gamma, 1.0)
        for i in range(4)
    ], axis=1)
    np.testing.assert_allclose(mc, online_mc, atol=1e-5)

    adaptive_cfg = AdaptiveCurriculumConfig.from_mapping({
        'phase_name': 'audit_probe',
        'primary': {'name': 'primary', 'direction': 'lower', 'meaningful_delta': 0.001},
        'guardrails': [{'name': 'tail', 'direction': 'lower', 'meaningful_delta': 0.001}],
    })
    baseline = observe_adaptive_curriculum(
        None, adaptive_cfg, optimizer_steps=0,
        metrics={'primary': 1., 'tail': 1.},
        cluster_records={k: [[i, 1., 1] for i in range(32)] for k in ('primary', 'tail')},
    )
    conflicting = observe_adaptive_curriculum(
        baseline.state, adaptive_cfg, optimizer_steps=50_000,
        metrics={'primary': .99, 'tail': 2.},
        cluster_records={'primary': [[i, .99, 1] for i in range(32)], 'tail': [[i, 2., 1] for i in range(32)]},
    )
    after = path.stat()
    result = {
        'device': 'cpu', 'critic_checkpoint': str(path.resolve()),
        'checkpoint_steps': state['steps'], 'contract': contract,
        'file_stable_during_read': (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        'game_log': str(Path(args.game_log).resolve()),
        'real_trajectory_steps': len(steps),
        'critic_output_shape': list(pred.shape), 'critic_predictions_finite': True,
        'critic_zero_sum_max_abs': pred.sum(dim=1).abs().max().item(),
        'mc_vs_online_gae_lambda_one_max_abs': float(np.max(np.abs(mc - online_mc))),
        'adaptive_guardrail_probe': {
            'primary_change': -.01, 'tail_change': 1.,
            'returned_action': conflicting.action,
            'tail_comparison': conflicting.comparisons['tail'],
            'interpretation': 'current guardrails are compensation signals, not vetoes on update_best',
        },
        'sealed_test_data_opened': False,
        'training_run_started': False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

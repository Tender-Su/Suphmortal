"""Small CPU diagnostic on existing simulated games, never the sealed human test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256
from mortal.data.dataloader import FileDatasetsIter
from mortal.data.oracle_value import discounted_returns_from_step_rewards, expand_kyoku_rewards_to_steps
from mortal.eval.paired_1v3 import bootstrap_interval, duplicate_sets
from mortal.online.pretrain_oracle_critic import batch_metrics, finalize_metrics
from mortal.research.audit_sl_rl_contracts import load_checkpoint_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--critic-checkpoint', required=True)
    parser.add_argument('--game-outcomes', required=True)
    parser.add_argument('--player-name', default='sl_canonical')
    parser.add_argument('--seed-sets', type=int, default=32)
    parser.add_argument('--feature-snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(20260905)
    games = json.loads(Path(args.game_outcomes).read_text(encoding='utf-8'))
    keys = list(duplicate_sets(games))
    if not 2 <= args.seed_sets <= len(keys):
        raise ValueError('seed-sets must be between 2 and the available count')
    selected_keys = {keys[i] for i in rng.choice(len(keys), args.seed_sets, replace=False)}
    selected = sorted(
        [g for g in games if (g['seed'], g['seed_key']) in selected_keys],
        key=lambda g: (g['seed'], g['seed_key'], g['challenger_seat']),
    )
    checkpoint = Path(args.critic_checkpoint)
    checkpoint_hash = file_sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    pretrain = state['oracle_critic_pretrain']
    brain, value = load_checkpoint_models(state)
    obs, invisible, targets, inputs = [], [], [], []
    for game in selected:
        path = game['log_path']
        dataset = FileDatasetsIter(
            version=cfg['control']['version'], file_list=[path], pts=cfg['env']['pts'],
            oracle=True, player_names=[args.player_name],
            emit_opponent_state_labels=False, track_danger_labels=False,
            track_regret_labels=False, value_target_mode='all_players',
            value_reward_source='score_rank',
        )
        trajectory = next(dataset.iter_game_trajectories([path]))
        reward = expand_kyoku_rewards_to_steps(trajectory['kyoku_value_target'], trajectory['at_kyoku'])
        target = discounted_returns_from_step_rewards(reward, pretrain['discount_gamma'])
        indices = np.sort(rng.choice(len(target), 4, replace=False))
        obs.extend(trajectory['obs'][indices])
        invisible.extend(trajectory['invisible_obs'][indices])
        targets.extend(target[indices])
        inputs.append({**game, 'sampled_steps': indices.tolist(), 'sha256': file_sha256(path)})
    obs = torch.from_numpy(np.stack(obs)).float()
    invisible = torch.from_numpy(np.stack(invisible)).float()
    targets = torch.from_numpy(np.stack(targets)).float()
    snapshot_path = Path(args.feature_snapshot)
    snapshot_contract = {
        'version': cfg['control']['version'], 'pts': cfg['env']['pts'],
        'gamma': pretrain['discount_gamma'], 'seed_sets': args.seed_sets,
        'player_name': args.player_name, 'game_outcomes_sha256': file_sha256(args.game_outcomes),
    }
    if snapshot_path.exists():
        snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
        if snapshot['contract'] != snapshot_contract:
            raise ValueError('feature snapshot does not match the requested diagnostic')
        obs, invisible, targets, inputs = (snapshot[k] for k in ('obs', 'invisible', 'targets', 'inputs'))
    else:
        atomic_torch_save({
            'contract': snapshot_contract, 'obs': obs, 'invisible': invisible,
            'targets': targets, 'inputs': inputs,
        }, snapshot_path)
    # Shuffle between seat-rotation games inside each bootstrap cluster.
    shift = int(rng.integers(1, 4))
    shuffled = invisible.reshape(args.seed_sets, 4, 4, *invisible.shape[1:]).roll(shift, 1).reshape_as(invisible)
    errors, modes = {}, {}
    for mode, oracle_input in [('true', invisible), ('zero', torch.zeros_like(invisible)), ('shuffled', shuffled)]:
        with torch.inference_mode():
            pred = torch.cat([
                value(brain(obs[i:i + 16], invisible_obs=oracle_input[i:i + 16]))
                for i in range(0, len(obs), 16)
            ])
        assert torch.isfinite(pred).all()
        modes[mode] = finalize_metrics([batch_metrics(pred, targets)])
        errors[mode] = (pred - targets).square().reshape(args.seed_sets, 4, 4, 4)
        print(json.dumps({'mode': mode, 'loss': modes[mode]['loss'], 'p0_loss': float(errors[mode][..., 0].mean())}), flush=True)
    delta = {}
    for control in ('zero', 'shuffled'):
        difference = errors['true'] - errors[control]
        delta[control] = {
            'all_players_mse_true_minus_control': bootstrap_interval(difference.mean((1, 2, 3)).numpy()),
            'p0_mse_true_minus_control': bootstrap_interval(difference[..., 0].mean((1, 2)).numpy()),
        }
    result = {
        'device': 'cpu', 'threads': 1, 'seed': 20260905,
        'critic_checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_steps': state['steps'], 'checkpoint_unchanged': checkpoint_hash == file_sha256(checkpoint),
        'feature_snapshot': str(snapshot_path.resolve()), 'feature_snapshot_sha256': file_sha256(snapshot_path),
        'oracle_input_semantics': 'native default: recorded hidden information plus one frozen random completion of unobserved wall tiles',
        'games': len(selected), 'independent_seed_sets': args.seed_sets, 'states': len(obs),
        'states_per_game': 4, 'sampling_uses_realized_outcome': False,
        'modes': modes, 'paired_cluster_bootstrap': delta, 'inputs': inputs,
        'shuffled_game_shift': shift,
        'shuffled_scope': 'another seat-rotation game within the same independent seed set',
        'scope': 'exploratory dependency diagnostic on archived canonical 1v3 games, not S70 on-policy qualification',
        'sealed_test_data_opened': False, 'actor_strength_evaluated': False,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({'paired_cluster_bootstrap': delta, 'output': args.output}, indent=2))


if __name__ == '__main__':
    main()

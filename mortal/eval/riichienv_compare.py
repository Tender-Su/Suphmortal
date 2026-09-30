"""Cross-architecture duplicate 1v3 evaluation on RiichiEnv.

This runner compares a Mortal-compatible checkpoint against the public
RiichiPPO ActorCritic checkpoint without converting either policy head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mortal.eval.one_vs_three import load_mortal_engine


RANK_POINTS = np.asarray([90.0, 45.0, 0.0, -135.0], dtype=np.float64)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=True),
        )

    def forward(self, x):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        return (avg_out + max_out).sigmoid().unsqueeze(-1) * x


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.ca(self.bn2(self.conv2(out)))
        return self.relu(out + residual)


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, conv_channels, num_blocks, fc_dim, tile_dim):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.ModuleList([ResBlock(conv_channels) for _ in range(num_blocks)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_channels * tile_dim, fc_dim)

    def forward(self, x):
        out = self.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
        return self.relu(self.fc(self.flatten(out)))


class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, conv_channels, num_blocks, fc_dim, tile_dim):
        super().__init__()
        self.backbone = ResNetBackbone(
            in_channels,
            conv_channels,
            num_blocks,
            fc_dim,
            tile_dim,
        )
        self.actor_head = nn.Linear(fc_dim, num_actions)
        self.critic_head = nn.Linear(fc_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as checkpoint_file:
        while chunk := checkpoint_file.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def infer_riichippo_spec(state):
    conv_in = state['backbone.conv_in.weight']
    fc_weight = state['backbone.fc.weight']
    actor_weight = state['actor_head.weight']
    block_indices = {
        int(key.split('.')[2])
        for key in state
        if key.startswith('backbone.res_blocks.')
    }
    conv_channels = int(conv_in.shape[0])
    flattened_dim = int(fc_weight.shape[1])
    if flattened_dim % conv_channels:
        raise ValueError('RiichiPPO flattened dimension is not divisible by conv channels')
    return {
        'in_channels': int(conv_in.shape[1]),
        'num_actions': int(actor_weight.shape[0]),
        'conv_channels': conv_channels,
        'num_blocks': max(block_indices) + 1,
        'fc_dim': int(fc_weight.shape[0]),
        'tile_dim': flattened_dim // conv_channels,
    }


class RiichiPPOAgent:
    def __init__(self, checkpoint_path, *, device):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        state = torch.load(self.checkpoint_path, weights_only=True, map_location='cpu')
        if not isinstance(state, dict):
            raise ValueError('RiichiPPO checkpoint root must be a state dict')
        self.spec = infer_riichippo_spec(state)
        self.model = ActorCriticNetwork(**self.spec).to(self.device).eval()
        self.model.load_state_dict(state, strict=True)

    @torch.inference_mode()
    def act(self, obs):
        feature = np.frombuffer(obs.encode(), dtype=np.float32).reshape(
            self.spec['in_channels'],
            self.spec['tile_dim'],
        ).copy()
        mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        feature_batch = torch.from_numpy(feature).to(self.device).unsqueeze(0)
        mask_batch = torch.from_numpy(mask).to(self.device).unsqueeze(0)
        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            logits, _ = self.model(feature_batch)
        action_id = logits.masked_fill(mask_batch == 0, -torch.inf).argmax(1).item()
        action = obs.find_action(action_id)
        if action is None:
            raise RuntimeError(f'RiichiPPO selected unavailable action {action_id}')
        return action


class MortalAgent:
    def __init__(self, engine, player_id):
        from libriichi.mjai import Bot

        self.bot = Bot(engine, player_id)

    def act(self, obs):
        response = None
        for event in obs.new_events():
            event_json = event if isinstance(event, str) else json.dumps(event)
            candidate = self.bot.react(event_json)
            if candidate is not None:
                response = json.loads(candidate) if isinstance(candidate, str) else candidate
        if response is None:
            raise RuntimeError('Mortal did not return an action for an actionable observation')
        action = obs.select_action_from_mjai(response)
        if action is None:
            raise RuntimeError(f'Mortal returned an unavailable MJAI action: {response}')
        return action


def _mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if values.size < 2:
        return mean, [mean, mean]
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return mean, [mean - 1.96 * se, mean + 1.96 * se]


def summarize_games(games):
    ranks = np.asarray([game['challenger_rank'] for game in games], dtype=np.int64)
    rank_counts = np.bincount(ranks, minlength=5)[1:5]
    set_ids = sorted({game['seed'] for game in games})
    set_rank = []
    set_pt = []
    for seed in set_ids:
        seed_ranks = np.asarray(
            [game['challenger_rank'] for game in games if game['seed'] == seed],
            dtype=np.int64,
        )
        set_rank.append(float(seed_ranks.mean()))
        set_pt.append(float(RANK_POINTS[seed_ranks - 1].mean()))
    avg_rank, rank_ci95 = _mean_ci95(set_rank)
    avg_pt, pt_ci95 = _mean_ci95(set_pt)
    return {
        'sets': len(set_ids),
        'games': len(games),
        'rankings': rank_counts.tolist(),
        'avg_rank': avg_rank,
        'avg_rank_ci95': rank_ci95,
        'avg_pt': avg_pt,
        'avg_pt_ci95': pt_ci95,
        'ci_unit': 'duplicate_seed_set',
    }


def _write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp_path.replace(path)


def run(args):
    from riichienv import GameRule, RiichiEnv

    mortal_cfg = {
        'state_file': str(Path(args.mortal_checkpoint).resolve()),
        'device': args.mortal_device,
        'enable_compile': False,
        'enable_amp': str(args.mortal_device).startswith('cuda'),
        'enable_rule_based_agari_guard': True,
        'enable_metadata': False,
        'name': args.mortal_name,
    }
    mortal_engine = load_mortal_engine(mortal_cfg, enable_metadata=False)
    riichippo = RiichiPPOAgent(args.riichippo_checkpoint, device=args.riichippo_device)
    output_path = Path(args.output).resolve()
    payload = {
        'schema_version': 1,
        'status': 'running',
        'started_at_unix': time.time(),
        'rules': 'RiichiEnv 4p-red-half / default_tenhou',
        'seed_start': args.seed_start,
        'seed_count': args.seed_count,
        'seat_rotation': [0, 1, 2, 3],
        'challenger_kind': args.challenger_kind,
        'games': [],
    }
    model_metadata = {
        'mortal': {
            'name': args.mortal_name,
            'checkpoint': str(Path(args.mortal_checkpoint).resolve()),
            'sha256': _sha256(Path(args.mortal_checkpoint)),
            'device': args.mortal_device,
        },
        'riichippo': {
            'name': args.riichippo_name,
            'checkpoint': str(Path(args.riichippo_checkpoint).resolve()),
            'sha256': _sha256(Path(args.riichippo_checkpoint)),
            'device': args.riichippo_device,
            'spec': riichippo.spec,
        },
    }
    payload['challenger'] = model_metadata[args.challenger_kind]
    champion_kind = 'riichippo' if args.challenger_kind == 'mortal' else 'mortal'
    payload['champion'] = model_metadata[champion_kind]
    _write_result(output_path, payload)

    for seed_offset in range(args.seed_count):
        seed = args.seed_start + seed_offset
        for challenger_seat in range(4):
            env = RiichiEnv(
                game_mode='4p-red-half',
                skip_mjai_logging=False,
                seed=seed,
                rule=GameRule.default_tenhou(),
            )
            agents = {
                player_id: (
                    MortalAgent(mortal_engine, player_id)
                    if (
                        (args.challenger_kind == 'mortal' and player_id == challenger_seat)
                        or (args.challenger_kind == 'riichippo' and player_id != challenger_seat)
                    )
                    else riichippo
                )
                for player_id in range(4)
            }
            obs_dict = env.reset()
            while not env.done():
                actions = {
                    player_id: agents[player_id].act(obs)
                    for player_id, obs in obs_dict.items()
                }
                obs_dict = env.step(actions)
            ranks = env.ranks()
            payload['games'].append({
                'seed': seed,
                'challenger_seat': challenger_seat,
                'challenger_rank': int(ranks[challenger_seat]),
                'ranks': [int(rank) for rank in ranks],
                'scores': [int(score) for score in env.scores()],
            })

        payload['summary'] = summarize_games(payload['games'])
        _write_result(output_path, payload)
        completed = seed_offset + 1
        if completed == 1 or completed % args.progress_every == 0:
            print(
                f"sets={completed}/{args.seed_count} games={completed * 4} "
                f"avg_rank={payload['summary']['avg_rank']:.4f} "
                f"avg_pt={payload['summary']['avg_pt']:+.4f}",
                flush=True,
            )

    payload['status'] = 'complete'
    payload['finished_at_unix'] = time.time()
    payload['summary'] = summarize_games(payload['games'])
    _write_result(output_path, payload)
    print(json.dumps(payload['summary'], ensure_ascii=False), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mortal-checkpoint', required=True)
    parser.add_argument('--riichippo-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mortal-name', default='s70_best_action_score')
    parser.add_argument('--riichippo-name', default='riichippo_2025_self_kldistill_s80_v1')
    parser.add_argument('--challenger-kind', choices=('mortal', 'riichippo'), default='mortal')
    parser.add_argument('--mortal-device', default='cuda:0')
    parser.add_argument('--riichippo-device', default='cuda:0')
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--seed-count', type=int, default=500)
    parser.add_argument('--progress-every', type=int, default=10)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == '__main__':
    main()

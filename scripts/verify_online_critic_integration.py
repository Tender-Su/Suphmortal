"""Load a real corrected Oracle checkpoint through the online constructor on CPU."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from mortal.core.evidence_contract import sha256_file
from mortal.data.oracle_value import OracleTerminalValueDataset
from mortal.online.train_online import build_online_value_models, validate_oracle_critic_init_checkpoint
from mortal.research.audit_sl_rl_contracts import load_checkpoint_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--game-log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    cfg, pre = deepcopy(state['config']), state['oracle_critic_pretrain']
    cfg['value'] = {'enabled': True, 'oracle_critic': True, 'oracle_critic_arch': pre['critic_arch'],
                    'target_mode': 'all_players', 'reward_source': 'score_rank',
                    'exact_zero_sum': pre['exact_zero_sum'], 'oracle_fusion_mode': pre['oracle_fusion_mode'],
                    'oracle_fusion_hidden': pre['oracle_fusion_hidden'],
                    'value_head_hidden': pre['value_head_hidden']}
    cfg['policy']['gae_gamma'] = pre['discount_gamma']
    contract = validate_oracle_critic_init_checkpoint(state, cfg)
    online_brain, online_head = build_online_value_models(cfg, device='cpu')
    online_brain.load_state_dict(state['oracle_brain'], strict=True)
    online_head.load_state_dict(state['value_net'], strict=True)
    offline_brain, offline_head = load_checkpoint_models(state)
    dataset = OracleTerminalValueDataset(version=4, file_list=[args.game_log], pts=cfg['env']['pts'],
                                          shuffle_files=False, oracle_imputation_seed=20260905,
                                          return_mode='score_rank_mc', discount_gamma=pre['discount_gamma'])
    rows = list(itertools.islice(dataset, 4))
    visible = torch.tensor(np.stack([item[0] for item in rows]))
    invisible = torch.tensor(np.stack([item[1] for item in rows]))
    target = torch.tensor(np.stack([item[2] for item in rows]))
    with torch.no_grad():
        a = online_head(online_brain(visible, invisible_obs=invisible))
        b = offline_head(offline_brain(visible, invisible_obs=invisible))
    assert a.shape == (4, 4) and torch.isfinite(a).all() and torch.equal(a, b)
    assert a.sum(-1).abs().max() < 1e-5
    loss = (online_head(online_brain(visible, invisible_obs=invisible)) - target).square().mean()
    loss.backward()
    grads = [p.grad for p in itertools.chain(online_brain.parameters(), online_head.parameters()) if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    result = {'checkpoint_sha256': sha256_file(args.checkpoint), 'steps': state['steps'],
              'contract': contract, 'prediction_max_abs_difference': 0.0,
              'all_players_zero_sum_max_abs': a.sum(-1).abs().max().item(),
              'finite_backward': True, 'grad_tensors': len(grads), 'device': 'cpu',
              'scope': 'native inputs, online constructor, strict weight load, forward and backward; not online qualification'}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

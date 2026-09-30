"""Prove whether repeated native decoding preserves Oracle validation inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from libriichi.dataset import GameplayLoader
from mortal.core.artifacts import atomic_write_json, file_sha256
from mortal.research.audit_sl_rl_contracts import load_checkpoint_models


def probe(path, brain, value, *, trust_seed):
    captures = []
    for _ in range(3):
        loader = GameplayLoader(version=4, oracle=True, trust_seed=trust_seed, augmented=False)
        game = loader.load_log_files([str(path)])[0][0]
        obs = np.asarray(game.take_obs_batch())
        invisible = np.asarray(game.take_invisible_obs_batch())
        indices = np.unique(np.linspace(0, len(obs) - 1, min(16, len(obs))).astype(int))
        with torch.inference_mode():
            pred = value(brain(torch.from_numpy(obs[indices]).float(), invisible_obs=torch.from_numpy(invisible[indices]).float())).numpy()
        captures.append((obs, invisible, pred))
    obs0, oracle0, pred0 = captures[0]
    return {
        'path': str(Path(path).resolve()), 'sha256': file_sha256(path), 'trust_seed': trust_seed,
        'states': len(obs0), 'prediction_states': len(pred0),
        'repeated_decodes': [
            {
                'visible_identical': bool(np.array_equal(obs0, obs)),
                'oracle_identical': bool(np.array_equal(oracle0, oracle)),
                'changed_oracle_state_fraction': float(np.any(oracle0 != oracle, axis=(1, 2)).mean()),
                'changed_oracle_channels': np.flatnonzero(np.any(oracle0 != oracle, axis=(0, 2))).tolist(),
                'p0_prediction_max_abs_difference': float(np.abs(pred0[:, 0] - pred[:, 0]).max()),
            }
            for obs, oracle, pred in captures[1:]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--critic-checkpoint', required=True)
    parser.add_argument('--dev-index', required=True)
    parser.add_argument('--native-log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.critic_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    brain, value = load_checkpoint_models(state)
    dev_files = torch.load(args.dev_index, map_location='cpu', weights_only=False)['file_list']
    result = {
        'device': 'cpu', 'checkpoint_steps': state['steps'],
        'human_dev_default': probe(dev_files[0], brain, value, trust_seed=False),
        'native_default': probe(args.native_log, brain, value, trust_seed=False),
        'native_trust_seed': probe(args.native_log, brain, value, trust_seed=True),
        'sealed_test_data_opened': False,
        'native_trust_seed_is_diagnostic_only': 'determinism is not proof that an older engine used the same wall generator',
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

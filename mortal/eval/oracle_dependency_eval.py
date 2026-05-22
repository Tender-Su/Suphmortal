import argparse
import json
from pathlib import Path

import torch

from mortal.core.checkpoint_utils import checkpoint_brain_is_oracle_structure, load_brain_state_with_input_bridge
from mortal.config import config
from mortal.core.model import Brain, CategoricalPolicy
from mortal.eval.oracle_eval import (
    evaluate_oracle_dependency_modes,
    oracle_dependency_eval_modes,
    write_oracle_dependency_report,
)
from mortal.eval.oracle_experiments import apply_oracle_experiment_to_config, normalize_oracle_input_mode
from mortal.eval.player import TestPlayer
from mortal.eval.search_runtime import build_search_runtime_bundle_from_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--state-file',
        default='',
        help='Checkpoint to evaluate. Defaults to [control].best_state_file, then [control].state_file.',
    )
    parser.add_argument(
        '--games',
        type=int,
        default=0,
        help='Total game count. Defaults to [test_play].games.',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Torch device for challenger inference. Default: cpu.',
    )
    parser.add_argument(
        '--modes',
        nargs='*',
        default=None,
        help='Oracle dependency eval modes. Default: config or true zero shuffled.',
    )
    parser.add_argument(
        '--output-json',
        default='',
        help='Optional JSON output path.',
    )
    return parser.parse_args()


def resolve_state_file(args):
    if args.state_file:
        return str(Path(args.state_file).resolve())
    best_state_file = str(config.get('control', {}).get('best_state_file', '') or '').strip()
    if best_state_file:
        return best_state_file
    return str(config.get('control', {}).get('state_file', '') or '').strip()


def load_checkpoint_runtime(state_file, device):
    state = torch.load(state_file, weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']
    mortal = Brain(
        version=version,
        num_blocks=num_blocks,
        conv_channels=conv_channels,
        is_oracle=checkpoint_brain_is_oracle_structure(state),
        Norm='GN',
    ).eval()
    dqn = CategoricalPolicy().eval()
    load_brain_state_with_input_bridge(mortal, state['mortal'])
    dqn.load_state_dict(state['policy_net'])
    search_runtime_bundle = build_search_runtime_bundle_from_state(
        state,
        device=device,
        enable_compile=False,
    )
    return mortal, dqn, search_runtime_bundle


def main():
    apply_oracle_experiment_to_config(config)
    args = parse_args()
    state_file = resolve_state_file(args)
    if not state_file:
        raise FileNotFoundError('no checkpoint path provided and [control].best_state_file is empty')
    if not Path(state_file).exists():
        raise FileNotFoundError(f'checkpoint not found: {state_file}')

    device = torch.device(args.device)
    games = int(args.games or config['test_play']['games'])
    if games <= 0 or games % 4 != 0:
        raise ValueError(f'games must be a positive multiple of 4, got {games}')

    if args.modes:
        modes = tuple(
            normalize_oracle_input_mode(mode, field_name='--modes')
            for mode in args.modes
        )
    else:
        modes = oracle_dependency_eval_modes(config)

    mortal, dqn, search_runtime_bundle = load_checkpoint_runtime(state_file, device)
    test_player = TestPlayer()
    results = evaluate_oracle_dependency_modes(
        test_player,
        mortal,
        dqn,
        device,
        seed_count=games // 4,
        modes=modes,
        search_runtime_bundle=search_runtime_bundle,
    )
    payload = {
        'state_file': state_file,
        'games': games,
        'device': str(device),
        'modes': list(modes),
        'results': results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        write_oracle_dependency_report(args.output_json, payload)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

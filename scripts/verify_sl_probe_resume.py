"""Validate isolated smoke parent state and continuous or checkpoint-resumed U2."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256
from mortal.supervised.curriculum_probe import learned_state_digest


def equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def read(path):
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def verify_parent(run):
    source = read(run / 'parent.pth')
    baseline = read(run / '20260907_A/update_0000000.pth')
    for key in ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net', 'scaler'):
        if not equal(source[key], baseline[key]):
            raise ValueError(f'parent learned state changed: {key}')
    if baseline['auxiliary_optimizer_steps'] != source['optimizer_steps']:
        raise ValueError('parent auxiliary clock changed')
    names = {name: pid for group, group_names in zip(source['optimizer']['param_groups'],
                                                    source['optimizer_param_groups'])
             for pid, name in zip(group['params'], group_names)}
    aliases = {'danger_aux_net.net.weight': ('danger_aux_net.any_net.weight',
                                           'danger_aux_net.value_net.weight',
                                           'danger_aux_net.player_net.weight')}
    checked = 0
    for group, group_names in zip(baseline['optimizer']['param_groups'], baseline['optimizer_param_groups']):
        for pid, name in zip(group['params'], group_names):
            if name in names:
                expected = source['optimizer']['state'][names[name]]
            else:
                pieces = [source['optimizer']['state'][names[alias]] for alias in aliases[name]]
                expected = {}
                for key in pieces[0]:
                    values = [piece[key] for piece in pieces]
                    if torch.is_tensor(values[0]) and values[0].ndim:
                        expected[key] = torch.cat(values, dim=0)
                    else:
                        if not all(equal(values[0], value) for value in values[1:]):
                            raise ValueError('fused Adam scalar states differ')
                        expected[key] = deepcopy(values[0])
            if not equal(expected, baseline['optimizer']['state'][pid]):
                raise ValueError(f'parent Adam state changed: {name}')
            checked += 1
    rows = [json.loads((run / f'20260907_{route}/update_0000000.json').read_text()) for route in 'ABC']
    if len({row['learned_state_sha256'] for row in rows}) != 1 or any(
            row['splits'] != rows[0]['splits'] for row in rows[1:]):
        raise ValueError('three same-parent observations differ')
    return {'parent_sha256': file_sha256(run / 'parent.pth'), 'all_heads_and_scaler_equal': True,
            'auxiliary_clock_equal': True, 'optimizer_parameter_states_verified': checked,
            'three_baseline_states_and_metrics_equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'compare'])
    parser.add_argument('--directory', required=True)
    parser.add_argument('--from-update', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    run = Path(args.directory).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    if not manifest['smoke'] or manifest['horizons'] != [1, 2]:
        raise ValueError('only isolated U1/U2 smoke outputs may be replayed')
    arm = run / '20260907_A'
    evidence = run / ('resume_verification' if args.from_update else 'continuous_verification')
    if args.action == 'stage':
        if json.loads((run / 'progress.json').read_text())['state'] != 'completed':
            raise ValueError('smoke has not completed')
        report = verify_parent(run)
        evidence.mkdir(exist_ok=False)
        for suffix in ('pth', 'json'):
            shutil.copy2(arm / f'update_0000002.{suffix}', evidence / f'expected_U2.{suffix}')
        atomic_torch_save(read(arm / f'update_{args.from_update:07d}.pth'), arm / 'state_file.pth')
        report['restart_from_update'] = args.from_update
        atomic_write_json(evidence / 'parent.json', report)
    else:
        expected = read(evidence / 'expected_U2.pth')
        actual = read(arm / 'update_0000002.pth')
        if learned_state_digest(expected) != learned_state_digest(actual):
            raise ValueError('resumed learned state differs')
        for key in ('steps', 'optimizer_steps', 'skipped_optimizer_steps', 'auxiliary_optimizer_steps'):
            if expected[key] != actual[key]:
                raise ValueError(f'resumed counter differs: {key}')
        for key in ('dataset', 'observed', 'rng'):
            if not equal(expected['curriculum_probe'][key], actual['curriculum_probe'][key]):
                raise ValueError(f'resumed state differs: {key}')
        before = json.loads((evidence / 'expected_U2.json').read_text())
        after = json.loads((arm / 'update_0000002.json').read_text())
        if before['splits'] != after['splits'] or before['exposure'] != after['exposure']:
            raise ValueError('resumed observations differ')
        report = {'learned_state_equal': True, 'rng_and_consumed_cursor_equal': True,
                  'validation_metrics_and_exposure_equal': True, 'optimizer_updates': actual['optimizer_steps'],
                  'restart_from_update': args.from_update,
                  'peak_cuda_allocated_bytes': after['peak_cuda_allocated_bytes']}
        atomic_write_json(evidence / 'comparison.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

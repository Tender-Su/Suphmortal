"""Real-log CPU regression: fixed completion and fold-invariant full-clock returns."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import libriichi
from libriichi.dataset import GameplayLoader
from mortal.core.evidence_contract import sha256_file, native_module_file
from mortal.data.oracle_value import full_decision_clock, oracle_step_value_targets


def decode(filename, *, fold_count=1, fold_index=0, seed=20260905):
    loader = GameplayLoader(version=4, oracle=True, trust_seed=False)
    loader.set_oracle_imputation_seed(seed)
    loader.set_sample_fold(fold_count, fold_index, 20260416)
    result = []
    for game in loader.load_log_files([str(filename)])[0]:
        obs = game.take_obs_batch()
        hidden = game.take_invisible_obs_batch()
        kyoku = game.take_at_kyoku_batch()
        full, indices = full_decision_clock(game, kyoku, require_metadata=True)
        grp = game.take_grp()
        features, ranks = grp.take_feature(), grp.take_rank_by_player()
        targets = {
            str(gamma): oracle_step_value_targets(
                features, ranks, [6, 4, 2, 0], full, 'score_rank_mc', gamma,
            )[indices] for gamma in (1.0, 0.999, 0.95)
        }
        result.append({'obs': obs, 'hidden': hidden, 'indices': indices, 'targets': targets})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline = decode(args.log)
    repeated = decode(args.log)
    other = decode(args.log, seed=20260906)
    assert all(np.array_equal(a['obs'], b['obs']) and np.array_equal(a['hidden'], b['hidden'])
               for a, b in zip(baseline, repeated))
    assert any(not np.array_equal(a['hidden'], b['hidden']) for a, b in zip(baseline, other))
    del repeated, other
    checks = []
    for count in (16, 64):
        covered = [set() for _ in baseline]
        for fold in range(count):
            for player, (full, part) in enumerate(zip(baseline, decode(
                args.log, fold_count=count, fold_index=fold,
            ))):
                indices = part['indices']
                assert not (covered[player] & set(indices.tolist()))
                covered[player].update(indices.tolist())
                assert np.array_equal(full['obs'][indices], part['obs'])
                assert np.array_equal(full['hidden'][indices], part['hidden'])
                for gamma in full['targets']:
                    assert np.array_equal(full['targets'][gamma][indices], part['targets'][gamma])
        assert all(indices == set(range(len(full['obs'])))
                   for indices, full in zip(covered, baseline))
        checks.append({'fold_count': count, 'states': sum(map(len, covered)),
                       'all_players_max_abs_target_difference': 0.0})
    result = {
        'source_sha256': sha256_file(args.log), 'native': str(native_module_file(libriichi)),
        'native_sha256': sha256_file(native_module_file(libriichi)), 'trust_seed': False,
        'fixed_seed_repeats_identical': True, 'other_completion_seed_changes_hidden': True,
        'gamma_values': [1.0, 0.999, 0.95], 'fold_checks': checks,
        'hidden_hashes': [hashlib.sha256(item['hidden'].tobytes()).hexdigest() for item in baseline],
        'sealed_test_opened': False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

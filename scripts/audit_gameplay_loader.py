import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np

from libriichi.dataset import GameplayLoader
from mortal.data.dataloader import iter_loaded_gameplay_batches


FIELDS = (
    ('obs', 'take_obs_batch'),
    ('invisible_obs', 'take_invisible_obs_batch'),
    ('actions', 'take_actions_batch'),
    ('masks', 'take_masks_batch'),
    ('at_kyoku', 'take_at_kyoku_batch'),
)


def update_array(digest, value):
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def audit(files, trust_seed, fold_count, fold_index, fold_seed):
    loader = GameplayLoader(
        version=4,
        oracle=True,
        augmented=False,
        trust_seed=trust_seed,
        track_opponent_states=False,
        track_danger_labels=False,
        track_regret_labels=False,
    )
    if fold_count > 1:
        setter = getattr(loader, 'set_sample_fold', None)
        if setter is None:
            raise RuntimeError('this libriichi build does not support native state folds')
        setter(fold_count, fold_index, fold_seed)

    digests = {name: hashlib.sha256() for name, _ in FIELDS}
    digests.update({
        'grp_feature': hashlib.sha256(),
        'rank_by_player': hashlib.sha256(),
        'player_id': hashlib.sha256(),
    })
    rows = defaultdict(int)
    gameplays = 0
    sample_xor = 0
    sample_sum = 0
    sample_count = 0
    modulus = 1 << 256
    for source_name, gameplay_batch in iter_loaded_gameplay_batches(
        loader,
        files,
        bulk_event_cache=all(str(file).endswith('.events.zst') for file in files),
    ):
        for gameplay_index, gameplay in enumerate(gameplay_batch):
            gameplays += 1
            values = {}
            for name, method_name in FIELDS:
                value = getattr(gameplay, method_name)()
                values[name] = np.asarray(value)
                update_array(digests[name], value)
                rows[name] += int(np.asarray(value).shape[0])
            grp = gameplay.take_grp()
            update_array(digests['grp_feature'], grp.take_feature())
            update_array(digests['rank_by_player'], grp.take_rank_by_player())
            player_id = int(gameplay.take_player_id())
            update_array(digests['player_id'], [player_id])
            for row_index in range(values['obs'].shape[0]):
                sample_digest = hashlib.sha256()
                sample_digest.update(str(source_name).encode('utf-8'))
                sample_digest.update(np.asarray(
                    (gameplay_index, player_id), dtype=np.int64
                ).tobytes())
                for name in ('obs', 'actions', 'masks', 'at_kyoku'):
                    update_array(sample_digest, values[name][row_index])
                sample_hash = int.from_bytes(sample_digest.digest(), 'little')
                sample_xor ^= sample_hash
                sample_sum = (sample_sum + sample_hash) % modulus
                sample_count += 1

    return {
        'module': __import__('libriichi').__file__,
        'files': len(files),
        'gameplays': gameplays,
        'trust_seed': trust_seed,
        'fold': [fold_count, fold_index, fold_seed],
        'rows': dict(rows),
        'sample_multiset': {
            'count': sample_count,
            'xor256': f'{sample_xor:064x}',
            'sum256': f'{sample_sum:064x}',
        },
        'sha256': {name: digest.hexdigest() for name, digest in digests.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description='Hash GameplayLoader outputs for binary-equivalence audits.'
    )
    parser.add_argument('files', nargs='+')
    parser.add_argument('--trust-seed', action='store_true')
    parser.add_argument('--fold-count', type=int, default=1)
    parser.add_argument('--fold-index', type=int, default=0)
    parser.add_argument('--fold-seed', type=int, default=0)
    parser.add_argument('--all-folds', action='store_true')
    args = parser.parse_args()
    if args.fold_count <= 0:
        parser.error('--fold-count must be positive')
    if not 0 <= args.fold_index < args.fold_count:
        parser.error('--fold-index must be in [0, fold-count)')

    if args.all_folds:
        folds = [
            audit(
                args.files,
                args.trust_seed,
                args.fold_count,
                fold_index,
                args.fold_seed,
            )
            for fold_index in range(args.fold_count)
        ]
        xor256 = 0
        sum256 = 0
        count = 0
        for result in folds:
            multiset = result['sample_multiset']
            xor256 ^= int(multiset['xor256'], 16)
            sum256 += int(multiset['sum256'], 16)
            count += int(multiset['count'])
        result = {
            'folds': folds,
            'aggregate_sample_multiset': {
                'count': count,
                'xor256': f'{xor256:064x}',
                'sum256': f'{sum256 % (1 << 256):064x}',
            },
        }
    else:
        result = audit(
            args.files,
            args.trust_seed,
            args.fold_count,
            args.fold_index,
            args.fold_seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

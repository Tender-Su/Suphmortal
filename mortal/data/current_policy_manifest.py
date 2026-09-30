"""Frozen current-policy rollout registration. Standard-library only; no training.

Native log headers and completed probe receipts are the authority for seed/seat
identity. Filenames are never used to infer a group or a player's perspective.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from mortal.core.evidence_contract import fingerprint, sha256_file

SCHEMA = 'current_policy_critic_manifest_v1'
CLUSTER_UNIT = 'full_seed_key_four_seat_group'
SOURCE_KEYS = (
    'mortal/data/dataloader.py', 'mortal/core/model.py',
    'mortal/core/checkpoint_utils.py', 'mortal/online/train_online.py',
    'mortal/eval/player.py', 'mortal/eval/engine.py',
    'libriichi/src/dataset/invisible.rs',
)
SAMPLING_KEYS = (
    'actor_explore_rate', 'opponent_explore_rate', 'actor_agari_guard',
    'opponent_agari_guard', 'search_enabled', 'actor_oracle_guiding',
)


def canonical_path(value):
    return str(Path(value).resolve())


def group_name(seed, seed_key):
    for value in (seed, seed_key):
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError('seed and seed_key must be unsigned 64-bit integers')
    return f'{seed}:{seed_key}'


def cluster_id(group_id):
    # Stable int63 fits the existing tensor/paired-cluster evaluator contract.
    return int(fingerprint({'seed_group': group_id})[:16], 16) & ((1 << 63) - 1)


def _require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('expected a lowercase SHA256 digest')
    return value


def _identity(receipt):
    weights = receipt['weights_and_config']
    sources = {name.replace('\\', '/'): digest for name, digest in receipt['source_hashes'].items()}
    native = sorted(_require_hash(digest) for digest in receipt['native_extension_sha256'].values())
    if not native:
        raise ValueError('probe receipt has no native extension identity')
    result = {
        'actor_sha256': _require_hash(weights['actor']['sha256']),
        'opponent_sha256': _require_hash(weights['opponent']['sha256']),
        'native_sha256': _require_hash(receipt['native_sha256']),
        'native_extension_sha256': native,
        'source_hashes': {name: _require_hash(sources[name]) for name in SOURCE_KEYS},
        'torch_version': receipt['torch_version'], 'numpy_version': receipt['numpy_version'],
        'sampling': {name: receipt[name] for name in SAMPLING_KEYS},
        'rank_points': receipt['probe_pts'],
        'actor_observation': 'visible_only', 'player_names': ['trainee'],
        'target_mode': 'all_players', 'return_mode': 'score_rank_mc', 'discount_gamma': 1.0,
        'oracle_input': 'recorded_hidden_plus_imputed_unobserved_wall', 'trust_seed': False,
    }
    if (result['rank_points'] != [2, 1, 0, -3]
            or result['sampling'] != dict(zip(SAMPLING_KEYS, (1.0, 0.0, False, True, False, False)))):
        raise ValueError('probe is not the frozen visible current-policy sampling contract')
    return result


def _verify_log(game):
    filename = game['path']
    if sha256_file(filename) != game['sha256']:
        raise ValueError(f'game log hash mismatch: {filename}')
    with gzip.open(filename, 'rt', encoding='utf-8') as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    if not events or events[0].get('type') != 'start_game' or events[-1].get('type') != 'end_game':
        raise ValueError(f'incomplete native game: {filename}')
    starts = sum(e.get('type') == 'start_kyoku' for e in events)
    if starts == 0 or starts != sum(e.get('type') == 'end_kyoku' for e in events):
        raise ValueError(f'unfinished kyoku: {filename}')
    header = events[0]
    names = header.get('names', [])
    if (len(names) != 4 or names.count('trainee') != 1
            or names.index('trainee') != game['trainee_seat']
            or header.get('seed') != [game['seed'], game['seed_key']]
            or starts != game['kyoku_count']):
        raise ValueError(f'registered seed/seat/completeness differs from native log: {filename}')


def register_probe_blocks(*, train_probe_dirs, dev_probe_dirs, test_probe_dirs=()):
    """Read finished receipts and create an immutable, explicit group split."""
    blocks, games, identity = [], [], None
    sources = [(source, split) for split, roots in (
        ('train', train_probe_dirs), ('dev', dev_probe_dirs), ('test', test_probe_dirs)
    ) for source in roots]
    if len({canonical_path(source) for source, _ in sources}) != len(sources):
        raise ValueError('a probe block cannot be registered twice or cross splits')
    for source, split in sources:
        root = Path(source).resolve()
        receipt_path, outcomes_path = root / 'provenance.json', root / 'outcomes.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('status') != 'complete':
            raise ValueError(f'probe block must be complete: {root}')
        block_identity = _identity(receipt)
        if identity is not None and identity != block_identity:
            raise ValueError('probe blocks have different actor/opponent/native/source/sampling identities')
        identity = block_identity
        outcomes_hash = sha256_file(outcomes_path)
        if receipt.get('artifact_sha256', {}).get('outcomes.json') != outcomes_hash:
            raise ValueError(f'outcomes receipt hash mismatch: {outcomes_path}')
        outcomes = json.loads(outcomes_path.read_text(encoding='utf-8'))
        arguments = receipt['arguments']
        count = arguments['games']
        if type(count) is not int or count < 4 or count % 4 or len(outcomes) != count:
            raise ValueError('probe outcomes count must match complete four-seat groups')
        expected_groups = {group_name(seed, arguments['seed_key']) for seed in
                           range(arguments['seed_start'], arguments['seed_start'] + count // 4)}
        registered_groups = set()
        block_id = sha256_file(receipt_path)
        blocks.append({
            'id': block_id, 'provenance_path': str(receipt_path),
            'provenance_sha256': block_id, 'outcomes_path': str(outcomes_path),
            'outcomes_sha256': outcomes_hash, 'source_commit': receipt['source_commit'],
            'weights_and_config': receipt['weights_and_config'],
            'arguments': arguments,
        })
        for outcome in outcomes:
            gid = group_name(outcome['seed'], outcome['seed_key'])
            registered_groups.add(gid)
            games.append({
                'path': canonical_path(outcome['log_path']), 'sha256': _require_hash(outcome['sha256']),
                'seed': outcome['seed'], 'seed_key': outcome['seed_key'],
                'trainee_seat': outcome['challenger_seat'], 'group_id': gid,
                'cluster_id': cluster_id(gid), 'block_id': block_id,
                'kyoku_count': outcome['kyoku_count'], 'split': split,
            })
        if registered_groups != expected_groups:
            raise ValueError('probe outcome seed groups differ from receipt arguments')
    payload = {
        'schema': SCHEMA, 'identity': identity, 'blocks': blocks,
        'split_unit': CLUSTER_UNIT, 'ci_cluster_unit': CLUSTER_UNIT,
        'games': sorted(games, key=lambda g: (g['seed'], g['seed_key'], g['trainee_seat'])),
    }
    payload['fingerprint'] = fingerprint(payload)
    validate_manifest(payload, verify_files=True)
    return payload


def validate_manifest(payload, *, verify_files):
    """Validate the whole fixed ledger; hashing is startup-only, never per batch."""
    if payload.get('schema') != SCHEMA:
        raise ValueError('unsupported current-policy manifest schema')
    unsigned = {key: value for key, value in payload.items() if key != 'fingerprint'}
    if payload.get('fingerprint') != fingerprint(unsigned):
        raise ValueError('current-policy manifest fingerprint mismatch')
    if payload.get('split_unit') != CLUSTER_UNIT or payload.get('ci_cluster_unit') != CLUSTER_UNIT:
        raise ValueError('current-policy split/CI must use full seed groups')
    blocks = {block['id']: block for block in payload['blocks']}
    if len(blocks) != len(payload['blocks']) or not blocks:
        raise ValueError('duplicate or empty probe blocks')
    registered = {}
    if verify_files:
        for block in blocks.values():
            for name in ('provenance', 'outcomes'):
                if sha256_file(block[f'{name}_path']) != block[f'{name}_sha256']:
                    raise ValueError(f'probe {name} changed after registration')
            receipt = json.loads(Path(block['provenance_path']).read_text(encoding='utf-8'))
            if (receipt.get('status') != 'complete' or _identity(receipt) != payload['identity']
                    or block['id'] != block['provenance_sha256']
                    or block['weights_and_config'] != receipt['weights_and_config']
                    or block['arguments'] != receipt['arguments']
                    or receipt['artifact_sha256']['outcomes.json'] != block['outcomes_sha256']):
                raise ValueError('manifest identity differs from its completed probe receipt')
            outcomes = json.loads(Path(block['outcomes_path']).read_text(encoding='utf-8'))
            if len(outcomes) != receipt['arguments']['games']:
                raise ValueError('registered block outcomes count mismatch')
            for outcome in outcomes:
                key = (block['id'], canonical_path(outcome['log_path']))
                if key in registered:
                    raise ValueError('duplicate registered outcome')
                registered[key] = outcome
    groups, paths, hashes, cluster_ids = defaultdict(list), set(), set(), {}
    splits = {'train': [], 'dev': [], 'test': []}
    for game in payload['games']:
        filename = canonical_path(game['path'])
        gid = group_name(game['seed'], game['seed_key'])
        if filename in paths or game['sha256'] in hashes:
            raise ValueError('duplicate game path or content across probe blocks')
        if filename != game['path'] or game['group_id'] != gid or game['cluster_id'] != cluster_id(gid):
            raise ValueError('noncanonical game path or invalid explicit seed-group identity')
        if game['block_id'] not in blocks or game['split'] not in splits:
            raise ValueError('unknown game block or split')
        if type(game['trainee_seat']) is not int or game['trainee_seat'] not in range(4):
            raise ValueError('invalid trainee seat')
        _require_hash(game['sha256'])
        if game['cluster_id'] in cluster_ids and cluster_ids[game['cluster_id']] != gid:
            raise ValueError('seed-group cluster ID collision')
        cluster_ids[game['cluster_id']] = gid
        groups[gid].append(game)
        paths.add(filename)
        hashes.add(game['sha256'])
        splits[game['split']].append(filename)
        if verify_files:
            outcome = registered.pop((game['block_id'], filename), None)
            if outcome is None or any(game[left] != outcome[right] for left, right in (
                ('sha256', 'sha256'), ('seed', 'seed'), ('seed_key', 'seed_key'),
                ('trainee_seat', 'challenger_seat'), ('kyoku_count', 'kyoku_count'),
            )):
                raise ValueError('manifest game differs from registered outcome')
            _verify_log(game)
    if registered:
        raise ValueError('manifest omits registered games from a probe block')
    for gid, members in groups.items():
        if len(members) != 4 or {g['trainee_seat'] for g in members} != set(range(4)):
            raise ValueError(f'incomplete or repeated four-seat group: {gid}')
        if len({g['split'] for g in members}) != 1 or len({g['block_id'] for g in members}) != 1:
            raise ValueError(f'seed group crosses splits or probe blocks: {gid}')
    if not splits['train'] or not splits['dev']:
        raise ValueError('current-policy training requires non-empty train and dev groups')
    if len(splits['dev']) < 8:
        raise ValueError('dev inference requires at least two complete independent seed groups')
    return splits


def load_current_policy_manifest(filename):
    payload = json.loads(Path(filename).read_text(encoding='utf-8'))
    splits = validate_manifest(payload, verify_files=True)
    return {
        'splits': splits,
        'group_ids': {g['path']: g['cluster_id'] for g in payload['games']},
        'file_sha256': {g['path']: g['sha256'] for g in payload['games']},
        'contract': {
            'schema': SCHEMA, 'manifest_fingerprint': payload['fingerprint'],
            'manifest_sha256': sha256_file(filename), 'identity': payload['identity'],
            'group_mapping_fingerprint': fingerprint([
                [g['path'], g['group_id'], g['cluster_id'], g['split']] for g in payload['games']]),
            'player_names': ['trainee'], 'ci_cluster_unit': CLUSTER_UNIT,
            'ci_estimand': 'state_weighted_mean_with_seed_group_cluster_se',
            'split_groups': {name: len(files) // 4 for name, files in splits.items()},
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-probe-dir', action='append', required=True,
                        help='completed training probe block; repeat to combine blocks')
    parser.add_argument('--dev-probe-dir', action='append', required=True,
                        help='completed held-out dev block; repeat to combine blocks')
    parser.add_argument('--test-probe-dir', action='append', default=[],
                        help='optional completed sealed-test block')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'never overwrite a registered manifest: {output}')
    payload = register_probe_blocks(train_probe_dirs=args.train_probe_dir,
                                    dev_probe_dirs=args.dev_probe_dir,
                                    test_probe_dirs=args.test_probe_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'manifest': str(output.resolve()), 'fingerprint': payload['fingerprint'],
                      'games': len(payload['games'])}, sort_keys=True))


if __name__ == '__main__':
    main()

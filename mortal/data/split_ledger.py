"""Cross-stage game identity ledger; reads indexes, never sealed game payloads."""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import torch

from mortal.core.evidence_contract import fingerprint, sha256_file


GAME_ID = re.compile(r'(\d{10}gm-[a-z0-9-]+)', re.IGNORECASE)


def game_identity(filename):
    match = GAME_ID.search(str(filename))
    if not match:
        raise ValueError('index does not expose original game ids; supply the source index: ' + str(filename))
    return match.group(1).lower()


def index_record(filename):
    filename, _, field = str(filename).partition('#')
    payload = torch.load(filename, map_location='cpu', weights_only=True)
    paths = payload[field or 'file_list'] if isinstance(payload, dict) else payload
    ids = {game_identity(item) for item in paths}
    return {'index_sha256': sha256_file(filename), 'entries': len(paths), 'unique_games': len(ids),
            'game_ids_sha256': fingerprint(sorted(ids)), 'first_month': min(ids)[:6],
            'last_month': max(ids)[:6]}, ids


def build_ledger(indexes, *, unknown_ancestors=()):
    records, identities = {}, {}
    for role, file in indexes.items():
        records[role], identities[role] = index_record(file)
    overlaps = []
    for left, right in combinations(identities, 2):
        common = identities[left] & identities[right]
        if common:
            overlaps.append({'left': left, 'right': right, 'games': len(common),
                             'examples': sorted(common)[:5]})
    holdout_overlaps = [item for item in overlaps if any(
        word in item[side] for side in ('left', 'right') for word in (':dev', ':test', ':validation')
    )]
    result = {'schema_version': 1, 'roles': records, 'overlaps': overlaps,
              'unknown_ancestors': list(unknown_ancestors), 'sealed_game_payloads_opened': False,
              'strict_pipeline_holdout_verified': not holdout_overlaps and not unknown_ancestors}
    result['fingerprint'] = fingerprint(result)
    return result


def require_disjoint(ledger, heldout_role, *, exposed_roles, require_known_ancestry=True):
    if heldout_role not in ledger['roles'] or any(role not in ledger['roles'] for role in exposed_roles):
        raise ValueError('split ledger is missing a required role')
    if require_known_ancestry and ledger['unknown_ancestors']:
        raise ValueError('unknown actor ancestors prevent a strict pipeline holdout claim')
    for overlap in ledger['overlaps']:
        pair = {overlap['left'], overlap['right']}
        if heldout_role in pair and any(role in pair for role in exposed_roles):
            raise ValueError(f'holdout overlaps an exposed stage: {overlap}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', action='append', required=True, help='stage:role=original_file_index.pth')
    parser.add_argument('--unknown-ancestor', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ledger = build_ledger(dict(item.split('=', 1) for item in args.index), unknown_ancestors=args.unknown_ancestor)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(ledger, stream, indent=2, sort_keys=True)
    print(json.dumps({'fingerprint': ledger['fingerprint'], 'roles': len(ledger['roles']),
                      'overlapping_role_pairs': len(ledger['overlaps']),
                      'strict_pipeline_holdout_verified': ledger['strict_pipeline_holdout_verified']}))


if __name__ == '__main__':
    main()

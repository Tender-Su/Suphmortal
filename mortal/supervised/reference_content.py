"""Optional immutable reference fingerprints for a saved SL continuation.

Reference ledgers constrain source bytes only. They never supply model state,
training counters, RNG, sampler positions, or consumed exposure.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from mortal.core.artifacts import file_sha256, stable_json_digest
from mortal.supervised.continuation import ledger_snapshot


def bind_reference_contract(descriptor, original, index, state, checkpoint_sha, reference, reference_index):
    """Validate identities and actual index arrays before creating any output."""
    required = {
        'format', 'source_identity', 'source_manifest_sha256', 'source_checkpoint_sha256',
        'source_ledger_content_sha256', 'reference_directory', 'reference_identity',
        'reference_manifest_sha256', 'reference_ledger_content_sha256',
        'domains_sha256', 'roles_sha256', 'phase', 'seed', 'recipe', 'order_evidence',
    }
    if set(descriptor) != required or descriptor['format'] != 'sl_reference_content_contract_v1':
        raise ValueError('invalid reference content contract schema')
    if (original['identity'] != descriptor['source_identity']
            or reference['identity'] != descriptor['reference_identity']
            or checkpoint_sha != descriptor['source_checkpoint_sha256']):
        raise ValueError('reference content source identity/checkpoint mismatch')
    for manifest, key in ((original, 'source'), (reference, 'reference')):
        if file_sha256(Path(manifest['directory']) / 'manifest.json') != descriptor[key + '_manifest_sha256']:
            raise ValueError('reference content manifest hash mismatch: ' + key)
    if Path(reference['directory']).resolve() != Path(descriptor['reference_directory']).resolve():
        raise ValueError('reference content directory mismatch')
    for key in ('domains', 'roles'):
        if (index[key] != reference_index[key]
                or stable_json_digest(index[key]) != descriptor[key + '_sha256']):
            raise ValueError('reference content index domain/role mismatch: ' + key)
    native = lambda m: sorted(digest for name, digest in m['source_sha256'].items()
        if Path(name).name.startswith('libriichi') and Path(name).suffix in ('.pyd', '.so'))
    if not native(original) or native(original) != native(reference):
        raise ValueError('reference native loader fingerprint mismatch')
    data = state['curriculum_probe']['dataset']
    if (state['run_provenance']['phase'] != descriptor['phase']
            or reference['phase'] != descriptor['phase']
            or data['sampler']['seed'] != descriptor['seed']
            or data['sampler']['recipe'] != descriptor['recipe']):
        raise ValueError('reference content phase/seed/recipe mismatch')
    # Continuation source config is immutable and separately manifest-pinned.
    cfg_path = Path(reference['directory']) / 'source_config.json'
    if file_sha256(cfg_path) != reference['input_sha256']['source_config.json']:
        raise ValueError('reference source config changed')
    from mortal.core.artifacts import read_shared_text
    cfg = json.loads(read_shared_text(cfg_path))
    sl = cfg['supervised']
    if sl['seed'] != descriptor['seed'] or sl['run_provenance']['phase'] != descriptor['phase']:
        raise ValueError('reference config phase/seed mismatch')
    evidence = descriptor['order_evidence']
    if (not isinstance(evidence, dict) or not evidence.get('artifact_sha256')
            or not evidence.get('next_logical_1024_row_identity_sha256')
            or not evidence.get('reference_end_checkpoint_sha256')):
        raise ValueError('reference input-order evidence missing')
    return dict(descriptor, contract_sha256=stable_json_digest(descriptor),
                semantics='content fingerprints only; no reference progress/state import')


def reference_ledger_snapshot(source, destination, source_identity, consumed_files, *,
                              current_hashes, binding):
    """Back up both read-only sources, verify, then atomically publish their union.

    Failed staging files remain as evidence. A successful union alone does not
    authorize run(): the final continuation receipt must also be present.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('new inherited reference content snapshot required')
    primary = destination.with_name('source_content_snapshot.sqlite3')
    reference = destination.with_name('reference_content_snapshot.sqlite3')
    building = destination.with_name(destination.name + '.building')
    if any(p.exists() for p in (primary, reference, building)):
        raise FileExistsError('reference snapshot staging already exists')
    if source_identity != binding['source_identity']:
        raise ValueError('source ledger identity does not match binding')
    first = ledger_snapshot(source, primary, source_identity, consumed_files,
                            current_hashes=current_hashes)
    second_source = Path(binding['reference_directory']) / 'training_content.sqlite3'
    second = ledger_snapshot(second_source, reference, binding['reference_identity'], ())
    for tag, result in (('source', first), ('reference', second)):
        if result['content_sha256'] != binding[tag + '_ledger_content_sha256']:
            raise ValueError('reference content snapshot digest mismatch: ' + tag)
    snapshots = {p.name: file_sha256(p) for p in (primary, reference)}
    union_identity = stable_json_digest({
        'format': 'sl_content_pin_union_v1', 'contract_sha256': binding['contract_sha256'],
        'snapshots': snapshots,
    })
    with primary.open('rb') as src, building.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    with closing(sqlite3.connect(building)) as db, db:
        db.execute('ATTACH DATABASE ? AS reference',
                   (reference.resolve().as_uri() + '?mode=ro',))
        mismatch = db.execute(
            'SELECT a.path FROM main.games a JOIN reference.games b ON a.path=b.path '
            'WHERE a.sha256!=b.sha256 LIMIT 1').fetchone()
        if mismatch:
            raise ValueError('reference content pin conflict: ' + mismatch[0])
        db.execute('INSERT OR IGNORE INTO main.games SELECT path,sha256 FROM reference.games')
        db.execute('UPDATE main.metadata SET identity=?', (union_identity,))
        digest, count = hashlib.sha256(), 0
        for row in db.execute('SELECT path,sha256 FROM main.games ORDER BY path'):
            digest.update(json.dumps(row, ensure_ascii=True).encode() + b'\n')
            count += 1
    # The destination belongs to a fresh, isolated output directory. Hard-link
    # publication is atomic and refuses an existing destination on both OSes.
    os.link(building, destination)
    building.unlink()
    return {'pinned_games': count, 'content_sha256': digest.hexdigest(),
            'identity': union_identity, 'snapshot_sha256': snapshots,
            'source_snapshot': first, 'reference_snapshot': second,
            'scope': 'immutable union of content pins; membership is not consumption',
            'reference_progress_imported': False}

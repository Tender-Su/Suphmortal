"""Content-bound contracts for fixed validation inputs and experimental provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def native_module_file(module):
    """Locate an extension behind a wheel's Python re-export package."""
    candidates = {Path(loaded.__file__).resolve()
                  for name, loaded in list(sys.modules.items())
                  if (name == module.__name__ or name.startswith(module.__name__ + '.'))
                  and getattr(loaded, '__file__', None)
                  and Path(loaded.__file__).suffix.lower() in ('.pyd', '.so', '.dll')}
    direct = Path(module.__file__).resolve()
    if direct.suffix.lower() in ('.pyd', '.so', '.dll'):
        candidates.add(direct)
    if len(candidates) != 1:
        raise ValueError('cannot identify a unique native extension for ' + module.__name__)
    return candidates.pop()


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()


def validation_input_contract(file_list, *, settings, native_file):
    # Hash actual validation bytes, not just a mutable index or filename list.
    # Preserve order: max_batches and state folds make it part of the input contract.
    files = [{'name': str(Path(name).resolve()), 'sha256': sha256_file(name)}
             for name in file_list]
    result = {'schema_version': 1, 'files': files, 'settings': settings,
              'native_sha256': sha256_file(native_file)}
    result['fingerprint'] = fingerprint(result)
    return result


def require_finalist_decision(decision_file, checkpoint_files):
    if not decision_file:
        raise ValueError('sealed test requires a frozen offline finalist decision')
    decision = json.loads(Path(decision_file).read_text(encoding='utf-8'))
    if decision.get('decision') != 'selected' or not decision.get('validation_input_fingerprint'):
        raise ValueError('invalid offline finalist decision or missing fixed-input provenance')
    allowed = set(decision.get('allowed_checkpoint_sha256', ()))
    if not allowed or any(sha256_file(file) not in allowed for file in checkpoint_files):
        raise ValueError('checkpoint is not frozen in the offline finalist decision')
    return fingerprint(decision)

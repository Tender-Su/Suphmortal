"""Run the bounded corrected critic branch from an immutable source manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest_file = Path(args.manifest).resolve()
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    source = Path(manifest['source_root'])
    for name, expected in manifest['source_sha256'].items():
        if sha256(source / name) != expected:
            raise ValueError('frozen runtime source changed: ' + name)
    for record in (manifest['config'], manifest['initial_checkpoint']):
        if sha256(record['path']) != record['sha256']:
            raise ValueError('frozen runtime input changed: ' + record['path'])
    if args.verify_only:
        print(json.dumps({'verified': True, 'files': len(manifest['source_sha256'])}))
        return 0
    env = os.environ.copy()
    env['MORTAL_CFG'] = manifest['config']['path']
    env['PYTHONPATH'] = str(source)
    result = subprocess.run([sys.executable, '-u', '-m', 'mortal.online.pretrain_oracle_critic'],
                            cwd=source, env=env)
    if result.returncode != 0:
        return result.returncode
    import torch
    state = torch.load(manifest['latest_checkpoint'], map_location='cpu', weights_only=False)
    adaptive = state.get('adaptive_curriculum_state', {})
    report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'steps': state['steps'],
              'adaptive_action': adaptive.get('last_action'), 'best_step': adaptive.get('best_step'),
              'status': 'calibration_branch_finished_requires_independent_qualification',
              'model_published': False, 'sealed_test_opened': False}
    output = manifest_file.parent / 'calibration_result.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

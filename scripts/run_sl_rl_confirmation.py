"""Run a frozen three-candidate maximum 1v3 protocol with resumable chunks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--chunk-seeds', type=int, default=8)
    parser.add_argument('--config')
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    source = Path(__file__).resolve().parents[1]
    if args.config:
        os.environ['MORTAL_CFG'] = str(Path(args.config).resolve(strict=True))

    def command(action, *extra):
        result = subprocess.run([
            sys.executable, '-u', '-m', 'mortal.eval.confirmation_protocol', action,
            '--directory', str(directory), '--device', args.device,
            '--chunk-seeds', str(args.chunk_seeds), *extra,
        ], cwd=source)
        if result.returncode:
            raise SystemExit(result.returncode)

    for name in ('reference', 'reference_repeat'):
        command('run', '--stage', 'aa', '--name', name)
    if not (directory / 'aa_decision.json').exists():
        command('check-aa')
    for name in ('reference', *protocol['candidates']):
        command('run', '--stage', 'screen', '--name', name)
    if not (directory / 'finalist.json').exists():
        command('select')
    selected = json.loads((directory / 'finalist.json').read_text(encoding='utf-8'))
    for name in ('reference', selected['finalist']):
        command('run', '--stage', 'confirmation', '--name', name)
    if not (directory / 'confirmation_decision.json').exists():
        command('decide')
    print((directory / 'confirmation_decision.json').read_text(encoding='utf-8'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

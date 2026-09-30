"""Create a new matched host experiment without mixing cross-runtime observations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.data.split_ledger import game_identity
from scripts.run_sl_curriculum_probe import freeze_search_parser


def path_lookup(paths):
    result = {}
    for path in paths:
        identity = game_identity(path)
        if identity in result and result[identity] != path:
            raise ValueError(f'ambiguous destination game: {identity}')
        result[identity] = path
    return result


def remap_indexes(indexes, lookup):
    def remap(value):
        if isinstance(value, str):
            identity = game_identity(value)
            if identity not in lookup:
                raise ValueError(f'destination is missing game: {identity}')
            return lookup[identity]
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        raise TypeError('unexpected probe index value')
    return remap(indexes)


def identity_digest(paths):
    return stable_json_digest([game_identity(path) for path in paths])


def migrate(args):
    output = Path(args.directory).resolve()
    if output.exists():
        raise FileExistsError('new experiment directory required')
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError('invalid GPU allocation fraction')
    original = json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
    unsigned = dict(original)
    if unsigned.pop('identity') != stable_json_digest(unsigned):
        raise ValueError('source manifest changed')
    for path, expected in ((args.parent, original['parent_sha256']),
                           (args.source_index, original['indexes_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError(f'source input changed: {path}')
    available = torch.load(args.data_index, map_location='cpu', weights_only=True)
    lookup = path_lookup(available['train_files'] + available['val_files'])
    del available
    old_indexes = torch.load(args.source_index, map_location='cpu', weights_only=True)
    indexes = remap_indexes(old_indexes, lookup)
    for group in ('domains', 'roles'):
        for name, paths in old_indexes[group].items():
            if identity_digest(paths) != identity_digest(indexes[group][name]):
                raise ValueError(f'game order changed: {group}/{name}')
    for old, expected in original['controller_source_sha256'].items():
        destination = lookup[game_identity(old)]
        if file_sha256(destination) != expected:
            raise ValueError(f'controller bytes changed: {destination}')
    del old_indexes
    if args.smoke:
        for name in ('controller_recent', 'controller_old'):
            indexes['roles'][name] = indexes['roles'][name][:2]
        indexes['monitor_recent_files'] = indexes['roles']['controller_recent']
        indexes['full_recent_files'] = indexes['roles']['controller_recent']
        indexes['old_regression_files'] = indexes['roles']['controller_old']
    output.mkdir(parents=True)
    runtime = output / 'source'
    for file in (ROOT / 'mortal').rglob('*.py'):
        if 'checkpoints' in file.parts or '__pycache__' in file.parts:
            continue
        target = runtime / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    for name in ('run_sl_curriculum_probe.py', 'start_oracle_critic_detached.ps1',
                 'supervise_oracle_critic_around_apex.ps1'):
        target = runtime / 'scripts' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, target)
    parser = runtime / 'mortal/eval/search_runtime.py'
    parser_hash = file_sha256(parser)
    parser.write_text(freeze_search_parser(parser.read_text(encoding='utf-8')), encoding='utf-8')
    shutil.copy2(args.native, runtime / 'libriichi.pyd')
    shutil.copy2(args.parent, output / 'parent.pth')
    atomic_torch_save(indexes, output / 'indexes.pth')
    manifest = deepcopy(original)
    manifest.pop('identity')
    controller = indexes['roles']['controller_recent'] + indexes['roles']['controller_old']
    manifest.update(parent=str(output / 'parent.pth'), indexes=str(output / 'indexes.pth'),
                    source_root=str(runtime), indexes_sha256=file_sha256(output / 'indexes.pth'),
                    source_sha256={file.relative_to(runtime).as_posix(): file_sha256(file)
                                   for file in runtime.rglob('*') if file.is_file()},
                    controller_source_sha256={file: file_sha256(file) for file in controller},
                    runtime_overrides={'mortal/eval/search_runtime.py': {
                        'main_source_sha256': parser_hash, 'reason': 'frozen keyword-only compatibility fix'}},
                    gpu_memory_fraction=args.gpu_memory_fraction, created_at=time.time(),
                    split_sizes={name: len(paths) for name, paths in indexes['roles'].items()},
                    split_identity_sha256={name: identity_digest(paths)
                                           for name, paths in indexes['roles'].items()},
                    smoke=args.smoke)
    if args.smoke:
        manifest.update(seeds=[20260907], horizons=[1, 2], delayed_transfer_routes=[])
    manifest['migration'] = {
        'source_identity': original['identity'], 'source_manifest_sha256': file_sha256(args.source_manifest),
        'all_domains_and_roles_order_verified': True, 'controller_bytes_verified': True,
        'source_observations_reused': False, 'cross_runtime_exact_resume': False,
        'initialization': 'all matched arms branch anew from identical common parent',
        'deployment_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'torch': torch.__version__,
    }
    manifest['identity'] = stable_json_digest(manifest)
    atomic_write_json(output / 'manifest.json', manifest)
    atomic_write_json(output / 'apex_supervisor_spec.json', {
        'format': 'oracle_critic_apex_supervisor_spec_v1', 'repo_root': str(runtime),
        'python_executable': sys.executable, 'search_root': str(output),
        'pause_file': str(output / 'apex_pause.request'),
        'status_file': str(output / 'apex_supervisor_status.json'),
        'log_file': str(output / 'apex_supervisor.log'),
        'runner_arguments': ['-u', str(runtime / 'scripts/run_sl_curriculum_probe.py'),
                             'run', '--directory', str(output)],
    })
    print(json.dumps({'identity': manifest['identity'], 'directory': str(output),
                      'migration': manifest['migration'], 'smoke': args.smoke}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'source-manifest', 'source-index', 'parent', 'data-index', 'native'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.45)
    parser.add_argument('--smoke', action='store_true')
    migrate(parser.parse_args())

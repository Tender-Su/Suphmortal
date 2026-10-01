"""Relocate one complete SL B/C checkpoint and extend that same phase continuously."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.supervised.continuation import (
    backend_protocol, ledger_snapshot, microbatch_migration, observation_plan, rebind_saved_state, relocate_config, trend_splits,
    validate_inherited_pins, validate_saved_phase,
)
from mortal.supervised.early_transition import parent_record
from scripts.run_sl_early_transition import (
    experiment_lock, fixed_source, fresh_directory, read_json, require_frozen_runtime, verify_manifest,
)


def separate_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('continuation requires a separate new directory; source experiment stays untouched')
    return output


def changed_runtime(original):
    return {name: {'source': digest, 'continuation': file_sha256(ROOT / name)}
            for name, digest in original['source_sha256'].items()
            if name.startswith('mortal/') and '/tests/' not in name and (ROOT / name).is_file()
            and file_sha256(ROOT / name) != digest}


def require_stable_checkpoint(path, expected):
    if file_sha256(path) != expected:
        raise ValueError('source checkpoint changed during preparation; select a complete immutable saved point')


def verify_rebind(source, rebound, *, migration=None):
    from mortal.supervised.curriculum_probe import learned_state_digest
    from scripts.verify_sl_probe_resume import equal

    before = learned_state_digest(source)
    if before != learned_state_digest(rebound):
        raise ValueError('learned/Adam/AMP/scheduler/auxiliary state changed during relocation')
    if migration is not None and (migration != microbatch_migration(
            source, migration['target_microbatch'], migration['reason']) or rebound['steps'] != migration['target_microsteps']):
        raise ValueError('saved microstep clock differs from the declared decision-based conversion')
    for key in source:
        if key == 'steps' and migration is not None:
            continue
        if key not in ('config', 'run_provenance', 'checkpoint_id', 'curriculum_probe') and not equal(source[key], rebound[key]):
            raise ValueError('saved phase state changed: ' + key)
    for key in source['curriculum_probe']:
        if key != 'identity' and not equal(source['curriculum_probe'][key], rebound['curriculum_probe'][key]):
            raise ValueError('consumed cursor/RNG/observation state changed: ' + key)
    return before


def prepare(args):
    import torch
    from mortal.core.toml_utils import write_toml_file
    from mortal.supervised.curriculum_probe import RECIPES

    source_root = Path(args.source_run).resolve()
    output = separate_output(source_root, args.directory)
    commit = fixed_source(args.source_commit)
    original = verify_manifest(source_root)
    differences = changed_runtime(original)
    if differences and not args.runtime_change_reason:
        raise ValueError('training runtime changed; inspect differences and explicitly document qualification before relocation: '
                         + ', '.join(differences))
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_sha = file_sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('run_provenance', {}).get('experiment_id') != original['identity']:
        raise ValueError('checkpoint does not belong to the selected source experiment')
    source_index = Path(state['config']['supervised']['file_index'])
    if file_sha256(source_index) != original['input_sha256']['indexes.pth']:
        raise ValueError('checkpoint index differs from the frozen experiment index')
    index = torch.load(source_index, map_location='cpu', weights_only=True)
    contract = validate_saved_phase(state, index['domains'], RECIPES)
    migration = microbatch_migration(state, getattr(args, 'microbatch', None),
                                    getattr(args, 'microbatch_change_reason', ''))
    backend = backend_protocol(getattr(args, 'backend', 'inherit'), getattr(args, 'torch_threads', None),
                               getattr(args, 'backend_change_reason', ''))
    if backend is None:
        history = state['config']['supervised']['run_provenance'].get('backend_protocols', [])
        if history:
            previous = history[-1]
            backend = backend_protocol(previous['mode'], previous['torch_threads'], previous['reason'])
            if backend != previous:
                raise ValueError('inherited backend protocol is not canonical')
    require_stable_checkpoint(checkpoint, checkpoint_sha)
    plan = observation_plan(start=contract['optimizer_updates'], until=args.until_update,
        save_updates=args.save_every_updates, save_seconds=args.save_every_seconds,
        trend_every=args.trend_every_updates, full_every=args.full_every_updates)
    trend = trend_splits(index['roles'], recent_games=args.trend_recent_games,
                         old_games=args.trend_old_games, seed=args.trend_seed)
    if index.get('trend_roles') is not None and index['trend_roles'] != trend:
        raise ValueError('continuing the same phase must retain its existing fixed trend panel')
    parent = parent_record(state, path=output / 'parent.pth', sha256=checkpoint_sha, phase=contract['phase'])
    if args.check_only:
        print(json.dumps({'state_contract': contract, 'plan': plan, 'runtime_differences': differences,
                          'microbatch_migration': migration, 'backend_protocol': backend,
                          'status': 'checkpoint_structure_only; ledger/copy/equality not yet verified'}))
        return
    fresh_directory(output)
    shutil.copy2(checkpoint, output / 'parent.pth')
    require_stable_checkpoint(checkpoint, checkpoint_sha)
    require_stable_checkpoint(output / 'parent.pth', checkpoint_sha)
    consumed_files = {state['curriculum_probe']['dataset']['files'][key]['file']
                      for key in state['curriculum_probe']['dataset']['consumed']}
    current_draws = (state['curriculum_probe']['dataset']['current'] or {}).get('draws', [])
    ledger = ledger_snapshot(state['config']['supervised']['probe_training_content_ledger'],
                             output / 'inherited_content.sqlite3', original['identity'], consumed_files,
                             current_hashes={row['file']: row['source_sha256'] for row in current_draws})
    index['trend_roles'] = trend
    atomic_torch_save(index, output / 'indexes.pth')
    atomic_write_json(output / 'source_config.json', state['config'])
    runtime = output / 'source'
    files = list((ROOT / 'mortal').rglob('*.py')) + [ROOT / 'scripts' / name for name in
        ('continue_sl_phase.py', 'run_sl_early_transition.py', 'verify_sl_probe_resume.py')]
    for path in files:
        if '__pycache__' in path.parts or 'checkpoints' in path.parts:
            continue
        target = runtime / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    native_names = [name for name in original['source_sha256']
                    if Path(name).name.startswith('libriichi') and Path(name).suffix in ('.so', '.pyd')]
    if not native_names:
        raise ValueError('source experiment lacks a frozen native extension')
    for name in native_names:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(original['source_root']) / name, target)
    manifest = {'format': 'sl_phase_continuation_v1', 'directory': str(output), 'source_root': str(runtime),
        'source_git_commit': commit, 'source_sha256': {p.relative_to(runtime).as_posix(): file_sha256(p)
            for p in runtime.rglob('*') if p.is_file()}, 'parents': {'same_phase': parent},
        'input_sha256': {name: file_sha256(output / name) for name in
                         ('indexes.pth', 'inherited_content.sqlite3', 'source_config.json')},
        'controller_source_sha256': original['controller_source_sha256'],
        'training_content': 'inherited all source pins; append-only new first-consumption pins',
        'phase': contract['phase'], 'arm': contract['arm'], 'plan': plan, 'source_state_contract': contract,
        'gpu_memory_fraction': original['gpu_memory_fraction'],
        'trend_panel': {'recent_games': args.trend_recent_games, 'old_games': args.trend_old_games, 'seed': args.trend_seed,
                        'identity': stable_json_digest(trend)},
        'source_experiment': {'identity': original['identity'], 'directory': str(source_root),
            'manifest_sha256': file_sha256(source_root / 'manifest.json'), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': checkpoint_sha}, 'ledger_snapshot': ledger,
        'runtime_differences': differences, 'runtime_change_reason': args.runtime_change_reason,
        'microbatch_migration': migration, 'backend_protocol': backend,
        'resume_scope': ('declared microbatch numerical branch; preserved learned/data/RNG state, converted microstep units'
                         if migration else ('declared backend numerical branch; all learned/data/RNG state preserved'
                                            if backend else 'exact saved-state relocation; not a cross-runtime bitwise trajectory proof')),
        'runtime_versions': {'torch': torch.__version__, 'python': sys.version},
        'phase_transfer': 'none; B-to-C requires a separate explicitly declared phase plan', 'created_at': time.time()}
    manifest['identity'] = stable_json_digest(manifest)
    shutil.copy2(output / 'inherited_content.sqlite3', output / 'training_content.sqlite3')
    with closing(sqlite3.connect(output / 'training_content.sqlite3')) as db, db:
        db.execute('UPDATE metadata SET identity=?', (manifest['identity'],))
    config = relocate_config(state['config'], output, manifest['identity'], parent, commit, original['identity'],
                             runtime_sha256=stable_json_digest(manifest['source_sha256']), migration=migration, backend=backend)
    rebound = rebind_saved_state(state, config, manifest['identity'], migration=migration, backend=backend)
    learned_digest = verify_rebind(state, rebound, migration=migration)
    rebound_contract = validate_saved_phase(rebound, index['domains'], RECIPES)
    write_toml_file(output / 'config.toml', config)
    atomic_torch_save(rebound, output / 'state_file.pth')
    require_stable_checkpoint(checkpoint, checkpoint_sha)
    atomic_write_json(output / 'manifest.json', manifest)
    verify_manifest(output)
    receipt = {'identity': manifest['identity'], 'state_contract': contract,
        'parent_sha256': checkpoint_sha, 'relocated_sha256': file_sha256(output / 'state_file.pth'),
        'learned_state_sha256': learned_digest, 'microbatch_migration': migration, 'backend_protocol': backend,
        'relocated_state_contract': rebound_contract,
        'all_saved_state_preserved_except_declared_metadata': migration is None and backend is None,
        'all_saved_state_preserved_except_metadata_and_declared_microstep_conversion': True,
        'ledger_snapshot': ledger, 'no_source_files_modified': True}
    atomic_write_json(output / 'continuation_receipt.json', receipt)
    print(json.dumps(receipt))


def run(args):
    manifest = verify_manifest(args.directory)
    require_frozen_runtime(manifest)
    import torch
    from mortal.core.toml_utils import load_toml_file
    from mortal.supervised.continuous_probe import ContinuousPhaseProbe

    output = Path(manifest['directory'])
    if (output / 'sealed.json').exists() and not args.seal:
        raise ValueError('sealed phase requires another explicit continuation experiment')
    config = load_toml_file(output / 'config.toml')
    expected_config = relocate_config(read_json(output / 'source_config.json'), output, manifest['identity'],
        manifest['parents']['same_phase'], manifest['source_git_commit'], manifest['source_experiment']['identity'],
        runtime_sha256=stable_json_digest(manifest['source_sha256']), migration=manifest.get('microbatch_migration'),
        backend=manifest.get('backend_protocol'))
    if config != expected_config:
        raise ValueError('continuation configuration differs from the declared relocation/numerical protocol')
    state = torch.load(output / 'state_file.pth', map_location='cpu', weights_only=False)
    if (state['config'] != config or state['run_provenance']['plan_id'] != manifest['identity']
            or state['run_provenance'] != config['supervised']['run_provenance']):
        raise ValueError('continuation config/provenance mismatch')
    index = torch.load(output / 'indexes.pth', map_location='cpu', weights_only=True)
    from mortal.supervised.curriculum_probe import RECIPES
    validate_saved_phase(state, index['domains'], RECIPES)
    data = state['curriculum_probe']['dataset']
    validate_inherited_pins(output / 'inherited_content.sqlite3', output / 'training_content.sqlite3',
                             manifest['identity'], (data['files'][key]['file'] for key in data['consumed']))
    seal_at = state['optimizer_steps'] if args.seal else None
    del state
    backend = manifest.get('backend_protocol')
    threads = backend['torch_threads'] if backend else 2
    os.environ.update(MORTAL_CFG=str(output / 'config.toml'),
        RAYON_NUM_THREADS=str(config['supervised']['rayon_num_threads']),
        OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(backend['deterministic'] if backend else True)
    # Preserve the source allocator cap; do not silently remove its capacity bound.
    if config['control']['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_fraction'])
    probe = ContinuousPhaseProbe(config, index['domains'], recipe=manifest['phase'],
        seed=config['supervised']['seed'], output=output, identity=manifest['identity'],
        plan=manifest['plan'], full_splits=index['roles'], trend_splits=index['trend_roles'], seal_at=seal_at)
    from mortal.supervised.train_supervised import sanitize_sys_path_for_spawn, train
    sanitize_sys_path_for_spawn()
    train(probe=probe, stage_label=f'Continuous {manifest["arm"]} {manifest["phase"]}',
          checkpoint_label=f'{manifest["arm"]}_{manifest["phase"]}_continuation')
    latest = torch.load(output / 'state_file.pth', map_location='cpu', weights_only=False)
    if latest['optimizer_steps'] != probe.stop_at or probe.stop_at not in probe.done['full']:
        raise ValueError('continuous phase returned before its complete full-evaluated endpoint')
    receipt = {'identity': manifest['identity'], 'optimizer_updates': probe.stop_at,
               'checkpoint': str(output / 'state_file.pth'), 'checkpoint_sha256': file_sha256(output / 'state_file.pth'),
               'phase': manifest['phase'], 'automatic_phase_transition': False, 'completed_at': time.time()}
    atomic_write_json(output / ('sealed.json' if args.seal else 'completed.json'), receipt)
    print(json.dumps(receipt))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for name in ('source-run', 'checkpoint', 'directory', 'source-commit'):
        prep.add_argument('--' + name, required=True)
    for name in ('until-update', 'save-every-updates', 'trend-every-updates', 'full-every-updates',
                 'trend-recent-games', 'trend-old-games', 'trend-seed'):
        prep.add_argument('--' + name, type=int, required=True)
    prep.add_argument('--save-every-seconds', type=float, required=True)
    prep.add_argument('--runtime-change-reason', default='')
    prep.add_argument('--microbatch', type=int, choices=(256, 512),
                      help='Explicit 256<->512 numerical branch; retains logical batch1024 and consumed cursor')
    prep.add_argument('--microbatch-change-reason', default='')
    prep.add_argument('--backend', choices=('inherit', 'strict', 'fast'), default='inherit')
    prep.add_argument('--torch-threads', type=int, choices=(1, 2, 4))
    prep.add_argument('--backend-change-reason', default='')
    prep.add_argument('--check-only', action='store_true')
    execute = commands.add_parser('run')
    execute.add_argument('--directory', required=True)
    execute.add_argument('--seal', action='store_true', help='Full-evaluate and seal latest without a training update')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        with experiment_lock(args.directory, '.training.lock'):
            run(args)

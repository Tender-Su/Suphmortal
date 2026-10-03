"""Prepare/run a single-parent B-to-C fork without weakening existing runners."""
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
from mortal.supervised.continuation import ledger_snapshot, validate_inherited_pins, validate_saved_phase
from mortal.supervised.early_transition import LEARNED_KEYS, parent_record, phase_spec, prepare_transition_state
from scripts.continue_sl_phase import changed_runtime, require_stable_checkpoint, separate_output
from scripts.run_sl_early_transition import (
    experiment_lock, fixed_source, fresh_directory, read_json, require_frozen_runtime, verify_manifest,
)


def prepare(args):
    import torch
    from mortal.core.toml_utils import write_toml_file
    from mortal.supervised.curriculum_probe import RECIPES, learned_state_digest
    from mortal.supervised.phase_fork import fork_config, validate_baseline
    from scripts.verify_sl_probe_resume import equal

    torch.set_num_threads(1)
    source_root = Path(args.source_run).resolve()
    output = separate_output(source_root, args.directory)
    commit = fixed_source(args.source_commit)
    original = verify_manifest(source_root)
    if changed_runtime(original):
        raise ValueError('fork requires the parent training/evaluation runtime unchanged')
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_sha = file_sha256(checkpoint)
    completion = read_json(source_root / 'completed.json')
    if (completion['identity'] != original['identity'] or completion['phase'] != 'B'
            or Path(completion['checkpoint']).resolve() != checkpoint
            or completion['checkpoint_sha256'] != checkpoint_sha):
        raise ValueError('parent must be its own completed B latest endpoint')
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    source_index = Path(source['config']['supervised']['file_index'])
    if file_sha256(source_index) != original['input_sha256']['indexes.pth']:
        raise ValueError('parent index differs from the frozen manifest')
    index = torch.load(source_index, map_location='cpu', weights_only=True)
    contract = validate_saved_phase(source, index['domains'], RECIPES)
    if (contract['phase'] != 'B' or contract['optimizer_updates'] != completion['optimizer_updates']
            or source['run_provenance']['experiment_id'] != original['identity']):
        raise ValueError('parent state does not match the completed B receipt')
    seed_path = Path(args.seed_manifest).resolve()
    seed_manifest = read_json(seed_path)
    unsigned = dict(seed_manifest)
    if stable_json_digest({k: v for k, v in unsigned.items() if k != 'identity'}) != unsigned['identity']:
        raise ValueError('seed manifest identity changed')
    chain = source['run_provenance'].get('parent_chain', [])
    origin = seed_manifest['parents'][contract['arm']]
    if not chain or any(chain[0].get(k) != origin.get(k) for k in ('phase', 'checkpoint_id', 'sha256')):
        raise ValueError('C seed must come from the original A-parent experiment')
    spec = phase_spec(updates=args.updates, observations=args.observations,
                      seed=seed_manifest['phases']['C']['seed'], peak=args.lr, init=args.lr, warmup=0)
    cfg = source['config']
    if (cfg['supervised']['batch_size'], cfg['control']['opt_step_every']) != (512, 2):
        raise ValueError('this fork inherits the validated 512x2 lane; no batch migration')
    if contract['next_update_lrs'] != [args.lr] * len(source['optimizer']['param_groups']):
        raise ValueError('fork LR must equal the parent effective constant LR')
    if {k: len(v) for k, v in index['roles'].items()} != {'controller_recent': 512, 'controller_old': 256}:
        raise ValueError('fork requires the existing full recent512/old256 panel')
    baseline_path = Path(args.baseline).resolve() if args.baseline else None
    baseline = read_json(baseline_path) if baseline_path else None
    if baseline is not None:
        validate_baseline(baseline, source, stable_json_digest(index['roles']))
        if file_sha256(baseline['checkpoint']) != baseline['checkpoint_sha256']:
            raise ValueError('parent observation checkpoint changed')
    require_stable_checkpoint(checkpoint, checkpoint_sha)
    fresh_directory(output)
    shutil.copy2(checkpoint, output / 'parent.pth')
    require_stable_checkpoint(output / 'parent.pth', checkpoint_sha)
    shutil.copy2(source_index, output / 'indexes.pth')
    shutil.copy2(seed_path, output / 'seed_manifest.json')
    atomic_write_json(output / 'source_config.json', cfg)
    data = source['curriculum_probe']['dataset']
    ledger = ledger_snapshot(cfg['supervised']['probe_training_content_ledger'],
        output / 'inherited_content.sqlite3', original['identity'],
        (data['files'][key]['file'] for key in data['consumed']),
        current_hashes={row['file']: row['source_sha256'] for row in (data['current'] or {}).get('draws', [])})
    if baseline_path:
        shutil.copy2(baseline_path, output / 'parent_observation.json')
    runtime = output / 'source'
    for name, digest in original['source_sha256'].items():
        source_file = Path(original['source_root']) / name
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        if file_sha256(target) != digest:
            raise ValueError('runtime copy changed: ' + name)
    for name in ('scripts/fork_sl_phase.py', 'mortal/supervised/phase_fork.py'):
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    parent = parent_record(source, path=output / 'parent.pth', sha256=checkpoint_sha, phase='B')
    inputs = ['indexes.pth', 'inherited_content.sqlite3', 'source_config.json', 'seed_manifest.json']
    if baseline_path:
        inputs.append('parent_observation.json')
    manifest = {'format': 'sl_single_parent_phase_fork_v1', 'directory': str(output),
        'source_root': str(runtime), 'source_git_commit': commit,
        'source_sha256': {p.relative_to(runtime).as_posix(): file_sha256(p)
                          for p in runtime.rglob('*') if p.is_file()},
        'input_sha256': {name: file_sha256(output / name) for name in inputs},
        'parents': {'single_B': parent}, 'phase': 'C', 'arm': contract['arm'], 'phase_spec': spec,
        'controller_source_sha256': original['controller_source_sha256'],
        'training_content': 'inherited all B pins; append new first-consumption C pins',
        'ledger_snapshot': ledger, 'source_state_contract': contract,
        'gpu_memory_fraction': original['gpu_memory_fraction'],
        'backend_protocol': original.get('backend_protocol'),
        'source_experiment': {'identity': original['identity'], 'directory': str(source_root),
            'manifest_sha256': file_sha256(source_root / 'manifest.json'),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha},
        'save_updates': 500, 'save_seconds': 300, 'baseline_reused': baseline is not None,
        'baseline_source': str(baseline_path) if baseline_path else None,
        'phase_transfer': 'explicit single completed B parent; no balanced-arm claim',
        'created_at': time.time(), 'automatic_extension': False, 'automatic_promotion': False}
    manifest['identity'] = stable_json_digest(manifest)
    shutil.copy2(output / 'inherited_content.sqlite3', output / 'training_content.sqlite3')
    with closing(sqlite3.connect(output / 'training_content.sqlite3')) as db, db:
        db.execute('UPDATE metadata SET identity=?', (manifest['identity'],))
    config = fork_config(cfg, output, manifest['identity'], parent, commit,
                         stable_json_digest(manifest['source_sha256']), spec)
    state = prepare_transition_state(source, config)
    for key in LEARNED_KEYS:
        expected = source[key]
        if key == 'optimizer':
            from copy import deepcopy
            expected = deepcopy(expected)
            for group, forked in zip(expected['param_groups'], state[key]['param_groups']):
                group['lr'], group['initial_lr'] = forked['lr'], forked['initial_lr']
        if not equal(expected, state[key]):
            raise ValueError('fork lost learned state: ' + key)
    if state['auxiliary_optimizer_steps'] != source['auxiliary_optimizer_steps']:
        raise ValueError('fork reset the mature auxiliary clock')
    write_toml_file(output / 'config.toml', config)
    atomic_torch_save(state, output / 'state_file.pth')
    atomic_write_json(output / 'manifest.json', manifest)
    require_stable_checkpoint(checkpoint, checkpoint_sha)
    verify_manifest(output)
    receipt = {'identity': manifest['identity'], 'source_commit': commit, 'parent': parent,
        'prepared_checkpoint_sha256': file_sha256(output / 'state_file.pth'), 'source_contract': contract,
        'phase_spec': spec, 'learned_heads_adam_scaler_preserved': True,
        'cumulative_auxiliary_clock': state['auxiliary_optimizer_steps'],
        'parent_learned_state_sha256': learned_state_digest(source),
        'C0_learned_state_sha256': learned_state_digest(state), 'ledger_snapshot': ledger,
        'adam_steps': {str(key): float(value['step']) for key, value in state['optimizer']['state'].items()},
        'baseline_reused': baseline is not None, 'baseline_source': manifest['baseline_source'],
        'parent_chain': config['supervised']['run_provenance']['parent_chain'],
        'source_files_unchanged': True, 'cuda_initialized': torch.cuda.is_initialized()}
    atomic_write_json(output / 'fork_receipt.json', receipt)
    print(json.dumps(receipt))


def run(args):
    import torch
    from mortal.core.toml_utils import load_toml_file
    from mortal.supervised.curriculum_probe import RECIPES
    from mortal.supervised.phase_fork import PhaseForkProbe, fork_config

    manifest = verify_manifest(args.directory)
    require_frozen_runtime(manifest)
    output = Path(manifest['directory'])
    if (output / 'completed.json').exists():
        raise ValueError('this C endpoint is already complete; no automatic extension')
    config = load_toml_file(output / 'config.toml')
    expected = fork_config(read_json(output / 'source_config.json'), output, manifest['identity'],
        manifest['parents']['single_B'], manifest['source_git_commit'],
        stable_json_digest(manifest['source_sha256']), manifest['phase_spec'])
    if config != expected:
        raise ValueError('fork config differs from the declared transition')
    state = torch.load(output / 'state_file.pth', map_location='cpu', weights_only=False)
    if state['config'] != config or state['run_provenance'] != config['supervised']['run_provenance']:
        raise ValueError('saved C state has different configuration or provenance')
    index = torch.load(output / 'indexes.pth', map_location='cpu', weights_only=True)
    if state['optimizer_steps']:
        validate_saved_phase(state, index['domains'], RECIPES)
        data = state['curriculum_probe']['dataset']
        consumed = (data['files'][key]['file'] for key in data['consumed'])
    else:
        if 'curriculum_probe' not in state:
            require_stable_checkpoint(output / 'state_file.pth',
                                      read_json(output / 'fork_receipt.json')['prepared_checkpoint_sha256'])
        consumed = ()
    validate_inherited_pins(output / 'inherited_content.sqlite3', output / 'training_content.sqlite3',
                           manifest['identity'], consumed)
    del state
    backend = manifest.get('backend_protocol')
    threads = backend['torch_threads'] if backend else 2
    os.environ.update(MORTAL_CFG=str(output / 'config.toml'),
        RAYON_NUM_THREADS=str(config['supervised']['rayon_num_threads']),
        OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(backend['deterministic'] if backend else True)
    if config['control']['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_fraction'])
    baseline = read_json(output / 'parent_observation.json') if manifest['baseline_reused'] else None
    probe = PhaseForkProbe(config, index['domains'], recipe='C', seed=manifest['phase_spec']['seed'],
        output=output / 'full', identity=manifest['identity'], horizons=manifest['phase_spec']['observations'],
        eval_splits=index['roles'], baseline=baseline, save_updates=manifest['save_updates'],
        save_seconds=manifest['save_seconds'])
    from mortal.supervised.train_supervised import sanitize_sys_path_for_spawn, train
    sanitize_sys_path_for_spawn()
    train(probe=probe, stage_label=f'Single-parent {manifest["arm"]} C', checkpoint_label='early_C_fork')
    latest = torch.load(output / 'state_file.pth', map_location='cpu', weights_only=False)
    final_contract = validate_saved_phase(latest, index['domains'], RECIPES)
    if final_contract['optimizer_updates'] != manifest['phase_spec']['updates']:
        raise ValueError('C returned before the declared successful-update endpoint')
    initial = read_json(output / 'fork_receipt.json')
    updates = final_contract['optimizer_updates']
    if latest['auxiliary_optimizer_steps'] != initial['cumulative_auxiliary_clock'] + updates:
        raise ValueError('C cumulative auxiliary clock did not advance with successful updates')
    expected_adam = {key: value + updates for key, value in initial['adam_steps'].items()}
    actual_adam = {str(key): float(value['step']) for key, value in latest['optimizer']['state'].items()}
    if actual_adam != expected_adam:
        raise ValueError('C Adam clocks did not advance with successful updates')
    data = latest['curriculum_probe']['dataset']
    validate_inherited_pins(output / 'inherited_content.sqlite3', output / 'training_content.sqlite3',
        manifest['identity'], (data['files'][key]['file'] for key in data['consumed']))
    for update in [0, *manifest['phase_spec']['observations']]:
        row = read_json(output / 'full' / f'update_{update:07d}.json')
        if (row['identity'], row['optimizer_updates']) != (manifest['identity'], update):
            raise ValueError('missing or foreign C observation')
    receipt = {'identity': manifest['identity'], 'optimizer_updates': final_contract['optimizer_updates'],
        'checkpoint': str(output / 'state_file.pth'), 'checkpoint_sha256': file_sha256(output / 'state_file.pth'),
        'phase': 'C', 'state_contract': final_contract, 'completed_at': time.time(),
        'parent_Adam_and_auxiliary_clock_advance_verified': True, 'inherited_ledger_pins_verified': True,
        'automatic_phase_transition': False, 'automatic_promotion': False}
    atomic_write_json(output / 'completed.json', receipt)
    print(json.dumps(receipt))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('source-run', 'checkpoint', 'seed-manifest', 'directory', 'source-commit'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--baseline')
    prep.add_argument('--updates', type=int, required=True)
    prep.add_argument('--observations', type=int, nargs='+', required=True)
    prep.add_argument('--lr', type=float, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--directory', required=True)
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.command == 'prepare':
        prepare(arguments)
    else:
        with experiment_lock(arguments.directory, '.training.lock'):
            run(arguments)

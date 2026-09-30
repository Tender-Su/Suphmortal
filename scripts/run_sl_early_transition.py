"""Prepare and run two matched early/late A→B→C branches in a frozen runtime."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.supervised.early_transition import (
    ARMS, PHASES, LEARNED_KEYS, TrainingContentLedger, observation_schedule, parent_record, phase_spec,
    prepare_transition_state, validate_corrected_recipe, validate_parent,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fresh_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


@contextmanager
def experiment_lock(directory, name):
    """OS-owned lock releases on crashes; never unlink an active lock inode."""
    with (Path(directory) / name).open('a+b') as stream:
        stream.write(b'0')
        stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('experiment already has an active runner/training process') from error
        yield


def fixed_source(commit):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    actual = git('rev-parse', 'HEAD')
    if commit != actual or git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('prepare requires a clean fixed Git checkout and its full HEAD commit')
    return actual


def branch_provenance(source, manifest, arm, phase, parent):
    spec = manifest['phases'][phase]
    chain = deepcopy(source.get('run_provenance', {}).get('parent_chain', [])) if phase == 'C' else []
    if phase == 'C' and (len(chain) != 1 or chain[0]['phase'] != 'A'):
        raise ValueError('B checkpoint lost its original A parent chain')
    chain.append(parent)
    return {'plan_id': stable_json_digest({'experiment': manifest['identity'], 'arm': arm,
                                                'phase': phase, 'parent': parent}),
                  'experiment_id': manifest['identity'], 'arm': arm, 'phase': phase,
                  'branch_mode': 'preserve_adam_declared_phase_lr',
                  'source_git_commit': manifest['source_git_commit'],
                  'source_runtime_sha256': stable_json_digest(manifest['source_sha256']),
                  'parent_checkpoint_id': source['checkpoint_id'], 'parent_chain': chain,
                  'source_lrs': parent['source_lrs'], 'phase_scheduler': spec['scheduler'],
                  'reset': ['microsteps', 'optimizer_updates', 'scheduler_clock', 'metrics',
                            'controllers', 'data_cursor', 'rng'],
                  'preserve': ['all_learned_heads', 'adam_moments_and_steps', 'parameter_mapping',
                               'adam_betas_eps_weight_decay', 'amp', 'cumulative_auxiliary_clock'],
                  'initialization': 'experimental_branch_not_exact_parent_resume'}


def apply_phase_lr(config, spec, provenance):
    config['supervised'].update(lr=spec['scheduler']['peak'], scheduler=deepcopy(spec['scheduler']),
                                run_provenance=deepcopy(provenance))
    # train_supervised reads supervised.lr with optim.scheduler.peak fallback.
    config['optim']['scheduler'] = deepcopy(spec['scheduler'])
    return config


def build_phase_config(source, template, manifest, arm, phase, output, parent):
    from scripts.run_sl_curriculum_probe import build_config

    spec = manifest['phases'][phase]
    provenance = branch_provenance(source, manifest, arm, phase, parent)
    # Both arms use one explicit configuration template, never their historical
    # controller settings. Adam group options are inherited from each checkpoint.
    config = build_config({'config': template, 'optimizer': source['optimizer']}, output,
                          Path(manifest['directory']) / 'indexes.pth', seed=spec['seed'],
                          device=manifest['device'], microbatch=manifest['microbatch'],
                          logical_batch=manifest['logical_batch'], identity=provenance['plan_id'],
                          runtime_performance=manifest['runtime_performance'])
    config['supervised']['val_batch_size'] = manifest['val_batch_size']
    config['supervised']['probe_training_content_ledger'] = str(Path(manifest['directory']) / 'training_content.sqlite3')
    return apply_phase_lr(config, spec, provenance)


def matching_parents(early, late):
    if validate_corrected_recipe(early['config']) != validate_corrected_recipe(late['config']):
        raise ValueError('early and late parents have different full auxiliary recipes')
    if early['optimizer_param_groups'] != late['optimizer_param_groups']:
        raise ValueError('early and late Adam parameter mappings differ')
    for left, right in zip(early['optimizer']['param_groups'], late['optimizer']['param_groups']):
        options = lambda group: {k: v for k, v in group.items() if k not in ('lr', 'initial_lr', 'params')}
        if options(left) != options(right):
            raise ValueError('early and late Adam hyperparameters differ')
    for name in LEARNED_KEYS[:5]:
        signature = lambda state: {key: (tuple(value.shape), str(value.dtype))
                                   for key, value in state[name].items()}
        if signature(early) != signature(late):
            raise ValueError(f'early and late learned architecture differs: {name}')


def prepare(args):
    import torch
    from mortal.data.split_ledger import game_identity
    from mortal.supervised.curriculum_probe import RECIPES, make_domains

    output = Path(args.directory).resolve()
    if output.exists():
        raise FileExistsError('new experiment directory required; old outputs are immutable')
    commit = fixed_source(args.source_commit)
    phases = {phase: phase_spec(updates=getattr(args, f'{phase.lower()}_updates'),
              observations=getattr(args, f'{phase.lower()}_observations'),
              seed=getattr(args, f'{phase.lower()}_seed'), peak=getattr(args, f'{phase.lower()}_lr'),
              init=getattr(args, f'{phase.lower()}_init_lr'), warmup=getattr(args, f'{phase.lower()}_warmup'))
              for phase in PHASES}
    if (args.microbatch <= 0 or args.logical_batch <= 0 or args.val_batch_size <= 0
            or args.logical_batch % args.microbatch):
        raise ValueError('positive logical batch must be divisible by microbatch')
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError('GPU memory fraction must be in (0, 1]')
    parent_hashes = {arm: file_sha256(getattr(args, arm)) for arm in ARMS}
    parents = {arm: torch.load(getattr(args, arm), map_location='cpu', weights_only=False) for arm in ARMS}
    for arm, state in parents.items():
        validate_parent(state, arm=arm)
    matching_parents(parents['early'], parents['late'])
    inventory = torch.load(args.index, map_location='cpu', weights_only=True)
    domains = make_domains(sorted(set(inventory['train_files'] + inventory.get('val_files', []))))
    evaluation = torch.load(args.validation_index, map_location='cpu', weights_only=True)
    roles = {'controller_recent': sorted(evaluation['full_recent_files']),
             'controller_old': sorted(evaluation['old_regression_files'])}
    train_files = sorted(set(file for files in domains.values() for file in files))
    identities = {name: [game_identity(f) for f in files] for name, files in roles.items()}
    all_eval = [item for files in identities.values() for item in files]
    if (any(not files for files in roles.values()) or len(all_eval) != len(set(all_eval))
            or set(all_eval) & {game_identity(f) for f in train_files}):
        raise ValueError('nonempty disjoint validation roles outside all training domains required')
    eval_hashes = {file: file_sha256(file) for files in roles.values() for file in files}
    fresh_directory(output)
    parent_info = {}
    for arm, state in parents.items():
        target = output / f'{arm}_A.pth'
        shutil.copy2(getattr(args, arm), target)
        if file_sha256(target) != parent_hashes[arm]:
            raise ValueError('A parent changed during preparation')
        parent_info[arm] = parent_record(state, path=target, sha256=file_sha256(target), phase='A')
    atomic_torch_save({'train_files': train_files, 'domains': domains, 'roles': roles,
                       'monitor_recent_files': roles['controller_recent'],
                       'full_recent_files': roles['controller_recent'],
                       'old_regression_files': roles['controller_old']}, output / 'indexes.pth')
    atomic_write_json(output / 'template.json', parents['late']['config'])
    runtime = output / 'source'
    files = list((ROOT / 'mortal').rglob('*.py')) + [ROOT / 'scripts' / name for name in
            ('run_sl_early_transition.py', 'run_sl_curriculum_probe.py')]
    for file in files:
        if '__pycache__' in file.parts or 'checkpoints' in file.parts:
            continue
        target = runtime / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    native = Path(args.native).resolve()
    if not native.name.startswith('libriichi') or native.suffix not in ('.so', '.pyd'):
        raise ValueError('native must be the explicit libriichi extension')
    shutil.copy2(native, runtime / native.name)
    manifest = {'format': 'sl_early_transition_v1', 'directory': str(output), 'parents': parent_info,
                'original_parent_paths': {arm: str(Path(getattr(args, arm)).resolve()) for arm in ARMS},
                'phases': phases, 'recipes': {phase: RECIPES[phase] for phase in PHASES},
                'source_git_commit': commit, 'source_root': str(runtime),
                'source_sha256': {file.relative_to(runtime).as_posix(): file_sha256(file)
                                  for file in runtime.rglob('*') if file.is_file()},
                'input_sha256': {name: file_sha256(output / name) for name in ('indexes.pth', 'template.json')},
                'controller_source_sha256': eval_hashes,
                'validation_identity': {name: stable_json_digest(ids) for name, ids in identities.items()},
                'domain_sizes': {name: len(files) for name, files in domains.items()},
                'microbatch': args.microbatch, 'logical_batch': args.logical_batch,
                'val_batch_size': args.val_batch_size, 'device': args.device,
                'gpu_memory_fraction': args.gpu_memory_fraction,
                'runtime_performance': {'probe_prepare_file_batch_size': args.prepare_file_batch_size,
                    'probe_prepare_workers': args.prepare_workers, 'val_prepare_workers': args.val_prepare_workers,
                    'prepare_rayon_threads': args.rayon_threads, 'val_file_batch_size': args.val_file_batch_size,
                    'rayon_num_threads': args.rayon_threads},
                'baseline': 'fresh phase-local evaluations; historical B/C losses are not comparable',
                'selection': 'descriptive paired early/late observations; no inherited veto or auto-promotion',
                'phase_transfer': 'own B latest endpoint to C; never historical best rollback',
                'training_content': 'shared append-only first-consumption SHA256 ledger; required on resume',
                'created_at': time.time()}
    manifest['identity'] = stable_json_digest(manifest)
    TrainingContentLedger(output / 'training_content.sqlite3', manifest['identity'], create=True)
    # Validate both initial configs before marking the preparation usable.
    for arm in ARMS:
        build_phase_config(parents[arm], parents['late']['config'], manifest, arm, 'B',
                           output / f'{arm}_B', parent_info[arm])
    atomic_write_json(output / 'manifest.json', manifest)
    print(json.dumps({'identity': manifest['identity'], 'schedule': observation_schedule(phases)}))


def verify_manifest(directory):
    output = Path(directory).resolve()
    manifest = read_json(output / 'manifest.json')
    unsigned = dict(manifest)
    identity = unsigned.pop('identity')
    if unsigned['directory'] != str(output) or stable_json_digest(unsigned) != identity:
        raise ValueError('experiment manifest identity or location changed')
    checks = [(output / name, digest) for name, digest in manifest['input_sha256'].items()]
    checks += [(Path(manifest['source_root']) / name, digest) for name, digest in manifest['source_sha256'].items()]
    checks += [(row['checkpoint'], row['sha256']) for row in manifest['parents'].values()]
    checks += list(manifest['controller_source_sha256'].items())
    for path, digest in checks:
        if file_sha256(path) != digest:
            raise ValueError(f'frozen experiment input changed: {path}')
    if manifest.get('training_content'):
        TrainingContentLedger(output / 'training_content.sqlite3', identity)
    return manifest


def phase_parent(manifest, arm, phase):
    if phase == 'B':
        return manifest['parents'][arm]
    # The direct phase CLI obeys the same balanced B→C gate as run().
    for other in ARMS:
        receipt = read_json(Path(manifest['directory']) / f'{other}_B' / 'completed.json')
        if (receipt['experiment_id'], receipt['arm'], receipt['phase'], receipt['until_update']) != (
                manifest['identity'], other, 'B', manifest['phases']['B']['updates']):
            raise ValueError('C requires both declared B endpoints to be completed')
    root = Path(manifest['directory']) / f'{arm}_B'
    receipt = read_json(root / 'completed.json')
    if receipt['until_update'] != manifest['phases']['B']['updates']:
        raise ValueError('C requires its own completed B endpoint')
    parent = receipt['endpoint']
    if parent['checkpoint'] != str(root / 'state_file.pth') or file_sha256(parent['checkpoint']) != parent['sha256']:
        raise ValueError('B latest endpoint identity changed')
    return parent


def require_frozen_runtime(manifest, root=ROOT):
    if Path(root).resolve() != Path(manifest['source_root']).resolve():
        raise ValueError('phase must execute from the manifest frozen source runtime; use run or its source/scripts entry')


def ensure_initializable_phase(output):
    if any(path.name != 'config.toml' for path in Path(output).iterdir()):
        raise RuntimeError('missing latest checkpoint in a previously used phase; refusing to restart')


def run_phase(args):
    manifest = verify_manifest(args.directory)
    require_frozen_runtime(manifest)
    import torch
    from mortal.core.toml_utils import load_toml_file, write_toml_file
    from mortal.supervised.curriculum_probe import CurriculumProbe

    spec = manifest['phases'][args.phase]
    if args.until_update not in spec['observations']:
        raise ValueError('segments must stop at declared observation points')
    output = Path(manifest['directory']) / f'{args.arm}_{args.phase}'
    parent = phase_parent(manifest, args.arm, args.phase)
    source = torch.load(parent['checkpoint'], map_location='cpu', weights_only=False)
    if parent != parent_record(source, path=parent['checkpoint'], sha256=parent['sha256'], phase=parent['phase']):
        raise ValueError('parent checkpoint metadata differs from receipt')
    if args.phase == 'C':
        provenance = source['run_provenance']
        if (provenance.get('experiment_id'), provenance.get('arm'), provenance.get('phase')) != (
                manifest['identity'], args.arm, 'B'):
            raise ValueError('C parent belongs to a different experiment or arm')
    config = build_phase_config(source, read_json(Path(manifest['directory']) / 'template.json'),
                               manifest, args.arm, args.phase, output, parent)
    config_path = output / 'config.toml'
    if output.exists():
        if not config_path.is_file() or load_toml_file(config_path) != config:
            raise FileExistsError('existing phase directory does not match this experiment')
    else:
        fresh_directory(output)
        write_toml_file(config_path, config)
    state_file = Path(config['supervised']['state_file'])
    if not state_file.exists():
        ensure_initializable_phase(output)
        atomic_torch_save(prepare_transition_state(source, config), state_file)
    else:
        current = torch.load(state_file, map_location='cpu', weights_only=False)
        if current['run_provenance'] != config['supervised']['run_provenance'] or current['config'] != config:
            raise ValueError('existing phase checkpoint config/provenance mismatch')
        del current
    del source
    os.environ.update(MORTAL_CFG=str(config_path), RAYON_NUM_THREADS=str(config['supervised']['rayon_num_threads']),
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if manifest['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_fraction'])
    torch.use_deterministic_algorithms(True)
    index = torch.load(Path(manifest['directory']) / 'indexes.pth', map_location='cpu', weights_only=True)
    identity = config['supervised']['run_provenance']['plan_id']
    probe = CurriculumProbe(config, index['domains'], recipe=args.phase, seed=spec['seed'], output=output,
                            horizons=spec['observations'], eval_splits=index['roles'], identity=identity,
                            reset_branch_rng=True)
    probe.stop_at = args.until_update
    from mortal.supervised.train_supervised import sanitize_sys_path_for_spawn, train
    sanitize_sys_path_for_spawn()
    train(probe=probe, stage_label=f'Early transition {args.arm} {args.phase}',
          checkpoint_label=f'{args.arm}_{args.phase}')
    for horizon in [0, *[n for n in spec['observations'] if n <= args.until_update]]:
        row = read_json(output / f'update_{horizon:07d}.json')
        if row['identity'] != identity or row['optimizer_updates'] != horizon:
            raise ValueError('missing or foreign phase observation')
    latest = torch.load(state_file, map_location='cpu', weights_only=False)
    if latest['optimizer_steps'] != args.until_update:
        raise ValueError('latest checkpoint is not the declared successful-update endpoint')
    receipt = {'identity': identity, 'experiment_id': manifest['identity'], 'arm': args.arm,
               'phase': args.phase, 'until_update': args.until_update, 'completed_at': time.time(),
               'endpoint': parent_record(latest, path=state_file, sha256=file_sha256(state_file), phase=args.phase)}
    atomic_write_json(output / f'segment_{args.until_update:07d}.json', receipt)
    if args.until_update == spec['updates']:
        atomic_write_json(output / 'completed.json', receipt)


def run(args):
    manifest = verify_manifest(args.directory)
    output = Path(manifest['directory'])
    for arm, phase, horizon in observation_schedule(manifest['phases']):
        segment = output / f'{arm}_{phase}' / f'segment_{horizon:07d}.json'
        if segment.exists():
            receipt = read_json(segment)
            if (receipt['experiment_id'], receipt['arm'], receipt['phase'], receipt['until_update']) != (
                    manifest['identity'], arm, phase, horizon):
                raise ValueError('existing segment belongs to a different experiment')
            observation = read_json(segment.parent / f'update_{horizon:07d}.json')
            if (observation['identity'] != receipt['identity']
                    or file_sha256(observation['checkpoint']) != observation['checkpoint_sha256']):
                raise ValueError('existing segment observation changed')
            continue
        progress = {'arm': arm, 'phase': phase, 'until_update': horizon, 'state': 'running'}
        atomic_write_json(output / 'progress.json', progress)
        log = output / f'{arm}_{phase}.log'
        command = [sys.executable, '-u', str(Path(manifest['source_root']) / 'scripts/run_sl_early_transition.py'),
                   'phase', '--directory', str(output), '--arm', arm, '--phase', phase,
                   '--until-update', str(horizon)]
        started = time.monotonic()
        with log.open('a', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=manifest['source_root'], stdout=stream, stderr=stream)
        progress.update(state='completed' if result.returncode == 0 else 'paused_or_failed',
                        returncode=result.returncode, wall_seconds=time.monotonic() - started)
        atomic_write_json(output / 'attempts' / f'{time.time_ns()}.json', progress)
        atomic_write_json(output / 'progress.json', progress)
        if result.returncode:
            raise SystemExit(result.returncode)
    atomic_write_json(output / 'completed.json', {'identity': manifest['identity'], 'completed_at': time.time(),
                      'selection': 'observations ready; no automatic winner, rollback, or publication'})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for name in ('directory', 'early', 'late', 'index', 'validation-index', 'native', 'source-commit'):
        prep.add_argument('--' + name, required=True)
    for phase in ('b', 'c'):
        for name in ('updates', 'seed', 'warmup'):
            prep.add_argument(f'--{phase}-{name}', type=int, required=True)
        prep.add_argument(f'--{phase}-observations', type=int, nargs='+', required=True)
        for name in ('lr', 'init-lr'):
            prep.add_argument(f'--{phase}-{name}', type=float, required=True)
    prep.add_argument('--device', default='cuda:0')
    prep.add_argument('--microbatch', type=int, required=True)
    prep.add_argument('--logical-batch', type=int, required=True)
    prep.add_argument('--val-batch-size', type=int, required=True)
    prep.add_argument('--gpu-memory-fraction', type=float, required=True)
    prep.add_argument('--prepare-workers', type=int, choices=range(9), default=0)
    prep.add_argument('--val-prepare-workers', type=int, choices=range(9), default=0)
    prep.add_argument('--prepare-file-batch-size', type=int, choices=(1, 2, 4), default=1)
    prep.add_argument('--val-file-batch-size', type=int, choices=(1, 2, 4, 8), default=1)
    prep.add_argument('--rayon-threads', type=int, choices=(1, 2, 4, 8), default=2)
    for name in ('run', 'phase'):
        sub = commands.add_parser(name)
        sub.add_argument('--directory', required=True)
        if name == 'phase':
            sub.add_argument('--arm', choices=ARMS, required=True)
            sub.add_argument('--phase', choices=PHASES, required=True)
            sub.add_argument('--until-update', type=int, required=True)
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.command == 'prepare':
        prepare(arguments)
    else:
        lock = '.runner.lock' if arguments.command == 'run' else '.training.lock'
        with experiment_lock(arguments.directory, lock):
            {'run': run, 'phase': run_phase}[arguments.command](arguments)

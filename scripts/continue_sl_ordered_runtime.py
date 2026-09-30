"""Continue paused initial curriculum arms in a qualified ordered runtime.

Every imported observation keeps its old provenance. Checkpoint metadata is
rebound explicitly; model, optimizer, RNG, consumed cursor and budgets stay exact.
Delayed-transfer arms must not have started yet.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.core.toml_utils import write_toml_file
from mortal.supervised.curriculum_probe import learned_state_digest
from scripts.continue_sl_curriculum_runtime import STATE_PATHS, assert_quiescent
from scripts.run_sl_curriculum_probe import build_config, freeze_search_parser, verify_manifest
from scripts.verify_sl_probe_resume import equal, read


PERFORMANCE_FIELDS = ('probe_prepare_file_batch_size', 'val_file_batch_size', 'rayon_num_threads',
                      'probe_prepare_workers', 'val_prepare_workers', 'prepare_rayon_threads')
VERIFIED_SOURCE = ('mortal/data/dataloader.py', 'mortal/supervised/train_supervised.py',
                   'mortal/supervised/curriculum_probe.py', 'mortal/supervised/ordered_preparation.py')


def semantic_config(config):
    result = deepcopy(config)
    sl = result['supervised']
    for name in (*STATE_PATHS, 'file_index', 'tensorboard_dir', *PERFORMANCE_FIELDS):
        sl.pop(name, None)
    sl['run_provenance'].pop('plan_id', None)
    return result


def rebind_checkpoint(state, config, identity):
    if semantic_config(state['config']) != semantic_config(config):
        raise ValueError('continuation would change training semantics')
    result = deepcopy(state)
    result['config'] = deepcopy(config)
    result['curriculum_probe']['identity'] = identity
    if 'run_provenance' in result:
        result['run_provenance'] = deepcopy(config['supervised']['run_provenance'])
    if learned_state_digest(result) != learned_state_digest(state):
        raise ValueError('checkpoint learned state changed during metadata relocation')
    for name in ('dataset', 'rng', 'observed', 'elapsed_seconds'):
        if not equal(state['curriculum_probe'][name], result['curriculum_probe'][name]):
            raise ValueError('checkpoint cursor or RNG changed during metadata relocation')
    return result


def qualify(proof, audit, inputs, original, native):
    required_checks = {'learned_state_exact', 'steps_exact', 'optimizer_steps_exact',
        'skipped_optimizer_steps_exact', 'auxiliary_optimizer_steps_exact', 'dataset_exact',
        'rng_exact', 'observed_exact', 'all_metrics_exact', 'exposure_exact'}
    if (proof.get('format') != 'sl_ordered_preparation_benchmark_v1' or not proof.get('passed')
            or not proof.get('resume_from') or not proof.get('checks') or not all(proof['checks'].values())
            or not required_checks <= proof['checks'].keys()
            or proof.get('final_update', 0) < 32 or proof.get('skipped_updates') != 0):
        raise ValueError('successful real training, complete metrics and resumed-state proof required')
    if (audit.get('format') != 'sl_ordered_preparation_input_audit_v1' or not audit.get('passed')
            or audit.get('samples', 0) < 8192 or audit.get('workers') != proof['workers']
            or not audit.get('ordered_fields_exact') or not audit.get('consumed_cursor_exact')):
        raise ValueError('full ordered input and consumed-cursor proof required')
    if (proof['inputs_identity'] != inputs['identity'] or audit['inputs_identity'] != inputs['identity']
            or inputs['source_identity'] != original['identity']
            or inputs['parent_sha256'] != original['parent_sha256']):
        raise ValueError('qualification uses different production inputs or parent')
    for record in (proof, audit):
        if record['runtime']['native_sha256'] != file_sha256(native):
            raise ValueError('qualification native extension differs')
        if any(record['runtime']['source_files'][name] != file_sha256(ROOT / name) for name in VERIFIED_SOURCE):
            raise ValueError('qualification preparation source differs')
    if file_sha256(native) != original['source_sha256']['libriichi.pyd']:
        raise ValueError('production native extension must remain unchanged')
    if proof['resources']['min_available_ram_bytes'] < 6 * 2**30:
        raise ValueError('qualification did not retain the declared RAM margin')
    return {'probe_prepare_file_batch_size': 4, 'probe_prepare_workers': proof['workers'],
        'val_prepare_workers': proof['val_workers'], 'val_file_batch_size': proof['val_file_batch_size'],
        'rayon_num_threads': proof['rayon_threads'], 'prepare_rayon_threads': proof['rayon_threads']}


def observation_metadata(observation, *, identity, checkpoint, previous_observation,
                         previous_checkpoint_sha256, source_identity):
    if observation['checkpoint_sha256'] != previous_checkpoint_sha256:
        raise ValueError('source observation checkpoint hash differs')
    result = deepcopy(observation)
    result.update(identity=identity, checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        imported_from={'observation': str(previous_observation), 'sha256': file_sha256(previous_observation),
            'checkpoint_sha256': previous_checkpoint_sha256, 'source_identity': source_identity,
            'method': 'same-host exact learned state and cursor, explicit metadata relocation'})
    if result['splits'] != observation['splits'] or result['exposure'] != observation['exposure']:
        raise ValueError('observation metrics or exposure changed')
    return result


def prepare(args):
    source, output = Path(args.source_run).resolve(), Path(args.directory).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('separate new continuation output required')
    original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    verify_manifest(original)
    if original['smoke']:
        raise ValueError('only the production curriculum queue may be continued')
    proof = json.loads(Path(args.resume_proof).read_text(encoding='utf-8'))
    audit = json.loads(Path(args.input_audit).read_text(encoding='utf-8'))
    inputs = json.loads(Path(args.inputs).read_text(encoding='utf-8'))
    native = Path(args.native).resolve()
    performance = qualify(proof, audit, inputs, original, native)
    arms = sorted(path.parent for path in source.glob('*/state_file.pth'))
    allowed = {f'{seed}_{route}' for seed in original['seeds'] for route in 'ABC'}
    if not arms or any(path.name not in allowed for path in arms):
        raise ValueError('delayed-transfer or unknown arms cannot use this migration')
    parent = read(original['parent'])
    statuses = {}
    for arm in arms:
        seed, route = arm.name.split('_')
        current = read(arm / 'state_file.pth')
        old_identity = stable_json_digest({'experiment': original['identity'], 'parent': original['parent_sha256'],
                                          'seed': int(seed), 'route': route, 'horizons': original['horizons']})
        if current['curriculum_probe']['identity'] != old_identity:
            raise ValueError('source checkpoint belongs to a different arm contract')
        config = build_config(parent, output / arm.name, output / 'indexes.pth', seed=int(seed),
            device=original['device'], microbatch=original['microbatch'], logical_batch=original['logical_batch'],
            identity='pending', runtime_performance=performance)
        if semantic_config(current['config']) != semantic_config(config):
            raise ValueError('source arm training semantics differ: ' + arm.name)
        probe = current['curriculum_probe']
        if (current['skipped_optimizer_steps'] or current['steps'] != current['optimizer_steps'] * 4
                or (current['optimizer_steps'] and sum(probe['dataset']['consumed'].values())
                    != current['optimizer_steps'] * original['logical_batch'])):
            raise ValueError('source arm is not at a complete successful-update cursor')
        statuses[arm.name] = {'updates': current['optimizer_steps'],
            'checkpoint_sha256': file_sha256(arm / 'state_file.pth'),
            'learned_state_sha256': learned_state_digest(current), 'observed': probe['observed']}
        del current
    if args.check_only:
        print(json.dumps({'ready_for_continuation': True, 'arms': statuses, 'performance': performance}), flush=True)
        return
    progress = json.loads((source / 'progress.json').read_text(encoding='utf-8'))
    if progress.get('returncode') != 75:
        raise ValueError('source runner must have completed its exact pause first')
    assert_quiescent(source, Path(original['source_root']))
    output.mkdir(parents=True)
    runtime = output / 'source'
    for path in (ROOT / 'mortal').rglob('*.py'):
        if '__pycache__' in path.parts or 'checkpoints' in path.parts:
            continue
        target = runtime / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ('run_sl_curriculum_probe.py', 'start_oracle_critic_detached.ps1',
                 'supervise_oracle_critic_around_apex.ps1'):
        target = runtime / 'scripts' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, target)
    parser = runtime / 'mortal/eval/search_runtime.py'
    parser_hash = file_sha256(parser)
    parser.write_text(freeze_search_parser(parser.read_text(encoding='utf-8')), encoding='utf-8')
    shutil.copy2(native, runtime / 'libriichi.pyd')
    shutil.copy2(original['parent'], output / 'parent.pth')
    shutil.copy2(original['indexes'], output / 'indexes.pth')
    manifest = deepcopy(original)
    manifest.pop('identity')
    manifest.update(parent=str(output / 'parent.pth'), indexes=str(output / 'indexes.pth'),
        source_root=str(runtime), source_sha256={p.relative_to(runtime).as_posix(): file_sha256(p)
            for p in runtime.rglob('*') if p.is_file()}, runtime_performance=performance, created_at=time.time(),
        runtime_overrides={'mortal/eval/search_runtime.py': {'main_source_sha256': parser_hash,
                            'reason': 'frozen keyword-only compatibility fix'}},
        continuation={'source_identity': original['identity'], 'source_directory': str(source),
            'source_manifest_sha256': file_sha256(source / 'manifest.json'), 'source_arms': statuses,
            'proof': str(Path(args.resume_proof).resolve()), 'proof_sha256': file_sha256(args.resume_proof),
            'input_audit': str(Path(args.input_audit).resolve()), 'input_audit_sha256': file_sha256(args.input_audit),
            'inputs_sha256': file_sha256(args.inputs), 'all_scientific_inputs_and_budgets_unchanged': True,
            'source_observations_reused_with_provenance': True,
            'deployment_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    manifest['identity'] = stable_json_digest(manifest)
    imported = []
    for arm in arms:
        seed, route = arm.name.split('_')
        old_identity = stable_json_digest({'experiment': original['identity'], 'parent': original['parent_sha256'],
                                          'seed': int(seed), 'route': route, 'horizons': original['horizons']})
        identity = stable_json_digest({'experiment': manifest['identity'], 'parent': original['parent_sha256'],
                                      'seed': int(seed), 'route': route, 'horizons': original['horizons']})
        target_arm = output / arm.name
        target_arm.mkdir()
        config = build_config(parent, target_arm, output / 'indexes.pth', seed=int(seed),
            device=manifest['device'], microbatch=manifest['microbatch'], logical_batch=manifest['logical_batch'],
            identity=identity, runtime_performance=performance)
        write_toml_file(target_arm / 'config.toml', config)
        checkpoint_hashes = {}
        for checkpoint in sorted(arm.glob('*.pth')):
            previous_sha = file_sha256(checkpoint)
            state = read(checkpoint)
            if state['curriculum_probe']['identity'] != old_identity:
                raise ValueError('source checkpoint belongs to a different observation contract')
            learned = learned_state_digest(state)
            rebound = rebind_checkpoint(state, config, identity)
            if file_sha256(checkpoint) != previous_sha:
                raise ValueError('paused source checkpoint changed during reading')
            atomic_torch_save(rebound, target_arm / checkpoint.name)
            checkpoint_hashes[checkpoint.name] = previous_sha
            imported.append({'source': str(checkpoint), 'source_sha256': previous_sha,
                'destination': str(target_arm / checkpoint.name),
                'destination_sha256': file_sha256(target_arm / checkpoint.name),
                'learned_state_sha256': learned, 'cursor_rng_exact': True})
            del state, rebound
            gc.collect()
        for observation in sorted(arm.glob('update_*.json')):
            result = json.loads(observation.read_text(encoding='utf-8'))
            if result['identity'] != old_identity:
                raise ValueError('source observation belongs to a different arm')
            checkpoint = target_arm / observation.with_suffix('.pth').name
            result = observation_metadata(result, identity=identity, checkpoint=checkpoint,
                previous_observation=observation, previous_checkpoint_sha256=checkpoint_hashes[checkpoint.name],
                source_identity=original['identity'])
            atomic_write_json(target_arm / observation.name, result)
        for segment in sorted(arm.glob('segment_*.json')):
            old = json.loads(segment.read_text(encoding='utf-8'))
            if old['identity'] != old_identity:
                raise ValueError('source segment belongs to a different arm')
            horizon = old['until_update']
            if not (target_arm / f'update_{horizon:07d}.json').is_file():
                raise ValueError('completed segment lacks its declared observation')
            receipt = {**old, 'identity': identity, 'imported_from': str(segment),
                       'source_sha256': file_sha256(segment)}
            atomic_write_json(target_arm / segment.name, receipt)
        complete = arm / 'completed.json'
        if complete.exists():
            old = json.loads(complete.read_text(encoding='utf-8'))
            atomic_write_json(target_arm / complete.name, {**old, 'identity': identity,
                'imported_from': str(complete), 'source_sha256': file_sha256(complete)})
    verify_manifest(manifest)
    assert_quiescent(source, Path(original['source_root']))
    atomic_write_json(output / 'manifest.json', manifest)
    atomic_write_json(output / 'continuation_receipt.json', {'identity': manifest['identity'],
        'source_identity': original['identity'], 'performance': performance, 'imported_checkpoints': imported,
        'all_scientific_inputs_and_budgets_unchanged': True, 'created_at': time.time()})
    atomic_write_json(output / 'apex_supervisor_spec.json', {
        'format': 'oracle_critic_apex_supervisor_spec_v1', 'repo_root': str(runtime),
        'python_executable': sys.executable, 'search_root': str(output),
        'pause_file': str(output / 'apex_pause.request'), 'status_file': str(output / 'apex_supervisor_status.json'),
        'log_file': str(output / 'apex_supervisor.log'),
        'runner_arguments': ['-u', str(runtime / 'scripts/run_sl_curriculum_probe.py'), 'run', '--directory', str(output)]})
    print(json.dumps({'ready': True, 'directory': str(output), 'identity': manifest['identity'],
        'performance': performance, 'imported_checkpoints': len(imported), 'arms': statuses}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'directory', 'resume-proof', 'input-audit', 'inputs', 'native'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args())

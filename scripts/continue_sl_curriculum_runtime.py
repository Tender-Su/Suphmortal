"""Relocate one verified, paused initial SL arm into an explicit new runtime."""
import argparse
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
from scripts.run_sl_curriculum_probe import build_config, freeze_search_parser, verify_manifest
from scripts.verify_sl_probe_resume import equal, read
from mortal.supervised.curriculum_probe import learned_state_digest


PERFORMANCE = {'probe_prepare_file_batch_size': 4, 'val_file_batch_size': 4, 'rayon_num_threads': 4}
STATE_PATHS = ('state_file', 'best_state_file', 'best_loss_state_file', 'best_acc_state_file',
               'best_rank_state_file', 'best_policy_state_file', 'adaptive_best_state_file')


def config_semantics(config):
    result = deepcopy(config)
    sl = result['supervised']
    for name in (*STATE_PATHS, 'file_index', 'tensorboard_dir', *PERFORMANCE):
        sl.pop(name, None)
    sl['run_provenance'].pop('plan_id', None)
    return result


def rebind_state(state, config, identity):
    if config_semantics(state['config']) != config_semantics(config):
        raise ValueError('runtime continuation would change training semantics')
    before = learned_state_digest(state)
    rebound = deepcopy(state)
    rebound['config'] = deepcopy(config)
    rebound['curriculum_probe']['identity'] = identity
    if 'run_provenance' in rebound:
        rebound['run_provenance'] = deepcopy(config['supervised']['run_provenance'])
    if learned_state_digest(rebound) != before:
        raise ValueError('learned state changed during metadata relocation')
    for key in ('dataset', 'rng', 'observed', 'elapsed_seconds'):
        if not equal(state['curriculum_probe'][key], rebound['curriculum_probe'][key]):
            raise ValueError('consumed-data or RNG state changed during relocation')
    return rebound


def assert_quiescent(source, source_root):
    if os.name != 'nt':
        raise RuntimeError('this activation barrier requires Windows process identities')
    paths = [str(path).replace('\\', '/') for path in (source, source_root)]
    if any("'" in path for path in paths):
        raise ValueError('unsupported quote in activation path')
    script = ("$me=$PID; @((Get-CimInstance Win32_Process) | Where-Object {"
              f"$_.ProcessId -ne $me -and $_.ProcessId -ne {os.getpid()} -and $_.CommandLine -and "
              f"($_.CommandLine.Replace('\\','/').IndexOf('{paths[0]}',[StringComparison]::OrdinalIgnoreCase) -ge 0 -or "
              f"$_.CommandLine.Replace('\\','/').IndexOf('{paths[1]}',[StringComparison]::OrdinalIgnoreCase) -ge 0)"
              "} | Select-Object ProcessId,Name) | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                            capture_output=True, text=True, check=True, timeout=20)
    if result.stdout.strip() not in ('', '[]', 'null'):
        raise RuntimeError('source process still references the run: ' + result.stdout.strip())


def prepare(args):
    source, output = Path(args.source_run).resolve(), Path(args.directory).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('a separate, nonexisting continuation output is required')
    manifest_path = source / 'manifest.json'
    original = json.loads(manifest_path.read_text(encoding='utf-8'))
    verify_manifest(original)
    if original['smoke'] or original.get('runtime_performance'):
        raise ValueError('only the original long-run runtime may use this one-time continuation')
    seed = original['seeds'][0]
    arm_name = f'{seed}_A'
    if list(sorted(p.parent.name for p in source.glob('*/state_file.pth'))) != [arm_name]:
        raise ValueError('this migration only supports the first active A arm')
    arm = source / arm_name
    current = read(arm / 'state_file.pth')
    updates = current['optimizer_steps']
    probe_state = current['curriculum_probe']
    if not 0 < updates < original['horizons'][0] or probe_state['observed'] != [0]:
        raise ValueError('expected a durable initial-arm cursor before the first observation horizon')
    if (not probe_state['dataset'] or current['skipped_optimizer_steps'] or
            current['steps'] != updates * current['config']['control']['opt_step_every']):
        raise ValueError('expected a complete, healthy consumed-data cursor')
    proof_path = Path(args.proof).resolve()
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if (proof.get('mode') != 'optimized_resume_equivalence' or not proof.get('passed') or
            not proof.get('checks') or not all(proof['checks'].values()) or
            proof['file_batch_size'] != 4 or proof['rayon_threads'] != 4):
        raise ValueError('changed preparation needs a successful exact recovery proof')
    for name in ('mortal/data/dataloader.py', 'mortal/supervised/train_supervised.py',
                 'mortal/supervised/curriculum_probe.py'):
        if file_sha256(ROOT / name) != proof['runtime']['source_files'][name]:
            raise ValueError('runtime differs from the verified preparation code: ' + name)
    native = Path(args.native).resolve()
    if file_sha256(native) != proof['runtime']['native_sha256'] or file_sha256(native) != original['source_sha256']['libriichi.pyd']:
        raise ValueError('native extension changed since verification')
    parent = read(original['parent'])
    config = build_config(parent, output / arm_name, output / 'indexes.pth', seed=seed,
        device=original['device'], microbatch=original['microbatch'], logical_batch=original['logical_batch'],
        identity='pending', runtime_performance=PERFORMANCE)
    if config_semantics(current['config']) != config_semantics(config):
        raise ValueError('training semantics differ outside the permitted preparation settings')
    if args.check_only:
        print(json.dumps({'ready_for_controlled_pause': True, 'source_updates': updates,
                          'source_checkpoint_sha256': file_sha256(arm / 'state_file.pth'),
                          'runtime_performance': PERFORMANCE}), flush=True)
        return
    progress = json.loads((source / 'progress.json').read_text(encoding='utf-8'))
    if progress.get('returncode') != 75:
        raise ValueError('source must exit at its controlled checkpoint boundary first')
    assert_quiescent(source, original['source_root'])
    output.mkdir(parents=True)
    runtime = output / 'source'
    for path in (ROOT / 'mortal').rglob('*.py'):
        if 'checkpoints' in path.parts or '__pycache__' in path.parts:
            continue
        target = runtime / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ('run_sl_curriculum_probe.py', 'start_oracle_critic_detached.ps1',
                 'supervise_oracle_critic_around_apex.ps1'):
        target = runtime / 'scripts' / name
        target.parent.mkdir(exist_ok=True)
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
             for p in runtime.rglob('*') if p.is_file()}, created_at=time.time(),
        runtime_performance=PERFORMANCE,
        runtime_overrides={'mortal/eval/search_runtime.py': {'main_source_sha256': parser_hash,
                           'reason': 'frozen keyword-only compatibility fix'}},
        continuation={'source_identity': original['identity'], 'source_directory': str(source),
            'source_manifest_sha256': file_sha256(manifest_path), 'source_updates': updates,
            'source_checkpoint_sha256': file_sha256(arm / 'state_file.pth'),
            'proof_path': str(proof_path), 'proof_sha256': file_sha256(proof_path),
            'all_inputs_unchanged': True, 'source_observations_reused_with_provenance': True,
            'method': 'same-host metadata relocation; ordered preparation and U1/U2 exact comparison verified',
            'deployment_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    manifest['identity'] = stable_json_digest(manifest)
    identity = stable_json_digest({'experiment': manifest['identity'], 'parent': original['parent_sha256'],
                                  'seed': seed, 'route': 'A', 'horizons': original['horizons']})
    config = build_config(parent, output / arm_name, output / 'indexes.pth', seed=seed,
        device=manifest['device'], microbatch=manifest['microbatch'], logical_batch=manifest['logical_batch'],
        identity=identity, runtime_performance=PERFORMANCE)
    rebound = rebind_state(current, config, identity)
    target_arm = output / arm_name
    target_arm.mkdir()
    from mortal.core.toml_utils import write_toml_file
    write_toml_file(target_arm / 'config.toml', config)
    atomic_torch_save(rebound, target_arm / 'state_file.pth')
    zero_path = arm / 'update_0000000.pth'
    zero = rebind_state(read(zero_path), config, identity)
    atomic_torch_save(zero, target_arm / zero_path.name)
    observation_path = arm / 'update_0000000.json'
    observation = json.loads(observation_path.read_text(encoding='utf-8'))
    if observation['checkpoint_sha256'] != file_sha256(zero_path):
        raise ValueError('initial observation checkpoint changed')
    observation.update(identity=identity, checkpoint=str(target_arm / zero_path.name),
        checkpoint_sha256=file_sha256(target_arm / zero_path.name),
        imported_from={'observation': str(observation_path), 'sha256': file_sha256(observation_path),
                       'checkpoint_sha256': file_sha256(zero_path), 'source_identity': original['identity']})
    atomic_write_json(target_arm / observation_path.name, observation)
    verify_manifest(manifest)
    atomic_write_json(output / 'manifest.json', manifest)
    receipt = {'manifest_identity': manifest['identity'], 'source_updates': updates,
        'learned_state_exact': learned_state_digest(current) == learned_state_digest(rebound),
        'cursor_rng_and_elapsed_exact': True, 'checkpoint_sha256': file_sha256(target_arm / 'state_file.pth'),
        'source_retained': str(source), 'created_at': time.time()}
    atomic_write_json(output / 'continuation_receipt.json', receipt)
    atomic_write_json(output / 'apex_supervisor_spec.json', {
        'format': 'oracle_critic_apex_supervisor_spec_v1', 'repo_root': str(runtime),
        'python_executable': sys.executable, 'search_root': str(output),
        'pause_file': str(output / 'apex_pause.request'), 'status_file': str(output / 'apex_supervisor_status.json'),
        'log_file': str(output / 'apex_supervisor.log'),
        'runner_arguments': ['-u', str(runtime / 'scripts/run_sl_curriculum_probe.py'), 'run', '--directory', str(output)]})
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'directory', 'proof', 'native'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args())

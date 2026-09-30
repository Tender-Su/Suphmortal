"""Prepare a fresh matched 1v3 runtime without importing old game results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mortal.core.artifacts import atomic_write_json, file_sha256, stable_json_digest
from mortal.eval.confirmation_protocol import runtime_record, verify_checkpoint


def verify_runtime_parity(previous, current, previous_root, current_root):
    if {k: v for k, v in previous.items() if k != 'source_sha256'} != {
            k: v for k, v in current.items() if k != 'source_sha256'}:
        raise ValueError('inference runtime versions or thread settings changed')
    if previous['source_sha256'].keys() != current['source_sha256'].keys():
        raise ValueError('inference source file set changed')
    newline_changes = []
    for name, old_hash in previous['source_sha256'].items():
        before, after = Path(previous_root) / name, Path(current_root) / name
        new_hash = current['source_sha256'][name]
        if file_sha256(before) != old_hash or file_sha256(after) != new_hash:
            raise ValueError('recorded runtime source changed: ' + name)
        # Python's source reader normalizes physical newlines before parsing.
        if before.read_text(encoding='utf-8') != after.read_text(encoding='utf-8'):
            raise ValueError('inference source text changed: ' + name)
        if old_hash != new_hash:
            newline_changes.append({'file': name, 'old_sha256': old_hash, 'new_sha256': new_hash})
    return newline_changes


def relocated_protocol(original, proof, provenance):
    if (proof.get('mode') != 'formal_inference' or proof.get('timing_mode') != 'throughput' or
            proof.get('protocol_fingerprint') != stable_json_digest(original) or
            proof.get('ordered_game_events_equal') is not True or proof.get('chunk_seeds') != 64 or
            proof.get('games', 0) < 256):
        raise ValueError('a complete matched chunk64 event and throughput proof is required')
    result = deepcopy(original)
    result['aa_games_per_arm'] = max(original['aa_games_per_arm'], 256)
    result['runtime_transition'] = provenance
    unchanged = deepcopy(result)
    unchanged['aa_games_per_arm'] = original['aa_games_per_arm']
    if 'runtime_transition' in original:
        unchanged['runtime_transition'] = original['runtime_transition']
    else:
        del unchanged['runtime_transition']
    if unchanged != original:
        raise ValueError('scientific protocol changed during runtime relocation')
    return result


def prepare(args):
    source = Path(args.source_run).resolve()
    output = Path(args.directory).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('a fresh, separate output is required')
    source_protocol = source / 'protocol.json'
    original = json.loads(source_protocol.read_text(encoding='utf-8'))
    aa = json.loads((source / 'aa_decision.json').read_text(encoding='utf-8'))
    if not aa['passed'] or aa['protocol_sha256'] != stable_json_digest(original):
        raise ValueError('source A/A provenance mismatch')
    if any(path.name != 'reference' for path in (source / 'screen').iterdir()) or (source / 'finalist.json').exists():
        raise ValueError('this transition is limited to an unfinished reference-only screen')
    manifest = json.loads((source / 'screen/reference/manifest.json').read_text(encoding='utf-8'))
    if manifest['chunk_seeds'] != 32 or manifest['protocol_sha256'] != stable_json_digest(original):
        raise ValueError('source screen identity mismatch')
    runtime = runtime_record('cuda:0')
    source_spec = json.loads((source / 'apex_supervisor_spec.json').read_text(encoding='utf-8'))
    newline_changes = verify_runtime_parity(manifest['runtime'], runtime, source_spec['repo_root'], ROOT)
    for record in (original['reference'], original['opponent'], *original['candidates'].values()):
        verify_checkpoint(record)
    proof_path = Path(args.proof).resolve()
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if (proof['runtime']['native_sha256'] != runtime['native_sha256'] or
            proof['runtime']['source_files']['mortal/eval/engine.py'] != runtime['source_sha256']['mortal/eval/engine.py']):
        raise ValueError('proof inference implementation differs')
    config_path = Path(args.config).resolve()
    provenance = {'source_directory': str(source), 'source_protocol_sha256': file_sha256(source_protocol),
        'source_protocol_fingerprint': stable_json_digest(original), 'source_chunk_seeds': 32,
        'new_chunk_seeds': 64, 'proof_path': str(proof_path), 'proof_sha256': file_sha256(proof_path),
        'old_results_imported': False, 'scientific_inputs_seeds_budgets_gates_unchanged': True,
        'inference_source_text_exact': True, 'physical_newline_changes': newline_changes,
        'configuration_source': str(config_path), 'configuration_sha256': file_sha256(config_path),
        'deployment_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'created_at': time.time(), 'reason': 'matched chunk64 throughput gain; larger tier not accepted'}
    protocol = relocated_protocol(original, proof, provenance)
    if args.check_only:
        print(json.dumps({'ready': True, 'unchanged_science': True, 'aa_games_per_arm': protocol['aa_games_per_arm'],
                          'old_results_imported': False, 'chunk_seeds': 64}), flush=True)
        return
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / 'runtime_config.toml')
    atomic_write_json(output / 'protocol.json', protocol)
    receipt = {'protocol_fingerprint': stable_json_digest(protocol), 'runtime': runtime,
        'provenance': provenance, 'runner_sha256': file_sha256(ROOT / 'scripts/run_sl_rl_confirmation.py'),
        'output_directory': str(output)}
    atomic_write_json(output / 'runtime_preparation.json', receipt)
    atomic_write_json(output / 'apex_supervisor_spec.json', {
        'format': 'oracle_critic_apex_supervisor_spec_v1', 'repo_root': str(ROOT),
        'python_executable': sys.executable, 'search_root': str(output),
        'pause_file': str(output / 'apex_pause.request'), 'status_file': str(output / 'apex_supervisor_status.json'),
        'log_file': str(output / 'apex_supervisor.log'),
        'runner_arguments': ['-u', str(ROOT / 'scripts/run_sl_rl_confirmation.py'), '--directory', str(output),
                             '--device', 'cuda:0', '--chunk-seeds', '64', '--config', str(output / 'runtime_config.toml')]})
    print(json.dumps({'protocol_fingerprint': receipt['protocol_fingerprint'], 'prepared_only': True,
                      'old_results_imported': False, 'directory': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'directory', 'proof', 'config'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args())

"""Frozen shortlists, independent 1v3 confirmation and content-bound evidence.

No command publishes a model. A non-positive confirmation lower bound is unresolved.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import os
import platform
import secrets
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import libriichi

from mortal.core.evidence_contract import fingerprint, native_module_file, sha256_file
from mortal.eval.paired_1v3 import compare_games, load_games


class EvaluationPaused(InterruptedError):
    """A complete chunk is preserved before yielding the evaluation GPU."""


def event_hashes(directory):
    hashes = []
    for log in sorted(Path(directory).rglob('*.json.gz')):
        with gzip.open(log, 'rt', encoding='utf-8') as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        for event in events:
            event.pop('meta', None)
        hashes.append(fingerprint(events))
    return hashes


def freeze_or_verify(filename, payload):
    filename = Path(filename)
    if filename.exists():
        if json.loads(filename.read_text(encoding='utf-8')) != payload:
            raise ValueError('frozen evaluation manifest changed: ' + str(filename))
    else:
        write_new(filename, payload)


def actor_fingerprint(state):
    digest = hashlib.sha256()
    for section in ('mortal', 'policy_net' if isinstance(state.get('policy_net'), dict) else 'current_dqn'):
        for key, value in sorted(state[section].items()):
            key = key.removeprefix('_orig_mod.').removeprefix('module.')
            tensor = value.detach().cpu().contiguous()
            digest.update(f'{section}/{key}/{tensor.dtype}/{tuple(tensor.shape)}'.encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(fingerprint({key: state['config'][key] for key in ('control', 'resnet')
                              if key == 'resnet'}).encode())
    digest.update(str(state['config']['control'].get('version', 1)).encode())
    return digest.hexdigest()


def checkpoint_record(filename):
    filename = Path(filename).resolve()
    state = torch.load(filename, map_location='cpu', weights_only=False)
    return {'path': str(filename), 'sha256': sha256_file(filename),
            'actor_sha256': actor_fingerprint(state), 'steps': int(state.get('steps', 0))}


def write_new(filename, payload):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)


def prepare_protocol(reference, opponent, candidates, *, screen_seed=None,
                     confirmation_seed=None):
    screen_seed = secrets.randbits(63) if screen_seed is None else screen_seed
    confirmation_seed = secrets.randbits(63) if confirmation_seed is None else confirmation_seed
    if not 1 <= len(candidates) <= 3:
        raise ValueError('freeze one to three candidate actors before screening')
    if screen_seed == confirmation_seed:
        raise ValueError('screen and confirmation seeds must be independent')
    actors = [record['actor_sha256'] for record in candidates.values()]
    if len(actors) != len(set(actors)) or reference['actor_sha256'] in actors:
        raise ValueError('shortlist contains duplicate actors or the unchanged reference')
    return {
        'schema_version': 1, 'primary': 'mean_pt', 'rank_points': [90, 45, 0, -135],
        'minimum_meaningful_pt': 0.0, 'screen_games_per_arm': 16000,
        'confirmation_games_per_arm': 64000, 'aa_games_per_arm': 8,
        'seed_start': 10000, 'screen_seed_key': screen_seed,
        'confirmation_seed_key': confirmation_seed, 'aa_seed_key': secrets.randbits(63),
        'reference': reference, 'opponent': opponent, 'candidates': candidates,
        'inference': {'oracle_input_mode': 'zero', 'search': False, 'explore_rate': 0,
                      'enable_amp': False, 'enable_compile': False, 'agari_guard': False},
        'selection_rule': 'one highest screen mean delta finalist; independent confirmation required',
        'confirmation_rule': 'paired cluster bootstrap 95% lower bound > minimum_meaningful_pt',
        'publication': 'manual; an inconclusive confirmation retains the reference',
    }


def runtime_record(device):
    root = Path(__file__).resolve().parents[2]
    sources = sorted(file.relative_to(root).as_posix()
                     for group in ('mortal/eval', 'mortal/core')
                     for file in (root / group).rglob('*.py'))
    return {'native_sha256': sha256_file(native_module_file(libriichi)), 'torch': torch.__version__,
            'python': platform.python_version(), 'device': device,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {key: os.environ.get(key) for key in (
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS',
                'CUBLAS_WORKSPACE_CONFIG')},
            'source_sha256': {name: sha256_file(root / name) for name in sources}}


def verify_checkpoint(record):
    if sha256_file(record['path']) != record['sha256']:
        raise ValueError('frozen checkpoint changed: ' + record['path'])


def run_arm(protocol, directory, *, stage, name, device='cpu', chunk_seeds=2):
    from mortal.eval.one_vs_three import load_mortal_engine, run_eval_once
    if stage not in {'aa', 'screen', 'confirmation'}:
        raise ValueError('invalid evaluation stage')
    directory = Path(directory)
    if stage != 'aa':
        aa = json.loads((directory / 'aa_decision.json').read_text(encoding='utf-8'))
        if aa['protocol_sha256'] != fingerprint(protocol) or not aa['passed']:
            raise ValueError('A/A must pass under this frozen protocol before screening')
    if stage == 'confirmation':
        selection = json.loads((directory / 'finalist.json').read_text(encoding='utf-8'))
        if selection['protocol_sha256'] != fingerprint(protocol):
            raise ValueError('finalist was selected under a different protocol')
        if name not in ('reference', selection['finalist']):
            raise ValueError('confirmation may only evaluate the frozen finalist and reference')
    if stage == 'aa' and name not in ('reference', 'reference_repeat'):
        raise ValueError('A/A evaluates reference and reference_repeat only')
    record = protocol['reference'] if name == 'reference' or stage == 'aa' else protocol['candidates'][name]
    opponent = protocol['opponent']
    verify_checkpoint(record)
    verify_checkpoint(opponent)
    settings = protocol['inference']
    if settings != prepare_inference_contract():
        raise ValueError('unsupported inference contract')
    run_directory = directory / stage / name
    native_logs = run_directory / 'games'
    manifest = {'protocol_sha256': fingerprint(protocol), 'stage': stage, 'name': name,
                'actor': record, 'opponent': opponent, 'runtime': runtime_record(device),
                'chunk_seeds': chunk_seeds, 'inference': settings}
    freeze_or_verify(run_directory / 'manifest.json', manifest)
    result_file = run_directory / 'result.json'
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding='utf-8'))
        if (result['manifest'] != manifest or result['event_stream_sha256'] != event_hashes(native_logs)
                or result['games'] != load_games(native_logs, 'candidate')):
            raise ValueError('completed evaluation evidence changed')
        return
    engine_cfg = {'state_file': record['path'], 'device': device, 'name': 'candidate',
                  'enable_compile': False, 'enable_amp': False,
                  'enable_rule_based_agari_guard': False, 'oracle_input_mode': 'zero'}
    challenger = load_mortal_engine(engine_cfg, enable_metadata=True)
    champion = load_mortal_engine({**engine_cfg, 'state_file': opponent['path'], 'name': 'opponent'}, enable_metadata=True)
    for engine in (challenger, champion):
        engine.search_runtime_bundle = None
        engine.explore_rate = 0
    context = {'engine_chal': challenger, 'engine_cham': champion}
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    count = protocol[f'{stage}_games_per_arm'] // 4
    seed_key = protocol[f'{stage}_seed_key']
    for offset in range(0, count, chunk_seeds):
        pause_file = os.environ.get('MORTAL_ORACLE_PAUSE_FILE')
        if pause_file and Path(pause_file).is_file():
            raise EvaluationPaused('evaluation paused at a complete chunk boundary')
        chunk = native_logs / f'{offset:07d}'
        chunk_file = chunk / 'complete.json'
        chunk_count = min(chunk_seeds, count - offset)
        if chunk_file.exists():
            saved = json.loads(chunk_file.read_text(encoding='utf-8'))
            if saved != {'offset': offset, 'count': chunk_count, 'events': event_hashes(chunk)}:
                raise ValueError('completed evaluation chunk changed')
            continue
        if chunk.exists():
            # Preserve interrupted evidence outside the result tree; replay the whole chunk.
            quarantine = run_directory / 'interrupted' / f'{offset:07d}_{secrets.token_hex(6)}'
            for target in (chunk, quarantine):
                if not target.resolve().is_relative_to(run_directory.resolve()):
                    raise ValueError('chunk move escapes the evaluation run')
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            chunk.rename(quarantine)
        run_eval_once(cfg={}, seed_start=protocol['seed_start'] + offset, seed_key=seed_key,
                      seed_count=chunk_count, log_dir=str(chunk),
                      disable_progress_bar=True, eval_context=context)
        if len(load_games(chunk, 'candidate')) != chunk_count * 4:
            raise ValueError('incomplete evaluation chunk')
        write_new(chunk_file, {'offset': offset, 'count': chunk_count, 'events': event_hashes(chunk)})
        print(f'{stage}/{name}: {(offset + chunk_count) * 4}/{count * 4} games sealed', flush=True)
    if runtime_record(device) != manifest['runtime']:
        raise ValueError('evaluation runtime changed while the arm ran')
    games = load_games(native_logs, 'candidate')
    if len(games) != count * 4:
        raise ValueError('incomplete evaluation arm')
    write_new(run_directory / 'result.json', {
        'manifest': manifest, 'games': games, 'event_stream_sha256': event_hashes(native_logs),
    })


def prepare_inference_contract():
    return {'oracle_input_mode': 'zero', 'search': False, 'explore_rate': 0,
            'enable_amp': False, 'enable_compile': False, 'agari_guard': False}


def paired_result(protocol, directory, stage, name):
    directory = Path(directory) / stage
    left = json.loads((directory / name / 'result.json').read_text(encoding='utf-8'))
    right = json.loads((directory / 'reference' / 'result.json').read_text(encoding='utf-8'))
    for field in ('protocol_sha256', 'opponent', 'runtime', 'chunk_seeds', 'inference'):
        if left['manifest'][field] != right['manifest'][field]:
            raise ValueError('unmatched evaluation provenance: ' + field)
    if left['manifest']['protocol_sha256'] != fingerprint(protocol):
        raise ValueError('evaluation belongs to a different frozen protocol')
    if any(len(arm['games']) != protocol[f'{stage}_games_per_arm'] for arm in (left, right)):
        raise ValueError('incomplete evaluation result cannot be used for selection')
    return compare_games(left['games'], right['games'])


def select_finalist(protocol, directory):
    comparisons = {name: paired_result(protocol, directory, 'screen', name)
                   for name in protocol['candidates']}
    finalist = max(comparisons, key=lambda name: comparisons[name]['delta_pt_candidate_minus_reference']['mean'])
    result = {'protocol_sha256': fingerprint(protocol), 'finalist': finalist,
              'actor_sha256': protocol['candidates'][finalist]['actor_sha256'],
              'screen_comparisons': comparisons, 'status': 'awaiting_independent_confirmation'}
    write_new(Path(directory) / 'finalist.json', result)
    return result


def check_aa(protocol, directory):
    comparison = paired_result(protocol, directory, 'aa', 'reference_repeat')
    runs = [json.loads((Path(directory) / 'aa' / name / 'result.json').read_text(encoding='utf-8'))
            for name in ('reference', 'reference_repeat')]
    passed = (comparison['changed_game_outcomes'] == 0
              and runs[0]['event_stream_sha256'] == runs[1]['event_stream_sha256'])
    result = {'protocol_sha256': fingerprint(protocol), 'passed': passed, 'comparison': comparison}
    write_new(Path(directory) / 'aa_decision.json', result)
    if not passed:
        raise ValueError('A/A game event streams differ; screening remains blocked')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'check-aa', 'select', 'decide'])
    parser.add_argument('--directory', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--opponent')
    parser.add_argument('--candidate', action='append', default=[])
    parser.add_argument('--stage', choices=['aa', 'screen', 'confirmation'])
    parser.add_argument('--name', default='reference')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chunk-seeds', type=int, default=2)
    args = parser.parse_args()
    directory = Path(args.directory)
    filename = directory / 'protocol.json'
    if args.action == 'prepare':
        candidates = {name: checkpoint_record(file) for name, file in
                      (spec.split('=', 1) for spec in args.candidate)}
        protocol = prepare_protocol(checkpoint_record(args.reference), checkpoint_record(args.opponent), candidates)
        write_new(filename, protocol)
        return
    protocol = json.loads(filename.read_text(encoding='utf-8'))
    if args.action == 'run':
        if args.chunk_seeds <= 0:
            raise ValueError('chunk-seeds must be positive')
        run_arm(protocol, directory, stage=args.stage, name=args.name, device=args.device, chunk_seeds=args.chunk_seeds)
    elif args.action == 'select':
        select_finalist(protocol, directory)
    elif args.action == 'check-aa':
        check_aa(protocol, directory)
    else:
        selected = json.loads((directory / 'finalist.json').read_text(encoding='utf-8'))
        comparison = paired_result(protocol, directory, 'confirmation', selected['finalist'])
        lower = comparison['delta_pt_candidate_minus_reference']['ci95'][0]
        result = {'protocol_sha256': fingerprint(protocol), 'finalist': selected['finalist'],
                  'comparison': comparison, 'decision': 'qualified' if lower > protocol['minimum_meaningful_pt']
                  else 'inconclusive_retain_reference'}
        write_new(directory / 'confirmation_decision.json', result)


if __name__ == '__main__':
    try:
        main()
    except EvaluationPaused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75) from None

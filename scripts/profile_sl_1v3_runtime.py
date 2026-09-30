"""Profile bounded frozen inputs in a separate runtime; never train or publish."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import cProfile
import gc
import itertools
import json
import os
from pathlib import Path
import pstats
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.research.runtime_profile import (
    PhaseTimes, ResourceSampler, install_tensor_and_model_timers,
    validation_input_hashes, validation_timers, windows_memory_snapshot,
)


def new_output(args, *, initialize_cuda=True):
    source = Path(args.source_run).resolve()
    output = Path(args.output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('diagnostic output must be outside the active run')
    output.mkdir(parents=True, exist_ok=False)
    if not 0 < args.gpu_memory_fraction <= 0.3:
        raise ValueError('diagnostic GPU allocation fraction must be in (0, 0.3]')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if initialize_cuda and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return source, output


def profile_summary(profile, output):
    profile.dump_stats(str(output / 'calls.pstats'))
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({'file': filename, 'line': line, 'name': name,
                     'calls': calls, 'primitive_calls': primitive,
                     'self_s': own, 'cumulative_s': cumulative})
    return sorted(rows, key=lambda row: -row['self_s'])[:45]


def runtime_identity():
    import libriichi
    from mortal.core.evidence_contract import native_module_file
    return {'python': sys.version, 'torch': torch.__version__,
            'native_sha256': file_sha256(native_module_file(libriichi)),
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'env': {name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS',
                     'CUBLAS_WORKSPACE_CONFIG')},
            'source_files': {name: file_sha256(ROOT / name) for name in
                ('mortal/data/dataloader.py', 'mortal/supervised/train_supervised.py',
                 'mortal/supervised/curriculum_probe.py',
                 'mortal/eval/engine.py', 'mortal/research/runtime_profile.py',
                 'scripts/profile_sl_1v3_runtime.py')}}


def finish(output, record, resources):
    record['resources'] = resources.report()
    record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    record['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
    record['runtime'] = runtime_identity()
    record['completed_at'] = time.time()
    atomic_write_json(output / 'profile.json', record)
    print(json.dumps({'profile': str(output / 'profile.json'), 'mode': record['mode'],
                      'wall_s': record['resources']['wall_s'],
                      'peak_cuda_mib': record['cuda_peak_allocated_bytes'] / 2**20,
                      'phases': record['phases']}, ensure_ascii=True), flush=True)


class ValidationOnlyProbe:
    horizons = [0]
    stop_at = 0

    def __init__(self, files, output, record, *, timing_mode='phases', repeats=2):
        self.files, self.output, self.record = files, output, record
        self.timing_mode, self.repeats = timing_mode, repeats

    def restore(self, state):
        if state['optimizer_steps'] != 0:
            raise ValueError('diagnostic must start from an isolated U0 branch')
        self.record['loaded_optimizer_updates'] = state['optimizer_steps']
        self.record['loaded_auxiliary_optimizer_steps'] = state['auxiliary_optimizer_steps']

    def observe(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        if optimizer_steps:
            raise ValueError('diagnostic cannot execute optimizer updates')
        if self.timing_mode == 'throughput':
            return self.measure_throughput(evaluate)
        phases = PhaseTimes(synchronize=torch.cuda.synchronize)
        hashes = []
        profile = cProfile.Profile()
        start = time.perf_counter()
        with validation_timers(evaluate, phases, hashes):
            profile.enable()
            try:
                metrics, batches = evaluate(self.files, 0, desc='Diagnostic validation',
                    scalar_prefix='diagnostic', collect_cluster_records=True)
            finally:
                profile.disable()
        self.record.update(phases=phases.report(), evaluation_s=time.perf_counter() - start,
                           batches=batches, samples=len(hashes), metrics=metrics,
                           hottest_calls=profile_summary(profile, self.output))
        self.record['samples_per_second_instrumented'] = len(hashes) / self.record['evaluation_s']
        atomic_write_json(self.output / 'ordered_sample_hashes.json', hashes)
        self.record['ordered_sample_hashes_sha256'] = file_sha256(self.output / 'ordered_sample_hashes.json')

    def measure_throughput(self, evaluate):
        # Hashing and warm-up are outside the unpatched, unprofiled timed passes.
        hashes = []
        with validation_input_hashes(hashes):
            metrics, batches = evaluate(self.files, 0, desc='Diagnostic input equivalence',
                scalar_prefix='diagnostic', collect_cluster_records=True)
        times = []
        for repeat in range(self.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            actual, actual_batches = evaluate(self.files, 0, desc=f'Diagnostic timing {repeat + 1}',
                scalar_prefix='diagnostic', collect_cluster_records=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            if actual != metrics or actual_batches != batches:
                raise RuntimeError('repeated validation changed metrics or batches')
        self.record.update(phases={}, evaluation_s=sum(times) / len(times),
            evaluation_times_s=times, batches=batches, samples=len(hashes), metrics=metrics,
            repeated_metrics_exact=True, samples_per_second=len(hashes) * len(times) / sum(times))
        atomic_write_json(self.output / 'ordered_sample_hashes.json', hashes)
        self.record['ordered_sample_hashes_sha256'] = file_sha256(self.output / 'ordered_sample_hashes.json')


def sl_inputs(args, source):
    if not 1 <= args.files <= 32 or args.offset < 0 or not 1 <= args.file_batch_size <= 8:
        raise ValueError('SL diagnostic input and native file batches must be bounded')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if args.input_profile:
        previous = json.loads(Path(args.input_profile).read_text(encoding='utf-8'))
        if (previous['production_identity'] != manifest['identity'] or
                previous['role'] != args.role or previous['offset'] != args.offset):
            raise ValueError('input profile is not the requested production subset')
        files = [item['path'] for item in previous['inputs']]
        if any(file_sha256(item['path']) != item['sha256'] for item in previous['inputs']):
            raise ValueError('diagnostic inputs changed')
    else:
        index = torch.load(manifest['indexes'], weights_only=True, map_location='cpu')
        files = index['roles'][args.role][args.offset:args.offset + args.files]
        del index
    if len(files) != args.files:
        raise ValueError('not enough declared controller files')
    return manifest, files


def profile_sl(args):
    source, output = new_output(args)
    manifest, files = sl_inputs(args, source)
    record = {'schema_version': 1, 'mode': 'sl_validation',
        'production_identity': manifest['identity'], 'production_run': str(source),
        'role': args.role, 'offset': args.offset,
        'timing_mode': args.timing_mode, 'file_batch_size': args.file_batch_size,
        'rayon_threads': args.rayon_threads,
        'inputs': [{'path': name, 'sha256': file_sha256(name)} for name in files],
        'notes': ['Diagnostic subset only; production splits and sample counts are unchanged.',
                  'CUDA boundaries are synchronized, so this is an instrumented decomposition, not a speedup benchmark.',
                  'Native read/parse/feature/label generation is initially one combined bin; cProfile cannot split its Rust internals.',
                  'Exclusive times avoid double counting. All auxiliary losses and detailed/sliced/cluster metrics are enabled.']}
    if args.timing_mode == 'throughput':
        record['notes'][1] = ('Timed passes have no phase patches, sample hashing or cProfile; CUDA is synchronized '
                              'only before/after the complete evaluation. One input-equivalence pass warms each process.')
    from scripts.run_sl_curriculum_probe import build_config, prepare_branch_state

    with ResourceSampler() as resources:
        parent = torch.load(manifest['parent'], weights_only=True, map_location='cpu', mmap=True)
        record['parent_sha256'] = file_sha256(manifest['parent'])
        local_index = output / 'indexes.pth'
        atomic_torch_save({'train_files': files[:1], 'monitor_recent_files': files,
                          'full_recent_files': files, 'old_regression_files': []}, local_index)
        config = build_config(parent, output, local_index, seed=manifest['seeds'][0],
            device='cuda:0', microbatch=manifest['microbatch'],
            logical_batch=manifest['logical_batch'], identity=stable_json_digest(record))
        config['supervised']['val_file_batch_size'] = args.file_batch_size
        config['supervised']['rayon_num_threads'] = args.rayon_threads
        config_path = output / 'config.toml'
        write_toml_file(config_path, config)
        atomic_torch_save(prepare_branch_state(parent, config), config['supervised']['state_file'])
        del parent
        os.environ['MORTAL_CFG'] = str(config_path)
        os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        from mortal.supervised.train_supervised import train
        probe = ValidationOnlyProbe(files, output, record, timing_mode=args.timing_mode, repeats=args.repeats)
        train(probe=probe, stage_label='Read-only diagnostic', checkpoint_label='diagnostic')
        if 'phases' not in record:
            raise RuntimeError('diagnostic exited before complete measurement')
    finish(output, record, resources)


def profile_block(args):
    source, output = new_output(args, initialize_cuda=False)
    manifest, files = sl_inputs(args, source)
    if len(files) != 4 or args.file_batch_size not in (1, 2, 4):
        raise ValueError('training preparation diagnostic uses exactly one four-draw block')
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from scripts.run_sl_curriculum_probe import build_config
    parent = torch.load(manifest['parent'], weights_only=True, map_location='cpu', mmap=True)
    config = build_config(parent, output, output / 'unused-index.pth', seed=manifest['seeds'][0],
        device='cpu', microbatch=manifest['microbatch'], logical_batch=manifest['logical_batch'],
        identity=manifest['identity'])
    config_path = output / 'config.toml'
    write_toml_file(config_path, config)
    del parent
    os.environ['MORTAL_CFG'] = str(config_path)
    from mortal.supervised.curriculum_probe import RotatingGameDataset
    from mortal.supervised.train_supervised import safe_default_collate
    from mortal.research.runtime_profile import batch_fingerprints
    kwargs = dict(version=config['control']['version'], enable_augmentation=True,
                  augmented_first=False, emit_opponent_state_labels=True, track_danger_labels=True)

    def dataset(size):
        return RotatingGameDataset({'recent': files}, {'recent': 1.0}, manifest['seeds'][0], kwargs,
                                   prepare_file_batch_size=size)

    record = {'schema_version': 1, 'mode': 'training_input_block', 'production_identity': manifest['identity'],
        'role': args.role, 'offset': args.offset, 'file_batch_size': args.file_batch_size,
        'rayon_threads': args.rayon_threads, 'phases': {}, 'optimizer_updates': 0,
        'notes': ['One four-draw block only, both views and every auxiliary label retained.',
                  'Input timings include iteration but no hashing/collation or model work; this is not full training throughput.']}
    with ResourceSampler() as resources:
        hashes, times, saved = [], [], None
        for repeat in range(args.repeats + 1):
            data = dataset(args.file_batch_size)
            stream = iter(data)
            started = time.perf_counter()
            first = next(stream)
            count = sum(row['available_decisions_per_draw'] for row in data.files.values())
            if repeat == 0:
                rows = [first, *itertools.islice(stream, min(255, count - 1))]
                saved = data.state_dict()
                while rows:
                    hashes.extend(batch_fingerprints(safe_default_collate(rows)))
                    rows = list(itertools.islice(stream, min(256, count - len(hashes))))
            else:
                for _ in range(count - 1):
                    next(stream)
                times.append(time.perf_counter() - started)
            if data.offset != count or data.exposure()['decisions'] != count:
                raise RuntimeError('preparation consumed outside the planned block')
            stream.close()
            del first, stream, data
            gc.collect()
        restored = dataset(args.file_batch_size)
        restored.load_state_dict(saved)
        stream = iter(restored)
        rows = list(itertools.islice(stream, min(256, len(hashes) - saved['offset'])))
        actual = batch_fingerprints(safe_default_collate(rows))
        record['resume_sample_hashes_equal'] = actual == hashes[saved['offset']:saved['offset'] + len(actual)]
        if not record['resume_sample_hashes_equal']:
            raise RuntimeError('mid-block restored inputs changed')
        stream.close()
        record.update(samples=len(hashes), preparation_times_s=times, saved_offset=saved['offset'],
                      resumed_offset=restored.offset, samples_per_second=len(hashes) * len(times) / sum(times))
        atomic_write_json(output / 'ordered_sample_hashes.json', hashes)
        record['ordered_sample_hashes_sha256'] = file_sha256(output / 'ordered_sample_hashes.json')
    finish(output, record, resources)


def thread_schedule_during(operation, *, interval=0.02):
    if not 0 < interval <= 1:
        raise ValueError('scheduling sample interval must be in (0, 1]')
    stop, ready = threading.Event(), threading.Event()
    ticks = []

    def sample():
        ready.set()
        while not stop.wait(interval):
            ticks.append(time.perf_counter())

    thread = threading.Thread(target=sample, name='native-scheduling-sentinel', daemon=True)
    thread.start()
    try:
        if not ready.wait(1.0):
            raise RuntimeError('scheduling sentinel did not start')
        started = time.perf_counter()
        result = operation()
        ended = time.perf_counter()
    finally:
        stop.set()
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError('scheduling sentinel did not stop')
    during = [tick for tick in ticks if started <= tick <= ended]
    boundaries = [started, *during, ended]
    return result, {'elapsed_s': ended - started, 'ticks_during_call': len(during),
        'max_gap_s': max(b - a for a, b in zip(boundaries, boundaries[1:])),
        'interval_s': interval, 'sentinel_stopped': True}


def profile_loader_scheduling(args):
    if not 1 <= args.files <= 4:
        raise ValueError('loader scheduling diagnostic is limited to four files')
    if windows_memory_snapshot()['available_ram_bytes'] < 12 * 2**30:
        raise ValueError('insufficient RAM headroom for a separate loader diagnostic')
    source, output = new_output(args, initialize_cuda=False)
    manifest, files = sl_inputs(args, source)
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parent = torch.load(manifest['parent'], weights_only=True, map_location='cpu', mmap=True)
    version = parent['config']['control']['version']
    del parent
    from libriichi.dataset import GameplayLoader
    loader = GameplayLoader(version=version, oracle=False, player_names=[], excludes=[],
        augmented=False, track_opponent_states=True, track_danger_labels=True)
    record = {'schema_version': 1, 'mode': 'loader_thread_scheduling',
        'production_identity': manifest['identity'], 'role': args.role, 'offset': args.offset,
        'inputs': [{'path': name, 'sha256': file_sha256(name)} for name in files],
        'rayon_threads': args.rayon_threads, 'optimizer_updates': 0,
        'notes': ['CPU-only scheduling observation, not a throughput or equivalence benchmark.',
                  'One native call, one unaugmented view; production sampling and labels are not changed.',
                  'A blocked sentinel is consistent with native GIL retention; OS scheduling is also a possible contributor.',
                  'The outer launcher must enforce a process timeout; a Python thread cannot cancel a native call.']}
    with ResourceSampler() as resources:
        _, control = thread_schedule_during(lambda: time.sleep(0.5))
        groups, native = thread_schedule_during(lambda: loader.load_gz_log_files(files))
        record['phases'] = {'sleep_control': control, 'native_loader': native}
        record['player_records'] = sum(len(group) for group in groups)
        del groups
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError('loader scheduling diagnostic unexpectedly initialized CUDA')
    finish(output, record, resources)


def validate_eval_capacity(proof, *, chunk_seeds, protocol_fingerprint, gpu_free_bytes,
                           gpu_total_bytes, ram_free_bytes, gpu_fraction):
    if (proof.get('mode') != 'formal_inference' or proof.get('timing_mode') != 'throughput' or
            proof.get('ordered_game_events_equal') is not True or
            proof.get('protocol_fingerprint') != protocol_fingerprint or
            proof.get('chunk_seeds', 0) < chunk_seeds / 2 or
            proof.get('games', 0) < proof.get('chunk_seeds', 0) * 4):
        raise ValueError('larger evaluation needs a matched, successful preceding capacity proof')
    scale = max(1.0, chunk_seeds / proof['chunk_seeds'])
    gpu = proof['cuda_peak_reserved_bytes'] * scale + 256 * 2**20
    ram = proof['resources']['process_lifetime_peak_commit_bytes'] * scale + 2 * 2**30
    if (gpu > gpu_total_bytes * gpu_fraction * 0.8 or
            gpu_free_bytes - gpu < 2 * 2**30 or ram_free_bytes - ram < 12 * 2**30):
        raise ValueError('insufficient measured RAM/VRAM headroom for the next capacity tier')
    return {'previous_chunk_seeds': proof['chunk_seeds'], 'projected_gpu_bytes': gpu,
            'projected_commit_bytes': ram, 'prelaunch_free_gpu_bytes': gpu_free_bytes,
            'prelaunch_free_ram_bytes': ram_free_bytes,
            'note': 'Conservative projection is a launch gate, not a measured peak guarantee.'}


def profile_eval(args):
    source, output = new_output(args)
    total_seeds = args.total_seeds or args.chunk_seeds
    if not 1 <= args.chunk_seeds <= total_seeds <= 256 or args.seed_offset < 0:
        raise ValueError('evaluation diagnostic is limited to 256 seeds')
    protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    config = deepcopy(load_toml_file(args.config))
    config['control'].update(device='cuda:0', enable_compile=False, enable_amp=False)
    config_path = output / 'config.toml'
    write_toml_file(config_path, config)
    os.environ['MORTAL_CFG'] = str(config_path)
    os.environ['MORTAL_AGENT_PROFILE'] = '1' if args.timing_mode == 'phases' else '0'
    os.environ['MORTAL_ENGINE_PROFILE'] = '1' if args.timing_mode == 'phases' else '0'
    from mortal.eval.confirmation_protocol import event_hashes, prepare_inference_contract, verify_checkpoint
    from mortal.eval.one_vs_three import load_mortal_engine, run_eval_once
    from mortal.eval.engine import MortalEngine
    if protocol['inference'] != prepare_inference_contract():
        raise ValueError('unknown inference semantics')
    for name in ('reference', 'opponent'):
        verify_checkpoint(protocol[name])
    capacity = None
    if args.chunk_seeds > 64:
        if not args.capacity_proof:
            raise ValueError('larger evaluation needs --capacity-proof from the preceding tier')
        proof_path = Path(args.capacity_proof).resolve()
        proof = json.loads(proof_path.read_text(encoding='utf-8'))
        from mortal.core.evidence_contract import native_module_file
        import libriichi
        if (proof['runtime']['native_sha256'] != file_sha256(native_module_file(libriichi)) or
                proof['runtime']['torch'] != torch.__version__ or
                proof['runtime']['source_files']['mortal/eval/engine.py'] != file_sha256(ROOT / 'mortal/eval/engine.py')):
            raise ValueError('capacity proof is from another inference runtime')
        free, total = torch.cuda.mem_get_info()
        capacity = validate_eval_capacity(proof, chunk_seeds=args.chunk_seeds,
            protocol_fingerprint=stable_json_digest(protocol), gpu_free_bytes=free,
            gpu_total_bytes=total, ram_free_bytes=windows_memory_snapshot()['available_ram_bytes'],
            gpu_fraction=args.gpu_memory_fraction)
        capacity.update(proof_path=str(proof_path), proof_sha256=file_sha256(proof_path))
    phases = PhaseTimes(synchronize=torch.cuda.synchronize)
    profile = cProfile.Profile()
    record = {'schema_version': 1, 'mode': 'formal_inference',
              'production_run': str(source), 'protocol_fingerprint': stable_json_digest(protocol),
              'seed_start': protocol['seed_start'] + args.seed_offset,
              'seed_key': protocol['screen_seed_key'], 'seed_count': total_seeds,
              'chunk_seeds': args.chunk_seeds, 'timing_mode': args.timing_mode, 'capacity_gate': capacity,
              'notes': ['Diagnostic only; no sealed production results are reused for scientific selection.',
                        'CUDA boundaries are synchronized; instrumented timing is not an unprofiled throughput comparison.',
                        'Native agent wait_encode/pack/decode counters are emitted to stderr; wait_encode is not total encoding CPU time.',
                        'Driver residual includes simulation, overlapped encoding, native/Python bridge and log dumping.']}
    with ResourceSampler() as resources:
        base = {'device': 'cuda:0', 'enable_compile': False, 'enable_amp': False,
                'enable_rule_based_agari_guard': False, 'oracle_input_mode': 'zero'}
        engines = [load_mortal_engine({**base, 'state_file': protocol[name]['path'],
                    'name': alias}, enable_metadata=True)
                   for name, alias in (('reference', 'candidate'), ('opponent', 'opponent'))]
        for engine in engines:
            engine.search_runtime_bundle = None
            engine.explore_rate = 0
        context = dict(engine_chal=engines[0], engine_cham=engines[1])
        from unittest.mock import patch
        with ExitStack() as stack:
            if args.timing_mode == 'phases':
                install_tensor_and_model_timers(stack, phases)
                stack.enter_context(patch.object(MortalEngine, '_react_batch',
                    phases.wrap(MortalEngine._react_batch, 'inference_python_other', sync=True)))
                stack.enter_context(patch.object(MortalEngine, '_prepare_batch_tensors',
                    phases.wrap(MortalEngine._prepare_batch_tensors, 'input_pack_other', sync=True)))
            else:
                record['notes'][1] = 'No phase timers, CUDA boundary patches, native/engine profiling or cProfile in the timed pass.'
            torch.cuda.synchronize()
            started = time.perf_counter()
            if args.timing_mode == 'phases':
                profile.enable()
            try:
                for chunk_offset in range(0, total_seeds, args.chunk_seeds):
                    run_eval_once(cfg={}, seed_start=record['seed_start'] + chunk_offset,
                        seed_key=record['seed_key'], seed_count=min(args.chunk_seeds, total_seeds - chunk_offset),
                        log_dir=str(output / 'games' / f'{chunk_offset:07d}'),
                        disable_progress_bar=True, eval_context=context)
                torch.cuda.synchronize()
            finally:
                profile.disable()
            record['evaluation_s'] = time.perf_counter() - started
        record['phases'] = phases.report()
        record['engine_profiles'] = [dict(name=e.name, **e._profile_stats) for e in engines]
        if args.timing_mode == 'phases':
            record['hottest_calls'] = profile_summary(profile, output)
        hashes = event_hashes(output / 'games')
        record['games'] = len(hashes)
        expected = []
        remaining = total_seeds
        offset = args.seed_offset
        while remaining:
            path = source / 'screen' / 'reference' / 'games' / f'{offset:07d}' / 'complete.json'
            if not path.is_file():
                break
            baseline = json.loads(path.read_text(encoding='utf-8'))
            if baseline['count'] > remaining:
                break
            names = [game.name for game in sorted(path.parent.rglob('*.json.gz'))]
            if len(names) != len(baseline['events']):
                raise ValueError('sealed chunk log count differs from its event hashes')
            expected.extend(zip(names, baseline['events']))
            offset += baseline['count']
            remaining -= baseline['count']
        if not remaining:
            actual_names = [game.name for game in sorted((output / 'games').rglob('*.json.gz'))]
            record['ordered_game_events_equal'] = sorted(zip(actual_names, hashes)) == sorted(expected)
        atomic_write_json(output / 'ordered_event_hashes.json', hashes)
    finish(output, record, resources)


def profile_train_resume(args):
    source, output = new_output(args)
    if args.file_batch_size not in (1, 2, 4):
        raise ValueError('training preparation is bounded by the existing four-draw block')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest['smoke'] or manifest['horizons'] != [1, 2]:
        raise ValueError('only completed isolated U1/U2 evidence can validate a changed preparation path')
    arm = source / '20260907_A'
    from scripts.verify_sl_probe_resume import equal, read
    from mortal.supervised.curriculum_probe import CurriculumProbe, learned_state_digest
    initial = read(arm / 'update_0000001.pth')
    config = deepcopy(initial['config'])
    config['control']['device'] = 'cuda:0'
    sl = config['supervised']
    for key in ('state_file', 'best_state_file', 'best_loss_state_file', 'best_acc_state_file',
                'best_rank_state_file', 'best_policy_state_file', 'adaptive_best_state_file'):
        sl[key] = str(output / (key + '.pth'))
    sl.update(tensorboard_dir=str(output / 'tensorboard'), val_file_batch_size=args.file_batch_size,
              rayon_num_threads=args.rayon_threads, probe_prepare_file_batch_size=args.file_batch_size)
    config_path = output / 'config.toml'
    write_toml_file(config_path, config)
    atomic_torch_save(initial, sl['state_file'])
    identity = initial['curriculum_probe']['identity']
    del initial
    os.environ['MORTAL_CFG'] = str(config_path)
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    index = torch.load(manifest['indexes'], weights_only=True, map_location='cpu')
    record = {'schema_version': 1, 'mode': 'optimized_resume_equivalence', 'phases': {},
        'source_run': str(source), 'source_U1_sha256': file_sha256(arm / 'update_0000001.pth'),
        'expected_U2_sha256': file_sha256(arm / 'update_0000002.pth'),
        'file_batch_size': args.file_batch_size, 'rayon_threads': args.rayon_threads,
        'notes': ['Reuses completed U1 as read-only input and U2 as expected evidence in a fresh output.',
                  'Only the changed preparation path is run. Production and previous smoke artifacts are not modified.']}
    with ResourceSampler() as resources:
        probe = CurriculumProbe(config, index['domains'], recipe='A', seed=manifest['seeds'][0],
            output=output, horizons=[1, 2], identity=identity,
            eval_splits={name: index['roles'][name] for name in ('controller_recent', 'controller_old')})
        from mortal.supervised.train_supervised import train
        started = time.perf_counter()
        train(probe=probe, stage_label='Changed preparation resume verification', checkpoint_label='diagnostic')
        record['one_update_and_observation_s'] = time.perf_counter() - started
        expected, actual = read(arm / 'update_0000002.pth'), read(output / 'update_0000002.pth')
        checks = {'learned_state_exact': learned_state_digest(expected) == learned_state_digest(actual)}
        for key in ('steps', 'optimizer_steps', 'skipped_optimizer_steps', 'auxiliary_optimizer_steps'):
            checks[key + '_exact'] = expected[key] == actual[key]
        for key in ('dataset', 'observed', 'rng'):
            checks[key + '_exact'] = equal(expected['curriculum_probe'][key], actual['curriculum_probe'][key])
        before = json.loads((arm / 'update_0000002.json').read_text(encoding='utf-8'))
        after = json.loads((output / 'update_0000002.json').read_text(encoding='utf-8'))
        checks.update(metrics_exact=before['splits'] == after['splits'], exposure_exact=before['exposure'] == after['exposure'])
        record['checks'] = checks
        record['passed'] = all(checks.values())
        record['optimizer_updates'] = actual['optimizer_steps']
    finish(output, record, resources)
    if not record['passed']:
        raise RuntimeError('changed preparation path failed exact recovery comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('sl', 'block', 'eval', 'train-resume', 'loader-scheduling'):
        item = sub.add_parser(mode)
        item.add_argument('--source-run', required=True)
        item.add_argument('--output', required=True)
        item.add_argument('--gpu-memory-fraction', type=float, default=0.25)
        if mode == 'loader-scheduling':
            item.add_argument('--role', choices=('controller_recent', 'controller_old'), default='controller_recent')
            item.add_argument('--offset', type=int, default=8)
            item.add_argument('--files', type=int, choices=(1, 2, 3, 4), default=1)
            item.add_argument('--rayon-threads', type=int, choices=(1, 2, 4), default=4)
            item.add_argument('--input-profile')
            item.set_defaults(file_batch_size=1)
        elif mode in ('sl', 'block'):
            item.add_argument('--role', choices=('controller_recent', 'controller_old'), default='controller_recent')
            item.add_argument('--offset', type=int, default=8)
            item.add_argument('--files', type=int, default=4)
            item.add_argument('--file-batch-size', type=int, default=1)
            item.add_argument('--rayon-threads', type=int, choices=(1, 2, 4, 8), default=2)
            item.add_argument('--timing-mode', choices=('phases', 'throughput'), default='phases')
            item.add_argument('--repeats', type=int, choices=(1, 2, 3), default=2)
            item.add_argument('--input-profile')
        elif mode == 'eval':
            item.add_argument('--config', required=True)
            item.add_argument('--chunk-seeds', type=int, default=32)
            item.add_argument('--total-seeds', type=int)
            item.add_argument('--seed-offset', type=int, default=0)
            item.add_argument('--capacity-proof')
            item.add_argument('--timing-mode', choices=('phases', 'throughput'), default='phases')
        else:
            item.add_argument('--file-batch-size', type=int, choices=(1, 2, 4), default=4)
            item.add_argument('--rayon-threads', type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    {'sl': profile_sl, 'block': profile_block, 'eval': profile_eval,
     'train-resume': profile_train_resume, 'loader-scheduling': profile_loader_scheduling}[args.mode](args)


if __name__ == '__main__':
    main()

"""Matched, bounded SL preparation experiments in a separate Git runtime."""
import argparse
from copy import deepcopy
import gc
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.core.toml_utils import write_toml_file
from mortal.research.runtime_profile import ResourceSampler, batch_fingerprints
from mortal.supervised.curriculum_probe import CurriculumProbe, RECIPES, capture_rng, learned_state_digest
from scripts.run_sl_curriculum_probe import build_config, prepare_branch_state
from scripts.verify_sl_probe_resume import equal, read


def prepare_inputs(args):
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('new diagnostic input directory outside production required')
    output.mkdir(parents=True)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    # Trusted, locally produced index. The restricted unpickler makes this
    # millions-of-strings metadata export disproportionately slow on Windows.
    index = torch.load(manifest['indexes'], map_location='cpu', weights_only=False, mmap=True)
    rng = np.random.default_rng(20260908)
    domains = {name: [files[int(i)] for i in rng.choice(len(files), min(64, len(files)), replace=False)]
               for name, files in index['domains'].items()}
    roles = {name: index['roles'][name][:args.eval_files] for name in ('controller_recent', 'controller_old')}
    selected = sorted(set(itertools.chain.from_iterable([*domains.values(), *roles.values()])))
    inputs = {'format': 'sl_ordered_preparation_inputs_v1', 'source_run': str(source),
        'source_identity': manifest['identity'], 'parent': manifest['parent'],
        'parent_sha256': file_sha256(manifest['parent']), 'seed': manifest['seeds'][0],
        'microbatch': manifest['microbatch'], 'logical_batch': manifest['logical_batch'],
        'domains': domains, 'roles': roles, 'files_sha256': {name: file_sha256(name) for name in selected},
        'purpose': 'Representative diagnostic subset only; no scientific selection or production index mutation.'}
    inputs['identity'] = stable_json_digest(inputs)
    atomic_write_json(output / 'inputs.json', inputs)
    print(json.dumps({'inputs': str(output / 'inputs.json'), 'games': len(selected),
                      'identity': inputs['identity']}), flush=True)


def configure(args):
    inputs = json.loads(Path(args.inputs).read_text(encoding='utf-8'))
    fingerprint = dict(inputs)
    expected = fingerprint.pop('identity')
    if stable_json_digest(fingerprint) != expected:
        raise ValueError('diagnostic input manifest changed')
    output = Path(args.output).resolve()
    source = Path(inputs['source_run']).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('new isolated output outside production required')
    output.mkdir(parents=True)
    for name, expected in inputs['files_sha256'].items():
        if file_sha256(name) != expected:
            raise ValueError('diagnostic source game changed: ' + name)
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', RAYON_NUM_THREADS=str(args.rayon_threads))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # The trainer seeds Python and Torch, but the unused legacy NumPy RNG is
    # still part of an exact checkpoint. Give cold benchmark runs one origin.
    # Restored runs subsequently replace it with their common saved RNG state.
    np.random.seed(inputs['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    parent = torch.load(inputs['parent'], weights_only=True, map_location='cpu', mmap=True)
    if file_sha256(inputs['parent']) != inputs['parent_sha256']:
        raise ValueError('frozen parent changed')
    index = output / 'indexes.pth'
    atomic_torch_save({'train_files': list(itertools.chain.from_iterable(inputs['domains'].values())),
        'monitor_recent_files': inputs['roles']['controller_recent'],
        'full_recent_files': inputs['roles']['controller_recent'],
        'old_regression_files': inputs['roles']['controller_old']}, index)
    config = build_config(parent, output, index, seed=inputs['seed'], device='cuda:0',
        microbatch=inputs['microbatch'], logical_batch=inputs['logical_batch'], identity=inputs['identity'])
    config['supervised'].update(probe_prepare_file_batch_size=4, probe_prepare_workers=args.workers,
        val_prepare_workers=args.val_workers, val_file_batch_size=args.val_file_batch_size,
        rayon_num_threads=args.rayon_threads, prepare_rayon_threads=args.rayon_threads,
        log_every=128)
    config_path = output / 'config.toml'
    write_toml_file(config_path, config)
    os.environ['MORTAL_CFG'] = str(config_path)
    atomic_torch_save(prepare_branch_state(parent, config), config['supervised']['state_file'])
    del parent
    gc.collect()
    return inputs, output, config


class BenchmarkProbe(CurriculumProbe):
    def __init__(self, *args, warmup, record, evaluate_final=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup, self.record, self.evaluate_final = warmup, record, evaluate_final
        self.interval_started = None
        self.times = []

    def observe(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        # U0 in this diagnostic only loads the identical parent; its full
        # controller baseline already exists in the production experiment.
        if optimizer_steps == 0:
            self.observed = [0]
            return
        if self.evaluate_final:
            super().observe(optimizer_steps, evaluate, build_state, save_latest, epoch)
        else:
            self.observed.append(optimizer_steps)
            state = build_state(epoch, epoch_complete=False)
            atomic_torch_save(state, self.output / f'update_{optimizer_steps:07d}.pth')

    def after_update(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.interval_started is not None:
            self.times.append(now - self.interval_started)
        if optimizer_steps >= self.warmup:
            self.interval_started = now
        if optimizer_steps == self.warmup:
            atomic_torch_save(build_state(epoch, epoch_complete=False), self.output / 'resume_point.pth')
            self.interval_started = time.perf_counter()
        if optimizer_steps == self.stop_at:
            self.record.update(timed_updates=len(self.times), update_intervals_s=self.times,
                training_seconds=sum(self.times), updates_per_second=len(self.times) / sum(self.times),
                successful_decisions_per_second=len(self.times) * self.config['supervised']['batch_size']
                    * self.config['control']['opt_step_every'] / sum(self.times))
            self.observe(optimizer_steps, evaluate, build_state, save_latest, epoch)
            return True
        return False


def runtime_record():
    import libriichi
    from mortal.core.evidence_contract import native_module_file
    return {'python': sys.version, 'torch': torch.__version__,
        'native_sha256': file_sha256(native_module_file(libriichi)),
        'source_files': {name: file_sha256(ROOT / name) for name in (
            'mortal/supervised/train_supervised.py', 'mortal/supervised/curriculum_probe.py',
            'mortal/supervised/ordered_preparation.py', 'mortal/data/dataloader.py',
            'scripts/benchmark_sl_ordered_preparation.py')},
        'parent_threads': torch.get_num_threads(), 'worker_torch_threads': 1,
        'session_id': _session_id()}


def _session_id():
    import ctypes
    session = ctypes.c_ulong()
    if not ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        raise ctypes.WinError()
    return session.value


def run_benchmark(args):
    inputs, output, config = configure(args)
    if not 0 < args.gpu_memory_fraction <= 0.4:
        raise ValueError('diagnostic GPU fraction must be in (0, .4]')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    record = {'format': 'sl_ordered_preparation_benchmark_v1', 'inputs_identity': inputs['identity'],
        'workers': args.workers, 'val_workers': args.val_workers,
        'val_file_batch_size': args.val_file_batch_size, 'rayon_threads': args.rayon_threads,
        'microbatch': inputs['microbatch'], 'logical_batch': inputs['logical_batch'],
        'warmup_updates': args.warmup, 'planned_timed_updates': args.updates,
        'timing_boundary': ('first resumed update warms reconstruction; later successful updates timed'
                            if args.resume_from else 'warmup checkpoint write excluded; later successful updates timed'),
        'resume_from': args.resume_from, 'recipe': args.recipe, 'runtime': runtime_record()}
    if args.resume_from:
        initial = read(Path(args.resume_from) / 'resume_point.pth')
        if initial['optimizer_steps'] != args.warmup or initial['curriculum_probe']['identity'] != inputs['identity']:
            raise ValueError('resume checkpoint does not match this benchmark')
        atomic_torch_save(initial, config['supervised']['state_file'])
        del initial
    from mortal.supervised.train_supervised import train
    probe = BenchmarkProbe(config, inputs['domains'], recipe=args.recipe, seed=inputs['seed'],
        output=output, horizons=[args.warmup + args.updates], eval_splits=inputs['roles'],
        identity=inputs['identity'], warmup=args.warmup, record=record, evaluate_final=not args.skip_validation)
    with ResourceSampler() as resources:
        train(probe=probe, stage_label='Isolated preparation benchmark', checkpoint_label='diagnostic')
    result_path = output / f'update_{probe.stop_at:07d}.pth'
    result = read(result_path)
    record.update(resources=resources.report(), final_update=result['optimizer_steps'],
        skipped_updates=result['skipped_optimizer_steps'], learned_state_sha256=learned_state_digest(result),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), completed_at=time.time())
    if args.compare_to:
        expected_dir = Path(args.compare_to)
        expected = read(expected_dir / result_path.name)
        checks = {'learned_state_exact': learned_state_digest(expected) == learned_state_digest(result)}
        for key in ('steps', 'optimizer_steps', 'skipped_optimizer_steps', 'auxiliary_optimizer_steps'):
            checks[key + '_exact'] = equal(expected[key], result[key])
        for key in ('dataset', 'rng', 'observed'):
            checks[key + '_exact'] = equal(expected['curriculum_probe'][key], result['curriculum_probe'][key])
        if not args.skip_validation:
            before = json.loads((expected_dir / result_path.with_suffix('.json').name).read_text())
            after = json.loads(result_path.with_suffix('.json').read_text())
            checks['all_metrics_exact'] = before['splits'] == after['splits']
            checks['exposure_exact'] = before['exposure'] == after['exposure']
        record['checks'] = checks
        record['passed'] = all(checks.values())
    atomic_write_json(output / 'result.json', record)
    print(json.dumps({'output': str(output), 'updates_per_second': record.get('updates_per_second'),
        'peak_cuda_gib': record['peak_cuda_allocated_bytes'] / 2**30,
        'minimum_system_free_ram_gib': record['resources']['min_available_ram_bytes'] / 2**30,
        'passed': record.get('passed')}), flush=True)
    if record.get('passed') is False:
        raise RuntimeError('ordered preparation changed matched training state or validation')


def audit_inputs(args):
    inputs, output, config = configure(args)
    from mortal.supervised.train_supervised import safe_default_collate
    from torch.utils.data import DataLoader
    probe = CurriculumProbe(config, inputs['domains'], recipe=args.recipe, seed=inputs['seed'],
        output=output, horizons=[1], eval_splits={}, identity=inputs['identity'])
    dataset = probe.build_dataset({'version': config['control']['version'],
        'enable_augmentation': True, 'augmented_first': False,
        'emit_opponent_state_labels': True, 'track_danger_labels': True})
    def loader(data):
        return iter(DataLoader(data, batch_size=256, num_workers=0, collate_fn=safe_default_collate,
            generator=torch.Generator().manual_seed(inputs['seed'])))
    iterator = loader(dataset)
    hashes, saved, saved_at = [], None, 0
    before = capture_rng()
    try:
        for index in range(args.batches):
            hashes.extend(batch_fingerprints(next(iterator)))
            if index == 2:
                saved, saved_at = dataset.state_dict(), len(hashes)
        final = dataset.state_dict()
    finally:
        iterator._dataset_fetcher.dataset_iter.close()
    probe.dataset, probe.pending_dataset = None, saved
    restored = probe.build_dataset(dataset.loader_kwargs)
    iterator = loader(restored)
    try:
        actual = list(itertools.chain.from_iterable(batch_fingerprints(next(iterator))
                      for _ in range(args.batches - 3)))
    finally:
        iterator._dataset_fetcher.dataset_iter.close()
    record = {'format': 'sl_ordered_preparation_input_audit_v1', 'inputs_identity': inputs['identity'],
        'workers': args.workers, 'samples': len(hashes), 'runtime': runtime_record(),
        'ordered_fields_sha256': stable_json_digest(hashes),
        'resumed_fields_exact': actual == hashes[saved_at:],
        'resumed_cursor_exact': equal(final, restored.state_dict()), 'model_rng_unchanged': equal(before, capture_rng())}
    atomic_write_json(output / 'ordered_sample_hashes.json', hashes)
    atomic_torch_save(final, output / 'cursor.pth')
    if args.compare_to:
        previous = Path(args.compare_to)
        expected = json.loads((previous / 'ordered_sample_hashes.json').read_text())
        record['ordered_fields_exact'] = expected == hashes
        record['consumed_cursor_exact'] = equal(read(previous / 'cursor.pth'), final)
    record['passed'] = all(value for key, value in record.items() if key.endswith(('_exact', '_unchanged')))
    atomic_write_json(output / 'result.json', record)
    print(json.dumps(record), flush=True)
    if not record['passed']:
        raise RuntimeError('prepared input audit failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--eval-files', type=int, choices=(4, 8, 16, 32), default=16)
    for mode in ('train', 'audit'):
        sub = modes.add_parser(mode)
        sub.add_argument('--inputs', required=True)
        sub.add_argument('--output', required=True)
        sub.add_argument('--workers', type=int, choices=range(9), default=0)
        sub.add_argument('--val-workers', type=int, choices=range(9), default=0)
        sub.add_argument('--val-file-batch-size', type=int, choices=(1, 2, 4, 8), default=4)
        sub.add_argument('--rayon-threads', type=int, choices=(1, 2, 4, 8), default=4)
        sub.add_argument('--recipe', choices=RECIPES, default='A')
        sub.add_argument('--compare-to')
        if mode == 'train':
            sub.add_argument('--updates', type=int, choices=(8, 16, 32, 64, 128), default=32)
            sub.add_argument('--warmup', type=int, choices=(4, 8), default=4)
            sub.add_argument('--skip-validation', action='store_true')
            sub.add_argument('--resume-from')
            sub.add_argument('--gpu-memory-fraction', type=float, default=0.3)
        else:
            sub.add_argument('--batches', type=int, choices=(16, 32, 64), default=32)
    args = parser.parse_args()
    {'prepare': prepare_inputs, 'train': run_benchmark, 'audit': audit_inputs}[args.mode](args)


if __name__ == '__main__':
    main()

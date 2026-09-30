"""Measure real Oracle updates from a copied full checkpoint without publishing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

from profile_oracle_runtime import Sampler, digest, dump


class TimedIterator:
    def __init__(self, iterator, waits, hashes, hash_batches):
        self.iterator, self.waits, self.hashes = iterator, waits, hashes
        self.hash_batches = hash_batches

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.iterator, name)

    def __next__(self):
        start = time.perf_counter()
        batch = next(self.iterator)
        self.waits.append(time.perf_counter() - start)
        if len(self.hashes) < self.hash_batches:
            h = hashlib.sha256()
            for tensor in batch:
                array = tensor.contiguous().numpy()
                h.update(f'{array.dtype}:{array.shape}'.encode())
                h.update(array.tobytes())
            self.hashes.append(h.hexdigest())
        return batch


class TimedLoader:
    def __init__(self, loader, waits, hashes, hash_batches):
        self.loader, self.waits, self.hashes = loader, waits, hashes
        self.hash_batches = hash_batches

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        return TimedIterator(iter(self.loader), self.waits, self.hashes, self.hash_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--rayon', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--torch-threads', type=int, default=0)
    parser.add_argument('--hash-batches', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--gpu-fraction', type=float, default=.72)
    parser.add_argument('--import-prelude', action='store_true',
                        help='Match the production module entry point initialization.')
    args = parser.parse_args()
    if not 0 < args.steps <= 2000 or not 0 <= args.warmup < args.steps:
        parser.error('invalid bounded update count')
    if not 0 < args.gpu_fraction <= .8 or args.rayon < 1 or args.prefetch < 1:
        parser.error('invalid resource limits')
    source_run, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    if output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('diagnostic outputs must not overlap the production run')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source_run / 'runtime_manifest.json').read_text(encoding='utf-8'))
    source_root = Path(manifest['source_root'])
    for name, expected in manifest['source_sha256'].items():
        if digest(source_root / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    if digest(source_run / 'critic_config.toml') != manifest['config']['sha256']:
        raise ValueError('frozen configuration changed')
    import psutil
    if psutil.virtual_memory().available < 10 * 2**30:
        raise RuntimeError('real-update benchmark requires at least 10 GiB available RAM')
    for process in psutil.process_iter(['name', 'cmdline']):
        if (process.info['name'] or '').lower() in ('r5apex.exe', 'r5apex_dx12.exe'):
            raise RuntimeError('Apex is running')
        if '-m' in (process.info['cmdline'] or []) and 'mortal.online.pretrain_oracle_critic' in process.info['cmdline']:
            raise RuntimeError('another production Oracle trainer is running')
    import toml
    cfg = toml.loads((source_run / 'critic_config.toml').read_text(encoding='utf-8'))
    pre_cfg = cfg['oracle_critic_pretrain']
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = digest(checkpoint)
    for key, filename in (('state_file', 'latest.pth'), ('best_state_file', 'best_dev.pth'),
                          ('best_primary_state_file', 'best_primary.pth'),
                          ('adaptive_best_state_file', 'adaptive_best.pth')):
        pre_cfg[key] = str(output / 'checkpoints' / filename)
    pre_cfg['metrics_file'] = str(output / 'metrics.jsonl')
    pre_cfg['tensorboard_dir'] = str(output / 'tb_log')
    pre_cfg['run_name'] = output.name
    pre_cfg['rayon_num_threads'] = args.rayon
    pre_cfg['prefetch_factor'] = args.prefetch
    (output / 'checkpoints').mkdir()
    shutil.copyfile(checkpoint, pre_cfg['state_file'])
    shutil.copyfile(source_run / 'critic/checkpoints/validation_inputs.json',
                    output / 'checkpoints/validation_inputs.json')
    case_config = output / 'config.toml'
    case_config.write_text(toml.dumps(cfg), encoding='utf-8', newline='\n')
    os.environ.update(MORTAL_CFG=str(case_config), PYTHONPATH=str(source_root),
                      MORTAL_ORACLE_PAUSE_FILE=str(output / 'pause.request'),
                      RAYON_NUM_THREADS=str(args.rayon), PYTHONDONTWRITEBYTECODE='1')
    if os.environ.get('MORTAL_CPU_AFFINITY'):
        raise RuntimeError('benchmark does not opt in to CPU affinity')
    sys.path.insert(0, str(source_root))
    import torch
    from mortal.online import pretrain_oracle_critic as pre
    if args.import_prelude:
        import mortal.core.prelude  # noqa: F401
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    start_step = int(state['steps'])
    target_step = start_step + args.steps
    if target_step >= ((start_step // 10000) + 1) * 10000:
        raise ValueError('training-only probe must not cross a scientific evaluation gate')
    del state
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    identity = {'source_run': str(source_run), 'source_manifest_sha256': digest(source_run / 'runtime_manifest.json'),
                'native_sha256': digest(source_root / 'libriichi.pyd'), 'config_sha256': digest(case_config),
                'input_checkpoint_sha256': checkpoint_hash, 'start_step': start_step,
                'target_step': target_step, 'torch_threads': torch.get_num_threads(),
                'arguments': vars(args), 'script_sha256': digest(__file__),
                'started_at': datetime.now(timezone.utc).isoformat()}
    dump(output / 'identity.json', identity)
    step_times, waits, hashes = [], [], []
    original_loader = pre.make_loader
    def make_loader(dataset, config, *, train):
        loader = original_loader(dataset, config, train=train)
        return TimedLoader(loader, waits, hashes, args.hash_batches) if train else loader
    pre.make_loader = make_loader
    original_pause_due = pre.external_pause_due
    started = time.perf_counter()
    reason = None
    with Sampler(output, os.getpid()) as sampler:
        def pause_due(path, steps, max_steps):
            nonlocal reason
            step_times.append({'step': steps, 'time': time.perf_counter() - started})
            if steps >= target_step:
                reason = 'requested_updates_completed'
            elif time.perf_counter() - started > args.max_seconds:
                reason = 'time_limit'
            elif sampler.rows and sampler.rows[-1]['ram_percent'] >= 88:
                reason = 'RAM_reserve'
            elif sampler.rows and sampler.rows[-1].get('gpu_used_mib', 0) >= 14500:
                reason = 'VRAM_reserve'
            elif any((p.info['name'] or '').lower() in ('r5apex.exe', 'r5apex_dx12.exe')
                     for p in psutil.process_iter(['name'])):
                reason = 'Apex_started'
            return reason is not None or original_pause_due(path, steps, max_steps)
        pre.external_pause_due = pause_due
        sys.argv = [str(source_root / 'mortal/online/pretrain_oracle_critic.py')]
        try:
            code = pre.train()
        except Exception:
            dump(output / 'failed_resources.json', sampler.report())
            raise
    durations = [b['time'] - a['time'] for a, b in zip(step_times, step_times[1:])][args.warmup:]
    report = {'returncode': code, 'reason': reason, 'start_step': start_step,
              'last_step': step_times[-1]['step'] if step_times else None,
              'updates': len(step_times), 'wall_s': time.perf_counter() - started,
              'steady_steps': len(durations), 'steady_steps_per_s': len(durations) / sum(durations) if durations else None,
              'steady_samples_per_s': pre_cfg['batch_size'] * len(durations) / sum(durations) if durations else None,
              'step_median_s': statistics.median(durations) if durations else None,
              'data_wait_s': sum(waits[args.warmup + 1:]), 'input_batch_hashes': hashes,
              'checkpoint_sha256': digest(pre_cfg['state_file']),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'resources': sampler.report()}
    dump(output / 'steps.json', step_times)
    dump(output / 'result.json', report)
    print(json.dumps(report), flush=True)
    if code != 75 or reason != 'requested_updates_completed':
        raise RuntimeError('bounded probe did not finish all requested updates')


if __name__ == '__main__':
    main()

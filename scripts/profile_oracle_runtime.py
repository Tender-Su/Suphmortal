"""Bounded Oracle resource diagnostics; read frozen runs, write separate artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_sample():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        return {'gpu_error': result.stderr[:200]}
    values = result.stdout.strip().splitlines()[0].split(',')
    return dict(zip(('gpu_percent', 'gpu_used_mib', 'gpu_total_mib', 'gpu_watts', 'gpu_c'),
                    map(float, values)))


class Sampler:
    """Independent, low-rate counters; no WMI, tracing injection or affinity changes."""
    def __init__(self, output, pid, interval=2.0):
        import psutil
        self.psutil = psutil
        self.output, self.pid, self.interval = output, pid, interval
        self.rows, self.processes = [], {}
        self.stop = threading.Event()
        self.thread = None
        self.error = None

    def __enter__(self):
        self.started = time.perf_counter()
        self.psutil.cpu_percent()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        try:
            with (self.output / 'resources.jsonl').open('w', encoding='utf-8') as stream:
                while not self.stop.is_set():
                    started = time.perf_counter()
                    ram = self.psutil.virtual_memory()
                    row = {'time': datetime.now(timezone.utc).isoformat(),
                           'elapsed_s': started - self.started,
                           'cpu_percent': self.psutil.cpu_percent(),
                           'ram_percent': ram.percent, 'available_ram_bytes': ram.available}
                    row.update(gpu_sample())
                    try:
                        root = self.psutil.Process(self.pid)
                        tree = [root, *root.children(recursive=True)]
                    except self.psutil.NoSuchProcess:
                        tree = []
                    members = []
                    for process in tree:
                        try:
                            process = self.processes.setdefault(process.pid, process)
                            mem = process.memory_info()
                            members.append({'pid': process.pid, 'cpu_percent': process.cpu_percent(),
                                            'rss': mem.rss, 'private': getattr(mem, 'private', None)})
                        except (self.psutil.AccessDenied, self.psutil.NoSuchProcess):
                            pass
                    row['processes'] = members
                    row['tree_rss_bytes'] = sum(p['rss'] for p in members)
                    row['tree_cpu_percent'] = sum(p['cpu_percent'] for p in members)
                    row['sample_cost_s'] = time.perf_counter() - started
                    self.rows.append(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    self.stop.wait(max(0, self.interval - row['sample_cost_s']))
        except Exception as exc:
            self.error = repr(exc)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)

    def report(self):
        result = {'samples': len(self.rows), 'error': self.error,
                  'limitation': 'Periodic counters can miss transient peaks; scheduler latency is not a UI latency measurement.'}
        for name in ('gpu_percent', 'gpu_used_mib', 'gpu_watts', 'gpu_c', 'cpu_percent',
                     'ram_percent', 'available_ram_bytes', 'tree_rss_bytes', 'tree_cpu_percent'):
            values = sorted(r[name] for r in self.rows if name in r)
            if values:
                result[name] = {'min': min(values), 'median': statistics.median(values),
                                'p95': values[int((len(values) - 1) * .95)], 'max': max(values)}
        return result


def initialize(args):
    source_run = Path(args.source_run).resolve()
    output = Path(args.output).resolve()
    if output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('output must be outside the source run')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source_run / 'runtime_manifest.json').read_text(encoding='utf-8'))
    source_root = Path(manifest['source_root'])
    identity = {'config_sha256': digest(source_run / 'critic_config.toml'),
                'source_manifest_sha256': digest(source_run / 'runtime_manifest.json'),
                'native_sha256': digest(source_root / 'libriichi.pyd'),
                'source_run': str(source_run), 'source_root': str(source_root),
                'script_sha256': digest(__file__), 'mode': args.mode,
                'started_at': datetime.now(timezone.utc).isoformat(), 'arguments': vars(args)}
    if identity['config_sha256'] != manifest['config']['sha256']:
        raise ValueError('source config differs from manifest')
    if identity['native_sha256'] != manifest['source_sha256']['libriichi.pyd']:
        raise ValueError('source native differs from manifest')
    dump(output / 'identity.json', identity)
    return source_run, source_root, output


def monitor(args):
    source_run, _, output = initialize(args)
    latencies = []
    with Sampler(output, args.pid) as sampler:
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            time.sleep(.02)
            latencies.append(max(0, time.perf_counter() - start - .02))
    latencies.sort()
    report = {'resources': sampler.report(), 'seconds': args.seconds,
              'scheduler_delay_p95_ms': 1000 * latencies[int((len(latencies) - 1) * .95)],
              'scheduler_delay_max_ms': 1000 * max(latencies)}
    dump(output / 'result.json', report)
    print(json.dumps(report), flush=True)


def inputs(args):
    source_run, source_root, output = initialize(args)
    os.environ['MORTAL_CFG'] = str(source_run / 'critic_config.toml')
    os.environ['PYTHONPATH'] = str(source_root)
    if args.rayon:
        os.environ['RAYON_NUM_THREADS'] = str(args.rayon)
    sys.path.insert(0, str(source_root))
    import torch
    from mortal.online import pretrain_oracle_critic as pre
    from mortal.data.oracle_value import deterministic_game_id
    pre.sanitize_sys_path_for_spawn()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    cfg = dict(pre.oracle_pretrain_cfg())
    train = args.split == 'train'
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    prefix = '' if train else 'val_'
    for key, value in (('num_workers', args.workers), ('file_batch_size', args.file_batch),
                       ('prefetch_factor', args.prefetch)):
        if value is not None:
            cfg[prefix + key] = value
    if args.rayon:
        cfg['rayon_num_threads'] = args.rayon
    files = pre.load_file_index(cfg['train_file_index' if train else 'dev_file_index'])
    if args.files:
        files = files[:args.files]
    original_files = len(files)
    modulus = int(cfg.get('val_game_id_modulus', 1))
    remainders = cfg.get('val_game_id_remainders', [0])
    if args.early_filter:
        if train:
            raise ValueError('early_filter is validation only')
        # Validation disables synthetic names in bulk_event_cache. Both raw logs
        # and .pt cache payloads emit the original filename as their game ID key.
        files = [f for f in files if int(deterministic_game_id(str(f))) % modulus in remainders]
    dataset = pre.make_dataset(files, cfg, train=train)
    loader = pre.make_loader(dataset, cfg, train=train)
    row_hashes, batch_times = [], []
    predictions, targets = [], []
    samples = selected = batches = 0
    measured_started = None
    measured_samples = 0
    import psutil
    if psutil.virtual_memory().available < args.min_available_gib * 2**30:
        raise RuntimeError('insufficient available RAM for diagnostic')
    brain = value_net = None
    if args.forward:
        if train:
            raise ValueError('forward-only diagnostic is validation only')
        gpu = gpu_sample()
        limit_mib = gpu['gpu_total_mib'] * args.gpu_fraction
        if gpu['gpu_total_mib'] - gpu['gpu_used_mib'] < limit_mib + 1024:
            raise RuntimeError('insufficient free VRAM plus reserve')
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        allow_tf32 = bool(cfg.get('allow_tf32', pre.config['control'].get('allow_tf32', True)))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(pre.config['control'].get('enable_cudnn_benchmark', True))
        torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
        model_keys = ('critic_arch', 'oracle_fusion_init', 'oracle_fusion_mode',
                      'oracle_fusion_hidden', 'exact_zero_sum', 'value_loss_mode', 'value_head_hidden')
        brain, value_net = pre.build_models(torch.device('cpu'), **{k: cfg[k] for k in model_keys})
        checkpoint = Path(args.checkpoint) if args.checkpoint else source_run / 'critic/checkpoints/adaptive_best.pth'
        checkpoint_hash = digest(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        brain.load_state_dict(state['oracle_brain'])
        value_net.load_state_dict(state['value_net'])
        del state
        brain.to('cuda').eval()
        value_net.to('cuda').eval()
    started = time.perf_counter()
    with Sampler(output, os.getpid()) as sampler:
        iterator = iter(loader)
        try:
            while True:
                before = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                batch_times.append(time.perf_counter() - before)
                batches += 1
                samples += len(batch[0])
                if not train:
                    keep = torch.tensor([int(g) % modulus in remainders for g in batch[4]])
                    kept = [v[keep] for v in batch]
                else:
                    kept = batch
                selected += len(kept[0])
                if args.forward and len(kept[0]):
                    with torch.inference_mode():
                        for offset in range(0, len(kept[0]), args.forward_batch):
                            visible = kept[0][offset:offset + args.forward_batch].to('cuda')
                            invisible = kept[1][offset:offset + args.forward_batch].to('cuda')
                            predictions.append(pre.model_forward(brain, value_net, visible, invisible,
                                enable_amp=bool(cfg.get('eval_enable_amp', False)), device_type='cuda').cpu())
                            targets.append(kept[2][offset:offset + args.forward_batch].clone())
                if args.hash_inputs:
                    arrays = [v.contiguous().numpy() for v in kept]
                    for row in zip(*arrays):
                        h = hashlib.sha256()
                        for value in row:
                            h.update(value.tobytes())
                        row_hashes.append(h.hexdigest())
                if args.consumer_seconds:
                    time.sleep(args.consumer_seconds)
                if batches == args.warmup_batches:
                    measured_started = time.perf_counter()
                elif batches > args.warmup_batches:
                    measured_samples += len(kept[0])
                if psutil.virtual_memory().available < args.min_available_gib * 2**30:
                    raise RuntimeError('available RAM crossed diagnostic reserve')
                if any(p.info['name'].lower() in ('r5apex.exe', 'r5apex_dx12.exe')
                       for p in psutil.process_iter(['name']) if p.info['name']):
                    raise RuntimeError('Apex appeared; diagnostic stopped')
                if args.batches and batches >= args.batches:
                    break
        finally:
            pre.shutdown_data_loader_iterator(loader, iterator)
    elapsed = time.perf_counter() - started
    report = {'files_before_subset': original_files, 'files_loaded': len(files),
              'batches': batches, 'samples_loaded': samples, 'selected_samples': selected,
              'elapsed_s': elapsed, 'input_wait_s': sum(batch_times),
              'selected_samples_per_s': selected / elapsed,
              'batch_wait_median_s': statistics.median(batch_times) if batch_times else None,
              'batch_wait_p95_s': sorted(batch_times)[int((len(batch_times) - 1) * .95)] if batch_times else None,
              'ordered_input_sha256': hashlib.sha256(''.join(row_hashes).encode()).hexdigest() if row_hashes else None,
              'multiset_input_sha256': hashlib.sha256(''.join(sorted(row_hashes)).encode()).hexdigest() if row_hashes else None,
              'data_stream_signature': pre.data_stream_signature(cfg),
              'resources': sampler.report()}
    if measured_started is not None:
        measured_elapsed = time.perf_counter() - measured_started
        waits = batch_times[args.warmup_batches:]
        report['steady'] = {'batches': len(waits), 'samples': measured_samples,
                            'elapsed_s': measured_elapsed, 'samples_per_s': measured_samples / measured_elapsed,
                            'input_wait_s': sum(waits), 'simulated_consumer_seconds': args.consumer_seconds}
    if row_hashes:
        dump(output / 'sample_hashes.json', row_hashes)
    if predictions:
        pred, target = torch.cat(predictions), torch.cat(targets)
        report['forward'] = {'checkpoint_sha256': checkpoint_hash, 'samples': len(pred),
                             'loss': float((pred - target).square().mean()),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        torch.save({'pred': pred, 'target': target}, output / 'predictions.pt')
    dump(output / 'result.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('monitor', 'inputs'))
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--split', choices=('dev', 'train'), default='dev')
    parser.add_argument('--files', type=int, default=256)
    parser.add_argument('--batches', type=int, default=0)
    parser.add_argument('--warmup-batches', type=int, default=4)
    parser.add_argument('--consumer-seconds', type=float, default=0)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--file-batch', type=int)
    parser.add_argument('--prefetch', type=int)
    parser.add_argument('--rayon', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--forward-batch', type=int, default=128)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--gpu-fraction', type=float, default=.2)
    parser.add_argument('--min-available-gib', type=float, default=6)
    parser.add_argument('--early-filter', action='store_true')
    parser.add_argument('--hash-inputs', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0 or args.files < 0 or args.batches < 0:
        parser.error('invalid diagnostic bounds')
    if not 0 < args.gpu_fraction <= .3 or args.forward_batch <= 0:
        parser.error('invalid diagnostic GPU limits')
    if args.mode == 'monitor' and not args.pid:
        parser.error('--pid is required for monitoring')
    {'monitor': monitor, 'inputs': inputs}[args.mode](args)


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'mortal'))

import run_stage05_ab as ab
from dataloader import SupervisedFileDatasetsIter, resolve_rayon_num_threads, worker_init_fn


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', wintypes.DWORD),
        ('dwMemoryLoad', wintypes.DWORD),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]


kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
psapi = ctypes.WinDLL('psapi', use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
    wintypes.DWORD,
]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def system_memory() -> dict[str, float]:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    used = stat.ullTotalPhys - stat.ullAvailPhys
    return {'used': used, 'total': stat.ullTotalPhys, 'percent': float(stat.dwMemoryLoad)}


def process_memory() -> dict[str, int]:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return {'rss': 0, 'private': 0}
    return {'rss': int(counters.WorkingSetSize), 'private': int(counters.PrivateUsage)}


def gb(value: float | int) -> float:
    return round(float(value) / (1024 ** 3), 3)


def build_dataloader_kwargs(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    val_prefetch_factor: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        'dataset': dataset,
        'batch_size': batch_size,
        'drop_last': False,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        kwargs['worker_init_fn'] = worker_init_fn
        kwargs['prefetch_factor'] = val_prefetch_factor
        kwargs['persistent_workers'] = False
        kwargs['in_order'] = True
    return kwargs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('monitor_recent', 'full_recent', 'old_regression'), default='monitor_recent')
    parser.add_argument('--max-batches', type=int, default=64)
    parser.add_argument('--val-file-batch-size', type=int, default=0)
    parser.add_argument('--val-prefetch-factor', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--enable-opponent', action='store_true')
    parser.add_argument('--enable-danger', action='store_true')
    parser.add_argument('--output', default=str(REPO_ROOT / 'logs' / 'validation_mem_probe.json'))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def dump(state: dict) -> None:
        output_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    base_cfg = ab.build_base_config()
    grouped = ab.group_files_by_month(ab.load_all_files())
    eval_splits = ab.build_eval_splits(grouped, ab.BASE_SCREENING['seed'] + 55, ab.BASE_SCREENING['eval_files'])

    supervised_cfg = base_cfg['supervised']
    dataset_cfg = base_cfg['dataset']
    control_cfg = base_cfg['control']

    version = int(control_cfg['version'])
    batch_size = int(supervised_cfg.get('batch_size', control_cfg['batch_size']))
    val_file_batch_size = int(supervised_cfg.get('val_file_batch_size', supervised_cfg.get('file_batch_size', dataset_cfg['file_batch_size'])))
    num_workers = int(supervised_cfg.get('num_workers', dataset_cfg['num_workers']))
    val_prefetch_factor = int(supervised_cfg.get('val_prefetch_factor', supervised_cfg.get('prefetch_factor', 2)))
    if args.val_file_batch_size > 0:
        val_file_batch_size = int(args.val_file_batch_size)
    if args.num_workers >= 0:
        num_workers = int(args.num_workers)
    if args.val_prefetch_factor > 0:
        val_prefetch_factor = int(args.val_prefetch_factor)
    worker_torch_num_threads = int(supervised_cfg.get('worker_torch_num_threads', dataset_cfg.get('worker_torch_num_threads', 1)))
    worker_torch_num_interop_threads = int(supervised_cfg.get('worker_torch_num_interop_threads', dataset_cfg.get('worker_torch_num_interop_threads', 1)))
    explicit_rayon_threads = int(supervised_cfg.get('rayon_num_threads', dataset_cfg.get('rayon_num_threads', 0)))
    rayon_num_threads = resolve_rayon_num_threads(num_workers, val_file_batch_size, explicit_rayon_threads)
    files = list(eval_splits[args.mode + '_files'])

    if rayon_num_threads > 0:
        os.environ['RAYON_NUM_THREADS'] = str(rayon_num_threads)

    vm0 = system_memory()
    pm0 = process_memory()
    state = {
        'phase': 'init',
        'mode': args.mode,
        'files': len(files),
        'batch_size': batch_size,
        'val_file_batch_size': val_file_batch_size,
        'num_workers': num_workers,
        'val_prefetch_factor': val_prefetch_factor,
        'rayon_num_threads': rayon_num_threads,
        'opponent_labels': bool(args.enable_opponent),
        'danger_labels': bool(args.enable_danger),
        'max_batches': args.max_batches,
        'batches_completed': 0,
        'start_sys_used_gb': gb(vm0['used']),
        'start_sys_percent': vm0['percent'],
        'start_proc_rss_gb': gb(pm0['rss']),
        'peak_sys_used_gb': gb(vm0['used']),
        'peak_sys_percent': vm0['percent'],
        'peak_proc_rss_gb': gb(pm0['rss']),
        'peak_proc_private_gb': gb(pm0['private']),
        'error': None,
    }
    dump(state)

    stop_flag = False
    peak_sys_used = vm0['used']
    peak_sys_percent = vm0['percent']
    peak_proc_rss = pm0['rss']
    peak_proc_private = pm0['private']

    def monitor() -> None:
        nonlocal peak_sys_used, peak_sys_percent, peak_proc_rss, peak_proc_private
        while not stop_flag:
            vm = system_memory()
            pm = process_memory()
            peak_sys_used = max(peak_sys_used, vm['used'])
            peak_sys_percent = max(peak_sys_percent, vm['percent'])
            peak_proc_rss = max(peak_proc_rss, pm['rss'])
            peak_proc_private = max(peak_proc_private, pm['private'])
            time.sleep(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    start = time.perf_counter()
    try:
        dataset = SupervisedFileDatasetsIter(
            version=version,
            file_list=files,
            oracle=False,
            file_batch_size=val_file_batch_size,
            reserve_ratio=0.0,
            player_names=None,
            num_epochs=1,
            enable_augmentation=False,
            augmented_first=False,
            shuffle_files=False,
            worker_torch_num_threads=worker_torch_num_threads,
            worker_torch_num_interop_threads=worker_torch_num_interop_threads,
            rayon_num_threads=rayon_num_threads,
            emit_opponent_state_labels=bool(args.enable_opponent),
            track_danger_labels=bool(args.enable_danger),
        )
        loader = DataLoader(
            **build_dataloader_kwargs(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                val_prefetch_factor=val_prefetch_factor,
            )
        )
        for batch in loader:
            _ = [torch.as_tensor(x) if (x is not None and not torch.is_tensor(x)) else x for x in batch]
            state['batches_completed'] += 1
            state['phase'] = 'running'
            state['peak_sys_used_gb'] = gb(peak_sys_used)
            state['peak_sys_percent'] = peak_sys_percent
            state['peak_proc_rss_gb'] = gb(peak_proc_rss)
            state['peak_proc_private_gb'] = gb(peak_proc_private)
            state['elapsed_sec'] = round(time.perf_counter() - start, 3)
            if state['batches_completed'] % 4 == 0:
                dump(state)
            if state['batches_completed'] >= args.max_batches:
                break
        state['phase'] = 'completed'
    except Exception as exc:
        state['phase'] = 'failed'
        state['error'] = repr(exc)
    finally:
        stop_flag = True
        thread.join(timeout=2.0)
        vm1 = system_memory()
        pm1 = process_memory()
        state['elapsed_sec'] = round(time.perf_counter() - start, 3)
        state['end_sys_used_gb'] = gb(vm1['used'])
        state['end_sys_percent'] = vm1['percent']
        state['end_proc_rss_gb'] = gb(pm1['rss'])
        state['peak_sys_used_gb'] = gb(peak_sys_used)
        state['peak_sys_percent'] = peak_sys_percent
        state['peak_proc_rss_gb'] = gb(peak_proc_rss)
        state['peak_proc_private_gb'] = gb(peak_proc_private)
        state['avg_batches_per_sec'] = round(state['batches_completed'] / state['elapsed_sec'], 3) if state['elapsed_sec'] > 0 else None
        dump(state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

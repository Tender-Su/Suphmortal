"""Process-local, opt-in measurements for isolated SL and evaluation diagnostics."""
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import inspect
import os
import threading
import time
from unittest.mock import patch


class PhaseTimes:
    def __init__(self, synchronize=None, clock=time.perf_counter):
        self.synchronize = synchronize or (lambda: None)
        self.clock = clock
        self.stack = []
        self.stats = defaultdict(lambda: {'calls': 0, 'inclusive_s': 0.0, 'exclusive_s': 0.0})

    @contextmanager
    def measure(self, name, *, sync=False):
        if sync:
            self.synchronize()
        start = self.clock()
        frame = [name, 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            if sync:
                self.synchronize()
            elapsed = self.clock() - start
            assert self.stack.pop() is frame
            item = self.stats[name]
            item['calls'] += 1
            item['inclusive_s'] += elapsed
            item['exclusive_s'] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, function, name, *, sync=False):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with self.measure(name, sync=sync):
                return function(*args, **kwargs)
        return wrapped

    def report(self):
        return dict(sorted(self.stats.items(), key=lambda item: -item[1]['exclusive_s']))


def closure_patch(stack, function, name, replacement):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    cell = cells[name]
    previous = cell.cell_contents
    cell.cell_contents = replacement
    stack.callback(setattr, cell, 'cell_contents', previous)


def batch_fingerprints(batch):
    """Hash every field of every ordered CPU sample, including auxiliary labels."""
    arrays = [value.detach().cpu().contiguous().numpy() for value in batch]
    if not arrays or any(len(value) != len(arrays[0]) for value in arrays):
        raise ValueError('inconsistent collated sample counts')
    headers = [f'{value.dtype}:{value.shape[1:]}'.encode() for value in arrays]
    hashes = []
    for index in range(len(arrays[0])):
        digest = hashlib.sha256()
        for header, value in zip(headers, arrays):
            digest.update(len(header).to_bytes(4, 'little'))
            digest.update(header)
            digest.update(value[index].tobytes())
        hashes.append(digest.hexdigest())
    return hashes


@contextmanager
def validation_input_hashes(sample_hashes):
    import mortal.supervised.train_supervised as train

    original = train.safe_default_collate

    def collate(batch):
        result = original(batch)
        sample_hashes.extend(batch_fingerprints(result))
        return result

    with patch.object(train, 'safe_default_collate', collate):
        yield


def install_tensor_and_model_timers(stack, phases):
    import torch
    from mortal.core.model import AuxNet, Brain, CategoricalPolicy, DangerAuxNet, OpponentStateAuxNet

    for cls, method, label in (
        (Brain, 'forward', 'brain_forward'),
        (CategoricalPolicy, 'logits', 'policy_head'),
        (AuxNet, 'forward', 'rank_head'),
        (OpponentStateAuxNet, 'forward', 'opponent_head'),
        (DangerAuxNet, 'forward', 'danger_head'),
    ):
        stack.enter_context(patch.object(cls, method, phases.wrap(getattr(cls, method), label, sync=True)))

    original_to = torch.Tensor.to

    @wraps(original_to)
    def timed_to(tensor, *args, **kwargs):
        destination = kwargs.get('device')
        if destination is None and args and isinstance(args[0], (str, torch.device)):
            destination = args[0]
        if destination is not None:
            destination = torch.device(destination)
            if tensor.device.type != destination.type:
                label = 'h2d' if destination.type == 'cuda' else 'd2h'
                with phases.measure(label, sync=True):
                    return original_to(tensor, *args, **kwargs)
        return original_to(tensor, *args, **kwargs)

    stack.enter_context(patch.object(torch.Tensor, 'to', timed_to))
    for method in ('cpu', 'tolist', 'item'):
        original = getattr(torch.Tensor, method)

        def timed_output(tensor, *args, _original=original, _method=method, **kwargs):
            if tensor.is_cuda:
                with phases.measure('gpu_output_' + _method, sync=True):
                    return _original(tensor, *args, **kwargs)
            return _original(tensor, *args, **kwargs)

        stack.enter_context(patch.object(torch.Tensor, method, timed_output))


@contextmanager
def validation_timers(evaluate, phases, sample_hashes):
    import mortal.data.dataloader as data
    import mortal.supervised.train_supervised as train
    from torch.utils.data._utils.fetch import _IterableDatasetFetcher
    from torch.utils.data._utils import pin_memory

    with ExitStack() as stack:
        install_tensor_and_model_timers(stack, phases)
        original_loaded = data.iter_loaded_gameplay_batches

        def timed_loaded(*args, **kwargs):
            iterator = iter(original_loaded(*args, **kwargs))
            while True:
                with phases.measure('native_read_parse_features_labels'):
                    try:
                        value = next(iterator)
                    except StopIteration:
                        return
                yield value

        stack.enter_context(patch.object(data, 'iter_loaded_gameplay_batches', timed_loaded))
        stack.enter_context(patch.object(data.SupervisedFileDatasetsIter, 'populate_buffer',
            phases.wrap(data.SupervisedFileDatasetsIter.populate_buffer, 'column_export_and_rows')))
        stack.enter_context(patch.object(_IterableDatasetFetcher, 'fetch',
            phases.wrap(_IterableDatasetFetcher.fetch, 'batch_input_other')))
        stack.enter_context(patch.object(pin_memory, 'pin_memory',
            phases.wrap(pin_memory.pin_memory, 'pin_memory')))
        original_collate = train.safe_default_collate

        def collate(batch):
            with phases.measure('collate'):
                result = original_collate(batch)
            with phases.measure('diagnostic_input_fingerprints'):
                sample_hashes.extend(batch_fingerprints(result))
            return result

        stack.enter_context(patch.object(train, 'safe_default_collate', collate))
        closure = inspect.getclosurevars(evaluate).nonlocals
        forward = closure['forward_loss']
        forward_closure = inspect.getclosurevars(forward).nonlocals
        for name in ('move_batch_to_device', 'compute_opponent_metrics', 'compute_danger_metrics'):
            original = forward_closure[name]
            closure_patch(stack, forward, name, phases.wrap(original, name, sync=True))
        closure_patch(stack, evaluate, 'forward_loss', phases.wrap(forward, 'loss_and_metrics_other', sync=True))
        for name in ('merge_metrics', 'finalize_metrics'):
            original = closure[name]
            closure_patch(stack, evaluate, name, phases.wrap(original, name, sync=True))
        yield


def windows_memory_snapshot():
    # Same Win32 counters as scripts/probe_validation_memory.py, without its config imports.
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                'total', 'available', 'total_page', 'available_page',
                'total_virtual', 'available_virtual', 'available_extended')]

    class ProcessCounters(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'peak_rss', 'rss', 'peak_paged', 'paged', 'peak_nonpaged',
                'nonpaged', 'pagefile', 'peak_commit', 'private')]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    system, process = MemoryStatus(), ProcessCounters()
    system.length = ctypes.sizeof(system)
    process.size = ctypes.sizeof(process)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'rss_bytes': process.rss, 'private_bytes': process.private,
            'available_ram_bytes': system.available, 'peak_rss_bytes': process.peak_rss,
            'peak_commit_bytes': process.peak_commit}


class ResourceSampler:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = None

    def __enter__(self):
        if os.name != 'nt':
            raise RuntimeError('this resource sampler currently requires Windows')
        self.cpu_start = time.process_time()
        self.started = time.perf_counter()
        self.samples.append({'elapsed_s': 0.0, **windows_memory_snapshot()})

        def sample():
            while not self.stop.is_set():
                self.samples.append({'elapsed_s': time.perf_counter() - self.started,
                                     **windows_memory_snapshot()})
                self.stop.wait(1.0)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('resource sampler did not stop')
        self.wall_s = time.perf_counter() - self.started
        self.cpu_s = time.process_time() - self.cpu_start
        self.samples.append({'elapsed_s': self.wall_s, **windows_memory_snapshot()})

    def report(self):
        return {'wall_s': self.wall_s, 'process_cpu_s': self.cpu_s,
            'average_cpu_cores': self.cpu_s / self.wall_s,
            'max_rss_bytes': max((s['rss_bytes'] for s in self.samples), default=0),
            'max_private_bytes': max((s['private_bytes'] or 0 for s in self.samples), default=0),
            'process_lifetime_peak_rss_bytes': max(s['peak_rss_bytes'] for s in self.samples),
            'process_lifetime_peak_commit_bytes': max(s['peak_commit_bytes'] for s in self.samples),
            'min_available_ram_bytes': min((s['available_ram_bytes'] for s in self.samples), default=0),
            'sample_count': len(self.samples),
            'max_sample_gap_s': max((b['elapsed_s'] - a['elapsed_s'] for a, b in
                                    zip(self.samples, self.samples[1:])), default=0),
            'limitation': 'One-second process sampling can miss transients or stall behind the native GIL.'}

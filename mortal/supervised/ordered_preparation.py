"""Bounded native preparation with ordered, consumer-owned training cursors.

Workers only prepare immutable four-game draws. Speculative sampler positions,
queues and shared buffers are never checkpoint state. Windows shared-memory
handles stay open until the parent acknowledges attachment; returned row views
keep their mapping alive through collation, including batches crossing blocks.
"""
from collections import deque
from copy import deepcopy
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
import time
import traceback

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from mortal.core.artifacts import file_sha256
from mortal.supervised.curriculum_probe import RotatingGameDataset, RotatingGameSampler, derived_seed


class _MappingLease:
    def __init__(self, memory):
        self.memory = memory

    def __del__(self):
        self.memory.close()


class _SharedArray(np.ndarray):
    def __array_finalize__(self, source):
        self.lease = getattr(source, 'lease', None)


def _scalar_kind(value):
    return type(value).__name__ if type(value) in (int, float, bool, str) else None


class PreparedBlock:
    def __init__(self, receipt):
        self.draws = receipt['draws']
        self.counts = receipt['counts']
        self.columns = []
        self.scalar_kinds = []
        for field in receipt['columns']:
            memory = SharedMemory(name=field['name'])
            # On Windows unlink is a no-op; mapping lifetime is handle based.
            memory.unlink()
            array = np.ndarray(field['shape'], dtype=field['dtype'], buffer=memory.buf).view(_SharedArray)
            array.lease = _MappingLease(memory)
            self.columns.append(array)
            self.scalar_kinds.append(field['scalar'])
        self.count = sum(self.counts)

    def row(self, index):
        row = []
        for column, kind in zip(self.columns, self.scalar_kinds):
            value = column[index]
            if kind:
                value = {'int': int, 'float': float, 'bool': bool, 'str': str}[kind](value)
            row.append(value)
        return tuple(row)


def _load_rows(draws, loader_kwargs, file_batch_size, sample_loader):
    from mortal.data.dataloader import SupervisedFileDatasetsIter, stable_source_game_id

    files = list(dict.fromkeys(draw['file'] for draw in draws))
    if sample_loader is not None:
        by_game = {stable_source_game_id(name): sample_loader(name) for name in files}
    else:
        by_game = {stable_source_game_id(name): [] for name in files}
        if len(by_game) != len(files):
            raise ValueError('source game ID collision in prepared block')
        data = SupervisedFileDatasetsIter(
            file_list=files, file_batch_size=file_batch_size, reserve_ratio=0,
            shuffle_files=False, num_epochs=1, emit_game_id=True, **loader_kwargs,
        )
        for sample in data:
            by_game[int(sample[-1])].append(sample)
    grouped = [by_game[stable_source_game_id(draw['file'])] for draw in draws]
    if any(not rows for rows in grouped):
        raise ValueError('empty game in prepared input block')
    return [sample for rows in grouped for sample in rows], [len(rows) for rows in grouped]


def _prepare_worker(connection, loader_kwargs, file_batch_size, rayon_threads, sample_loader):
    os.environ['RAYON_NUM_THREADS'] = str(rayon_threads)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        while True:
            draws = connection.recv()
            if draws is None:
                return
            buffers = []
            try:
                draws = deepcopy(draws)
                if sample_loader is None:
                    for draw in draws:
                        digest = file_sha256(draw['file'])
                        if draw.get('source_sha256', digest) != digest:
                            raise ValueError('prepared source differs from checkpoint')
                        draw['source_sha256'] = digest
                rows, counts = _load_rows(draws, loader_kwargs, file_batch_size, sample_loader)
                columns = []
                for index, first in enumerate(rows[0]):
                    value = np.asarray(first)
                    shape = (len(rows), *value.shape)
                    size = int(np.prod(shape)) * value.dtype.itemsize
                    memory = SharedMemory(create=True, size=max(size, 1))
                    buffers.append(memory)
                    column = np.ndarray(shape, dtype=value.dtype, buffer=memory.buf)
                    for row_index, row in enumerate(rows):
                        column[row_index] = row[index]
                    columns.append({'name': memory.name, 'shape': shape,
                                    'dtype': value.dtype.str, 'scalar': _scalar_kind(first)})
                    del column
                del rows
                if sample_loader is None and any(file_sha256(d['file']) != d['source_sha256'] for d in draws):
                    raise ValueError('source changed during preparation')
                connection.send({'ok': True, 'draws': draws, 'counts': counts, 'columns': columns})
                if connection.recv() != 'attached':
                    return
            except Exception:
                connection.send({'ok': False, 'error': traceback.format_exc()[-6000:]})
                return
            finally:
                for memory in buffers:
                    memory.close()
                    # The parent unlinks after attachment; on failed jobs no
                    # other process owns the mapping yet.
                    try:
                        memory.unlink()
                    except FileNotFoundError:
                        pass
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class OrderedBlockPool:
    """One outstanding block per CPU worker, with small pipe messages only."""
    def __init__(self, workers, loader_kwargs, *, file_batch_size=4, rayon_threads=4,
                 sample_loader=None, timeout=180):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('preparation workers must be in 1..8')
        if file_batch_size not in (1, 2, 4, 8) or rayon_threads not in (1, 2, 4, 8):
            raise ValueError('unsupported preparation capacity tier')
        self.slots, self.pending, self.timeout = [], deque(), timeout
        context = mp.get_context('spawn')
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_prepare_worker,
                    args=(child, loader_kwargs, file_batch_size, rayon_threads, sample_loader), daemon=True)
                process.start()
                child.close()
                self.slots.append({'connection': parent, 'process': process, 'busy': False})
        except BaseException:
            self.close()
            raise

    @property
    def pids(self):
        return [slot['process'].pid for slot in self.slots]

    def submit(self, draws):
        slot = next((slot for slot in self.slots if not slot['busy']), None)
        if slot is None:
            raise RuntimeError('bounded preparation queue is full')
        slot['connection'].send(deepcopy(draws))
        slot['busy'] = True
        self.pending.append(slot)

    def take(self):
        slot = self.pending.popleft()
        connection = slot['connection']
        if not connection.poll(self.timeout):
            raise TimeoutError('ordered native preparation exceeded its bounded timeout')
        result = connection.recv()
        if not result['ok']:
            raise RuntimeError('preparation worker failed: ' + result['error'])
        block = PreparedBlock(result)
        connection.send('attached')
        slot['busy'] = False
        return block

    def close(self):
        slots, self.slots = self.slots, []
        for slot in slots:
            try:
                slot['connection'].send(None)
            except (OSError, EOFError):
                pass
        deadline = time.monotonic() + 10
        for slot in slots:
            process = slot['process']
            process.join(max(0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(5)
            slot['connection'].close()
            process.close()
        self.pending.clear()


class OrderedRotatingGameDataset(RotatingGameDataset):
    def __init__(self, *args, prepare_workers, prepare_rayon_threads=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_workers = prepare_workers
        self.prepare_rayon_threads = prepare_rayon_threads
        self.worker_pids = []

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError('the consumed cursor must stay in the training process')
        from mortal.data.dataloader import stable_source_game_id

        planner = RotatingGameSampler(self.sampler.domains, self.sampler.recipe, self.sampler.seed)
        planner.load_state_dict(self.sampler.state_dict())
        pool = OrderedBlockPool(self.prepare_workers, self.loader_kwargs,
            file_batch_size=self.prepare_file_batch_size, rayon_threads=self.prepare_rayon_threads,
            sample_loader=self.sample_loader)
        self.worker_pids = pool.pids
        try:
            for index in range(self.prepare_workers):
                draws = self.current['draws'] if index == 0 and self.current else [planner.draw() for _ in range(4)]
                pool.submit(draws)
            while True:
                block = pool.take()
                pool.submit([planner.draw() for _ in range(4)])
                if self.current is None:
                    self.current = {'draws': [self.sampler.draw() for _ in range(4)]}
                    self.offset = 0
                for draw, prepared in zip(self.current['draws'], block.draws):
                    if any(draw[key] != prepared[key] for key in ('file', 'domain', 'cycle', 'draw')):
                        raise ValueError('speculative preparation changed draw order')
                    if self.sample_loader is None:
                        digest = file_sha256(draw['file'])
                        if prepared['source_sha256'] != digest or draw.get('source_sha256', digest) != digest:
                            raise ValueError('prepared input changed before consumption')
                        draw['source_sha256'] = digest
                if self.content_ledger is not None:
                    self.content_ledger.verify(self.current['draws'])
                game_ids = []
                for draw, count in zip(block.draws, block.counts):
                    game_id = stable_source_game_id(draw['file'])
                    self.files[game_id] = {'file': draw['file'], 'available_decisions_per_draw': count}
                    game_ids.extend([game_id] * count)
                order = np.random.default_rng(derived_seed(
                    self.sampler.seed, 'rows', self.current['draws'][0]['draw'])).permutation(block.count)
                if not 0 <= self.offset <= block.count:
                    raise ValueError('consumed cursor exceeds prepared block')
                while self.offset < block.count:
                    index = int(order[self.offset])
                    self.offset += 1
                    self.consumed[game_ids[index]] += 1
                    yield block.row(index)
                self.current = None
                self.offset = 0
                del block
        finally:
            pool.close()
            self.worker_pids = []


class OrderedValidationDataset(IterableDataset):
    def __init__(self, original, *, prepare_workers, prepare_file_batch_size, prepare_rayon_threads):
        super().__init__()
        if original.shuffle_files or original.enable_augmentation or not original.emit_game_id:
            raise ValueError('ordered validation requires fixed files, one view and game IDs')
        self.original = original
        self.workers = prepare_workers
        self.file_batch_size = prepare_file_batch_size
        self.rayon_threads = prepare_rayon_threads

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError('nested validation DataLoader workers are unsupported')
        original = self.original
        kwargs = {name: getattr(original, name) for name in (
            'version', 'player_names', 'excludes', 'enable_augmentation', 'augmented_first',
            'emit_opponent_state_labels', 'track_danger_labels')}
        chunks = iter([{'file': name} for name in original.file_list[index:index + self.file_batch_size]]
                      for index in range(0, len(original.file_list), self.file_batch_size))
        pool = OrderedBlockPool(self.workers, kwargs, file_batch_size=self.file_batch_size,
                                rayon_threads=self.rayon_threads)
        try:
            for _ in range(self.workers):
                draws = next(chunks, None)
                if draws is not None:
                    pool.submit(draws)
            while pool.pending:
                block = pool.take()
                draws = next(chunks, None)
                if draws is not None:
                    pool.submit(draws)
                for index in range(block.count):
                    yield block.row(index)
                del block
        finally:
            pool.close()

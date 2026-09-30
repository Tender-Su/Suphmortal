from copy import deepcopy
import itertools
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from mortal.supervised.curriculum_probe import RotatingGameDataset, capture_rng
from mortal.supervised.ordered_preparation import OrderedRotatingGameDataset, OrderedBlockPool
from scripts.verify_sl_probe_resume import equal


def numeric_rows(filename):
    number = int(filename.rsplit('-', 1)[1])
    return [(np.full((2, 3), number + index / 8, dtype=np.float32),
             np.int64(index), np.array([True, False]), number, np.uint16(index),
             np.bool_(index % 2), np.float32(index / 8)) for index in range(7 + number % 3)]


def broken_rows(filename):
    raise ValueError('intentional native preparation failure')


class OrderedPreparationTests(unittest.TestCase):
    def dataset(self, workers=None, unique=3):
        arguments = ({'recent': [f'20240101-game-{i}' for i in range(unique)]},
                     {'recent': 1.0}, 41, {})
        if workers is None:
            return RotatingGameDataset(*arguments, sample_loader=numeric_rows)
        return OrderedRotatingGameDataset(*arguments, sample_loader=numeric_rows,
            prepare_workers=workers, prepare_file_batch_size=4)

    def loader(self, data):
        from mortal.supervised.train_supervised import safe_default_collate
        return iter(DataLoader(data, batch_size=13, num_workers=0, collate_fn=safe_default_collate,
                               generator=torch.Generator().manual_seed(19)))

    def close(self, iterator):
        iterator._dataset_fetcher.dataset_iter.close()

    def test_order_cursor_dtype_rng_and_resume_across_blocks_and_duplicates(self):
        baseline, prepared = self.dataset(), self.dataset(2)
        left, right = self.loader(baseline), self.loader(prepared)
        before = capture_rng()
        try:
            for _ in range(6):
                self.assertTrue(equal(next(left), next(right)))
                self.assertTrue(equal(baseline.state_dict(), prepared.state_dict()))
            self.assertTrue(equal(before, capture_rng()))
            saved = deepcopy(prepared.state_dict())
            self.assertEqual(sum(saved['consumed'].values()), 78)
            restored = self.dataset(1)
            restored.load_state_dict(saved)
            resumed = self.loader(restored)
            try:
                for _ in range(5):
                    self.assertTrue(equal(next(left), next(resumed)))
                    self.assertTrue(equal(baseline.state_dict(), restored.state_dict()))
            finally:
                self.close(resumed)
            self.assertEqual(restored.worker_pids, [])
        finally:
            self.close(left)
            self.close(right)
        self.assertEqual(prepared.worker_pids, [])

    def test_row_views_survive_block_retirement_and_worker_shutdown(self):
        data = self.dataset(2, unique=1)
        stream = iter(data)
        rows = list(itertools.islice(stream, 70))
        stream.close()
        expected = list(itertools.islice(iter(self.dataset(unique=1)), 70))
        self.assertTrue(equal(rows, expected))

    def test_worker_error_leaves_no_consumed_rows_or_child_process(self):
        data = self.dataset(1)
        data.sample_loader = broken_rows
        state = data.state_dict()
        with self.assertRaisesRegex(RuntimeError, 'intentional native preparation failure'):
            next(iter(data))
        self.assertTrue(equal(state, data.state_dict()))
        self.assertEqual(data.worker_pids, [])

    def test_worker_capacity_is_explicitly_bounded(self):
        for workers in (0, 9, True):
            with self.assertRaises(ValueError):
                OrderedBlockPool(workers, {})


if __name__ == '__main__':
    unittest.main()

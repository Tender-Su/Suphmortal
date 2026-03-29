import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import probe_validation_memory as probe


class ProbeValidationMemoryTests(unittest.TestCase):
    def test_build_dataloader_kwargs_skips_worker_only_options_when_num_workers_zero(self):
        kwargs = probe.build_dataloader_kwargs(
            dataset=object(),
            batch_size=32,
            num_workers=0,
            val_prefetch_factor=2,
        )

        self.assertEqual(0, kwargs['num_workers'])
        self.assertNotIn('worker_init_fn', kwargs)
        self.assertNotIn('prefetch_factor', kwargs)
        self.assertNotIn('persistent_workers', kwargs)
        self.assertNotIn('in_order', kwargs)

    def test_build_dataloader_kwargs_keeps_worker_options_when_num_workers_positive(self):
        kwargs = probe.build_dataloader_kwargs(
            dataset=object(),
            batch_size=32,
            num_workers=4,
            val_prefetch_factor=2,
        )

        self.assertEqual(4, kwargs['num_workers'])
        self.assertIs(probe.worker_init_fn, kwargs['worker_init_fn'])
        self.assertEqual(2, kwargs['prefetch_factor'])
        self.assertFalse(kwargs['persistent_workers'])
        self.assertTrue(kwargs['in_order'])


if __name__ == '__main__':
    unittest.main()

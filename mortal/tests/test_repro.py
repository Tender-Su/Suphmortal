import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.core.repro as repro


def make_config(*, enabled=False, seed=0, allow_cudnn_benchmark=None, strict_cuda=False):
    cfg = {
        "control": {
            "enable_cudnn_benchmark": True,
        },
        "repro": {
            "enabled": enabled,
            "seed": seed,
            "strict_cuda": strict_cuda,
        },
        "baseline": {
            "train": {},
        },
    }
    if allow_cudnn_benchmark is not None:
        cfg["repro"]["allow_cudnn_benchmark"] = allow_cudnn_benchmark
    return cfg


class ReproTests(unittest.TestCase):
    def test_resolve_train_key_returns_none_when_disabled(self):
        self.assertIsNone(repro.resolve_train_key(make_config(enabled=False)))

    def test_resolve_train_key_derives_stable_u64_when_enabled(self):
        config = make_config(enabled=True, seed=20260415)
        self.assertEqual(repro.resolve_train_key(config), repro.resolve_train_key(config))

    def test_resolve_baseline_pool_seed_prefers_explicit_pool_seed(self):
        config = make_config(enabled=True, seed=20260415)
        baseline_cfg = {"pool_seed": 12345}
        self.assertEqual(12345, repro.resolve_baseline_pool_seed(config, baseline_cfg))

    def test_resolve_baseline_pool_seed_derives_from_repro_seed(self):
        config = make_config(enabled=True, seed=20260415)
        self.assertEqual(
            repro.resolve_baseline_pool_seed(config, {}),
            repro.resolve_baseline_pool_seed(config, {}),
        )

    def test_effective_cudnn_benchmark_defaults_to_disabled_in_repro_mode(self):
        config = make_config(enabled=True, seed=1)
        self.assertFalse(repro.effective_cudnn_benchmark(config))

    def test_apply_reproducibility_reseeds_python_numpy_and_torch(self):
        config = make_config(enabled=True, seed=20260415)

        repro.apply_reproducibility(config, process_name="trainer")
        sample_a = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        repro.apply_reproducibility(config, process_name="trainer")
        sample_b = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        self.assertEqual(sample_a[0], sample_b[0])
        self.assertEqual(sample_a[1], sample_b[1])
        self.assertTrue(torch.equal(sample_a[2], sample_b[2]))


if __name__ == "__main__":
    unittest.main()

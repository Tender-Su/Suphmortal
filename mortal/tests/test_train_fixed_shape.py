"""Exercise actual nested trainer functions without importing native runtime.

NumPy executes the same extracted AST in a small float32/int64 adapter on cloud
hosts without Torch. Real Torch CPU tests run when available; CUDA timing and
whole-update equivalence are separate bounded runner checks, never implied here.
"""
import ast
from copy import deepcopy
from pathlib import Path
import os
import types
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

SOURCE = Path(__file__).resolve().parents[1] / 'supervised/train_supervised.py'


def extracted(name, namespace):
    node = next(n for n in ast.walk(ast.parse(SOURCE.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(node)], type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return namespace[name]


class Array(np.ndarray):
    def to(self, dtype=None, **kwargs):
        return self.astype(dtype) if dtype is not None else self

    def mul_(self, value):
        self *= value
        return self

    def clamp_(self, min=None, max=None):
        np.clip(self, min, max, out=self)
        return self

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def sum(self, dim=None, dtype=None):
        return array(np.asarray(self).sum(axis=dim, dtype=dtype))


def array(value):
    return np.asarray(value).view(Array)


NUMPY = types.SimpleNamespace(float32=np.float32, int64=np.int64, bool=np.bool_,
    full=lambda shape, value, dtype, device: array(np.full(shape, value, dtype=dtype)),
    where=lambda condition, yes, no: array(np.where(condition, yes, no)),
    minimum=lambda lhs, rhs: array(np.minimum(lhs, rhs)))


def check_rank(test, backend, make, as_numpy, device='cpu'):
    rng = np.random.default_rng(37)
    contexts = rng.integers(0, 4, size=(4096, 8), dtype=np.int64)
    contexts[:, 0] = rng.integers(0, 21, size=len(contexts))
    contexts[:, 6:] = rng.integers(0, 65536, size=(len(contexts), 2))
    context = make(contexts)
    turns = np.select([contexts[:, 0] <= 4, contexts[:, 0] >= 12], [1.0, 1.15], 1.05).astype(np.float32)
    for south, last, gap in ((1.59, 1.617, 0.0), (1.0, 1.0, 0.0), (0.0, -1.0, 0.0), (2.0, 3.0, 0.25)):
        namespace = {'torch': backend, 'device': device, 'rank_aux_base_weight': 0.001548,
            'rank_turn_weighting': {}, 'compute_context_turn_weights': lambda *_: make(turns),
            'rank_aux_south_factor': south, 'rank_aux_all_last_factor': last,
            'rank_aux_gap_focus_points': 4000.0, 'rank_aux_gap_close_bonus': gap,
            'rank_aux_max_weight': 0.00516,
            'context_meta_specs': {'round_stage': 1, 'is_all_last': 3, 'up_gap_100': 6, 'down_gap_100': 7}}
        actual = extracted('compute_rank_aux_sample_weights', namespace)(context)
        # Original masked multiplies in original order, followed by original gap/clamp.
        expected = backend.full((len(contexts),), 0.001548, dtype=backend.float32, device=device)
        expected.mul_(make(turns))
        if south > 0 and south != 1.0:
            expected[context[:, 1] == 1] *= float(south)
        if last > 0 and last != 1.0:
            expected[context[:, 3].to(backend.bool)] *= float(last)
        if gap > 0:
            nearest = backend.minimum(context[:, 6], context[:, 7]).to(backend.float32)
            nearest.mul_(100.0)
            closeness = (1.0 - nearest / 4000.0).clamp_(0.0, 1.0)
            expected.mul_(1.0 + gap * closeness)
        expected.clamp_(max=0.00516)
        test.assertTrue(np.array_equal(as_numpy(expected), as_numpy(actual)))
        test.assertEqual(np.dtype('float32'), as_numpy(actual).dtype)


def check_groups(test, backend, make, as_numpy):
    mapping = np.array([0] * 37 + [1] + [2] * 3 + list(range(3, 8)), dtype=np.int64)
    namespace = {'torch': backend, 'action_to_group': make(mapping),
                 'action_group_ids': make(np.arange(8)), 'num_action_groups': 8}
    function = extracted('compute_action_group_stats', namespace)
    rng = np.random.default_rng(91)
    for size in (0, 1, 46, 256, 1024):
        actions = np.arange(46) if size == 46 else rng.integers(0, 46, size=size, dtype=np.int64)
        for pred in (actions.copy(), (actions + 1) % 46, rng.integers(0, 46, size=size, dtype=np.int64)):
            actual = function(make(pred), make(actions))
            for key, indices in (('count', mapping[actions]), ('correct', mapping[actions[pred == actions]])):
                expected = np.bincount(indices, minlength=8)
                test.assertTrue(np.array_equal(expected, as_numpy(actual[key])))
                test.assertEqual(np.dtype('int64'), as_numpy(actual[key]).dtype)
            test.assertEqual(size, int(as_numpy(actual['count']).sum()))
            test.assertEqual(int((pred == actions).sum()), int(as_numpy(actual['correct']).sum()))


class FixedShapeTests(unittest.TestCase):
    def test_actual_rank_ast_numpy_float32_exact(self):
        check_rank(self, NUMPY, array, np.asarray)

    def test_actual_group_ast_numpy_int64_exact(self):
        check_groups(self, NUMPY, array, np.asarray)

    @unittest.skipIf(torch is None, 'Torch unavailable; actual Torch CPU equivalence remains required')
    def test_actual_rank_torch_cpu_exact(self):
        check_rank(self, torch, torch.from_numpy, lambda value: value.numpy())

    @unittest.skipIf(torch is None, 'Torch unavailable; actual Torch CPU equivalence remains required')
    def test_actual_groups_torch_cpu_exact(self):
        check_groups(self, torch, torch.from_numpy, lambda value: value.numpy())

    @unittest.skipUnless(torch is not None and os.environ.get('MORTAL_FIXED_SHAPE_CUDA_TEST') == '1',
                         'CUDA tests require explicit isolated-runner opt-in')
    def test_actual_rank_torch_cuda_amp_exact(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        with torch.autocast('cuda', enabled=True):
            check_rank(self, torch, lambda value: torch.from_numpy(value).cuda(),
                       lambda value: value.cpu().numpy(), device='cuda')

    @unittest.skipUnless(torch is not None and os.environ.get('MORTAL_FIXED_SHAPE_CUDA_TEST') == '1',
                         'CUDA tests require explicit isolated-runner opt-in')
    def test_actual_groups_torch_cuda_exact(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        with torch.inference_mode():
            check_groups(self, torch, lambda value: torch.from_numpy(value).cuda(),
                         lambda value: value.cpu().numpy())

    def test_no_dynamic_metric_index_or_scalar_extraction(self):
        tree = ast.parse(SOURCE.read_text())
        rank = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_rank_aux_sample_weights')
        group = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_action_group_stats')
        self.assertFalse(any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                             and n.value.id == 'weights' for n in ast.walk(rank)))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('bincount', 'nonzero', 'item', 'cpu')
                             for n in ast.walk(group)))


if __name__ == '__main__':
    unittest.main()

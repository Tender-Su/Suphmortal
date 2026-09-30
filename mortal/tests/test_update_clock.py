import ast
import json
from pathlib import Path
import math
import unittest
from types import SimpleNamespace

from mortal.core.update_clock import OptimizerUpdateClock, observed_scaler_step


class FakeOptimizer:
    def __init__(self, *, grad=True, fused=False):
        self.defaults = {'fused': fused}
        self.param_groups = [{'params': [SimpleNamespace(grad=1 if grad else None)]}]
        self.hooks = []
        self.updates = 0

    def register_step_post_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def step(self):
        self.updates += 1
        for hook in self.hooks:
            hook(self, (), {})
        return None  # AdamW's normal return value is not a success signal.


class FakeScaler:
    def __init__(self, *, skip=False, enabled=True, scale=1.0, next_scale=1.0):
        self.skip = skip
        self.enabled = enabled
        self.scale = scale
        self.next_scale = next_scale
        self.updates = 0

    def step(self, optimizer):
        if not self.enabled or not self.skip:
            return optimizer.step()

    def update(self):
        self.scale = self.next_scale
        self.updates += 1

    def get_scale(self):
        raise AssertionError('success observation must not read the scale')


class UpdateClockTests(unittest.TestCase):
    def test_disabled_growth_skip_and_numerical_scale_edges(self):
        for enabled, skip, scale, next_scale in (
            (False, False, 1, 1), (True, False, 2, 4),
            (True, False, 2, 2), (True, True, 2, 1),
            (True, True, 0, 0), (True, False, math.inf, math.inf),
            (True, True, math.nan, math.nan),
        ):
            with self.subTest(enabled=enabled, skip=skip, scale=scale):
                clock, optimizer = OptimizerUpdateClock(), FakeOptimizer()
                scaler = FakeScaler(enabled=enabled, skip=skip, scale=scale, next_scale=next_scale)
                result = observed_scaler_step(scaler, optimizer, clock)
                self.assertEqual(result, not skip)
                self.assertEqual((clock.attempts, clock.successes, clock.skips), (1, int(not skip), int(skip)))
                self.assertEqual(optimizer.updates, int(not skip))
                self.assertEqual(scaler.updates, 1)
                self.assertEqual(optimizer.hooks, [])

    def test_serialization_preserves_exact_counts_and_legacy_offset(self):
        for state, source, offset in (
            ({'optimizer_steps': 17, 'steps': 100}, 'optimizer_steps', 17),
            ({'steps': 9}, 'inferred_microbatches', 3),
        ):
            clock = OptimizerUpdateClock.from_checkpoint(state, opt_step_every=4)
            clock.record(False)
            clock.record(True)
            self.assertEqual(clock.progress, offset + 1)
            self.assertEqual(clock.legacy_source, source)
            saved = json.loads(json.dumps(clock.state_dict()))
            restored = OptimizerUpdateClock.from_checkpoint({'optimizer_update_clock': saved}, opt_step_every=8)
            self.assertEqual(clock, restored)
            self.assertEqual((restored.attempts, restored.successes, restored.skips), (2, 1, 1))

    def test_accumulation_attempts_and_skips_do_not_change_microbatch_units(self):
        clock, optimizer = OptimizerUpdateClock(), FakeOptimizer()
        microbatches = 0
        for index in range(1, 7):
            microbatches += 1
            if index % 2 == 0:
                observed_scaler_step(FakeScaler(skip=index == 4), optimizer, clock)
        self.assertEqual(microbatches, 6)
        self.assertEqual((clock.attempts, clock.successes, clock.skips), (3, 2, 1))
        self.assertEqual(clock.progress, 2)

    def test_train_wiring_preserves_scheduler_and_cleanup_on_skips(self):
        source = Path(__file__).resolve().parents[1] / 'online' / 'train_online.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        update = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.If)
                      and ast.unparse(node.test) == 'idx % opt_step_every == 0')
        calls = [ast.unparse(node.value) for node in update.body if isinstance(node, ast.Expr)]
        self.assertEqual(calls[-2:], ['observed_scaler_step(scaler, optimizer, update_clock)',
                                     'optimizer.zero_grad(set_to_none=True)'])
        self.assertFalse(any(isinstance(node, (ast.Return, ast.Continue)) for node in ast.walk(update)))
        self.assertFalse(any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'scheduler.step'
                             for node in ast.walk(update)))
        self.assertIn('scheduler.step()', source.read_text(encoding='utf-8'))

    def test_fresh_roundtrip(self):
        clock = OptimizerUpdateClock()
        clock.record(True)
        self.assertEqual(OptimizerUpdateClock.from_checkpoint(
            {'optimizer_update_clock': clock.state_dict()}, opt_step_every=1), clock)

    def test_bad_checkpoint_fails_closed(self):
        for field, value in (('version', 2), ('attempts', 1), ('skips', -1), ('successes', 1.5), ('legacy_source', 'exact')):
            saved = OptimizerUpdateClock().state_dict()
            saved[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                OptimizerUpdateClock.from_checkpoint({'optimizer_update_clock': saved}, opt_step_every=1)

    def test_weights_only_keeps_progress_but_resets_exact_optimizer_counts(self):
        clock = OptimizerUpdateClock.from_checkpoint({'optimizer_steps': 10}, opt_step_every=2)
        clock.record(True)
        clock.record(False)
        state = {'optimizer_steps': clock.progress, 'optimizer_update_clock': clock.state_dict()}
        branch = OptimizerUpdateClock.from_checkpoint(state, opt_step_every=2, weights_only=True)
        self.assertEqual(branch.progress, clock.progress)
        self.assertEqual(branch.legacy_attempt_offset, 10)
        self.assertEqual(branch.inherited_progress_offset, 1)
        self.assertEqual(branch.inherited_source, 'weights_only')
        self.assertEqual((branch.attempts, branch.successes, branch.skips), (0, 0, 0))
        branch.record(True)
        self.assertEqual(branch.progress, 12)
        resumed = OptimizerUpdateClock.from_checkpoint(
            {'optimizer_steps': 12, 'optimizer_update_clock': branch.state_dict()}, opt_step_every=2)
        self.assertEqual(resumed, branch)

    def test_missing_fields_and_inconsistent_compatibility_alias_fail_clearly(self):
        saved = OptimizerUpdateClock().state_dict()
        del saved['successes']
        with self.assertRaisesRegex(ValueError, 'missing fields.*successes'):
            OptimizerUpdateClock.from_checkpoint({'optimizer_update_clock': saved}, opt_step_every=1)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            OptimizerUpdateClock.from_checkpoint(
                {'optimizer_steps': 7, 'optimizer_update_clock': OptimizerUpdateClock().state_dict()},
                opt_step_every=1)

    def test_no_gradient_and_fused_rejected(self):
        for optimizer in (FakeOptimizer(grad=False), FakeOptimizer(fused=True)):
            clock = OptimizerUpdateClock()
            with self.assertRaises(ValueError):
                observed_scaler_step(FakeScaler(enabled=False), optimizer, clock)
            self.assertEqual(clock.attempts, 0)
            self.assertEqual(optimizer.hooks, [])

    def test_scaler_update_exception_does_not_commit_partial_attempt(self):
        optimizer, clock = FakeOptimizer(), OptimizerUpdateClock()
        scaler = FakeScaler()
        def fail():
            raise RuntimeError('scale update failed')
        scaler.update = fail
        with self.assertRaisesRegex(RuntimeError, 'scale update failed'):
            observed_scaler_step(scaler, optimizer, clock)
        self.assertEqual(optimizer.updates, 1)
        self.assertEqual(optimizer.hooks, [])
        # The exception aborts training before persistence. It is not an AMP skip.
        self.assertEqual((clock.attempts, clock.successes, clock.skips), (0, 0, 0))

    def test_exception_removes_hook_and_is_not_a_skip(self):
        optimizer, clock = FakeOptimizer(), OptimizerUpdateClock()
        scaler = FakeScaler()
        def fail(_optimizer):
            raise RuntimeError('step failed')
        scaler.step = fail
        with self.assertRaises(RuntimeError):
            observed_scaler_step(scaler, optimizer, clock)
        self.assertEqual(optimizer.hooks, [])
        self.assertEqual(clock.attempts, 0)


if __name__ == '__main__':
    unittest.main()

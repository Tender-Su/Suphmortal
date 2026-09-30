"""Pure configuration/clock and AST wiring regressions; no Torch required."""
import ast
from pathlib import Path
import unittest

from mortal.core.update_clock import OptimizerUpdateClock
from mortal.online.critic_calibration import (
    critic_only_enabled, successful_optimizer_step_limit,
    successful_optimizer_step_limit_reached, validate_calibration_resume,
)


class CriticCalibrationTests(unittest.TestCase):
    def test_opt_in_and_invalid_configuration(self):
        self.assertFalse(critic_only_enabled({}))
        self.assertEqual(successful_optimizer_step_limit({}), 0)
        self.assertTrue(critic_only_enabled({'value': {'enabled': True, 'critic_only': True}}))
        with self.assertRaisesRegex(ValueError, 'value.enabled'):
            critic_only_enabled({'value': {'critic_only': True}})
        for invalid in (-1, 1.5, True, '10'):
            with self.assertRaises(ValueError):
                successful_optimizer_step_limit({'online': {'max_successful_optimizer_steps': invalid}})

    def test_budget_ignores_skips_legacy_and_inherited_offsets(self):
        config = {'online': {'max_successful_optimizer_steps': 2}}
        clock = OptimizerUpdateClock(legacy_attempt_offset=999, inherited_progress_offset=999)
        clock.record(False)
        clock.record(True)
        self.assertFalse(successful_optimizer_step_limit_reached(config, clock))
        clock.record(False)
        clock.record(True)
        self.assertTrue(successful_optimizer_step_limit_reached(config, clock))
        self.assertFalse(successful_optimizer_step_limit_reached({}, clock))

    def test_same_phase_resume_preserves_budget_weights_only_resets(self):
        config = {'value': {'enabled': True, 'critic_only': True},
                  'online': {'max_successful_optimizer_steps': 2}}
        clock = OptimizerUpdateClock(attempts=3, successes=2, skips=1)
        state = {'config': config, 'optimizer_update_clock': clock.state_dict()}
        validate_calibration_resume(state, config)
        resumed = OptimizerUpdateClock.from_checkpoint(state, opt_step_every=4)
        self.assertTrue(successful_optimizer_step_limit_reached(config, resumed))
        fresh = OptimizerUpdateClock.from_checkpoint(state, opt_step_every=4, weights_only=True)
        self.assertEqual((fresh.attempts, fresh.successes, fresh.skips), (0, 0, 0))
        self.assertFalse(successful_optimizer_step_limit_reached(config, fresh))
        with self.assertRaisesRegex(ValueError, 'new phase'):
            validate_calibration_resume(state, {})
        with self.assertRaisesRegex(ValueError, 'exact optimizer_update_clock'):
            validate_calibration_resume({'config': config}, config)

    def test_actual_warmup_helper_never_releases_critic_only(self):
        tree = ast.parse(Path('mortal/online/train_online.py').read_text())
        names = {'value_training_cfg', 'value_critic_warmup_steps', 'value_critic_warmup_active'}
        module = ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        scope = {'critic_only_enabled': critic_only_enabled}
        exec(compile(module, '<warmup helpers>', 'exec'), scope)
        cfg = {'value': {'enabled': True, 'critic_only': True, 'critic_warmup_steps': 3}}
        for step in (0, 3, 1000000):
            self.assertTrue(scope['value_critic_warmup_active'](cfg, step))
        cfg['value']['critic_only'] = False
        self.assertTrue(scope['value_critic_warmup_active'](cfg, 2))
        self.assertFalse(scope['value_critic_warmup_active'](cfg, 3))

    def test_boundary_stop_and_mode_restore_are_wired(self):
        tree = ast.parse(Path('mortal/online/train_online.py').read_text())
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        restore = ast.unparse(functions['restore_training_mode'])
        self.assertIn('restore_actor_training_mode(mortal, policy_net, critic_only=critic_only)', restore)
        stop = ast.unparse(functions['stop_at_successful_optimizer_step_limit'])
        self.assertLess(stop.index('persist_live_training_state('), stop.index('sys.exit('))
        self.assertIn('ONLINE_MAX_STEPS_EXIT_CODE', stop)
        self.assertIn('atomic_torch_save', ast.unparse(functions['persist_live_training_state']))
        boundary = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'idx % opt_step_every == 0'
                    and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                            and c.func.id == 'stop_at_successful_optimizer_step_limit' for c in ast.walk(n))]
        self.assertEqual(len(boundary), 1)
        self.assertIn('validate_calibration_resume(state, config)', ast.unparse(tree))


if __name__ == '__main__':
    unittest.main()

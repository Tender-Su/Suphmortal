"""CPU-light orchestration contracts; not real-checkpoint training evidence."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mortal.core.artifacts import file_sha256
from mortal.core.update_clock import OptimizerUpdateClock, observed_scaler_step
from mortal.research import critic_only_update_check as check
from mortal.tests.test_update_clock import FakeOptimizer, FakeScaler


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            'control': {'state_file': 'old.pth', 'enable_compile': True},
            'resnet': {'conv_channels': 192, 'num_blocks': 40},
            'aux': {'next_rank_weight': 0.01},
            'supervised': {'rank_aux': {'base_weight': 0.02}},
            'dataset': {'file_batch_size': 8, 'reserve_ratio': 0, 'num_epochs': 1},
            'optim': {'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .1,
                      'scheduler': {'init': 1e-8, 'peak': 1e-4}},
            'value': {'weight': .05, 'zero_sum_weight': .01},
            'oracle_guiding': {'actor_enabled': True},
            'online': {'importance_sampling': {'enabled': True, 'vtrace_mode': 'always'}},
        }
        self.actor = {
            'control': {'version': 4}, 'resnet': self.base['resnet'],
            'aux': {'next_rank_weight': .2, 'opponent_state_weight': .3,
                    'danger_enabled': True, 'danger_weight': .4,
                    'tile_efficiency_weight': .5, 'furo_regret_weight': .6,
                    'hand_value_regret_weight': .7, 'custom': {'unchanged': [1, 2]}},
            'supervised': {'rank_aux': {'base_weight': .8, 'turn_weighting': {'mid_factor': 1.1}},
                           'arbitrary_future_recipe': {'retain': True}},
        }
        self.critic = {'config': {'resnet': self.base['resnet']}, 'oracle_critic_pretrain': {
            'critic_arch': 'dual_tower', 'oracle_fusion_mode': 'mlp', 'oracle_fusion_hidden': 128,
            'value_head_hidden': 64, 'value_loss_mode': 'mse', 'exact_zero_sum': True}}
        self.args = SimpleNamespace(actor='actor.pth', critic='critic.pth', opponent='canonical.pth',
                                    device='cpu', batch_size=8, enable_amp=False,
                                    engineering_lr=1e-5, sampling_seed=99, seed_key=22, seed_start=100)

    def test_configuration_is_fresh_and_preserves_entire_inherited_recipe(self):
        before = deepcopy((self.base, self.actor, self.critic))
        cfg = check.build_config(self.base, self.actor, self.critic, self.args, Path('/new'))
        self.assertEqual((self.base, self.actor, self.critic), before)
        for section in ('aux', 'supervised'):
            self.assertEqual(cfg[section], self.actor[section])
            self.assertIsNot(cfg[section], self.actor[section])
        cfg['aux']['custom']['unchanged'].append(3)
        self.assertEqual(self.actor['aux']['custom']['unchanged'], [1, 2])
        self.assertTrue(cfg['control']['online'])
        self.assertFalse(cfg['control']['enable_compile'])
        self.assertEqual(cfg['control']['opt_step_every'], 1)
        self.assertEqual(cfg['online']['max_successful_optimizer_steps'], 2)
        self.assertFalse(cfg['online']['stop_at_max_steps'])
        self.assertFalse(cfg['online']['importance_sampling']['enabled'])
        self.assertEqual(cfg['online']['importance_sampling']['vtrace_mode'], 'disabled')
        self.assertEqual((cfg['policy']['gae_gamma'], cfg['policy']['gae_lambda']), (1, 1))
        self.assertTrue(cfg['value']['critic_only'])
        self.assertEqual(cfg['value']['oracle_fusion_hidden'], 128)
        self.assertEqual(cfg['value']['value_head_hidden'], 64)
        self.assertTrue(cfg['value']['exact_zero_sum'])
        self.assertEqual(cfg['value']['zero_sum_weight'], .01)
        self.assertFalse(cfg['test_play']['enable'])
        self.assertFalse(cfg['test_play']['initial_enable'])
        self.assertEqual(cfg['baseline']['test']['state_file'], str(Path('canonical.pth').resolve()))
        self.assertEqual(cfg['optim']['weight_decay'], .1)
        self.assertEqual(cfg['optim']['scheduler']['init'], 1e-5)
        self.assertFalse(cfg['oracle_guiding']['actor_enabled'])
        self.assertEqual(cfg['control']['test_every'] % cfg['control']['save_every'], 0)

    def test_mismatched_architecture_and_zero_value_weight_fail(self):
        bad = deepcopy(self.critic)
        bad['config']['resnet']['num_blocks'] = 20
        with self.assertRaisesRegex(ValueError, 'architecture'):
            check.build_config(self.base, self.actor, bad, self.args, Path('/new'))
        self.base['value']['weight'] = 0
        with self.assertRaisesRegex(ValueError, 'value.weight'):
            check.build_config(self.base, self.actor, self.critic, self.args, Path('/new'))

    def test_cli_requires_positive_explicit_engineering_lr(self):
        base = []
        for name in ('config', 'actor', 'critic', 'opponent', 'reuse-rollout-dir', 'output-dir'):
            base += ['--' + name, 'placeholder']
        base += ['--seed-start', '1', '--seed-key', '2', '--sampling-seed', '3']
        with self.assertRaises(SystemExit):
            check.parse_args(base)
        for lr in ('0', '-1', 'nan', 'inf'):
            with self.subTest(lr=lr), self.assertRaises(SystemExit):
                check.parse_args(base + ['--engineering-lr', lr])
        self.assertEqual(check.parse_args(base + ['--engineering-lr', '.00001']).engineering_lr, 1e-5)


class ClosedRolloutTests(unittest.TestCase):
    def test_output_collision_and_original_tree_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / 'original'
            original.mkdir()
            marker = original / 'keep'
            marker.write_text('immutable')
            with self.assertRaises(ValueError):
                check.reserve_check_output(original / 'new', original)
            fresh = check.reserve_check_output(root / 'fresh', original)
            with self.assertRaises(FileExistsError):
                check.reserve_check_output(fresh, original)
            self.assertEqual(marker.read_text(), 'immutable')

    def test_exact_registered_checkpoint_roles_and_imputation_seed(self):
        hashes = {'actor': 'actorhash', 'opponent': 'opponenthash', 'critic': 'warmhash'}
        previous = {'weights_and_config': {
            'actor': {'sha256': hashes['actor']}, 'opponent': {'sha256': hashes['opponent']},
            'warm40k': {'sha256': hashes['critic']}}, 'arguments': {'imputation_seed': 20260905}}
        check.validate_registered_roles(previous, hashes)
        for key in hashes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check.validate_registered_roles(previous, {**hashes, key: 'wrong'})
        previous['arguments']['imputation_seed'] = 0
        with self.assertRaisesRegex(ValueError, 'imputation'):
            check.validate_registered_roles(previous, hashes)

    def test_one_shot_drain_preserves_files_and_refuses_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'original.json.gz'
            path.write_bytes(b'registered')
            games = [{'log_path': str(path), 'sha256': file_sha256(path)}]
            drain = check.OneShotDrain(tmp, games)
            self.assertEqual(drain(), str(Path(tmp).resolve()))
            with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                drain()
            self.assertEqual(path.read_bytes(), b'registered')
            self.assertEqual(drain.calls, 1)
            extra = Path(tmp) / 'extra'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'registered'):
                check.OneShotDrain(tmp, games)()
            extra.unlink()
            path.write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                check.OneShotDrain(tmp, games)()
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'registered'):
                check.OneShotDrain(tmp, games)()

    def test_hash_verifier_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'original')
            hashes = {path: file_sha256(path)}
            check.verify_hashes(hashes)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                check.verify_hashes(hashes)
            self.assertEqual(path.read_bytes(), b'changed')


class TransparentWrapperTests(unittest.TestCase):
    def test_publication_captures_live_references_without_network_or_mutation(self):
        actor, policy = object(), object()
        critic = {'oracle_brain': object(), 'value_net': object()}
        runtime = {'actor_oracle_enabled': False, 'actor_oracle_keep_prob': 0.0,
                   'oracle_critic_enabled': True, 'nested': {'unchanged': True}}
        before = deepcopy(runtime)
        observer = check.TrainingObservation(None, {}, critic, (), Path('/unused'))
        observer.actor_unchanged = Mock()
        observer.logits = Mock(return_value='fixed-logits')
        with patch.object(check, 'assert_state_equal') as equality, \
                patch('socket.socket', side_effect=AssertionError('unexpected network')):
            self.assertEqual(observer.submit(actor, policy, is_idle=True,
                                            runtime=runtime, aux_payload=critic), 0)
            self.assertEqual(observer.submit(actor, policy, is_idle=False,
                                            runtime=runtime, aux_payload=critic), 1)
            self.assertEqual(equality.call_count, 2)
        self.assertEqual(observer.models, (actor, policy))
        self.assertEqual(observer.actor_unchanged.call_count, 2)
        observer.logits.assert_called_once_with((actor, policy))
        self.assertEqual(runtime, before)
        runtime['nested']['unchanged'] = False
        self.assertTrue(observer.publications[0]['runtime']['nested']['unchanged'])
        self.assertEqual(observer.steps, [])

    def test_seed_wrapper_calls_original_once_and_preserves_identity(self):
        dataset, files, returned = object(), ['same-original-file'], object()
        original = Mock(return_value=returned)
        wrapper = check.seeded_trajectories(original)
        self.assertIs(wrapper(dataset, files), returned)
        original.assert_called_once_with(dataset, files, oracle_imputation_seed=20260905)
        with self.assertRaises(ValueError):
            wrapper(dataset, files, oracle_imputation_seed=1)
        self.assertEqual(original.call_count, 1)

    def test_step_wrapper_observes_real_helper_once_including_skips(self):
        events = []
        original = Mock(wraps=observed_scaler_step)
        before = lambda local, optimizer, clock: events.append(('before', clock.state_dict()))
        after = lambda result, optimizer, clock: events.append(('after', result, clock.state_dict()))
        wrapped = check.observed_steps(original, before, after)
        clock, optimizer = OptimizerUpdateClock(), FakeOptimizer()
        for skip in (False, True, False):
            result = wrapped(FakeScaler(skip=skip), optimizer, clock)
            self.assertEqual(result, not skip)
        self.assertEqual(original.call_count, 3)
        self.assertEqual((clock.attempts, clock.successes, clock.skips), (3, 2, 1))
        self.assertEqual(optimizer.updates, 2)
        self.assertEqual([event[0] for event in events], ['before', 'after'] * 3)
        self.assertEqual(events[0][1]['attempts'], 0)

    def test_step_exception_does_not_invent_a_result_or_clock(self):
        clock, optimizer = OptimizerUpdateClock(), FakeOptimizer()
        scaler = FakeScaler()
        scaler.step = Mock(side_effect=RuntimeError('real failure'))
        after = Mock()
        real = Mock(wraps=observed_scaler_step)
        wrapped = check.observed_steps(real, Mock(), after)
        with self.assertRaisesRegex(RuntimeError, 'real failure'):
            wrapped(scaler, optimizer, clock)
        real.assert_called_once_with(scaler, optimizer, clock)
        self.assertEqual(clock.attempts, 0)
        after.assert_not_called()

    def test_checkpoint_evidence_uses_exact_clock_not_inferred_steps(self):
        clock, observations = OptimizerUpdateClock(), []
        for success in (False, True, True):
            clock.record(success)
            observations.append({'succeeded': success, 'clock': clock.state_dict()})
        state = {'steps': 3, 'optimizer_steps': 2, 'optimizer_update_clock': clock.state_dict()}
        before = json.dumps((state, observations))
        self.assertEqual(check.validate_saved_clock(state, observations), clock.state_dict())
        self.assertEqual(json.dumps((state, observations)), before)
        for bad in ({'steps': 2, 'optimizer_steps': 2}, {**state, 'steps': 2},
                    {**state, 'optimizer_steps': 3}):
            with self.assertRaises(ValueError):
                check.validate_saved_clock(bad, observations)
        bad = deepcopy(state)
        bad['optimizer_update_clock']['attempts'] = 4
        with self.assertRaises(ValueError):
            check.validate_saved_clock(bad, observations)

    def test_only_allowed_boundaries_are_patched_and_production_train_is_used(self):
        source = Path(check.__file__).read_text()
        tree = ast.parse(source)
        patched = [ast.unparse(node.args[0]) + '.' + node.args[1].value
                   for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == 'patch.object']
        self.assertEqual(set(patched), {'common.drain', 'common.submit_param',
                         'FileDatasetsIter.iter_game_trajectories', 'train_online.observed_scaler_step'})
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertIn('train_online.train', calls)
        self.assertNotIn('train_online.main', calls)
        self.assertIn('load_verified_rollout', calls)
        self.assertNotIn('clock.record', calls)
        self.assertNotIn('optimizer.step', calls)
        self.assertIn("player_names=['trainee']", source)


if __name__ == '__main__':
    unittest.main()

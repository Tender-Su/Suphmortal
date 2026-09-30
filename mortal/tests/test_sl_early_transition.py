"""CPU contracts; real torch state tests skip explicitly on torch-free hosts."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mortal.core.artifacts import file_sha256, stable_json_digest
from mortal.supervised.early_transition import (
    LEARNED_KEYS, TrainingContentLedger, observation_schedule, parent_record, phase_spec,
    prepare_transition_state, validate_corrected_recipe, validate_optimizer_mapping, validate_parent,
)
from scripts.run_sl_early_transition import (
    apply_phase_lr, branch_provenance, experiment_lock, fresh_directory, matching_parents, parser,
    ensure_initializable_phase, phase_parent, require_frozen_runtime, verify_manifest,
)

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_TOML = importlib.util.find_spec('toml') is not None


def source_fixture():
    return {'checkpoint_id': 'early-A', 'steps': 100, 'optimizer_steps': 99,
            'auxiliary_optimizer_steps': 103, 'scheduler': {'old_clock': 99}, 'scaler': {'scale': 128},
            'config': {'control': {}, 'dataset': {},
                       'optim': {'eps': 1e-8, 'betas': [0.9, 0.999], 'weight_decay': 0.01,
                                 'scheduler': {'peak': 0.001}},
                       'supervised': {'lr': 0.001, 'rank_aux': {'base_weight': 0.001548,
                                                                              'max_weight': 0.00516}},
                       'aux': {'next_rank_weight': 0.03, 'opponent_state_weight': 0.00135,
                               'danger_enabled': True, 'danger_weight': 0.00804,
                               'danger_turn_weighting': {'early_factor': 0.05}}},
            'optimizer': {'param_groups': [dict(params=[i], lr=4.91502392e-5, initial_lr=1,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01 if i == 0 else 0.0)
                           for i in range(2)],
                          'state': {i: {'step': 99., 'exp_avg': [0.7], 'exp_avg_sq': [0.4]} for i in range(2)}},
            'optimizer_param_groups': [('mortal.a',), ('policy_net.b',)],
            **{name: {} for name in LEARNED_KEYS[:5]},
            'best_val_loss': 0.1, 'patience_counter': 42,
            'adaptive_curriculum_state': {'completed': True},
            'curriculum_probe': {'identity': 'old', 'dataset': 'old_cursor', 'rng': 'old_rng'}}


def manifest_fixture(directory='new'):
    spec = phase_spec(updates=4, observations=[2, 4], seed=31, peak=1e-5, init=5e-6, warmup=2)
    return {'identity': 'experiment', 'directory': str(directory), 'source_git_commit': 'a' * 40,
            'source_sha256': {'trainer.py': 'hash'}, 'phases': {'B': spec, 'C': {**spec, 'seed': 32}},
            'device': 'cpu', 'microbatch': 2, 'logical_batch': 4, 'val_batch_size': 2,
            'runtime_performance': {'probe_prepare_file_batch_size': 1,
                                    'val_file_batch_size': 1, 'rayon_num_threads': 2}}


class TransitionContracts(unittest.TestCase):
    def test_budgets_observations_lr_and_seeds_are_explicit(self):
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['prepare', '--directory', 'new'])
        for change in ({'updates': 0}, {'observations': [4, 2]}, {'observations': [2]},
                       {'observations': [2, 2, 4]}, {'peak': float('nan')}, {'init': 0},
                       {'warmup': 0}, {'warmup': 5}, {'seed': -1}):
            args = dict(updates=4, observations=[2, 4], seed=31, peak=1e-5, init=5e-6, warmup=2)
            with self.subTest(change=change), self.assertRaises(ValueError):
                phase_spec(**{**args, **change})

    def test_two_arms_only_and_both_b_end_before_any_c(self):
        schedule = observation_schedule(manifest_fixture()['phases'])
        self.assertEqual(schedule, [('early', 'B', 2), ('late', 'B', 2),
                                   ('late', 'B', 4), ('early', 'B', 4),
                                   ('early', 'C', 2), ('late', 'C', 2),
                                   ('late', 'C', 4), ('early', 'C', 4)])

    def test_parent_full_recipe_and_adam_structure_fail_closed(self):
        source = source_fixture()
        validate_parent(source)
        for key, value in (('opponent_state_weight', 0), ('danger_weight', 0), ('danger_enabled', False)):
            changed = deepcopy(source)
            changed['config']['aux'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'corrected'):
                validate_parent(changed)
        changed = deepcopy(source)
        changed['config']['supervised']['rank_aux']['base_weight'] = 0.03
        with self.assertRaisesRegex(ValueError, 'corrected'):
            validate_corrected_recipe(changed['config'])
        changed = deepcopy(source)
        changed['config']['aux']['danger_turn_weighting']['early_factor'] = 0.2
        with self.assertRaisesRegex(ValueError, 'full auxiliary'):
            matching_parents(source, changed)
        for edit in (lambda s: s['optimizer']['state'][0].pop('exp_avg_sq'),
                     lambda s: s['optimizer_param_groups'].pop(),
                     lambda s: s['optimizer']['state'].pop(1),
                     lambda s: s.pop('danger_aux_net')):
            changed = deepcopy(source)
            edit(changed)
            with self.assertRaises(ValueError):
                validate_parent(changed)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            validate_parent(source, arm='early')

    def test_same_size_adam_mapping_cannot_silently_remap(self):
        validate_optimizer_mapping([['a'], ['b']], (('a',), ('b',)))
        with self.assertRaisesRegex(ValueError, 'exact Adam'):
            validate_optimizer_mapping([['a'], ['b']], (('b',), ('a',)))

    def test_source_lrs_differ_but_declared_phase_configs_match(self):
        early = source_fixture()
        late = deepcopy(early)
        for group in late['optimizer']['param_groups']:
            group['lr'] = 5e-6
        matching_parents(early, late)
        spec = manifest_fixture()['phases']['B']
        configs = [apply_phase_lr(deepcopy(parent['config']), spec, {}) for parent in (early, late)]
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0]['supervised']['lr'], 1e-5)
        self.assertEqual(configs[0]['optim']['scheduler']['peak'], 1e-5)
        self.assertEqual(configs[0]['supervised']['scheduler']['init'], 5e-6)

    def test_provenance_persists_full_parent_chain(self):
        source = source_fixture()
        manifest = manifest_fixture()
        a = parent_record(source, path='frozen/early_A.pth', sha256='a', phase='A')
        b_provenance = branch_provenance(source, manifest, 'early', 'B', a)
        source.update(run_provenance=b_provenance, checkpoint_id='early-B',
                      steps=8, optimizer_steps=4, auxiliary_optimizer_steps=107)
        b = parent_record(source, path='new/early_B/state_file.pth', sha256='b', phase='B')
        c_provenance = branch_provenance(source, manifest, 'early', 'C', b)
        self.assertEqual(c_provenance['parent_chain'], [a, b])
        self.assertEqual(b_provenance['parent_chain'], [a])
        self.assertEqual(c_provenance['parent_checkpoint_id'], 'early-B')
        self.assertEqual(c_provenance['source_git_commit'], 'a' * 40)
        self.assertEqual(c_provenance['phase_scheduler'], manifest['phases']['C']['scheduler'])

    def test_output_and_phase_parent_collisions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileExistsError):
                fresh_directory(root)
            manifest = manifest_fixture(root)
            b = fresh_directory(root / 'early_B')
            checkpoint = b / 'state_file.pth'
            checkpoint.write_text('latest', encoding='utf-8')
            record = {'checkpoint': str(checkpoint), 'sha256': file_sha256(checkpoint)}
            receipt = {'experiment_id': 'experiment', 'arm': 'early', 'phase': 'B',
                       'until_update': 4, 'endpoint': record}
            (b / 'completed.json').write_text(json.dumps(receipt))
            with self.assertRaises(FileNotFoundError):
                phase_parent(manifest, 'early', 'C')
            late = fresh_directory(root / 'late_B')
            (late / 'completed.json').write_text(json.dumps({**receipt, 'arm': 'late'}))
            self.assertEqual(phase_parent(manifest, 'early', 'C'), record)
            checkpoint.write_text('different', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                phase_parent(manifest, 'early', 'C')
            record['checkpoint'] = str(b / 'best_loss.pth')
            (b / 'completed.json').write_text(json.dumps({**receipt, 'endpoint': record}))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                phase_parent(manifest, 'early', 'C')

    def test_concurrent_runner_lock_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            with experiment_lock(tmp, '.runner.lock'):
                with self.assertRaisesRegex(RuntimeError, 'active runner'):
                    with experiment_lock(tmp, '.runner.lock'):
                        self.fail('concurrent runner admitted')
            with experiment_lock(tmp, '.runner.lock'):
                pass

    def test_direct_phase_cannot_execute_unpinned_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / 'frozen'
            require_frozen_runtime({'source_root': str(frozen)}, frozen)
            with self.assertRaisesRegex(ValueError, 'frozen source'):
                require_frozen_runtime({'source_root': str(frozen)}, Path(tmp) / 'different')

    def test_missing_latest_never_reinitializes_used_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.toml').write_text('prepared')
            ensure_initializable_phase(root)
            for name in ('update_0000000.pth', 'segment_0000002.json', 'completed.json', 'tensorboard'):
                artifact = root / name
                artifact.write_text('historical result')
                with self.assertRaisesRegex(RuntimeError, 'refusing to restart'):
                    ensure_initializable_phase(root)
                self.assertEqual(artifact.read_text(), 'historical result')
                artifact.unlink()

    def test_shared_content_ledger_pins_before_consumption_across_arms_and_cycles(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'content.sqlite3'
            game = Path(tmp) / 'game.json'
            game.write_bytes(b'original')
            draw = {'file': str(game), 'source_sha256': file_sha256(game)}
            early = TrainingContentLedger(path, 'experiment', create=True)
            early.verify([draw])
            late = TrainingContentLedger(path, 'experiment')
            late.verify([draw])
            game.write_bytes(b'changed')
            changed = {'file': str(game), 'source_sha256': file_sha256(game)}
            for ledger in (early, late):
                with self.assertRaisesRegex(ValueError, 'across arms/cycles'):
                    ledger.verify([changed])
            with self.assertRaisesRegex(ValueError, 'another experiment'):
                TrainingContentLedger(path, 'other')
            with self.assertRaises(FileExistsError):
                TrainingContentLedger(path, 'experiment', create=True)
            path.unlink()
            with self.assertRaises(sqlite3.OperationalError):
                TrainingContentLedger(path, 'experiment')

    def test_frozen_manifest_identity_and_file_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'indexes.pth').write_bytes(b'fixed index')
            manifest = {'directory': str(root), 'input_sha256': {'indexes.pth': file_sha256(root / 'indexes.pth')},
                        'source_sha256': {}, 'parents': {}, 'controller_source_sha256': {}}
            manifest['identity'] = stable_json_digest(manifest)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_manifest(root), manifest)
            (root / 'indexes.pth').write_bytes(b'changed index')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_manifest(root)


@unittest.skipUnless(HAS_TORCH and HAS_TOML, 'requires real torch and toml; not a production trainer smoke')
class TransitionTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.source = source_fixture()
        self.source['scaler'] = torch.amp.GradScaler('cpu', init_scale=128).state_dict()
        for name in LEARNED_KEYS[:5]:
            self.source[name] = {'weight': torch.tensor([1.])}
        for state in self.source['optimizer']['state'].values():
            state.update({key: torch.tensor(value) for key, value in state.items()})
        self.manifest = manifest_fixture()
        self.parent = parent_record(self.source, path='early_A.pth', sha256='a', phase='A')
        from scripts.run_sl_early_transition import build_phase_config
        self.config = build_phase_config(self.source, self.source['config'], self.manifest,
                                        'early', 'B', Path('early_B'), self.parent)

    def assert_nested_equal(self, left, right):
        if self.torch.is_tensor(left):
            self.torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_keep_all_moments_heads_amp_and_aux_clock_reset_every_local_state(self):
        before = deepcopy(self.source)
        state = prepare_transition_state(self.source, self.config)
        self.assert_nested_equal(before, self.source)
        for name in LEARNED_KEYS:
            if name != 'optimizer':
                self.assert_nested_equal(state[name], self.source[name])
        self.assert_nested_equal(state['optimizer']['state'], self.source['optimizer']['state'])
        for old, new in zip(self.source['optimizer']['param_groups'], state['optimizer']['param_groups']):
            self.assertEqual({k: v for k, v in old.items() if k not in ('lr', 'initial_lr')},
                             {k: v for k, v in new.items() if k not in ('lr', 'initial_lr')})
        self.assertEqual(state['auxiliary_optimizer_steps'], 103)
        self.assertEqual((state['steps'], state['optimizer_steps'], state['epoch']), (0, 0, 0))
        for key in ('best_val_loss', 'patience_counter', 'curriculum_probe', 'adaptive_curriculum_state'):
            self.assertNotIn(key, state)
        self.assertEqual(state['scheduler']['_last_lr'], [5e-6, 5e-6])
        self.assertEqual(state['scheduler']['last_epoch'], 0)
        scaler = self.torch.amp.GradScaler('cpu', init_scale=1)
        scaler.load_state_dict(state['scaler'])
        self.assert_nested_equal(scaler.state_dict(), self.source['scaler'])
        self.assertEqual(scaler.get_scale(), 128)
        changed = deepcopy(self.config)
        changed['aux']['danger_turn_weighting']['early_factor'] = 0.2
        with self.assertRaisesRegex(ValueError, 'objective differs'):
            prepare_transition_state(self.source, changed)
        changed = deepcopy(self.config)
        changed['supervised']['lr'] = 1e-4
        with self.assertRaisesRegex(ValueError, 'trainer LR'):
            prepare_transition_state(self.source, changed)

    def optimizer_and_scheduler(self, state):
        from mortal.supervised.lr_scheduler import LinearWarmUpConstantLR
        params = [self.torch.nn.Parameter(self.torch.zeros(1)) for _ in range(2)]
        optimizer = self.torch.optim.AdamW([{'params': [p]} for p in params], lr=1)
        scheduler = LinearWarmUpConstantLR(optimizer, peak=1e-5, init=5e-6, warm_up_steps=2)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        return params, optimizer, scheduler

    def test_real_adam_first_update_and_serialized_scheduler_restart(self):
        state = prepare_transition_state(self.source, self.config)
        params, optimizer, scheduler = self.optimizer_and_scheduler(state)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [5e-6, 5e-6])
        for p in params:
            p.grad = self.torch.ones_like(p)
        optimizer.step()
        scheduler.step()
        self.assertEqual([int(s['step']) for s in optimizer.state.values()], [100, 100])
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 7.5e-6)
        state.update(optimizer=deepcopy(optimizer.state_dict()), scheduler=deepcopy(scheduler.state_dict()))
        serialized = io.BytesIO()
        self.torch.save(state, serialized)
        serialized.seek(0)
        state = self.torch.load(serialized, map_location='cpu', weights_only=False)
        resumed_params, resumed, resumed_scheduler = self.optimizer_and_scheduler(state)
        self.assertEqual(scheduler.get_last_lr(), resumed_scheduler.get_last_lr())
        for p in resumed_params:
            p.grad = self.torch.ones_like(p)
        resumed.step()
        resumed_scheduler.step()
        optimizer.step()
        scheduler.step()
        self.assert_nested_equal(optimizer.state_dict(), resumed.state_dict())
        self.assert_nested_equal(scheduler.state_dict(), resumed_scheduler.state_dict())
        self.assertEqual(resumed_scheduler.get_last_lr(), [1e-5, 1e-5])

    def test_b_to_c_resets_local_clock_and_keeps_cumulative_aux_and_chain(self):
        from scripts.run_sl_early_transition import build_phase_config
        b = prepare_transition_state(self.source, self.config)
        b.update(steps=8, optimizer_steps=4, auxiliary_optimizer_steps=107, checkpoint_id='B-end')
        parent = parent_record(b, path='early_B/state_file.pth', sha256='b', phase='B')
        config = build_phase_config(b, self.source['config'], self.manifest,
                                   'early', 'C', Path('early_C'), parent)
        c = prepare_transition_state(b, config)
        self.assertEqual(c['auxiliary_optimizer_steps'], 107)
        self.assertEqual(c['optimizer_steps'], 0)
        self.assertEqual(c['scheduler']['last_epoch'], 0)
        self.assertEqual([p['phase'] for p in c['run_provenance']['parent_chain']], ['A', 'B'])
        self.assert_nested_equal(c['optimizer']['state'], b['optimizer']['state'])

    def test_phase_common_data_order_full_domain_and_fresh_rng(self):
        import random
        import numpy as np
        from mortal.supervised.curriculum_probe import CurriculumProbe, RECIPES, RotatingGameSampler
        domains = {'recent': list(range(50)), 'latest': list(range(30)), 'replay': list(range(100, 150))}
        for phase in ('B', 'C'):
            seed = self.manifest['phases'][phase]['seed']
            left, right = [RotatingGameSampler(domains, RECIPES[phase], seed) for _ in range(2)]
            self.assertEqual([left.draw() for _ in range(1000)], [right.draw() for _ in range(1000)])
        full = RotatingGameSampler(domains, {'recent': 1}, 10)
        self.assertEqual({full.draw()['file'] for _ in range(50)}, set(domains['recent']))
        values = []
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                probe = CurriculumProbe(self.config, domains, recipe='B', seed=31, output=tmp,
                    horizons=[2, 4], eval_splits={}, identity='id', reset_branch_rng=True)
                probe.restore(prepare_transition_state(self.source, self.config))
                values.append((random.random(), np.random.rand(), self.torch.rand(1).item()))
        self.assertEqual(values[0], values[1])

    def test_dataset_hook_rejects_changes_after_completed_block_and_during_parse(self):
        from mortal.supervised.curriculum_probe import RotatingGameDataset
        # Only the native parser is stubbed; exercise the real dataset's block
        # lifecycle and durable ledger, without a GPU or compiled extension.
        module = SimpleNamespace(stable_source_game_id=lambda _: 1)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'mortal.data.dataloader': module}):
            game = Path(tmp) / 'game.json'
            game.write_bytes(b'original')
            ledger = TrainingContentLedger(Path(tmp) / 'content.sqlite3', 'experiment', create=True)

            def dataset():
                result = RotatingGameDataset({'recent': [str(game)]}, {'recent': 1}, 10, {},
                                             content_ledger=ledger)
                result.load_sample_block = lambda draws: [[('row',)] for _ in draws]
                return result

            stream = iter(dataset())
            for _ in range(4):
                next(stream)
            game.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'across arms/cycles'):
                next(stream)
            with self.assertRaisesRegex(ValueError, 'across arms/cycles'):
                next(iter(dataset()))
            game.write_bytes(b'original')
            changed_while_parsing = dataset()

            def parse(draws):
                game.write_bytes(b'mutated during parse')
                return [[('row',)] for _ in draws]

            changed_while_parsing.load_sample_block = parse
            with self.assertRaisesRegex(ValueError, 'during preparation'):
                next(iter(changed_while_parsing))

    def test_old_data_only_probe_still_rejects_changed_lr(self):
        from scripts.run_sl_curriculum_probe import prepare_branch_state
        with self.assertRaisesRegex(ValueError, 'preserve current parent LR'):
            prepare_branch_state(self.source, self.config)


if __name__ == '__main__':
    unittest.main()

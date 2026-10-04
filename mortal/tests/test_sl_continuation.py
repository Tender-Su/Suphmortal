"""State-preserving phase extension and independently scheduled observations."""
from copy import deepcopy
from contextlib import closing
import importlib.util
import io
from pathlib import Path
import json
import random
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mortal.core.artifacts import atomic_write_json, file_sha256, stable_json_digest
from mortal.supervised.continuation import (
    evaluation_horizons, evaluation_kind, ledger_snapshot, microbatch_migration, observation_plan, rebind_saved_state,
    relocate_config, training_semantics, trend_splits, validate_inherited_pins, validate_saved_phase,
)
from mortal.supervised.early_transition import TrainingContentLedger, parent_record
from mortal.tests.test_sl_early_transition import source_fixture
from scripts.continue_sl_phase import parser, require_stable_checkpoint, separate_output

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('toml') is not None
RECIPES = {'B': {'recent': 0.90, 'replay': 0.10}, 'C': {'latest': 0.98, 'replay': 0.02}}
DOMAINS = {'recent': ['2024game'], 'latest': ['2024game'], 'replay': ['2012game']}


def saved_fixture():
    state = source_fixture()
    provenance = {'plan_id': 'old-phase', 'experiment_id': 'old-experiment', 'phase': 'B', 'arm': 'late',
                  'branch_mode': 'preserve_adam_declared_phase_lr', 'parent_chain': []}
    sl = state['config']['supervised']
    sl.update(seed=31, batch_size=4, val_batch_size=1024, num_workers=0, max_steps=0, val_every_steps=0, save_every=0,
              run_provenance=deepcopy(provenance), probe_training_content_ledger='old-ledger.sqlite3',
              lr=1e-5, scheduler={'type': 'constant', 'init': 5e-6, 'peak': 1e-5, 'warm_up_steps': 20})
    state['config']['control'].update(device='cpu', enable_cuda_prefetch=False, enable_amp=False, opt_step_every=2)
    state.update(optimizer_steps=10, steps=20, skipped_optimizer_steps=0, nonfinite_batches=0,
                 epoch=0, epoch_complete=False, timestamp=1.0, config_section='supervised', run_provenance=provenance)
    for group in state['optimizer']['param_groups']:
        group['lr'] = 7.5e-6
    state['scheduler'] = {'last_epoch': 10, '_last_lr': [7.5e-6, 7.5e-6]}
    state['curriculum_probe'] = {'identity': 'old-phase', 'observed': [0, 10], 'elapsed_seconds': 27,
        'rng': {'python': random.Random(12).getstate(), 'numpy': [1], 'torch': [2], 'cuda': []},
        'dataset': {'sampler': {'seed': 31, 'recipe': RECIPES['B'], 'positions': {'recent': [35, 1], 'replay': [3, 1]},
                               'rng': random.Random(31).getstate(), 'draws': 40},
                    'current': None, 'offset': 0, 'consumed': {1: 80},
                    'files': {1: {'file': '2024game', 'available_decisions_per_draw': 2}}}}
    return state


def plan_fixture(start=10, until=30):
    return observation_plan(start=start, until=until, save_updates=3, save_seconds=1800,
                            trend_every=5, full_every=10)


def microbatch_fixture(state=None, batch=256):
    state = deepcopy(state if state is not None else saved_fixture())
    state['config']['supervised'].update(batch_size=batch, log_every=32768 // batch)
    state['config']['control']['opt_step_every'] = 1024 // batch
    state.update(steps=(state['optimizer_steps'] + state['skipped_optimizer_steps']) * (1024 // batch),
                 convergence_state=None, adaptive_curriculum_state=None)
    state['curriculum_probe']['dataset']['consumed'] = {1: state['steps'] * batch}
    return state


class ContinuationContracts(unittest.TestCase):
    def test_microbatch_migration_converts_only_microstep_units_in_both_directions(self):
        for old, new in ((256, 512), (512, 256)):
            source = microbatch_fixture(batch=old)
            source['optimizer_steps'] = 1000
            source['skipped_optimizer_steps'] = 1
            source = microbatch_fixture(source, old)
            migration = microbatch_migration(source, new, 'measured throughput')
            self.assertEqual(migration['source_microsteps'], 1001 * (1024 // old))
            self.assertEqual(migration['target_microsteps'], 1001 * (1024 // new))
            self.assertEqual(migration['consumed_decisions'], 1001 * 1024)
            parent = parent_record(source, path='new/parent.pth', sha256='source', phase='B')
            config = relocate_config(source['config'], 'new', 'new', parent, 'commit', 'old',
                                     runtime_sha256='new-runtime', migration=migration)
            rebound = rebind_saved_state(source, config, 'new', migration=migration)
            self.assertEqual(rebound['steps'], 1001 * (1024 // new))
            for key in source:
                if key not in ('steps', 'config', 'run_provenance', 'checkpoint_id', 'curriculum_probe'):
                    self.assertEqual(source[key], rebound[key])
            self.assertEqual(rebound['curriculum_probe']['dataset'], source['curriculum_probe']['dataset'])
            self.assertEqual(rebound['config']['supervised']['log_every'] * new, 32768)
            self.assertEqual(rebound['run_provenance']['microbatch_migrations'], [migration])
            self.assertEqual(rebound['run_provenance']['continuation']['mode'], 'declared_microbatch_numerical_branch')
            self.assertEqual(rebound['config']['supervised']['val_batch_size'], 1024)
            for section, key, value in (('supervised', 'lr', 1e-3), ('supervised', 'seed', 3),
                                        ('supervised', 'log_every', 13), ('control', 'opt_step_every', 1)):
                bad = deepcopy(config)
                bad[section][key] = value
                with self.assertRaisesRegex(ValueError, 'training settings'):
                    rebind_saved_state(source, bad, 'new', migration=migration)

    def test_microbatch_migration_rejects_unknown_or_incomplete_clocks(self):
        source = microbatch_fixture()
        edits = [lambda x: x.__setitem__('steps', 39), lambda x: x.__setitem__('nonfinite_batches', 1),
                 lambda x: x.__setitem__('epoch_complete', True), lambda x: x.__setitem__('future_microsteps', 2),
                 lambda x: x.__setitem__('skipped_optimizer_steps', 0.0),
                 lambda x: x['config']['control'].__setitem__('opt_step_every', 2),
                 lambda x: x['config']['supervised'].__setitem__('batch_size', 128),
                 lambda x: x['config']['supervised'].__setitem__('log_every', 127),
                 lambda x: x['config']['supervised'].__setitem__('save_every', 128),
                 lambda x: x['config']['supervised'].__setitem__('convergence', {'enabled': True}),
                 lambda x: x.__setitem__('adaptive_curriculum_state', {'last_gate_step': 32}),
                 lambda x: x['curriculum_probe']['dataset']['consumed'].__setitem__(1, 10241),
                 lambda x: x['curriculum_probe']['dataset'].__setitem__('micro_counter', 20)]
        for edit in edits:
            changed = deepcopy(source)
            edit(changed)
            with self.assertRaises(ValueError):
                microbatch_migration(changed, 512, 'throughput')
        for target, reason in ((256, 'same'), (128, 'unsupported'), (512, ''), (None, 'missing target')):
            with self.assertRaises(ValueError):
                microbatch_migration(source, target, reason)
        self.assertIsNone(microbatch_migration(source, None, ''))

    def test_large_horizon_and_cadences_are_explicit_not_maturity(self):
        plan = observation_plan(start=1000, until=200000, save_updates=1000, save_seconds=1800,
                                trend_every=2000, full_every=10000)
        self.assertIsNone(evaluation_kind(plan, 1000))
        self.assertEqual(evaluation_kind(plan, 2000), 'trend')
        self.assertEqual(evaluation_kind(plan, 10000), 'full')
        self.assertEqual(evaluation_horizons(plan)[-1], 200000)
        self.assertNotIn(1000, evaluation_horizons(plan))
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['prepare', '--source-run', 'old'])
        for change in ({'until': 10}, {'save_updates': 0}, {'save_seconds': float('inf')}, {'trend_every': -1}):
            args = dict(start=10, until=200000, save_updates=1000, save_seconds=1800, trend_every=2000, full_every=10000)
            with self.assertRaises(ValueError):
                observation_plan(**{**args, **change})

    def test_endpoint_always_full_even_off_cadence(self):
        plan = plan_fixture(until=23)
        self.assertEqual(evaluation_horizons(plan), [15, 20, 23])
        self.assertEqual(evaluation_kind(plan, 23), 'full')

    def test_full_panel_unchanged_fixed_trend_subset(self):
        full = {'controller_recent': list(range(512)), 'controller_old': list(range(1000, 1256))}
        before = deepcopy(full)
        a = trend_splits(full, recent_games=128, old_games=64, seed=17)
        self.assertEqual(a, trend_splits(full, recent_games=128, old_games=64, seed=17))
        self.assertEqual(full, before)
        self.assertEqual([len(a[k]) for k in a], [128, 64])
        for key in a:
            self.assertLessEqual(set(a[key]), set(full[key]))
        with self.assertRaises(ValueError):
            trend_splits(full, recent_games=513, old_games=64, seed=17)

    def test_validate_real_boundary_not_inferred_cursor_or_rng(self):
        source = saved_fixture()
        self.assertEqual(validate_saved_phase(source, DOMAINS, RECIPES)['next_update_lrs'], [7.5e-6] * 2)
        edits = [lambda x: x['curriculum_probe'].pop('rng'),
                 lambda x: x['curriculum_probe'].__setitem__('dataset', None),
                 lambda x: x.pop('auxiliary_optimizer_steps'),
                 lambda x: x.__setitem__('steps', 19),
                 lambda x: x.__setitem__('epoch_complete', True),
                 lambda x: x['curriculum_probe']['dataset']['consumed'].__setitem__(1, 79),
                 lambda x: x['scheduler'].__setitem__('last_epoch', 0),
                 lambda x: x['scheduler'].__setitem__('_last_lr', [5e-6] * 2),
                 lambda x: x['curriculum_probe']['dataset']['sampler'].__setitem__('seed', 42),
                 lambda x: x['config']['control'].__setitem__('device', 'cuda:0'),
                 lambda x: x['config']['control'].__setitem__('enable_amp', True),
                 lambda x: x['config']['supervised'].__setitem__('val_batch_size', 32),
                 lambda x: x['config']['supervised'].pop('val_batch_size')]
        for edit in edits:
            changed = deepcopy(source)
            edit(changed)
            with self.assertRaises((ValueError, KeyError)):
                validate_saved_phase(changed, DOMAINS, RECIPES)

    def test_amp_skips_preserve_real_consumption_not_success_count_guess(self):
        source = saved_fixture()
        source.update(skipped_optimizer_steps=1, steps=22)
        source['curriculum_probe']['dataset']['consumed'][1] = 88
        self.assertEqual(validate_saved_phase(source, DOMAINS, RECIPES)['consumed_decisions'], 88)

    def test_single_arm_metadata_relocation_keeps_every_training_field(self):
        source = saved_fixture()
        parent = parent_record(source, path='new/parent.pth', sha256='source', phase='B')
        source['config']['supervised']['run_provenance']['source_runtime_sha256'] = 'old-runtime'
        config = relocate_config(source['config'], 'new', 'new-id', parent, 'new-commit', 'old-experiment',
                                 runtime_sha256='new-runtime')
        self.assertEqual(training_semantics(source['config']), training_semantics(config))
        result = rebind_saved_state(source, config, 'new-id')
        for key in source:
            if key not in ('config', 'run_provenance', 'checkpoint_id', 'curriculum_probe'):
                self.assertEqual(source[key], result[key])
        self.assertEqual(result['curriculum_probe']['dataset'], source['curriculum_probe']['dataset'])
        self.assertEqual(result['curriculum_probe']['rng'], source['curriculum_probe']['rng'])
        self.assertEqual(result['run_provenance']['parent_chain'][-1], parent)
        self.assertEqual(result['run_provenance']['source_runtime_sha256'], 'new-runtime')
        self.assertEqual(result['run_provenance']['continuation']['source_runtime_sha256'], 'old-runtime')
        self.assertEqual(result['run_provenance']['phase'], 'B')
        self.assertEqual(result['optimizer_steps'], 10)
        self.assertEqual(source['curriculum_probe']['identity'], 'old-phase')
        for field, value in (('lr', 1e-4), ('seed', 42), ('batch_size', 8)):
            changed = deepcopy(config)
            changed['supervised'][field] = value
            with self.assertRaisesRegex(ValueError, 'training settings'):
                rebind_saved_state(source, changed, 'new-id')

    def test_ledger_backup_preserves_all_pins_including_other_arm_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'source.sqlite3', Path(tmp) / 'snapshot.sqlite3'
            ledger = TrainingContentLedger(source, 'old', create=True)
            ledger.verify([{'file': name, 'source_sha256': name * 8} for name in ('consumed', 'otherarm')])
            before = file_sha256(source)
            receipt = ledger_snapshot(source, target, 'old', {'consumed'})
            self.assertEqual(receipt['pinned_games'], 2)
            self.assertEqual(file_sha256(source), before)
            with closing(sqlite3.connect(source)) as left, closing(sqlite3.connect(target)) as right:
                self.assertEqual(left.execute('SELECT * FROM games ORDER BY path').fetchall(),
                                 right.execute('SELECT * FROM games ORDER BY path').fetchall())
            with self.assertRaises(FileExistsError):
                ledger_snapshot(source, target, 'old', {'consumed'})
            with self.assertRaisesRegex(ValueError, 'lacks a consumed'):
                ledger_snapshot(source, Path(tmp) / 'missing.sqlite3', 'old', {'never_pinned'})
            with self.assertRaisesRegex(ValueError, 'current-block fingerprint'):
                ledger_snapshot(source, Path(tmp) / 'mismatch.sqlite3', 'old', {'consumed'},
                                current_hashes={'consumed': 'different'})

    def test_output_and_mutable_latest_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.mkdir()
            separate_output(source, Path(tmp) / 'new')
            for bad in (source, source / 'child', Path(tmp)):
                with self.assertRaises(ValueError):
                    separate_output(source, bad)
            checkpoint = source / 'latest.pth'
            checkpoint.write_text('old')
            digest = file_sha256(checkpoint)
            checkpoint.write_text('new')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                require_stable_checkpoint(checkpoint, digest)

    def test_working_ledger_cannot_forget_or_change_inherited_pins(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            inherited, live = Path(tmp) / 'inherited.sqlite3', Path(tmp) / 'working.sqlite3'
            ledger = TrainingContentLedger(inherited, 'old', create=True)
            ledger.verify([{'file': 'game', 'source_sha256': 'a' * 64}])
            shutil.copy2(inherited, live)
            with closing(sqlite3.connect(live)) as db, db:
                db.execute('UPDATE metadata SET identity=?', ('new',))
            validate_inherited_pins(inherited, live, 'new', ['game'])
            with closing(sqlite3.connect(live)) as db, db:
                db.execute('UPDATE games SET sha256=? WHERE path=?', ('b' * 64, 'game'))
            with self.assertRaisesRegex(ValueError, 'lost or changed'):
                validate_inherited_pins(inherited, live, 'new')
            with closing(sqlite3.connect(live)) as db, db:
                db.execute('DELETE FROM games')
            with self.assertRaisesRegex(ValueError, 'lost or changed'):
                validate_inherited_pins(inherited, live, 'new')


@unittest.skipUnless(HAS_TORCH, 'requires real torch/toml CPU; not a native trainer acceptance')
class ContinuationTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        from mortal.supervised.curriculum_probe import capture_rng
        from mortal.supervised.lr_scheduler import LinearWarmUpConstantLR
        self.torch = torch
        self.source = saved_fixture()
        self.params = [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))]
        optimizer = torch.optim.AdamW([{'params': [p]} for p in self.params], lr=1)
        scheduler = LinearWarmUpConstantLR(optimizer, peak=1e-5, init=5e-6, warm_up_steps=20)
        for _ in range(10):
            for param in self.params:
                param.grad = torch.ones_like(param)
            optimizer.step()
            scheduler.step()
        self.source.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           scaler=torch.amp.GradScaler('cpu').state_dict())
        for name in ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net'):
            self.source[name] = {'weight': torch.ones(1)}
        self.source['curriculum_probe']['rng'] = capture_rng()

    def test_actual_adam_scaler_scheduler_rng_and_next_update_survive_torch_roundtrip(self):
        import numpy as np
        from mortal.supervised.curriculum_probe import restore_rng
        from mortal.supervised.lr_scheduler import LinearWarmUpConstantLR
        from scripts.continue_sl_phase import verify_rebind
        from scripts.verify_sl_probe_resume import equal
        parent = parent_record(self.source, path='new/parent.pth', sha256='parent', phase='B')
        config = relocate_config(self.source['config'], 'new', 'new', parent, 'commit', 'old', runtime_sha256='new-runtime')
        rebound = rebind_saved_state(self.source, config, 'new')
        verify_rebind(self.source, rebound)
        stream = io.BytesIO()
        self.torch.save(rebound, stream)
        stream.seek(0)
        rebound = self.torch.load(stream, weights_only=False)
        outputs = []
        for state in (self.source, rebound):
            params = [self.torch.nn.Parameter(p.detach().clone()) for p in self.params]
            optimizer = self.torch.optim.AdamW([{'params': [p]} for p in params], lr=1)
            scheduler = LinearWarmUpConstantLR(optimizer, peak=1e-5, init=5e-6, warm_up_steps=20)
            optimizer.load_state_dict(deepcopy(state['optimizer']))
            scheduler.load_state_dict(deepcopy(state['scheduler']))
            scaler = self.torch.amp.GradScaler('cpu')
            scaler.load_state_dict(state['scaler'])
            restore_rng(state['curriculum_probe']['rng'])
            draw = random.random() + np.random.rand()
            loss = sum((p * self.torch.rand_like(p) * draw).sum() for p in params)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            outputs.append((params, optimizer.state_dict(), scaler.state_dict(), scheduler.state_dict()))
        self.assertTrue(equal(outputs[0], outputs[1]))
        self.assertEqual(outputs[1][-1]['last_epoch'], 11)
        self.assertAlmostEqual(outputs[1][-1]['_last_lr'][0], 7.75e-6)

    def make_probe(self, directory, *, seal_at=None):
        from mortal.supervised.continuous_probe import ContinuousPhaseProbe
        return ContinuousPhaseProbe(self.source['config'], DOMAINS, recipe='B', seed=31,
            output=directory, identity='old-phase', plan=plan_fixture(),
            full_splits={'controller_recent': ['full'], 'controller_old': ['old']},
            trend_splits={'controller_recent': ['trend'], 'controller_old': ['old-trend']}, seal_at=seal_at)

    def test_restore_does_not_repeat_start_validation_or_reset_cursor_and_rng(self):
        from scripts.verify_sl_probe_resume import equal
        with tempfile.TemporaryDirectory() as tmp:
            probe = self.make_probe(tmp)
            probe.restore(self.source)
            self.assertTrue(equal(probe.pending_dataset, self.source['curriculum_probe']['dataset']))
            evaluate, save = Mock(), Mock()
            probe.observe(10, evaluate, None, save, 0)
            evaluate.assert_not_called()
            save.assert_not_called()

    def test_actual_mid_block_cursor_yields_same_next_rows_after_metadata_relocation(self):
        from mortal.supervised.curriculum_probe import RotatingGameDataset
        from scripts.verify_sl_probe_resume import equal
        names = {'2024game': 1, '2012game': 2}
        module = SimpleNamespace(stable_source_game_id=lambda filename: names[filename])
        loader = lambda filename: [(filename, row) for row in range(7)]
        with patch.dict('sys.modules', {'mortal.data.dataloader': module}):
            original = RotatingGameDataset(DOMAINS, RECIPES['B'], 31, {}, sample_loader=loader)
            stream = iter(original)
            for _ in range(80):
                next(stream)
            self.source['curriculum_probe']['dataset'] = original.state_dict()
            # sample_loader fixtures skip native file hashing; provide its
            # explicit pinned-content field, not an invented data position.
            for draw in self.source['curriculum_probe']['dataset']['current']['draws']:
                draw['source_sha256'] = 'f' * 64
            validate_saved_phase(self.source, DOMAINS, RECIPES)
            parent = parent_record(self.source, path='new/parent.pth', sha256='parent', phase='B')
            config = relocate_config(self.source['config'], 'new', 'new', parent, 'commit', 'old', runtime_sha256='new-runtime')
            rebound = rebind_saved_state(self.source, config, 'new')
            resumed = RotatingGameDataset(DOMAINS, RECIPES['B'], 31, {}, sample_loader=loader)
            resumed.load_state_dict(rebound['curriculum_probe']['dataset'])
            continued = iter(resumed)
            self.assertEqual([next(stream) for _ in range(64)], [next(continued) for _ in range(64)])
            self.assertTrue(equal(original.state_dict(), resumed.state_dict()))

    def test_microbatch_migration_preserves_real_state_and_next_logical_1024_rows(self):
        from mortal.supervised.curriculum_probe import RotatingGameDataset
        from scripts.continue_sl_phase import verify_rebind
        from scripts.verify_sl_probe_resume import equal
        names = {'2024game': 1, '2012game': 2}
        module = SimpleNamespace(stable_source_game_id=lambda filename: names[filename])
        loader = lambda filename: [(filename, row) for row in range(513)]
        source = microbatch_fixture(self.source)
        with patch.dict('sys.modules', {'mortal.data.dataloader': module}):
            original = RotatingGameDataset(DOMAINS, RECIPES['B'], 31, {}, sample_loader=loader)
            stream = iter(original)
            for _ in range(10240):
                next(stream)
            source['curriculum_probe']['dataset'] = original.state_dict()
            for draw in source['curriculum_probe']['dataset']['current']['draws']:
                draw['source_sha256'] = 'f' * 64
            validate_saved_phase(source, DOMAINS, RECIPES)
            migration = microbatch_migration(source, 512, 'declared throughput protocol')
            parent = parent_record(source, path='new/parent.pth', sha256='parent', phase='B')
            config = relocate_config(source['config'], 'new', 'new', parent, 'commit', 'old',
                                     runtime_sha256='new-runtime', migration=migration)
            rebound = rebind_saved_state(source, config, 'new', migration=migration)
            verify_rebind(source, rebound, migration=migration)
            buffer = io.BytesIO()
            self.torch.save(rebound, buffer)
            buffer.seek(0)
            rebound = self.torch.load(buffer, weights_only=False)
            verify_rebind(source, rebound, migration=migration)
            validate_saved_phase(rebound, DOMAINS, RECIPES)
            rows, datasets = [], []
            for state, microbatch in ((source, 256), (rebound, 512)):
                dataset = RotatingGameDataset(DOMAINS, RECIPES['B'], 31, {}, sample_loader=loader)
                dataset.load_state_dict(state['curriculum_probe']['dataset'])
                batches = iter(self.torch.utils.data.DataLoader(dataset, batch_size=microbatch,
                    num_workers=0, generator=self.torch.Generator().manual_seed(31)))
                logical = []
                for _ in range(1024 // microbatch):
                    filenames, offsets = next(batches)
                    logical.extend(zip(filenames, offsets.tolist()))
                rows.append(logical)
                datasets.append(dataset)
            self.assertEqual(len(rows[0]), 1024)
            self.assertEqual(rows[0], rows[1])
            self.assertTrue(equal(datasets[0].state_dict(), datasets[1].state_dict()))
            self.assertEqual(rebound['optimizer_steps'], 10)
            self.assertEqual(rebound['steps'], 20)
            self.assertEqual(rebound['scheduler'], source['scheduler'])

    def test_prepare_materializes_independent_identity_and_frozen_inputs_without_reset(self):
        self.check_prepare_materialization(migrate=False)

    def test_prepare_materializes_declared_microbatch_protocol_identity(self):
        self.check_prepare_materialization(migrate=True)

    def test_reference_pins_preserve_late_state_and_require_complete_receipt(self):
        self.check_prepare_materialization(migrate=True, reference=True)

    def check_prepare_materialization(self, *, migrate, reference=False):
        from scripts.continue_sl_phase import prepare, verify_rebind
        from scripts.run_sl_early_transition import verify_manifest
        from mortal.core.toml_utils import load_toml_file
        if migrate:
            self.source = microbatch_fixture(self.source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root, target = root / 'old', root / 'new'
            source_root.mkdir()
            runtime = source_root / 'source'
            runtime.mkdir()
            # File-copy fixture only: no native module or trainer is executed.
            native = runtime / 'libriichi.pyd'
            native.write_bytes(b'not-an-executable-native-fixture')
            recent, old = source_root / 'recent.json', source_root / 'old.json'
            recent.write_text('recent')
            old.write_text('old')
            index = {'domains': DOMAINS, 'train_files': DOMAINS['recent'] + DOMAINS['replay'],
                     'roles': {'controller_recent': [str(recent)], 'controller_old': [str(old)]}}
            self.torch.save(index, source_root / 'indexes.pth')
            manifest = {'directory': str(source_root), 'source_root': str(runtime), 'parents': {},
                'source_sha256': {'libriichi.pyd': file_sha256(native)},
                'input_sha256': {'indexes.pth': file_sha256(source_root / 'indexes.pth')},
                'controller_source_sha256': {str(p): file_sha256(p) for p in (recent, old)},
                'gpu_memory_fraction': 0.5, 'training_content': 'fixture'}
            manifest['identity'] = stable_json_digest(manifest)
            atomic_write_json(source_root / 'manifest.json', manifest)
            source_ledger = source_root / 'training_content.sqlite3'
            ledger = TrainingContentLedger(source_ledger, manifest['identity'], create=True)
            ledger.verify([{'file': '2024game', 'source_sha256': 'a' * 64}])
            self.source['run_provenance']['experiment_id'] = manifest['identity']
            self.source['config']['supervised'].update(
                file_index=str(source_root / 'indexes.pth'), probe_training_content_ledger=str(source_ledger),
                run_provenance=deepcopy(self.source['run_provenance']))
            checkpoint = source_root / 'point.pth'
            self.torch.save(self.source, checkpoint)
            reference_contract = None
            if reference:
                ref = root / 'reference'
                ref.mkdir()
                self.torch.save(index, ref / 'indexes.pth')
                atomic_write_json(ref / 'source_config.json', self.source['config'])
                ref_manifest = dict(manifest, directory=str(ref), phase='B',
                    input_sha256={name: file_sha256(ref / name) for name in
                                  ('indexes.pth', 'source_config.json')})
                ref_manifest.pop('identity')
                ref_manifest['identity'] = stable_json_digest(ref_manifest)
                atomic_write_json(ref / 'manifest.json', ref_manifest)
                ref_ledger = TrainingContentLedger(ref / 'training_content.sqlite3',
                                                   ref_manifest['identity'], create=True)
                ref_ledger.verify([{'file': '2024game', 'source_sha256': 'a' * 64},
                                   {'file': '2012game', 'source_sha256': 'b' * 64}])
                first = ledger_snapshot(source_ledger, root / 'first.sqlite3',
                                         manifest['identity'], ('2024game',))
                second = ledger_snapshot(ref_ledger.path, root / 'second.sqlite3',
                                          ref_manifest['identity'], ())
                descriptor = dict(format='sl_reference_content_contract_v1',
                    source_identity=manifest['identity'], source_manifest_sha256=file_sha256(source_root / 'manifest.json'),
                    source_checkpoint_sha256=file_sha256(checkpoint), source_ledger_content_sha256=first['content_sha256'],
                    reference_directory=str(ref), reference_identity=ref_manifest['identity'],
                    reference_manifest_sha256=file_sha256(ref / 'manifest.json'),
                    reference_ledger_content_sha256=second['content_sha256'],
                    domains_sha256=stable_json_digest(index['domains']), roles_sha256=stable_json_digest(index['roles']),
                    phase='B', seed=31, recipe=RECIPES['B'],
                    order_evidence={'artifact_sha256': 'fixture', 'next_logical_1024_row_identity_sha256': 'fixture',
                                    'reference_end_checkpoint_sha256': 'fixture'})
                reference_contract = root / 'reference_contract.json'
                atomic_write_json(reference_contract, descriptor)
            before = {path: file_sha256(path) for path in source_root.rglob('*') if path.is_file()}
            args = SimpleNamespace(source_run=str(source_root), directory=str(target), source_commit='new-fixed-commit',
                checkpoint=str(checkpoint), until_update=100, save_every_updates=10, save_every_seconds=1800,
                trend_every_updates=20, full_every_updates=50, trend_recent_games=1, trend_old_games=1,
                trend_seed=3, runtime_change_reason='', check_only=False,
                microbatch=512 if migrate else None, microbatch_change_reason='measured throughput' if migrate else '',
                reference_content_contract=str(reference_contract) if reference_contract else None)
            with patch('scripts.continue_sl_phase.fixed_source', return_value='new-fixed-commit'), patch('sys.stdout', new=io.StringIO()):
                prepare(args)
            actual_manifest = verify_manifest(target)
            actual = self.torch.load(target / 'state_file.pth', weights_only=False)
            self.assertEqual(actual['optimizer_steps'], 10)
            self.assertNotEqual(actual_manifest['identity'], manifest['identity'])
            self.assertEqual(actual['curriculum_probe']['identity'], actual_manifest['identity'])
            self.assertEqual(actual['config'], load_toml_file(target / 'config.toml'))
            verify_rebind(self.source, actual, migration=actual_manifest['microbatch_migration'])
            if migrate:
                self.assertEqual(actual['steps'], 20)
                self.assertEqual(actual['config']['control']['opt_step_every'], 2)
                self.assertEqual(actual['config']['supervised']['batch_size'], 512)
                self.assertEqual(actual_manifest['source_state_contract']['microsteps'], 40)
                self.assertEqual(actual_manifest['microbatch_migration']['source_microsteps'], 40)
                self.assertIn('declared microbatch numerical branch', actual_manifest['resume_scope'])
            self.assertEqual(before, {path: file_sha256(path) for path in before})
            self.assertEqual(actual_manifest['gpu_memory_fraction'], 0.5)
            with closing(sqlite3.connect(target / 'training_content.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT identity FROM metadata').fetchone()[0], actual_manifest['identity'])
                self.assertEqual(db.execute('SELECT count(*) FROM games').fetchone()[0], 2 if reference else 1)
            if reference:
                self.assertEqual(actual['curriculum_probe']['dataset'], self.source['curriculum_probe']['dataset'])
                self.assertFalse(actual_manifest['ledger_snapshot']['reference_progress_imported'])
                self.assertEqual(actual_manifest['reference_content']['reference_identity'], ref_manifest['identity'])
                self.assertEqual(file_sha256(ref / 'manifest.json'), descriptor['reference_manifest_sha256'])
                (target / 'continuation_receipt.json').unlink()
                from scripts.continue_sl_phase import run
                with patch('scripts.continue_sl_phase.require_frozen_runtime'), self.assertRaises(FileNotFoundError):
                    run(SimpleNamespace(directory=str(target), seal=False))

    def test_latest_update_and_wall_clock_saves_do_not_call_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = self.make_probe(tmp)
            probe.restore(self.source)
            evaluate, save = Mock(), Mock()
            probe.after_update(11, evaluate, None, save, 0)
            save.assert_not_called()
            probe.after_update(13, evaluate, None, save, 0)
            self.assertEqual(save.call_count, 1)
            probe.last_saved_time -= 1801
            probe.after_update(14, evaluate, None, save, 0)
            self.assertEqual(save.call_count, 2)
            evaluate.assert_not_called()

    def test_trend_is_lightweight_full_archives_and_rng_is_preserved(self):
        from mortal.supervised.curriculum_probe import capture_rng
        from scripts.verify_sl_probe_resume import equal
        with tempfile.TemporaryDirectory() as tmp:
            probe = self.make_probe(tmp)
            probe.restore(self.source)
            before = capture_rng()
            def evaluate(*args, **kwargs):
                self.torch.rand(3)
                random.random()
                return {'policy_loss': 0.4}, 1
            evaluate = Mock(side_effect=evaluate)
            save = Mock()
            build = Mock(side_effect=lambda *args, **kwargs: {**self.source, 'curriculum_probe': probe.state_dict()})
            probe.after_update(15, evaluate, build, save, 0)
            self.assertEqual(evaluate.call_count, 2)
            self.assertFalse(evaluate.call_args.kwargs['collect_cluster_records'])
            build.assert_not_called()
            self.assertTrue(equal(before, capture_rng()))
            self.assertTrue((Path(tmp) / 'trend/update_000000015.json').exists())
            self.assertFalse(list(Path(tmp).rglob('*.pth')))
            probe.after_update(20, evaluate, build, save, 0)
            self.assertTrue(evaluate.call_args.kwargs['collect_cluster_records'])
            self.assertTrue((Path(tmp) / 'full/update_000000020.pth').exists())
            self.assertTrue(equal(before, capture_rng()))
            self.assertEqual(probe.done, {'trend': [15], 'full': [0, 10, 20]})

    def test_resume_after_result_before_latest_reuses_result_without_revalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = self.make_probe(tmp)
            probe.restore(self.source)
            evaluate = Mock(return_value=({'policy_loss': 0.4}, 1))
            probe.observe(15, evaluate, None, Mock(), 0)
            resumed = self.make_probe(tmp)
            resumed.restore(self.source)
            no_evaluation = Mock(side_effect=AssertionError('must reuse completed trend receipt'))
            resumed.observe(15, no_evaluation, None, Mock(), 0)
            no_evaluation.assert_not_called()
            self.assertEqual(resumed.done['trend'], [15])

    def test_seal_forces_full_at_current_update_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = self.make_probe(tmp, seal_at=17)
            probe.restore(self.source)
            self.assertIn(17, probe.horizons)
            self.assertEqual(probe.stop_at, 17)
            evaluate = Mock(return_value=({'policy_loss': 0.4}, 1))
            build = lambda *args, **kwargs: {**self.source, 'curriculum_probe': probe.state_dict()}
            probe.observe(17, evaluate, build, Mock(), 0)
            self.assertEqual(evaluate.call_count, 2)
            self.assertIn(17, probe.done['full'])


if __name__ == '__main__':
    unittest.main()

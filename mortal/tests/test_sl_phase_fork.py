"""Small CPU tests for the new fork boundary, baseline reuse, and C resume."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock

import torch

from mortal.core.artifacts import atomic_torch_save, stable_json_digest
from mortal.supervised.curriculum_probe import capture_rng, learned_state_digest
from mortal.supervised.early_transition import LEARNED_KEYS, parent_record, phase_spec, prepare_transition_state
from mortal.supervised.phase_fork import PhaseForkProbe, fork_config, validate_baseline
from mortal.tests.test_sl_continuation import DOMAINS, microbatch_fixture
from scripts.run_sl_early_transition import branch_provenance
from scripts.verify_sl_probe_resume import equal


def fixture(output):
    source = microbatch_fixture(batch=512)
    source['checkpoint_id'] = 'B50'
    chain = [{'phase': phase, 'checkpoint_id': ident, 'sha256': ident, 'checkpoint': ident}
             for phase, ident in [('A', 'A'), ('B', 'B5'), ('B', 'B7')]]
    source['run_provenance']['parent_chain'] = chain
    source['config']['supervised']['run_provenance'] = deepcopy(source['run_provenance'])
    parent = parent_record(source, path=output / 'parent.pth', sha256='parent-sha', phase='B')
    spec = phase_spec(updates=50000, observations=[1000, 5000, 10000, 20000, 50000],
                      seed=2026100202, peak=1e-5, init=1e-5, warmup=0)
    config = fork_config(source['config'], output, 'new-C', parent, 'commit', 'runtime', spec)
    return source, config, parent, spec


class PhaseForkTests(unittest.TestCase):
    def test_append_full_chain_without_mutating_source_or_old_gate(self):
        source, config, parent, spec = fixture(Path('new'))
        self.assertEqual([r['checkpoint_id'] for r in config['supervised']['run_provenance']['parent_chain']],
                         ['A', 'B5', 'B7', 'B50'])
        self.assertEqual(len(source['run_provenance']['parent_chain']), 3)
        self.assertEqual(config['supervised']['seed'], 2026100202)
        self.assertEqual(config['supervised']['scheduler']['warm_up_steps'], 0)
        self.assertEqual(config['aux'], source['config']['aux'])
        self.assertEqual(config['control'], source['config']['control'])
        with self.assertRaisesRegex(ValueError, 'original A parent chain'):
            branch_provenance(source, {'phases': {'C': spec}}, 'late', 'C', parent)

    def test_reject_missing_foreign_and_duplicate_ancestry(self):
        source, _, parent, spec = fixture(Path('new'))
        for chain in [[], [{'phase': 'C', 'checkpoint_id': 'C'}],
                      [{'phase': 'A', 'checkpoint_id': 'B50'}],
                      [{'phase': 'A', 'checkpoint_id': 'A'}, {'phase': 'C', 'checkpoint_id': 'C'}]]:
            bad = deepcopy(source['config'])
            bad['supervised']['run_provenance']['parent_chain'] = chain
            with self.assertRaises(ValueError):
                fork_config(bad, Path('new'), 'id', parent, 'commit', 'runtime', spec)

    def test_actual_transition_keeps_all_learned_state_and_mature_clocks(self):
        source, config, _, _ = fixture(Path('new'))
        fork = prepare_transition_state(source, config)
        for key in LEARNED_KEYS:
            if key == 'optimizer':
                self.assertTrue(equal(source[key]['state'], fork[key]['state']))
                for old, new in zip(source[key]['param_groups'], fork[key]['param_groups']):
                    self.assertEqual({k: v for k, v in old.items() if k not in ('lr', 'initial_lr')},
                                     {k: v for k, v in new.items() if k not in ('lr', 'initial_lr')})
            else:
                self.assertTrue(equal(source[key], fork[key]), key)
        self.assertEqual(fork['auxiliary_optimizer_steps'], source['auxiliary_optimizer_steps'])
        self.assertEqual(fork['optimizer_steps'], 0)
        self.assertEqual(fork['steps'], 0)
        self.assertEqual(fork['scheduler']['last_epoch'], 0)
        self.assertEqual(fork['scheduler']['_last_lr'], [1e-5, 1e-5])
        self.assertNotIn('curriculum_probe', fork)
        self.assertNotIn('adaptive_curriculum_state', fork)

    def test_baseline_requires_exact_state_identity_update_and_panel(self):
        source, _, _, _ = fixture(Path('new'))
        row = {'identity': source['run_provenance']['plan_id'], 'optimizer_updates': source['optimizer_steps'],
               'kind': 'full', 'split_identity': 'roles', 'learned_state_sha256': learned_state_digest(source)}
        validate_baseline(row, source, 'roles')
        for key, bad in [('identity', 'foreign'), ('optimizer_updates', 9), ('kind', 'trend'),
                         ('split_identity', 'other'), ('learned_state_sha256', 'changed')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_baseline({**row, key: bad}, source, 'roles')

    def make_probe(self, output, baseline=None):
        source, config, _, spec = fixture(output)
        probe = PhaseForkProbe(config, DOMAINS, recipe='C', seed=spec['seed'], output=output / 'full',
            horizons=spec['observations'], eval_splits={'controller_recent': ['a'], 'controller_old': ['b']},
            identity='new-C', baseline=baseline)
        return source, prepare_transition_state(source, config), probe

    def test_C_rng_resets_once_then_restores_saved_C_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state, probe = self.make_probe(Path(tmp))
            probe.restore(state)
            self.assertEqual(random.random(), random.Random(2026100202).random())
            saved = probe.state_dict()
            saved['dataset'] = {'C_cursor': 123}
            expected_next = random.random()
            random.seed(8)
            _, _, resumed = self.make_probe(Path(tmp))
            resumed.restore({**state, 'optimizer_steps': 10, 'curriculum_probe': saved})
            self.assertEqual(random.random(), expected_next)
            self.assertEqual(resumed.pending_dataset, {'C_cursor': 123})
            self.assertEqual(resumed.last_saved_update, 10)

    def test_U0_reuses_metrics_without_evaluation_and_saves_real_C_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = {'identity': 'B-parent', 'optimizer_updates': 50000, 'seed': 1, 'recipe': 'B',
                        'splits': {'controller_recent': {'policy_loss': .45}}, 'evaluation_seconds': 369}
            _, state, probe = self.make_probe(Path(tmp), baseline)
            probe.restore(state)
            evaluate, save = Mock(), Mock()
            build = lambda *a, **kw: {**state, 'curriculum_probe': probe.state_dict()}
            probe.observe(0, evaluate, build, save, 0)
            evaluate.assert_not_called()
            row = json.loads((Path(tmp) / 'full/update_0000000.json').read_text())
            self.assertEqual(row['splits'], baseline['splits'])
            self.assertEqual(row['optimizer_updates'], 0)
            self.assertEqual(row['evaluation_seconds'], 0)
            self.assertEqual(row['reused_parent_observation']['optimizer_updates'], 50000)
            self.assertEqual(probe.observed, [0])
            self.assertEqual(save.call_count, 2)

    def test_partial_observation_is_preserved_and_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state, probe = self.make_probe(Path(tmp))
            path = Path(tmp) / 'full/update_0001000.pth'
            path.write_bytes(b'existing partial evidence')
            with self.assertRaises(FileExistsError):
                probe.observe(1000, Mock(), Mock(return_value=state), Mock(), 0)
            self.assertEqual(path.read_bytes(), b'existing partial evidence')

    def interrupt_after_observation(self, output, update):
        baseline = {'identity': 'B-parent', 'optimizer_updates': 50000, 'seed': 1, 'recipe': 'B',
                    'splits': {name: {'policy_loss': .45}
                               for name in ('controller_recent', 'controller_old')}} if update == 0 else None
        _, state, probe = self.make_probe(output, baseline)
        probe.restore(state)
        probe.observed = [value for value in [0, *probe.horizons] if value < update]
        probe.pending_dataset = {'C_cursor': 123}
        state.update(optimizer_steps=update, steps=update * state['config']['control']['opt_step_every'])
        latest = output / 'latest.pth'

        def build(*args, **kwargs):
            return {**state, 'curriculum_probe': probe.state_dict()}

        def save(*args, reason, **kwargs):
            if reason != 'before_phase_fork_observation':
                raise OSError('injected latest write failure')
            atomic_torch_save(build(), latest)

        def evaluate(*args, **kwargs):
            random.random()
            return {'policy_loss': .45}, 1

        with self.assertRaisesRegex(OSError, 'injected latest write failure'):
            probe.observe(update, evaluate, build, save, 0)
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        self.assertNotIn(update, saved['curriculum_probe']['observed'])
        paths = [output / f'full/update_{update:07d}.{suffix}' for suffix in ('json', 'pth')]
        self.assertTrue(all(path.is_file() for path in paths))
        return saved, baseline, latest, paths

    def test_recover_completed_observation_after_latest_write_failure(self):
        for update in (0, 1000, 5000, 10000, 20000, 50000):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                saved, baseline, latest, paths = self.interrupt_after_observation(output, update)
                evidence = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
                _, _, resumed = self.make_probe(output, baseline)
                resumed.restore(saved)

                def build(*args, **kwargs):
                    return {**saved, 'curriculum_probe': resumed.state_dict()}

                save = Mock(side_effect=lambda *a, **kw: atomic_torch_save(build(), latest))
                evaluate = Mock()
                resumed.observe(update, evaluate, build, save, 0)
                evaluate.assert_not_called()
                save.assert_called_once_with(0, epoch_complete=False, reason='recover_completed_observation')
                self.assertEqual(resumed.observed, [*saved['curriculum_probe']['observed'], update])
                self.assertEqual(resumed.last_saved_update, update)
                recovered = torch.load(latest, map_location='cpu', weights_only=False)
                self.assertEqual(recovered['curriculum_probe']['observed'], resumed.observed)
                self.assertEqual(recovered['curriculum_probe']['dataset'], saved['curriculum_probe']['dataset'])
                self.assertTrue(equal(recovered['curriculum_probe']['rng'], saved['curriculum_probe']['rng']))
                self.assertEqual(evidence, [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths])
                resumed.observe(update, evaluate, build, save, 0)
                self.assertEqual(save.call_count, 1)
                self.assertFalse(torch.cuda.is_initialized())

    def test_recovery_rejects_partial_or_mismatched_evidence_without_writing(self):
        changes = {'identity': 'foreign', 'optimizer_updates': 999, 'recipe': 'B', 'seed': 17,
                   'kind': 'trend', 'split_identity': 'foreign-panel', 'checkpoint': 'foreign.pth',
                   'checkpoint_sha256': 'changed', 'learned_state_sha256': 'changed',
                   'splits': {'controller_recent': {'policy_loss': .45}}}
        cases = [*changes, 'empty_metrics', 'missing_field', 'invalid_json', 'missing_json',
                 'missing_checkpoint', 'changed_checkpoint']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                saved, baseline, latest, paths = self.interrupt_after_observation(output, 1000)
                result_path, checkpoint = paths
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if case in changes:
                    result[case] = changes[case]
                    result_path.write_text(json.dumps(result), encoding='utf-8')
                elif case == 'empty_metrics':
                    result['splits']['controller_old'] = {}
                    result_path.write_text(json.dumps(result), encoding='utf-8')
                elif case == 'missing_field':
                    del result['evaluation_seconds']
                    result_path.write_text(json.dumps(result), encoding='utf-8')
                elif case == 'invalid_json':
                    result_path.write_text('{', encoding='utf-8')
                elif case == 'missing_json':
                    result_path.unlink()
                elif case == 'missing_checkpoint':
                    checkpoint.unlink()
                else:
                    checkpoint.write_bytes(b'changed archive')

                def evidence():
                    return [(path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None
                            for path in [*paths, latest]]

                before = evidence()
                _, _, resumed = self.make_probe(output, baseline)
                resumed.restore(saved)
                evaluate, save = Mock(), Mock()
                with self.assertRaises((FileExistsError, ValueError)):
                    resumed.observe(1000, evaluate, Mock(return_value=saved), save, 0)
                evaluate.assert_not_called()
                save.assert_not_called()
                self.assertEqual(resumed.observed, saved['curriculum_probe']['observed'])
                self.assertEqual(before, evidence())

    def test_recovery_requires_latest_to_have_the_observed_learned_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            saved, baseline, _, paths = self.interrupt_after_observation(output, 1000)
            before = [path.read_bytes() for path in paths]
            saved['auxiliary_optimizer_steps'] += 1
            _, _, resumed = self.make_probe(output, baseline)
            resumed.restore(saved)
            evaluate, save = Mock(), Mock()
            with self.assertRaisesRegex(ValueError, 'different learned state'):
                resumed.observe(1000, evaluate, Mock(return_value=saved), save, 0)
            evaluate.assert_not_called()
            save.assert_not_called()
            self.assertEqual(resumed.observed, saved['curriculum_probe']['observed'])
            self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_sparse_observations_periodic_save_and_fixed_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, probe = self.make_probe(Path(tmp))
            self.assertEqual([0, *probe.horizons], [0, 1000, 5000, 10000, 20000, 50000])
            probe.observe = Mock()
            save = Mock()
            self.assertFalse(probe.after_update(64, None, None, save, 0))
            save.assert_not_called()
            self.assertFalse(probe.after_update(500, None, None, save, 0))
            self.assertEqual(save.call_count, 1)
            self.assertFalse(probe.after_update(1000, None, None, save, 0))
            self.assertEqual(probe.observe.call_args.args[0], 1000)
            self.assertTrue(probe.after_update(50000, None, None, save, 0))
            self.assertEqual(probe.observe.call_args.args[0], 50000)


if __name__ == '__main__':
    unittest.main()

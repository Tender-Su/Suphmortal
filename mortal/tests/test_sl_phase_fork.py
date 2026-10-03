"""Small CPU tests for the new fork boundary, baseline reuse, and C resume."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock

from mortal.core.artifacts import stable_json_digest
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

    def test_sparse_observations_periodic_save_and_fixed_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, probe = self.make_probe(Path(tmp))
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

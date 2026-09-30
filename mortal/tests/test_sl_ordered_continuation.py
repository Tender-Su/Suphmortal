from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mortal.core.artifacts import file_sha256
from mortal.tests.test_sl_runtime_continuation import RuntimeContinuationTests
from scripts.continue_sl_ordered_runtime import rebind_checkpoint, observation_metadata, qualify
from scripts.verify_sl_probe_resume import equal


class OrderedContinuationTests(unittest.TestCase):
    def test_complete_cursor_and_optimizer_survive_only_performance_and_path_rebinding(self):
        source = RuntimeContinuationTests().state()
        before = deepcopy(source)
        config = deepcopy(source['config'])
        config['supervised'].update(probe_prepare_workers=4, val_prepare_workers=4,
            prepare_rayon_threads=4, probe_prepare_file_batch_size=4, val_file_batch_size=4,
            rayon_num_threads=4, state_file='new', file_index='new-index')
        config['supervised']['run_provenance']['plan_id'] = 'new-id'
        rebound = rebind_checkpoint(source, config, 'new-id')
        self.assertTrue(equal(source, before))
        rebound['config'] = source['config']
        rebound['curriculum_probe']['identity'] = source['curriculum_probe']['identity']
        self.assertTrue(equal(source, rebound))
        for section, key, value in (('aux', 'danger_weight', 0), ('control', 'opt_step_every', 2),
                                    ('supervised', 'num_workers', 4), ('supervised', 'batch_size', 512)):
            bad = deepcopy(config)
            bad[section][key] = value
            with self.assertRaisesRegex(ValueError, 'training semantics'):
                rebind_checkpoint(source, bad, 'new-id')

    def test_observation_preserves_metrics_and_explicit_chain_to_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / 'old.json', root / 'new.pth'
            old = {'identity': 'old-arm', 'checkpoint_sha256': 'old-sha',
                   'splits': {'recent': {'cluster': [1, 2, 3]}}, 'exposure': {'decisions': 99}}
            previous.write_text(json.dumps(old))
            current.write_bytes(b'rebound checkpoint')
            result = observation_metadata(old, identity='new-arm', checkpoint=current,
                previous_observation=previous, previous_checkpoint_sha256='old-sha', source_identity='old-run')
            self.assertEqual(result['splits'], old['splits'])
            self.assertEqual(result['exposure'], old['exposure'])
            self.assertEqual(result['checkpoint_sha256'], file_sha256(current))
            self.assertEqual(result['imported_from']['source_identity'], 'old-run')
            self.assertEqual(old['identity'], 'old-arm')
            with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
                observation_metadata(old, identity='new-arm', checkpoint=current,
                    previous_observation=previous, previous_checkpoint_sha256='changed', source_identity='old-run')

    def test_throughput_only_or_nonresumed_proof_cannot_activate_runtime(self):
        base = {'format': 'sl_ordered_preparation_benchmark_v1', 'passed': True,
            'resume_from': 'baseline', 'checks': {'learned_state_exact': True},
            'final_update': 36, 'skipped_updates': 0}
        with self.assertRaisesRegex(ValueError, 'complete metrics'):
            qualify(base, {}, {}, {}, Path('unused'))
        with self.assertRaisesRegex(ValueError, 'resumed-state'):
            qualify({**base, 'resume_from': None}, {}, {}, {}, Path('unused'))


if __name__ == '__main__':
    unittest.main()

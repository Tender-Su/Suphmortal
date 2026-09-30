from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.continue_sl_curriculum_runtime import PERFORMANCE, assert_quiescent, rebind_state
from scripts.verify_sl_probe_resume import equal


class RuntimeContinuationTests(unittest.TestCase):
    def state(self):
        state = {name: {'weight': torch.tensor([1.0, 2.0])} for name in
                 ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net')}
        state.update(optimizer={'state': {1: {'step': torch.tensor(123), 'exp_avg': torch.tensor([0.1])}}},
            optimizer_param_groups=[('mortal.weight',)], scheduler={'lr': 5e-6}, scaler={'scale': 128},
            auxiliary_optimizer_steps=123, optimizer_steps=4, steps=16,
            config={'control': {'opt_step_every': 4}, 'aux': {'danger_weight': 0.01},
                    'supervised': {'state_file': 'old', 'file_index': 'old-index', 'batch_size': 256,
                                   'val_batch_size': 1024, 'num_workers': 0,
                                   'run_provenance': {'plan_id': 'old-id', 'probe': True}}},
            curriculum_probe={'identity': 'old-id', 'dataset': {'offset': 7, 'sampler': {'draws': 4}},
                              'rng': {'torch': torch.tensor([1, 2], dtype=torch.uint8)},
                              'observed': [0], 'elapsed_seconds': 19.5})
        return state

    def test_metadata_relocation_preserves_all_numerical_and_cursor_state(self):
        source = self.state()
        original = deepcopy(source)
        config = deepcopy(source['config'])
        config['supervised'].update(PERFORMANCE, state_file='new', file_index='new-index')
        config['supervised']['run_provenance']['plan_id'] = 'new-id'
        actual = rebind_state(source, config, 'new-id')
        self.assertTrue(equal(source, original))
        actual['config'] = source['config']
        actual['curriculum_probe']['identity'] = source['curriculum_probe']['identity']
        self.assertTrue(equal(source, actual))

    def test_semantic_changes_are_rejected(self):
        for section, name, value in (('aux', 'danger_weight', 0), ('supervised', 'num_workers', 2),
                                     ('supervised', 'batch_size', 128), ('supervised', 'val_batch_size', 512)):
            source = self.state()
            config = deepcopy(source['config'])
            config[section][name] = value
            with self.assertRaisesRegex(ValueError, 'training semantics'):
                rebind_state(source, config, 'new-id')

    def test_activation_refuses_any_source_process(self):
        with patch('scripts.continue_sl_curriculum_runtime.subprocess.run',
                   return_value=Mock(stdout='{"ProcessId":42,"Name":"python.exe"}')):
            with self.assertRaisesRegex(RuntimeError, 'still references'):
                assert_quiescent(Path('C:/old-run'), Path('C:/old-runtime'))
        with patch('scripts.continue_sl_curriculum_runtime.subprocess.run', return_value=Mock(stdout='[]')):
            assert_quiescent(Path('C:/old-run'), Path('C:/old-runtime'))


if __name__ == '__main__':
    unittest.main()

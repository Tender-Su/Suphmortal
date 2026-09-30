import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.eval.oracle_experiments as oracle_experiments


def make_config():
    return {
        'control': {
            'state_file': './checkpoints/mortal.pth',
            'best_state_file': './checkpoints/best.pth',
            'tensorboard_dir': './tb_log',
        },
        'test_play': {
            'log_dir': './logs/test_play',
        },
        'train_play': {
            'default': {
                'log_dir': './logs/train_play',
            },
        },
        'online': {
            'server': {
                'buffer_dir': './buffer',
                'drain_dir': './drain',
            },
        },
        '1v3': {
            'log_dir': './logs/1v3',
        },
        'value': {
            'enabled': True,
            'oracle_critic': True,
        },
        'oracle_guiding': {
            'actor_enabled': True,
            'actor_source': 'true',
        },
        'oracle_experiments': {
            'default_arm': 'current_config',
            'suffix_artifacts': True,
        },
        'oracle_dependency_eval': {
            'log_dir': './logs/oracle_dependency',
        },
    }


class OracleInputModeTests(unittest.TestCase):
    def test_zero_mode_zeros_tensor(self):
        tensor = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
        out = oracle_experiments.apply_oracle_input_mode(tensor, 'zero')
        self.assertTrue(torch.equal(out, torch.zeros_like(tensor)))

    def test_shuffled_mode_rolls_batch(self):
        tensor = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        out = oracle_experiments.apply_oracle_input_mode(tensor, 'shuffled')
        self.assertTrue(torch.equal(out[0], tensor[1]))
        self.assertTrue(torch.equal(out[1], tensor[0]))

    def test_shuffled_mode_rolls_channels_for_singleton_batch(self):
        tensor = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
        out = oracle_experiments.apply_oracle_input_mode(tensor, 'shuffled')
        self.assertTrue(torch.equal(out[:, 0], tensor[:, -1]))


class OracleExperimentArmTests(unittest.TestCase):
    def test_env_override_selects_builtin_arm(self):
        cfg = make_config()
        with mock.patch.dict(os.environ, {'MORTAL_ORACLE_ARM': 'critic_only'}, clear=False):
            arm = oracle_experiments.resolve_oracle_experiment_arm(cfg)
        self.assertEqual('critic_only', arm.name)
        self.assertFalse(arm.actor_oracle_enabled)
        self.assertTrue(arm.oracle_critic_enabled)

    def test_apply_experiment_to_config_suffixes_runtime_paths(self):
        cfg = make_config()
        with mock.patch.dict(
            os.environ,
            {
                'MORTAL_ORACLE_ARM': 'actor_shuffled',
                'MORTAL_ORACLE_ARTIFACT_SUFFIX': 'actor_shuffled',
            },
            clear=False,
        ):
            arm, suffix = oracle_experiments.apply_oracle_experiment_to_config(cfg)
        self.assertEqual('actor_shuffled', arm.name)
        self.assertEqual('shuffled', cfg['oracle_guiding']['actor_source'])
        self.assertTrue(cfg['value']['oracle_critic'])
        self.assertTrue(cfg['control']['state_file'].endswith('mortal_actor_shuffled.pth'))
        self.assertTrue(cfg['test_play']['log_dir'].endswith('test_play_actor_shuffled'))
        self.assertTrue(cfg['oracle_dependency_eval']['log_dir'].endswith('oracle_dependency_actor_shuffled'))


if __name__ == '__main__':
    unittest.main()

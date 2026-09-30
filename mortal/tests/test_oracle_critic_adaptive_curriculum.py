import unittest

import torch

from mortal.core.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    initial_adaptive_curriculum_state,
)
from mortal.research.oracle_critic_curriculum import (
    checkpoint_training_state_hashes,
    migrate_checkpoint_for_phase,
)
from scripts.run_oracle_critic_adaptive_curriculum import (
    FINAL_LR_LEVELS,
    GATE_EVERY_STEPS,
    PHASES,
    REQUIRED_FUTILE_GATES,
    adaptive_mapping,
)


def fake_checkpoint():
    return {
        'oracle_brain': {'weight': torch.arange(4, dtype=torch.float32)},
        'value_net': {'weight': torch.arange(3, dtype=torch.float32)},
        'optimizer': {'state': {}, 'param_groups': [{'lr': 2e-4}]},
        'scheduler': {'last_epoch': 100, 'tail_lr': 2e-4},
        'scaler': {'scale': 2048.0},
        'steps': 100,
        'training_contract': {
            'optimizer': {'type': 'schedule_free_adamw'},
            'adaptive_curriculum': {'phase_name': 'phase_a'},
        },
        'convergence_state': None,
        'adaptive_curriculum_state': {'phase_name': 'phase_a'},
        'resume_supported': True,
        'file_splits': {
            'seed': 7,
            'train': {'files': 2, 'sha256': 'old'},
            'dev': {'files': 1, 'sha256': 'dev'},
            'test': {'files': 1, 'sha256': 'test'},
        },
        'data_progress': {
            'signature': {'batch_size': 640},
            'cycle': 4,
            'resume_cursors': {0: [10, 20]},
            'samples_consumed': 64_000,
            'batches_consumed': 100,
        },
        'config': {'oracle_critic_pretrain': {'run_name': 'source'}},
        'oracle_critic_pretrain': {'run_name': 'source'},
        'init_info': {},
    }


class OracleCriticAdaptiveCurriculumTests(unittest.TestCase):
    def test_nonfinal_phases_have_no_fixed_lengths_or_lr_levels(self):
        self.assertEqual(('phase_a', 'phase_b', 'phase_c'), PHASES)
        for phase in PHASES[:-1]:
            config = AdaptiveCurriculumConfig.from_mapping(adaptive_mapping(phase))
            self.assertFalse(config.final_phase)
            self.assertEqual((), config.lr_levels)
            self.assertEqual(GATE_EVERY_STEPS, config.gate_every_steps)
            self.assertEqual(
                REQUIRED_FUTILE_GATES,
                config.required_futile_gates,
            )

    def test_final_phase_owns_dynamic_lr_levels(self):
        config = AdaptiveCurriculumConfig.from_mapping(
            adaptive_mapping(PHASES[-1])
        )

        self.assertTrue(config.final_phase)
        self.assertEqual(FINAL_LR_LEVELS, config.lr_levels)

    def test_adaptive_phase_migration_replaces_only_audited_metadata(self):
        state = fake_checkpoint()
        before = checkpoint_training_state_hashes(state)
        destination_splits = {
            'seed': 7,
            'train': {'files': 3, 'sha256': 'new'},
            'dev': state['file_splits']['dev'],
            'test': state['file_splits']['test'],
        }
        config = AdaptiveCurriculumConfig.from_mapping(adaptive_mapping('phase_b'))
        destination_contract = {
            **state['training_contract'],
            'adaptive_curriculum': {'phase_name': 'phase_b'},
        }
        destination_state = initial_adaptive_curriculum_state(config)

        migrated = migrate_checkpoint_for_phase(
            state,
            destination_config={
                'oracle_critic_pretrain': {'run_name': 'phase_b'}
            },
            destination_file_splits=destination_splits,
            source_phase='phase_a',
            destination_phase='phase_b',
            source_checkpoint='adaptive_best.pth',
            source_checkpoint_sha256='abc',
            destination_training_contract=destination_contract,
            destination_adaptive_curriculum_state=destination_state,
        )

        self.assertEqual(before, checkpoint_training_state_hashes(migrated))
        self.assertEqual(destination_contract, migrated['training_contract'])
        self.assertEqual('phase_b', migrated['adaptive_curriculum_state']['phase_name'])
        self.assertEqual({}, migrated['data_progress']['resume_cursors'])
        record = migrated['curriculum_provenance'][-1]
        self.assertEqual(
            'replace_for_audited_adaptive_phase',
            record['training_contract_action'],
        )


if __name__ == '__main__':
    unittest.main()

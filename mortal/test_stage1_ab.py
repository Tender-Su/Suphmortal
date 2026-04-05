import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage1_ab as stage1_ab


def make_base_cfg() -> dict:
    return stage1_ab.ensure_stage1_section(
        {
            'supervised': {
                'best_loss_state_file': './checkpoints/stage0_5_supervised.pth',
                'rank_aux': {
                    'base_weight': 0.001548,
                    'max_weight': 0.00516,
                },
            },
            'aux': {
                'next_rank_weight': 0.2,
                'opponent_state_weight': 0.00135,
                'danger_enabled': True,
                'danger_weight': 0.00804,
            },
            'optim': {
                'scheduler': {
                    'peak': 3e-4,
                    'final': 1e-5,
                },
            },
            'stage1': {
                'lr': 3e-4,
                'max_steps': 12000,
                'scheduler': {
                    'type': 'cosine',
                    'warm_up_steps': 1000,
                    'init': 1e-8,
                    'final': 1e-5,
                    'max_steps': 12000,
                },
            },
        }
    )


class Stage1AbSummaryTests(unittest.TestCase):
    def test_resolve_stage1_splits_excludes_eval_files_from_reused_train_split(self):
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }
        base_cfg = {'stage1': {'max_train_files': 0}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_index_path = Path(tmp_dir) / 'file_index_supervised_json.pth'
            torch_payload = {
                'train_files': [
                    'train_a.json.gz',
                    'monitor.json.gz',
                    'recent.json.gz',
                    'old.json.gz',
                    'train_b.json.gz',
                ]
            }
            import torch

            torch.save(torch_payload, base_index_path)

            with (
                patch.object(stage1_ab.s05, 'BASE_INDEX_PATH', base_index_path),
                patch.object(stage1_ab.s05, 'load_all_files', return_value=['placeholder.json.gz']),
                patch.object(stage1_ab.s05, 'group_files_by_month', return_value={}),
                patch.object(stage1_ab.s05, 'build_eval_splits', return_value=eval_splits),
            ):
                train_files, actual_eval_splits, grouped = stage1_ab.resolve_stage1_splits(base_cfg, seed=17)

        self.assertEqual(['train_a.json.gz', 'train_b.json.gz'], train_files)
        self.assertEqual(eval_splits, actual_eval_splits)
        self.assertEqual({}, grouped)

    def test_ensure_stage1_section_reuses_oracle_init_state_file_when_stage1_missing(self):
        cfg = {
            'supervised': {
                'state_file': './checkpoints/stage0_5_latest.pth',
                'best_state_file': './checkpoints/stage0_5_supervised.pth',
            },
            'oracle': {
                'init_state_file': './checkpoints/stage0_5_supervised.pth',
            },
        }

        stage1_cfg = stage1_ab.ensure_stage1_section(cfg)['stage1']

        self.assertEqual('./checkpoints/stage0_5_supervised.pth', stage1_cfg['init_state_file'])
        self.assertEqual('linear', stage1_cfg['oracle_dropout']['schedule'])
        self.assertGreater(stage1_cfg['oracle_dropout']['decay_steps'], 0)

    def test_ensure_stage1_section_uses_supervised_seed_when_stage1_init_missing(self):
        cfg = {
            'supervised': {
                'best_state_file': './checkpoints/stage0_5_supervised_best_state.pth',
                'best_loss_state_file': './checkpoints/stage0_5_supervised_best_loss.pth',
            },
            'stage1': {
                'best_state_file': './checkpoints/stage1_best_loss_oracle.pth',
                'best_loss_state_file': './checkpoints/stage1_best_loss_oracle.pth',
            },
        }

        stage1_cfg = stage1_ab.ensure_stage1_section(cfg)['stage1']

        self.assertEqual('./checkpoints/stage0_5_supervised_best_loss.pth', stage1_cfg['init_state_file'])

    def test_stage1_recipe_problems_requires_full_aux_recipe(self):
        base_cfg = make_base_cfg()
        self.assertEqual([], stage1_ab.stage1_recipe_problems(base_cfg))

        broken_cfg = make_base_cfg()
        broken_cfg['stage1']['aux']['opponent_state_weight'] = 0.0

        self.assertIn('opponent_state_weight <= 0', stage1_ab.stage1_recipe_problems(broken_cfg))

    def test_stage1_arm_exp_dir_supports_profile_phase_layout(self):
        exp_dir = stage1_ab.stage1_arm_exp_dir(
            ab_name='stage1_profile_test',
            profile_arm_name='linear_075',
            phase_name='oracle_transition',
        )

        self.assertEqual(
            stage1_ab.STAGE1_AB_ROOT / 'stage1_profile_test' / 'linear_075' / 'oracle_transition',
            exp_dir,
        )

    def test_build_phase_overrides_switches_to_normal_continuation(self):
        base_cfg = make_base_cfg()
        ckpts = {
            'state_file': Path('latest.pth'),
            'best_state_file': Path('best.pth'),
            'best_loss_state_file': Path('best_loss.pth'),
            'best_acc_state_file': Path('best_acc.pth'),
            'best_rank_state_file': Path('best_rank.pth'),
            'best_loss_normal_state_file': Path('best_loss_normal.pth'),
            'best_acc_normal_state_file': Path('best_acc_normal.pth'),
            'best_rank_normal_state_file': Path('best_rank_normal.pth'),
            'tensorboard_dir': Path('tb'),
            'file_index': Path('file_index.pth'),
        }

        overrides = stage1_ab.build_phase_overrides(
            base_cfg,
            ckpts=ckpts,
            seed=17,
            max_steps=3000,
            init_state_file='init_state.pth',
            enable_oracle=False,
            oracle_dropout=stage1_ab.disabled_oracle_dropout(),
            lr_scale=0.1,
            warm_up_steps_override=0,
        )
        stage1_cfg = overrides['stage1']

        self.assertFalse(stage1_cfg['enable_oracle'])
        self.assertAlmostEqual(3e-5, stage1_cfg['lr'])
        self.assertEqual(0, stage1_cfg['scheduler']['warm_up_steps'])
        self.assertAlmostEqual(1e-6, stage1_cfg['scheduler']['final'])
        self.assertGreater(stage1_cfg['aux']['opponent_state_weight'], 0.0)
        self.assertTrue(stage1_cfg['aux']['danger_enabled'])
        self.assertGreater(stage1_cfg['aux']['danger_weight'], 0.0)
        self.assertFalse(stage1_cfg['publish_stage2_handoff'])

    def test_run_profile_ab_uses_selected_arm_metrics_not_selection_metadata(self):
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }
        selection = {'eligible': ['cosine_075']}

        def fake_run_profile_arm(*args, profile, **kwargs):
            return {
                'arm_name': profile.arm_name,
                'description': profile.description,
                'final': {
                    'best_loss': {'marker': profile.arm_name},
                },
            }

        with (
            patch.object(stage1_ab, 'resolve_stage1_splits', return_value=(['train.json.gz'], eval_splits, {})),
            patch.object(stage1_ab, 'run_profile_arm', side_effect=fake_run_profile_arm),
            patch.object(stage1_ab, 'rank_results', return_value=[]),
            patch.object(stage1_ab, 'save_results', return_value=Path('X:/virtual/stage1_profile_summary.json')),
            patch.object(stage1_ab.s05, 'select_winner_by_policy', return_value=('cosine_075', selection)),
        ):
            payload = stage1_ab.run_profile_ab(
                make_base_cfg(),
                ab_name='stage1_profile_test',
                seed=33,
                step_scale=1.0,
                profile_arm_names=['linear_075', 'cosine_075'],
                init_state_file='init_state.pth',
            )

        self.assertEqual('cosine_075', payload['winner'])
        self.assertEqual({'marker': 'cosine_075'}, payload['winner_metrics'])
        self.assertEqual(selection, payload['selection'])

    def test_run_profile_arm_continuation_uses_transition_terminal_state_file(self):
        base_cfg = make_base_cfg()
        profile = stage1_ab.PROFILE_ARMS['linear_075']
        train_files = ['train.json.gz']
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            transition_latest = tmp_path / 'transition_latest.pth'
            transition_latest.write_text('latest', encoding='utf-8')
            best_loss_normal = tmp_path / 'transition_best_loss_normal.pth'
            best_loss_normal.write_text('best-loss', encoding='utf-8')

            transition_result = {
                'final': {'best_loss': {'loss': 1.0}},
                'paths': {
                    'state_file': str(transition_latest),
                    'best_loss_normal_state_file': str(best_loss_normal),
                },
            }
            continuation_result = {
                'final': {'best_loss': {'loss': 0.9}},
                'paths': {
                    'state_file': str(tmp_path / 'continuation_latest.pth'),
                },
            }
            init_state_files: list[str] = []

            def fake_run_training_phase(*args, **kwargs):
                init_state_files.append(kwargs['cfg_overrides']['stage1']['init_state_file'])
                if kwargs['phase_name'] == 'oracle_transition':
                    return transition_result
                if kwargs['phase_name'] == 'normal_continue':
                    return continuation_result
                raise AssertionError(f'unexpected phase: {kwargs["phase_name"]}')

            with patch.object(stage1_ab, 'run_training_phase', side_effect=fake_run_training_phase):
                result = stage1_ab.run_profile_arm(
                    base_cfg,
                    ab_name='stage1_profile_test',
                    profile=profile,
                    train_files=train_files,
                    eval_splits=eval_splits,
                    init_state_file='initial_seed.pth',
                    seed=17,
                    step_scale=1.0,
                )

        self.assertEqual(['initial_seed.pth', str(transition_latest)], init_state_files)
        self.assertEqual(continuation_result['final'], result['final'])

    def test_main_blocks_pending_formal_1v3_before_stage1_launch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_state_file = Path(tmp_dir) / 'stage0_5_supervised.pth'
            init_state_file.write_text('checkpoint', encoding='utf-8')
            base_cfg = make_base_cfg()
            base_cfg['stage1']['init_state_file'] = str(init_state_file)

            with (
                patch.object(sys, 'argv', ['run_stage1_ab.py']),
                patch.object(stage1_ab.s05, 'build_base_config', return_value=base_cfg),
                patch.object(stage1_ab, 'ensure_stage1_section', return_value=base_cfg),
                patch.object(
                    stage1_ab.stage05_formal,
                    'ensure_stage1_canonical_handoff_ready',
                    side_effect=RuntimeError('pending formal_1v3'),
                ),
                patch.object(stage1_ab, 'run_profile_ab') as run_profile_ab,
            ):
                with self.assertRaisesRegex(RuntimeError, 'pending formal_1v3'):
                    stage1_ab.main()

        run_profile_ab.assert_not_called()


if __name__ == '__main__':
    unittest.main()

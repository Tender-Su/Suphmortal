import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage1_ab as stage1_ab


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

    def test_main_blocks_pending_formal_1v3_before_stage1_launch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_state_file = Path(tmp_dir) / 'stage0_5_supervised.pth'
            init_state_file.write_text('checkpoint', encoding='utf-8')
            base_cfg = {'stage1': {'init_state_file': str(init_state_file)}}

            with (
                patch.object(sys, 'argv', ['run_stage1_ab.py']),
                patch.object(stage1_ab.s05, 'build_base_config', return_value=base_cfg),
                patch.object(stage1_ab, 'ensure_stage1_section', return_value=base_cfg),
                patch.object(
                    stage1_ab.stage05_formal,
                    'ensure_stage1_canonical_handoff_ready',
                    side_effect=RuntimeError('pending formal_1v3'),
                ),
                patch.object(stage1_ab, 'run_block_c') as run_block_c,
            ):
                with self.assertRaisesRegex(RuntimeError, 'pending formal_1v3'):
                    stage1_ab.main()

        run_block_c.assert_not_called()

    def test_stage1_arm_exp_dir_keeps_recipe_mode_layout_stable(self):
        exp_dir = stage1_ab.stage1_arm_exp_dir(
            ab_name='stage1_recipe_test',
            recipe_arm_name='S1-B',
            gamma_arm='G1',
            isolate_gamma_artifacts=False,
        )

        self.assertEqual(
            stage1_ab.STAGE1_AB_ROOT / 'stage1_recipe_test' / 'S1-B',
            exp_dir,
        )

    def test_stage1_arm_exp_dir_isolates_gamma_artifacts(self):
        g1_dir = stage1_ab.stage1_arm_exp_dir(
            ab_name='stage1_gamma_test',
            recipe_arm_name='S1-D',
            gamma_arm='G1',
            isolate_gamma_artifacts=True,
        )
        g2_dir = stage1_ab.stage1_arm_exp_dir(
            ab_name='stage1_gamma_test',
            recipe_arm_name='S1-D',
            gamma_arm='G2',
            isolate_gamma_artifacts=True,
        )

        self.assertEqual(
            stage1_ab.STAGE1_AB_ROOT / 'stage1_gamma_test' / 'S1-D' / 'G1',
            g1_dir,
        )
        self.assertEqual(
            stage1_ab.STAGE1_AB_ROOT / 'stage1_gamma_test' / 'S1-D' / 'G2',
            g2_dir,
        )
        self.assertNotEqual(g1_dir, g2_dir)

    def test_build_recipe_overrides_force_enable_aux_heads_for_enabled_recipe(self):
        base_cfg = stage1_ab.ensure_stage1_section({
            'supervised': {
                'rank_aux': {
                    'base_weight': 0.03,
                },
            },
            'aux': {
                'next_rank_weight': 0.2,
                'opponent_state_weight': 0.0,
                'danger_enabled': False,
                'danger_weight': 0.0,
            },
        })
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

        overrides = stage1_ab.build_recipe_overrides(
            base_cfg,
            stage1_ab.RECIPE_ARMS['S1-D'],
            gamma_arm='G1',
            max_steps=12000,
            init_state_file='init_state.pth',
            ckpts=ckpts,
            seed=17,
        )
        aux_cfg = overrides['stage1']['aux']

        self.assertGreater(aux_cfg['opponent_state_weight'], 0.0)
        self.assertTrue(aux_cfg['danger_enabled'])
        self.assertGreater(aux_cfg['danger_weight'], 0.0)
        self.assertFalse(overrides['stage1']['publish_stage2_handoff'])
        self.assertIsNone(stage1_ab.ineffective_reason(base_cfg, stage1_ab.RECIPE_ARMS['S1-C']))
        self.assertIsNone(stage1_ab.ineffective_reason(base_cfg, stage1_ab.RECIPE_ARMS['S1-D']))

    def test_run_recipe_ab_uses_selected_arm_metrics_not_selection_metadata(self):
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }
        selection = {'eligible': ['S1-C']}

        def fake_run_recipe_arm(*args, recipe_arm, **kwargs):
            return {
                'arm_name': recipe_arm.arm_name,
                'final': {
                    'best_loss': {'marker': recipe_arm.arm_name},
                },
            }

        with (
            patch.object(stage1_ab, 'resolve_stage1_splits', return_value=(['train.json.gz'], eval_splits, {})),
            patch.object(stage1_ab, 'run_recipe_arm', side_effect=fake_run_recipe_arm),
            patch.object(stage1_ab, 'rank_results', return_value=[]),
            patch.object(stage1_ab, 'collect_effectiveness_warnings', return_value=[]),
            patch.object(stage1_ab, 'save_results', return_value=Path('X:/virtual/stage1_recipe_summary.json')),
            patch.object(stage1_ab.s05, 'select_winner_by_policy', return_value=('S1-C', selection)),
        ):
            payload = stage1_ab.run_recipe_ab(
                {},
                ab_name='stage1_recipe_test',
                seed=17,
                step_scale=1.0,
                gamma_arm='G1',
                init_state_file='init_state.pth',
            )

        self.assertEqual('S1-C', payload['winner'])
        self.assertEqual({'marker': 'S1-C'}, payload['winner_metrics'])
        self.assertEqual(selection, payload['selection'])

    def test_run_gamma_ab_uses_selected_arm_metrics_not_selection_metadata(self):
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }
        selection = {'eligible': ['S1-D_G2']}

        def fake_run_recipe_arm(*args, recipe_arm, gamma_arm, **kwargs):
            return {
                'arm_name': recipe_arm.arm_name,
                'final': {
                    'best_loss': {'marker': f'{recipe_arm.arm_name}_{gamma_arm}'},
                },
            }

        with (
            patch.object(stage1_ab, 'resolve_stage1_splits', return_value=(['train.json.gz'], eval_splits, {})),
            patch.object(stage1_ab, 'run_recipe_arm', side_effect=fake_run_recipe_arm),
            patch.object(stage1_ab, 'rank_results', return_value=[]),
            patch.object(stage1_ab, 'collect_effectiveness_warnings', return_value=[]),
            patch.object(stage1_ab, 'save_results', return_value=Path('X:/virtual/stage1_gamma_summary.json')),
            patch.object(stage1_ab.s05, 'select_winner_by_policy', return_value=('S1-D_G2', selection)),
        ):
            payload = stage1_ab.run_gamma_ab(
                {},
                ab_name='stage1_gamma_test',
                seed=33,
                step_scale=1.0,
                recipe_arm_name='S1-D',
                init_state_file='init_state.pth',
            )

        self.assertEqual('S1-D_G2', payload['winner'])
        self.assertEqual({'marker': 'S1-D_G2'}, payload['winner_metrics'])
        self.assertEqual(selection, payload['selection'])

    def test_run_gamma_ab_requests_isolated_gamma_artifacts(self):
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }
        isolate_flags = []

        def fake_run_recipe_arm(*args, isolate_gamma_artifacts, **kwargs):
            isolate_flags.append(isolate_gamma_artifacts)
            return {
                'arm_name': 'placeholder',
                'final': {
                    'best_loss': {'marker': 'placeholder'},
                },
            }

        with (
            patch.object(stage1_ab, 'resolve_stage1_splits', return_value=(['train.json.gz'], eval_splits, {})),
            patch.object(stage1_ab, 'run_recipe_arm', side_effect=fake_run_recipe_arm),
            patch.object(stage1_ab, 'rank_results', return_value=[]),
            patch.object(stage1_ab, 'collect_effectiveness_warnings', return_value=[]),
            patch.object(stage1_ab, 'save_results', return_value=Path('X:/virtual/stage1_gamma_summary.json')),
            patch.object(stage1_ab.s05, 'select_winner_by_policy', return_value=('S1-B_G0', {'eligible': ['S1-B_G0']})),
        ):
            stage1_ab.run_gamma_ab(
                {},
                ab_name='stage1_gamma_test',
                seed=33,
                step_scale=1.0,
                recipe_arm_name='S1-B',
                init_state_file='init_state.pth',
            )

        self.assertEqual([True, True, True], isolate_flags)


if __name__ == '__main__':
    unittest.main()

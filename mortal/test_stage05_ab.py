import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_ab as stage05_ab


class Stage05ABTests(unittest.TestCase):
    def test_phase_a_training_pool_excludes_old_regression_months(self):
        grouped = {
            '200912': ['early.json.gz'],
            '202112': ['mid_keep.json.gz'],
            '202201': ['old_reg_a.json.gz'],
            '202212': ['old_reg_b.json.gz'],
            '202401': ['recent.json.gz'],
        }

        train_files = stage05_ab.phase_train_files(
            grouped,
            'phase_a',
            weight_profile='mild',
            window_profile='24m_12m',
            pool_size=0,
            seed=123,
        )

        self.assertIn('mid_keep.json.gz', train_files)
        self.assertNotIn('old_reg_a.json.gz', train_files)
        self.assertNotIn('old_reg_b.json.gz', train_files)

    def test_phase_b_replay_pool_excludes_old_regression_months(self):
        grouped = {
            '200912': ['early.json.gz'],
            '202112': ['mid_keep.json.gz'],
            '202201': ['old_reg_a.json.gz'],
            '202212': ['old_reg_b.json.gz'],
            '202401': ['recent.json.gz'],
        }

        unit_profile = {
            'phase_b': ([1.0], ['replay']),
        }
        with patch.dict(stage05_ab.WEIGHT_PROFILES, {'unit_test': unit_profile}, clear=False):
            train_files = stage05_ab.phase_train_files(
                grouped,
                'phase_b',
                weight_profile='unit_test',
                window_profile='24m_12m',
                pool_size=0,
                seed=456,
            )

        self.assertIn('early.json.gz', train_files)
        self.assertIn('mid_keep.json.gz', train_files)
        self.assertNotIn('old_reg_a.json.gz', train_files)
        self.assertNotIn('old_reg_b.json.gz', train_files)

    def test_transient_training_failure_marker_requires_explicit_resource_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / 'train.log'

            log_path.write_text(
                'RuntimeError: DataLoader worker (pid(s) 1234) exited unexpectedly\n',
                encoding='utf-8',
                newline='\n',
            )
            self.assertIsNone(stage05_ab.transient_training_failure_marker(log_path))

            log_path.write_text(
                '\n'.join([
                    'RuntimeError: DataLoader worker (pid(s) 1234) exited unexpectedly',
                    'OSError: WinError 1455: paging file is too small',
                ]),
                encoding='utf-8',
                newline='\n',
            )
            self.assertEqual('WinError 1455', stage05_ab.transient_training_failure_marker(log_path))

    def test_transient_training_failure_marker_ignores_previous_attempt_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / 'train.log'
            first_attempt = '\n'.join([
                '=== train_supervised attempt 1 ===',
                'OSError: WinError 1455: paging file is too small',
            ]) + '\n'
            log_path.write_text(first_attempt, encoding='utf-8', newline='\n')
            start_offset = log_path.stat().st_size
            with log_path.open('a', encoding='utf-8', newline='\n') as f:
                f.write('=== train_supervised attempt 2 ===\n')
                f.write('ValueError: permanent config failure\n')

            self.assertIsNone(
                stage05_ab.transient_training_failure_marker(log_path, start_offset=start_offset)
            )

    def test_run_training_stops_retrying_when_only_previous_attempt_had_transient_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / 'config.toml'
            cfg_path.write_text('', encoding='utf-8', newline='\n')
            log_path = Path(tmp_dir) / 'train.log'
            attempts = {'count': 0}

            def fake_run(*args, **kwargs):
                attempts['count'] += 1
                stdout = kwargs['stdout']
                if attempts['count'] == 1:
                    stdout.write('OSError: WinError 1455: paging file is too small\n')
                elif attempts['count'] == 2:
                    stdout.write('ValueError: permanent config failure\n')
                else:
                    raise AssertionError('run_training retried after a non-transient failure')
                stdout.flush()
                return SimpleNamespace(returncode=1)

            with (
                patch.object(stage05_ab.subprocess, 'run', side_effect=fake_run),
                patch.object(stage05_ab.time, 'sleep', return_value=None),
            ):
                with self.assertRaisesRegex(RuntimeError, 'train_supervised.py failed'):
                    stage05_ab.run_training(cfg_path, log_path)

            self.assertEqual(2, attempts['count'])

    def test_load_state_summary_with_fallback_uses_latest_when_best_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            latest_path = tmp_path / 'latest.pth'
            latest_payload = {
                'steps': 375,
                'optimizer_steps': 375,
                'epoch': 1,
                'optimizer': {'param_groups': [{'lr': 3e-4}]},
                'last_full_recent_metrics': {'loss': 0.5},
            }
            import torch

            torch.save(latest_payload, latest_path)

            summary = stage05_ab.load_state_summary_with_fallback(
                tmp_path / 'best_loss.pth',
                latest_path,
            )

            self.assertEqual(str(latest_path), summary['path'])
            self.assertEqual(375, summary['optimizer_steps'])
            self.assertEqual(3e-4, summary['lr'])

    def test_checkpoint_paths_compact_mode_keeps_distinct_best_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir) / 'exp'
            storage_root = Path(tmp_dir) / 'scratch'

            ckpts = stage05_ab.checkpoint_paths(
                exp_dir,
                storage_root=storage_root,
                compact_checkpoints=True,
            )

            self.assertEqual(storage_root / 'checkpoints' / 'best_loss.pth', ckpts['best_loss_state_file'])
            self.assertEqual(storage_root / 'checkpoints' / 'best_action_score.pth', ckpts['best_acc_state_file'])
            self.assertEqual(storage_root / 'checkpoints' / 'best_rank.pth', ckpts['best_rank_state_file'])
            self.assertEqual(storage_root / 'file_index.pth', ckpts['file_index'])
            self.assertTrue((storage_root / 'checkpoints').exists())
            self.assertTrue((storage_root / 'tb').exists())

    def test_compact_checkpoint_artifacts_removes_non_loss_metric_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratch_root = Path(tmp_dir)
            ckpts = stage05_ab.checkpoint_paths(
                scratch_root / 'exp',
                storage_root=scratch_root,
                compact_checkpoints=True,
            )
            for checkpoint_path in (
                ckpts['best_loss_state_file'],
                ckpts['best_acc_state_file'],
                ckpts['best_rank_state_file'],
            ):
                checkpoint_path.write_text('placeholder', encoding='utf-8', newline='\n')

            stage05_ab.compact_checkpoint_artifacts(ckpts)

            self.assertTrue(ckpts['best_loss_state_file'].exists())
            self.assertFalse(ckpts['best_acc_state_file'].exists())
            self.assertFalse(ckpts['best_rank_state_file'].exists())

    def test_cleanup_phase_artifacts_only_removes_scratch_roots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_scratch_root = stage05_ab.AB_SCRATCH_ROOT
            try:
                stage05_ab.AB_SCRATCH_ROOT = Path(tmp_dir) / 'scratch_base'
                scratch_phase = stage05_ab.AB_SCRATCH_ROOT / 'token' / 'phase_a'
                scratch_phase.mkdir(parents=True, exist_ok=True)
                (scratch_phase / 'marker.txt').write_text('scratch', encoding='utf-8', newline='\n')
                scratch_result = {
                    'artifact_root': str(scratch_phase),
                    'artifacts_retained': True,
                }

                persistent_root = Path(tmp_dir) / 'logs' / 'phase_c'
                persistent_root.mkdir(parents=True, exist_ok=True)
                (persistent_root / 'marker.txt').write_text('persistent', encoding='utf-8', newline='\n')
                persistent_result = {
                    'artifact_root': str(persistent_root),
                    'artifacts_retained': True,
                }

                stage05_ab.cleanup_phase_artifacts(scratch_result)
                stage05_ab.cleanup_phase_artifacts(persistent_result)

                self.assertFalse(scratch_phase.exists())
                self.assertFalse(scratch_result['artifacts_retained'])
                self.assertEqual(str(scratch_phase.resolve()), scratch_result['cleaned_artifact_root'])
                self.assertTrue(persistent_root.exists())
                self.assertTrue(persistent_result['artifacts_retained'])
                self.assertNotIn('cleaned_artifact_root', persistent_result)
            finally:
                stage05_ab.AB_SCRATCH_ROOT = original_scratch_root

    def test_run_arm_cleans_previous_scratch_phase_after_successful_handoff(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            original_scratch_root = stage05_ab.AB_SCRATCH_ROOT
            stage05_ab.AB_SCRATCH_ROOT = tmp_path / 'scratch_base'
            phase_roots: dict[str, Path] = {}

            def fake_run_phase(
                base_cfg,
                grouped,
                *,
                ab_name,
                arm_name,
                phase_name,
                scheduler_type,
                weight_profile,
                window_profile,
                seed,
                eval_splits,
                init_state_file,
                step_scale,
                storage_root=None,
                compact_checkpoints=False,
            ):
                root = storage_root or (tmp_path / 'persistent' / phase_name)
                root.mkdir(parents=True, exist_ok=True)
                best_loss = root / 'checkpoints' / 'best_loss.pth'
                best_loss.parent.mkdir(parents=True, exist_ok=True)
                best_loss.write_text(phase_name, encoding='utf-8', newline='\n')
                phase_roots[phase_name] = root
                if phase_name == 'phase_a':
                    self.assertIsNone(init_state_file)
                elif phase_name == 'phase_b':
                    self.assertEqual(str(phase_roots['phase_a'] / 'checkpoints' / 'best_loss.pth'), init_state_file)
                elif phase_name == 'phase_c':
                    self.assertEqual(str(phase_roots['phase_b'] / 'checkpoints' / 'best_loss.pth'), init_state_file)
                return {
                    'latest': {'phase': phase_name},
                    'best_loss': {'phase': phase_name, 'last_full_recent_metrics': {'loss': 0.1, 'action_quality_score': 0.5, 'rank_acc': 0.2}},
                    'best_acc': {'phase': phase_name},
                    'best_rank': {'phase': phase_name},
                    'artifact_root': str(root.resolve()),
                    'artifacts_retained': True,
                    'paths': {
                        'best_loss_state_file': str(best_loss),
                    },
                    'log_path': str(root / 'train.log'),
                    'config_path': str(root / 'config.toml'),
                }

            try:
                with patch.object(stage05_ab, 'run_phase', side_effect=fake_run_phase):
                    result = stage05_ab.run_arm(
                        base_cfg={},
                        grouped={},
                        ab_name='unit_ab',
                        arm_name='unit_arm',
                        scheduler_profile='cosine',
                        curriculum_profile='broad_to_recent',
                        weight_profile='strong',
                        window_profile='24m_12m',
                        seed=123,
                        eval_splits={
                            'monitor_recent_files': [],
                            'full_recent_files': [],
                            'old_regression_files': [],
                        },
                        step_scale=1.0,
                    )

                self.assertFalse(phase_roots['phase_a'].exists())
                self.assertFalse(phase_roots['phase_b'].exists())
                self.assertTrue(phase_roots['phase_c'].exists())
                self.assertFalse(result['phase_results']['phase_a']['artifacts_retained'])
                self.assertFalse(result['phase_results']['phase_b']['artifacts_retained'])
                self.assertTrue(result['phase_results']['phase_c']['artifacts_retained'])
                self.assertEqual(str(phase_roots['phase_a'].resolve()), result['phase_results']['phase_a']['cleaned_artifact_root'])
                self.assertEqual(str(phase_roots['phase_b'].resolve()), result['phase_results']['phase_b']['cleaned_artifact_root'])
            finally:
                stage05_ab.AB_SCRATCH_ROOT = original_scratch_root

    def test_scratch_phase_root_is_stable_for_same_inputs(self):
        phase_a = stage05_ab.scratch_phase_root(
            ab_name='demo_ab',
            arm_name='demo_arm',
            phase_name='phase_a',
            scratch_token='token123',
        )
        phase_a_repeat = stage05_ab.scratch_phase_root(
            ab_name='demo_ab',
            arm_name='demo_arm',
            phase_name='phase_a',
            scratch_token='token123',
        )
        phase_b = stage05_ab.scratch_phase_root(
            ab_name='demo_ab',
            arm_name='demo_arm',
            phase_name='phase_b',
            scratch_token='token123',
        )

        self.assertEqual(phase_a, phase_a_repeat)
        self.assertNotEqual(phase_a, phase_b)
        self.assertIn('mahjongai_stage05_ab', str(phase_a))


if __name__ == '__main__':
    unittest.main()

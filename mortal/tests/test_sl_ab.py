import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.supervised.run_sl_ab as sl_ab


class Stage05ABTests(unittest.TestCase):
    def test_phase_a_training_pool_excludes_old_regression_months(self):
        grouped = {
            '200912': ['early.json.gz'],
            '202112': ['mid_keep.json.gz'],
            '202201': ['old_reg_a.json.gz'],
            '202212': ['old_reg_b.json.gz'],
            '202401': ['recent.json.gz'],
        }

        train_files = sl_ab.phase_train_files(
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
        with patch.dict(sl_ab.WEIGHT_PROFILES, {'unit_test': unit_profile}, clear=False):
            train_files = sl_ab.phase_train_files(
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
            self.assertIsNone(sl_ab.transient_training_failure_marker(log_path))

            log_path.write_text(
                '\n'.join([
                    'RuntimeError: DataLoader worker (pid(s) 1234) exited unexpectedly',
                    'OSError: WinError 1455: paging file is too small',
                ]),
                encoding='utf-8',
                newline='\n',
            )
            self.assertEqual('WinError 1455', sl_ab.transient_training_failure_marker(log_path))

    def test_transient_training_failure_marker_handles_pin_memory_shared_event(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / 'train.log'
            log_path.write_text(
                '\n'.join([
                    'RuntimeError: Couldn\'t open shared event: <torch_123_event>, error code: <2>',
                    'RuntimeError: Pin memory thread exited unexpectedly',
                ]),
                encoding='utf-8',
                newline='\n',
            )

            self.assertEqual(
                "Couldn't open shared event",
                sl_ab.transient_training_failure_marker(log_path),
            )

    def test_transient_training_failure_marker_handles_cublas_internal_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / 'train.log'
            log_path.write_text(
                '\n'.join([
                    'RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasLtMatmul',
                    'RuntimeError: matmul failed during warmup',
                ]),
                encoding='utf-8',
                newline='\n',
            )

            self.assertEqual(
                'CUBLAS_STATUS_INTERNAL_ERROR',
                sl_ab.transient_training_failure_marker(log_path),
            )

    def test_transient_training_failure_marker_handles_cudnn_host_allocation_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / 'train.log'
            log_path.write_text(
                '\n'.join([
                    'RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED',
                    'Unhandled exception caught in c10/util/AbortHandler.h',
                ]),
                encoding='utf-8',
                newline='\n',
            )

            self.assertEqual(
                'CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED',
                sl_ab.transient_training_failure_marker(log_path),
            )

    def test_transient_training_failure_marker_handles_cudnn_execution_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / 'train.log'
            log_path.write_text(
                '\n'.join([
                    'RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED',
                    'Unhandled exception caught in c10/util/AbortHandler.h',
                ]),
                encoding='utf-8',
                newline='\n',
            )

            self.assertEqual(
                'CUDNN_STATUS_EXECUTION_FAILED',
                sl_ab.transient_training_failure_marker(log_path),
            )

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
                sl_ab.transient_training_failure_marker(log_path, start_offset=start_offset)
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
                patch.object(sl_ab.subprocess, 'run', side_effect=fake_run),
                patch.object(sl_ab.time, 'sleep', return_value=None),
            ):
                with self.assertRaisesRegex(RuntimeError, 'train_supervised.py failed'):
                    sl_ab.run_training(cfg_path, log_path)

            self.assertEqual(2, attempts['count'])

    def test_run_training_invokes_train_supervised_script_directly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / 'config.toml'
            cfg_path.write_text('', encoding='utf-8', newline='\n')
            log_path = Path(tmp_dir) / 'train.log'

            with patch.object(
                sl_ab.subprocess,
                'run',
                return_value=SimpleNamespace(returncode=0),
            ) as run_mock:
                sl_ab.run_training(cfg_path, log_path)

            run_args = run_mock.call_args
            self.assertEqual(
                [sl_ab.sys.executable, '-m', 'mortal.supervised.train_supervised'],
                run_args.args[0],
            )
            self.assertEqual(sl_ab.MORTAL_DIR, run_args.kwargs['cwd'])
            self.assertEqual(str(cfg_path), run_args.kwargs['env']['MORTAL_CFG'])
            self.assertFalse(run_args.kwargs['check'])

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

            summary = sl_ab.load_state_summary_with_fallback(
                tmp_path / 'best_loss.pth',
                latest_path,
            )

            self.assertEqual(str(latest_path), summary['path'])
            self.assertEqual(375, summary['optimizer_steps'])
            self.assertEqual(3e-4, summary['lr'])

    def test_checkpoint_complete_when_max_steps_reached(self):
        self.assertTrue(
            sl_ab.checkpoint_is_complete_for_config(
                {'steps': 126000, 'validation_checks': 0},
                {'max_steps': 126000, 'early_stopping_patience_checks': 8},
            )
        )

    def test_checkpoint_complete_when_early_stopping_state_reached(self):
        self.assertTrue(
            sl_ab.checkpoint_is_complete_for_config(
                {
                    'steps': 490000,
                    'validation_checks': 48,
                    'patience_counter': 8,
                    'num_lr_reductions': 0,
                },
                {
                    'max_steps': 1260000,
                    'min_validation_checks': 2,
                    'early_stopping_patience_checks': 8,
                    'early_stopping_min_lr_reductions': 0,
                },
            )
        )

    def test_checkpoint_incomplete_before_budget_or_early_stop(self):
        self.assertFalse(
            sl_ab.checkpoint_is_complete_for_config(
                {
                    'steps': 126000,
                    'validation_checks': 12,
                    'patience_counter': 1,
                    'num_lr_reductions': 0,
                },
                {
                    'max_steps': 840000,
                    'min_validation_checks': 2,
                    'early_stopping_patience_checks': 8,
                    'early_stopping_min_lr_reductions': 0,
                },
            )
        )

    def test_checkpoint_paths_keep_distinct_metric_winners_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir) / 'exp'
            ckpts = sl_ab.checkpoint_paths(exp_dir)

            self.assertEqual(exp_dir / 'checkpoints' / 'best_loss.pth', ckpts['best_loss_state_file'])
            self.assertEqual(exp_dir / 'checkpoints' / 'best_action_score.pth', ckpts['best_acc_state_file'])
            self.assertEqual(exp_dir / 'checkpoints' / 'best_rank.pth', ckpts['best_rank_state_file'])
            self.assertEqual(exp_dir / 'file_index.pth', ckpts['file_index'])
            self.assertEqual(exp_dir / 'phase_manifest.json', ckpts['manifest_file'])
            self.assertTrue((exp_dir / 'checkpoints').exists())
            self.assertTrue((exp_dir / 'tb').exists())

    def test_semantic_config_digest_ignores_runtime_and_artifact_paths(self):
        base = {
            'control': {'batch_size': 128},
            'supervised': {
                'max_steps': 100,
                'state_file': 'one/latest.pth',
                'num_workers': 2,
                'prefetch_factor': 2,
            },
        }
        changed_runtime = {
            'control': {'batch_size': 128},
            'supervised': {
                'max_steps': 100,
                'state_file': 'two/latest.pth',
                'num_workers': 8,
                'prefetch_factor': 6,
            },
        }
        changed_semantics = {
            'control': {'batch_size': 128},
            'supervised': {
                'max_steps': 200,
                'state_file': 'two/latest.pth',
                'num_workers': 8,
                'prefetch_factor': 6,
            },
        }

        self.assertEqual(
            sl_ab.semantic_config_digest(base),
            sl_ab.semantic_config_digest(changed_runtime),
        )
        self.assertNotEqual(
            sl_ab.semantic_config_digest(base),
            sl_ab.semantic_config_digest(changed_semantics),
        )

    def test_validate_existing_phase_artifacts_accepts_exact_plan(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpts = sl_ab.checkpoint_paths(Path(tmp_dir) / 'phase_a')
            plan = {'plan_id': 'plan-a', 'phase_name': 'phase_a'}
            sl_ab.atomic_write_json(
                ckpts['manifest_file'],
                {'plan': plan, 'status': 'running'},
            )
            torch.save(
                {
                    'checkpoint_id': 'checkpoint-a',
                    'run_provenance': plan,
                    'steps': 50,
                },
                ckpts['state_file'],
            )

            state = sl_ab.validate_existing_phase_artifacts(ckpts, plan)

            self.assertEqual('checkpoint-a', state['checkpoint_id'])

    def test_validate_existing_phase_artifacts_rejects_wrong_plan(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpts = sl_ab.checkpoint_paths(Path(tmp_dir) / 'phase_a')
            sl_ab.atomic_write_json(
                ckpts['manifest_file'],
                {'plan': {'plan_id': 'old-plan'}, 'status': 'completed'},
            )
            torch.save(
                {
                    'checkpoint_id': 'old-checkpoint',
                    'run_provenance': {'plan_id': 'old-plan'},
                    'steps': 100,
                },
                ckpts['state_file'],
            )

            with self.assertRaisesRegex(RuntimeError, 'plan mismatch'):
                sl_ab.validate_existing_phase_artifacts(
                    ckpts,
                    {'plan_id': 'new-plan'},
                )

    def test_validate_existing_phase_artifacts_rejects_legacy_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpts = sl_ab.checkpoint_paths(Path(tmp_dir) / 'phase_a')
            torch.save({'steps': 100}, ckpts['state_file'])

            with self.assertRaisesRegex(RuntimeError, 'predates phase provenance'):
                sl_ab.validate_existing_phase_artifacts(
                    ckpts,
                    {'plan_id': 'new-plan'},
                )

    def test_run_arm_uses_policy_selected_checkpoint_as_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
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
                allow_early_stopping=True,
            ):
                root = storage_root or (tmp_path / 'persistent' / phase_name)
                root.mkdir(parents=True, exist_ok=True)
                best_loss = root / 'checkpoints' / 'best_loss.pth'
                best_acc = root / 'checkpoints' / 'best_action_score.pth'
                best_rank = root / 'checkpoints' / 'best_rank.pth'
                best_loss.parent.mkdir(parents=True, exist_ok=True)
                best_loss.write_text(phase_name, encoding='utf-8', newline='\n')
                best_acc.write_text(phase_name, encoding='utf-8', newline='\n')
                best_rank.write_text(phase_name, encoding='utf-8', newline='\n')
                manifest = root / 'phase_manifest.json'
                manifest.write_text(
                    '{"plan": {"plan_id": "unit"}, "status": "completed"}',
                    encoding='utf-8',
                    newline='\n',
                )
                phase_roots[phase_name] = root
                if phase_name == 'phase_a':
                    self.assertIsNone(init_state_file)
                elif phase_name == 'phase_b':
                    self.assertEqual(str(phase_roots['phase_a'] / 'checkpoints' / 'best_action_score.pth'), init_state_file)
                elif phase_name == 'phase_c':
                    self.assertEqual(str(phase_roots['phase_b'] / 'checkpoints' / 'best_action_score.pth'), init_state_file)
                loss_metrics = {
                    'loss': 0.1000,
                    'action_quality_score': 0.3,
                    'selection_quality_score': 0.3,
                    'rank_acc': 0.2,
                }
                action_metrics = {
                    'loss': 0.1002,
                    'action_quality_score': 0.5,
                    'selection_quality_score': 0.5,
                    'rank_acc': 0.3,
                }
                rank_metrics = {
                    'loss': 0.2,
                    'action_quality_score': 0.6,
                    'selection_quality_score': 0.6,
                    'rank_acc': 0.8,
                }
                return {
                    'latest': {'phase': phase_name},
                    'best_loss': {
                        'phase': phase_name,
                        'checkpoint_id': f'{phase_name}-loss',
                        'path': str(best_loss),
                        'last_full_recent_metrics': loss_metrics,
                    },
                    'best_acc': {
                        'phase': phase_name,
                        'checkpoint_id': f'{phase_name}-action',
                        'path': str(best_acc),
                        'last_full_recent_metrics': action_metrics,
                    },
                    'best_rank': {
                        'phase': phase_name,
                        'checkpoint_id': f'{phase_name}-rank',
                        'path': str(best_rank),
                        'last_full_recent_metrics': rank_metrics,
                    },
                    'artifact_root': str(root.resolve()),
                    'artifacts_retained': True,
                    'paths': {
                        'best_loss_state_file': str(best_loss),
                        'manifest_file': str(manifest),
                    },
                    'log_path': str(root / 'train.log'),
                    'config_path': str(root / 'config.toml'),
                }

            with (
                patch.object(sl_ab, 'run_phase', side_effect=fake_run_phase),
                patch.object(sl_ab, 'phase_storage_root_override', return_value=None),
            ):
                result = sl_ab.run_arm(
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

            self.assertEqual(
                'best_acc',
                result['phase_results']['phase_a']['handoff']['checkpoint_type'],
            )
            self.assertEqual(
                'best_acc',
                result['phase_results']['phase_b']['handoff']['checkpoint_type'],
            )
            self.assertTrue(phase_roots['phase_a'].exists())
            self.assertTrue(phase_roots['phase_b'].exists())
            self.assertTrue(phase_roots['phase_c'].exists())

    def test_run_arm_reuses_exact_lineage_and_rejects_changed_parent(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_ab_root = sl_ab.AB_ROOT
            sl_ab.AB_ROOT = Path(tmp_dir) / 'sl_ab'
            training_calls = []
            plans = {}

            def fake_run_training(cfg_path, log_path):
                cfg = sl_ab.load_toml_file(cfg_path)
                supervised = cfg['supervised']
                provenance = supervised['run_provenance']
                phase_name = provenance['phase_name']
                training_calls.append(phase_name)
                plans[phase_name] = provenance
                state = {
                    'checkpoint_id': f'{phase_name}-winner',
                    'run_provenance': provenance,
                    'steps': supervised['max_steps'],
                    'optimizer_steps': supervised['max_steps'],
                    'epoch': 0,
                    'timestamp': 1.0,
                    'optimizer': {'param_groups': [{'lr': 1e-5}]},
                    'last_full_recent_metrics': {
                        'loss': 0.1,
                        'action_quality_score': 0.5,
                        'selection_quality_score': 0.5,
                        'rank_acc': 0.2,
                    },
                }
                for key in (
                    'state_file',
                    'best_loss_state_file',
                    'best_acc_state_file',
                    'best_rank_state_file',
                ):
                    checkpoint_path = Path(supervised[key])
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state, checkpoint_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text('completed\n', encoding='utf-8', newline='\n')

            grouped = {
                '200901': ['early.json.gz'],
                '202101': ['mid.json.gz'],
                '202401': ['recent.json.gz'],
            }
            eval_splits = {
                'monitor_recent_files': ['monitor.json.gz'],
                'full_recent_files': ['full.json.gz'],
                'old_regression_files': ['old.json.gz'],
            }
            screening_overrides = {
                'phase_steps': {
                    'phase_a': 1,
                    'phase_b': 1,
                    'phase_c': 1,
                },
                'phase_train_pool': {
                    'phase_a': 3,
                    'phase_b': 3,
                    'phase_c': 3,
                },
            }

            def run_unit_arm():
                return sl_ab.run_arm(
                    base_cfg={'control': {'batch_size': 128}},
                    grouped=grouped,
                    ab_name='lineage_ab',
                    arm_name='lineage_arm',
                    scheduler_profile='cosine',
                    curriculum_profile='broad_to_recent',
                    weight_profile='strong',
                    window_profile='24m_12m',
                    seed=123,
                    eval_splits=eval_splits,
                    step_scale=1.0,
                    allow_early_stopping=False,
                )

            try:
                with (
                    patch.dict(
                        sl_ab.BASE_SCREENING,
                        screening_overrides,
                        clear=False,
                    ),
                    patch.object(
                        sl_ab,
                        'phase_storage_root_override',
                        return_value=None,
                    ),
                    patch.object(
                        sl_ab,
                        'run_training',
                        side_effect=fake_run_training,
                    ),
                ):
                    first = run_unit_arm()
                    second = run_unit_arm()

                    self.assertEqual(
                        ['phase_a', 'phase_b', 'phase_c'],
                        training_calls,
                    )
                    self.assertEqual(
                        'phase_a-winner',
                        plans['phase_b']['parent_checkpoint_id'],
                    )
                    self.assertEqual(
                        'phase_b-winner',
                        plans['phase_c']['parent_checkpoint_id'],
                    )
                    self.assertTrue(
                        second['phase_results']['phase_a'][
                            'reused_completed_checkpoint'
                        ]
                    )

                    phase_a_handoff = Path(
                        first['phase_results']['phase_a']['handoff']['path']
                    )
                    tampered = torch.load(
                        phase_a_handoff,
                        map_location='cpu',
                        weights_only=False,
                    )
                    tampered['checkpoint_id'] = 'phase_a-replaced'
                    torch.save(tampered, phase_a_handoff)

                    with self.assertRaisesRegex(
                        RuntimeError,
                        'phase artifact plan mismatch',
                    ):
                        run_unit_arm()
            finally:
                sl_ab.AB_ROOT = original_ab_root

    def test_formal_checkpoint_run_disables_loss_only_early_stopping(self):
        fake_result = {
            'final': {
                name: {
                    'last_full_recent_metrics': {
                        'loss': 0.1,
                        'action_quality_score': 0.5,
                        'selection_quality_score': 0.5,
                        'rank_acc': 0.2,
                    },
                }
                for name in ('best_loss', 'best_acc', 'best_rank', 'latest')
            },
        }
        with (
            patch.object(sl_ab, 'build_eval_splits', return_value={}),
            patch.object(sl_ab, 'run_arm', return_value=fake_result) as run_arm_mock,
            patch.object(sl_ab, 'save_results'),
        ):
            sl_ab.run_ab6_checkpoint(
                {},
                {},
                seed=1,
                scheduler_profile='cosine',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                step_scale=140,
                ab_name='s140',
            )

        self.assertFalse(run_arm_mock.call_args.kwargs['allow_early_stopping'])


if __name__ == '__main__':
    unittest.main()

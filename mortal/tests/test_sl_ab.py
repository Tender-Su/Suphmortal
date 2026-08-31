import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.supervised.run_sl_ab as sl_ab
from mortal.supervised.sl_selection import SCENARIO_SCORE_VERSION


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

    def test_convergence_checkpoint_completes_only_after_convergence(self):
        config = {
            'max_steps': 8_000_000,
            'convergence': {'enabled': True},
        }
        self.assertFalse(
            sl_ab.checkpoint_is_complete_for_config(
                {
                    'steps': 8_000_000,
                    'convergence_state': {'converged': False},
                },
                config,
            )
        )
        self.assertTrue(
            sl_ab.checkpoint_is_complete_for_config(
                {
                    'steps': 3_000_000,
                    'convergence_state': {'converged': True},
                },
                config,
            )
        )

    def test_adaptive_checkpoint_completes_only_after_controller_completion(self):
        config = {
            'max_steps': 8_000_000,
            'adaptive_curriculum': {'enabled': True},
        }
        self.assertFalse(
            sl_ab.checkpoint_is_complete_for_config(
                {
                    'steps': 8_000_000,
                    'adaptive_curriculum_state': {'completed': False},
                },
                config,
            )
        )
        self.assertTrue(
            sl_ab.checkpoint_is_complete_for_config(
                {
                    'steps': 250_000,
                    'adaptive_curriculum_state': {'completed': True},
                },
                config,
            )
        )

    def test_longabc_convergence_decouples_core_from_training_cap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpts = sl_ab.checkpoint_paths(Path(tmp_dir) / 'phase_a')
            overrides = sl_ab.make_phase_overrides(
                ckpts,
                seed=7,
                phase_name='phase_a',
                max_steps=8_000_000,
                scheduler_core_steps=2_520_000,
                scheduler_type='cosine',
                init_state_file=None,
                allow_early_stopping=False,
                convergence_profile='longabc',
            )

        supervised = overrides['supervised']
        self.assertEqual(8_000_000, supervised['max_steps'])
        self.assertEqual(2_520_000, supervised['scheduler']['max_steps'])
        self.assertEqual(
            2_520_000,
            supervised['convergence']['core_optimizer_steps'],
        )
        self.assertEqual(
            [1e-5, 5e-6, 2.5e-6, 1e-6],
            supervised['convergence']['tail_lr_levels'],
        )
        self.assertEqual(0, supervised['early_stopping_patience_checks'])

    def test_full_dynamic_overrides_use_evidence_gates_and_constant_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpts = sl_ab.checkpoint_paths(Path(tmp_dir) / 'phase_c')
            overrides = sl_ab.make_phase_overrides(
                ckpts,
                seed=7,
                phase_name='phase_c',
                max_steps=8_000_000,
                scheduler_core_steps=8_000_000,
                scheduler_type='constant',
                init_state_file=None,
                allow_early_stopping=False,
                adaptive_curriculum_profile='full_dynamic',
            )

        supervised = overrides['supervised']
        adaptive = supervised['adaptive_curriculum']
        self.assertEqual('constant', supervised['scheduler']['type'])
        self.assertEqual(5_000, supervised['scheduler']['warm_up_steps'])
        self.assertEqual(10_000, supervised['val_every_steps'])
        self.assertEqual(10_000, supervised['save_every'])
        self.assertEqual(50_000, adaptive['gate_every_steps'])
        self.assertEqual(2, adaptive['required_futile_gates'])
        self.assertTrue(adaptive['final_phase'])
        self.assertEqual(
            [1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6, 2.5e-6, 1e-6],
            adaptive['lr_levels'],
        )
        self.assertEqual(0, supervised['early_stopping_patience_checks'])

    def test_mature_bootstrap_caps_dynamic_lr_ladder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpts = sl_ab.checkpoint_paths(Path(tmp_dir) / 'phase_c')
            overrides = sl_ab.make_phase_overrides(
                ckpts,
                seed=7,
                phase_name='phase_c',
                max_steps=8_000_000,
                scheduler_core_steps=8_000_000,
                scheduler_type='constant',
                init_state_file=None,
                allow_early_stopping=False,
                adaptive_curriculum_profile='full_dynamic',
                adaptive_peak_lr=5e-5,
            )

        supervised = overrides['supervised']
        self.assertEqual(5e-5, supervised['lr'])
        self.assertEqual(
            [5e-5, 2.5e-5, 1e-5, 5e-6, 2.5e-6, 1e-6],
            supervised['adaptive_curriculum']['lr_levels'],
        )

    def test_adaptive_bootstrap_starts_at_phase_b_and_preserves_anchor(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / 'legacy_a_best.pth'
            eval_splits = {
                'monitor_recent_files': ['monitor.json.gz'],
                'full_recent_files': ['full.json.gz'],
                'old_regression_files': ['old.json.gz'],
            }
            source_plan = {
                'plan_id': 'legacy-a-plan',
                'phase_name': 'phase_a',
                'monitor_recent_files_digest': sl_ab.stable_digest(
                    eval_splits['monitor_recent_files']
                ),
                'full_recent_files_digest': sl_ab.stable_digest(
                    eval_splits['full_recent_files']
                ),
                'old_regression_files_digest': sl_ab.stable_digest(
                    eval_splits['old_regression_files']
                ),
            }
            torch.save(
                {
                    'checkpoint_id': 'legacy-a-best',
                    'run_provenance': source_plan,
                    'steps': 2_880_000,
                    'optimizer_steps': 2_880_000,
                    'epoch': 8,
                    'timestamp': 1.0,
                    'optimizer': {'param_groups': [{'lr': 5e-6}]},
                    'last_full_recent_metrics': {
                        'policy_loss': 0.44,
                        'action_quality_score': -0.20,
                        'rank_acc': 0.30,
                    },
                    'last_old_regression_metrics': {'policy_loss': 0.48},
                },
                source_path,
            )
            calls = []

            def fake_run_phase(**kwargs):
                calls.append(kwargs)
                phase_name = kwargs['phase_name']
                checkpoint_path = root / phase_name / 'adaptive_best.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    phase_name,
                    encoding='utf-8',
                    newline='\n',
                )
                manifest_path = root / phase_name / 'phase_manifest.json'
                manifest_path.write_text(
                    '{"plan": {"plan_id": "unit"}, "status": "completed"}',
                    encoding='utf-8',
                    newline='\n',
                )
                summary = {
                    'checkpoint_id': f'{phase_name}-best',
                    'path': str(checkpoint_path),
                    'last_full_recent_metrics': {
                        'policy_loss': 0.43 if phase_name == 'phase_b' else 0.42,
                        'action_quality_score': -0.19,
                        'rank_acc': 0.31,
                    },
                    'last_old_regression_metrics': {'policy_loss': 0.47},
                    'adaptive_curriculum_state': {'completed': True},
                }
                return {
                    'latest': summary,
                    'best_policy': summary,
                    'best_loss': summary,
                    'best_acc': summary,
                    'best_rank': summary,
                    'adaptive_best': summary,
                    'portfolio': {},
                    'paths': {'manifest_file': str(manifest_path)},
                }

            original_ab_root = sl_ab.AB_ROOT
            sl_ab.AB_ROOT = root / 'sl_ab'
            try:
                with (
                    patch.object(sl_ab, 'run_phase', side_effect=fake_run_phase),
                    patch.object(sl_ab, 'phase_storage_root_override', return_value=None),
                ):
                    result = sl_ab.run_arm(
                        base_cfg={},
                        grouped={},
                        ab_name='bootstrap_unit',
                        arm_name='dynamic_from_b',
                        scheduler_profile='phasewise',
                        curriculum_profile='broad_to_recent',
                        weight_profile='strong',
                        window_profile='24m_12m',
                        seed=123,
                        eval_splits=eval_splits,
                        step_scale=1.0,
                        allow_early_stopping=False,
                        adaptive_curriculum_profile='full_dynamic',
                        adaptive_start_phase='phase_b',
                        adaptive_bootstrap_state_file=str(source_path),
                    )
            finally:
                sl_ab.AB_ROOT = original_ab_root

            self.assertEqual(['phase_b', 'phase_c'], result['phase_order'])
            self.assertEqual(str(source_path), calls[0]['init_state_file'])
            self.assertIsNone(calls[0]['adaptive_handoff_source'])
            self.assertIsNone(calls[1]['init_state_file'])
            self.assertEqual(
                str(root / 'phase_b' / 'adaptive_best.pth'),
                calls[1]['adaptive_handoff_source'],
            )
            self.assertEqual([1e-5, 1e-5], [call['adaptive_peak_lr'] for call in calls])
            self.assertEqual(5e-6, calls[0]['adaptive_warmup_init_lr'])
            self.assertIsNone(calls[1]['adaptive_warmup_init_lr'])
            self.assertTrue(result['adaptive_bootstrap']['eval_split_digests_match'])
            self.assertFalse(result['adaptive_bootstrap']['optimizer_state_preserved'])
            self.assertEqual(5e-6, result['adaptive_bootstrap']['source_lr'])
            self.assertEqual(2.0, result['adaptive_bootstrap']['rewarm_ratio'])
            self.assertIn('bootstrap_anchor', result['cross_phase_candidates'])
            self.assertTrue(
                (root / 'sl_ab' / 'bootstrap_unit' / 'adaptive_bootstrap.json').exists()
            )

    def test_adaptive_phase_b_requires_bootstrap_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'requires a bootstrap checkpoint'):
            sl_ab.run_arm(
                base_cfg={},
                grouped={},
                ab_name='unit',
                arm_name='unit',
                scheduler_profile='phasewise',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                seed=1,
                eval_splits={},
                step_scale=1.0,
                adaptive_curriculum_profile='full_dynamic',
                adaptive_start_phase='phase_b',
            )

    def test_checkpoint_paths_keep_distinct_metric_winners_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir) / 'exp'
            ckpts = sl_ab.checkpoint_paths(exp_dir)

            self.assertEqual(exp_dir / 'checkpoints' / 'best_loss.pth', ckpts['best_loss_state_file'])
            self.assertEqual(exp_dir / 'checkpoints' / 'best_policy.pth', ckpts['best_policy_state_file'])
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
                'scheduler': {'max_steps': 80},
            },
        }

        self.assertEqual(
            sl_ab.semantic_config_digest(base),
            sl_ab.semantic_config_digest(changed_runtime),
        )
        changed_cap_only = deepcopy(changed_runtime)
        changed_cap_only['supervised']['max_steps'] = 200
        self.assertEqual(
            sl_ab.semantic_config_digest(base),
            sl_ab.semantic_config_digest(changed_cap_only),
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

    def test_phase_plan_id_ignores_only_monotonic_training_cap(self):
        base_cfg = {
            'control': {'batch_size': 128},
            'supervised': {
                'max_steps': 8_000_000,
                'scheduler': {'type': 'cosine', 'max_steps': 2_520_000},
            },
        }
        kwargs = {
            'ab_name': 'converge',
            'arm_name': 'anchor',
            'phase_name': 'phase_a',
            'scheduler_type': 'cosine',
            'weight_profile': 'strong',
            'window_profile': '24m_12m',
            'seed': 7,
            'step_scale': 140.0,
            'scheduler_core_steps': 2_520_000,
            'train_files': ['train'],
            'eval_splits': {
                'monitor_recent_files': ['monitor'],
                'full_recent_files': ['full'],
                'old_regression_files': ['old'],
            },
            'init_state_file': None,
        }
        first = sl_ab.build_phase_plan(
            max_steps=8_000_000,
            cfg=base_cfg,
            **kwargs,
        )
        extended_cfg = deepcopy(base_cfg)
        extended_cfg['supervised']['max_steps'] = 10_000_000
        second = sl_ab.build_phase_plan(
            max_steps=10_000_000,
            cfg=extended_cfg,
            **kwargs,
        )

        self.assertEqual(first['plan_id'], second['plan_id'])
        self.assertNotEqual(
            first['training_cap_steps'],
            second['training_cap_steps'],
        )

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

    def test_run_arm_uses_best_offline_candidate_including_portfolio_as_parent(self):
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
                pareto = root / 'candidate_portfolio' / 'pareto.pth'
                best_loss.parent.mkdir(parents=True, exist_ok=True)
                pareto.parent.mkdir(parents=True, exist_ok=True)
                best_loss.write_text(phase_name, encoding='utf-8', newline='\n')
                best_acc.write_text(phase_name, encoding='utf-8', newline='\n')
                best_rank.write_text(phase_name, encoding='utf-8', newline='\n')
                pareto.write_text(phase_name, encoding='utf-8', newline='\n')
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
                    self.assertEqual(
                        str(phase_roots['phase_a'] / 'candidate_portfolio' / 'pareto.pth'),
                        init_state_file,
                    )
                elif phase_name == 'phase_c':
                    self.assertEqual(
                        str(phase_roots['phase_b'] / 'candidate_portfolio' / 'pareto.pth'),
                        init_state_file,
                    )
                loss_metrics = {
                    'loss': 0.1000,
                    'policy_loss': 0.1000,
                    'action_quality_score': 0.3,
                    'selection_quality_score': 0.3,
                    'rank_acc': 0.2,
                }
                action_metrics = {
                    'loss': 0.1002,
                    'policy_loss': 0.1002,
                    'action_quality_score': 0.5,
                    'selection_quality_score': 0.5,
                    'rank_acc': 0.3,
                }
                rank_metrics = {
                    'loss': 0.2,
                    'policy_loss': 0.2,
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
                    'portfolio': {
                        'pareto_00': {
                            'phase': phase_name,
                            'checkpoint_id': f'{phase_name}-pareto',
                            'path': str(pareto),
                            'last_full_recent_metrics': {
                                'loss': 0.1001,
                                'policy_loss': 0.1001,
                                'action_quality_score': 0.7,
                                'selection_quality_score': 0.7,
                                'rank_acc': 0.4,
                            },
                        },
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
                'pareto_00',
                result['phase_results']['phase_a']['handoff']['checkpoint_type'],
            )
            self.assertEqual(
                'pareto_00',
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

    def test_phase_extension_migration_preserves_training_state(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / 'source'
            source_checkpoints = source_root / 'checkpoints'
            source_checkpoints.mkdir(parents=True)
            target_root = root / 'target'
            target_ckpts = sl_ab.checkpoint_paths(target_root)
            source_plan = {
                'schema_version': 1,
                'plan_id': 'source-plan',
                'phase_name': 'phase_a',
                'scheduler_type': 'cosine',
                'weight_profile': 'strong',
                'window_profile': '24m_12m',
                'training_seed': 7,
                'file_order_seed': 108,
                'train_files_digest': 'train',
                'monitor_recent_files_digest': 'monitor',
                'full_recent_files_digest': 'full',
                'old_regression_files_digest': 'old',
                'parent_checkpoint_id': '',
                'parent_plan_id': '',
            }
            source_cfg = {
                'control': {'batch_size': 128},
                'supervised': {
                    'max_steps': 100,
                    'state_file': 'source/latest.pth',
                    'scheduler': {'type': 'cosine', 'max_steps': 100},
                    'early_stopping_patience_checks': 0,
                },
            }
            expected_plan = {
                **source_plan,
                'schema_version': 2,
                'plan_id': 'target-plan',
                'scheduler_core_steps': 100,
                'training_cap_steps': 1000,
            }
            target_cfg = deepcopy(source_cfg)
            target_cfg['supervised'].update({
                'max_steps': 1000,
                'state_file': str(target_ckpts['state_file']),
                'best_policy_state_file': str(target_ckpts['best_policy_state_file']),
                'candidate_portfolio_limit': 12,
                'convergence': {
                    'enabled': True,
                    'core_optimizer_steps': 100,
                    'tail_lr_levels': [1e-5, 1e-6],
                },
            })

            filenames = {
                'latest.pth': 0.40,
                'best_loss.pth': 0.35,
                'best_action_score.pth': 0.36,
                'best_rank.pth': 0.37,
            }
            source_ids = {}
            for index, (filename, policy_loss) in enumerate(filenames.items()):
                checkpoint_id = f'source-{index}'
                source_ids[filename] = checkpoint_id
                torch.save(
                    {
                        'checkpoint_id': checkpoint_id,
                        'run_provenance': source_plan,
                        'config': source_cfg,
                        'mortal': {'weight': torch.tensor([1.0, 2.0])},
                        'policy_net': {'weight': torch.tensor([3.0])},
                        'aux_net': {'weight': torch.tensor([4.0])},
                        'opponent_aux_net': None,
                        'danger_aux_net': None,
                        'optimizer': {'state': {0: {'step': torch.tensor(9.0)}}},
                        'scheduler': {'max_steps': 100, 'last_epoch': 49},
                        'scaler': {'scale': 1.0},
                        'steps': 50,
                        'optimizer_steps': 49,
                        'best_full_recent_policy_loss': policy_loss,
                        'last_full_recent_metrics': {'policy_loss': policy_loss},
                    },
                    source_checkpoints / filename,
                )

            migration = sl_ab.migrate_phase_extension(
                source_root=source_root,
                target_root=target_root,
                target_ckpts=target_ckpts,
                expected_plan=expected_plan,
                target_cfg=target_cfg,
            )

            migrated_latest = torch.load(
                target_ckpts['state_file'],
                map_location='cpu',
                weights_only=False,
            )
            source_latest = torch.load(
                source_checkpoints / 'latest.pth',
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual('target-plan', migrated_latest['run_provenance']['plan_id'])
            self.assertNotEqual(source_latest['checkpoint_id'], migrated_latest['checkpoint_id'])
            self.assertTrue(sl_ab.nested_state_equal(source_latest['mortal'], migrated_latest['mortal']))
            self.assertTrue(sl_ab.nested_state_equal(source_latest['optimizer'], migrated_latest['optimizer']))
            self.assertTrue(sl_ab.nested_state_equal(source_latest['scheduler'], migrated_latest['scheduler']))
            self.assertEqual('source-plan', source_latest['run_provenance']['plan_id'])
            self.assertEqual('best_loss_state_file', migration['best_policy_source'])
            self.assertEqual(0.35, migration['best_policy_loss'])
            self.assertEqual(0.35, migrated_latest['best_full_recent_policy_loss'])
            self.assertTrue(target_ckpts['best_policy_state_file'].exists())
            migrated_best_policy = torch.load(
                target_ckpts['best_policy_state_file'],
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual(
                0.35,
                migrated_best_policy['last_full_recent_metrics']['policy_loss'],
            )
            self.assertEqual(0.35, migrated_best_policy['best_full_recent_policy_loss'])
            self.assertTrue(target_ckpts['manifest_file'].exists())
            self.assertFalse((target_root / '.migration_in_progress.json').exists())

    def test_adaptive_phase_handoff_preserves_training_state_and_resets_gate(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / 'source' / 'checkpoints' / 'adaptive_best.pth'
            target_path = root / 'target' / 'checkpoints' / 'latest.pth'
            source_path.parent.mkdir(parents=True)
            source_plan = {
                'plan_id': 'source-plan',
                'phase_name': 'phase_a',
            }
            source_cfg = {
                'control': {'version': 4, 'batch_size': 1024},
                'dataset': {'enable_augmentation': True},
                'optim': {'lr': 1e-4, 'weight_decay': 0.01},
                'resnet': {'conv_channels': 256},
                'aux': {'next_rank_weight': 0.1},
                'supervised': {
                    'state_file': str(source_path.parent / 'latest.pth'),
                    'adaptive_best_state_file': str(source_path),
                    'max_steps': 8_000_000,
                    'scheduler': {
                        'type': 'constant',
                        'warm_up_steps': 5_000,
                        'max_steps': 8_000_000,
                    },
                    'adaptive_curriculum': {
                        'enabled': True,
                        'phase_name': 'phase_a',
                        'final_phase': False,
                        'gate_every_steps': 50_000,
                        'required_futile_gates': 2,
                        'primary': {
                            'name': 'policy_loss',
                            'direction': 'lower',
                            'meaningful_delta': 2e-4,
                        },
                    },
                },
            }
            source_state = {
                'checkpoint_id': 'source-checkpoint',
                'run_provenance': source_plan,
                'config': source_cfg,
                'mortal': {'weight': torch.tensor([1.0, 2.0])},
                'policy_net': {'weight': torch.tensor([3.0])},
                'aux_net': {'weight': torch.tensor([4.0])},
                'opponent_aux_net': None,
                'danger_aux_net': None,
                'optimizer': {
                    'state': {0: {'step': torch.tensor(9.0)}},
                    'param_groups': [{'lr': 1e-4}],
                },
                'optimizer_param_groups': [{'lr': 1e-4}],
                'scheduler': {'last_epoch': 49, 'tail_lr': 1e-4},
                'scaler': {'scale': 1024.0},
                'steps': 150_000,
                'optimizer_steps': 149_997,
                'skipped_optimizer_steps': 3,
                'nonfinite_batches': 1,
                'epoch': 12,
                'validation_checks': 15,
                'last_full_recent_metrics': {'policy_loss': 0.35},
                'adaptive_curriculum_state': {
                    'schema_version': 1,
                    'phase_name': 'phase_a',
                    'completed': True,
                    'best_step': 100_000,
                    'best_metrics': {'policy_loss': 0.35},
                    'best_cluster_records': {
                        'policy_loss': [[1, 0.35, 1], [2, 0.36, 1]],
                    },
                },
            }
            torch.save(source_state, source_path)

            target_cfg = deepcopy(source_cfg)
            target_cfg['supervised'].update({
                'state_file': str(target_path),
                'adaptive_best_state_file': str(
                    target_path.parent / 'adaptive_best.pth'
                ),
                'adaptive_curriculum': {
                    'enabled': True,
                    'phase_name': 'phase_b',
                    'final_phase': False,
                    'gate_every_steps': 50_000,
                    'required_futile_gates': 2,
                    'primary': {
                        'name': 'policy_loss',
                        'direction': 'lower',
                        'meaningful_delta': 2e-4,
                    },
                },
            })
            expected_plan = {
                'plan_id': 'target-plan',
                'phase_name': 'phase_b',
                'parent_checkpoint_id': 'source-checkpoint',
                'parent_plan_id': 'source-plan',
            }

            migration = sl_ab.migrate_adaptive_phase_handoff(
                source_path=source_path,
                target_path=target_path,
                expected_plan=expected_plan,
                target_cfg=target_cfg,
            )

            migrated = torch.load(
                target_path,
                map_location='cpu',
                weights_only=False,
            )
            for key in sl_ab.ADAPTIVE_PHASE_PRESERVED_STATE_KEYS:
                self.assertTrue(
                    sl_ab.nested_state_equal(source_state.get(key), migrated.get(key)),
                    key,
                )
            adaptive_state = migrated['adaptive_curriculum_state']
            self.assertEqual('phase_b', adaptive_state['phase_name'])
            self.assertFalse(adaptive_state['completed'])
            self.assertEqual(0, adaptive_state['gate_index'])
            self.assertEqual(100_000, adaptive_state['best_step'])
            self.assertEqual('inherit_baseline', adaptive_state['last_action'])
            self.assertEqual(0, migrated['validation_checks'])
            self.assertIsNone(migrated['last_full_recent_metrics'])
            self.assertEqual(-1, migrated['epoch'])
            self.assertTrue(
                migrated['adaptive_phase_handoff_migration'][
                    'data_traversal_restart'
                ]
            )
            self.assertEqual('source-checkpoint', migration['source_checkpoint_id'])
            self.assertTrue(
                (target_path.parent.parent / 'adaptive_phase_handoff.json').exists()
            )
            adaptive_best_path = Path(
                target_cfg['supervised']['adaptive_best_state_file']
            )
            self.assertTrue(adaptive_best_path.exists())
            inherited_best = torch.load(
                adaptive_best_path,
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual(
                migrated['checkpoint_id'],
                inherited_best['checkpoint_id'],
            )
            original = torch.load(
                source_path,
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual('phase_a', original['run_provenance']['phase_name'])

    def test_phase_extension_rejects_scheduler_horizon_change(self):
        source_state = {
            'run_provenance': {
                'plan_id': 'source',
                'phase_name': 'phase_a',
                'scheduler_type': 'cosine',
                'weight_profile': 'strong',
                'window_profile': '24m_12m',
                'training_seed': 7,
                'file_order_seed': 108,
                'train_files_digest': 'train',
                'monitor_recent_files_digest': 'monitor',
                'full_recent_files_digest': 'full',
                'old_regression_files_digest': 'old',
                'parent_checkpoint_id': '',
                'parent_plan_id': '',
            },
            'config': {'supervised': {'scheduler': {'max_steps': 100}}},
            'scheduler': {'max_steps': 100},
            'steps': 50,
            'optimizer_steps': 50,
        }
        expected_plan = {
            **source_state['run_provenance'],
            'plan_id': 'target',
            'scheduler_core_steps': 200,
        }

        with self.assertRaisesRegex(RuntimeError, 'scheduler horizon mismatch'):
            sl_ab.validate_phase_extension_source(
                source_state,
                source_path=Path('source.pth'),
                expected_plan=expected_plan,
                target_cfg=source_state['config'],
            )

    def test_extension_config_ignores_unrelated_rebased_paths(self):
        source = {
            'control': {'version': 4, 'batch_size': 1024, 'state_file': 'old.pth'},
            'dataset': {
                'enable_augmentation': True,
                'augmented_first': False,
                'globs': [r'C:\data\*.json'],
            },
            'optim': {'eps': 1e-8},
            'resnet': {'conv_channels': 256},
            'aux': {'next_rank_weight': 0.1},
            'supervised': {
                'batch_size': 1024,
                'state_file': r'C:\old\latest.pth',
                'max_steps': 2_520_000,
            },
            '1v3': {'log_dir': r'C:\old\1v3'},
        }
        target = deepcopy(source)
        target['control']['state_file'] = './mortal.pth'
        target['dataset']['globs'] = ['C:/data/*.json']
        target['supervised']['state_file'] = r'C:\new\latest.pth'
        target['supervised']['max_steps'] = 8_000_000
        target['1v3']['log_dir'] = './1v3'

        self.assertEqual(
            sl_ab.extension_immutable_config_digest(source),
            sl_ab.extension_immutable_config_digest(target),
        )
        self.assertEqual([], sl_ab.extension_config_differences(source, target))

    def test_extension_config_reports_training_semantic_difference(self):
        source = {
            'control': {'version': 4},
            'dataset': {'enable_augmentation': True},
            'optim': {'weight_decay': 0.01},
            'resnet': {'conv_channels': 256},
            'supervised': {'batch_size': 1024},
        }
        target = deepcopy(source)
        target['optim']['weight_decay'] = 0.02

        self.assertEqual(
            [('optim.weight_decay', 0.01, 0.02)],
            sl_ab.extension_config_differences(source, target),
        )

    def test_load_candidate_portfolio_preserves_compact_selection_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint = root / 'candidate.pth'
            checkpoint.touch()
            sl_ab.atomic_write_json(
                root / 'portfolio.json',
                {
                    'schema_version': 1,
                    'candidates': [{
                        'checkpoint_id': 'candidate-id',
                        'path': str(checkpoint),
                        'step': 40,
                        'optimizer_steps': 39,
                        'policy_loss': 0.40,
                        'action_quality_score': 0.50,
                        'old_regression_policy_loss': 0.60,
                        'rank_acc': 0.30,
                        'full_recent_metrics': {
                            'policy_loss': 0.40,
                            'action_quality_score': 0.50,
                            'scenario_quality_score': 0.25,
                            'scenario_quality_score_version': SCENARIO_SCORE_VERSION,
                            'rank_acc': 0.30,
                        },
                    }],
                },
            )

            portfolio = sl_ab.load_candidate_portfolio(root)

            metrics = next(iter(portfolio.values()))['last_full_recent_metrics']
            self.assertEqual(0.25, metrics['scenario_quality_score'])
            self.assertGreater(metrics['selection_quality_score'], 0.50)

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

    def test_run_arm_rejects_extension_without_convergence_profile(self):
        with self.assertRaisesRegex(ValueError, 'convergence profile'):
            sl_ab.run_arm(
                {},
                {},
                ab_name='unit',
                arm_name='unit',
                scheduler_profile='cosine',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                seed=1,
                eval_splits={},
                step_scale=140,
                phase_extension_sources={'phase_a': 'source'},
            )


if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_supervised


class FakeLoader:
    def __init__(self):
        self.iterator = FakeLoaderIterator()

    def __iter__(self):
        return self.iterator


class FakeLoaderIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class TrainSupervisedResumeAuxTests(unittest.TestCase):
    def test_safe_default_collate_normalizes_numpy_bool_scalars(self):
        batch = [
            (np.bool_(True), np.array([1.0, 2.0], dtype=np.float32)),
            (np.bool_(False), np.array([3.0, 4.0], dtype=np.float32)),
        ]

        collated = train_supervised.safe_default_collate(batch)

        self.assertTrue(torch.equal(collated[0], torch.tensor([True, False], dtype=torch.bool)))
        self.assertTrue(
            torch.equal(
                collated[1],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            )
        )

    def test_loader_uses_oracle_disables_visible_only_validation_loading(self):
        self.assertTrue(
            train_supervised.loader_uses_oracle(
                training=True,
                use_oracle=True,
                validation_use_oracle=False,
            )
        )
        self.assertFalse(
            train_supervised.loader_uses_oracle(
                training=False,
                use_oracle=True,
                validation_use_oracle=False,
            )
        )
        self.assertFalse(
            train_supervised.loader_uses_oracle(
                training=False,
                use_oracle=False,
                validation_use_oracle=True,
            )
        )

    def test_batch_includes_oracle_distinguishes_visible_only_validation_batches(self):
        self.assertFalse(train_supervised.batch_includes_oracle(5, enable_danger_aux=False))
        self.assertTrue(train_supervised.batch_includes_oracle(6, enable_danger_aux=False))
        self.assertFalse(train_supervised.batch_includes_oracle(9, enable_danger_aux=True))
        self.assertTrue(train_supervised.batch_includes_oracle(10, enable_danger_aux=True))

    def test_missing_init_state_file_errors_immediately(self):
        with self.assertRaisesRegex(FileNotFoundError, r'stage1\.init_state_file does not exist'):
            train_supervised.ensure_init_state_file_exists(
                r'X:\missing\stage0_5_seed.pth',
                cfg_prefix='stage1',
            )

    def test_full_validation_zero_disables_monitor_checks(self):
        self.assertFalse(
            train_supervised.should_run_full_validation_this_check(
                full_val_every_checks=0,
                validation_checks=1,
                has_full_recent_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_full_validation_this_check(
                full_val_every_checks=0,
                validation_checks=1,
                has_full_recent_files=False,
            )
        )
        self.assertFalse(
            train_supervised.should_run_full_validation_this_check(
                full_val_every_checks=2,
                validation_checks=1,
                has_full_recent_files=True,
            )
        )
        self.assertTrue(
            train_supervised.should_run_full_validation_this_check(
                full_val_every_checks=2,
                validation_checks=2,
                has_full_recent_files=True,
            )
        )

    def test_fallback_full_validation_restores_epoch_end_and_budget_sync_passes(self):
        self.assertTrue(
            train_supervised.should_run_fallback_full_validation(
                ran_full_val=False,
                has_full_recent_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_fallback_full_validation(
                ran_full_val=True,
                has_full_recent_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_fallback_full_validation(
                ran_full_val=False,
                has_full_recent_files=False,
            )
        )

    def test_old_regression_zero_disables_monitor_checks(self):
        self.assertFalse(
            train_supervised.should_run_old_regression_validation_this_check(
                old_regression_every_checks=0,
                validation_checks=1,
                has_old_regression_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_old_regression_validation_this_check(
                old_regression_every_checks=2,
                validation_checks=1,
                has_old_regression_files=True,
            )
        )
        self.assertTrue(
            train_supervised.should_run_old_regression_validation_this_check(
                old_regression_every_checks=2,
                validation_checks=2,
                has_old_regression_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_old_regression_validation_this_check(
                old_regression_every_checks=2,
                validation_checks=2,
                has_old_regression_files=False,
            )
        )

    def test_old_regression_fallback_runs_after_full_validation_when_periodic_checks_disabled(self):
        self.assertTrue(
            train_supervised.should_run_old_regression_after_full_validation(
                old_regression_every_checks=0,
                ran_full_val=True,
                has_old_regression_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_old_regression_after_full_validation(
                old_regression_every_checks=2,
                ran_full_val=True,
                has_old_regression_files=True,
            )
        )
        self.assertFalse(
            train_supervised.should_run_old_regression_after_full_validation(
                old_regression_every_checks=0,
                ran_full_val=False,
                has_old_regression_files=True,
            )
        )

    def test_make_closeable_batch_iter_returns_iterator_without_prefetch(self):
        loader = FakeLoader()

        batch_iter, batches_on_device = train_supervised.make_closeable_batch_iter(
            loader,
            enable_cuda_prefetch=False,
            prefetcher_factory=lambda _: self.fail('prefetcher should not be constructed'),
        )

        self.assertIs(loader.iterator, batch_iter)
        self.assertFalse(batches_on_device)

    def test_make_closeable_batch_iter_wraps_loader_with_prefetcher(self):
        loader = FakeLoader()
        created = []

        class FakePrefetcher:
            def __init__(self, wrapped_loader):
                created.append(wrapped_loader)

        batch_iter, batches_on_device = train_supervised.make_closeable_batch_iter(
            loader,
            enable_cuda_prefetch=True,
            prefetcher_factory=FakePrefetcher,
        )

        self.assertIsInstance(batch_iter, FakePrefetcher)
        self.assertEqual([loader], created)
        self.assertTrue(batches_on_device)

    def test_backfill_missing_oracle_obs_creates_zero_batch_when_model_still_requires_it(self):
        obs = torch.randn((3, 1012, 34), dtype=torch.float32)

        oracle_obs = train_supervised.backfill_missing_oracle_obs(
            obs,
            None,
            require_oracle=True,
            oracle_channels=217,
        )

        self.assertEqual((3, 217, 34), tuple(oracle_obs.shape))
        self.assertEqual(obs.dtype, oracle_obs.dtype)
        self.assertEqual(obs.device, oracle_obs.device)
        self.assertTrue(torch.equal(torch.zeros_like(oracle_obs), oracle_obs))

    def test_resume_optimizer_steps_prefers_explicit_optimizer_steps(self):
        self.assertEqual(
            42,
            train_supervised.resume_optimizer_steps_from_state(
                {'steps': 300, 'optimizer_steps': 42},
                default=0,
            ),
        )

    def test_resume_optimizer_steps_falls_back_to_legacy_steps_without_accumulation(self):
        self.assertEqual(
            300,
            train_supervised.resume_optimizer_steps_from_state(
                {'steps': 300},
                opt_step_every=1,
                default=0,
            ),
        )

    def test_resume_optimizer_steps_scales_legacy_steps_by_accumulation(self):
        self.assertEqual(
            75,
            train_supervised.resume_optimizer_steps_from_state(
                {'steps': 300},
                opt_step_every=4,
                default=0,
            ),
        )

    def test_resume_optimizer_steps_rounds_up_flushed_partial_accumulation(self):
        self.assertEqual(
            76,
            train_supervised.resume_optimizer_steps_from_state(
                {'steps': 301},
                opt_step_every=4,
                default=0,
            ),
        )

    def test_post_optimizer_actions_force_max_steps_validation_even_without_periodic_val(self):
        actions = train_supervised.plan_post_optimizer_step_actions(
            steps=10,
            save_every=4000,
            val_every_steps=4000,
            max_steps=10,
        )

        self.assertFalse(actions['save_periodic'])
        self.assertTrue(actions['save_budget_checkpoint'])
        self.assertTrue(actions['release_train_loader'])
        self.assertEqual('max_steps', actions['validation_reason'])
        self.assertTrue(actions['stop_due_to_budget'])

    def test_post_optimizer_actions_reuse_periodic_save_when_budget_and_save_coincide(self):
        actions = train_supervised.plan_post_optimizer_step_actions(
            steps=4000,
            save_every=4000,
            val_every_steps=0,
            max_steps=4000,
        )

        self.assertTrue(actions['save_periodic'])
        self.assertFalse(actions['save_budget_checkpoint'])
        self.assertTrue(actions['release_train_loader'])
        self.assertEqual('max_steps', actions['validation_reason'])
        self.assertTrue(actions['stop_due_to_budget'])

    def test_checkpoint_head_flags_use_stage1_section_aux_enable_overrides(self):
        state = {
            'config': {
                'aux': {
                    'opponent_state_weight': 0.0,
                    'danger_enabled': False,
                    'danger_weight': 0.0,
                },
                'stage1': {
                    'aux': {
                        'opponent_state_weight': 0.25,
                        'danger_weight': 0.4,
                    },
                },
            },
            'opponent_aux_net': {'weights': 1},
            'danger_aux_net': {'weights': 1},
        }

        flags = train_supervised.checkpoint_optional_head_flags_for_state(
            state,
            config_section='stage1',
        )

        self.assertTrue(flags['opponent_aux_net'])
        self.assertTrue(flags['danger_aux_net'])

    def test_checkpoint_head_flags_use_stage1_section_aux_disable_overrides(self):
        state = {
            'config': {
                'aux': {
                    'opponent_state_weight': 0.25,
                    'danger_enabled': True,
                    'danger_weight': 0.4,
                },
                'stage1': {
                    'aux': {
                        'opponent_state_weight': 0.0,
                        'danger_enabled': False,
                        'danger_weight': 0.0,
                    },
                },
            },
            'opponent_aux_net': None,
            'danger_aux_net': None,
        }

        flags = train_supervised.checkpoint_optional_head_flags_for_state(
            state,
            config_section='stage1',
        )

        self.assertFalse(flags['opponent_aux_net'])
        self.assertFalse(flags['danger_aux_net'])

    def test_resolve_stage2_handoff_state_file_respects_stage1_publish_flag(self):
        self.assertEqual(
            '',
            train_supervised.resolve_stage2_handoff_state_file(
                cfg_prefix='stage1',
                supervised_cfg={'publish_stage2_handoff': False},
                control_cfg={'state_file': './checkpoints/mortal.pth'},
            ),
        )
        self.assertEqual(
            './checkpoints/mortal.pth',
            train_supervised.resolve_stage2_handoff_state_file(
                cfg_prefix='stage1',
                supervised_cfg={},
                control_cfg={'state_file': './checkpoints/mortal.pth'},
            ),
        )

    def test_stage2_handoff_export_enables_normal_export_without_named_normal_paths(self):
        self.assertTrue(
            train_supervised.should_enable_normal_export(
                export_normal_checkpoints=False,
                best_loss_normal_state_file='',
                best_acc_normal_state_file='',
                best_rank_normal_state_file='',
                stage2_handoff_state_file='./checkpoints/stage2_handoff.pth',
            )
        )

    def test_normal_export_stays_disabled_without_any_export_targets(self):
        self.assertFalse(
            train_supervised.should_enable_normal_export(
                export_normal_checkpoints=False,
                best_loss_normal_state_file='',
                best_acc_normal_state_file='',
                best_rank_normal_state_file='',
                stage2_handoff_state_file='',
            )
        )

    def test_retryable_validation_error_requires_explicit_resource_marker(self):
        try:
            try:
                raise OSError('WinError 1455: paging file is too small')
            except OSError as inner:
                raise RuntimeError('DataLoader worker (pid(s) 1234) exited unexpectedly') from inner
        except RuntimeError as exc:
            self.assertTrue(train_supervised.is_retryable_validation_error(exc))

    def test_retryable_validation_error_does_not_retry_generic_worker_crash(self):
        exc = RuntimeError('DataLoader worker (pid(s) 1234) exited unexpectedly')
        self.assertFalse(train_supervised.is_retryable_validation_error(exc))

    def test_run_with_validation_retries_retries_retryable_loader_failures(self):
        attempts = []
        sleeps = []

        def flaky_validation():
            attempts.append('call')
            if len(attempts) == 1:
                try:
                    raise OSError('WinError 1455: paging file is too small')
                except OSError as inner:
                    raise RuntimeError('DataLoader worker (pid(s) 1234) exited unexpectedly') from inner
            return 'ok'

        with self.assertLogs(level='ERROR') as logs:
            result = train_supervised.run_with_validation_retries(
                flaky_validation,
                device_type='cpu',
                context='unit-test validation',
                sleep_fn=sleeps.append,
            )

        self.assertEqual('ok', result)
        self.assertEqual(['call', 'call'], attempts)
        self.assertEqual([1.0], sleeps)
        self.assertEqual(1, len(logs.output))

    def test_run_with_validation_retries_does_not_retry_generic_failures(self):
        with self.assertRaisesRegex(RuntimeError, 'generic failure'):
            train_supervised.run_with_validation_retries(
                lambda: (_ for _ in ()).throw(RuntimeError('generic failure')),
                device_type='cpu',
                context='unit-test validation',
            )

    def test_run_with_validation_retries_clears_cuda_cache_before_retry(self):
        attempts = []
        sleeps = []
        cache_clears = []

        def flaky_validation():
            attempts.append('call')
            if len(attempts) == 1:
                try:
                    raise OSError("Couldn't open shared file mapping")
                except OSError as inner:
                    raise RuntimeError('DataLoader worker (pid(s) 4321) exited unexpectedly') from inner
            return 'ok'

        with self.assertLogs(level='ERROR') as logs:
            result = train_supervised.run_with_validation_retries(
                flaky_validation,
                device_type='cuda',
                context='unit-test validation',
                sleep_fn=sleeps.append,
                empty_cache_fn=lambda: cache_clears.append('cleared'),
            )

        self.assertEqual('ok', result)
        self.assertEqual(['call', 'call'], attempts)
        self.assertEqual([1.0], sleeps)
        self.assertEqual(['cleared'], cache_clears)
        self.assertEqual(1, len(logs.output))

    def test_gradient_probe_helpers_compute_expected_geometry(self):
        left = torch.tensor([3.0, 4.0], dtype=torch.float32)
        right = torch.tensor([0.0, 5.0], dtype=torch.float32)

        self.assertAlmostEqual(5.0 / (2.0 ** 0.5), train_supervised.gradient_probe_rms(left), places=6)
        self.assertAlmostEqual(0.8, train_supervised.gradient_probe_cosine(left, right), places=6)
        self.assertAlmostEqual(90.0 ** 0.5 / 10.0, train_supervised.gradient_probe_combo_factor(left, right), places=6)


class TrainSupervisedPostStepPlanTests(unittest.TestCase):
    def test_budget_tail_step_still_runs_budget_validation_off_validation_boundary(self):
        plan = train_supervised.plan_post_optimizer_step_actions(
            steps=12,
            save_every=8,
            val_every_steps=16,
            max_steps=12,
        )

        self.assertFalse(plan['save_periodic'])
        self.assertTrue(plan['save_budget_checkpoint'])
        self.assertTrue(plan['release_train_loader'])
        self.assertEqual('max_steps', plan['validation_reason'])
        self.assertTrue(plan['stop_due_to_budget'])

    def test_budget_on_validation_boundary_prefers_budget_path_and_releases_loader_first(self):
        plan = train_supervised.plan_post_optimizer_step_actions(
            steps=16,
            save_every=8,
            val_every_steps=16,
            max_steps=16,
        )

        self.assertTrue(plan['save_periodic'])
        self.assertFalse(plan['save_budget_checkpoint'])
        self.assertTrue(plan['release_train_loader'])
        self.assertEqual('max_steps', plan['validation_reason'])
        self.assertTrue(plan['stop_due_to_budget'])

    def test_budget_stop_final_actions_force_fallback_validations_when_periodic_checks_disabled(self):
        plan = train_supervised.plan_budget_stop_final_actions(
            stop_due_to_budget=True,
            ran_full_val=False,
            has_full_recent_files=True,
            has_old_regression_files=True,
            old_regression_every_checks=0,
        )

        self.assertTrue(plan['run_full_validation'])
        self.assertTrue(plan['run_old_regression_validation'])
        self.assertTrue(plan['resave_latest_state'])

    def test_budget_stop_final_actions_only_force_full_validation_when_old_regression_has_own_schedule(self):
        plan = train_supervised.plan_budget_stop_final_actions(
            stop_due_to_budget=True,
            ran_full_val=False,
            has_full_recent_files=True,
            has_old_regression_files=True,
            old_regression_every_checks=2,
        )

        self.assertTrue(plan['run_full_validation'])
        self.assertFalse(plan['run_old_regression_validation'])
        self.assertTrue(plan['resave_latest_state'])

    def test_budget_stop_final_actions_skip_resave_when_no_fallback_validation_runs(self):
        plan = train_supervised.plan_budget_stop_final_actions(
            stop_due_to_budget=True,
            ran_full_val=False,
            has_full_recent_files=False,
            has_old_regression_files=True,
            old_regression_every_checks=0,
        )

        self.assertFalse(plan['run_full_validation'])
        self.assertFalse(plan['run_old_regression_validation'])
        self.assertFalse(plan['resave_latest_state'])

    def test_budget_stop_final_actions_skip_when_full_validation_already_ran(self):
        plan = train_supervised.plan_budget_stop_final_actions(
            stop_due_to_budget=True,
            ran_full_val=True,
            has_full_recent_files=True,
            has_old_regression_files=True,
            old_regression_every_checks=0,
        )

        self.assertFalse(plan['run_full_validation'])
        self.assertFalse(plan['run_old_regression_validation'])
        self.assertFalse(plan['resave_latest_state'])


class TrainSupervisedTurnWeightingTests(unittest.TestCase):
    def test_resolve_turn_weighting_cfg_clamps_invalid_boundaries(self):
        cfg = train_supervised.resolve_turn_weighting_cfg(
            {
                'early_factor': 0.2,
                'mid_factor': 1.1,
                'late_factor': 2.2,
                'early_max_turn': -3,
                'late_min_turn': 0,
            },
            default_early_factor=0.5,
            default_mid_factor=1.0,
            default_late_factor=1.5,
        )

        self.assertEqual(0, cfg['early_max_turn'])
        self.assertEqual(1, cfg['late_min_turn'])
        self.assertAlmostEqual(0.2, cfg['early_factor'])
        self.assertAlmostEqual(1.1, cfg['mid_factor'])
        self.assertAlmostEqual(2.2, cfg['late_factor'])

    def test_compute_turn_bucket_weights_matches_early_mid_late_schedule(self):
        turns = torch.tensor([0, 6, 7, 12, 13, 18], dtype=torch.int64)

        weights = train_supervised.compute_turn_bucket_weights(
            turns,
            early_factor=0.1,
            mid_factor=1.0,
            late_factor=2.5,
            early_max_turn=6,
            late_min_turn=13,
        )

        expected = torch.tensor([0.1, 0.1, 1.0, 1.0, 2.5, 2.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(expected, weights))


class TrainSupervisedExactMetricTests(unittest.TestCase):
    def test_compute_exact_action_metric_stats_renormalizes_chi_slice(self):
        probs = torch.zeros((2, 46), dtype=torch.float32)
        probs[0, 0] = 0.40
        probs[0, 1] = 0.15
        probs[0, 38] = 0.20
        probs[0, 39] = 0.15
        probs[0, 40] = 0.10
        probs[1, 2] = 0.45
        probs[1, 3] = 0.15
        probs[1, 38] = 0.10
        probs[1, 39] = 0.12
        probs[1, 40] = 0.18
        actions = torch.tensor([38, 40], dtype=torch.int64)

        raw_stats = train_supervised.compute_exact_action_metric_stats(
            probs,
            actions,
            start=38,
            end=41,
            topk_size=3,
            normalize_within_slice=False,
        )
        chi_stats = train_supervised.compute_exact_action_metric_stats(
            probs,
            actions,
            start=38,
            end=41,
            topk_size=3,
            normalize_within_slice=True,
        )

        expected_nll = -torch.log(torch.tensor([0.20 / 0.45, 0.18 / 0.40], dtype=torch.float32)).sum().item()

        self.assertEqual(2, int(chi_stats['count'].item()))
        self.assertAlmostEqual(expected_nll, chi_stats['nll_sum'].item(), places=6)
        self.assertEqual(2, int(chi_stats['top1_correct'].item()))
        self.assertEqual(2, int(chi_stats['top3_correct'].item()))
        self.assertEqual(0, int(raw_stats['top1_correct'].item()))
        self.assertGreater(raw_stats['nll_sum'].item(), chi_stats['nll_sum'].item())


if __name__ == '__main__':
    unittest.main()

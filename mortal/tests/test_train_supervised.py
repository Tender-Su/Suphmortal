import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.supervised.train_supervised as train_supervised


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
    def test_external_pause_file_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pause_path = Path(tmp_dir) / 'apex_pause.request'
            with patch.dict(
                train_supervised.os.environ,
                {train_supervised.EXTERNAL_PAUSE_ENV_VAR: str(pause_path)},
                clear=False,
            ):
                self.assertEqual(
                    pause_path.resolve(),
                    train_supervised.resolve_external_pause_file(),
                )
                self.assertFalse(
                    train_supervised.external_pause_requested(pause_path)
                )
                pause_path.write_text('{}', encoding='utf-8')
                self.assertTrue(
                    train_supervised.external_pause_requested(pause_path)
                )

    def test_sanitize_sys_path_for_spawn_keeps_repo_root_first(self):
        original_sys_path = list(sys.path)
        repo_root = str(train_supervised.REPO_ROOT)
        legacy_entries = [
            str(train_supervised.MORTAL_ROOT),
            str(train_supervised.MORTAL_ROOT / 'eval'),
            str(train_supervised.MORTAL_ROOT / 'core'),
            str(train_supervised.REPO_ROOT / 'scripts'),
        ]
        unrelated = str(Path(tempfile.gettempdir()) / 'mahjongai_extra_path')
        try:
            sys.path[:] = [
                legacy_entries[1],
                unrelated,
                repo_root,
                *legacy_entries,
            ]

            train_supervised.sanitize_sys_path_for_spawn()

            self.assertEqual(repo_root, sys.path[0])
            self.assertIn(unrelated, sys.path)
            for entry in legacy_entries:
                self.assertNotIn(entry, sys.path)
            self.assertEqual(1, sys.path.count(repo_root))
        finally:
            sys.path[:] = original_sys_path

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

    def test_missing_init_state_file_errors_immediately(self):
        with self.assertRaisesRegex(FileNotFoundError, r'supervised\.init_state_file does not exist'):
            train_supervised.ensure_init_state_file_exists(
                r'X:\missing\sl_seed.pth',
                cfg_prefix='supervised',
            )

    def test_exact_resume_requires_matching_provenance(self):
        provenance = {'plan_id': 'phase-a'}
        train_supervised.validate_checkpoint_provenance(
            {'run_provenance': provenance},
            provenance,
            cfg_prefix='supervised',
        )

        with self.assertRaisesRegex(RuntimeError, 'provenance mismatch'):
            train_supervised.validate_checkpoint_provenance(
                {'run_provenance': {'plan_id': 'old-phase'}},
                provenance,
                cfg_prefix='supervised',
            )

    def test_phase_init_requires_exact_parent_checkpoint(self):
        provenance = {
            'plan_id': 'phase-b',
            'parent_checkpoint_id': 'phase-a-winner',
        }
        train_supervised.validate_init_checkpoint_identity(
            {'checkpoint_id': 'phase-a-winner'},
            provenance,
            cfg_prefix='supervised',
        )

        with self.assertRaisesRegex(RuntimeError, 'checkpoint mismatch'):
            train_supervised.validate_init_checkpoint_identity(
                {'checkpoint_id': 'different-phase-a'},
                provenance,
                cfg_prefix='supervised',
            )

    def test_atomic_torch_save_replaces_checkpoint_without_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / 'checkpoints' / 'latest.pth'

            train_supervised.atomic_torch_save({'value': 1}, target)
            train_supervised.atomic_torch_save({'value': 2}, target)

            self.assertEqual(2, torch.load(target, weights_only=True)['value'])
            self.assertEqual([], list(target.parent.glob('*.tmp')))

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

    def test_best_loss_checkpoint_improvement_is_stricter_than_patience_delta(self):
        best_loss = 0.4560101177
        small_but_real_improvement = 0.4559505714

        self.assertTrue(
            train_supervised.is_strict_loss_improvement(
                small_but_real_improvement,
                best_loss,
            )
        )
        self.assertFalse(
            train_supervised.is_patience_loss_improvement(
                small_but_real_improvement,
                best_loss,
                min_delta=0.0005,
            )
        )
        self.assertFalse(train_supervised.is_strict_loss_improvement(best_loss, best_loss))

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

    def test_checkpoint_head_flags_use_supervised_section_aux_enable_overrides(self):
        state = {
            'config': {
                'aux': {
                    'opponent_state_weight': 0.0,
                    'danger_enabled': False,
                    'danger_weight': 0.0,
                },
                'supervised': {
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
            config_section='supervised',
        )

        self.assertTrue(flags['opponent_aux_net'])
        self.assertTrue(flags['danger_aux_net'])

    def test_checkpoint_head_flags_use_supervised_section_aux_disable_overrides(self):
        state = {
            'config': {
                'aux': {
                    'opponent_state_weight': 0.25,
                    'danger_enabled': True,
                    'danger_weight': 0.4,
                },
                'supervised': {
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
            config_section='supervised',
        )

        self.assertFalse(flags['opponent_aux_net'])
        self.assertFalse(flags['danger_aux_net'])

    def test_resolve_rl_handoff_state_file_is_disabled(self):
        self.assertFalse(hasattr(train_supervised, 'resolve_rl_handoff_state_file'))

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


class ValidationBoundedRetryTests(unittest.TestCase):
    def test_retry_budget_counts_retries_and_raises_last_original_error(self):
        for max_retries in (None, 0, 1, 2):
            with self.subTest(max_retries=max_retries):
                budget = 1 if max_retries is None else max_retries
                errors = []
                sleeps = []
                empty_cache = Mock()

                def fail_validation():
                    if len(errors) >= budget + 1:
                        self.fail('validation exceeded its retry budget')
                    error = RuntimeError(f'WinError 1455: validation attempt {len(errors)}')
                    errors.append(error)
                    raise error

                kwargs = {} if max_retries is None else {'max_retries': max_retries}
                with (
                    patch.object(train_supervised.logging, 'exception') as log,
                    self.assertRaises(RuntimeError) as raised,
                ):
                    train_supervised.run_with_validation_retries(
                        fail_validation,
                        device_type='cuda',
                        context='bounded retry test',
                        sleep_fn=sleeps.append,
                        empty_cache_fn=empty_cache,
                        **kwargs,
                    )

                self.assertEqual(budget + 1, len(errors))
                self.assertIs(errors[-1], raised.exception)
                self.assertEqual([float(i) for i in range(1, budget + 1)], sleeps)
                self.assertEqual(budget, empty_cache.call_count)
                self.assertEqual(budget, log.call_count)

    def test_success_on_last_allowed_attempt_returns_result(self):
        result = object()
        validate = Mock(side_effect=[
            RuntimeError('WinError 1455: first attempt'),
            RuntimeError('WinError 1455: second attempt'),
            result,
        ])
        sleeps = []
        empty_cache = Mock()
        with self.assertLogs(level='ERROR'):
            actual = train_supervised.run_with_validation_retries(
                validate,
                device_type='cpu',
                context='bounded success test',
                max_retries=2,
                sleep_fn=sleeps.append,
                empty_cache_fn=empty_cache,
            )

        self.assertIs(result, actual)
        self.assertEqual(3, validate.call_count)
        self.assertEqual([1.0, 2.0], sleeps)
        empty_cache.assert_not_called()

    def test_negative_budget_rejected_before_validation_or_cleanup(self):
        validate = Mock()
        sleep = Mock()
        empty_cache = Mock()
        with self.assertRaisesRegex(ValueError, 'max_retries'):
            train_supervised.run_with_validation_retries(
                validate,
                device_type='cuda',
                context='negative budget test',
                max_retries=-1,
                sleep_fn=sleep,
                empty_cache_fn=empty_cache,
            )
        validate.assert_not_called()
        sleep.assert_not_called()
        empty_cache.assert_not_called()

    def test_retry_cleanup_failure_preserves_validation_error_without_retrying(self):
        primary = RuntimeError('WinError 1455: original validation failure')
        cleanup_error = RuntimeError('CUDA error: out of memory during cleanup')
        validate = Mock(side_effect=primary)
        sleep = Mock()
        empty_cache = Mock(side_effect=cleanup_error)
        with self.assertLogs(level='ERROR') as logs, self.assertRaises(RuntimeError) as raised:
            train_supervised.run_with_validation_retries(
                validate,
                device_type='cuda',
                context='retry cleanup failure test',
                max_retries=2,
                sleep_fn=sleep,
                empty_cache_fn=empty_cache,
            )

        self.assertIs(primary, raised.exception)
        validate.assert_called_once_with()
        empty_cache.assert_called_once_with()
        sleep.assert_not_called()
        self.assertEqual(1, len(logs.records))
        self.assertIs(cleanup_error, logs.records[0].exc_info[1])


class ValidationBrainMicrobatchTests(unittest.TestCase):
    def test_group_norm_matches_full_logical_batch(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = torch.nn.Sequential(
                torch.nn.Conv1d(4, 8, kernel_size=3, padding=1),
                torch.nn.GroupNorm(2, 8),
                torch.nn.Mish(),
                torch.nn.Flatten(),
                torch.nn.Linear(8 * 5, 7),
            ).eval()
            obs = torch.randn(11, 4, 5, dtype=torch.float64)

        with torch.inference_mode():
            expected = model(obs.float())
            actual = train_supervised.forward_validation_brain(
                model, obs, device=torch.device('cpu'), microbatch_size=4,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual((11, 7), actual.shape)
        self.assertEqual(torch.float32, actual.dtype)
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(actual.grad_fn)
        self.assertFalse(model.training)

    def test_eval_batch_norm_matches_full_batch_without_updating_buffers(self):
        model = torch.nn.BatchNorm1d(4).eval()
        with torch.no_grad():
            model.running_mean.copy_(torch.tensor([0.2, -0.5, 1.0, 0.8]))
            model.running_var.copy_(torch.tensor([0.5, 2.0, 1.5, 3.0]))
        buffers_before = {
            name: value.clone() for name, value in model.named_buffers()
        }
        obs = torch.arange(11 * 4 * 5, dtype=torch.float64).reshape(11, 4, 5) / 20

        with torch.inference_mode():
            expected = model(obs.float())
            actual = train_supervised.forward_validation_brain(
                model, obs, device=torch.device('cpu'), microbatch_size=4,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(buffers_before[name], value), name)

    def test_microbatch_sizes_cover_each_sample_once_in_order(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, value):
                self.inputs.append(value.clone())
                return value * 2 + 1

        cases = (
            (1, 1, [1]),
            (3, 8, [3]),
            (8, 4, [4, 4]),
            (11, 4, [4, 4, 3]),
            (513, None, [256, 256, 1]),
        )
        for size, microbatch_size, expected_sizes in cases:
            with self.subTest(size=size, microbatch_size=microbatch_size):
                model = RecordingModel().eval()
                obs = torch.arange(size * 2, dtype=torch.float64).reshape(size, 2)
                before = obs.clone()
                kwargs = {} if microbatch_size is None else {
                    'microbatch_size': microbatch_size,
                }
                with torch.inference_mode():
                    actual = train_supervised.forward_validation_brain(
                        model, obs, device=torch.device('cpu'), **kwargs,
                    )

                self.assertEqual(expected_sizes, [len(value) for value in model.inputs])
                for value in model.inputs:
                    self.assertEqual(torch.float32, value.dtype)
                    self.assertEqual('cpu', value.device.type)
                self.assertTrue(torch.equal(torch.cat(model.inputs), obs.float()))
                self.assertTrue(torch.equal(actual, obs.float() * 2 + 1))
                self.assertTrue(torch.equal(obs, before))

    def test_chunk_inputs_die_before_next_transfer_and_outputs_after_concat(self):
        input_refs = []
        output_refs = []
        transfers = []
        original_to = torch.Tensor.to

        class TrackingModel(torch.nn.Module):
            def forward(self, value):
                input_refs.append(weakref.ref(value))
                result = value.sum(dim=1, keepdim=True)
                output_refs.append(weakref.ref(result))
                return result

        def tracked_to(value, *args, **kwargs):
            self.assertTrue(all(ref() is None for ref in input_refs))
            transfers.append(value.shape[0])
            return original_to(value, *args, **kwargs)

        # Float64 CPU input forces a fresh float32 allocation without using CUDA.
        obs = torch.arange(18, dtype=torch.float64).reshape(9, 2)
        model = TrackingModel().eval()
        with torch.inference_mode(), patch.object(torch.Tensor, 'to', new=tracked_to):
            actual = train_supervised.forward_validation_brain(
                model, obs, device=torch.device('cpu'), microbatch_size=4,
            )

        self.assertEqual([4, 4, 1], transfers)
        self.assertEqual(3, len(input_refs))
        self.assertTrue(all(ref() is None for ref in input_refs))
        self.assertTrue(all(ref() is None for ref in output_refs))
        self.assertTrue(torch.equal(actual, obs.float().sum(dim=1, keepdim=True)))

    def test_requires_inference_mode_even_inside_no_grad(self):
        model = torch.nn.Identity().eval()
        obs = torch.ones(2, 4)
        for context in (torch.enable_grad, torch.no_grad):
            with self.subTest(context=context.__name__), context():
                with patch.object(model, 'forward') as forward:
                    with self.assertRaisesRegex(RuntimeError, 'inference_mode'):
                        train_supervised.forward_validation_brain(
                            model, obs, device=torch.device('cpu'),
                        )
                    forward.assert_not_called()

    def test_rejects_training_model_without_changing_its_mode(self):
        model = torch.nn.Identity().train()
        with torch.inference_mode(), patch.object(model, 'forward') as forward:
            with self.assertRaisesRegex(RuntimeError, 'eval'):
                train_supervised.forward_validation_brain(
                    model, torch.ones(2, 4), device=torch.device('cpu'),
                )
        forward.assert_not_called()
        self.assertTrue(model.training)

    def test_rejects_empty_batch_and_nonpositive_microbatch_size(self):
        for size, microbatch_size in ((0, 256), (2, 0), (2, -1)):
            with self.subTest(size=size, microbatch_size=microbatch_size):
                model = torch.nn.Identity().eval()
                with torch.inference_mode(), patch.object(model, 'forward') as forward:
                    with self.assertRaises(ValueError):
                        train_supervised.forward_validation_brain(
                            model, torch.ones(size, 4), device=torch.device('cpu'),
                            microbatch_size=microbatch_size,
                        )
                forward.assert_not_called()


class ValidationResourceCleanupTests(unittest.TestCase):
    def test_cleanup_order_and_cpu_skips_cuda_cache(self):
        for device_type in ('cpu', 'cuda'):
            with self.subTest(device_type=device_type):
                events = []
                with (
                    patch.object(train_supervised.gc, 'collect', side_effect=lambda: events.append('gc')),
                    patch.object(train_supervised.torch.cuda, 'empty_cache', side_effect=lambda: events.append('cache')),
                ):
                    train_supervised.cleanup_validation_resources(
                        lambda: events.append('close'), device_type=device_type,
                    )
                expected = ['close', 'gc'] + (['cache'] if device_type == 'cuda' else [])
                self.assertEqual(expected, events)

    def test_all_cleanup_actions_run_and_first_failure_is_raised_and_logged(self):
        for failed_actions in (('close',), ('gc',), ('cache',), ('close', 'gc', 'cache')):
            with self.subTest(failed_actions=failed_actions):
                events = []
                errors = {name: RuntimeError(f'{name} failed') for name in failed_actions}

                def action(name):
                    events.append(name)
                    if name in errors:
                        raise errors[name]

                with (
                    patch.object(train_supervised.gc, 'collect', side_effect=lambda: action('gc')),
                    patch.object(train_supervised.torch.cuda, 'empty_cache', side_effect=lambda: action('cache')),
                    self.assertLogs(level='ERROR') as logs,
                    self.assertRaises(RuntimeError) as raised,
                ):
                    train_supervised.cleanup_validation_resources(
                        lambda: action('close'), device_type='cuda',
                    )

                self.assertIs(errors[failed_actions[0]], raised.exception)
                self.assertEqual(['close', 'gc', 'cache'], events)
                self.assertEqual(len(failed_actions), len(logs.records))
                self.assertEqual(
                    [errors[name] for name in failed_actions],
                    [record.exc_info[1] for record in logs.records],
                )

    def test_active_original_exception_survives_all_cleanup_failures(self):
        for primary in (RuntimeError('forward CUDA OOM'), SystemExit(75)):
            with self.subTest(primary=type(primary).__name__):
                events = []

                def fail(name):
                    events.append(name)
                    raise RuntimeError(f'{name} cleanup failed')

                with (
                    patch.object(train_supervised.gc, 'collect', side_effect=lambda: fail('gc')),
                    patch.object(train_supervised.torch.cuda, 'empty_cache', side_effect=lambda: fail('cache')),
                    self.assertLogs(level='ERROR') as logs,
                    self.assertRaises(type(primary)) as raised,
                ):
                    try:
                        raise primary
                    finally:
                        train_supervised.cleanup_validation_resources(
                            lambda: fail('close'), device_type='cuda',
                        )

                self.assertIs(primary, raised.exception)
                self.assertEqual(['close', 'gc', 'cache'], events)
                self.assertEqual(3, len(logs.records))
                if isinstance(primary, SystemExit):
                    self.assertEqual(75, raised.exception.code)

    def test_successful_cleanup_preserves_active_exception(self):
        primary = RuntimeError('original failure')
        with (
            patch.object(train_supervised.gc, 'collect') as collect,
            patch.object(train_supervised.torch.cuda, 'empty_cache') as empty_cache,
            self.assertRaises(RuntimeError) as raised,
        ):
            try:
                raise primary
            finally:
                train_supervised.cleanup_validation_resources(
                    lambda: None, device_type='cpu',
                )
        self.assertIs(primary, raised.exception)
        collect.assert_called_once_with()
        empty_cache.assert_not_called()


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

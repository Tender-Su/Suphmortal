import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_fidelity as fidelity
import run_stage05_loader_ab as loader_ab


class LoaderAbConfigTest(unittest.TestCase):
    def test_coarse_train_configs_focus_on_current_nw4_winner_and_nearby_correlated_controls(self):
        names = {cfg.name for cfg in loader_ab.coarse_train_configs()}
        self.assertIn('nw4_fb8_pf3_vfb8_vpf5_bs1024', names)
        self.assertIn('nw4_fb9_pf3_vfb8_vpf5_bs1024', names)
        self.assertIn('nw4_fb10_pf3_vfb8_vpf5_bs1024', names)
        self.assertIn('nw4_fb10_pf4_vfb8_vpf5_bs1024', names)
        self.assertIn('nw4_fb11_pf3_vfb8_vpf5_bs1024', names)
        self.assertIn('nw6_fb7_pf3_vfb8_vpf5_bs1024', names)
        self.assertIn('nw6_fb8_pf3_vfb8_vpf5_bs1024', names)
        self.assertIn('nw3_fb13_pf3_vfb8_vpf5_bs1024', names)
        self.assertNotIn('nw8_fb12_pf4_vfb8_vpf5_bs1024', names)
        self.assertNotIn('nw4_fb7_pf3_vfb8_vpf5_bs1024', names)
        self.assertNotIn('nw6_fb4_pf2_vfb8_vpf5_bs1024', names)
        self.assertNotIn('nw6_fb10_pf3_vfb8_vpf5_bs1024', names)

    def test_choose_best_stable_prefers_fastest_stable_result(self):
        best = loader_ab.choose_best_stable(
            [
                {'stable': False, 'total_steps_per_sec': 9.0, 'total_runtime_sec': 100.0},
                {'stable': True, 'total_steps_per_sec': 5.0, 'total_runtime_sec': 100.0},
                {'stable': True, 'total_steps_per_sec': 6.0, 'total_runtime_sec': 110.0},
            ]
        )
        self.assertIsNotNone(best)
        self.assertEqual(6.0, best['total_steps_per_sec'])

    def test_top_stable_results_limits_and_orders(self):
        top = loader_ab.top_stable_results(
            [
                {'stable': True, 'total_steps_per_sec': 5.0, 'total_runtime_sec': 100.0},
                {'stable': True, 'total_steps_per_sec': 6.0, 'total_runtime_sec': 110.0},
                {'stable': True, 'total_steps_per_sec': 6.0, 'total_runtime_sec': 90.0},
                {'stable': False, 'total_steps_per_sec': 9.0, 'total_runtime_sec': 50.0},
            ],
            2,
        )
        self.assertEqual(2, len(top))
        self.assertEqual(90.0, top[0]['total_runtime_sec'])
        self.assertEqual(110.0, top[1]['total_runtime_sec'])

    def test_validation_configs_are_deduped(self):
        base = loader_ab.make_loader_config(
            num_workers=6,
            file_batch_size=8,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
        )
        validate = loader_ab.validation_configs(base)
        self.assertEqual(len(validate), len({cfg.name for cfg in validate}))

    def test_loader_benchmark_inputs_signature_changes_when_benchmark_inputs_change(self):
        eval_splits = {
            'monitor_recent_files': ['monitor_a.json.gz'],
            'full_recent_files': ['recent_a.json.gz'],
            'old_regression_files': ['old_a.json.gz'],
        }
        base_signature = loader_ab.loader_benchmark_inputs_signature(
            base_cfg={'optim': {'lr': 0.001}},
            grouped={'202401': ['a.json.gz']},
            eval_splits=eval_splits,
        )

        self.assertNotEqual(
            base_signature,
            loader_ab.loader_benchmark_inputs_signature(
                base_cfg={'optim': {'lr': 0.002}},
                grouped={'202401': ['a.json.gz']},
                eval_splits=eval_splits,
            ),
        )
        self.assertNotEqual(
            base_signature,
            loader_ab.loader_benchmark_inputs_signature(
                base_cfg={'optim': {'lr': 0.001}},
                grouped={'202401': ['a.json.gz'], '202402': ['b.json.gz']},
                eval_splits=eval_splits,
            ),
        )
        self.assertNotEqual(
            base_signature,
            loader_ab.loader_benchmark_inputs_signature(
                base_cfg={'optim': {'lr': 0.001}},
                grouped={'202401': ['a.json.gz']},
                eval_splits={
                    'monitor_recent_files': ['monitor_a.json.gz'],
                    'full_recent_files': ['recent_b.json.gz'],
                    'old_regression_files': ['old_a.json.gz'],
                },
            ),
        )

    def test_run_loader_config_invalidates_cache_when_benchmark_inputs_signature_changes(self):
        candidate = fidelity.CandidateSpec(
            arm_name='candidate_arm',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='mild',
            window_profile='24m_12m',
            cfg_overrides={'optim': {'lr': 0.001}},
            meta={},
        )
        config = loader_ab.make_loader_config(
            num_workers=4,
            file_batch_size=10,
            prefetch_factor=3,
            val_file_batch_size=8,
            val_prefetch_factor=5,
        )
        eval_splits = {
            'monitor_recent_files': ['monitor_a.json.gz'],
            'full_recent_files': ['recent_a.json.gz'],
            'old_regression_files': ['old_a.json.gz'],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            loader_root = Path(tmp_dir)
            run_phase_calls: list[dict] = []

            def fake_run_phase(*args, **kwargs):
                call_index = len(run_phase_calls)
                run_phase_calls.append({'args': args, 'kwargs': kwargs})
                phase_dir = loader_root / f'phase_{call_index}'
                log_path = phase_dir / 'train.log'
                cfg_path = phase_dir / 'config.toml'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text('', encoding='utf-8', newline='\n')
                cfg_path.write_text('', encoding='utf-8', newline='\n')
                return {
                    'latest': {'optimizer_steps': 10},
                    'log_path': str(log_path),
                    'config_path': str(cfg_path),
                }

            with (
                patch.object(loader_ab, 'LOADER_AB_ROOT', loader_root),
                patch.object(loader_ab.ab, 'run_phase', side_effect=fake_run_phase),
                patch.object(loader_ab, 'count_retry_markers', return_value=0),
                patch.object(loader_ab.time, 'perf_counter', side_effect=[1.0, 3.0, 4.0, 7.0]),
            ):
                base_cfg_a = {'optim': {'lr': 0.001}}
                signature_a = loader_ab.loader_benchmark_inputs_signature(
                    base_cfg=base_cfg_a,
                    grouped={'202401': ['a.json.gz']},
                    eval_splits=eval_splits,
                )
                first = loader_ab.run_loader_config(
                    suite_name='suite_a',
                    round_name='train_scan',
                    config=config,
                    base_cfg=base_cfg_a,
                    grouped={'202401': ['a.json.gz']},
                    eval_splits=eval_splits,
                    benchmark_inputs_signature=signature_a,
                    candidate=candidate,
                    seed=123,
                    step_scale=0.25,
                    val_every_steps=750,
                    monitor_val_batches=64,
                    full_recent_files=64,
                    old_regression_files=32,
                    phase_name='phase_a',
                )
                cached = loader_ab.run_loader_config(
                    suite_name='suite_a',
                    round_name='train_scan',
                    config=config,
                    base_cfg=base_cfg_a,
                    grouped={'202401': ['a.json.gz']},
                    eval_splits=eval_splits,
                    benchmark_inputs_signature=signature_a,
                    candidate=candidate,
                    seed=123,
                    step_scale=0.25,
                    val_every_steps=750,
                    monitor_val_batches=64,
                    full_recent_files=64,
                    old_regression_files=32,
                    phase_name='phase_a',
                )

                base_cfg_b = {'optim': {'lr': 0.002}}
                signature_b = loader_ab.loader_benchmark_inputs_signature(
                    base_cfg=base_cfg_b,
                    grouped={'202401': ['a.json.gz']},
                    eval_splits=eval_splits,
                )
                rerun = loader_ab.run_loader_config(
                    suite_name='suite_a',
                    round_name='train_scan',
                    config=config,
                    base_cfg=base_cfg_b,
                    grouped={'202401': ['a.json.gz']},
                    eval_splits=eval_splits,
                    benchmark_inputs_signature=signature_b,
                    candidate=candidate,
                    seed=123,
                    step_scale=0.25,
                    val_every_steps=750,
                    monitor_val_batches=64,
                    full_recent_files=64,
                    old_regression_files=32,
                    phase_name='phase_a',
                )

        self.assertEqual(2, len(run_phase_calls))
        self.assertEqual(first['signature'], cached['signature'])
        self.assertEqual(signature_a, first['benchmark_inputs_signature'])
        self.assertNotEqual(first['signature'], rerun['signature'])
        self.assertEqual(signature_b, rerun['benchmark_inputs_signature'])


if __name__ == '__main__':
    unittest.main()

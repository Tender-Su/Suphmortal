import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_train_throughput_once
import run_stage05_loader_ab as loader_ab


class RunTrainThroughputOnceTests(unittest.TestCase):
    def test_parse_args_uses_current_validation_loader_defaults(self):
        with patch.object(
            sys,
            'argv',
            [
                'run_train_throughput_once.py',
                '--run-name', 'demo_run',
                '--suite-name', 'demo_suite',
                '--round-name', 'demo_round',
                '--num-workers', '4',
                '--file-batch-size', '10',
                '--prefetch-factor', '3',
            ],
        ):
            args = run_train_throughput_once.parse_args()

        self.assertEqual(loader_ab.DEFAULT_VAL_FILE_BATCH_SIZE, args.val_file_batch_size)
        self.assertEqual(loader_ab.DEFAULT_VAL_PREFETCH_FACTOR, args.val_prefetch_factor)

    def test_main_passes_loader_benchmark_inputs_to_run_loader_config(self):
        fake_candidate = object()
        fake_config = object()
        fake_result = {
            'config': {'name': 'nw4_fb10_pf3'},
            'candidate': {'arm_name': 'candidate'},
            'total_runtime_sec': 10.0,
            'total_steps': 100,
            'total_steps_per_sec': 10.0,
            'stable': True,
            'total_retry_count': 0,
            'phase_timings': {'phase_a': {'log_path': 'train.log', 'config_path': 'config.toml'}},
        }

        with patch.object(
            sys,
            'argv',
            [
                'run_train_throughput_once.py',
                '--run-name', 'demo_run',
                '--suite-name', 'demo_suite',
                '--round-name', 'demo_round',
                '--num-workers', '4',
                '--file-batch-size', '10',
                '--prefetch-factor', '3',
            ],
        ), patch(
            'run_train_throughput_once.ab.build_base_config',
            return_value={'base': 'cfg'},
        ), patch(
            'run_train_throughput_once.ab.load_all_files',
            return_value=[],
        ), patch(
            'run_train_throughput_once.ab.group_files_by_month',
            return_value={'202601': ['a.json.gz']},
        ), patch(
            'run_train_throughput_once.loader_ab.loader_benchmark_inputs_signature',
            return_value='sig123',
        ) as signature_mock, patch(
            'run_train_throughput_once.loader_ab.load_reference_candidate',
            return_value=fake_candidate,
        ), patch(
            'run_train_throughput_once.loader_ab.make_loader_config',
            return_value=fake_config,
        ) as make_loader_mock, patch(
            'run_train_throughput_once.loader_ab.run_loader_config',
            return_value=fake_result,
        ) as run_loader_mock, patch(
            'run_train_throughput_once.loader_ab.LOADER_AB_ROOT',
            Path('C:/virtual/loader_ab'),
        ), patch(
            'run_train_throughput_once.loader_ab.fidelity.atomic_write_json',
        ), patch(
            'sys.stdout',
            new_callable=io.StringIO,
        ) as stdout:
            run_train_throughput_once.main()

        signature_mock.assert_called_once_with(
            base_cfg={'base': 'cfg'},
            grouped={'202601': ['a.json.gz']},
            eval_splits={
                'monitor_recent_files': [],
                'full_recent_files': [],
                'old_regression_files': [],
            },
        )
        make_loader_mock.assert_called_once_with(
            num_workers=4,
            file_batch_size=10,
            prefetch_factor=3,
            val_file_batch_size=loader_ab.DEFAULT_VAL_FILE_BATCH_SIZE,
            val_prefetch_factor=loader_ab.DEFAULT_VAL_PREFETCH_FACTOR,
            batch_size=1024,
        )
        run_loader_mock.assert_called_once_with(
            suite_name='demo_suite',
            round_name='demo_round',
            config=fake_config,
            base_cfg={'base': 'cfg'},
            grouped={'202601': ['a.json.gz']},
            eval_splits={
                'monitor_recent_files': [],
                'full_recent_files': [],
                'old_regression_files': [],
            },
            benchmark_inputs_signature='sig123',
            candidate=fake_candidate,
            seed=20260326,
            step_scale=1.0,
            val_every_steps=0,
            monitor_val_batches=0,
            full_recent_files=0,
            old_regression_files=0,
            phase_name='phase_a',
        )
        payload = json.loads(stdout.getvalue())
        self.assertEqual('demo_suite', payload['suite_name'])
        self.assertEqual(10.0, payload['total_steps_per_sec'])


if __name__ == '__main__':
    unittest.main()

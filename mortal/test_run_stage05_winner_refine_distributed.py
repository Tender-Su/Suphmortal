from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import distributed_dispatch as dispatch_module
import run_stage05_fidelity as fidelity
import run_stage05_winner_refine_distributed as distributed


def make_candidate(name: str) -> fidelity.CandidateSpec:
    return fidelity.CandidateSpec(
        arm_name=name,
        scheduler_profile='cosine',
        curriculum_profile='broad_to_recent',
        weight_profile='strong',
        window_profile='24m_12m',
        cfg_overrides={},
        meta={
            'protocol_arm': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
            'aux_family': 'all_three',
        },
    )


def make_ranking_entry(
    name: str,
    *,
    selection_quality_score: float,
    comparison_recent_loss: float,
    eligible: bool = True,
    valid: bool = True,
) -> dict:
    return {
        'arm_name': name,
        'valid': valid,
        'eligible': eligible,
        'selection_quality_score': selection_quality_score,
        'comparison_recent_loss': comparison_recent_loss,
        'recent_policy_loss': comparison_recent_loss,
        'candidate_meta': {
            'protocol_arm': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
            'aux_family': 'all_three',
        },
    }


class WinnerRefineDistributedTests(unittest.TestCase):
    def test_initialize_dispatch_control_state_sets_remote_launch_mode(self):
        state = distributed.initialize_dispatch_control_state(
            local_label='desktop',
            remote_label='laptop',
            remote_launch_mode='interactive_window',
        )

        self.assertFalse(state['workers']['desktop']['paused'])
        self.assertEqual('interactive_window', state['workers']['laptop']['launch_mode'])

    def test_set_worker_pause_can_request_interrupt(self):
        state = distributed.initialize_dispatch_control_state(
            local_label='desktop',
            remote_label='laptop',
            remote_launch_mode='interactive_window',
        )

        entry = distributed.set_worker_pause(
            state,
            worker_label='laptop',
            paused=True,
            stop_active=True,
        )

        self.assertTrue(entry['paused'])
        self.assertTrue(entry['interrupt_requested'])

        entry = distributed.set_worker_pause(
            state,
            worker_label='laptop',
            paused=False,
        )

        self.assertFalse(entry['paused'])
        self.assertFalse(entry['interrupt_requested'])

    def test_reset_task_after_operator_interrupt_requeues_without_consuming_retry_budget(self):
        task_state = {
            'status': 'running',
            'attempts': 2,
            'started_at': '2026-03-31 10:00:00',
            'worker_label': 'laptop',
            'local_result_path': 'result.json',
            'remote_result_path': 'remote.json',
            'log_path': 'task.log',
        }

        distributed.reset_task_after_operator_interrupt(
            task_state,
            note='worker `laptop` paused by operator',
        )

        self.assertEqual('pending', task_state['status'])
        self.assertEqual(1, task_state['attempts'])
        self.assertIn('paused by operator', task_state['error'])
        self.assertNotIn('started_at', task_state)
        self.assertNotIn('worker_label', task_state)

    def test_patched_base_screening_restores_defaults(self):
        original_num_workers = distributed.ab.BASE_SCREENING['num_workers']
        original_file_batch_size = distributed.ab.BASE_SCREENING['file_batch_size']

        with distributed.patched_base_screening({'num_workers': 9, 'file_batch_size': 13}):
            self.assertEqual(9, distributed.ab.BASE_SCREENING['num_workers'])
            self.assertEqual(13, distributed.ab.BASE_SCREENING['file_batch_size'])

        self.assertEqual(original_num_workers, distributed.ab.BASE_SCREENING['num_workers'])
        self.assertEqual(original_file_batch_size, distributed.ab.BASE_SCREENING['file_batch_size'])

    def test_build_remote_python_command_preserves_mortal_relative_path(self):
        worker = dispatch_module.WorkerSpec(
            kind='remote',
            label='laptop',
            python=r'C:\Python\python.exe',
            host='mahjong-laptop',
            repo=r'C:\Users\numbe\Desktop\MahjongAI',
            ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
        )

        command = dispatch_module.build_remote_python_command(
            worker=worker,
            script_path=Path(r'C:\Users\numbe\Desktop\MahjongAI\mortal\run_stage05_winner_refine_distributed.py'),
            remote_result_path=Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\result.json'),
            command_args=['run-task', '--run-name', 'demo'],
        )

        self.assertEqual('ssh', command[0])
        self.assertIn(
            r"C:\Users\numbe\Desktop\MahjongAI\mortal\run_stage05_winner_refine_distributed.py",
            command[-1],
        )

    def test_build_remote_interactive_window_command_uses_helper_script(self):
        worker = dispatch_module.WorkerSpec(
            kind='remote',
            label='laptop',
            python=r'C:\Python\python.exe',
            host='mahjong-laptop',
            repo=r'C:\Users\numbe\Desktop\MahjongAI',
            ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
        )
        task_state = {
            'task_id': 'seed1__s1__demo_arm',
            'candidate_arm': 'demo_arm',
            'seed': 1,
        }

        command = distributed.build_remote_interactive_window_command(
            worker=worker,
            run_name='demo_run',
            task_state=task_state,
            remote_result_path=Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\result.json'),
            remote_runtime_root=Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\runtime\seed1__s1__demo_arm'),
            screening_overrides={
                'num_workers': 4,
                'file_batch_size': 10,
                'prefetch_factor': 4,
                'val_file_batch_size': 7,
                'val_prefetch_factor': 5,
            },
        )

        self.assertEqual('ssh', command[0])
        self.assertIn(
            r"C:\Users\numbe\Desktop\MahjongAI\scripts\start_interactive_remote_python.ps1",
            command[-1],
        )
        self.assertIn('-PythonArgsBase64', command[-1])

    def test_handle_finished_json_task_retries_remote_fetch_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_result_path = Path(tmp_dir) / 'result.json'
            task_state = {
                'status': 'running',
                'attempts': 1,
            }
            active = dispatch_module.ActiveTask(
                worker=dispatch_module.WorkerSpec(
                    kind='remote',
                    label='laptop',
                    python='python',
                    host='mahjong-laptop',
                ),
                stage_name='seed1',
                task_id='task_a',
                task_state=task_state,
                process=type('Proc', (), {'returncode': 0})(),
                log_path=Path(tmp_dir) / 'task.log',
                local_result_path=local_result_path,
                remote_result_path=Path(r'C:\remote\result.json'),
            )

            with patch.object(
                dispatch_module,
                'fetch_remote_result',
                side_effect=RuntimeError('scp failed'),
            ):
                dispatch_module.handle_finished_json_task(
                    active=active,
                    max_attempts=2,
                    finished_at='2026-03-31 10:00:00',
                    validate_result=lambda path: None,
                )

            self.assertEqual('pending', task_state['status'])
            self.assertIn('result handling failed', task_state['error'])

    def test_handle_finished_json_task_retries_validate_failure_and_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_result_path = Path(tmp_dir) / 'result.json'
            local_result_path.write_text('partial', encoding='utf-8')
            task_state = {
                'status': 'running',
                'attempts': 1,
            }
            active = dispatch_module.ActiveTask(
                worker=dispatch_module.WorkerSpec(
                    kind='local',
                    label='desktop',
                    python='python',
                ),
                stage_name='seed1',
                task_id='task_b',
                task_state=task_state,
                process=type('Proc', (), {'returncode': 0})(),
                log_path=Path(tmp_dir) / 'task.log',
                local_result_path=local_result_path,
                remote_result_path=None,
            )

            dispatch_module.handle_finished_json_task(
                active=active,
                max_attempts=2,
                finished_at='2026-03-31 10:00:00',
                validate_result=lambda path: (_ for _ in ()).throw(ValueError('bad json')),
            )

            self.assertEqual('pending', task_state['status'])
            self.assertIn('bad json', task_state['error'])
            self.assertFalse(local_result_path.exists())

    def test_load_task_result_rejects_unsuccessful_training_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / 'result.json'
            result_path.write_text(
                json.dumps(
                    {
                        'summary': {
                            'ok': False,
                            'valid': False,
                            'error': 'train_supervised entrypoint failed',
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(RuntimeError, 'reported training failure'):
                distributed.load_task_result(result_path)

    def test_select_winner_refine_seed2_candidates_keeps_floor_and_gap_band(self):
        candidates = [make_candidate(name) for name in ('a', 'b', 'c', 'd')]
        ranking = [
            make_ranking_entry('a', selection_quality_score=0.5500, comparison_recent_loss=0.3000),
            make_ranking_entry('b', selection_quality_score=0.5497, comparison_recent_loss=0.3002),
            make_ranking_entry('c', selection_quality_score=0.5492, comparison_recent_loss=0.3005),
            make_ranking_entry('d', selection_quality_score=0.5475, comparison_recent_loss=0.3012),
        ]

        selected, details = distributed.select_winner_refine_seed2_candidates(
            ranking,
            candidates,
            min_keep=2,
            selection_gap=0.001,
            max_keep=9,
        )

        self.assertEqual(['a', 'b', 'c'], [candidate.arm_name for candidate in selected])
        self.assertEqual('eligible_then_selection_gap', details['mode'])
        self.assertEqual(3, details['candidate_count'])

    def test_select_winner_refine_seed2_candidates_prefers_eligible_pool(self):
        candidates = [make_candidate(name) for name in ('a', 'b', 'c')]
        ranking = [
            make_ranking_entry('a', selection_quality_score=0.5500, comparison_recent_loss=0.3000, eligible=True),
            make_ranking_entry('b', selection_quality_score=0.5494, comparison_recent_loss=0.3006, eligible=True),
            make_ranking_entry('c', selection_quality_score=0.5498, comparison_recent_loss=0.3003, eligible=False),
        ]

        selected, details = distributed.select_winner_refine_seed2_candidates(
            ranking,
            candidates,
            min_keep=2,
            selection_gap=0.001,
            max_keep=9,
        )

        self.assertEqual(['a', 'b'], [candidate.arm_name for candidate in selected])
        self.assertEqual('eligible', details['pool_mode'])

    def test_reset_running_tasks_for_resume_requeues_inflight_tasks(self):
        state = {
            'seed1': {
                'tasks': {
                    't1': {
                        'status': 'running',
                        'started_at': '2026-03-31 01:00:00',
                        'worker_label': 'desktop',
                        'local_result_path': 'x.json',
                        'remote_result_path': 'y.json',
                        'log_path': 'z.log',
                    }
                }
            },
            'seed2': {
                'tasks': {
                    't2': {
                        'status': 'completed',
                    }
                }
            },
        }

        distributed.reset_running_tasks_for_resume(state)

        self.assertEqual('pending', state['seed1']['tasks']['t1']['status'])
        self.assertNotIn('started_at', state['seed1']['tasks']['t1'])
        self.assertNotIn('worker_label', state['seed1']['tasks']['t1'])
        self.assertEqual('completed', state['seed2']['tasks']['t2']['status'])

    def test_update_run_state_for_dispatch_marks_stopped_after_refine(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'running_p1_winner_refine',
                        'p1': {},
                        'final_conclusion': {},
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            dispatch_state = {
                'stage': 'completed',
                'status': 'completed',
                'seed1': {'candidate_count': 27},
                'seed2': {'candidate_count': 5},
                'seed2_selector': {'selected_arm_names': ['arm_a', 'arm_b']},
                'winner_refine_centers': ['center_a'],
                'local_label': 'desktop',
                'remote_label': 'laptop',
            }
            final_round = {
                'round_name': 'p1_winner_refine_round',
                'ranking': [
                    {
                        'arm_name': 'arm_a',
                        'valid': True,
                    }
                ],
            }
            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                    front_runner='arm_a',
                    final_round=final_round,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual('stopped_after_p1_winner_refine', persisted['status'])
            self.assertEqual('arm_a', persisted['final_conclusion']['p1_refine_front_runner'])
            self.assertEqual(final_round, persisted['p1']['winner_refine_round'])
            self.assertEqual(['center_a'], persisted['p1']['winner_refine_centers'])

    def test_update_run_state_for_dispatch_accepts_seed2_not_initialized_yet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'stopped_after_p1_protocol_decide',
                        'p1': {},
                        'final_conclusion': {},
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            dispatch_state = {
                'stage': 'seed1',
                'status': 'running',
                'seed1': {'candidate_count': 27},
                'seed2': None,
                'seed2_selector': None,
                'winner_refine_centers': ['center_a'],
                'local_label': 'desktop',
                'remote_label': 'laptop',
            }
            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            dispatch_summary = persisted['p1']['winner_refine_dispatch']
            self.assertEqual('running', dispatch_summary['status'])
            self.assertEqual(27, dispatch_summary['seed1_candidate_count'])
            self.assertIsNone(dispatch_summary['seed2_candidate_count'])
            self.assertEqual([], dispatch_summary['seed2_selected_arm_names'])

    def test_load_refine_context_recovers_seed_from_round_payload_when_top_level_seed_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'winner_refine_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            custom_seed = 987654
            state = {
                'status': 'stopped_after_p1_protocol_decide',
                'selected_protocol_arms': [fidelity.P1_WINNER_REFINE_PROTOCOL_ARM],
                'p1': {
                    'protocol_arms': [fidelity.P1_WINNER_REFINE_PROTOCOL_ARM],
                    'calibration': {'dummy': True},
                    'protocol_decide_round': {
                        'seed': custom_seed + 505,
                        'ranking': [],
                    },
                    'selected_protocol_arm': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
                },
                'final_conclusion': {
                    'p1_protocol_winner': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(distributed.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(distributed.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(distributed.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    distributed.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(
                    distributed.p1_only,
                    'build_protocol_candidates',
                    return_value=[make_candidate(fidelity.P1_WINNER_REFINE_PROTOCOL_ARM)],
                ),
                patch.object(
                    distributed.fidelity,
                    'current_p1_winner_refine_explicit_center_arm_names',
                    return_value=['center_a'],
                ),
                patch.object(
                    distributed.p1_only,
                    'select_protocol_centers',
                    return_value=[make_candidate('center_a')],
                ),
                patch.object(
                    distributed.fidelity,
                    'build_p1_winner_refine_candidates',
                    return_value=[make_candidate('cand_a')],
                ),
            ):
                ctx = distributed.load_refine_context(run_dir)

            self.assertEqual(custom_seed, ctx['seed'])
            self.assertEqual(custom_seed + 606, ctx['seed_base'])


if __name__ == '__main__':
    unittest.main()

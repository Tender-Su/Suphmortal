from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from io import StringIO
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
    def test_dispatch_root_for_protocol_decide_uses_separate_directory(self):
        run_dir = Path(r'C:\tmp\demo_run')

        path = distributed.dispatch_root_for_run(run_dir, distributed.ROUND_KIND_PROTOCOL_DECIDE)

        self.assertEqual(run_dir / 'distributed' / 'protocol_decide_dispatch', path)

    def test_dispatch_root_for_ablation_uses_separate_directory(self):
        run_dir = Path(r'C:\tmp\demo_run')

        path = distributed.dispatch_root_for_run(run_dir, distributed.ROUND_KIND_ABLATION)

        self.assertEqual(run_dir / 'distributed' / 'ablation_dispatch', path)

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

    def test_update_run_state_for_protocol_decide_marks_stopped_after_protocol_decide(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'running_p1_protocol_decide',
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
                'seed1': {'candidate_count': 33},
                'seed2': {'candidate_count': 14},
                'seed2_plan': {
                    'probe_candidate_names': ['arm_a', 'arm_b'],
                    'decision_candidate_names': ['arm_a', 'arm_b', 'arm_c'],
                    'expanded_groups': ['proto_a'],
                },
                'local_label': 'desktop',
                'remote_label': 'laptop',
                'final_protocol_winner': 'proto_a',
                'final_protocol_compare': {
                    'round_name': 'p1_protocol_compare',
                    'ranking': [{'arm_name': 'proto_a', 'candidate_meta': {'protocol_arm': 'proto_a'}}],
                },
            }
            final_round = {
                'round_name': 'p1_protocol_decide_round',
                'ranking': [
                    {
                        'arm_name': 'arm_a',
                        'valid': True,
                        'candidate_meta': {'protocol_arm': 'proto_a', 'aux_family': 'all_three'},
                    }
                ],
            }
            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                    round_kind=distributed.ROUND_KIND_PROTOCOL_DECIDE,
                    front_runner='proto_a',
                    final_round=final_round,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual('stopped_after_p1_protocol_decide', persisted['status'])
            self.assertEqual('proto_a', persisted['final_conclusion']['p1_protocol_winner'])
            self.assertEqual(final_round, persisted['p1']['protocol_decide_round'])
            self.assertEqual('p1_protocol_compare', persisted['p1']['protocol_compare']['round_name'])

    def test_update_run_state_for_protocol_decide_does_not_rollback_completed_run_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'completed',
                        'p1': {
                            'winner_refine_round': {'round_name': 'p1_winner_refine_round'},
                            'ablation_round': {'round_name': 'p1_ablation_round'},
                        },
                        'final_conclusion': {
                            'p1_winner': 'final_winner',
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            dispatch_state = {
                'stage': 'completed',
                'status': 'completed',
                'seed1': {'candidate_count': 33},
                'seed2': {'candidate_count': 14},
                'seed2_plan': {
                    'probe_candidate_names': ['arm_a'],
                    'decision_candidate_names': ['arm_a'],
                    'expanded_groups': [],
                },
                'local_label': 'desktop',
                'remote_label': 'laptop',
                'final_protocol_winner': 'proto_a',
            }
            final_round = {
                'round_name': 'p1_protocol_decide_round',
                'ranking': [
                    {
                        'arm_name': 'arm_a',
                        'valid': True,
                        'candidate_meta': {'protocol_arm': 'proto_a', 'aux_family': 'all_three'},
                    }
                ],
            }
            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                    round_kind=distributed.ROUND_KIND_PROTOCOL_DECIDE,
                    front_runner='proto_a',
                    final_round=final_round,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual('completed', persisted['status'])
            self.assertEqual('final_winner', persisted['final_conclusion']['p1_winner'])

    def test_update_run_state_for_ablation_marks_completed_and_sets_final_compare(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'stopped_after_p1_winner_refine',
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
                'seed1': {'candidate_count': 5},
                'seed2': {'candidate_count': 5},
                'local_label': 'desktop',
                'remote_label': 'laptop',
                'final_p1_winner': 'all_three_winner',
            }
            final_round = {
                'round_name': 'p1_ablation_round',
                'ranking': [
                    {
                        'arm_name': 'all_three_winner',
                        'valid': True,
                        'candidate_meta': {'aux_family': 'all_three'},
                    }
                ],
            }

            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                    round_kind=distributed.ROUND_KIND_ABLATION,
                    front_runner='all_three_winner',
                    final_round=final_round,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual('completed', persisted['status'])
            self.assertEqual(final_round, persisted['p1']['ablation_round'])
            self.assertEqual('p1_final_compare', persisted['p1']['final_compare']['round_name'])
            self.assertEqual('all_three_winner', persisted['p1']['winner'])
            self.assertEqual('all_three_winner', persisted['final_conclusion']['p1_winner'])

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
                    'search_space': {
                        'winner_refine_center_mode': 'explicit_arm_names',
                        'winner_refine_center_protocol_arm': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
                        'winner_refine_center_arm_names': ['center_a'],
                        'budget_ratio_digits': 5,
                        'aux_weight_digits': 6,
                    },
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
                    'winner_refine_center_selection_from_search_space',
                    return_value={'keep': None, 'explicit_arm_names': ['center_a']},
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

    def test_load_ablation_context_uses_refine_front_runner_and_recovers_seed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'ablation_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            custom_seed = 2468
            protocol_arm = fidelity.P1_WINNER_REFINE_PROTOCOL_ARM
            refine_winner = make_candidate('refine_winner')
            refine_winner.meta.update(
                {
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                }
            )
            runner_up = make_candidate('runner_up')
            runner_up.meta.update(
                {
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                }
            )
            state = {
                'status': 'stopped_after_p1_winner_refine',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration': {'dummy': True},
                    'search_space': {
                        'budget_ratio_digits': 5,
                        'aux_weight_digits': 6,
                    },
                    'winner_refine_front_runner': refine_winner.arm_name,
                    'winner_refine_round': {
                        'seed': custom_seed + 606,
                        'ranking': [
                            {
                                'arm_name': runner_up.arm_name,
                                'scheduler_profile': runner_up.scheduler_profile,
                                'curriculum_profile': runner_up.curriculum_profile,
                                'weight_profile': runner_up.weight_profile,
                                'window_profile': runner_up.window_profile,
                                'cfg_overrides': runner_up.cfg_overrides,
                                'candidate_meta': runner_up.meta,
                                'valid': True,
                            },
                            {
                                'arm_name': refine_winner.arm_name,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'candidate_meta': refine_winner.meta,
                                'valid': True,
                            },
                        ],
                    },
                },
                'final_conclusion': {
                    'p1_refine_front_runner': refine_winner.arm_name,
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
                    return_value=[make_candidate(protocol_arm)],
                ),
                patch.object(
                    distributed.fidelity,
                    'build_p1_ablation_candidates',
                    return_value=[make_candidate('ablation_all_three')],
                ) as build_candidates,
            ):
                ctx = distributed.load_ablation_context(run_dir)

            self.assertEqual(custom_seed, ctx['seed'])
            self.assertEqual(custom_seed + 707, ctx['seed_base'])
            self.assertEqual(['ablation_all_three'], [candidate.arm_name for candidate in ctx['candidates']])
            self.assertEqual(refine_winner.arm_name, build_candidates.call_args.args[2].arm_name)

    def test_load_protocol_decide_context_uses_persisted_search_space(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'protocol_decide_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            custom_seed = 321
            protocol_arm = fidelity.P1_WINNER_REFINE_PROTOCOL_ARM
            state = {
                'status': 'stopped_after_p1_calibration',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration': {'dummy': True},
                    'search_space': {
                        'protocol_decide_total_budget_ratios': [0.11],
                        'protocol_decide_mixes': [
                            {'name': 'custom_anchor', 'rank_share': 0.5, 'opp_share': 0.2, 'danger_share': 0.3},
                        ],
                        'protocol_decide_progressive_ambiguity_mode': fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_AMBIGUITY_MODE,
                        'protocol_decide_progressive_gap_threshold': fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_GAP_THRESHOLD,
                        'protocol_decide_progressive_noise_margin_mult': fidelity.P1_PROGRESSIVE_NOISE_MARGIN_MULT,
                    },
                    'calibration_round': {'seed': custom_seed + 404},
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
                    return_value=[make_candidate(protocol_arm)],
                ),
                patch.object(
                    distributed.fidelity,
                    'build_p1_protocol_decide_candidates',
                    return_value=[make_candidate('cand_a')],
                ) as build_candidates,
            ):
                ctx = distributed.load_protocol_decide_context(run_dir)

            self.assertEqual(custom_seed, ctx['seed'])
            self.assertEqual(custom_seed + 505, ctx['seed_base'])
            self.assertEqual(['cand_a'], [candidate.arm_name for candidate in ctx['candidates']])
            self.assertEqual([0.11], build_candidates.call_args.kwargs['search_space']['protocol_decide_total_budget_ratios'])
            self.assertEqual('custom_anchor', build_candidates.call_args.kwargs['search_space']['protocol_decide_mixes'][0]['name'])

    def test_load_protocol_decide_context_legacy_search_space_keeps_budget_coordinate_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'protocol_decide_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            custom_seed = 222
            protocol_arm = fidelity.P1_WINNER_REFINE_PROTOCOL_ARM
            state = {
                'status': 'stopped_after_p1_calibration',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration': {
                        'opp_weight_per_budget_unit': 0.052,
                        'danger_weight_per_budget_unit': 0.18,
                        'rank_effective_base': 0.05121494575615136,
                        'triple_combo_factor': 0.902,
                        'protocol_triple_combo_factors': {protocol_arm: 0.902},
                    },
                    'search_space': {
                        'protocol_decide_total_budget_ratios': [0.12],
                        'protocol_decide_mixes': [
                            {'name': 'opp_lean', 'rank_share': 0.38, 'opp_share': 0.31, 'danger_share': 0.31},
                        ],
                        'winner_refine_center_mode': 'explicit_arm_names',
                        'winner_refine_center_protocol_arm': protocol_arm,
                        'winner_refine_center_arm_names': [f'{protocol_arm}__B_r0046_o0037_d0037'],
                    },
                    'calibration_round': {'seed': custom_seed + 404},
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
                    return_value=[make_candidate(protocol_arm)],
                ),
            ):
                ctx = distributed.load_protocol_decide_context(run_dir)

            self.assertTrue(any(candidate.arm_name.endswith('__B_r0046_o0037_d0037') for candidate in ctx['candidates']))

    def test_load_protocol_decide_context_prefers_persisted_dispatch_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'protocol_decide_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            custom_seed = 654
            protocol_arm = fidelity.P1_WINNER_REFINE_PROTOCOL_ARM
            state = {
                'status': 'stopped_after_p1_calibration',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration': {'dummy': True},
                    'search_space': {
                        'protocol_decide_total_budget_ratios': [0.11],
                        'protocol_decide_mixes': [
                            {'name': 'custom_anchor', 'rank_share': 0.5, 'opp_share': 0.2, 'danger_share': 0.3},
                        ],
                        'protocol_decide_progressive_ambiguity_mode': fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_AMBIGUITY_MODE,
                        'protocol_decide_progressive_gap_threshold': fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_GAP_THRESHOLD,
                        'protocol_decide_progressive_noise_margin_mult': fidelity.P1_PROGRESSIVE_NOISE_MARGIN_MULT,
                    },
                    'calibration_round': {'seed': custom_seed + 404},
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            dispatch_state_path = distributed.dispatch_state_path_for_run(
                run_dir,
                distributed.ROUND_KIND_PROTOCOL_DECIDE,
            )
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            persisted = make_candidate('persisted_cand')
            dispatch_state_path.write_text(
                json.dumps(
                    {
                        'candidate_payloads': [
                            distributed.fidelity.candidate_cache_payload(persisted, include_meta=True),
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

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
                    return_value=[make_candidate(protocol_arm)],
                ),
                patch.object(
                    distributed.fidelity,
                    'build_p1_protocol_decide_candidates',
                ) as build_candidates,
            ):
                ctx = distributed.load_protocol_decide_context(run_dir)

            build_candidates.assert_not_called()
            self.assertEqual(['persisted_cand'], [candidate.arm_name for candidate in ctx['candidates']])

    def test_main_run_task_prints_compact_summary_instead_of_full_payload(self):
        args = argparse.Namespace(
            command='run-task',
            run_name='demo_run',
            candidate_arm='demo_arm',
            seed=7,
            machine_label='laptop',
            result_json=r'C:\tmp\result.json',
            screening_num_workers=None,
            screening_file_batch_size=None,
            screening_prefetch_factor=None,
            screening_val_file_batch_size=None,
            screening_val_prefetch_factor=None,
        )
        payload = {
            'run_name': 'demo_run',
            'candidate_arm': 'demo_arm',
            'seed': 7,
            'seed_label': 's7',
            'machine_label': 'laptop',
            'completed_at': '2026-03-31 12:00:00',
            'summary': {
                'valid': True,
                'selection_quality_score': -0.12,
                'recent_policy_loss': 0.51,
            },
            'payload': {
                'run': {
                    'phase_results': {
                        'phase_a': {
                            'latest': {
                                'last_full_recent_metrics': {
                                    'discard_top1_acc': 0.78,
                                }
                            }
                        }
                    }
                }
            },
        }
        stdout = StringIO()

        with (
            patch.object(distributed, 'parse_args', return_value=args),
            patch.object(distributed, 'execute_single_task', return_value=payload) as execute_mock,
            patch('sys.stdout', stdout),
        ):
            distributed.main()

        rendered = json.loads(stdout.getvalue())
        execute_mock.assert_called_once()
        self.assertEqual('demo_run', rendered['run_name'])
        self.assertEqual('demo_arm', rendered['candidate_arm'])
        self.assertEqual(r'C:\tmp\result.json', rendered['result_json'])
        self.assertTrue(rendered['valid'])
        self.assertNotIn('payload', rendered)

    def test_print_status_for_protocol_decide_includes_final_protocol_winner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            dispatch_state_path = distributed.dispatch_state_path_for_run(
                run_dir,
                distributed.ROUND_KIND_PROTOCOL_DECIDE,
            )
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            dispatch_state_path.write_text(
                json.dumps(
                    {
                        'stage': 'completed',
                        'status': 'completed',
                        'seed1': {'tasks': {}},
                        'seed2': {'tasks': {}},
                        'seed2_plan': {'probe_candidate_names': ['arm_a']},
                        'final_protocol_winner': 'proto_a',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            control_path = distributed.dispatch_control_path_for_run(
                run_dir,
                distributed.ROUND_KIND_PROTOCOL_DECIDE,
            )
            control_path.write_text(json.dumps({'workers': {}}, ensure_ascii=False), encoding='utf-8')
            args = argparse.Namespace(
                run_name='demo_run',
                round_kind=distributed.ROUND_KIND_PROTOCOL_DECIDE,
            )
            stdout = StringIO()

            with (
                patch.object(distributed.fidelity, 'FIDELITY_ROOT', Path(tmp_dir)),
                patch('sys.stdout', stdout),
            ):
                distributed.print_status(args)

            rendered = json.loads(stdout.getvalue())
            self.assertEqual('proto_a', rendered['final_protocol_winner'])
            self.assertIn('seed2_plan', rendered)
            self.assertNotIn('seed2_selector', rendered)

    def test_print_status_for_ablation_includes_final_p1_winner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            dispatch_state_path = distributed.dispatch_state_path_for_run(
                run_dir,
                distributed.ROUND_KIND_ABLATION,
            )
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            dispatch_state_path.write_text(
                json.dumps(
                    {
                        'stage': 'completed',
                        'status': 'completed',
                        'seed1': {'tasks': {}},
                        'seed2': {'tasks': {}},
                        'final_p1_winner': 'ablation_winner',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            control_path = distributed.dispatch_control_path_for_run(
                run_dir,
                distributed.ROUND_KIND_ABLATION,
            )
            control_path.write_text(json.dumps({'workers': {}}, ensure_ascii=False), encoding='utf-8')
            args = argparse.Namespace(
                run_name='demo_run',
                round_kind=distributed.ROUND_KIND_ABLATION,
            )
            stdout = StringIO()

            with (
                patch.object(distributed.fidelity, 'FIDELITY_ROOT', Path(tmp_dir)),
                patch('sys.stdout', stdout),
            ):
                distributed.print_status(args)

            rendered = json.loads(stdout.getvalue())
            self.assertEqual('ablation_winner', rendered['final_p1_winner'])
            self.assertNotIn('seed2_selector', rendered)
            self.assertNotIn('seed2_plan', rendered)


if __name__ == '__main__':
    unittest.main()

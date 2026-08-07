from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.supervised.run_sl_fidelity as fidelity
import mortal.supervised.run_sl_formal_distributed as formal_dist


def make_candidate_entry(
    arm_name: str,
    *,
    rank: int,
    protocol_arm: str = 'proto_arm',
    candidate_name: str | None = None,
    mix_name: str | None = None,
    source_arm: str | None = None,
    rank_budget_ratio: float = 0.05,
    opp_budget_ratio: float = 0.025,
    danger_budget_ratio: float = 0.04,
) -> dict:
    return {
        'arm_name': arm_name,
        'scheduler_profile': 'cosine',
        'curriculum_profile': 'broad_to_recent',
        'weight_profile': 'strong',
        'window_profile': '24m_12m',
        'cfg_overrides': {'aux': {'dummy': rank}},
        'candidate_meta': {
            'protocol_arm': protocol_arm,
            'aux_family': 'all_three',
            'candidate_name': candidate_name,
            'mix_name': mix_name,
            'source_arm': source_arm,
            'rank_budget_ratio': rank_budget_ratio,
            'opp_budget_ratio': opp_budget_ratio,
            'danger_budget_ratio': danger_budget_ratio,
        },
        'valid': True,
        'rank': rank,
    }


class RunStage05FormalDistributedTests(unittest.TestCase):
    def test_formal_step_scale_35_matches_one_epoch_long_abc_budget(self):
        phase_steps = formal_dist.formal.FORMAL_DEFAULTS['phase_steps']

        scaled = {
            phase: int(round(steps * 35.0))
            for phase, steps in phase_steps.items()
        }

        self.assertEqual(
            {'phase_a': 630000, 'phase_b': 420000, 'phase_c': 210000},
            scaled,
        )

    def test_formal_step_scale_140_matches_full_s140_budget(self):
        phase_steps = formal_dist.formal.FORMAL_DEFAULTS['phase_steps']

        scaled = {
            phase: int(round(steps * 140.0))
            for phase, steps in phase_steps.items()
        }

        self.assertEqual(
            {'phase_a': 2520000, 'phase_b': 1680000, 'phase_c': 840000},
            scaled,
        )

    def test_remote_only_builds_only_remote_worker(self):
        workers = formal_dist.common_dispatch.build_workers(
            enable_remote=True,
            enable_local=False,
            local_python='local-python',
            local_label='desktop',
            remote_host='mahjong-laptop',
            remote_repo=r'C:\Users\numbe\Desktop\MahjongAI',
            remote_python=r'C:\Users\numbe\miniconda3\envs\mortal\python.exe',
            remote_label='laptop',
            ssh_key=None,
        )

        self.assertEqual(['remote'], [worker.kind for worker in workers])
        self.assertEqual(['laptop'], [worker.label for worker in workers])

    def test_remote_only_control_state_omits_local_worker(self):
        control_state = formal_dist.common_dispatch.initialize_dispatch_control_state(
            local_label=None,
            remote_label='laptop',
            remote_launch_mode='interactive_window',
        )

        self.assertNotIn('desktop', control_state['workers'])
        self.assertEqual({'laptop'}, set(control_state['workers']))
        self.assertFalse(
            formal_dist.common_dispatch.ensure_control_state_workers(
                control_state=control_state,
                local_label=None,
                remote_label='laptop',
                remote_launch_mode='interactive_window',
            )
        )

    def test_remote_interactive_window_command_uses_explicit_powershell(self):
        worker = formal_dist.WorkerSpec(
            kind='remote',
            label='laptop',
            python=r'C:\Python\python.exe',
            host='mahjong-laptop',
            repo=r'C:\Users\numbe\Desktop\MahjongAI',
            ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
        )
        command = formal_dist.build_remote_interactive_window_command(
            worker=worker,
            run_name='demo_run',
            task_state={
                'task_id': 'formal__anchor',
                'candidate_arm': 'anchor',
            },
            remote_result_path=Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\result.json'),
            remote_runtime_root=Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\runtime\formal__anchor'),
            formal_overrides={
                'num_workers': 4,
                'file_batch_size': 10,
                'prefetch_factor': 4,
                'val_file_batch_size': 7,
                'val_prefetch_factor': 5,
            },
        )

        self.assertEqual('ssh', command[0])
        self.assertIn('powershell', command)
        self.assertIn('-NoProfile', command)
        self.assertIn('-EncodedCommand', command)
        script = formal_dist.dispatch.decode_remote_powershell_command_arg(command[-1])
        self.assertIn(
            r"C:\Users\numbe\Desktop\MahjongAI\scripts\start_interactive_remote_python.ps1",
            script,
        )
        self.assertIn('-WaitForStartOnly', script)

    def test_remote_task_launcher_has_no_execution_time_limit(self):
        launcher_path = Path(__file__).resolve().parents[2] / 'scripts' / 'start_interactive_remote_python.ps1'
        launcher = launcher_path.read_text(encoding='utf-8')

        self.assertIn('-ExecutionTimeLimit ([TimeSpan]::Zero)', launcher)

    def test_task_command_is_resume_safe_by_default(self):
        command = formal_dist.build_task_command_args(
            run_name='demo_run',
            task_state={'candidate_arm': 'anchor'},
            machine_label='laptop',
        )

        self.assertEqual(1, command.count('--resume-existing'))

    def test_task_storage_key_is_stable_short_and_collision_resistant_enough_for_paths(self):
        key = formal_dist.task_storage_key(
            'formal__C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00516_o000135_d000804'
        )

        self.assertEqual(16, len(key))
        self.assertEqual(key, formal_dist.task_storage_key(
            'formal__C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00516_o000135_d000804'
        ))
        self.assertNotEqual(key, formal_dist.task_storage_key('formal__different'))

    def test_launch_remote_task_maps_runtime_paths_to_remote_repo(self):
        worker = formal_dist.WorkerSpec(
            kind='remote',
            label='laptop',
            python=r'C:\Python\python.exe',
            host='mahjong-laptop',
            repo=r'C:\Users\numbe\Desktop\MahjongAI_longabc_runner',
            ssh_key=None,
        )
        task_state = {
            'task_id': 'formal__anchor',
            'candidate_arm': 'anchor',
        }

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(
                formal_dist,
                'REPO_ROOT',
                Path(tmp_dir),
            ),
            patch.object(
                formal_dist,
                'build_remote_interactive_window_command',
                return_value=[sys.executable, '-c', 'pass'],
            ) as build_command,
        ):
            dispatch_root = Path(tmp_dir) / 'logs' / 'sl_fidelity' / 'run' / 'distributed' / 'formal_dispatch'
            active = formal_dist.launch_remote_task(
                worker,
                run_name='run',
                task_state=task_state,
                dispatch_root=dispatch_root,
                launch_mode='interactive_window',
                formal_overrides=None,
            )
            active.process.wait(timeout=10)

        self.assertIn(
            r'C:\Users\numbe\Desktop\MahjongAI_longabc_runner',
            str(build_command.call_args.kwargs['remote_result_path']),
        )
        self.assertIn(
            r'C:\Users\numbe\Desktop\MahjongAI_longabc_runner',
            str(build_command.call_args.kwargs['remote_runtime_root']),
        )
        self.assertTrue(task_state['remote_detached'])
        self.assertEqual(1, task_state['attempts'])
        self.assertIn('attempt_001_', task_state['remote_runtime_root'])
        self.assertIn('attempt_001_', task_state['remote_result_path'])
        self.assertIn('attempt_001_', task_state['log_path'])
        self.assertTrue(task_state['remote_task_name'].startswith('MahjongAI-WinnerRefine-'))

    def test_launch_remote_task_persists_identity_before_start(self):
        worker = formal_dist.WorkerSpec(
            kind='remote',
            label='laptop',
            python='python',
            host='mahjong-laptop',
            repo=r'C:\runner',
            ssh_key=None,
        )
        task_state = {'task_id': 'formal__anchor', 'candidate_arm': 'anchor'}
        snapshots = []

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(formal_dist, 'REPO_ROOT', Path(tmp_dir)),
            patch.object(
                formal_dist,
                'build_remote_interactive_window_command',
                return_value=[sys.executable, '-c', 'pass'],
            ),
        ):
            active = formal_dist.launch_remote_task(
                worker,
                run_name='run',
                task_state=task_state,
                dispatch_root=Path(tmp_dir) / 'dispatch',
                launch_mode='interactive_window',
                formal_overrides=None,
                persist_state=lambda: snapshots.append(dict(task_state)),
            )
            active.process.wait(timeout=10)

        self.assertEqual(1, len(snapshots))
        self.assertEqual('running', snapshots[0]['status'])
        self.assertEqual(task_state['remote_launch_id'], snapshots[0]['remote_launch_id'])
        self.assertEqual(task_state['remote_result_path'], snapshots[0]['remote_result_path'])

    def test_reset_running_tasks_preserves_recoverable_remote_task(self):
        task = {
            'status': 'running',
            'worker_label': 'laptop',
            'remote_launch_mode': 'interactive_window',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        dispatch_state = {'formal': {'tasks': {'task': task}}}

        formal_dist.reset_running_tasks_for_resume(dispatch_state)

        self.assertEqual('running', task['status'])
        self.assertIn('coordinator_recovered_at', task)
        self.assertEqual('laptop', task['worker_label'])

    def test_reset_running_tasks_preserves_recoverable_ssh_inline_task(self):
        task = {
            'status': 'running',
            'worker_label': 'laptop',
            'remote_launch_mode': 'ssh_inline',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        dispatch_state = {
            'remote_label': 'laptop',
            'formal': {'tasks': {'task': task}},
        }

        formal_dist.reset_running_tasks_for_resume(dispatch_state)

        self.assertEqual('running', task['status'])
        self.assertIn('coordinator_recovered_at', task)

    def test_reopen_failed_task_when_operator_raises_max_attempts(self):
        task = {
            'status': 'failed',
            'attempts': 3,
            'error': 'remote task is no longer running (orphaned)',
            'finished_at': '2026-08-07 06:25:55',
            'worker_label': 'laptop',
            'remote_launch_mode': 'interactive_window',
            'remote_runtime_root': r'C:\runtime\attempt_003',
            'remote_result_path': r'C:\results\attempt_003.json',
            'remote_task_name': 'MahjongAI-formal-attempt-003',
            'last_remote_probe_status': 'orphaned',
        }
        dispatch_state = {'formal': {'tasks': {'task': task}}}

        with patch.object(formal_dist.fidelity, 'ts_now', return_value='2026-08-07 22:00:00'):
            changed = formal_dist.reopen_retriable_failed_tasks_for_resume(
                dispatch_state,
                max_attempts=4,
            )

        self.assertTrue(changed)
        self.assertEqual('pending', task['status'])
        self.assertEqual(3, task['attempts'])
        self.assertEqual('2026-08-07 22:00:00', task['retry_reopened_at'])
        self.assertNotIn('error', task)
        self.assertNotIn('finished_at', task)
        self.assertNotIn('remote_task_name', task)
        self.assertEqual(
            {
                'reopened_at': '2026-08-07 22:00:00',
                'attempts': 3,
                'error': 'remote task is no longer running (orphaned)',
                'finished_at': '2026-08-07 06:25:55',
                'remote_task_name': 'MahjongAI-formal-attempt-003',
                'remote_runtime_root': r'C:\runtime\attempt_003',
                'remote_result_path': r'C:\results\attempt_003.json',
                'last_remote_probe_status': 'orphaned',
            },
            task['recovered_failures'][0],
        )

    def test_failed_task_stays_terminal_without_extra_attempt_budget(self):
        task = {'status': 'failed', 'attempts': 3, 'error': 'failed'}

        changed = formal_dist.reopen_retriable_failed_tasks_for_resume(
            {'formal': {'tasks': {'task': task}}},
            max_attempts=3,
        )

        self.assertFalse(changed)
        self.assertEqual({'status': 'failed', 'attempts': 3, 'error': 'failed'}, task)

    def test_poll_running_remote_tasks_supports_ssh_inline(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'worker_label': 'laptop',
            'remote_launch_mode': 'ssh_inline',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')

        with patch.object(formal_dist, 'poll_remote_task', return_value=True) as poll:
            changed = formal_dist.poll_running_remote_tasks(
                stage_state={'tasks': {'formal__anchor': task}},
                workers_by_label={'laptop': worker},
                active_task_ids=set(),
                max_attempts=3,
            )

        self.assertTrue(changed)
        poll.assert_called_once_with(worker=worker, task_state=task, max_attempts=3)

    def test_reset_running_tasks_requeues_nonrecoverable_task(self):
        task = {
            'status': 'running',
            'worker_label': 'desktop',
            'remote_launch_mode': 'ssh_inline',
        }
        dispatch_state = {'formal': {'tasks': {'task': task}}}

        formal_dist.reset_running_tasks_for_resume(dispatch_state)

        self.assertEqual('pending', task['status'])
        self.assertNotIn('worker_label', task)

    def test_reset_running_tasks_does_not_adopt_local_task_with_stale_remote_metadata(self):
        task = {
            'status': 'running',
            'worker_label': 'desktop',
            'remote_launch_mode': 'interactive_window',
            'remote_runtime_root': r'C:\stale-runtime',
            'remote_result_path': r'C:\stale-result.json',
        }
        dispatch_state = {
            'remote_label': 'laptop',
            'formal': {'tasks': {'task': task}},
        }

        formal_dist.reset_running_tasks_for_resume(dispatch_state)

        self.assertEqual('pending', task['status'])
        self.assertNotIn('worker_label', task)

    def test_launch_local_task_clears_stale_remote_attempt_metadata(self):
        worker = formal_dist.WorkerSpec(kind='local', label='desktop', python=sys.executable)
        task = {
            'task_id': 'formal__anchor',
            'candidate_arm': 'anchor',
            'status': 'pending',
            'error': 'old remote failure',
            'remote_launch_mode': 'interactive_window',
            'remote_runtime_root': r'C:\stale-runtime',
            'remote_result_path': r'C:\stale-result.json',
            'remote_task_name': 'MahjongAI-WinnerRefine-stale',
        }

        def fake_launch(_worker, *, task_state, **_kwargs):
            task_state['status'] = 'running'
            task_state['attempts'] = 2
            task_state['worker_label'] = 'desktop'
            task_state['pid'] = 123
            return SimpleNamespace(task_state=task_state)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(formal_dist.dispatch, 'launch_json_task', side_effect=fake_launch),
        ):
            formal_dist.launch_local_task(
                worker,
                run_name='run',
                task_state=task,
                dispatch_root=Path(tmp_dir),
            )

        self.assertEqual('running', task['status'])
        self.assertEqual(123, task['pid'])
        self.assertEqual('run', task['run_name'])
        self.assertNotIn('error', task)
        for key in formal_dist.REMOTE_TASK_METADATA_FIELDS:
            self.assertNotIn(key, task)

    def test_detached_launch_transport_failure_waits_for_remote_probe(self):
        task = {'status': 'running', 'remote_detached': True}
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(255)'])
        process.wait(timeout=10)
        active = formal_dist.ActiveTask(
            worker=worker,
            stage_name='formal',
            task_id='task',
            task_state=task,
            process=process,
            log_path=Path('task.log'),
            local_result_path=Path('result.json'),
        )

        formal_dist.handle_finished_task(active=active, max_attempts=3)

        self.assertEqual('running', task['status'])
        self.assertIn('code 255', task['remote_launch_transport_error'])

    def test_remote_probe_unreachable_never_consumes_retry(self):
        task = {
            'status': 'running',
            'attempts': 2,
            'remote_missing_polls': 1,
            'remote_missing_status': 'orphaned',
            'remote_done_result_polls': 1,
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        with patch.object(
            formal_dist,
            'probe_remote_task',
            return_value={'reachable': False, 'status': 'unreachable', 'error': 'offline'},
        ):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('running', task['status'])
        self.assertEqual(2, task['attempts'])
        self.assertNotIn('remote_missing_polls', task)
        self.assertNotIn('remote_missing_status', task)
        self.assertNotIn('remote_done_result_polls', task)
        self.assertEqual('offline', task['remote_probe_error'])

    def test_running_probe_clears_stale_terminal_state_and_records_evidence(self):
        task = {
            'status': 'running',
            'attempts': 2,
            'error': 'previous transport exited 255',
            'finished_at': '2026-08-01 06:24:03',
            'remote_launch_transport_error': 'ssh exited 255',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {
            'reachable': True,
            'status': 'running',
            'result_exists': False,
            'process_ids': [12928, 28128],
            'scheduled_tasks': [
                {'task_name': 'MahjongAI-WinnerRefine-formal__anchor', 'state': 'Running'}
            ],
        }

        with patch.object(formal_dist, 'probe_remote_task', return_value=probe):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('running', task['status'])
        self.assertNotIn('error', task)
        self.assertNotIn('finished_at', task)
        self.assertNotIn('remote_launch_transport_error', task)
        self.assertIn('remote_launch_recovered_at', task)
        self.assertEqual([12928, 28128], task['last_remote_process_ids'])
        self.assertEqual(probe['scheduled_tasks'], task['last_remote_scheduled_tasks'])

    def test_remote_orphan_requires_repeated_confirmation_before_retry(self):
        task = {'status': 'running', 'attempts': 1}
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {'reachable': True, 'status': 'orphaned', 'result_exists': False}
        with patch.object(formal_dist, 'probe_remote_task', return_value=probe):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)
            self.assertEqual('running', task['status'])
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('pending', task['status'])
        self.assertEqual(1, task['attempts'])

    def test_remote_missing_confirmation_resets_when_status_changes(self):
        task = {
            'status': 'running',
            'attempts': 1,
            'remote_missing_polls': formal_dist.REMOTE_MISSING_CONFIRM_POLLS - 1,
            'remote_missing_status': 'not_started',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {'reachable': True, 'status': 'orphaned', 'result_exists': False}

        with patch.object(formal_dist, 'probe_remote_task', return_value=probe):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('running', task['status'])
        self.assertEqual('orphaned', task['remote_missing_status'])
        self.assertEqual(1, task['remote_missing_polls'])

    def test_remote_nonzero_done_retries_without_deleting_artifact_metadata(self):
        task = {
            'status': 'running',
            'attempts': 1,
            'remote_runtime_root': r'C:\runtime\attempt_001',
            'remote_result_path': r'C:\results\attempt_001.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {
            'reachable': True,
            'status': 'done',
            'result_exists': False,
            'done_exit_code': 1,
            'done_error': 'CUDA failure',
        }
        with patch.object(formal_dist, 'probe_remote_task', return_value=probe):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('pending', task['status'])
        self.assertEqual(r'C:\runtime\attempt_001', task['remote_runtime_root'])
        self.assertEqual(r'C:\results\attempt_001.json', task['remote_result_path'])
        self.assertIn('CUDA failure', task['error'])

    def test_invalid_done_exit_code_retries_without_crashing_coordinator(self):
        task = {'status': 'running', 'attempts': 1}
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {
            'reachable': True,
            'status': 'done',
            'result_exists': False,
            'done_exit_code': 'not-an-int',
        }

        with (
            patch.object(formal_dist, 'probe_remote_task', return_value=probe),
            patch.object(formal_dist, 'cleanup_remote_task_registration') as cleanup,
        ):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('pending', task['status'])
        self.assertIn('invalid done exit code', task['error'])
        cleanup.assert_called_once_with(worker, task)

    def test_parse_remote_probe_output_ignores_powershell_noise(self):
        payload = formal_dist.parse_remote_probe_output(
            '#< CLIXML\nnoise\n{"status":"running","process_ids":[123]}\n'
        )

        self.assertEqual('running', payload['status'])
        self.assertEqual([123], payload['process_ids'])

    def test_remote_result_completion_closes_running_task(self):
        task = {
            'status': 'running',
            'remote_result_path': r'C:\remote\result.json',
            'local_result_path': r'C:\local\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        payload = {
            'child_run_name': 'child',
            'offline_checkpoint_winner': 'best_loss',
            'completed_at': '2026-08-01 12:00:00',
        }
        with (
            patch.object(formal_dist, 'fetch_remote_result_file'),
            patch.object(formal_dist, 'load_task_result', return_value=payload),
            patch.object(formal_dist, 'sync_remote_task_outputs'),
        ):
            completion = formal_dist.try_complete_remote_task(worker, task)

        self.assertTrue(completion.completed)
        self.assertIsNone(completion.error)
        self.assertFalse(completion.invalid_result)
        self.assertEqual('completed', task['status'])
        self.assertEqual('remote_result_json', task['completion_source'])
        self.assertEqual('child', task['child_run_name'])

    def test_successful_remote_exit_waits_for_transient_result_sync_failure(self):
        task = {'status': 'running', 'attempts': 2}
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {
            'reachable': True,
            'status': 'done',
            'result_exists': True,
            'done_exit_code': 0,
        }
        with (
            patch.object(formal_dist, 'probe_remote_task', return_value=probe),
            patch.object(
                formal_dist,
                'try_complete_remote_task',
                return_value=formal_dist.RemoteCompletionAttempt(error='scp offline'),
            ),
        ):
            for _ in range(formal_dist.REMOTE_MISSING_CONFIRM_POLLS + 1):
                formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('running', task['status'])
        self.assertEqual(2, task['attempts'])
        self.assertEqual('scp offline', task['remote_result_error'])

    def test_operator_interrupt_handles_detached_remote_task(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'attempts': 2,
            'worker_label': 'laptop',
            'remote_launch_mode': 'interactive_window',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        control_state = {
            'workers': {
                'laptop': {
                    'paused': True,
                    'interrupt_requested': True,
                }
            }
        }
        with patch.object(formal_dist, 'interrupt_persisted_remote_task', return_value=True):
            changed = formal_dist.apply_formal_worker_control_requests(
                control_state=control_state,
                active={},
                stage_state={'tasks': {'formal__anchor': task}},
                workers_by_label={'laptop': worker},
            )

        self.assertTrue(changed)
        self.assertEqual('pending', task['status'])
        self.assertEqual(1, task['attempts'])
        self.assertFalse(control_state['workers']['laptop']['interrupt_requested'])

    def test_coordinator_restart_keeps_detached_running_task_nonterminal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir) / 'fidelity'
            run_dir = fidelity_root / 'coordinator'
            dispatch_root = run_dir / 'distributed' / 'formal_dispatch'
            dispatch_root.mkdir(parents=True)
            task = {
                'task_id': 'formal__anchor',
                'candidate_arm': 'anchor',
                'status': 'running',
                'attempts': 1,
                'worker_label': 'laptop',
                'remote_launch_mode': 'interactive_window',
                'remote_runtime_root': r'C:\runtime',
                'remote_result_path': r'C:\result.json',
            }
            state = {
                'status': 'running',
                'stage': 'formal',
                'formal': {'tasks': {'formal__anchor': task}},
            }
            formal_dist.write_dispatch_state(dispatch_root / 'dispatch_state.json', state)
            control = formal_dist.common_dispatch.initialize_dispatch_control_state(
                local_label=None,
                remote_label='laptop',
                remote_launch_mode='interactive_window',
            )
            formal_dist.common_dispatch.write_dispatch_control(
                dispatch_root / 'dispatch_control.json',
                control,
            )
            worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
            args = SimpleNamespace(
                local_only=False,
                remote_only=True,
                run_name='coordinator',
                source_run_name='unused',
                candidate_arm=['anchor'],
                seed_offset=2000,
                formal_step_scale=140.0,
                local_python='python',
                local_label='desktop',
                remote_host='laptop',
                remote_repo=r'C:\runner',
                remote_python='python',
                remote_label='laptop',
                ssh_key=None,
                remote_launch_mode='interactive_window',
                remote_num_workers=4,
                remote_file_batch_size=10,
                remote_prefetch_factor=4,
                remote_val_file_batch_size=7,
                remote_val_prefetch_factor=5,
                max_attempts=3,
                poll_seconds=15.0,
            )

            with (
                patch.object(formal_dist.fidelity, 'FIDELITY_ROOT', fidelity_root),
                patch.object(formal_dist.fidelity, 'acquire_run_lock', return_value=run_dir / 'lock'),
                patch.object(formal_dist.fidelity, 'release_run_lock'),
                patch.object(formal_dist.common_dispatch, 'build_workers', return_value=[worker]),
                patch.object(formal_dist, 'poll_running_remote_tasks', return_value=False),
                patch.object(formal_dist, 'launch_task_for_worker') as launch_task,
                patch.object(formal_dist.time, 'sleep', side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    formal_dist.run_dispatch(args)

            persisted = json.loads(
                (dispatch_root / 'dispatch_state.json').read_text(encoding='utf-8')
            )
            persisted_task = persisted['formal']['tasks']['formal__anchor']
            self.assertEqual('running', persisted['status'])
            self.assertEqual('formal', persisted['stage'])
            self.assertEqual('running', persisted_task['status'])
            self.assertIn('coordinator_recovered_at', persisted_task)
            launch_task.assert_not_called()

    def test_load_source_context_builds_child_run_names_from_explicit_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir)
            source_run_dir = fidelity_root / 'source_run'
            source_run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'seed': 20260329,
                'selected_protocol_arms': ['proto_arm'],
                'p1': {
                    'selected_protocol_arm': 'proto_arm',
                    'winner_refine_front_runner': 'front_runner_arm',
                    'winner_refine_round': {
                        'ranking': [
                            make_candidate_entry('arm_b', rank=2),
                            make_candidate_entry('arm_a', rank=1),
                        ]
                    },
                },
                'final_conclusion': {
                    'p1_protocol_winner': 'proto_arm',
                    'p1_refine_front_runner': 'front_runner_arm',
                },
            }
            (source_run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_dist.load_source_context(
                source_run_dir=source_run_dir,
                coordinator_run_name='triplet_formal_run',
                candidate_arms=['arm_a', 'arm_b'],
                formal_seed_offset=2000,
                formal_step_scale=5.0,
            )

            self.assertEqual('source_run', context['source_run_name'])
            self.assertEqual(20262329, context['formal_seed'])
            self.assertEqual('proto_arm', context['selected_protocol_arm'])
            self.assertEqual(
                ['triplet_formal_run__arm_a', 'triplet_formal_run__arm_b'],
                [payload['child_run_name'] for payload in context['candidate_payloads']],
            )

    def test_load_source_context_resolves_structural_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir)
            source_run_dir = fidelity_root / 'source_run'
            source_run_dir.mkdir(parents=True, exist_ok=True)
            center_arm = 'opp_center'
            scaled_arm = 'opp_center_scaled'
            shifted_arm = 'opp_center_shifted'
            state = {
                'seed': 20260329,
                'selected_protocol_arms': ['proto_arm'],
                'p1': {
                    'selected_protocol_arm': 'proto_arm',
                    'winner_refine_front_runner': scaled_arm,
                    'protocol_decide_round': {
                        'ranking': [
                            make_candidate_entry(
                                center_arm,
                                rank=1,
                                candidate_name='opp_lean_12',
                                mix_name='opp_lean',
                                source_arm=None,
                                rank_budget_ratio=0.0456,
                                opp_budget_ratio=0.0372,
                                danger_budget_ratio=0.0372,
                            ),
                        ]
                    },
                    'winner_refine_round': {
                        'ranking': [
                            make_candidate_entry(
                                scaled_arm,
                                rank=1,
                                candidate_name='scaled',
                                source_arm=center_arm,
                                rank_budget_ratio=0.0388,
                                opp_budget_ratio=0.0316,
                                danger_budget_ratio=0.0316,
                            ),
                            make_candidate_entry(
                                shifted_arm,
                                rank=2,
                                candidate_name='shifted',
                                source_arm=center_arm,
                                rank_budget_ratio=0.0356,
                                opp_budget_ratio=0.0370,
                                danger_budget_ratio=0.0472,
                            ),
                        ]
                    },
                },
                'final_conclusion': {
                    'p1_protocol_winner': 'proto_arm',
                    'p1_refine_front_runner': scaled_arm,
                },
            }
            (source_run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_dist.load_source_context(
                source_run_dir=source_run_dir,
                coordinator_run_name='triplet_formal_run',
                candidate_arms=['opp_lean*0.85', 'opp_lean(rank--/danger++)'],
                formal_seed_offset=2000,
                formal_step_scale=5.0,
            )

            self.assertEqual(scaled_arm, context['candidate_alias_to_arm']['opp_lean*0.85'])
            self.assertEqual(shifted_arm, context['candidate_alias_to_arm']['opp_lean(rank--/danger++)'])
            self.assertEqual('opp_lean*0.85', context['candidate_payloads'][0]['candidate_alias'])
            self.assertEqual('opp_lean(rank--/danger++)', context['candidate_payloads'][1]['candidate_alias'])
            self.assertEqual([scaled_arm, shifted_arm], [item['arm_name'] for item in context['candidate_payloads']])

    def test_initialize_dispatch_state_creates_one_task_per_candidate(self):
        source_context = {
            'source_run_name': 'source_run',
            'source_seed': 1,
            'selected_protocol_arm': 'proto_arm',
            'selected_protocol_arms': ['proto_arm'],
            'source_refine_front_runner': 'front_runner',
            'formal_seed': 2001,
            'formal_step_scale': 5.0,
            'candidate_alias_to_arm': {'anchor*1.0': 'arm_a', 'opp_lean*0.85': 'arm_b'},
            'candidate_arm_to_alias': {'arm_a': 'anchor*1.0', 'arm_b': 'opp_lean*0.85'},
            'candidate_payloads': [
                {
                    **fidelity.candidate_cache_payload(
                        fidelity.CandidateSpec(
                            arm_name='arm_a',
                            scheduler_profile='cosine',
                            curriculum_profile='broad_to_recent',
                            weight_profile='strong',
                            window_profile='24m_12m',
                            cfg_overrides={},
                            meta={'protocol_arm': 'proto_arm'},
                        ),
                        include_meta=True,
                    ),
                    'source_rank': 1,
                    'candidate_alias': 'anchor*1.0',
                    'child_run_name': 'triplet__arm_a',
                },
                {
                    **fidelity.candidate_cache_payload(
                        fidelity.CandidateSpec(
                            arm_name='arm_b',
                            scheduler_profile='cosine',
                            curriculum_profile='broad_to_recent',
                            weight_profile='strong',
                            window_profile='24m_12m',
                            cfg_overrides={},
                            meta={'protocol_arm': 'proto_arm'},
                        ),
                        include_meta=True,
                    ),
                    'source_rank': 2,
                    'candidate_alias': 'opp_lean*0.85',
                    'child_run_name': 'triplet__arm_b',
                },
            ],
        }

        dispatch_state = formal_dist.initialize_dispatch_state(
            run_name='triplet_formal_run',
            source_context=source_context,
            local_label='desktop',
            remote_label='laptop',
        )

        self.assertEqual('formal', dispatch_state['stage'])
        self.assertEqual(2, dispatch_state['formal']['task_count'])
        self.assertEqual({'anchor*1.0', 'opp_lean*0.85'}, set(dispatch_state['candidate_alias_to_arm']))
        self.assertEqual(
            {'arm_a', 'arm_b'},
            {task['candidate_arm'] for task in dispatch_state['formal']['tasks'].values()},
        )
        for task in dispatch_state['formal']['tasks'].values():
            self.assertEqual('triplet_formal_run', task['run_name'])
            self.assertEqual(2001, task['formal_seed'])
            self.assertEqual(5.0, task['formal_step_scale'])

    def test_execute_single_task_writes_child_state_and_result_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fidelity_root = root / 'fidelity'
            ab_root = root / 'sl_ab'
            fidelity_root.mkdir()
            ab_root.mkdir()
            coordinator_run = fidelity_root / 'triplet_formal_run'
            coordinator_run.mkdir()

            candidate = fidelity.CandidateSpec(
                arm_name='arm_a',
                scheduler_profile='cosine',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                cfg_overrides={'aux': {'dummy': 1}},
                meta={'protocol_arm': 'proto_arm'},
            )
            source_context = {
                'source_run_name': 'source_run',
                'source_seed': 20260329,
                'selected_protocol_arm': 'proto_arm',
                'selected_protocol_arms': ['proto_arm'],
                'source_refine_front_runner': 'front_runner',
                'formal_seed': 20262329,
                'formal_step_scale': 5.0,
                'candidate_payloads': [
                    {
                        **fidelity.candidate_cache_payload(candidate, include_meta=True),
                        'source_rank': 1,
                        'child_run_name': 'triplet_formal_run__arm_a',
                    }
                ],
            }
            dispatch_state = formal_dist.initialize_dispatch_state(
                run_name='triplet_formal_run',
                source_context=source_context,
                local_label='desktop',
                remote_label=None,
            )
            (coordinator_run / 'distributed' / 'formal_dispatch').mkdir(parents=True, exist_ok=True)
            (coordinator_run / 'distributed' / 'formal_dispatch' / 'dispatch_state.json').write_text(
                json.dumps(dispatch_state, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            def fake_finalize_formal_result(_cfg, result, *, protocol_arm):
                checkpoint_root = (
                    ab_root
                    / 'triplet_formal_run__arm_a_formal'
                    / 'checkpoint_compare'
                    / 'phase_c'
                    / 'checkpoints'
                )
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                best_loss = checkpoint_root / 'best_loss.pth'
                best_acc = checkpoint_root / 'best_acc.pth'
                best_rank = checkpoint_root / 'best_rank.pth'
                for path in (best_loss, best_acc, best_rank):
                    path.write_text('ckpt', encoding='utf-8')
                result.update(
                    {
                        'offline_checkpoint_winner': 'best_loss',
                        'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
                        'checkpoint_pack_types': ['best_loss', 'best_acc', 'best_rank'],
                        'candidates': {
                            'best_loss': {'path': str(best_loss)},
                            'best_acc': {'path': str(best_acc)},
                            'best_rank': {'path': str(best_rank)},
                        },
                    }
                )
                return result

            result_json = root / 'result.json'
            with (
                patch.object(formal_dist.fidelity, 'FIDELITY_ROOT', fidelity_root),
                patch.object(formal_dist.ab, 'AB_ROOT', ab_root),
                patch.object(formal_dist.ab, 'build_base_config', return_value={'supervised': {}}),
                patch.object(formal_dist.ab, 'group_files_by_month', return_value={}),
                patch.object(formal_dist.ab, 'load_all_files', return_value=[]),
                patch.object(formal_dist.ab, 'run_ab6_checkpoint', return_value={'winner': 'best_loss'}),
                patch.object(formal_dist.formal, 'finalize_formal_result', side_effect=fake_finalize_formal_result),
            ):
                payload = formal_dist.execute_single_task(
                    run_name='triplet_formal_run',
                    candidate_arm='arm_a',
                    result_json=result_json,
                    machine_label='desktop',
                )

            self.assertEqual('triplet_formal_run__arm_a', payload['child_run_name'])
            self.assertEqual('best_loss', payload['offline_checkpoint_winner'])
            child_state = json.loads(
                (fidelity_root / 'triplet_formal_run__arm_a' / 'state.json').read_text(
                    encoding='utf-8'
                )
            )
            self.assertEqual('completed', child_state['formal']['status'])
            self.assertEqual('pending', child_state['formal_1v3']['status'])
            self.assertTrue(result_json.exists())

    def test_execute_single_task_preserves_child_and_ab_dirs_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fidelity_root = root / 'fidelity'
            ab_root = root / 'sl_ab'
            fidelity_root.mkdir()
            ab_root.mkdir()
            coordinator_run = fidelity_root / 'triplet_formal_run'
            coordinator_run.mkdir()
            child_run_dir = fidelity_root / 'triplet_formal_run__arm_a'
            child_run_dir.mkdir()
            child_marker = child_run_dir / 'existing_child_marker.txt'
            child_marker.write_text('keep', encoding='utf-8')
            ab_dir = ab_root / 'triplet_formal_run__arm_a_formal'
            ab_dir.mkdir()
            ab_marker = ab_dir / 'existing_ab_marker.txt'
            ab_marker.write_text('keep', encoding='utf-8')

            candidate = fidelity.CandidateSpec(
                arm_name='arm_a',
                scheduler_profile='cosine',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                cfg_overrides={},
                meta={'protocol_arm': 'proto_arm'},
            )
            source_context = {
                'source_run_name': 'source_run',
                'source_seed': 20260329,
                'selected_protocol_arm': 'proto_arm',
                'selected_protocol_arms': ['proto_arm'],
                'source_refine_front_runner': 'front_runner',
                'formal_seed': 20262329,
                'formal_step_scale': 5.0,
                'candidate_payloads': [
                    {
                        **fidelity.candidate_cache_payload(candidate, include_meta=True),
                        'source_rank': 1,
                        'child_run_name': 'triplet_formal_run__arm_a',
                    }
                ],
            }
            dispatch_state = formal_dist.initialize_dispatch_state(
                run_name='triplet_formal_run',
                source_context=source_context,
                local_label='desktop',
                remote_label=None,
            )
            dispatch_root = coordinator_run / 'distributed' / 'formal_dispatch'
            dispatch_root.mkdir(parents=True, exist_ok=True)
            (dispatch_root / 'dispatch_state.json').write_text(
                json.dumps(dispatch_state, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            def fake_finalize_formal_result(_cfg, result, *, protocol_arm):
                checkpoint_root = ab_dir / 'checkpoint_compare' / 'phase_c' / 'checkpoints'
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                best_loss = checkpoint_root / 'best_loss.pth'
                best_acc = checkpoint_root / 'best_acc.pth'
                best_rank = checkpoint_root / 'best_rank.pth'
                for path in (best_loss, best_acc, best_rank):
                    path.write_text('ckpt', encoding='utf-8')
                result.update(
                    {
                        'offline_checkpoint_winner': 'best_loss',
                        'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
                        'checkpoint_pack_types': ['best_loss', 'best_acc', 'best_rank'],
                        'candidates': {
                            'best_loss': {'path': str(best_loss)},
                            'best_acc': {'path': str(best_acc)},
                            'best_rank': {'path': str(best_rank)},
                        },
                    }
                )
                return result

            result_json = root / 'result.json'
            with (
                patch.object(formal_dist.fidelity, 'FIDELITY_ROOT', fidelity_root),
                patch.object(formal_dist.ab, 'AB_ROOT', ab_root),
                patch.object(formal_dist.ab, 'build_base_config', return_value={'supervised': {}}),
                patch.object(formal_dist.ab, 'group_files_by_month', return_value={}),
                patch.object(formal_dist.ab, 'load_all_files', return_value=[]),
                patch.object(formal_dist.ab, 'run_ab6_checkpoint', return_value={'winner': 'best_loss'}),
                patch.object(formal_dist.formal, 'finalize_formal_result', side_effect=fake_finalize_formal_result),
            ):
                formal_dist.execute_single_task(
                    run_name='triplet_formal_run',
                    candidate_arm='arm_a',
                    result_json=result_json,
                    machine_label='desktop',
                )

            self.assertEqual('keep', child_marker.read_text(encoding='utf-8'))
            self.assertEqual('keep', ab_marker.read_text(encoding='utf-8'))
            self.assertTrue(result_json.exists())

    def test_rewrite_repo_paths_rehomes_repo_relative_paths(self):
        remote_repo = Path(r'C:\remote\MahjongAI')
        local_repo = Path(r'C:\local\MahjongAI')
        payload = {
            'state': str(remote_repo / 'logs' / 'sl_fidelity' / 'demo' / 'state.json'),
            'nested': {
                'best_loss': str(remote_repo / 'logs' / 'sl_ab' / 'demo_formal' / 'best_loss.pth'),
                'outside': r'D:\dataset\file.json.gz',
            },
        }

        rewritten = formal_dist.rewrite_repo_paths(payload, remote_repo=remote_repo, local_repo=local_repo)

        self.assertEqual(
            str(local_repo / 'logs' / 'sl_fidelity' / 'demo' / 'state.json'),
            rewritten['state'],
        )
        self.assertEqual(r'D:\dataset\file.json.gz', rewritten['nested']['outside'])

    def test_fetch_remote_tree_preserves_existing_tree_on_scp_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            local_path = root / 'child'
            local_path.mkdir()
            marker = local_path / 'marker.txt'
            marker.write_text('keep', encoding='utf-8')
            worker = formal_dist.WorkerSpec(
                kind='remote',
                label='laptop',
                python='python',
                host='mahjong-laptop',
            )

            with patch.object(
                formal_dist.subprocess,
                'run',
                return_value=subprocess.CompletedProcess([], 1, 'network down'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'network down'):
                    formal_dist.fetch_remote_tree(
                        worker,
                        remote_path=Path(r'C:\runner\child'),
                        local_path=local_path,
                    )

            self.assertEqual('keep', marker.read_text(encoding='utf-8'))
            self.assertEqual([], list(root.glob('.child.sync-*')))

    def test_fetch_remote_tree_installs_complete_tree_atomically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            local_path = root / 'child'
            local_path.mkdir()
            (local_path / 'old.txt').write_text('old', encoding='utf-8')
            remote_path = Path(r'C:\runner\child')
            worker = formal_dist.WorkerSpec(
                kind='remote',
                label='laptop',
                python='python',
                host='mahjong-laptop',
            )

            def fake_scp(command, **_kwargs):
                staged_tree = Path(command[-1]) / remote_path.name
                staged_tree.mkdir()
                (staged_tree / 'new.txt').write_text('new', encoding='utf-8')
                return subprocess.CompletedProcess(command, 0, '')

            with patch.object(formal_dist.subprocess, 'run', side_effect=fake_scp):
                formal_dist.fetch_remote_tree(
                    worker,
                    remote_path=remote_path,
                    local_path=local_path,
                )

            self.assertFalse((local_path / 'old.txt').exists())
            self.assertEqual('new', (local_path / 'new.txt').read_text(encoding='utf-8'))
            self.assertFalse((root / '.child.sync-backup').exists())
            self.assertEqual([], list(root.glob('.child.sync-*')))

    def test_install_staged_tree_restores_old_tree_when_install_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            local_path = root / 'child'
            local_path.mkdir()
            (local_path / 'old.txt').write_text('old', encoding='utf-8')
            missing_staged_tree = root / 'missing-staged-tree'

            with self.assertRaises(FileNotFoundError):
                formal_dist.install_staged_tree(
                    staged_tree=missing_staged_tree,
                    local_path=local_path,
                )

            self.assertEqual('old', (local_path / 'old.txt').read_text(encoding='utf-8'))
            self.assertFalse((root / '.child.sync-backup').exists())

    def test_load_task_result_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / 'result.json'
            result_path.write_text(
                json.dumps(
                    {
                        'schema_version': formal_dist.TASK_RESULT_SCHEMA_VERSION + 1,
                        'round_kind': formal_dist.ROUND_KIND_FORMAL,
                    }
                ),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(RuntimeError, 'unsupported schema version'):
                formal_dist.load_task_result(result_path)

    def test_task_result_identity_checks_seed_and_scale(self):
        task = {
            'run_name': 'run',
            'candidate_arm': 'anchor',
            'child_run_name': 'child',
            'formal_seed': 2001,
            'formal_step_scale': 140.0,
        }
        payload = {
            'run_name': 'run',
            'candidate_arm': 'anchor',
            'child_run_name': 'child',
            'formal_seed': 2002,
            'formal_step_scale': 140.0,
        }

        with self.assertRaisesRegex(RuntimeError, 'formal_seed'):
            formal_dist.validate_task_result_identity(payload, task)

    def test_remote_result_identity_mismatch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = {
                'status': 'running',
                'run_name': 'expected_run',
                'candidate_arm': 'expected_arm',
                'child_run_name': 'expected_child',
                'remote_result_path': r'C:\remote\result.json',
                'local_result_path': str(Path(tmp_dir) / 'result.json'),
            }
            worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
            wrong_payload = {
                'run_name': 'wrong_run',
                'candidate_arm': 'expected_arm',
                'child_run_name': 'expected_child',
            }

            with (
                patch.object(formal_dist, 'fetch_remote_result_file'),
                patch.object(formal_dist, 'load_task_result', return_value=wrong_payload),
                patch.object(formal_dist, 'sync_remote_task_outputs') as sync_outputs,
            ):
                completion = formal_dist.try_complete_remote_task(worker, task)

            self.assertFalse(completion.completed)
            self.assertTrue(completion.invalid_result)
            self.assertIn('identity mismatch', str(completion.error))
            self.assertEqual('running', task['status'])
            sync_outputs.assert_not_called()

    def test_invalid_remote_result_retries_instead_of_polling_forever(self):
        task = {'status': 'running', 'attempts': 1}
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        probe = {
            'reachable': True,
            'status': 'done',
            'result_exists': True,
            'done_exit_code': 0,
        }
        completion = formal_dist.RemoteCompletionAttempt(
            error='task result identity mismatch',
            invalid_result=True,
        )

        with (
            patch.object(formal_dist, 'probe_remote_task', return_value=probe),
            patch.object(formal_dist, 'try_complete_remote_task', return_value=completion),
            patch.object(formal_dist, 'cleanup_remote_task_registration') as cleanup,
        ):
            formal_dist.poll_remote_task(worker=worker, task_state=task, max_attempts=3)

        self.assertEqual('pending', task['status'])
        self.assertEqual(1, task['attempts'])
        self.assertIn('invalid result', task['error'])
        cleanup.assert_called_once_with(worker, task)

    def test_operator_interrupt_stops_detached_task_during_launch_handshake(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'attempts': 2,
            'worker_label': 'laptop',
            'remote_launch_mode': 'interactive_window',
            'remote_detached': True,
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        active_task = SimpleNamespace(worker=worker, task_state=task)
        active = {'laptop': active_task}
        control_state = {
            'workers': {
                'laptop': {'paused': True, 'interrupt_requested': True},
            }
        }

        with (
            patch.object(formal_dist.common_dispatch, 'interrupt_local_active_task') as stop_transport,
            patch.object(formal_dist, 'interrupt_persisted_remote_task', return_value=True) as stop_remote,
        ):
            changed = formal_dist.apply_formal_worker_control_requests(
                control_state=control_state,
                active=active,
                stage_state={'tasks': {'formal__anchor': task}},
                workers_by_label={'laptop': worker},
            )

        self.assertTrue(changed)
        self.assertEqual({}, active)
        stop_transport.assert_called_once_with(active_task)
        stop_remote.assert_called_once_with(worker, task)
        self.assertEqual('pending', task['status'])
        self.assertEqual(1, task['attempts'])
        self.assertFalse(control_state['workers']['laptop']['interrupt_requested'])

    def test_operator_interrupt_uses_verified_stop_for_active_ssh_inline_task(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'attempts': 2,
            'worker_label': 'laptop',
            'remote_launch_mode': 'ssh_inline',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        active_task = SimpleNamespace(worker=worker, task_state=task)
        active = {'laptop': active_task}
        control_state = {
            'workers': {
                'laptop': {'paused': True, 'interrupt_requested': True},
            }
        }

        with (
            patch.object(formal_dist.common_dispatch, 'interrupt_local_active_task') as stop_transport,
            patch.object(formal_dist, 'interrupt_persisted_remote_task', return_value=True) as stop_remote,
        ):
            formal_dist.apply_formal_worker_control_requests(
                control_state=control_state,
                active=active,
                stage_state={'tasks': {'formal__anchor': task}},
                workers_by_label={'laptop': worker},
            )

        stop_transport.assert_called_once_with(active_task)
        stop_remote.assert_called_once_with(worker, task)
        self.assertEqual('pending', task['status'])

    def test_operator_interrupt_keeps_request_when_detached_worker_is_unreachable(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'attempts': 2,
            'worker_label': 'laptop',
            'remote_launch_mode': 'interactive_window',
            'remote_detached': True,
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        active_task = SimpleNamespace(worker=worker, task_state=task)
        active = {'laptop': active_task}
        control_state = {
            'workers': {
                'laptop': {'paused': True, 'interrupt_requested': True},
            }
        }

        with (
            patch.object(formal_dist.common_dispatch, 'interrupt_local_active_task'),
            patch.object(formal_dist, 'interrupt_persisted_remote_task', return_value=False),
        ):
            changed = formal_dist.apply_formal_worker_control_requests(
                control_state=control_state,
                active=active,
                stage_state={'tasks': {'formal__anchor': task}},
                workers_by_label={'laptop': worker},
            )

        self.assertTrue(changed)
        self.assertEqual({}, active)
        self.assertEqual('running', task['status'])
        self.assertEqual(2, task['attempts'])
        self.assertTrue(control_state['workers']['laptop']['interrupt_requested'])
    def test_interrupt_persisted_remote_task_verifies_remote_process_stopped(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
            'remote_task_name': 'MahjongAI-Formal-formal__anchor',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')

        with (
            patch.object(
                formal_dist,
                'run_remote_powershell',
                return_value=subprocess.CompletedProcess([], 0, ''),
            ) as run_remote,
            patch.object(
                formal_dist,
                'probe_remote_task',
                return_value={'reachable': True, 'status': 'running', 'result_exists': False},
            ),
        ):
            interrupted = formal_dist.interrupt_persisted_remote_task(worker, task)

        self.assertFalse(interrupted)
        self.assertIn('still running', task['remote_interrupt_error'])
        interrupt_script = run_remote.call_args.kwargs['script']
        self.assertIn('MahjongAI-Formal-formal__anchor', interrupt_script)
        self.assertIn('MahjongAI-WinnerRefine-formal__anchor', interrupt_script)

    def test_operator_interrupt_accepts_task_that_finished_during_stop(self):
        task = {
            'task_id': 'formal__anchor',
            'status': 'running',
            'attempts': 2,
            'worker_label': 'laptop',
            'remote_launch_mode': 'interactive_window',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        control_state = {
            'workers': {
                'laptop': {'paused': True, 'interrupt_requested': True},
            }
        }

        def finish_during_stop(_worker, task_state):
            task_state['status'] = 'completed'
            return True

        with patch.object(
            formal_dist,
            'interrupt_persisted_remote_task',
            side_effect=finish_during_stop,
        ):
            changed = formal_dist.apply_formal_worker_control_requests(
                control_state=control_state,
                active={},
                stage_state={'tasks': {'formal__anchor': task}},
                workers_by_label={'laptop': worker},
            )

        self.assertTrue(changed)
        self.assertEqual('completed', task['status'])
        self.assertEqual(2, task['attempts'])
        self.assertFalse(control_state['workers']['laptop']['interrupt_requested'])

    def test_probe_remote_task_treats_wmi_failure_as_unknown(self):
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        task = {
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        probe_payload = {
            'status': 'orphaned',
            'started_exists': True,
            'done_exists': False,
            'result_exists': False,
            'process_probe_error': 'access denied',
        }

        with patch.object(
            formal_dist,
            'run_remote_powershell',
            return_value=subprocess.CompletedProcess([], 0, json.dumps(probe_payload)),
        ):
            result = formal_dist.probe_remote_task(worker, task)

        self.assertFalse(result['reachable'])
        self.assertEqual('unreadable', result['status'])
        self.assertIn('access denied', result['error'])

    def test_probe_remote_task_uses_done_marker_when_wmi_probe_fails(self):
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        task = {
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        probe_payload = {
            'status': 'done',
            'started_exists': True,
            'done_exists': True,
            'result_exists': True,
            'process_probe_error': 'access denied',
        }

        with patch.object(
            formal_dist,
            'run_remote_powershell',
            return_value=subprocess.CompletedProcess([], 0, json.dumps(probe_payload)),
        ):
            result = formal_dist.probe_remote_task(worker, task)

        self.assertTrue(result['reachable'])
        self.assertEqual('done', result['status'])

    def test_probe_remote_task_checks_legacy_and_helper_task_names(self):
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        task = {
            'task_id': 'formal__anchor',
            'remote_task_name': 'MahjongAI-Formal-formal__anchor',
            'remote_runtime_root': r'C:\runtime',
            'remote_result_path': r'C:\result.json',
        }
        payload = {
            'status': 'running',
            'started_exists': True,
            'done_exists': False,
            'result_exists': False,
            'process_ids': [123],
            'process_probe_error': None,
        }

        with patch.object(
            formal_dist,
            'run_remote_powershell',
            return_value=subprocess.CompletedProcess([], 0, json.dumps(payload)),
        ) as run_remote:
            result = formal_dist.probe_remote_task(worker, task)

        self.assertTrue(result['reachable'])
        probe_script = run_remote.call_args.kwargs['script']
        self.assertIn('MahjongAI-Formal-formal__anchor', probe_script)
        self.assertIn('MahjongAI-WinnerRefine-formal__anchor', probe_script)
        self.assertIn('$_.ProcessId -ne $PID', probe_script)

    def test_remote_launch_preparation_failure_keeps_task_pending(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir) / 'fidelity'
            run_dir = fidelity_root / 'coordinator'
            dispatch_root = run_dir / 'distributed' / 'formal_dispatch'
            dispatch_root.mkdir(parents=True)
            task = {
                'task_id': 'formal__anchor',
                'candidate_arm': 'anchor',
                'child_run_name': 'child',
                'status': 'pending',
                'attempts': 0,
            }
            state = {
                'status': 'running',
                'stage': 'formal',
                'formal': {'tasks': {'formal__anchor': task}},
            }
            formal_dist.write_dispatch_state(dispatch_root / 'dispatch_state.json', state)
            control = formal_dist.common_dispatch.initialize_dispatch_control_state(
                local_label=None,
                remote_label='laptop',
                remote_launch_mode='interactive_window',
            )
            formal_dist.common_dispatch.write_dispatch_control(
                dispatch_root / 'dispatch_control.json',
                control,
            )
            worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
            args = SimpleNamespace(
                local_only=False,
                remote_only=True,
                run_name='coordinator',
                source_run_name='unused',
                candidate_arm=['anchor'],
                seed_offset=2000,
                formal_step_scale=140.0,
                local_python='python',
                local_label='desktop',
                remote_host='laptop',
                remote_repo=r'C:\runner',
                remote_python='python',
                remote_label='laptop',
                ssh_key=None,
                remote_launch_mode='interactive_window',
                remote_num_workers=4,
                remote_file_batch_size=10,
                remote_prefetch_factor=4,
                remote_val_file_batch_size=7,
                remote_val_prefetch_factor=5,
                max_attempts=3,
                poll_seconds=15.0,
            )

            with (
                patch.object(formal_dist.fidelity, 'FIDELITY_ROOT', fidelity_root),
                patch.object(formal_dist.fidelity, 'acquire_run_lock', return_value=run_dir / 'lock'),
                patch.object(formal_dist.fidelity, 'release_run_lock'),
                patch.object(formal_dist.common_dispatch, 'build_workers', return_value=[worker]),
                patch.object(formal_dist, 'poll_running_remote_tasks', return_value=False),
                patch.object(
                    formal_dist,
                    'launch_task_for_worker',
                    side_effect=RuntimeError('ssh offline'),
                ) as launch_task,
                patch.object(formal_dist.time, 'sleep', side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    formal_dist.run_dispatch(args)

            persisted = json.loads(
                (dispatch_root / 'dispatch_state.json').read_text(encoding='utf-8')
            )
            persisted_task = persisted['formal']['tasks']['formal__anchor']
            self.assertEqual('running', persisted['coordinator_status'])
            self.assertEqual('pending', persisted_task['status'])
            self.assertEqual(0, persisted_task['attempts'])
            self.assertEqual('ssh offline', persisted_task['remote_launch_error'])
            self.assertEqual(1, persisted_task['remote_launch_error_count'])
            launch_task.assert_called_once()

    def test_cleanup_remote_task_unregisters_attempt_scoped_scheduled_task(self):
        worker = formal_dist.WorkerSpec(kind='remote', label='laptop', python='python')
        task = {
            'task_id': 'formal__anchor',
            'remote_task_name': 'MahjongAI-Formal-formal__anchor',
        }

        with patch.object(
            formal_dist,
            'run_remote_powershell',
            return_value=subprocess.CompletedProcess([], 0, ''),
        ) as run_remote:
            cleaned = formal_dist.cleanup_remote_task_registration(worker, task)

        self.assertTrue(cleaned)
        self.assertIn('remote_task_cleaned_at', task)
        self.assertNotIn('remote_cleanup_error', task)
        self.assertEqual(
            [
                'MahjongAI-Formal-formal__anchor',
                'MahjongAI-WinnerRefine-formal__anchor',
            ],
            task['remote_task_cleaned_names'],
        )
        cleanup_script = run_remote.call_args.kwargs['script']
        self.assertIn('MahjongAI-Formal-formal__anchor', cleanup_script)
        self.assertIn('MahjongAI-WinnerRefine-formal__anchor', cleanup_script)

    def test_remote_scheduled_task_names_support_legacy_and_attempt_state(self):
        legacy = {
            'task_id': 'formal__anchor',
            'remote_task_name': 'MahjongAI-Formal-formal__anchor',
        }
        current = {
            'task_id': 'formal__anchor',
            'remote_launch_task_id': 'formal_deadbeef_002_12345678',
            'remote_task_name': 'MahjongAI-WinnerRefine-attempt-specific',
        }

        self.assertEqual(
            [
                'MahjongAI-Formal-formal__anchor',
                'MahjongAI-WinnerRefine-formal__anchor',
            ],
            formal_dist.remote_scheduled_task_names(legacy),
        )
        self.assertEqual(
            [
                'MahjongAI-WinnerRefine-attempt-specific',
                'MahjongAI-WinnerRefine-formal_deadbeef_002_12345678',
            ],
            formal_dist.remote_scheduled_task_names(current),
        )


if __name__ == '__main__':
    unittest.main()

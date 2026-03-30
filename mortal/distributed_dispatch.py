from __future__ import annotations

import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_REMOTE_HOST = 'mahjong-laptop'


@dataclass(frozen=True)
class WorkerSpec:
    kind: str
    label: str
    python: str | None
    host: str | None = None
    repo: str | None = None
    ssh_key: str | None = None


@dataclass(frozen=True)
class JsonTaskLaunchSpec:
    task_id: str
    stage_name: str
    local_result_path: Path
    log_path: Path
    command_args: list[str]
    cwd: Path
    remote_result_path: Path | None = None


@dataclass
class ActiveTask:
    worker: WorkerSpec
    stage_name: str
    task_id: str
    task_state: dict[str, Any]
    process: subprocess.Popen[Any]
    log_path: Path
    local_result_path: Path
    remote_result_path: Path | None = None


def quote_ps(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def path_to_scp_remote(path: str | Path) -> str:
    text = str(path).replace('\\', '/')
    if len(text) >= 2 and text[1] == ':':
        return text
    return text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hostname_fallback(default_label: str = 'desktop') -> str:
    try:
        return socket.gethostname().strip() or default_label
    except Exception:
        return default_label


def summarize_task_status(tasks: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}
    for task in tasks.values():
        status = str(task.get('status', 'pending'))
        counts.setdefault(status, 0)
        counts[status] += 1
    return counts


def build_workers(
    *,
    enable_remote: bool,
    local_python: str,
    local_label: str,
    remote_host: str,
    remote_repo: str,
    remote_python: str,
    remote_label: str,
    ssh_key: str | None,
) -> list[WorkerSpec]:
    workers = [
        WorkerSpec(
            kind='local',
            label=local_label,
            python=local_python,
        )
    ]
    if enable_remote:
        workers.append(
            WorkerSpec(
                kind='remote',
                label=remote_label,
                python=remote_python,
                host=remote_host,
                repo=remote_repo,
                ssh_key=ssh_key,
            )
        )
    return workers


def build_remote_python_command(
    *,
    worker: WorkerSpec,
    script_path: Path,
    remote_result_path: Path,
    command_args: list[str],
) -> list[str]:
    remote_script = Path(worker.repo or str(script_path.parent.parent)) / script_path.name
    quoted_args = ' '.join(quote_ps(str(arg)) for arg in command_args)
    ps_command = (
        f"Set-Location {quote_ps(worker.repo or str(script_path.parent.parent))}; "
        f"& {quote_ps(worker.python or sys.executable)} "
        f"{quote_ps(str(remote_script))} "
        f"{quoted_args} "
        f"--result-json {quote_ps(str(remote_result_path))}"
    )
    command = ['ssh']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.append(worker.host or DEFAULT_REMOTE_HOST)
    command.append(ps_command)
    return command


def launch_json_task(
    worker: WorkerSpec,
    *,
    task_state: dict[str, Any],
    script_path: Path,
    spec: JsonTaskLaunchSpec,
) -> ActiveTask:
    spec.local_result_path.parent.mkdir(parents=True, exist_ok=True)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    if worker.kind == 'local':
        command = [
            worker.python or sys.executable,
            str(script_path),
            *spec.command_args,
            '--result-json',
            str(spec.local_result_path),
        ]
    elif worker.kind == 'remote':
        if spec.remote_result_path is None:
            raise ValueError('remote worker requires remote_result_path')
        command = build_remote_python_command(
            worker=worker,
            script_path=script_path,
            remote_result_path=spec.remote_result_path,
            command_args=spec.command_args,
        )
    else:
        raise ValueError(f'unknown worker kind `{worker.kind}`')
    log_handle = spec.log_path.open('w', encoding='utf-8', newline='\n')
    process = subprocess.Popen(
        command,
        cwd=str(spec.cwd),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    task_state['status'] = 'running'
    task_state['attempts'] = int(task_state.get('attempts', 0)) + 1
    task_state['worker_label'] = worker.label
    task_state['local_result_path'] = str(spec.local_result_path)
    task_state['log_path'] = str(spec.log_path)
    if spec.remote_result_path is not None:
        task_state['remote_result_path'] = str(spec.remote_result_path)
    if worker.kind == 'local':
        task_state['pid'] = process.pid
    return ActiveTask(
        worker=worker,
        stage_name=spec.stage_name,
        task_id=spec.task_id,
        task_state=task_state,
        process=process,
        log_path=spec.log_path,
        local_result_path=spec.local_result_path,
        remote_result_path=spec.remote_result_path,
    )


def fetch_remote_result(worker: WorkerSpec, remote_result_path: str | Path, local_result_path: Path) -> None:
    local_result_path.parent.mkdir(parents=True, exist_ok=True)
    command = ['scp']
    if worker.ssh_key:
        command.extend(['-i', worker.ssh_key])
    command.extend(
        [
            f'{worker.host}:{path_to_scp_remote(remote_result_path)}',
            str(local_result_path),
        ]
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f'scp failed: {completed.stdout.strip()}')


def mark_task_failed(task_state: dict[str, Any], message: str, *, max_attempts: int, finished_at: str) -> None:
    if int(task_state.get('attempts', 0)) < max_attempts:
        task_state['status'] = 'pending'
    else:
        task_state['status'] = 'failed'
    task_state['error'] = message
    task_state['finished_at'] = finished_at


def handle_finished_json_task(
    *,
    active: ActiveTask,
    max_attempts: int,
    finished_at: str,
    validate_result: Callable[[Path], Any],
) -> None:
    task_state = active.task_state
    return_code = active.process.returncode
    if return_code != 0:
        mark_task_failed(
            task_state,
            f'worker `{active.worker.label}` exited with code {return_code}; see {active.log_path}',
            max_attempts=max_attempts,
            finished_at=finished_at,
        )
        return
    local_result_path = active.local_result_path
    if active.worker.kind == 'remote':
        if active.remote_result_path is None:
            raise ValueError('remote active task requires remote_result_path')
        fetch_remote_result(active.worker, active.remote_result_path, local_result_path)
    if not local_result_path.exists():
        mark_task_failed(
            task_state,
            f'missing task result json at {local_result_path}',
            max_attempts=max_attempts,
            finished_at=finished_at,
        )
        return
    validate_result(local_result_path)
    task_state['status'] = 'completed'
    task_state['finished_at'] = finished_at
    task_state.pop('error', None)

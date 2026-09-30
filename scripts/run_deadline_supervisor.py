"""Own a bounded set of trusted workloads; standard-library only.

See docs/agent/deadline-supervisor.md. Never recover ownership from saved PIDs.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

UTC = dt.timezone.utc


def utc_now():
    return dt.datetime.now(UTC)


def parse_deadline(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('deadline requires an explicit UTC offset')
    return result.astimezone(UTC)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity(cwd):
    commit = subprocess.check_output(['git', '-C', str(cwd), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    dirty = subprocess.check_output(['git', '-C', str(cwd), 'status', '--porcelain'], text=True, timeout=10).strip()
    return commit, bool(dirty)


def verify_role(role):
    commit, dirty = source_identity(role['cwd'])
    if commit != role['source_commit']:
        raise ValueError(f"source commit changed before launch: {role['name']}")
    if dirty and not role['allow_dirty']:
        raise ValueError(f"dirty source refused before launch: {role['name']}")
    if role['config']:
        actual = hashlib.sha256(Path(role['config']['path']).read_bytes()).hexdigest()
        if actual != role['config']['sha256']:
            raise ValueError(f"configuration changed before launch: {role['name']}")


def prepare(spec_path, output):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    deadline = parse_deadline(spec['deadline_utc'])
    grace = float(spec.get('grace_seconds', 120))
    if not math.isfinite(grace) or grace < 0:
        raise ValueError('grace_seconds must be finite and nonnegative')
    if utc_now().timestamp() >= deadline.timestamp() - grace:
        raise ValueError('lease expired or already inside graceful shutdown margin')
    roles = []
    names = set()
    for role in spec['roles']:
        name = role['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('role names must be unique safe filename components')
        names.add(name)
        completion_role = role.get('stop_when_complete', False)
        if not isinstance(completion_role, bool):
            raise ValueError('stop_when_complete must be an explicit boolean')
        argv = role['argv']
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or '\0' in x for x in argv):
            raise ValueError('argv must be a nonempty array of strings, never shell text')
        cwd = Path(role['cwd']).resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError('cwd must be a directory')
        # Resolve an explicit executable, not a shell or PATH alias.
        executable = Path(argv[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError('argv[0] must name an existing absolute executable path')
        commit, dirty = source_identity(cwd)
        allow_dirty = spec.get('allow_dirty', False)
        if not isinstance(allow_dirty, bool):
            raise ValueError('allow_dirty must be an explicit boolean')
        if dirty and not allow_dirty:
            raise ValueError(f'dirty source refused: {cwd}; use a clean committed checkout')
        expected_commit = role.get('expected_commit', spec.get('expected_commit'))
        if expected_commit is not None and expected_commit != commit:
            raise ValueError(f'expected_commit mismatch: {name}; actual HEAD={commit}')
        config = None
        if role.get('config'):
            config_path = Path(role['config']).resolve(strict=True)
            config = {'path': str(config_path), 'sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
        roles.append({'name': name, 'argv': argv, 'cwd': str(cwd), 'source_commit': commit,
                      'stop_when_complete': completion_role,
                      'source_dirty': dirty, 'allow_dirty': allow_dirty, 'expected_commit': expected_commit, 'config': config,
                      'stdout': str(output / (name + '.stdout.log')),
                      'stderr': str(output / (name + '.stderr.log')),
                      'result': str(output / (name + '.result.json'))})
    if not roles:
        raise ValueError('at least one role is required')
    if os.name != 'nt' and spec.get('posix_process_group_acknowledged') is not True:
        raise ValueError('POSIX requires posix_process_group_acknowledged=true; escaping groups is unsupported')
    return {'schema_version': 1, 'run_id': uuid.uuid4().hex, 'created_utc': utc_now().isoformat(),
            'deadline_utc': deadline.isoformat(), 'grace_seconds': grace, 'roles': roles,
            'spec_path': str(spec_path), 'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            'stop_file': str(output / 'STOP'), 'output': str(output), 'owner_pid': None,
            'status': 'prepared', 'checkpoint_status': 'unverified; never implies exact resume'}


class WindowsJob:
    """Uninherited kill-on-close job; bootstrap is assigned before its launch gate opens."""
    def __init__(self):
        from ctypes import wintypes as w
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                        ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount',
                         'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', Io),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.SetInformationJobObject.restype = w.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = w.BOOL
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.TerminateJobObject.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        # Popen's Windows handle references the exact process object, immune to PID reuse.
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def bootstrap(manifest_path, role_name):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    role = next(r for r in manifest['roles'] if r['name'] == role_name)
    # EOF before GO means owner disappeared before establishing ownership.
    if sys.stdin.buffer.readline() != b'GO\n':
        return 2
    if utc_now() >= parse_deadline(manifest['deadline_utc']):
        return 2
    try:
        verify_role(role)
        if utc_now() >= parse_deadline(manifest['deadline_utc']):
            raise ValueError('deadline reached before workload creation')
    except Exception as exc:
        atomic_json(role['result'], {'returncode': None, 'error': str(exc), 'finished_utc': utc_now().isoformat()})
        sys.stdin.buffer.read()
        if os.name != 'nt':
            os.killpg(os.getpgrp(), signal.SIGKILL)
        return 2
    env = dict(os.environ, MORTAL_STOP_FILE=manifest['stop_file'],
               MORTAL_ORACLE_PAUSE_FILE=manifest['stop_file'],
               MORTAL_RUN_ID=manifest['run_id'], MORTAL_RUN_DEADLINE_UTC=manifest['deadline_utc'])
    if role['config']:
        env['MORTAL_CFG'] = role['config']['path']
    try:
        child = subprocess.Popen(role['argv'], cwd=role['cwd'], env=env, stdin=subprocess.DEVNULL,
                                 close_fds=True, shell=False)
    except Exception as exc:
        atomic_json(role['result'], {'returncode': None, 'error': str(exc), 'finished_utc': utc_now().isoformat()})
    else:
        atomic_json(Path(manifest['output']) / (role_name + '.started.json'), {'pid': child.pid, 'started_utc': utc_now().isoformat()})
        def reap():
            code = child.wait()
            atomic_json(role['result'], {'returncode': code, 'finished_utc': utc_now().isoformat()})
        threading.Thread(target=reap, daemon=True).start()
    # Keep the leader alive and unreaped even after command exit. Its PID/PGID cannot
    # be reused while the owner has this live anchor. A broken owner pipe kills group.
    sys.stdin.buffer.read()
    if os.name != 'nt':
        os.killpg(os.getpgrp(), signal.SIGKILL)
    return 0  # Windows owner job close already kills the entire job.


def supervise(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['owner_pid'] = os.getpid()
    manifest['owner_started_utc'] = utc_now().isoformat()
    deadline = parse_deadline(manifest['deadline_utc']).timestamp()
    # Monotonic ceiling prevents a backwards wall-clock adjustment extending lease.
    hard_monotonic = time.monotonic() + max(0, deadline - time.time())
    grace = manifest['grace_seconds']
    processes, streams = [], []
    job = None
    reason = None
    interrupted = False
    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def remaining():
        return min(deadline - time.time(), hard_monotonic - time.monotonic())
    def persist():
        atomic_json(manifest_path, manifest)
    def stop_request():
        Path(manifest['stop_file']).touch(exist_ok=True)
    try:
        if remaining() <= grace:
            reason = 'refused_late_start'
            return 2
        if os.name == 'nt':
            job = WindowsJob()
        manifest['status'] = 'starting'
        persist()
        for role in manifest['roles']:
            if remaining() <= grace:
                reason = 'deadline'
                break
            out = open(role['stdout'], 'xb')
            streams.append(out)
            err = open(role['stderr'], 'xb')
            streams.append(err)
            kwargs = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_bootstrap',
                                        str(manifest_path), role['name']], stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, close_fds=True, **kwargs)
            processes.append(process)
            role['bootstrap_pid'] = process.pid
            if job:
                job.assign(process)
            persist()  # Record ownership before opening the workload gate.
            verify_role(role)
            if remaining() <= grace:
                reason = 'deadline'
                break
            process.stdin.write(b'GO\n')
            process.stdin.flush()
        manifest['status'] = 'running'
        persist()
        stopping_at = None
        while True:
            results = []
            for role in manifest['roles']:
                path = Path(role['result'])
                results.append(json.loads(path.read_text(encoding='utf-8')) if path.exists() else None)
            if Path(manifest['stop_file']).exists():
                reason = reason or 'interrupted'
            if any(result and result.get('returncode') not in ((0, 75, 87) if reason is not None else (0,))
                   for result in results):
                reason = reason or 'external_failure'
            completion_results = [(role['name'], result) for role, result in zip(manifest['roles'], results)
                                  if role.get('stop_when_complete', False)]
            if reason is None and completion_results and all(
                result is not None and result.get('returncode') == 0
                for _, result in completion_results
            ):
                reason = 'completed'
                manifest['completion_roles'] = [name for name, _ in completion_results]
            # Anchors must remain alive. Do not reap them before group cleanup.
            if interrupted:
                reason = reason or 'interrupted'
            left = remaining()
            if left <= grace:
                reason = reason or 'deadline'
            if reason and stopping_at is None:
                stopping_at = time.monotonic()
                stop_request()
                manifest['status'] = 'stopping'
                manifest['stop_reason'] = reason
                persist()
            if all(result is not None for result in results):
                reason = reason or 'completed'
                manifest['graceful_exit'] = True
                break
            if left <= 0 or (stopping_at is not None and time.monotonic() - stopping_at >= grace):
                manifest['forced_stop'] = True
                reason = reason or 'deadline'
                break
            time.sleep(min(0.1, max(0.001, left)))
        return 0 if reason == 'completed' else (124 if reason == 'deadline' else 1)
    except Exception as exc:
        reason = 'supervisor_failure'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        return 1
    finally:
        # Kernel ownership handles on Windows; unreaped session leaders on POSIX.
        # Never recover from a manifest PID or kill unrelated interpreters.
        if job:
            job.close()
        for process in processes:
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                # Assignment may have failed; the gate is still closed in that case.
                if process.poll() is None:
                    process.kill()
            if process.stdin:
                process.stdin.close()
            process.wait()
        for stream in streams:
            stream.close()
        manifest['status'] = reason or 'supervisor_failure'
        manifest['role_results'] = {}
        for role in manifest['roles']:
            result_path = Path(role['result'])
            manifest['role_results'][role['name']] = (json.loads(result_path.read_text(encoding='utf-8'))
                                                       if result_path.exists() else {'returncode': None, 'result': 'incomplete'})
        manifest['finished_utc'] = utc_now().isoformat()
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run')
    run.add_argument('--spec', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--detach', action='store_true')
    run.add_argument('--dry-run', action='store_true')
    worker = sub.add_parser('_supervise')
    worker.add_argument('manifest')
    boot = sub.add_parser('_bootstrap')
    boot.add_argument('manifest')
    boot.add_argument('role')
    args = parser.parse_args()
    if args.action == '_bootstrap':
        return bootstrap(args.manifest, args.role)
    if args.action == '_supervise':
        return supervise(args.manifest)
    output = Path(args.output).resolve()
    manifest = prepare(args.spec, output)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite/reuse a lease.
    path = output / 'manifest.json'
    atomic_json(path, manifest)
    if not args.detach:
        return supervise(path)
    with (output / 'supervisor.log').open('xb') as log:
        flags = {'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_supervise', str(path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True, **flags)
    print(json.dumps({'run_id': manifest['run_id'], 'supervisor_pid': child.pid, 'manifest': str(path),
                      'status': 'launch_requested; inspect manifest for actual startup'}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f'deadline supervisor: {exc}', file=sys.stderr)
        raise SystemExit(2)

"""Bind a single fork command to the existing laptop resource/APEX/QoS guard."""
import argparse
from datetime import datetime, timezone
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mortal.core.artifacts import atomic_write_json, atomic_write_text, file_sha256, read_shared_text


def read_guard_json(path):
    return json.loads(read_shared_text(path))


def write_guard_json(directory, name, value):
    # Reuse bounded WinError 5/32 replacement retries. Persistent failures still
    # propagate into the unchanged guard failure/STOP path; never report success.
    atomic_write_text(Path(directory) / name, json.dumps(value, indent=2, allow_nan=False) + '\n')


def existing_guard(path, expected, directory):
    if file_sha256(path) != expected:
        raise ValueError('existing guard source changed')
    spec = importlib.util.spec_from_file_location('existing_sl_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.D = Path(directory)
    module.LEASE = module.D / 'lease'
    module.read = read_guard_json
    module.put = partial(write_guard_json, module.D)
    return module


def worker(base, directory):
    pause = Path(os.environ['MORTAL_ORACLE_PAUSE_FILE'])
    until = time.monotonic() + 30
    while not (directory / 'guard_ready.json').exists():
        if pause.exists() or time.monotonic() >= until:
            raise RuntimeError('existing resource guard did not become ready')
        time.sleep(.2)
    spec = read_guard_json(directory / 'training_command.json')
    if file_sha256(spec['config']) != spec['config_sha256']:
        raise ValueError('prepared C configuration changed')
    qos = base.module('existing_sl_qos', base.QOS_TOOL)

    def high(pid):
        handle = qos.Process(pid)
        try:
            handle.high()
            observed = handle.read()
            if not observed['control'] & 1 or observed['state'] & 1:
                raise RuntimeError('existing HighQoS setting did not apply')
            return observed
        finally:
            handle.close()

    high(os.getpid())
    child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], stdin=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    observed = high(child.pid)
    atomic_write_json(directory / 'trainer.started.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'pid': child.pid,
        'argv': spec['argv'], 'qos': observed, 'target_successful_updates': spec['updates']})
    code = child.wait()
    atomic_write_json(directory / 'trainer.result.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'returncode': code})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guard', 'worker'))
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--guard-path', required=True, type=Path)
    parser.add_argument('--guard-sha256', required=True)
    args = parser.parse_args()
    guard = existing_guard(args.guard_path, args.guard_sha256, args.directory)
    raise SystemExit(guard.guard() if args.mode == 'guard' else worker(guard, args.directory))

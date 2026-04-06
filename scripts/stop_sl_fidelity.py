from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


DEFAULT_RUN_NAME = 'sl_fidelity_main'
RUN_NAME_FLAG_RE = re.compile(r'--run-name(?:\s+|=)(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
RUN_NAME_FLAG_PRESENT_RE = re.compile(r'--run-name(?:\s|=)')


def command_targets_run(command_line: str | None, run_name: str) -> bool:
    if not command_line or 'run_sl_fidelity.py' not in command_line:
        return False
    match = RUN_NAME_FLAG_RE.search(command_line)
    if match is not None:
        explicit_run_name = next(group for group in match.groups() if group is not None)
        return explicit_run_name == run_name
    return run_name == DEFAULT_RUN_NAME and RUN_NAME_FLAG_PRESENT_RE.search(command_line) is None


def query_sl_processes() -> list[dict[str, object]]:
    command = (
        "$procs = Get-CimInstance Win32_Process | Where-Object { "
        + "$_.Name -match '^python(\\.exe)?$' -and "
        + "$_.CommandLine -match 'run_sl_fidelity\\.py' "
        + "}; "
        + "$procs | Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ['powershell', '-NoProfile', '-Command', command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'failed to query processes')
    raw = result.stdout.strip()
    if not raw:
        return []
    payload = json.loads(raw)
    if isinstance(payload, list):
        return payload
    return [payload]


def find_matching_pids(run_name: str) -> list[int]:
    return [
        int(proc['ProcessId'])
        for proc in query_sl_processes()
        if command_targets_run(proc.get('CommandLine'), run_name)
    ]


def kill_process_tree(pid: int) -> tuple[int, str]:
    result = subprocess.run(
        ['taskkill', '/PID', str(pid), '/T', '/F'],
        capture_output=True,
        text=True,
        check=False,
    )
    output = '\n'.join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return result.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=DEFAULT_RUN_NAME)
    args = parser.parse_args()

    pids = find_matching_pids(args.run_name)
    if not pids:
        print(json.dumps({'run_name': args.run_name, 'killed': [], 'status': 'no_process'}, ensure_ascii=False))
        return 0

    killed: list[dict[str, object]] = []
    overall_ok = True
    for pid in sorted(set(pids)):
        return_code, output = kill_process_tree(pid)
        ok = return_code == 0
        overall_ok = overall_ok and ok
        killed.append({'pid': pid, 'ok': ok, 'output': output})

    print(json.dumps({'run_name': args.run_name, 'killed': killed, 'status': 'ok' if overall_ok else 'partial_failure'}, ensure_ascii=False, indent=2))
    return 0 if overall_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

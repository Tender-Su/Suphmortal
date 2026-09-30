from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from probe_train_resources import (
    atomic_write_json,
    atomic_write_text,
    query_gpu_metrics,
    stop_process_tree,
    summarize_numeric,
    sys_executable,
)
from probe_oracle_critic_resources import merge_windows_and_tree_metrics
from mortal.core.toml_utils import load_toml_file


CDE_ARMS = frozenset({"C", "D", "E"})

TOTAL_STEPS_RE = re.compile(r"total steps:\s*([0-9,]+)")
MAX_STEPS_RE = re.compile(r"reached configured max steps=([0-9,]+)")
SUBMIT_RE = re.compile(r"param has been submitted: version=([0-9]+)")
CLIENT_RANK_RE = re.compile(
    r"trainee rankings:\s*\[([^\]]+)\]\s*\(([0-9.eE+-]+),\s*([0-9.eE+-]+)pt\)"
)
ERROR_RE = re.compile(r"\b(?:ERROR|CRITICAL|Traceback|RuntimeError|ValueError|FileNotFoundError)\b")
SHUTDOWN_DISCONNECT_RE = re.compile(
    r"Traceback \(most recent call last\):.*?ConnectionResetError: \[WinError 10054\].*",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Oracle critic C/D/E online probe with role logs and resource monitoring.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--python-exe", default=sys_executable())
    parser.add_argument(
        "--arm",
        default="current_config",
        help=(
            "Oracle experiment arm for online_role_runner. Passing C/D/E is accepted "
            "as a CDE label and runs the config as current_config."
        ),
    )
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--server-start-wait-sec", type=float, default=4.0)
    parser.add_argument("--trainer-start-wait-sec", type=float, default=8.0)
    parser.add_argument("--max-system-mem-percent", type=float, default=85.0)
    parser.add_argument("--max-gpu-mem-mb", type=float, default=15000.0)
    parser.add_argument("--hard-system-mem-percent", type=float, default=92.0)
    parser.add_argument("--hard-gpu-mem-mb", type=float, default=16000.0)
    parser.add_argument("--resource-breach-samples", type=int, default=3)
    parser.add_argument("--max-run-sec", type=float, default=0.0)
    return parser.parse_args()


def configured_cde_arm(config_path: Path) -> str:
    cfg = load_toml_file(config_path)
    cde_cfg = cfg.get("oracle_cde_experiment", {})
    if not isinstance(cde_cfg, dict):
        return ""
    return str(cde_cfg.get("arm", "") or "").strip().upper()


def normalize_runner_arm(config_path: Path, arm: str | None) -> str:
    value = str(arm or "current_config").strip()
    if value.upper() not in CDE_ARMS:
        return value or "current_config"

    config_arm = configured_cde_arm(config_path)
    if config_arm and value.upper() != config_arm:
        raise ValueError(
            f"--arm {value!r} does not match oracle_cde_experiment.arm={config_arm!r}"
        )
    return "current_config"


def configured_max_steps(config_path: Path) -> int:
    try:
        cfg = load_toml_file(config_path)
        scheduler = cfg.get("optim", {}).get("scheduler", {})
        if isinstance(scheduler, dict):
            return max(int(scheduler.get("max_steps", 0) or 0), 0)
    except Exception:
        return 0
    return 0


def read_stdout(role: str, pipe, chunk_queue: queue.Queue[tuple[str, bytes]]) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            chunk_queue.put((role, chunk))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def launch_role(
    *,
    role: str,
    args: argparse.Namespace,
    env: dict[str, str],
    runner_arm: str,
) -> subprocess.Popen:
    cmd = [
        args.python_exe,
        "-m",
        "mortal.online.online_role_runner",
        role,
        "--config",
        str(Path(args.config).resolve()),
        "--arm",
        runner_arm,
    ]
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        bufsize=0,
    )


def latest_int(pattern: re.Pattern[str], text: str) -> int:
    matches = pattern.findall(text)
    if not matches:
        return 0
    return int(str(matches[-1]).replace(",", ""))


def parse_client_rank(text: str) -> dict[str, Any] | None:
    matches = CLIENT_RANK_RE.findall(text)
    if not matches:
        return None
    counts_text, avg_rank, avg_pt = matches[-1]
    counts = [int(item) for item in counts_text.split() if item.strip().lstrip("-").isdigit()]
    return {
        "rankings": counts,
        "avg_rank": float(avg_rank),
        "avg_pt": float(avg_pt),
    }


def console_has_error(text: str, *, allow_shutdown_disconnect: bool = False) -> bool:
    if not ERROR_RE.search(text):
        return False
    if allow_shutdown_disconnect:
        text = SHUTDOWN_DISCONNECT_RE.sub("", text)
    return bool(ERROR_RE.search(text))


def parse_console_summary(
    console: dict[str, str],
    *,
    allow_shutdown_disconnect: bool = False,
) -> dict[str, Any]:
    trainer = console.get("trainer", "")
    client = console.get("client", "")
    server = console.get("server", "")
    return {
        "latest_step": max(
            latest_int(TOTAL_STEPS_RE, trainer),
            latest_int(MAX_STEPS_RE, trainer),
        ),
        "last_submit_version": latest_int(SUBMIT_RE, trainer),
        "last_client_rank": parse_client_rank(client),
        "trainer_error": console_has_error(trainer),
        "client_error": console_has_error(
            client,
            allow_shutdown_disconnect=allow_shutdown_disconnect,
        ),
        "server_error": console_has_error(
            server,
            allow_shutdown_disconnect=allow_shutdown_disconnect,
        ),
    }


def write_role_consoles(log_dir: Path, console: dict[str, str]) -> None:
    for role, text in console.items():
        atomic_write_text(log_dir / f"{role}.console.live.txt", text.replace("\r", "\n"))


def process_snapshot(procs: dict[str, subprocess.Popen]) -> tuple[dict[str, Any], set[int]]:
    pids: set[int] = set()
    role_snapshots: dict[str, Any] = {}
    totals = {
        "process_count": 0,
        "tree_cpu_percent": 0.0,
        "tree_rss_gb": 0.0,
    }
    system_cpu = None
    system_mem_used = None
    system_mem_percent = None

    for role, proc in procs.items():
        if proc.poll() is not None:
            continue
        metrics, role_pids = merge_windows_and_tree_metrics(proc.pid)
        role_snapshots[role] = metrics
        pids.update(role_pids)
        for key in ("process_count", "tree_cpu_percent", "tree_rss_gb"):
            value = metrics.get(key)
            if value is not None:
                totals[key] += float(value)
        for key, current in (
            ("system_cpu_percent", system_cpu),
            ("system_mem_used_gb", system_mem_used),
            ("system_mem_percent", system_mem_percent),
        ):
            value = metrics.get(key)
            if value is None:
                continue
            value = float(value)
            if current is not None and value <= current:
                continue
            if key == "system_cpu_percent":
                system_cpu = value
            elif key == "system_mem_used_gb":
                system_mem_used = value
            else:
                system_mem_percent = value

    snapshot: dict[str, Any] = {
        "process_count": int(totals["process_count"]),
        "tree_cpu_percent": totals["tree_cpu_percent"],
        "tree_rss_gb": totals["tree_rss_gb"],
        "system_cpu_percent": system_cpu,
        "system_mem_used_gb": system_mem_used,
        "system_mem_percent": system_mem_percent,
        "roles": role_snapshots,
    }
    snapshot.update(query_gpu_metrics(pids))
    return snapshot, pids


def stop_all(procs: dict[str, subprocess.Popen]) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            stop_process_tree(proc.pid)


def resource_stop_reason(
    *,
    sample: dict[str, Any],
    reached_target: bool,
    args: argparse.Namespace,
    breach_counts: dict[str, int],
) -> str:
    if reached_target:
        breach_counts.clear()
        return ""

    mem_percent = sample.get("system_mem_percent")
    gpu_mem = sample.get("gpu_mem_used_mb")
    if (
        args.hard_system_mem_percent > 0
        and mem_percent is not None
        and float(mem_percent) >= args.hard_system_mem_percent
    ):
        return f"system_mem_percent>={args.hard_system_mem_percent}"
    if (
        args.hard_gpu_mem_mb > 0
        and gpu_mem is not None
        and float(gpu_mem) >= args.hard_gpu_mem_mb
    ):
        return f"gpu_mem_mb>={args.hard_gpu_mem_mb}"

    required = max(int(args.resource_breach_samples), 1)
    checks = (
        ("system_mem_percent", mem_percent, args.max_system_mem_percent),
        ("gpu_mem_mb", gpu_mem, args.max_gpu_mem_mb),
    )
    for key, value, limit in checks:
        if limit > 0 and value is not None and float(value) >= float(limit):
            breach_counts[key] = breach_counts.get(key, 0) + 1
            if breach_counts[key] >= required:
                return f"{key}>={limit}x{required}"
        else:
            breach_counts[key] = 0
    return ""


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config does not exist: {config_path}")

    run_dir = config_path.parent
    log_dir = run_dir / "online_run"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.name or run_dir.name
    max_steps = configured_max_steps(config_path)
    runner_arm = normalize_runner_arm(config_path, args.arm)

    env = os.environ.copy()
    env["MORTAL_CFG"] = str(config_path)
    env["MORTAL_ORACLE_ARM"] = runner_arm

    procs: dict[str, subprocess.Popen] = {}
    console: dict[str, str] = {"server": "", "trainer": "", "client": ""}
    raw_chunks: dict[str, list[bytes]] = {"server": [], "trainer": [], "client": []}
    chunk_queue: queue.Queue[tuple[str, bytes]] = queue.Queue()
    samples: list[dict[str, Any]] = []
    stop_reason = ""
    resource_breach_counts: dict[str, int] = {}
    started_at = time.time()
    last_sample_at = 0.0

    try:
        procs["server"] = launch_role(role="server", args=args, env=env, runner_arm=runner_arm)
        threading.Thread(
            target=read_stdout,
            args=("server", procs["server"].stdout, chunk_queue),
            daemon=True,
        ).start()
        time.sleep(args.server_start_wait_sec)

        procs["trainer"] = launch_role(role="trainer", args=args, env=env, runner_arm=runner_arm)
        threading.Thread(
            target=read_stdout,
            args=("trainer", procs["trainer"].stdout, chunk_queue),
            daemon=True,
        ).start()
        time.sleep(args.trainer_start_wait_sec)

        procs["client"] = launch_role(role="client", args=args, env=env, runner_arm=runner_arm)
        threading.Thread(
            target=read_stdout,
            args=("client", procs["client"].stdout, chunk_queue),
            daemon=True,
        ).start()

        while True:
            drained = False
            while True:
                try:
                    role, chunk = chunk_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                raw_chunks[role].append(chunk)
                console[role] += chunk.decode("utf-8", errors="ignore")

            now = time.time()
            parsed = parse_console_summary(console)
            if now - last_sample_at >= args.sample_interval:
                metrics, _ = process_snapshot(procs)
                sample = {
                    "elapsed_sec": round(now - started_at, 3),
                    **parsed,
                    **metrics,
                    "returncodes": {
                        role: proc.poll()
                        for role, proc in procs.items()
                    },
                }
                samples.append(sample)
                last_sample_at = now
                write_role_consoles(log_dir, console)
                atomic_write_json(
                    log_dir / "live.json",
                    {
                        "name": run_name,
                        "elapsed_sec": sample["elapsed_sec"],
                        "stop_reason": stop_reason,
                        "latest_step": parsed["latest_step"],
                        "last_client_rank": parsed["last_client_rank"],
                        "last_sample": sample,
                    },
                )

                reached_target = max_steps > 0 and int(parsed["latest_step"]) >= max_steps
                resource_reason = resource_stop_reason(
                    sample=sample,
                    reached_target=reached_target,
                    args=args,
                    breach_counts=resource_breach_counts,
                )
                if resource_reason:
                    stop_reason = resource_reason
                    stop_all(procs)

            trainer_code = procs["trainer"].poll() if "trainer" in procs else None
            if trainer_code is not None:
                stop_reason = stop_reason or "trainer_exit"
                stop_all({k: v for k, v in procs.items() if k != "trainer"})
                break

            if args.max_run_sec > 0 and now - started_at >= args.max_run_sec:
                stop_reason = "max_run_sec"
                stop_all(procs)
                break

            if any(proc.poll() not in (None, 0) for role, proc in procs.items() if role != "trainer"):
                stop_reason = "role_exit"
                stop_all(procs)
                break

            if not drained:
                time.sleep(0.2)
    finally:
        stop_all(procs)
        time.sleep(1.0)
        while True:
            try:
                role, chunk = chunk_queue.get_nowait()
            except queue.Empty:
                break
            raw_chunks[role].append(chunk)
            console[role] += chunk.decode("utf-8", errors="ignore")

    elapsed_sec = time.time() - started_at
    write_role_consoles(log_dir, console)
    for role, chunks in raw_chunks.items():
        (log_dir / f"{role}.console.bin").write_bytes(b"".join(chunks))
        atomic_write_text(log_dir / f"{role}.console.txt", console[role].replace("\r", "\n"))

    atomic_write_text(
        log_dir / "samples.jsonl",
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
    )

    reached_configured_target = (
        max_steps > 0
        and latest_int(MAX_STEPS_RE, console.get("trainer", "")) >= max_steps
    )
    parsed = parse_console_summary(
        console,
        allow_shutdown_disconnect=stop_reason == "trainer_exit" and reached_configured_target,
    )
    summary: dict[str, Any] = {
        "name": run_name,
        "config_path": str(config_path),
        "log_dir": str(log_dir),
        "elapsed_sec": elapsed_sec,
        "stop_reason": stop_reason,
        "returncodes": {role: proc.poll() for role, proc in procs.items()},
        "latest_step": parsed["latest_step"],
        "last_submit_version": parsed["last_submit_version"],
        "last_client_rank": parsed["last_client_rank"],
        "errors": {
            "server": parsed["server_error"],
            "trainer": parsed["trainer_error"],
            "client": parsed["client_error"],
        },
        "sample_count": len(samples),
        "resource_summary_all": {},
    }
    if elapsed_sec > 0:
        summary["steps_per_sec_all"] = parsed["latest_step"] / elapsed_sec
    for key in (
        "gpu_util_percent",
        "gpu_mem_util_percent",
        "gpu_mem_used_mb",
        "tree_gpu_mem_mb",
        "system_cpu_percent",
        "tree_cpu_percent",
        "tree_rss_gb",
        "system_mem_used_gb",
        "system_mem_percent",
        "gpu_temperature_c",
    ):
        stats = summarize_numeric(samples, key)
        if stats is not None:
            summary["resource_summary_all"][key] = stats
    atomic_write_json(log_dir / "summary.json", summary)
    atomic_write_json(
        log_dir / "live.json",
        {
            "name": run_name,
            "elapsed_sec": round(elapsed_sec, 3),
            "stop_reason": stop_reason,
            "latest_step": parsed["latest_step"],
            "last_client_rank": parsed["last_client_rank"],
            "returncodes": summary["returncodes"],
            "summary_path": str(log_dir / "summary.json"),
        },
    )
    return summary


def main() -> None:
    summary = run_probe(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

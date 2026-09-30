from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from mortal.core.toml_utils import load_toml_file, write_toml_file
from probe_train_resources import (
    atomic_write_json,
    atomic_write_text,
    POWERSHELL_EXE,
    query_gpu_metrics,
    query_windows_metrics,
    read_stdout,
    stop_process_tree,
    summarize_numeric,
    sys_executable,
)


STEP_RE = re.compile(r"step=(\d+)\s+(?:train_loss|val_loss)=")
TRAIN_RE = re.compile(
    r"step=(\d+)\s+train_loss=([0-9.eE+-]+)\s+objective_loss=([0-9.eE+-]+).*?"
    r"corr=([0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"step=(\d+)\s+val_loss=([0-9.eE+-]+)\s+val_mae=([0-9.eE+-]+)\s+"
    r"val_corr=([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Oracle critic pretrain throughput and CPU/GPU/memory pressure."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--base-config", default="mortal/config.toml")
    parser.add_argument("--output-root", default="logs/oracle_critic_resource_probe")
    parser.add_argument("--python-exe", default=sys_executable())
    parser.add_argument("--critic-arch", choices=("single_tower", "dual_tower"), default="dual_tower")
    parser.add_argument("--train-scope", default="all")
    parser.add_argument("--init-state-file", default="")
    parser.add_argument("--teacher-state-file", default="")
    parser.add_argument("--teacher-loss-weight", type=float, default=None)
    parser.add_argument("--target-loss-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--file-batch-size", type=int, default=10)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--val-file-batch-size", type=int, default=8)
    parser.add_argument("--val-prefetch-factor", type=int, default=5)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--target-step", type=int, default=240)
    parser.add_argument("--steady-step", type=int, default=120)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=1000000)
    parser.add_argument("--val-every-steps", type=int, default=1000000)
    parser.add_argument("--dependency-val-every-steps", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--max-train-files", type=int, default=256)
    parser.add_argument("--max-val-files", type=int, default=64)
    parser.add_argument(
        "--full-file-pool",
        action="store_true",
        help="Use the base config train/val file pool instead of probe-size file caps.",
    )
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--post-target-wait-sec", type=float, default=4.0)
    parser.add_argument("--fresh", action="store_true", default=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the case config without launching the resource monitor.",
    )
    return parser.parse_args()


def latest_step(console_text: str) -> int:
    matches = STEP_RE.findall(console_text)
    if not matches:
        return 0
    return int(matches[-1])


def parse_training_summary(console_text: str) -> dict[str, Any]:
    train = [(int(step), float(loss), float(obj), float(corr)) for step, loss, obj, corr in TRAIN_RE.findall(console_text)]
    val = [(int(step), float(loss), float(mae), float(corr)) for step, loss, mae, corr in VAL_RE.findall(console_text)]
    result: dict[str, Any] = {
        "train_log_count": len(train),
        "val_log_count": len(val),
    }
    if train:
        result["last_train"] = {
            "step": train[-1][0],
            "loss": train[-1][1],
            "objective_loss": train[-1][2],
            "corr": train[-1][3],
        }
        if len(train) >= 2:
            elapsed_steps = max(train[-1][0] - train[0][0], 1)
            result["logged_step_span"] = elapsed_steps
    if val:
        result["last_val"] = {
            "step": val[-1][0],
            "loss": val[-1][1],
            "mae": val[-1][2],
            "corr": val[-1][3],
        }
    return result


def resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_mortal_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    text = str(path_text).replace("\\", "/")
    if text.startswith("./checkpoints/") or text == "./checkpoints":
        return str((REPO_ROOT / "mortal" / text[2:]).resolve())
    return str((REPO_ROOT / path).resolve())


def make_case_config(base_cfg: dict[str, Any], case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    pretrain_cfg = cfg.setdefault("oracle_critic_pretrain", {})
    run_name = args.run_name or f"resource_probe_{args.name}"
    init_state_file = args.init_state_file or str(pretrain_cfg.get("init_state_file", ""))

    pretrain_cfg.update(
        {
            "run_name": run_name,
            "state_file": str(case_dir / "checkpoints" / "latest.pth"),
            "best_state_file": str(case_dir / "checkpoints" / "best.pth"),
            "tensorboard_dir": str(case_dir / "tb_log"),
            "metrics_file": str(case_dir / "metrics.jsonl"),
            "critic_arch": args.critic_arch,
            "train_scope": args.train_scope,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "val_num_workers": int(args.val_num_workers),
            "file_batch_size": int(args.file_batch_size),
            "val_file_batch_size": int(args.val_file_batch_size),
            "prefetch_factor": int(args.prefetch_factor),
            "val_prefetch_factor": int(args.val_prefetch_factor),
            "max_steps": int(args.max_steps),
            "log_every": int(args.log_every),
            "save_every": int(args.save_every),
            "val_every_steps": int(args.val_every_steps),
            "dependency_val_every_steps": int(args.dependency_val_every_steps),
            "val_batches": int(args.val_batches),
            "max_train_files": int(args.max_train_files),
            "max_val_files": int(args.max_val_files),
        }
    )
    if args.full_file_pool:
        pretrain_cfg.pop("max_train_files", None)
        pretrain_cfg.pop("max_val_files", None)
    if init_state_file:
        pretrain_cfg["init_state_file"] = resolve_mortal_path(init_state_file)
    if args.teacher_state_file:
        pretrain_cfg["teacher_state_file"] = resolve_mortal_path(args.teacher_state_file)
    if args.teacher_loss_weight is not None:
        pretrain_cfg["teacher_loss_weight"] = float(args.teacher_loss_weight)
    if args.target_loss_weight is not None:
        pretrain_cfg["target_loss_weight"] = float(args.target_loss_weight)

    scheduler_cfg = pretrain_cfg.setdefault("scheduler", {})
    scheduler_cfg["max_steps"] = max(int(args.max_steps), int(scheduler_cfg.get("warm_up_steps", 0) or 0))
    return cfg


def current_pid_set(root_pid: int) -> tuple[dict[str, Any], set[int]]:
    windows_metrics = query_windows_metrics(root_pid)
    raw_pid_set = windows_metrics.get("pid_set", [])
    if isinstance(raw_pid_set, (int, float, str)):
        raw_pid_items = [raw_pid_set]
    else:
        raw_pid_items = list(raw_pid_set)
    pids = {
        int(pid)
        for pid in raw_pid_items
        if isinstance(pid, (int, float, str)) and str(pid).strip()
    }
    return windows_metrics, pids


def query_tree_process_snapshot(root_pid: int) -> dict[str, Any]:
    script = rf"""
$ErrorActionPreference = 'SilentlyContinue'
$rootPid = {int(root_pid)}
function Get-DescendantPids([int]$startPid) {{
    $seen = @{{}}
    $queue = New-Object System.Collections.Queue
    $queue.Enqueue($startPid)
    while ($queue.Count -gt 0) {{
        $current = [int]$queue.Dequeue()
        if ($seen.ContainsKey($current)) {{
            continue
        }}
        $seen[$current] = $true
        Get-CimInstance Win32_Process -Filter ('ParentProcessId = ' + $current) |
            ForEach-Object {{ $queue.Enqueue([int]$_.ProcessId) }}
    }}
    return [int[]]$seen.Keys
}}
$pids = Get-DescendantPids $rootPid
$processes = @()
foreach ($procId in $pids) {{
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($null -ne $proc) {{
        $processes += $proc
    }}
}}
$rssBytes = 0.0
$privateBytes = 0.0
foreach ($proc in $processes) {{
    $rssBytes += [double]$proc.WorkingSet64
    $privateBytes += [double]$proc.PrivateMemorySize64
}}
[pscustomobject]@{{
    pid_set = $pids
    process_count = $processes.Count
    tree_rss_gb = ($rssBytes / 1GB)
    tree_private_gb = ($privateBytes / 1GB)
}} | ConvertTo-Json -Compress
"""
    try:
        proc = subprocess.run(
            [POWERSHELL_EXE, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        text = proc.stdout.strip()
        if text:
            return json.loads(text)
    except Exception:
        pass
    return {
        "pid_set": [root_pid],
        "process_count": 0,
        "tree_rss_gb": None,
        "tree_private_gb": None,
    }


def query_tree_cpu_percent(pids: set[int]) -> float | None:
    if not pids:
        return None
    pid_filter = ",".join(str(pid) for pid in sorted(pids))
    script = rf"""
$ErrorActionPreference = 'SilentlyContinue'
$target = @({pid_filter})
$proc = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
    Where-Object {{ $target -contains [int]$_.IDProcess }}
$sum = 0.0
foreach ($item in $proc) {{
    $sum += [double]$item.PercentProcessorTime
}}
[pscustomobject]@{{ tree_cpu_percent = $sum }} | ConvertTo-Json -Compress
"""
    try:
        proc = subprocess.run(
            [POWERSHELL_EXE, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        text = proc.stdout.strip()
        if text:
            payload = json.loads(text)
            return float(payload["tree_cpu_percent"])
    except Exception:
        pass
    return None


def merge_windows_and_tree_metrics(root_pid: int) -> tuple[dict[str, Any], set[int]]:
    windows_metrics, pids = current_pid_set(root_pid)
    tree_metrics = query_tree_process_snapshot(root_pid)
    raw_pid_set = tree_metrics.get("pid_set", [])
    if isinstance(raw_pid_set, (int, float, str)):
        raw_pid_items = [raw_pid_set]
    else:
        raw_pid_items = list(raw_pid_set)
    tree_pids = {
        int(pid)
        for pid in raw_pid_items
        if isinstance(pid, (int, float, str)) and str(pid).strip()
    }
    if tree_pids:
        pids = tree_pids
    tree_cpu_percent = query_tree_cpu_percent(pids)
    if tree_cpu_percent is not None:
        windows_metrics["tree_cpu_percent"] = tree_cpu_percent
    for key in ("process_count", "tree_rss_gb", "tree_private_gb"):
        value = tree_metrics.get(key)
        if value is not None:
            windows_metrics[key] = value
    return windows_metrics, pids


def prepare_case(args: argparse.Namespace) -> tuple[Path, Path]:
    output_root = resolve_config_path(args.output_root)
    case_dir = output_root / args.name
    case_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_toml_file(resolve_config_path(args.base_config))
    cfg = make_case_config(base_cfg, case_dir, args)
    config_path = case_dir / "config.toml"
    write_toml_file(config_path, cfg)
    return case_dir, config_path


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    case_dir, config_path = prepare_case(args)

    env = os.environ.copy()
    env["MORTAL_CFG"] = str(config_path)
    cmd = [args.python_exe, "-m", "mortal.online.pretrain_oracle_critic"]
    if args.fresh:
        cmd.append("--fresh")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    if proc.stdout is None:
        raise RuntimeError("failed to capture oracle critic pretrain stdout")

    chunk_queue: queue.Queue[bytes] = queue.Queue()
    reader = threading.Thread(target=read_stdout, args=(proc.stdout, chunk_queue), daemon=True)
    reader.start()

    raw_chunks: list[bytes] = []
    console_text = ""
    samples: list[dict[str, Any]] = []
    started_at = time.time()
    last_sample_at = 0.0
    target_seen_at: float | None = None
    live_console_path = case_dir / "console.live.txt"
    live_path = case_dir / "live.json"

    def write_live_snapshot() -> None:
        latest_sample = samples[-1] if samples else {}
        atomic_write_text(live_console_path, console_text.replace("\r", "\n"))
        atomic_write_json(
            live_path,
            {
                "name": args.name,
                "elapsed_sec": round(time.time() - started_at, 3),
                "latest_step": latest_step(console_text),
                "returncode": proc.poll(),
                "training": parse_training_summary(console_text),
                "last_sample": latest_sample,
            },
        )

    while True:
        drained = False
        while True:
            try:
                chunk = chunk_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            raw_chunks.append(chunk)
            console_text += chunk.decode("utf-8", errors="ignore")

        step = latest_step(console_text)
        now = time.time()
        if step >= args.target_step and target_seen_at is None:
            target_seen_at = now

        if now - last_sample_at >= args.sample_interval:
            windows_metrics, pids = merge_windows_and_tree_metrics(proc.pid)
            gpu_metrics = query_gpu_metrics(pids)
            samples.append(
                {
                    "elapsed_sec": round(now - started_at, 3),
                    "latest_step": step,
                    "process_count": windows_metrics.get("process_count"),
                    "system_cpu_percent": windows_metrics.get("system_cpu_percent"),
                    "tree_cpu_percent": windows_metrics.get("tree_cpu_percent"),
                    "tree_rss_gb": windows_metrics.get("tree_rss_gb"),
                    "tree_private_gb": windows_metrics.get("tree_private_gb"),
                    "system_mem_used_gb": windows_metrics.get("system_mem_used_gb"),
                    "system_mem_percent": windows_metrics.get("system_mem_percent"),
                    **gpu_metrics,
                }
            )
            last_sample_at = now
            write_live_snapshot()

        should_force_stop_at_target = int(args.target_step) < int(args.max_steps)
        if (
            should_force_stop_at_target
            and target_seen_at is not None
            and now - target_seen_at >= args.post_target_wait_sec
        ):
            stop_process_tree(proc.pid)
            target_seen_at = None

        if proc.poll() is not None:
            if not drained:
                time.sleep(0.2)
                while True:
                    try:
                        chunk = chunk_queue.get_nowait()
                    except queue.Empty:
                        break
                    raw_chunks.append(chunk)
                    console_text += chunk.decode("utf-8", errors="ignore")
            break

        time.sleep(0.2)

    raw = b"".join(raw_chunks)
    console_path = case_dir / "console.txt"
    (case_dir / "console.bin").write_bytes(raw)
    atomic_write_text(console_path, console_text.replace("\r", "\n"))
    atomic_write_text(
        case_dir / "samples.jsonl",
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
    )

    steady_samples = [sample for sample in samples if sample.get("latest_step", 0) >= args.steady_step]
    summary: dict[str, Any] = {
        "name": args.name,
        "critic_arch": args.critic_arch,
        "train_scope": args.train_scope,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "file_batch_size": args.file_batch_size,
        "prefetch_factor": args.prefetch_factor,
        "val_file_batch_size": args.val_file_batch_size,
        "val_prefetch_factor": args.val_prefetch_factor,
        "max_steps": args.max_steps,
        "target_step": args.target_step,
        "steady_step": args.steady_step,
        "elapsed_sec": time.time() - started_at,
        "returncode": proc.returncode,
        "config_path": str(config_path),
        "console_path": str(console_path),
        "sample_count": len(samples),
        "steady_sample_count": len(steady_samples),
        "training": parse_training_summary(console_text),
        "resource_summary_all": {},
        "resource_summary_steady": {},
    }
    completed_step = latest_step(console_text)
    if summary["elapsed_sec"] > 0:
        summary["steps_per_sec_all"] = completed_step / summary["elapsed_sec"]
    for key in (
        "gpu_util_percent",
        "gpu_mem_util_percent",
        "gpu_mem_used_mb",
        "tree_gpu_mem_mb",
        "system_cpu_percent",
        "tree_cpu_percent",
        "tree_rss_gb",
        "tree_private_gb",
        "system_mem_used_gb",
        "system_mem_percent",
        "gpu_temperature_c",
    ):
        all_summary = summarize_numeric(samples, key)
        if all_summary is not None:
            summary["resource_summary_all"][key] = all_summary
        steady_summary = summarize_numeric(steady_samples, key)
        if steady_summary is not None:
            summary["resource_summary_steady"][key] = steady_summary

    atomic_write_json(case_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        case_dir, config_path = prepare_case(args)
        print(json.dumps(
            {
                "name": args.name,
                "case_dir": str(case_dir),
                "config_path": str(config_path),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return
    summary = run_probe(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

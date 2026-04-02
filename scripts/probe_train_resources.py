from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import re
import random
import statistics
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

MORTAL_DIR = Path(__file__).resolve().parents[1] / "mortal"
if str(MORTAL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(MORTAL_DIR))

from toml_utils import load_toml_file, write_toml_file


TRAIN_RE = re.compile(r"TRAIN E1:\s*(\d+)batch .*? ([0-9.]+)batch/s")
BAD_FILE_RE = re.compile(r"error when reading ([A-Za-z]:\\[^\r\n]+?\.json(?:\.gz)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--glob-pattern", default="*.json")
    parser.add_argument("--output-root", default="logs/train_resource_probe")
    parser.add_argument("--python-exe", default=sys_executable())
    parser.add_argument("--train-script", default="mortal/train_supervised.py")
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--file-batch-size", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, required=True)
    parser.add_argument("--val-file-batch-size", type=int, default=8)
    parser.add_argument("--val-prefetch-factor", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--target-step", type=int, default=240)
    parser.add_argument("--steady-step", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--post-target-wait-sec", type=float, default=4.0)
    parser.add_argument("--shared-file-index", default="")
    parser.add_argument("--exclude-file", action="append", default=[])
    parser.add_argument("--rebuild-file-index", action="store_true")
    parser.add_argument("--auto-exclude-invalid-utf8", action="store_true")
    parser.add_argument("--auto-exclude-limit", type=int, default=16)
    return parser.parse_args()


def sys_executable() -> str:
    return os.environ.get("PYTHON", os.sys.executable)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def read_stdout(pipe, chunk_queue: queue.Queue[bytes]) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            chunk_queue.put(chunk)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def latest_train_step(console_text: str) -> int:
    matches = TRAIN_RE.findall(console_text)
    if not matches:
        return 0
    return int(matches[-1][0])


def parse_throughput(console_text: str) -> dict[str, Any]:
    matches = [(int(step), float(rate)) for step, rate in TRAIN_RE.findall(console_text)]
    result: dict[str, Any] = {"count": len(matches)}
    for floor in (120, 150, 180):
        values = [rate for step, rate in matches if step >= floor]
        if values:
            ordered = sorted(values)
            result[f"median_from_{floor}"] = statistics.median(values)
            result[f"p75_from_{floor}"] = ordered[(len(ordered) * 3) // 4]
            result[f"max_from_{floor}"] = max(values)
    return result


def detect_invalid_utf8_bad_file(console_text: str) -> str:
    if "stream did not contain valid UTF-8" not in console_text:
        return ""
    matches = BAD_FILE_RE.findall(console_text)
    if not matches:
        return ""
    return matches[-1]


def query_gpu_metrics(pid_set: set[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gpu_util_percent": None,
        "gpu_mem_util_percent": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_temperature_c": None,
        "tree_gpu_mem_mb": 0,
    }
    try:
        gpu_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        row = gpu_proc.stdout.strip().splitlines()
        if row:
            parts = [part.strip() for part in row[0].split(",")]
            if len(parts) >= 5:
                result["gpu_util_percent"] = float(parts[0])
                result["gpu_mem_util_percent"] = float(parts[1])
                result["gpu_mem_used_mb"] = float(parts[2])
                result["gpu_mem_total_mb"] = float(parts[3])
                result["gpu_temperature_c"] = float(parts[4])
    except Exception:
        pass

    try:
        app_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        total = 0.0
        for line in app_proc.stdout.strip().splitlines():
            if not line.strip():
                continue
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) < 2:
                continue
            try:
                app_pid = int(pieces[0])
                used_mb = float(pieces[1])
            except ValueError:
                continue
            if app_pid in pid_set:
                total += used_mb
        result["tree_gpu_mem_mb"] = total
    except Exception:
        pass

    return result


def query_windows_metrics(root_pid: int) -> dict[str, Any]:
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
$os = Get-CimInstance Win32_OperatingSystem
$totalKb = [double]$os.TotalVisibleMemorySize
$freeKb = [double]$os.FreePhysicalMemory
$usedKb = $totalKb - $freeKb
$memPercent = $null
if ($totalKb -gt 0) {{
    $memPercent = ($usedKb * 100.0 / $totalKb)
}}
$counters = Get-Counter '\Processor(_Total)\% Processor Time','\Process(python*)\% Processor Time','\Process(python*)\Working Set - Private','\Process(python*)\ID Process'
$systemCpu = $null
$instances = @{{}}
foreach ($sample in $counters.CounterSamples) {{
    $path = $sample.Path.ToLowerInvariant()
    if ($path -like '*\processor(_total)\% processor time') {{
        $systemCpu = [double]$sample.CookedValue
        continue
    }}
    if ($path -match '\\process\(([^)]+)\)\\(.+)$') {{
        $instance = $matches[1]
        if (-not $instances.ContainsKey($instance)) {{
            $instances[$instance] = @{{
                pid = $null
                cpu = 0.0
                ws = 0.0
            }}
        }}
        if ($path -like '*\id process') {{
            $instances[$instance]['pid'] = [int]$sample.CookedValue
        }} elseif ($path -like '*\% processor time') {{
            $instances[$instance]['cpu'] = [double]$sample.CookedValue
        }} elseif ($path -like '*\working set - private') {{
            $instances[$instance]['ws'] = [double]$sample.CookedValue
        }}
    }}
}}
$treeCpu = 0.0
$treeWsBytes = 0.0
$processCount = 0
foreach ($entry in $instances.GetEnumerator()) {{
    $pid = $entry.Value['pid']
    if ($null -eq $pid) {{
        continue
    }}
    if ($pids -contains [int]$pid) {{
        $treeCpu += [double]$entry.Value['cpu']
        $treeWsBytes += [double]$entry.Value['ws']
        $processCount += 1
    }}
}}
[pscustomobject]@{{
    pid_set = $pids
    process_count = $processCount
    system_cpu_percent = $systemCpu
    tree_cpu_percent = $treeCpu
    tree_rss_gb = ($treeWsBytes / 1GB)
    system_mem_used_gb = ($usedKb / 1MB)
    system_mem_percent = $memPercent
}} | ConvertTo-Json -Compress
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        text = proc.stdout.strip()
        if not text:
            return {
                "pid_set": [root_pid],
                "process_count": 0,
                "system_cpu_percent": None,
                "tree_cpu_percent": None,
                "tree_rss_gb": None,
                "system_mem_used_gb": None,
                "system_mem_percent": None,
            }
        return json.loads(text)
    except Exception:
        return {
            "pid_set": [root_pid],
            "process_count": 0,
            "system_cpu_percent": None,
            "tree_cpu_percent": None,
            "tree_rss_gb": None,
            "system_mem_used_gb": None,
            "system_mem_percent": None,
        }


def summarize_numeric(samples: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "p75": ordered[(len(ordered) * 3) // 4],
        "max": max(values),
    }


def make_case_config(
    *,
    base_cfg: dict[str, Any],
    case_dir: Path,
    dataset_root: Path,
    glob_pattern: str,
    shared_file_index: str,
    batch_size: int,
    max_steps: int,
    log_every: int,
    num_workers: int,
    file_batch_size: int,
    prefetch_factor: int,
    val_file_batch_size: int,
    val_prefetch_factor: int,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    supervised_cfg = cfg["supervised"]
    dataset_cfg = cfg["dataset"]

    supervised_cfg["state_file"] = str(case_dir / "stage0_5_latest.pth")
    supervised_cfg["best_state_file"] = str(case_dir / "stage0_5_supervised.pth")
    supervised_cfg["best_loss_state_file"] = str(case_dir / "stage0_5_supervised.pth")
    supervised_cfg["best_acc_state_file"] = str(case_dir / "stage0_5_supervised_best_acc.pth")
    supervised_cfg["best_rank_state_file"] = str(case_dir / "stage0_5_supervised_best_rank.pth")
    supervised_cfg["tensorboard_dir"] = str(case_dir / "tb")
    supervised_cfg["file_index"] = shared_file_index or str(case_dir / "file_index_supervised_json.pth")
    supervised_cfg["batch_size"] = batch_size
    supervised_cfg["save_every"] = max_steps
    supervised_cfg["log_every"] = log_every
    supervised_cfg["max_steps"] = max_steps
    supervised_cfg["val_every_steps"] = 0
    supervised_cfg["monitor_val_batches"] = 0
    supervised_cfg["full_val_every_checks"] = 0
    supervised_cfg["min_validation_checks"] = 0
    supervised_cfg["old_regression_every_checks"] = 0
    supervised_cfg["num_workers"] = num_workers
    supervised_cfg["file_batch_size"] = file_batch_size
    supervised_cfg["val_file_batch_size"] = val_file_batch_size
    supervised_cfg["prefetch_factor"] = prefetch_factor
    supervised_cfg["val_prefetch_factor"] = val_prefetch_factor

    dataset_cfg["globs"] = [str(dataset_root / "**" / glob_pattern)]
    dataset_cfg["num_workers"] = num_workers
    dataset_cfg["file_batch_size"] = file_batch_size
    dataset_cfg["prefetch_factor"] = prefetch_factor

    return cfg


def build_supervised_index(
    *,
    globs: list[str],
    index_path: Path,
    seed: int,
    val_ratio: float,
    min_val_files: int,
    max_train_files: int,
    max_val_files: int,
    exclude_files: list[str],
) -> dict[str, Any]:
    file_list: list[str] = []
    for pattern in globs:
        file_list.extend(glob.glob(pattern, recursive=True))
    exclude_set = {str(Path(item)) for item in exclude_files}
    file_list = [str(Path(item)) for item in file_list if str(Path(item)) not in exclude_set]
    file_list.sort()
    rng = random.Random(seed)
    rng.shuffle(file_list)
    if len(file_list) < 2:
        raise RuntimeError("not enough files to split train/val")
    val_count = max(int(len(file_list) * val_ratio), min_val_files)
    val_count = min(val_count, len(file_list) - 1)
    val_files = file_list[:val_count]
    train_files = file_list[val_count:]
    if max_train_files > 0:
        train_files = train_files[:max_train_files]
    if max_val_files > 0:
        val_files = val_files[:max_val_files]
    payload = {
        "train_files": train_files,
        "val_files": val_files,
        "monitor_recent_files": list(val_files),
        "full_recent_files": list(val_files),
        "old_regression_files": [],
        "seed": seed,
        "val_ratio": val_ratio,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, index_path)
    return payload


def stop_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    case_dir = output_root / args.name
    case_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    base_cfg = load_toml_file(args.base_config)
    cfg = make_case_config(
        base_cfg=base_cfg,
        case_dir=case_dir,
        dataset_root=dataset_root,
        glob_pattern=args.glob_pattern,
        shared_file_index=args.shared_file_index,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        num_workers=args.num_workers,
        file_batch_size=args.file_batch_size,
        prefetch_factor=args.prefetch_factor,
        val_file_batch_size=args.val_file_batch_size,
        val_prefetch_factor=args.val_prefetch_factor,
    )
    config_path = case_dir / "config.toml"
    write_toml_file(config_path, cfg)
    file_index_path = Path(cfg["supervised"]["file_index"])
    exclude_files = list(args.exclude_file)
    total_started_at = time.time()
    auto_excluded_files: list[str] = []
    auto_exclude_attempts = 0
    force_rebuild = bool(args.rebuild_file_index)

    while True:
        if force_rebuild or not file_index_path.exists():
            build_supervised_index(
                globs=list(cfg["dataset"]["globs"]),
                index_path=file_index_path,
                seed=int(cfg["supervised"].get("seed", 0)),
                val_ratio=float(cfg["supervised"].get("val_ratio", 0.02)),
                min_val_files=int(cfg["supervised"].get("min_val_files", 64)),
                max_train_files=int(cfg["supervised"].get("max_train_files", 0)),
                max_val_files=int(cfg["supervised"].get("max_val_files", 0)),
                exclude_files=exclude_files,
            )
        force_rebuild = False

        env = os.environ.copy()
        env["MORTAL_CFG"] = str(config_path)
        train_cmd = [args.python_exe, str((repo_root / args.train_script).resolve())]
        proc = subprocess.Popen(
            train_cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if proc.stdout is None:
            raise RuntimeError("failed to capture train stdout")

        chunk_queue: queue.Queue[bytes] = queue.Queue()
        reader = threading.Thread(target=read_stdout, args=(proc.stdout, chunk_queue), daemon=True)
        reader.start()

        raw_chunks: list[bytes] = []
        console_text = ""
        samples: list[dict[str, Any]] = []
        target_seen_at: float | None = None

        started_at = time.time()
        last_sample_at = 0.0
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

            current_step = latest_train_step(console_text)
            now = time.time()
            if current_step >= args.target_step and target_seen_at is None:
                target_seen_at = now

            if now - last_sample_at >= args.sample_interval:
                windows_metrics = query_windows_metrics(proc.pid)
                raw_pid_set = windows_metrics.get("pid_set", [])
                if isinstance(raw_pid_set, (int, float, str)):
                    raw_pid_items = [raw_pid_set]
                else:
                    raw_pid_items = list(raw_pid_set)
                current_pids = {
                    int(pid)
                    for pid in raw_pid_items
                    if isinstance(pid, (int, float, str)) and str(pid).strip()
                }
                gpu_metrics = query_gpu_metrics(current_pids)
                samples.append(
                    {
                        "elapsed_sec": round(now - started_at, 3),
                        "latest_step": current_step,
                        "process_count": windows_metrics.get("process_count"),
                        "system_cpu_percent": windows_metrics.get("system_cpu_percent"),
                        "tree_cpu_percent": windows_metrics.get("tree_cpu_percent"),
                        "tree_rss_gb": windows_metrics.get("tree_rss_gb"),
                        "system_mem_used_gb": windows_metrics.get("system_mem_used_gb"),
                        "system_mem_percent": windows_metrics.get("system_mem_percent"),
                        **gpu_metrics,
                    }
                )
                last_sample_at = now

            if target_seen_at is not None and now - target_seen_at >= args.post_target_wait_sec:
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
        bad_file = detect_invalid_utf8_bad_file(console_text)
        if (
            args.auto_exclude_invalid_utf8
            and proc.returncode != 0
            and bad_file
            and bad_file not in exclude_files
            and auto_exclude_attempts < args.auto_exclude_limit
        ):
            exclude_files.append(bad_file)
            auto_excluded_files.append(bad_file)
            auto_exclude_attempts += 1
            force_rebuild = True
            continue
        break

    elapsed_sec = time.time() - started_at
    raw = b"".join(raw_chunks)
    console_path = case_dir / "console.txt"
    (case_dir / "console.bin").write_bytes(raw)
    atomic_write_text(console_path, console_text.replace("\r", "\n"))
    atomic_write_text(
        case_dir / "samples.jsonl",
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
    )

    steady_samples = [sample for sample in samples if sample.get("latest_step", 0) >= args.steady_step]
    summary = {
        "name": args.name,
        "num_workers": args.num_workers,
        "file_batch_size": args.file_batch_size,
        "prefetch_factor": args.prefetch_factor,
        "val_file_batch_size": args.val_file_batch_size,
        "val_prefetch_factor": args.val_prefetch_factor,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "target_step": args.target_step,
        "steady_step": args.steady_step,
        "elapsed_sec": elapsed_sec,
        "total_elapsed_sec": time.time() - total_started_at,
        "returncode": proc.returncode,
        "config_path": str(config_path),
        "console_path": str(console_path),
        "sample_count": len(samples),
        "steady_sample_count": len(steady_samples),
        "excluded_files": exclude_files,
        "auto_excluded_files": auto_excluded_files,
        "auto_exclude_attempts": auto_exclude_attempts,
        "throughput": parse_throughput(console_text),
        "resource_summary_all": {},
        "resource_summary_steady": {},
    }
    metric_keys = [
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
    ]
    for key in metric_keys:
        all_summary = summarize_numeric(samples, key)
        if all_summary is not None:
            summary["resource_summary_all"][key] = all_summary
        steady_summary = summarize_numeric(steady_samples, key)
        if steady_summary is not None:
            summary["resource_summary_steady"][key] = steady_summary

    atomic_write_json(case_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

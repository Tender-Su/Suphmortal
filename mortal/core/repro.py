from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproRuntime:
    enabled: bool
    base_seed: int
    process_name: str
    process_seed: int | None
    train_key: int | None
    train_seed_start: int
    cudnn_benchmark: bool
    strict_cuda: bool


def repro_cfg(config):
    cfg = config.get("repro", {})
    return cfg if isinstance(cfg, dict) else {}


def repro_enabled(config) -> bool:
    return bool(repro_cfg(config).get("enabled", False))


def repro_base_seed(config) -> int:
    return max(int(repro_cfg(config).get("seed", 0) or 0), 0)


def repro_strict_cuda(config) -> bool:
    return bool(repro_cfg(config).get("strict_cuda", False))


def effective_cudnn_benchmark(config) -> bool:
    control_cfg = config.get("control", {})
    default = bool(control_cfg.get("enable_cudnn_benchmark", False)) if isinstance(control_cfg, dict) else False
    if not repro_enabled(config):
        return default
    return bool(repro_cfg(config).get("allow_cudnn_benchmark", False))


def _stable_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def derive_named_seed(base_seed: int, namespace: str) -> int:
    return max(_stable_u64(f"{int(base_seed)}:{namespace}") & 0x7FFF_FFFF_FFFF_FFFF, 1)


def resolve_process_seed(config, process_name: str) -> int | None:
    if not repro_enabled(config):
        return None
    return derive_named_seed(repro_base_seed(config), f"process:{process_name}")


def resolve_train_key(config) -> int | None:
    cfg = repro_cfg(config)
    raw_value = cfg.get("train_key")
    if raw_value is not None:
        return max(int(raw_value), 0)
    if not repro_enabled(config):
        return None
    return _stable_u64(f"{repro_base_seed(config)}:train_key")


def resolve_train_seed_start(config, default: int = 10000) -> int:
    cfg = repro_cfg(config)
    return max(int(cfg.get("train_seed_start", default) or default), 0)


def resolve_baseline_pool_seed(config, baseline_cfg=None) -> int | None:
    if baseline_cfg is None:
        baseline_root = config.get("baseline", {})
        baseline_cfg = baseline_root.get("train", {}) if isinstance(baseline_root, dict) else {}
    if isinstance(baseline_cfg, dict):
        explicit = baseline_cfg.get("pool_seed")
        if explicit is not None:
            return max(int(explicit), 0)
    if not repro_enabled(config):
        return None
    return derive_named_seed(repro_base_seed(config), "baseline_pool") % (2**32)


def apply_reproducibility(config, *, process_name: str) -> ReproRuntime:
    enabled = repro_enabled(config)
    base_seed = repro_base_seed(config)
    strict_cuda = repro_strict_cuda(config)
    process_seed = resolve_process_seed(config, process_name)
    train_key = resolve_train_key(config)
    train_seed_start = resolve_train_seed_start(config)
    cudnn_benchmark = effective_cudnn_benchmark(config)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    if strict_cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    if enabled and process_seed is not None:
        os.environ.setdefault("PYTHONHASHSEED", str(base_seed))
        random.seed(process_seed)
        np.random.seed(process_seed % (2**32))
        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(process_seed)

    return ReproRuntime(
        enabled=enabled,
        base_seed=base_seed,
        process_name=process_name,
        process_seed=process_seed,
        train_key=train_key,
        train_seed_start=train_seed_start,
        cudnn_benchmark=cudnn_benchmark,
        strict_cuda=strict_cuda,
    )

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


ROLE_TO_MODULE = {
    "server": "mortal.online.server",
    "trainer": "mortal.online.train_online",
    "client": "mortal.online.client",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one online role with explicit config / Oracle arm env setup.",
    )
    parser.add_argument("role", choices=tuple(ROLE_TO_MODULE))
    parser.add_argument("--config", default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--control-device", default=None)
    parser.add_argument("--baseline-train-device", default=None)
    parser.add_argument("--baseline-test-device", default=None)
    parser.add_argument("--train-play-profile", default=None)
    parser.add_argument("--windows-high-qos", action="store_true",
                        help="Disable execution-speed throttling for this role only.")
    return parser.parse_args()


def apply_runtime_env(*, config_path: str | None, arm: str | None, artifact_suffix: str | None) -> None:
    if config_path:
        os.environ["MORTAL_CFG"] = str(Path(config_path).resolve())
    if arm:
        os.environ["MORTAL_ORACLE_ARM"] = str(arm)
    if artifact_suffix is not None:
        os.environ["MORTAL_ORACLE_ARTIFACT_SUFFIX"] = str(artifact_suffix)


def apply_runtime_config_overrides(
    *,
    control_device: str | None,
    baseline_train_device: str | None,
    baseline_test_device: str | None,
    train_play_profile: str | None,
) -> None:
    if train_play_profile:
        os.environ["TRAIN_PLAY_PROFILE"] = str(train_play_profile)

    if not any((control_device, baseline_train_device, baseline_test_device)):
        return

    config_module = importlib.import_module("mortal.config")
    config_dict = config_module.config

    if control_device:
        config_dict.setdefault("control", {})["device"] = str(control_device)
    if baseline_train_device:
        baseline_cfg = config_dict.setdefault("baseline", {})
        baseline_cfg.setdefault("train", {})["device"] = str(baseline_train_device)
    if baseline_test_device:
        baseline_cfg = config_dict.setdefault("baseline", {})
        baseline_cfg.setdefault("test", {})["device"] = str(baseline_test_device)


def main() -> None:
    args = parse_args()
    if args.windows_high_qos:
        from mortal.core.process_resources import configure_windows_high_qos
        configure_windows_high_qos(True)
    apply_runtime_env(
        config_path=args.config,
        arm=args.arm,
        artifact_suffix=args.artifact_suffix,
    )
    apply_runtime_config_overrides(
        control_device=args.control_device,
        baseline_train_device=args.baseline_train_device,
        baseline_test_device=args.baseline_test_device,
        train_play_profile=args.train_play_profile,
    )
    module = importlib.import_module(ROLE_TO_MODULE[args.role])
    entry = getattr(module, "main", None)
    if entry is None:
        raise AttributeError(f"module {module.__name__!r} does not expose main()")
    entry()


if __name__ == "__main__":
    main()

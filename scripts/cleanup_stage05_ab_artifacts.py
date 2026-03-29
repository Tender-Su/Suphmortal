from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


DEFAULT_SCRATCH_ROOT = Path(tempfile.gettempdir()) / "mahjongai_stage05_ab"


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def collect_repo_cleanup_targets(stage05_ab_root: Path) -> list[Path]:
    targets: list[Path] = []
    for arm_result_path in stage05_ab_root.rglob("arm_result.json"):
        try:
            payload = json.loads(arm_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        run_payload = payload.get("run") or {}
        phase_order = run_payload.get("phase_order") or []
        if not phase_order:
            continue
        final_phase = str(phase_order[-1])
        arm_root = arm_result_path.parent
        for phase_name in phase_order:
            phase_name = str(phase_name)
            if phase_name == final_phase:
                continue
            phase_dir = arm_root / phase_name
            for candidate in (
                phase_dir / "checkpoints",
                phase_dir / "tb",
                phase_dir / "file_index.pth",
            ):
                if candidate.exists():
                    targets.append(candidate)
    return targets


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean historical stage05_ab intermediate artifacts without touching final-phase outputs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to this script's parent repo.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Default is dry-run.",
    )
    parser.add_argument(
        "--remove-scratch",
        action="store_true",
        help="Also remove the temp scratch root used by compact intermediate phase storage.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root to remove with --remove-scratch.",
    )
    args = parser.parse_args()

    stage05_ab_root = args.repo_root / "logs" / "stage05_ab"
    targets = collect_repo_cleanup_targets(stage05_ab_root) if stage05_ab_root.exists() else []
    if args.remove_scratch and args.scratch_root.exists():
        targets.append(args.scratch_root)

    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_targets.append(target)

    total_bytes = sum(path_size(target) for target in unique_targets)
    mode = "delete" if args.delete else "dry-run"
    print(f"[cleanup] mode={mode} targets={len(unique_targets)} reclaim={format_bytes(total_bytes)}")
    for target in unique_targets:
        print(f"[cleanup] {target}")

    if not args.delete:
        return

    for target in unique_targets:
        remove_path(target)

    print("[cleanup] done")


if __name__ == "__main__":
    main()

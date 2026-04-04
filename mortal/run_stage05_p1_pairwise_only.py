from __future__ import annotations

import argparse


DEPRECATION_MESSAGE = """
run_stage05_p1_pairwise_only.py is deprecated.

The 2026-03-28 P1 redesign replaced the old:
  calibration -> solo -> pairwise -> joint_refine
flow with:
  calibration -> protocol_decide -> winner_refine

`ablation` is now a manual backlog-only confirmation path, not part of the
default P1 mainline.

Use run_stage05_p1_only.py instead:
  python run_stage05_p1_only.py --run-name <name>
  python run_stage05_p1_only.py --run-name <name> --continue-to-winner-refine
  python run_stage05_p1_only.py --run-name <name> --continue-to-ablation
""".strip()


def build_deprecation_message(
    *,
    run_name: str | None,
    dry_run: bool,
    ignored_args: list[str],
) -> str:
    extra_lines: list[str] = []
    if run_name:
        extra_lines.append(f'Legacy --run-name was accepted and ignored here: {run_name}')
    if dry_run:
        extra_lines.append('Legacy --dry-run was accepted and ignored here.')
    if ignored_args:
        extra_lines.append(f'Additional legacy args were ignored: {" ".join(ignored_args)}')

    if not extra_lines:
        return DEPRECATION_MESSAGE
    return f'{DEPRECATION_MESSAGE}\n\n' + '\n'.join(extra_lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Deprecated legacy entry. See message below for the current P1 flow.',
        add_help=False,
    )
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('--run-name')
    parser.add_argument('--dry-run', action='store_true')
    args, unknown_args = parser.parse_known_args(argv)

    raise SystemExit(
        build_deprecation_message(
            run_name=args.run_name,
            dry_run=args.dry_run,
            ignored_args=unknown_args,
        )
    )


if __name__ == '__main__':
    main()

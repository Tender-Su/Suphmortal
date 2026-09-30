from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .browser import DEFAULT_MAJSOUL_URL, BrowserCaptureOptions, run_browser_capture


def parse_viewport(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError("viewport must look like 1280x720") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("viewport dimensions must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture Mahjong Soul web client websocket traffic from a controlled Chromium profile.",
    )
    parser.add_argument("--url", default=DEFAULT_MAJSOUL_URL, help="Mahjong Soul web URL to open.")
    parser.add_argument(
        "--out",
        default="logs/majsoul/capture.jsonl",
        help="JSONL capture path. Use '-' for stdout.",
    )
    parser.add_argument(
        "--profile-dir",
        default="logs/majsoul/browser-profile",
        help="Persistent browser profile directory; keep it stable to preserve login state.",
    )
    parser.add_argument(
        "--payload-mode",
        choices=("metadata", "full"),
        default="metadata",
        help="Store only payload hashes/lengths by default; use full only for authorized local debugging.",
    )
    parser.add_argument(
        "--include-url-query",
        action="store_true",
        help="Keep websocket URL query strings in logs. They are redacted by default.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless. Headed mode is the default because Mahjong Soul login is interactive.",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Optional local browser channel such as chrome or msedge. Omit to use Playwright Chromium.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to capture. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--viewport",
        type=parse_viewport,
        default=(1280, 720),
        help="Browser viewport, for example 1280x720.",
    )
    parser.add_argument(
        "--integration-mode",
        choices=("observe", "assist", "autoplay"),
        default="observe",
        help="Declared integration mode for logs. observe captures only; assist/autoplay need an event adapter.",
    )
    parser.add_argument(
        "--enable-ui-clicks",
        action="store_true",
        help="Permit visible UI clicks when an authorized autoplay adapter is attached. Default is dry-run.",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Optional normalized coordinate calibration JSON for authorized UI execution.",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> BrowserCaptureOptions:
    width, height = args.viewport
    return BrowserCaptureOptions(
        url=args.url,
        output=Path(args.out),
        profile_dir=Path(args.profile_dir),
        headless=args.headless,
        channel=args.channel,
        duration_seconds=args.duration,
        viewport_width=width,
        viewport_height=height,
        payload_mode=args.payload_mode,
        include_url_query=args.include_url_query,
        integration_mode=args.integration_mode,
        enable_ui_clicks=args.enable_ui_clicks,
        calibration_file=Path(args.calibration_file) if args.calibration_file else None,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    options = options_from_args(args)

    try:
        asyncio.run(run_browser_capture(options))
    except KeyboardInterrupt:
        logging.info("capture stopped")
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

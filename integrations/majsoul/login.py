from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from .browser import DEFAULT_MAJSOUL_URL, _load_async_playwright
from .capture import parse_viewport


async def run_login_profile(
    *,
    url: str,
    profile_dir: Path,
    channel: str | None,
    viewport_width: int,
    viewport_height: int,
) -> None:
    async_playwright = _load_async_playwright()
    logging.info("opening Mahjong Soul login profile at %s", profile_dir)
    logging.info("credentials are entered only in the browser; this helper does not read or log them")

    async with async_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {}
        if channel:
            launch_kwargs["channel"] = channel

        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            viewport={"width": viewport_width, "height": viewport_height},
            **launch_kwargs,
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=0)
            logging.info("opened %s", url)
            logging.info("log in manually, then close the browser window when finished")
            while context.pages:
                await asyncio.sleep(1)
        finally:
            await context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a persistent Mahjong Soul browser profile for manual test-account login.",
    )
    parser.add_argument("--url", default=DEFAULT_MAJSOUL_URL, help="Mahjong Soul web URL to open.")
    parser.add_argument(
        "--profile-dir",
        default="logs/majsoul/browser-profile",
        help="Persistent browser profile directory that will store the login session.",
    )
    parser.add_argument(
        "--channel",
        default="chrome",
        help="Local browser channel such as chrome or msedge. Use an installed browser for login.",
    )
    parser.add_argument(
        "--viewport",
        type=parse_viewport,
        default=(1280, 720),
        help="Browser viewport, for example 1280x720.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    width, height = args.viewport
    try:
        asyncio.run(
            run_login_profile(
                url=args.url,
                profile_dir=Path(args.profile_dir),
                channel=args.channel,
                viewport_width=width,
                viewport_height=height,
            )
        )
    except KeyboardInterrupt:
        logging.info("login helper stopped")
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .records import JsonlRecorder, PayloadMode


DEFAULT_MAJSOUL_URL = "https://game.maj-soul.com/1/"


@dataclass(frozen=True)
class BrowserCaptureOptions:
    url: str = DEFAULT_MAJSOUL_URL
    output: Path = Path("logs/majsoul/capture.jsonl")
    profile_dir: Path = Path("logs/majsoul/browser-profile")
    headless: bool = False
    channel: str | None = None
    duration_seconds: float = 0.0
    viewport_width: int = 1280
    viewport_height: int = 720
    payload_mode: PayloadMode = "metadata"
    include_url_query: bool = False
    integration_mode: str = "observe"
    enable_ui_clicks: bool = False
    calibration_file: Path | None = None


def _load_async_playwright():
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is required for Mahjong Soul browser capture. "
            "Install it with `pip install playwright` and then run "
            "`python -m playwright install chromium`."
        ) from exc
    return async_playwright


def _record_safely(recorder: JsonlRecorder, action: str, **kwargs: Any) -> None:
    try:
        if action == "open":
            recorder.write_socket_open(kwargs["url"])
        elif action == "close":
            recorder.write_socket_close(kwargs["url"])
        elif action == "frame":
            recorder.write_frame(
                direction=kwargs["direction"],
                url=kwargs["url"],
                payload=kwargs["payload"],
            )
        else:
            raise ValueError(f"unknown recorder action: {action}")
    except Exception:
        logging.exception("failed to record Mahjong Soul websocket %s", action)


def _attach_page(page, recorder: JsonlRecorder, attached_pages: set[int]) -> None:
    page_id = id(page)
    if page_id in attached_pages:
        return
    attached_pages.add(page_id)

    def on_websocket(ws) -> None:
        url = ws.url
        logging.info("capturing websocket: %s", url)
        _record_safely(recorder, "open", url=url)
        ws.on(
            "framesent",
            lambda payload: _record_safely(
                recorder,
                "frame",
                direction="sent",
                url=url,
                payload=payload,
            ),
        )
        ws.on(
            "framereceived",
            lambda payload: _record_safely(
                recorder,
                "frame",
                direction="received",
                url=url,
                payload=payload,
            ),
        )
        ws.on("close", lambda: _record_safely(recorder, "close", url=url))

    page.on("websocket", on_websocket)


async def run_browser_capture(options: BrowserCaptureOptions) -> None:
    async_playwright = _load_async_playwright()
    logging.info("writing Mahjong Soul websocket capture to %s", options.output)
    logging.info("using persistent browser profile at %s", options.profile_dir)

    with JsonlRecorder(
        options.output,
        payload_mode=options.payload_mode,
        include_url_query=options.include_url_query,
    ) as recorder:
        recorder.write_integration_event(
            "session_start",
            {
                "integration_mode": options.integration_mode,
                "enable_ui_clicks": options.enable_ui_clicks,
                "calibration_file": str(options.calibration_file) if options.calibration_file else None,
            },
        )
        async with async_playwright() as playwright:
            launch_kwargs: dict[str, Any] = {}
            if options.channel:
                launch_kwargs["channel"] = options.channel

            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(options.profile_dir),
                headless=options.headless,
                viewport={"width": options.viewport_width, "height": options.viewport_height},
                **launch_kwargs,
            )
            attached_pages: set[int] = set()

            try:
                context.on("page", lambda page: _attach_page(page, recorder, attached_pages))
                for page in context.pages:
                    _attach_page(page, recorder, attached_pages)

                page = context.pages[0] if context.pages else await context.new_page()
                _attach_page(page, recorder, attached_pages)
                await page.goto(options.url, wait_until="domcontentloaded", timeout=0)
                logging.info("opened %s", options.url)
                recorder.write_page_ready(url=page.url, title=await page.title())

                if options.duration_seconds > 0:
                    await page.wait_for_timeout(int(options.duration_seconds * 1000))
                else:
                    while True:
                        await asyncio.sleep(3600)
            finally:
                await context.close()

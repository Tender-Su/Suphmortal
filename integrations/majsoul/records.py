from __future__ import annotations

import base64
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


SCHEMA = "integrations.majsoul.websocket.v1"

Direction = Literal["sent", "received"]
PayloadKind = Literal["text", "binary"]
PayloadMode = Literal["metadata", "full"]


def now_ms() -> int:
    return int(time.time() * 1000)


def redact_url_query(url: str) -> str:
    parts = urlsplit(url)
    if not parts.query and not parts.fragment:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def normalize_payload(payload: str | bytes | bytearray | memoryview) -> tuple[bytes, PayloadKind, str | None]:
    if isinstance(payload, str):
        return payload.encode("utf-8"), "text", payload
    if isinstance(payload, bytes):
        return payload, "binary", None
    if isinstance(payload, bytearray):
        return bytes(payload), "binary", None
    if isinstance(payload, memoryview):
        return payload.tobytes(), "binary", None
    raise TypeError(f"unsupported websocket payload type: {type(payload).__name__}")


@dataclass(frozen=True)
class WebSocketFrameRecord:
    direction: Direction
    url: str
    payload: bytes
    payload_kind: PayloadKind
    payload_text: str | None = None
    ts_ms: int = field(default_factory=now_ms)

    @classmethod
    def from_payload(
        cls,
        *,
        direction: Direction,
        url: str,
        payload: str | bytes | bytearray | memoryview,
        ts_ms: int | None = None,
    ) -> "WebSocketFrameRecord":
        payload_bytes, payload_kind, payload_text = normalize_payload(payload)
        return cls(
            direction=direction,
            url=url,
            payload=payload_bytes,
            payload_kind=payload_kind,
            payload_text=payload_text,
            ts_ms=now_ms() if ts_ms is None else int(ts_ms),
        )

    def to_json_record(
        self,
        *,
        payload_mode: PayloadMode,
        include_url_query: bool = False,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "websocket_frame",
            "ts_ms": self.ts_ms,
            "direction": self.direction,
            "url": self.url if include_url_query else redact_url_query(self.url),
            "payload_kind": self.payload_kind,
            "payload_len": len(self.payload),
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        if payload_mode == "metadata":
            return record
        if payload_mode != "full":
            raise ValueError(f"unsupported payload mode: {payload_mode!r}")

        if self.payload_kind == "text" and self.payload_text is not None:
            record["payload_encoding"] = "utf-8"
            record["payload_text"] = self.payload_text
        else:
            record["payload_encoding"] = "base64"
            record["payload_base64"] = base64.b64encode(self.payload).decode("ascii")
        return record


class JsonlRecorder:
    def __init__(
        self,
        path: str | Path,
        *,
        payload_mode: PayloadMode = "metadata",
        include_url_query: bool = False,
    ) -> None:
        if payload_mode not in {"metadata", "full"}:
            raise ValueError(f"unsupported payload mode: {payload_mode!r}")
        self.path = Path(path) if str(path) != "-" else Path("-")
        self.payload_mode = payload_mode
        self.include_url_query = include_url_query
        self._lock = threading.Lock()
        self._handle = None

    def __enter__(self) -> "JsonlRecorder":
        if str(self.path) == "-":
            self._handle = sys.stdout
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is None:
            return
        if self._handle is not sys.stdout:
            self._handle.close()
        self._handle = None

    def write_record(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("recorder is not open")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def write_socket_open(self, url: str) -> None:
        self.write_record(
            {
                "schema": SCHEMA,
                "kind": "websocket_open",
                "ts_ms": now_ms(),
                "url": url if self.include_url_query else redact_url_query(url),
            }
        )

    def write_socket_close(self, url: str) -> None:
        self.write_record(
            {
                "schema": SCHEMA,
                "kind": "websocket_close",
                "ts_ms": now_ms(),
                "url": url if self.include_url_query else redact_url_query(url),
            }
        )

    def write_page_ready(self, *, url: str, title: str) -> None:
        self.write_record(
            {
                "schema": SCHEMA,
                "kind": "page_ready",
                "ts_ms": now_ms(),
                "url": url if self.include_url_query else redact_url_query(url),
                "title": title,
            }
        )

    def write_integration_event(self, event: str, payload: dict[str, Any]) -> None:
        self.write_record(
            {
                "schema": SCHEMA,
                "kind": "integration_event",
                "ts_ms": now_ms(),
                "event": event,
                "payload": payload,
            }
        )

    def write_frame(
        self,
        *,
        direction: Direction,
        url: str,
        payload: str | bytes | bytearray | memoryview,
    ) -> None:
        frame = WebSocketFrameRecord.from_payload(direction=direction, url=url, payload=payload)
        self.write_record(
            frame.to_json_record(
                payload_mode=self.payload_mode,
                include_url_query=self.include_url_query,
            )
        )

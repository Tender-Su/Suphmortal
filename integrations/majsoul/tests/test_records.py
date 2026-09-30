import json
import tempfile
import unittest
from pathlib import Path

from integrations.majsoul.browser import DEFAULT_MAJSOUL_URL
from integrations.majsoul.capture import parse_viewport
from integrations.majsoul.records import JsonlRecorder, WebSocketFrameRecord, redact_url_query


class MajsoulRecordTests(unittest.TestCase):
    def test_metadata_mode_redacts_payload_and_url_query(self):
        frame = WebSocketFrameRecord.from_payload(
            direction="received",
            url="wss://example.test/gateway?token=secret",
            payload=b"\x01\x02\x03",
            ts_ms=123,
        )
        record = frame.to_json_record(payload_mode="metadata")

        self.assertEqual("wss://example.test/gateway", record["url"])
        self.assertEqual("binary", record["payload_kind"])
        self.assertEqual(3, record["payload_len"])
        self.assertIn("payload_sha256", record)
        self.assertNotIn("payload_base64", record)
        self.assertNotIn("payload_text", record)

    def test_full_mode_keeps_text_payload(self):
        frame = WebSocketFrameRecord.from_payload(
            direction="sent",
            url="wss://example.test/gateway",
            payload="hello",
            ts_ms=123,
        )
        record = frame.to_json_record(payload_mode="full")

        self.assertEqual("text", record["payload_kind"])
        self.assertEqual("utf-8", record["payload_encoding"])
        self.assertEqual("hello", record["payload_text"])

    def test_jsonl_recorder_writes_lifecycle_and_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.jsonl"
            with JsonlRecorder(path, payload_mode="full") as recorder:
                recorder.write_page_ready(url="https://example.test/?token=secret", title="Ready")
                recorder.write_integration_event("session_start", {"mode": "observe"})
                recorder.write_socket_open("wss://example.test/ws?a=1")
                recorder.write_frame(
                    direction="received",
                    url="wss://example.test/ws?a=1",
                    payload=b"\x00\xff",
                )
                recorder.write_socket_close("wss://example.test/ws?a=1")

            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(
            ["page_ready", "websocket_open", "websocket_frame", "websocket_close"],
            [line["kind"] for line in lines if line["kind"] != "integration_event"],
        )
        self.assertEqual("https://example.test/", lines[0]["url"])
        self.assertEqual("Ready", lines[0]["title"])
        self.assertEqual("session_start", lines[1]["event"])
        self.assertEqual("wss://example.test/ws", lines[2]["url"])
        self.assertEqual("base64", lines[3]["payload_encoding"])
        self.assertEqual("AP8=", lines[3]["payload_base64"])

    def test_redact_url_query_also_drops_fragment(self):
        self.assertEqual(
            "wss://example.test/ws",
            redact_url_query("wss://example.test/ws?token=secret#frag"),
        )

    def test_parse_viewport(self):
        self.assertEqual((1600, 900), parse_viewport("1600x900"))
        with self.assertRaises(Exception):
            parse_viewport("1600")

    def test_default_url_uses_yostar_test_entrypoint(self):
        self.assertEqual("https://game.maj-soul.com/1/", DEFAULT_MAJSOUL_URL)


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from integrations.majsoul.actions import MajsoulAction
from integrations.majsoul.records import JsonlRecorder
from integrations.majsoul.ui import CoordinateTarget, UiCalibration, build_ui_plan


class MajsoulActionTests(unittest.TestCase):
    def test_maps_mjai_dahai_to_ui_plan(self):
        action = MajsoulAction.from_mjai_reaction(
            '{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}'
        )
        plan = build_ui_plan(action)

        self.assertEqual("dahai", action.kind)
        self.assertEqual("5m", action.tile)
        self.assertTrue(plan.executable)
        self.assertEqual("tile:5m", plan.steps[0].target_key)

    def test_unknown_reaction_is_not_executable(self):
        action = MajsoulAction.from_mjai_reaction({"type": "custom"})
        plan = build_ui_plan(action)

        self.assertEqual("unknown", action.kind)
        self.assertFalse(plan.executable)

    def test_coordinate_target_requires_normalized_coordinates(self):
        CoordinateTarget(0.5, 1.0)
        with self.assertRaises(ValueError):
            CoordinateTarget(1.2, 0.5)

    def test_dry_run_executor_records_plan_without_clicking(self):
        class DummyPage:
            viewport_size = {"width": 1000, "height": 800}

            def __init__(self):
                self.clicked = False

            @property
            def mouse(self):
                return self

            async def click(self, x, y):
                self.clicked = True

        from integrations.majsoul.ui import CoordinateUiExecutor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.jsonl"
            with JsonlRecorder(path) as recorder:
                page = DummyPage()
                executor = CoordinateUiExecutor(
                    page=page,
                    calibration=UiCalibration({"tile:5m": CoordinateTarget(0.5, 0.8)}),
                    recorder=recorder,
                    dry_run=True,
                )
                plan = build_ui_plan(MajsoulAction.from_mjai_reaction({"type": "dahai", "pai": "5m"}))
                result = asyncio.run(executor.execute(plan))

            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertFalse(result)
        self.assertFalse(page.clicked)
        self.assertEqual("integration_event", lines[0]["kind"])
        self.assertEqual("ui_plan", lines[0]["event"])


if __name__ == "__main__":
    unittest.main()

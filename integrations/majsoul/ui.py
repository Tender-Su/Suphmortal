from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import MajsoulAction
from .records import JsonlRecorder


@dataclass(frozen=True)
class UiStep:
    label: str
    target_key: str


@dataclass(frozen=True)
class UiPlan:
    action: MajsoulAction
    steps: tuple[UiStep, ...]
    executable: bool
    reason: str | None = None

    def to_json_record(self) -> dict[str, Any]:
        return {
            "action": self.action.kind,
            "tile": self.action.tile,
            "consumed": list(self.action.consumed),
            "steps": [{"label": step.label, "target_key": step.target_key} for step in self.steps],
            "executable": self.executable,
            "reason": self.reason,
        }


def build_ui_plan(action: MajsoulAction) -> UiPlan:
    if action.kind in {"none", "unknown"}:
        return UiPlan(action=action, steps=(), executable=False, reason=f"no UI plan for {action.kind}")
    if action.kind == "dahai":
        if not action.tile:
            return UiPlan(action=action, steps=(), executable=False, reason="dahai action has no tile")
        return UiPlan(
            action=action,
            steps=(UiStep(label=f"discard {action.tile}", target_key=f"tile:{action.tile}"),),
            executable=True,
        )
    if action.kind == "reach":
        return UiPlan(
            action=action,
            steps=(UiStep(label="reach", target_key="button:reach"),),
            executable=True,
        )
    if action.kind == "hora":
        return UiPlan(
            action=action,
            steps=(UiStep(label="win", target_key="button:hora"),),
            executable=True,
        )
    if action.kind == "ryukyoku":
        return UiPlan(
            action=action,
            steps=(UiStep(label="abortive draw", target_key="button:ryukyoku"),),
            executable=True,
        )
    if action.kind in {"chi", "pon", "daiminkan", "kakan", "ankan"}:
        call_key = "kan" if action.kind in {"daiminkan", "kakan", "ankan"} else action.kind
        steps = [UiStep(label=call_key, target_key=f"button:{call_key}")]
        if action.tile:
            steps.append(UiStep(label=f"select {action.tile}", target_key=f"tile:{action.tile}"))
        return UiPlan(action=action, steps=tuple(steps), executable=True)
    return UiPlan(action=action, steps=(), executable=False, reason=f"unhandled action {action.kind}")


@dataclass(frozen=True)
class CoordinateTarget:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("coordinate targets must be normalized to [0, 1]")


class UiCalibration:
    def __init__(self, targets: dict[str, CoordinateTarget]) -> None:
        self.targets = dict(targets)

    @classmethod
    def empty(cls) -> "UiCalibration":
        return cls({})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "UiCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        targets = {
            str(key): CoordinateTarget(x=float(value["x"]), y=float(value["y"]))
            for key, value in payload.get("targets", {}).items()
        }
        return cls(targets)

    def get(self, key: str) -> CoordinateTarget | None:
        return self.targets.get(key)


class CoordinateUiExecutor:
    def __init__(
        self,
        *,
        page,
        calibration: UiCalibration,
        recorder: JsonlRecorder,
        dry_run: bool = True,
    ) -> None:
        self.page = page
        self.calibration = calibration
        self.recorder = recorder
        self.dry_run = dry_run

    async def execute(self, plan: UiPlan) -> bool:
        self.recorder.write_integration_event(
            "ui_plan",
            {
                **plan.to_json_record(),
                "dry_run": self.dry_run,
            },
        )
        if self.dry_run or not plan.executable:
            return False

        viewport = self.page.viewport_size
        if viewport is None:
            viewport = await self.page.evaluate(
                "() => ({ width: window.innerWidth, height: window.innerHeight })"
            )

        for step in plan.steps:
            target = self.calibration.get(step.target_key)
            if target is None:
                self.recorder.write_integration_event(
                    "ui_step_skipped",
                    {"target_key": step.target_key, "reason": "missing calibration"},
                )
                return False
            await self.page.mouse.click(target.x * viewport["width"], target.y * viewport["height"])
            self.recorder.write_integration_event(
                "ui_step_clicked",
                {"target_key": step.target_key, "x": target.x, "y": target.y},
            )
        return True

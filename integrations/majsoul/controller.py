from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .actions import MajsoulAction
from .records import JsonlRecorder
from .ui import CoordinateUiExecutor, UiPlan, build_ui_plan


class ReactionSource(Protocol):
    def react(self, mjai_event: dict) -> str | dict | None:
        ...


@dataclass
class IntegrationController:
    recorder: JsonlRecorder
    reaction_source: ReactionSource | None = None
    ui_executor: CoordinateUiExecutor | None = None

    async def handle_mjai_event(self, event: dict) -> UiPlan | None:
        self.recorder.write_integration_event("mjai_event", event)
        if self.reaction_source is None:
            return None

        reaction = self.reaction_source.react(event)
        action = MajsoulAction.from_mjai_reaction(reaction)
        self.recorder.write_integration_event(
            "bot_reaction",
            {"reaction": action.raw or {"type": action.kind}},
        )
        plan = build_ui_plan(action)
        if self.ui_executor is not None:
            await self.ui_executor.execute(plan)
        return plan

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


ActionKind = Literal[
    "dahai",
    "reach",
    "chi",
    "pon",
    "daiminkan",
    "kakan",
    "ankan",
    "hora",
    "ryukyoku",
    "none",
    "unknown",
]


@dataclass(frozen=True)
class MajsoulAction:
    kind: ActionKind
    tile: str | None = None
    consumed: tuple[str, ...] = ()
    actor: int | None = None
    target: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mjai_reaction(cls, reaction: str | dict[str, Any] | None) -> "MajsoulAction":
        if reaction is None:
            return cls(kind="none")
        if isinstance(reaction, str):
            text = reaction.strip()
            if not text:
                return cls(kind="none")
            payload = json.loads(text)
        else:
            payload = dict(reaction)

        kind = str(payload.get("type", "unknown")).strip().lower()
        if kind == "none":
            return cls(kind="none", raw=payload)
        if kind not in {
            "dahai",
            "reach",
            "chi",
            "pon",
            "daiminkan",
            "kakan",
            "ankan",
            "hora",
            "ryukyoku",
        }:
            return cls(kind="unknown", raw=payload)

        consumed = payload.get("consumed") or ()
        return cls(
            kind=kind,  # type: ignore[arg-type]
            tile=payload.get("pai"),
            consumed=tuple(str(tile) for tile in consumed),
            actor=_optional_int(payload.get("actor")),
            target=_optional_int(payload.get("target")),
            raw=payload,
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)

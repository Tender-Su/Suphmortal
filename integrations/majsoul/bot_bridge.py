from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BotCommand:
    argv: tuple[str, ...]

    @classmethod
    def from_shell_text(cls, command: str) -> "BotCommand":
        return cls(tuple(shlex.split(command)))


class MjaiBotProcess:
    def __init__(self, command: BotCommand | Sequence[str]) -> None:
        argv = command.argv if isinstance(command, BotCommand) else tuple(command)
        if not argv:
            raise ValueError("bot command must not be empty")
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def react(self, mjai_event: dict) -> dict | None:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("bot process pipes are closed")
        self.process.stdin.write(json.dumps(mjai_event, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            return None
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()

    def __enter__(self) -> "MjaiBotProcess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

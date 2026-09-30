from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any


def get_dict_section(node: Any, key: str) -> dict[str, Any]:
    """Return an existing dict section without mutating the input."""
    if not isinstance(node, Mapping):
        return {}
    section = node.get(key)
    return section if isinstance(section, dict) else {}


def ensure_dict_section(
    node: MutableMapping[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return a mutable dict section, replacing an invalid value if needed."""
    section = node.get(key)
    if isinstance(section, dict):
        return section
    section = {}
    node[key] = section
    return section


def deep_merge_dict(
    destination: MutableMapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            deep_merge_dict(ensure_dict_section(destination, key), value)
        else:
            destination[key] = deepcopy(value)


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    parsed = str(value).strip().lower()
    if parsed in {'1', 'true', 'yes', 'on'}:
        return True
    if parsed in {'0', 'false', 'no', 'off'}:
        return False
    return default

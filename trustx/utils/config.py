"""YAML configuration loading with shallow overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config, optionally merging dotted-key overrides.

    >>> load_config("configs/td3.yaml", {"train.episodes": 10})  # doctest: +SKIP
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)
    return cfg


def set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings from the CLI, YAML-decoding each value."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        out[key] = yaml.safe_load(raw)
    return out


def merged(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

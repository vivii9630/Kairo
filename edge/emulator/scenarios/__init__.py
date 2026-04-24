"""Scenario registry for the edge emulator.

A scenario is a ``(ctx: EmulatorContext) -> dict`` callable registered
here so the CLI can run ``--scenario smoke`` without a direct import.
Scenarios live in this subpackage and auto-register via
``register_scenario``.
"""

from __future__ import annotations

from typing import Callable, Dict

from ..runner import EmulatorContext


ScenarioFn = Callable[[EmulatorContext], dict]


_REGISTRY: Dict[str, ScenarioFn] = {}


def register_scenario(name: str, fn: ScenarioFn) -> None:
    if name in _REGISTRY:
        raise ValueError(f"scenario {name!r} already registered")
    _REGISTRY[name] = fn


def get_scenario(name: str) -> ScenarioFn:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown scenario {name!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_scenarios() -> list:
    return sorted(_REGISTRY)


# Import scenario modules so their @register side-effects run.
from . import smoke  # noqa: E402, F401

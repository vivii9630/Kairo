"""Registry for analytics tools.

Each tool is a pure callable `(df, plan) -> AnalyticsResult` registered under a
stable name. The `analyst` agent picks names from `tool_names()`; the runner
dispatches through `lookup_tool`. Adding a new chart or model = one file in the
appropriate submodule + one `register(...)` call in its `__init__.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult


class ToolHandler(Protocol):
    def __call__(self, df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult: ...


@dataclass(frozen=True)
class AnalyticsTool:
    name: str
    description: str
    handler: ToolHandler
    required_columns: int = 1


_REGISTRY: Dict[str, AnalyticsTool] = {}


def register(tool: AnalyticsTool) -> None:
    _REGISTRY[tool.name] = tool


def lookup_tool(name: str) -> AnalyticsTool | None:
    return _REGISTRY.get(name)


def tool_names() -> List[str]:
    return sorted(_REGISTRY.keys())


def list_tools() -> List[AnalyticsTool]:
    return [_REGISTRY[name] for name in tool_names()]

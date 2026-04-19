"""Kairo analytics: charts, descriptive stats, forecasting, and ML helpers."""

from .planner import build_plans, has_chart_intent
from .registry import (
    AnalyticsTool,
    ToolHandler,
    list_tools,
    lookup_tool,
    register,
    tool_names,
)
from .runner import AnalyticsRunner
from .schema import ColumnInfo, ColumnKind, infer_columns

__all__ = [
    "AnalyticsRunner",
    "AnalyticsTool",
    "ColumnInfo",
    "ColumnKind",
    "ToolHandler",
    "build_plans",
    "has_chart_intent",
    "infer_columns",
    "list_tools",
    "lookup_tool",
    "register",
    "tool_names",
]

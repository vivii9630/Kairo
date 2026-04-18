"""Single public entrypoint for executing an `AnalyticsPlan`."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult

from .registry import lookup_tool, tool_names

# Importing submodules registers their tools as a side effect.
from . import charting  # noqa: F401
from . import stats  # noqa: F401


class AnalyticsRunner:
    """Executes AnalyticsPlans against a DataFrame. Thin and stateless."""

    def available_tools(self) -> list[str]:
        return tool_names()

    def run(self, df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
        tool = lookup_tool(plan.tool)
        if tool is None:
            return AnalyticsResult(
                plan=plan,
                error=f"unknown analytics tool '{plan.tool}'. available: {tool_names()}",
            )

        missing = [c for c in plan.columns if c not in df.columns]
        if missing:
            return AnalyticsResult(
                plan=plan,
                error=f"columns not found in data: {missing}. available: {list(df.columns)}",
            )

        if len(plan.columns) < tool.required_columns:
            return AnalyticsResult(
                plan=plan,
                error=(
                    f"tool '{tool.name}' requires at least "
                    f"{tool.required_columns} column(s); got {len(plan.columns)}"
                ),
            )

        return tool.handler(df, plan)

    def run_many(self, df: pd.DataFrame, plans: Iterable[AnalyticsPlan]) -> list[AnalyticsResult]:
        return [self.run(df, plan) for plan in plans]

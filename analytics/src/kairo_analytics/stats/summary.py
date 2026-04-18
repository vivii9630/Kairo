"""Descriptive stats over one or more numeric columns."""

from __future__ import annotations

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, StatSummary

from ..registry import AnalyticsTool, register


def _describe_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    summaries: list[StatSummary] = []
    for col in plan.columns:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            summaries.append(StatSummary(column=col, count=0))
            continue
        summaries.append(
            StatSummary(
                column=col,
                count=int(numeric.count()),
                mean=float(numeric.mean()),
                std=float(numeric.std()) if len(numeric) > 1 else None,
                min=float(numeric.min()),
                max=float(numeric.max()),
                median=float(numeric.median()),
            )
        )
    return AnalyticsResult(plan=plan, stats=summaries)


register(
    AnalyticsTool(
        name="describe",
        description="Descriptive stats (count/mean/std/min/median/max) for each column listed.",
        handler=_describe_handler,
        required_columns=1,
    )
)

"""Correlation matrix over 2+ numeric columns. Emits ChartSpec(kind='pairwise')."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


_VALID_METHODS = ("pearson", "spearman", "kendall")


def _correlation_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    params: Dict[str, Any] = plan.params or {}
    method = str(params.get("method", "pearson")).lower()
    if method not in _VALID_METHODS:
        method = "pearson"

    numeric_df = df[plan.columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method=method)
    cols = list(corr.columns)

    series = [
        ChartSeries(
            name=row,
            data=[
                float(corr.at[row, col]) if pd.notna(corr.at[row, col]) else None
                for col in cols
            ],
        )
        for row in cols
    ]

    title = params.get("title") or f"{method.title()} correlation ({len(cols)} cols)"
    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="pairwise",
            title=title,
            x_values=cols,
            series=series,
            metadata={"method": method, "n_rows": int(len(numeric_df.dropna()))},
        ),
    )


register(
    AnalyticsTool(
        name="correlation",
        description=(
            "Correlation matrix over 2+ numeric columns. "
            "params.method = pearson | spearman | kendall (default pearson)."
        ),
        handler=_correlation_handler,
        required_columns=2,
    )
)

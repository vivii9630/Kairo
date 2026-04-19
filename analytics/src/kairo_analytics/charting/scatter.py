"""Scatter plot: numeric y vs numeric x. Emits ChartSpec(kind='scatter')."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


def _scatter_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    params: Dict[str, Any] = plan.params or {}
    x_col = plan.columns[0]
    y_col = plan.columns[1]

    x_num = pd.to_numeric(df[x_col], errors="coerce")
    y_num = pd.to_numeric(df[y_col], errors="coerce")
    mask = x_num.notna() & y_num.notna()
    x_vals = x_num[mask]
    y_vals = y_num[mask]

    pearson: float | None = None
    if len(x_vals) > 1:
        try:
            pearson = float(x_vals.corr(y_vals))
        except Exception:
            pearson = None

    xv = [float(v) for v in x_vals.tolist()]
    series = [ChartSeries(name=y_col, data=[float(v) for v in y_vals.tolist()])]

    title = params.get("title") or f"{y_col} vs {x_col}"
    subtitle = None
    if pearson is not None:
        subtitle = f"Pearson r = {pearson:.2f} over {int(mask.sum())} points"

    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="scatter",
            title=title,
            subtitle=subtitle,
            x_label=x_col,
            y_label=y_col,
            x_values=xv,
            series=series,
            metadata={"n": int(mask.sum()), "pearson_r": pearson},
        ),
    )


register(
    AnalyticsTool(
        name="scatter",
        description="Scatter plot of numeric y vs numeric x. Reports Pearson r in subtitle.",
        handler=_scatter_handler,
        required_columns=2,
    )
)

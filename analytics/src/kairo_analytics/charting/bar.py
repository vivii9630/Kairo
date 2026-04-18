"""Bar chart: one categorical x axis, one or more numeric series, optional aggregation."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


_AGG_MAP: Dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "median": "median",
}


def _bar_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    params: Dict[str, Any] = plan.params or {}
    x_col = plan.columns[0]
    y_cols = plan.columns[1:]
    agg_name = str(params.get("agg", "sum"))
    agg_fn = _AGG_MAP.get(agg_name, "sum")
    top_n = params.get("top_n")

    if y_cols:
        grouped = df.groupby(x_col)[y_cols].agg(agg_fn).reset_index()
    else:
        grouped = df.groupby(x_col).size().reset_index(name="count")
        y_cols = ["count"]

    if isinstance(top_n, int) and top_n > 0 and len(y_cols) == 1:
        grouped = grouped.sort_values(by=y_cols[0], ascending=False).head(top_n)

    x_values = grouped[x_col].astype(str).tolist()
    series = [
        ChartSeries(name=col, data=[_to_number(v) for v in grouped[col].tolist()])
        for col in y_cols
    ]

    title = params.get("title") or f"{agg_name.title()} of {', '.join(y_cols)} by {x_col}"
    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="bar",
            title=title,
            x_label=x_col,
            y_label=", ".join(y_cols),
            x_values=x_values,
            series=series,
            metadata={"agg": agg_name},
        ),
    )


def _to_number(v: Any) -> float | int | None:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


register(
    AnalyticsTool(
        name="bar_chart",
        description=(
            "Aggregate rows by the first column, compute agg (default sum) over "
            "remaining columns. With only one column, counts rows per category."
        ),
        handler=_bar_handler,
        required_columns=1,
    )
)

"""Line chart: ordered x axis (numeric or datetime), one or more numeric series."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


def _line_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    params: Dict[str, Any] = plan.params or {}
    x_col = plan.columns[0]
    y_cols = plan.columns[1:]

    if not y_cols:
        return AnalyticsResult(
            plan=plan,
            error="line_chart requires at least one y column in addition to x",
        )

    work = df[[x_col, *y_cols]].copy()
    parsed_dt = _try_parse_datetime(work[x_col])
    if parsed_dt is not None:
        work[x_col] = parsed_dt
    work = work.sort_values(by=x_col)

    x_values = [_stringify_x(v) for v in work[x_col].tolist()]
    series = [
        ChartSeries(name=col, data=[_to_number(v) for v in work[col].tolist()])
        for col in y_cols
    ]

    title = params.get("title") or f"{', '.join(y_cols)} over {x_col}"
    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="line",
            title=title,
            x_label=x_col,
            y_label=", ".join(y_cols),
            x_values=x_values,
            series=series,
            metadata={"x_is_datetime": parsed_dt is not None},
        ),
    )


def _try_parse_datetime(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (ValueError, TypeError):
        return None
    return parsed


def _stringify_x(v: Any) -> str | float | int:
    if pd.isna(v):
        return ""
    if isinstance(v, (int, float)):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


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
        name="line_chart",
        description="Line chart over an ordered x axis with one or more y series. Auto-parses datetime x.",
        handler=_line_handler,
        required_columns=2,
    )
)

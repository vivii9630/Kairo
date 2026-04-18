"""Histogram: distribution of a single numeric column into bins."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


def _histogram_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    params: Dict[str, Any] = plan.params or {}
    col = plan.columns[0]
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return AnalyticsResult(
            plan=plan,
            error=f"column '{col}' has no numeric values to histogram",
        )

    bins = int(params.get("bins", 20))
    counts, edges = np.histogram(values.to_numpy(), bins=bins)
    labels = [f"{edges[i]:.3g}" for i in range(len(edges) - 1)]

    title = params.get("title") or f"Distribution of {col}"
    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="histogram",
            title=title,
            x_label=col,
            y_label="count",
            x_values=labels,
            series=[ChartSeries(name="count", data=[int(c) for c in counts])],
            metadata={"bins": bins, "n": int(len(values))},
        ),
    )


register(
    AnalyticsTool(
        name="histogram",
        description="Distribution of one numeric column as a histogram. Default 20 bins.",
        handler=_histogram_handler,
        required_columns=1,
    )
)

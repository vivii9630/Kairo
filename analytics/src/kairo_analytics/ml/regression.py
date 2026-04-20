"""Simple linear regression over a numeric x -> numeric y pair.

Emits a ChartSpec(kind='scatter') with the observed points as the only
series, plus enough metadata (slope, intercept, r_squared, x_line, y_line)
for the UI to overlay the fitted line.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


def _regression_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return AnalyticsResult(
            plan=plan,
            error="scikit-learn is not installed. pip install 'kairo-analytics[ml]'",
        )

    params: Dict[str, Any] = plan.params or {}
    x_col = plan.columns[0]
    y_col = plan.columns[1]

    x_num = pd.to_numeric(df[x_col], errors="coerce")
    y_num = pd.to_numeric(df[y_col], errors="coerce")
    mask = x_num.notna() & y_num.notna()
    x_vals = x_num[mask].to_numpy()
    y_vals = y_num[mask].to_numpy()

    if len(x_vals) < 3:
        return AnalyticsResult(
            plan=plan,
            error=f"need at least 3 valid rows to fit a regression; got {len(x_vals)}",
        )

    X = x_vals.reshape(-1, 1)
    try:
        model = LinearRegression().fit(X, y_vals)
    except Exception as exc:  # noqa: BLE001
        return AnalyticsResult(plan=plan, error=f"regression failed: {exc}")

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = float(model.score(X, y_vals))

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min_pred = slope * x_min + intercept
    y_max_pred = slope * x_max + intercept

    title = params.get("title") or f"Linear fit: {y_col} ~ {x_col}"
    subtitle = (
        f"y = {slope:.3f}·x + {intercept:.2f} | R^2 = {r_squared:.3f} "
        f"| n = {int(mask.sum())}"
    )
    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="scatter",
            title=title,
            subtitle=subtitle,
            x_label=x_col,
            y_label=y_col,
            x_values=[float(v) for v in x_vals.tolist()],
            series=[
                ChartSeries(
                    name=y_col,
                    data=[float(v) for v in y_vals.tolist()],
                )
            ],
            metadata={
                "model": "LinearRegression",
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "x_line": [x_min, x_max],
                "y_line": [y_min_pred, y_max_pred],
                "n": int(mask.sum()),
            },
        ),
    )


register(
    AnalyticsTool(
        name="linear_regression",
        description=(
            "Simple linear regression over x -> y numeric columns. Returns a "
            "scatter ChartSpec with slope, intercept, r_squared, and fitted "
            "line endpoints in metadata so the UI can overlay the fit."
        ),
        handler=_regression_handler,
        required_columns=2,
    )
)

"""Exponential-smoothing forecast over a datetime + numeric column pair.

Statsmodels is a soft dep — this tool returns an explicit error AnalyticsResult
when `statsmodels` isn't importable, so the caller can surface a friendly
message instead of a 500.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ForecastSpec

from ..registry import AnalyticsTool, register


def _forecast_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return AnalyticsResult(
            plan=plan,
            error="statsmodels is not installed. pip install 'kairo-analytics[forecast]'",
        )

    params: Dict[str, Any] = plan.params or {}
    dt_col = plan.columns[0]
    y_col = plan.columns[1]
    periods = int(params.get("periods", 14))
    if periods <= 0:
        periods = 14
    alpha = float(params.get("alpha", 0.1))  # confidence band width (1 - CL)

    dt = pd.to_datetime(df[dt_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = dt.notna() & y.notna()
    series = (
        pd.Series(y[mask].values, index=dt[mask])
        .sort_index()
        .groupby(level=0)
        .mean()
    )

    if len(series) < 4:
        return AnalyticsResult(
            plan=plan,
            error=f"need at least 4 historical points to forecast; got {len(series)}",
        )

    freq = _infer_freq(series.index)

    try:
        model = ExponentialSmoothing(
            series.values,
            trend="add" if len(series) >= 6 else None,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
    except Exception as exc:  # noqa: BLE001 — statsmodels raises many shapes
        return AnalyticsResult(plan=plan, error=f"forecast failed: {exc}")

    forecast_values = np.asarray(model.forecast(periods), dtype=float)
    resid = np.asarray(model.resid, dtype=float)
    sigma = float(np.nanstd(resid)) if np.any(~np.isnan(resid)) else 0.0
    # Gaussian one-sigma widening per step (random walk approximation)
    z = _z_score(alpha)
    step_sigma = sigma * np.sqrt(np.arange(1, periods + 1))
    lower = forecast_values - z * step_sigma
    upper = forecast_values + z * step_sigma

    last = series.index[-1]
    forecast_index = _future_index(last, periods, freq)

    history_x = [_fmt_ts(ts) for ts in series.index]
    forecast_x = [_fmt_ts(ts) for ts in forecast_index]

    title = params.get("title") or f"{y_col} forecast ({periods} {freq or 'steps'})"
    spec = ForecastSpec(
        title=title,
        x_label=dt_col,
        y_label=y_col,
        history_x=history_x,
        history_y=[float(v) for v in series.values],
        forecast_x=forecast_x,
        forecast_y=[float(v) for v in forecast_values],
        forecast_lower=[float(v) for v in lower],
        forecast_upper=[float(v) for v in upper],
        metadata={
            "model": "ExponentialSmoothing",
            "periods": periods,
            "freq": freq,
            "sigma": sigma,
            "confidence": 1 - alpha,
        },
    )
    return AnalyticsResult(plan=plan, forecast=spec)


def _infer_freq(idx: pd.DatetimeIndex) -> str | None:
    try:
        inferred = pd.infer_freq(idx)
        if inferred:
            return inferred
    except Exception:
        pass
    if len(idx) < 2:
        return None
    delta = idx[1:] - idx[:-1]
    median_delta = pd.Series(delta).median()
    days = median_delta.total_seconds() / 86400 if median_delta else 0
    if days >= 28:
        return "M"
    if days >= 6:
        return "W"
    if days >= 0.9:
        return "D"
    return "H"


def _future_index(last: pd.Timestamp, periods: int, freq: str | None) -> pd.DatetimeIndex:
    step_freq = freq or "D"
    return pd.date_range(start=last, periods=periods + 1, freq=step_freq)[1:]


def _fmt_ts(ts: pd.Timestamp) -> str:
    return ts.isoformat()


def _z_score(alpha: float) -> float:
    table = {0.32: 1.0, 0.1: 1.645, 0.05: 1.96, 0.01: 2.576}
    closest = min(table.keys(), key=lambda k: abs(k - alpha))
    return table[closest]


register(
    AnalyticsTool(
        name="forecast",
        description=(
            "Forecast a numeric column forward over a datetime index using "
            "exponential smoothing. columns=[datetime_col, numeric_col]; "
            "params.periods (default 14), params.alpha (default 0.1 → 90% band)."
        ),
        handler=_forecast_handler,
        required_columns=2,
    )
)

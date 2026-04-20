"""Query + schema → List[AnalyticsPlan].

Heuristic-first: prefer precision over recall. An empty plan list means the
query wasn't really about analytics and we should stay out of the way.

The planner is pure Python (no pandas, no LLM). Swap in an LLM-backed planner
later by replacing `build_plans` with an async variant — the output contract
stays the same.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from kairo_core import AnalyticsPlan

from .schema import ColumnInfo


_CHART_INTENT = (
    "chart", "plot", "visualize", "visualise", "visualization", "visualisation",
    "distribution", "histogram", "breakdown", "pairwise",
    "compare", "trend", "trends", "over time", "top", "bottom",
    "summary", "summarize", "describe", "statistics", "stats",
    "scatter", " vs ", "versus", "against", "relationship",
    "correlation", "correlate", "correlates", "corr matrix",
    "forecast", "forecasts", "predict", "prediction", "projection", "project ",
    "next week", "next month", "next year", "future", "upcoming",
)

_HIST_HINT = ("distribution", "histogram", "spread", "range of")
_LINE_HINT = ("trend", "trends", "over time", "time series", "daily", "weekly", "monthly", "yearly")
_BAR_HINT = ("compare", "breakdown", "top", "bottom", " by ", " per ")
_STATS_HINT = ("summary", "summarize", "describe", "statistics", "stats", "mean of", "average of")
_SCATTER_HINT = ("scatter", " vs ", "versus", "against", "relationship between")
_CORR_HINT = ("correlation", "correlate", "correlates", "pairwise", "corr matrix")
_FORECAST_HINT = (
    "forecast", "forecasts", "predict", "prediction", "projection",
    "project ", "next week", "next month", "next year", "upcoming",
    "future values", "what's next", "whats next",
)


def has_chart_intent(query: str) -> bool:
    q = f" {query.lower()} "
    return any(kw in q for kw in _CHART_INTENT)


def build_plans(
    query: str,
    columns: List[ColumnInfo],
    *,
    max_plans: int = 3,
) -> List[AnalyticsPlan]:
    """Pick analytics tools based on query keywords + available columns.

    Returns an empty list if no clear intent is present — callers should
    treat that as "user didn't ask for a chart".
    """
    if not columns:
        return []

    q = f" {query.lower()} "
    numeric = [c for c in columns if c.kind == "numeric"]
    datetime_cols = [c for c in columns if c.kind == "datetime"]
    categorical = [c for c in columns if c.kind == "categorical"]
    mentioned = [c for c in columns if re.search(rf"\b{re.escape(c.name.lower())}\b", q)]

    plans: List[AnalyticsPlan] = []

    # 1. Distribution / histogram
    if _hits(q, _HIST_HINT) and numeric:
        target = _pick(mentioned, "numeric") or numeric[0]
        plans.append(
            AnalyticsPlan(
                tool="histogram",
                columns=[target.name],
                params={"bins": 20},
                rationale=f"distribution keyword + numeric column '{target.name}'",
            )
        )

    # 2. Trend / time series
    if _hits(q, _LINE_HINT) and datetime_cols and numeric:
        dt = _pick(mentioned, "datetime") or datetime_cols[0]
        y = _pick(mentioned, "numeric") or numeric[0]
        plans.append(
            AnalyticsPlan(
                tool="line_chart",
                columns=[dt.name, y.name],
                rationale=f"trend keyword + datetime '{dt.name}' over numeric '{y.name}'",
            )
        )

    # 3. Breakdown / bar by category
    if _hits(q, _BAR_HINT) and categorical:
        cat = _pick(mentioned, "categorical") or categorical[0]
        y = _pick(mentioned, "numeric") or (numeric[0] if numeric else None)
        cols = [cat.name] + ([y.name] if y else [])
        params: Dict[str, Any] = {
            "agg": "mean" if ("average" in q or "mean" in q) else "sum",
        }
        top_n = _extract_top_n(q)
        if top_n:
            params["top_n"] = top_n
        plans.append(
            AnalyticsPlan(
                tool="bar_chart",
                columns=cols,
                params=params,
                rationale=f"breakdown keyword + categorical column '{cat.name}'",
            )
        )

    # 4. Stats summary
    if _hits(q, _STATS_HINT) and numeric:
        mentioned_numeric = [c for c in mentioned if c.kind == "numeric"]
        chosen = mentioned_numeric or numeric[:3]
        plans.append(
            AnalyticsPlan(
                tool="describe",
                columns=[c.name for c in chosen],
                rationale="stats keyword triggered descriptive summary",
            )
        )

    # 5. Scatter (x vs y)
    if _hits(q, _SCATTER_HINT):
        mentioned_numeric = [c for c in mentioned if c.kind == "numeric"]
        if len(mentioned_numeric) >= 2:
            x, y = mentioned_numeric[0], mentioned_numeric[1]
        elif len(numeric) >= 2:
            x, y = numeric[0], numeric[1]
        else:
            x = y = None
        if x and y:
            plans.append(
                AnalyticsPlan(
                    tool="scatter",
                    columns=[x.name, y.name],
                    rationale=f"scatter keyword + numeric columns '{x.name}', '{y.name}'",
                )
            )

    # 6. Correlation matrix
    if _hits(q, _CORR_HINT) and len(numeric) >= 2:
        mentioned_numeric = [c for c in mentioned if c.kind == "numeric"]
        chosen = mentioned_numeric if len(mentioned_numeric) >= 2 else numeric[:6]
        plans.append(
            AnalyticsPlan(
                tool="correlation",
                columns=[c.name for c in chosen],
                params={"method": "pearson"},
                rationale=f"correlation keyword over {len(chosen)} numeric columns",
            )
        )

    # 7. Forecast (needs datetime + numeric)
    if _hits(q, _FORECAST_HINT) and datetime_cols and numeric:
        dt = _pick(mentioned, "datetime") or datetime_cols[0]
        y = _pick(mentioned, "numeric") or numeric[0]
        periods = _extract_horizon(q) or 14
        plans.append(
            AnalyticsPlan(
                tool="forecast",
                columns=[dt.name, y.name],
                params={"periods": periods, "alpha": 0.1},
                rationale=f"forecast keyword + datetime '{dt.name}' over numeric '{y.name}' for {periods} steps",
            )
        )

    # Fallback: chart intent without a specific match — default by schema shape.
    if not plans and has_chart_intent(query):
        if categorical and numeric:
            plans.append(
                AnalyticsPlan(
                    tool="bar_chart",
                    columns=[categorical[0].name, numeric[0].name],
                    params={"agg": "sum"},
                    rationale="chart intent fallback → bar_chart on first categorical + numeric",
                )
            )
        elif categorical:
            plans.append(
                AnalyticsPlan(
                    tool="bar_chart",
                    columns=[categorical[0].name],
                    rationale="chart intent fallback → category counts",
                )
            )
        elif numeric:
            plans.append(
                AnalyticsPlan(
                    tool="histogram",
                    columns=[numeric[0].name],
                    params={"bins": 20},
                    rationale="chart intent fallback → histogram of first numeric",
                )
            )

    return plans[:max_plans]


def _hits(query: str, hints: tuple[str, ...]) -> bool:
    return any(h in query for h in hints)


def _pick(columns: List[ColumnInfo], kind: str) -> Optional[ColumnInfo]:
    for c in columns:
        if c.kind == kind:
            return c
    return None


def _extract_top_n(q: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)", q)
    if m:
        return int(m.group(1))
    return None


def _extract_horizon(q: str) -> Optional[int]:
    m = re.search(r"\bnext\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("week"):
            return n * 7
        if unit.startswith("month"):
            return n * 30
        if unit.startswith("year"):
            return n * 365
        return n
    if re.search(r"\bnext\s+week\b", q):
        return 7
    if re.search(r"\bnext\s+month\b", q):
        return 30
    if re.search(r"\bnext\s+year\b", q):
        return 365
    return None

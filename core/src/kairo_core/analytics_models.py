"""Analytics data contracts — shared by kairo-analytics, kairo-agents, kairo-ai, and the UI.

These models never import pandas, numpy, sklearn, or any heavy analytics dep.
They exist so that the UI can render charts and so that kairo-agents can emit
declarative plans without pulling in the analytics runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


ChartKind = Literal[
    "bar",
    "line",
    "pie",
    "scatter",
    "histogram",
    "pairwise",
]

AggregationKind = Literal["sum", "mean", "count", "min", "max", "median"]


class ChartSeries(BaseModel):
    """One named series inside a chart (e.g. one line on a multi-line chart)."""

    name: str
    data: List[Union[float, int, str, None]]


class ChartSpec(BaseModel):
    """Precomputed chart payload. UI renders directly; no transformation on the client."""

    kind: ChartKind
    title: str
    subtitle: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_values: List[Union[float, int, str]] = Field(default_factory=list)
    series: List[ChartSeries] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForecastSpec(BaseModel):
    """Time-series forecast output — renders as a line chart with a confidence band."""

    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    history_x: List[Union[float, int, str]] = Field(default_factory=list)
    history_y: List[float] = Field(default_factory=list)
    forecast_x: List[Union[float, int, str]] = Field(default_factory=list)
    forecast_y: List[float] = Field(default_factory=list)
    forecast_lower: List[float] = Field(default_factory=list)
    forecast_upper: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StatSummary(BaseModel):
    """Descriptive stats for a column or pair of columns."""

    column: str
    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsPlan(BaseModel):
    """What the `analyst` agent emits. The runner dispatches on `tool` via the registry."""

    tool: str
    columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class AnalyticsResult(BaseModel):
    """Runner output — a plan produces at most one of these per execution."""

    plan: AnalyticsPlan
    chart: Optional[ChartSpec] = None
    forecast: Optional[ForecastSpec] = None
    stats: List[StatSummary] = Field(default_factory=list)
    error: Optional[str] = None

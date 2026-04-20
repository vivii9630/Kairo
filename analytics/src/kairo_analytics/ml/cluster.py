"""KMeans clustering over 2 numeric columns. Emits a multi-series scatter."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from kairo_core import AnalyticsPlan, AnalyticsResult, ChartSeries, ChartSpec

from ..registry import AnalyticsTool, register


def _kmeans_handler(df: pd.DataFrame, plan: AnalyticsPlan) -> AnalyticsResult:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return AnalyticsResult(
            plan=plan,
            error="scikit-learn is not installed. pip install 'kairo-analytics[ml]'",
        )

    params: Dict[str, Any] = plan.params or {}
    x_col = plan.columns[0]
    y_col = plan.columns[1]
    k = int(params.get("k", 3))
    if k < 2:
        k = 2
    if k > 10:
        k = 10

    x_num = pd.to_numeric(df[x_col], errors="coerce")
    y_num = pd.to_numeric(df[y_col], errors="coerce")
    mask = x_num.notna() & y_num.notna()
    x_vals = x_num[mask].to_numpy()
    y_vals = y_num[mask].to_numpy()

    if len(x_vals) < k:
        return AnalyticsResult(
            plan=plan,
            error=f"need at least k={k} valid rows to cluster; got {len(x_vals)}",
        )

    X = np.column_stack([x_vals, y_vals])
    try:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    except Exception as exc:  # noqa: BLE001
        return AnalyticsResult(plan=plan, error=f"kmeans failed: {exc}")

    labels = model.labels_
    centroids = model.cluster_centers_.tolist()
    inertia = float(model.inertia_)

    # Sparse series — one per cluster. Point i appears in series[labels[i]].data[i]
    # with all other series[j].data[i] = None. This keeps x_values as one shared
    # array while letting the UI color points by cluster.
    x_values = [float(v) for v in x_vals.tolist()]
    series = []
    for cluster_id in range(k):
        data: list[float | None] = []
        for i, yi in enumerate(y_vals):
            data.append(float(yi) if labels[i] == cluster_id else None)
        series.append(ChartSeries(name=f"cluster {cluster_id}", data=data))

    title = params.get("title") or f"KMeans k={k} over {x_col}, {y_col}"
    subtitle = f"inertia = {inertia:.1f}"
    return AnalyticsResult(
        plan=plan,
        chart=ChartSpec(
            kind="scatter",
            title=title,
            subtitle=subtitle,
            x_label=x_col,
            y_label=y_col,
            x_values=x_values,
            series=series,
            metadata={
                "model": "KMeans",
                "k": k,
                "inertia": inertia,
                "centroids": centroids,
                "n": int(mask.sum()),
            },
        ),
    )


register(
    AnalyticsTool(
        name="kmeans",
        description=(
            "KMeans clustering over 2 numeric columns. params.k = number of "
            "clusters (default 3, clamped to [2, 10]). Emits a multi-series "
            "scatter (one series per cluster) with centroids in metadata."
        ),
        handler=_kmeans_handler,
        required_columns=2,
    )
)

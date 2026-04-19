"""Infer a typed column schema from a pandas DataFrame.

The planner (and downstream agents) only deal with `ColumnInfo` — a pure
dataclass with no pandas dependency — so intent-based rules can stay
readable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import pandas as pd


ColumnKind = Literal["numeric", "datetime", "categorical", "text"]


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: ColumnKind
    nunique: int
    non_null: int


def infer_columns(df: pd.DataFrame, *, sample_rows: int = 200) -> List[ColumnInfo]:
    """Classify each column as numeric / datetime / categorical / text.

    Heuristics (in order):
      - pandas dtype says numeric → numeric
      - pandas dtype says datetime → datetime
      - values parse cleanly as datetime → datetime
      - unique count ≤ 20 or unique/non_null ≤ 0.2 → categorical
      - otherwise → text
    """
    info: List[ColumnInfo] = []
    head = df.head(sample_rows) if len(df) > sample_rows else df
    for name in df.columns:
        series = df[name]
        non_null = int(series.notna().sum())
        nunique = int(series.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(series):
            info.append(ColumnInfo(name=name, kind="numeric", nunique=nunique, non_null=non_null))
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            info.append(ColumnInfo(name=name, kind="datetime", nunique=nunique, non_null=non_null))
            continue
        if _looks_like_datetime(head[name]):
            info.append(ColumnInfo(name=name, kind="datetime", nunique=nunique, non_null=non_null))
            continue
        if non_null > 0 and (nunique <= 20 or nunique / max(non_null, 1) <= 0.2):
            info.append(ColumnInfo(name=name, kind="categorical", nunique=nunique, non_null=non_null))
            continue
        info.append(ColumnInfo(name=name, kind="text", nunique=nunique, non_null=non_null))
    return info


def _looks_like_datetime(series: pd.Series) -> bool:
    try:
        parsed = pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        return False
    # At least 80% of non-null values must parse to be called a datetime column.
    non_null = series.notna().sum()
    if non_null == 0:
        return False
    return (parsed.notna().sum() / non_null) >= 0.8

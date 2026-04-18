# kairo-analytics

Analytics and ML capabilities for Kairo. Produces `ChartSpec`, `ForecastSpec`,
and `StatSummary` payloads defined in `kairo-core` — the UI and agents layers
depend only on those contracts, never on this package's heavy runtime
(pandas, sklearn, statsmodels).

## Shape

```
kairo_analytics/
├── charting/     # pandas-backed chart builders  (bar, line, pie, scatter, histogram, pairwise)
├── stats/        # descriptive statistics and correlation
├── forecasting/  # time-series forecasting  (optional extra: [forecast])
├── ml/           # clustering / classification / regression  (optional extra: [ml])
├── registry.py   # name → callable dispatch used by the analyst agent
└── runner.py     # AnalyticsPlan → AnalyticsResult, single public entrypoint
```

## Install

```
pip install -e ./analytics                 # charts + stats only
pip install -e "./analytics[forecast]"     # adds statsmodels
pip install -e "./analytics[ml]"           # adds scikit-learn
pip install -e "./analytics[all]"          # everything
```

## Rule

Nothing in `kairo-analytics` imports from `kairo-agents` or `kairo-ai`.
The dependency arrow points inward: agents → analytics → core.

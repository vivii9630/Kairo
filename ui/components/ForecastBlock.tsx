"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastSpec } from "@/lib/types";

const HISTORY_COLOR = "#8bb1ff";
const FORECAST_COLOR = "#ffd56b";
const BAND_COLOR = "#ffd56b";

interface ForecastBlockProps {
  forecast: ForecastSpec;
}

export function ForecastBlock({ forecast }: ForecastBlockProps) {
  const rows = toRows(forecast);
  const confidencePct = formatConfidence(forecast.metadata);

  return (
    <div className="rounded-lg border border-hair bg-panel p-4 my-3">
      <div className="mb-2">
        <div className="text-sm text-accent font-medium">{forecast.title}</div>
        {confidencePct && (
          <div className="text-xs text-muted mt-0.5">
            {confidencePct} confidence band
          </div>
        )}
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={rows}
            margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
          >
            <CartesianGrid stroke="#2a2b2b" strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              stroke="#8a8a8a"
              tick={{ fontSize: 11 }}
              tickFormatter={shortDate}
            />
            <YAxis stroke="#8a8a8a" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{
                background: "#1a1a1a",
                border: "1px solid #2a2b2b",
                fontSize: 12,
              }}
              labelFormatter={shortDate}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Area
              type="monotone"
              dataKey="band"
              stroke="none"
              fill={BAND_COLOR}
              fillOpacity={0.15}
              legendType="none"
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="history"
              name="history"
              stroke={HISTORY_COLOR}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="forecast"
              name="forecast"
              stroke={FORECAST_COLOR}
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={false}
              connectNulls
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

type Row = {
  x: string | number;
  history: number | null;
  forecast: number | null;
  band: [number, number] | null;
};

function toRows(forecast: ForecastSpec): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < forecast.history_x.length; i++) {
    rows.push({
      x: forecast.history_x[i],
      history: forecast.history_y[i] ?? null,
      forecast: null,
      band: null,
    });
  }
  // Bridge point: stitch last historical value into the forecast line so
  // the two series visually connect without a gap.
  const lastHist =
    forecast.history_y.length > 0
      ? forecast.history_y[forecast.history_y.length - 1]
      : null;
  for (let i = 0; i < forecast.forecast_x.length; i++) {
    const y = forecast.forecast_y[i] ?? null;
    const lo = forecast.forecast_lower[i];
    const hi = forecast.forecast_upper[i];
    rows.push({
      x: forecast.forecast_x[i],
      history: i === 0 ? lastHist : null,
      forecast: y,
      band: Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : null,
    });
  }
  return rows;
}

function shortDate(v: string | number): string {
  if (typeof v !== "string") return String(v);
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toISOString().slice(0, 10);
}

function formatConfidence(metadata: Record<string, unknown>): string | null {
  const raw = metadata?.confidence;
  if (typeof raw !== "number" || !Number.isFinite(raw)) return null;
  return `${Math.round(raw * 100)}%`;
}

export function ForecastStack({ forecasts }: { forecasts: ForecastSpec[] }) {
  if (!forecasts || forecasts.length === 0) return null;
  return (
    <div className="space-y-0">
      {forecasts.map((f, i) => (
        <ForecastBlock key={i} forecast={f} />
      ))}
    </div>
  );
}

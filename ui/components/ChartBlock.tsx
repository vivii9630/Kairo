"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartSpec } from "@/lib/types";

const SERIES_COLORS = [
  "#8bb1ff",
  "#8bffaf",
  "#ffa28b",
  "#ffd56b",
  "#c78bff",
  "#8be1ff",
];

interface ChartBlockProps {
  chart: ChartSpec;
}

export function ChartBlock({ chart }: ChartBlockProps) {
  return (
    <div className="rounded-lg border border-hair bg-panel p-4 my-3">
      <div className="mb-2">
        <div className="text-sm text-accent font-medium">{chart.title}</div>
        {chart.subtitle && (
          <div className="text-xs text-muted mt-0.5">{chart.subtitle}</div>
        )}
      </div>
      <div className="h-64 w-full">
        <ChartBody chart={chart} />
      </div>
    </div>
  );
}

function ChartBody({ chart }: { chart: ChartSpec }) {
  const rows = toRows(chart);
  const seriesNames = chart.series.map((s) => s.name);

  switch (chart.kind) {
    case "line":
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="#2a2b2b" strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              stroke="#8a8a8a"
              tick={{ fontSize: 11 }}
              label={axisLabel(chart.x_label, "bottom")}
            />
            <YAxis
              stroke="#8a8a8a"
              tick={{ fontSize: 11 }}
              label={axisLabel(chart.y_label, "left")}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1a1a",
                border: "1px solid #2a2b2b",
                fontSize: 12,
              }}
            />
            {seriesNames.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {seriesNames.map((name, i) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    case "bar":
    case "histogram":
    default:
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="#2a2b2b" strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              stroke="#8a8a8a"
              tick={{ fontSize: 11 }}
              label={axisLabel(chart.x_label, "bottom")}
            />
            <YAxis
              stroke="#8a8a8a"
              tick={{ fontSize: 11 }}
              label={axisLabel(chart.y_label, "left")}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1a1a",
                border: "1px solid #2a2b2b",
                fontSize: 12,
              }}
            />
            {seriesNames.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {seriesNames.map((name, i) => (
              <Bar
                key={name}
                dataKey={name}
                fill={SERIES_COLORS[i % SERIES_COLORS.length]}
                radius={[2, 2, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      );
  }
}

function toRows(chart: ChartSpec): Array<Record<string, number | string | null>> {
  return chart.x_values.map((x, i) => {
    const row: Record<string, number | string | null> = { x };
    for (const s of chart.series) {
      row[s.name] = s.data[i] ?? null;
    }
    return row;
  });
}

function axisLabel(
  value: string | null | undefined,
  position: "bottom" | "left"
) {
  if (!value) return undefined;
  return {
    value,
    position: position === "bottom" ? "insideBottom" : "insideLeft",
    fill: "#8a8a8a",
    fontSize: 11,
    offset: position === "bottom" ? -2 : 0,
    angle: position === "left" ? -90 : 0,
  };
}

export function ChartStack({ charts }: { charts: ChartSpec[] }) {
  if (!charts || charts.length === 0) return null;
  return (
    <div className="space-y-0">
      {charts.map((c, i) => (
        <ChartBlock key={i} chart={c} />
      ))}
    </div>
  );
}

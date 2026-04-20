"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
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
    case "scatter":
      return <ScatterBody chart={chart} />;
    case "pairwise":
      return <PairwiseHeatmap chart={chart} />;
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

function ScatterBody({ chart }: { chart: ChartSpec }) {
  const xs = chart.x_values as Array<number | string>;
  const seriesGroups = chart.series.map((s, i) => ({
    name: s.name,
    color: SERIES_COLORS[i % SERIES_COLORS.length],
    points: xs
      .map((x, j) => ({ x: Number(x), y: (s.data[j] ?? null) as number | null }))
      .filter(
        (p) =>
          p.y !== null &&
          Number.isFinite(p.x) &&
          Number.isFinite(p.y as number)
      ),
  }));

  const fitLine = extractFitLine(chart);
  const showLegend = seriesGroups.length > 1;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#2a2b2b" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          stroke="#8a8a8a"
          tick={{ fontSize: 11 }}
          label={axisLabel(chart.x_label, "bottom")}
        />
        <YAxis
          type="number"
          dataKey="y"
          stroke="#8a8a8a"
          tick={{ fontSize: 11 }}
          label={axisLabel(chart.y_label, "left")}
        />
        <ZAxis range={[50, 50]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3", stroke: "#555" }}
          contentStyle={{
            background: "#1a1a1a",
            border: "1px solid #2a2b2b",
            fontSize: 12,
          }}
        />
        {showLegend && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {seriesGroups.map((g) => (
          <Scatter key={g.name} name={g.name} data={g.points} fill={g.color} />
        ))}
        {fitLine && (
          <Scatter
            name="fit"
            data={fitLine}
            fill="#ffd56b"
            line={{ stroke: "#ffd56b", strokeWidth: 2 }}
            shape={() => null as unknown as React.ReactElement}
            legendType="none"
          />
        )}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function extractFitLine(chart: ChartSpec): Array<{ x: number; y: number }> | null {
  const md = chart.metadata ?? {};
  const xs = md.x_line as unknown;
  const ys = md.y_line as unknown;
  if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length !== 2 || ys.length !== 2) {
    return null;
  }
  const [x0, x1] = xs.map(Number);
  const [y0, y1] = ys.map(Number);
  if (![x0, x1, y0, y1].every(Number.isFinite)) return null;
  return [
    { x: x0, y: y0 },
    { x: x1, y: y1 },
  ];
}

function PairwiseHeatmap({ chart }: { chart: ChartSpec }) {
  const cols = chart.x_values as string[];
  const matrix = chart.series.map((s) => s.data as Array<number | null>);

  return (
    <div className="h-full w-full overflow-auto">
      <table className="border-collapse text-xs">
        <thead>
          <tr>
            <th className="p-1 text-muted text-right font-normal"></th>
            {cols.map((c) => (
              <th
                key={c}
                className="p-1 text-muted font-normal text-center"
                style={{ minWidth: 48 }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cols.map((rowCol, i) => (
            <tr key={rowCol}>
              <th className="p-1 text-muted text-right font-normal">{rowCol}</th>
              {matrix[i].map((v, j) => (
                <td
                  key={j}
                  className="p-1 text-center tabular-nums"
                  style={{
                    background: corrColor(v),
                    color: corrTextColor(v),
                    minWidth: 48,
                    border: "1px solid #1a1a1a",
                  }}
                >
                  {v === null ? "—" : v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function corrColor(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "#151515";
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const a = 0.15 + 0.6 * t;
    return `rgba(139, 177, 255, ${a})`;
  }
  const a = 0.15 + 0.6 * -t;
  return `rgba(255, 162, 139, ${a})`;
}

function corrTextColor(v: number | null): string {
  if (v === null) return "#666";
  return Math.abs(v) > 0.6 ? "#0a0a0a" : "#d4d4d4";
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

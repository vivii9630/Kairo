"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import type {
  TraceStep,
  LayeredGraphData,
  TraversalTrace,
} from "@/lib/types";

// Lazy-load Graph3D to avoid SSR issues with Three.js
const Graph3D = dynamic(
  () => import("./Graph3D").then((m) => ({ default: m.Graph3D })),
  { ssr: false, loading: () => <div className="flex-1 bg-[#0a0a0a] animate-pulse" /> }
);

interface TracePanelProps {
  trace: TraceStep[];
  stubbed: boolean;
  graphData?: LayeredGraphData | null;
  traversalTrace?: TraversalTrace | null;
}

type Tab = "graph" | "steps";

const KIND_COLORS: Record<TraceStep["kind"], string> = {
  plan: "bg-[#2a2e3a] text-[#8bb1ff]",
  retrieve: "bg-[#2a3a2e] text-[#8bffaf]",
  synthesize: "bg-[#3a2a2e] text-[#ffa28b]",
  tool: "bg-[#2a363a] text-[#8be1ff]",
  note: "bg-[#2a2a2a] text-[#bdbdbd]",
};

export function TracePanel({
  trace,
  stubbed,
  graphData,
  traversalTrace,
}: TracePanelProps) {
  const hasGraph = !!graphData;
  const [activeTab, setActiveTab] = useState<Tab>(hasGraph ? "graph" : "steps");

  // Stats for the header
  const nodeCount = graphData
    ? graphData.document.nodes.length +
      graphData.semantic.nodes.length +
      graphData.detail.nodes.length
    : 0;
  const visitedCount = traversalTrace
    ? Object.values(traversalTrace.visited_nodes).reduce(
        (sum, ids) => sum + ids.length,
        0
      )
    : 0;

  return (
    <aside className="w-96 shrink-0 border-l border-hair bg-surface h-full flex flex-col">
      {/* Header */}
      <div className="px-4 h-14 flex items-center justify-between border-b border-hair">
        <div className="flex items-center gap-3">
          {/* Tab buttons */}
          <button
            onClick={() => setActiveTab("graph")}
            className={`text-sm font-medium transition-colors ${
              activeTab === "graph" ? "text-accent" : "text-muted hover:text-accent"
            }`}
          >
            3D Graph
          </button>
          <button
            onClick={() => setActiveTab("steps")}
            className={`text-sm font-medium transition-colors ${
              activeTab === "steps" ? "text-accent" : "text-muted hover:text-accent"
            }`}
          >
            Steps
          </button>
        </div>
        <div className="flex items-center gap-2">
          {hasGraph && (
            <span className="text-[10px] text-muted">
              {visitedCount}/{nodeCount} nodes
            </span>
          )}
          {stubbed && (
            <span className="text-[10px] uppercase tracking-wide text-muted border border-hair rounded px-1.5 py-0.5">
              stub
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 flex flex-col">
        {activeTab === "graph" ? (
          <GraphTab graphData={graphData} traversalTrace={traversalTrace} />
        ) : (
          <StepsTab trace={trace} />
        )}
      </div>

      {/* Graph legend */}
      {activeTab === "graph" && hasGraph && (
        <div className="px-4 py-2 border-t border-hair flex items-center gap-3 text-[10px] text-muted">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#8bb1ff]" /> Document
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#8bffaf]" /> Semantic
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#ffa28b]" /> Detail
          </span>
        </div>
      )}
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Tab content
// ---------------------------------------------------------------------------

function GraphTab({
  graphData,
  traversalTrace,
}: {
  graphData?: LayeredGraphData | null;
  traversalTrace?: TraversalTrace | null;
}) {
  if (!graphData) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted px-6 text-center">
        Ask something to see the 3-layer knowledge graph. Drag to rotate, scroll
        to zoom.
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0">
      <Graph3D graph={graphData} traversal={traversalTrace} />
    </div>
  );
}

function StepsTab({ trace }: { trace: TraceStep[] }) {
  if (trace.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto p-4">
        <div className="text-sm text-muted">
          Ask something to see how Kairo plans, retrieves, and synthesizes.
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <ol className="relative space-y-4 pl-4 border-l border-hair">
        {trace.map((step) => (
          <li key={step.step} className="relative">
            <span className="absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full bg-panel border border-muted" />
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-muted">#{step.step}</span>
              <span
                className={`text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded ${
                  KIND_COLORS[step.kind] ?? KIND_COLORS.note
                }`}
              >
                {step.kind}
              </span>
            </div>
            <div className="text-sm text-accent leading-relaxed">
              {step.summary}
            </div>
            {step.snapshot_id && (
              <div className="text-xs text-muted mt-1">
                snapshot {step.snapshot_id.slice(0, 8)}
              </div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

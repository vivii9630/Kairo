"use client";

import type { TraceStep } from "@/lib/types";

interface TracePanelProps {
  trace: TraceStep[];
  stubbed: boolean;
}

const KIND_COLORS: Record<TraceStep["kind"], string> = {
  plan: "bg-[#2a2e3a] text-[#8bb1ff]",
  retrieve: "bg-[#2a3a2e] text-[#8bffaf]",
  synthesize: "bg-[#3a2a2e] text-[#ffa28b]",
  tool: "bg-[#2a363a] text-[#8be1ff]",
  note: "bg-[#2a2a2a] text-[#bdbdbd]",
};

export function TracePanel({ trace, stubbed }: TracePanelProps) {
  return (
    <aside className="w-80 shrink-0 border-l border-hair bg-surface h-full flex flex-col">
      <div className="px-4 h-14 flex items-center justify-between border-b border-hair">
        <span className="text-sm font-medium">Agent trace</span>
        {stubbed && (
          <span className="text-[10px] uppercase tracking-wide text-muted border border-hair rounded px-1.5 py-0.5">
            stub
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {trace.length === 0 ? (
          <div className="text-sm text-muted">
            Ask something to see how Kairo plans, retrieves, and synthesizes. The
            temporal graph overlay lands in Phase 7.
          </div>
        ) : (
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
        )}
      </div>
    </aside>
  );
}

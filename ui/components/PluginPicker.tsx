"use client";

import { useEffect, useRef } from "react";
import type { PluginSummary } from "@/lib/types";

interface PluginPickerProps {
  plugins: PluginSummary[];
  selected: string | null;
  onSelect: (name: string | null) => void;
  open: boolean;
  onClose: () => void;
}

export function PluginPicker({
  plugins,
  selected,
  onSelect,
  open,
  onClose,
}: PluginPickerProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={ref}
      className="absolute bottom-full left-0 mb-2 w-80 rounded-lg border border-hair bg-panel shadow-xl overflow-hidden z-10"
    >
      <div className="px-3 py-2 text-xs uppercase tracking-wide text-muted border-b border-hair">
        Connect a source
      </div>
      <ul className="max-h-80 overflow-y-auto">
        <li>
          <button
            onClick={() => {
              onSelect(null);
              onClose();
            }}
            className={`w-full flex items-start gap-3 px-3 py-3 text-left hover:bg-panel2 ${
              selected === null ? "bg-panel2" : ""
            }`}
          >
            <span className="w-8 h-8 rounded-md bg-hair grid place-items-center text-muted">
              –
            </span>
            <div className="flex-1">
              <div className="text-sm text-accent">No connector</div>
              <div className="text-xs text-muted">
                Ask a question without grounding to a specific source.
              </div>
            </div>
          </button>
        </li>
        {plugins.map((p) => (
          <li key={p.name}>
            <button
              onClick={() => {
                onSelect(p.name);
                onClose();
              }}
              className={`w-full flex items-start gap-3 px-3 py-3 text-left hover:bg-panel2 ${
                selected === p.name ? "bg-panel2" : ""
              }`}
            >
              <span className="w-8 h-8 rounded-md bg-hair grid place-items-center text-accent uppercase text-xs font-semibold">
                {p.label.slice(0, 2)}
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-sm text-accent flex items-center gap-2">
                  <span>{p.label}</span>
                  {p.auth_required && (
                    <span className="text-[10px] uppercase tracking-wide text-muted border border-hair rounded px-1 py-0.5">
                      auth
                    </span>
                  )}
                </div>
                <div className="text-xs text-muted truncate">
                  {p.description}
                </div>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

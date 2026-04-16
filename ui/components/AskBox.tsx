"use client";

import { useState } from "react";
import type { PluginSummary } from "@/lib/types";
import { PluginPicker } from "./PluginPicker";

interface AskBoxProps {
  plugins: PluginSummary[];
  selectedPlugin: string | null;
  onSelectPlugin: (name: string | null) => void;
  value: string;
  onChange: (value: string) => void;
  onSubmit: (query: string) => void;
  disabled?: boolean;
  autoFocus?: boolean;
}

export function AskBox({
  plugins,
  selectedPlugin,
  onSelectPlugin,
  value,
  onChange,
  onSubmit,
  disabled,
  autoFocus,
}: AskBoxProps) {
  const [pickerOpen, setPickerOpen] = useState(false);

  const selected = plugins.find((p) => p.name === selectedPlugin) || null;

  function submit() {
    const q = value.trim();
    if (!q || disabled) return;
    onSubmit(q);
  }

  return (
    <div className="w-full rounded-xl border border-hair bg-panel focus-within:border-muted transition-colors">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Ask anything…"
        rows={2}
        autoFocus={autoFocus}
        className="block w-full bg-transparent resize-none px-4 pt-4 pb-2 text-sm text-accent placeholder-muted focus:outline-none"
      />
      <div className="flex items-center gap-2 px-3 pb-3">
        <div className="relative">
          <button
            onClick={() => setPickerOpen((o) => !o)}
            className="flex items-center gap-2 px-2.5 h-8 rounded-md bg-panel2 border border-hair text-sm text-accent hover:bg-hair"
          >
            <span className="w-4 h-4 rounded bg-hair grid place-items-center text-[10px] uppercase">
              {selected ? selected.label.slice(0, 1) : "+"}
            </span>
            <span>{selected ? selected.label : "Connector"}</span>
            <span className="text-muted">+</span>
          </button>
          <PluginPicker
            plugins={plugins}
            selected={selectedPlugin}
            onSelect={onSelectPlugin}
            open={pickerOpen}
            onClose={() => setPickerOpen(false)}
          />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button className="flex items-center gap-1 px-2.5 h-8 rounded-md text-sm text-muted hover:text-accent">
            <span>Model</span>
            <span>⌄</span>
          </button>
          <button
            onClick={submit}
            disabled={disabled || !value.trim()}
            className="w-8 h-8 rounded-full bg-accent text-surface grid place-items-center disabled:opacity-40 disabled:cursor-not-allowed hover:bg-white transition-colors"
            aria-label="Ask"
          >
            ↑
          </button>
        </div>
      </div>
    </div>
  );
}

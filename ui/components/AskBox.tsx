"use client";

import { useRef, useState } from "react";
import type { IngestResponse, PluginSummary } from "@/lib/types";
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

  // Ingest controls
  ingest: IngestResponse | null;
  ingestPending: boolean;
  ingestError: string | null;
  onIngestUrl: (url: string) => void;
  onIngestFile: (file: File) => void;
  onClearIngest: () => void;
}

const KIND_ICON: Record<IngestResponse["kind"], string> = {
  github: "⌘",
  csv: "▦",
  xlsx: "▦",
};

export function AskBox({
  plugins,
  selectedPlugin,
  onSelectPlugin,
  value,
  onChange,
  onSubmit,
  disabled,
  autoFocus,
  ingest,
  ingestPending,
  ingestError,
  onIngestUrl,
  onIngestFile,
  onClearIngest,
}: AskBoxProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [githubUrl, setGithubUrl] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selected = plugins.find((p) => p.name === selectedPlugin) || null;

  function submit() {
    const q = value.trim();
    if (!q || disabled) return;
    onSubmit(q);
  }

  function submitGithubUrl() {
    const url = githubUrl.trim();
    if (!url || ingestPending) return;
    onIngestUrl(url);
  }

  function onFilePicked(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    onIngestFile(f);
    // allow selecting the same file again later
    e.target.value = "";
  }

  return (
    <div className="w-full space-y-2">
      {/* Source row — GitHub URL + file attach + active-ingest chip */}
      <div className="flex items-center gap-2 text-xs">
        <input
          type="text"
          value={githubUrl}
          onChange={(e) => setGithubUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              submitGithubUrl();
            }
          }}
          placeholder="Paste a public GitHub URL…"
          disabled={ingestPending}
          className="flex-1 min-w-0 bg-panel border border-hair rounded-md px-3 h-8 text-accent placeholder-muted focus:outline-none focus:border-muted disabled:opacity-50"
        />
        <button
          onClick={submitGithubUrl}
          disabled={ingestPending || !githubUrl.trim()}
          className="h-8 px-3 rounded-md border border-hair text-muted hover:text-accent hover:bg-panel disabled:opacity-40"
        >
          {ingestPending ? "…" : "Load"}
        </button>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={ingestPending}
          title="Attach CSV or XLSX"
          className="h-8 w-8 rounded-md border border-hair grid place-items-center text-muted hover:text-accent hover:bg-panel disabled:opacity-40"
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xlsm"
          onChange={onFilePicked}
          className="hidden"
        />
      </div>

      {/* Active ingest chip */}
      {ingest && (
        <div className="flex items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-panel border border-hair text-accent">
            <span className="text-muted">{KIND_ICON[ingest.kind]}</span>
            <span>
              {ingest.label}{" "}
              <span className="text-muted">· {ingest.doc_count} docs</span>
            </span>
            <button
              onClick={onClearIngest}
              className="ml-1 text-muted hover:text-accent"
              aria-label="Clear attached source"
            >
              ✕
            </button>
          </span>
        </div>
      )}

      {ingestError && (
        <div className="text-[11px] text-red-400">{ingestError}</div>
      )}

      {/* Ask textarea + controls */}
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
          placeholder={
            ingest
              ? `Ask anything about ${ingest.label}…`
              : "Ask anything…"
          }
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
    </div>
  );
}

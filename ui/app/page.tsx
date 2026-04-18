"use client";

import { useCallback, useEffect, useState } from "react";
import Image from "next/image";
import { api } from "@/lib/api";
import type {
  AskResponse,
  IngestResponse,
  LayeredGraphData,
  Message,
  PluginSummary,
  Thread,
  TraceStep,
  TraversalTrace,
} from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { TopNav } from "@/components/TopNav";
import { AskBox } from "@/components/AskBox";
import { MessageList } from "@/components/MessageList";
import { TracePanel } from "@/components/TracePanel";

const SUGGESTIONS = [
  "Summarize the most recent changes in my GitHub repo",
  "What did the team discuss about the auth migration in Slack last week?",
  "Pull my Q3 financial spreadsheet from Drive and flag anomalies",
  "Find emails from legal about the compliance review",
];

export default function HomePage() {
  const [plugins, setPlugins] = useState<PluginSummary[]>([]);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThread, setActiveThread] = useState<Thread | null>(null);
  const [selectedPlugin, setSelectedPlugin] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [graphData, setGraphData] = useState<LayeredGraphData | null>(null);
  const [traversalTrace, setTraversalTrace] = useState<TraversalTrace | null>(null);
  const [stubbed, setStubbed] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const [ingest, setIngest] = useState<IngestResponse | null>(null);
  const [ingestPending, setIngestPending] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  useEffect(() => {
    api.getPlugins().then(setPlugins).catch((e) => setError(String(e)));
    api.listThreads().then(setThreads).catch(() => {});
  }, []);

  const refreshThreads = useCallback(async () => {
    try {
      const list = await api.listThreads();
      setThreads(list);
    } catch {
      /* ignore */
    }
  }, []);

  const loadThread = useCallback(async (id: string) => {
    const t = await api.getThread(id);
    setActiveThread(t);
    setMessages(t.messages);
    setSelectedPlugin(t.plugin ?? null);
    setTrace([]);
    setGraphData(null);
    setTraversalTrace(null);
    setDraft("");
    setIngest(null);
    setIngestError(null);
  }, []);

  function newThread() {
    setActiveThread(null);
    setMessages([]);
    setTrace([]);
    setGraphData(null);
    setTraversalTrace(null);
    setError(null);
    setDraft("");
    setIngest(null);
    setIngestError(null);
  }

  async function handleIngestUrl(url: string) {
    setIngestError(null);
    setIngestPending(true);
    try {
      const res = await api.ingestUrl(url);
      setIngest(res);
    } catch (e) {
      setIngestError(String(e));
    } finally {
      setIngestPending(false);
    }
  }

  async function handleIngestFile(file: File) {
    setIngestError(null);
    setIngestPending(true);
    try {
      const res = await api.ingestFile(file);
      setIngest(res);
    } catch (e) {
      setIngestError(String(e));
    } finally {
      setIngestPending(false);
    }
  }

  function clearIngest() {
    setIngest(null);
    setIngestError(null);
  }

  async function handleAsk(query: string) {
    setError(null);
    setPending(true);
    setDraft("");
    const now = new Date().toISOString();
    setMessages((m) => [...m, { role: "user", content: query, created_at: now }]);
    try {
      const res: AskResponse = await api.ask({
        query,
        plugin: selectedPlugin,
        thread_id: activeThread?.id ?? null,
        ingest_id: ingest?.ingest_id ?? null,
      });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.answer,
          created_at: new Date().toISOString(),
        },
      ]);
      setTrace(res.trace);
      setGraphData(res.graph_data ?? null);
      setTraversalTrace(res.traversal_trace ?? null);
      setStubbed(res.stubbed);
      if (!activeThread) {
        const t = await api.getThread(res.thread_id);
        setActiveThread(t);
      }
      await refreshThreads();
    } catch (e) {
      setError(String(e));
    } finally {
      setPending(false);
    }
  }

  const isLanding = !activeThread && messages.length === 0;
  const showLogo = isLanding && draft.length === 0;

  return (
    <div className="flex h-full">
      <Sidebar
        threads={threads}
        activeThreadId={activeThread?.id ?? null}
        onNewThread={newThread}
        onSelectThread={(id) => loadThread(id).catch((e) => setError(String(e)))}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <TopNav />

        <div className="flex-1 flex min-h-0">
          <div className="flex-1 flex flex-col min-w-0">
            {isLanding ? (
              <LandingView
                plugins={plugins}
                selectedPlugin={selectedPlugin}
                onSelectPlugin={setSelectedPlugin}
                onAsk={handleAsk}
                pending={pending}
                draft={draft}
                onDraftChange={setDraft}
                showLogo={showLogo}
                ingest={ingest}
                ingestPending={ingestPending}
                ingestError={ingestError}
                onIngestUrl={handleIngestUrl}
                onIngestFile={handleIngestFile}
                onClearIngest={clearIngest}
              />
            ) : (
              <ThreadView
                plugins={plugins}
                selectedPlugin={selectedPlugin}
                onSelectPlugin={setSelectedPlugin}
                messages={messages}
                pending={pending}
                onAsk={handleAsk}
                draft={draft}
                onDraftChange={setDraft}
                ingest={ingest}
                ingestPending={ingestPending}
                ingestError={ingestError}
                onIngestUrl={handleIngestUrl}
                onIngestFile={handleIngestFile}
                onClearIngest={clearIngest}
              />
            )}
            {error && (
              <div className="mx-auto w-full max-w-3xl px-6 pb-4">
                <div className="text-xs text-red-400 border border-red-900/60 bg-red-950/40 rounded-md px-3 py-2">
                  {error}
                </div>
              </div>
            )}
          </div>

          <TracePanel
            trace={trace}
            stubbed={stubbed}
            graphData={graphData}
            traversalTrace={traversalTrace}
          />
        </div>
      </main>
    </div>
  );
}

interface IngestProps {
  ingest: IngestResponse | null;
  ingestPending: boolean;
  ingestError: string | null;
  onIngestUrl: (url: string) => void;
  onIngestFile: (file: File) => void;
  onClearIngest: () => void;
}

function LandingView({
  plugins,
  selectedPlugin,
  onSelectPlugin,
  onAsk,
  pending,
  draft,
  onDraftChange,
  showLogo,
  ingest,
  ingestPending,
  ingestError,
  onIngestUrl,
  onIngestFile,
  onClearIngest,
}: {
  plugins: PluginSummary[];
  selectedPlugin: string | null;
  onSelectPlugin: (name: string | null) => void;
  onAsk: (q: string) => void;
  pending: boolean;
  draft: string;
  onDraftChange: (v: string) => void;
  showLogo: boolean;
} & IngestProps) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 pt-16 pb-12">
        <div
          className={`flex justify-center transition-all duration-300 ease-out ${
            showLogo
              ? "opacity-100 h-44 mb-8"
              : "opacity-0 h-0 mb-0 overflow-hidden"
          }`}
        >
          <Image
            src="/logo_kairoAI.png"
            alt="Kairo AI"
            width={320}
            height={176}
            priority
            className="h-44 w-auto object-contain"
          />
        </div>

        <AskBox
          plugins={plugins}
          selectedPlugin={selectedPlugin}
          onSelectPlugin={onSelectPlugin}
          value={draft}
          onChange={onDraftChange}
          onSubmit={onAsk}
          disabled={pending}
          autoFocus
          ingest={ingest}
          ingestPending={ingestPending}
          ingestError={ingestError}
          onIngestUrl={onIngestUrl}
          onIngestFile={onIngestFile}
          onClearIngest={onClearIngest}
        />

        {showLogo && (
          <div className="mt-8">
            <div className="flex items-center gap-2 mb-3 text-sm text-muted">
              <span>Try Kairo</span>
            </div>
            <div className="grid gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => onAsk(s)}
                  disabled={pending}
                  className="text-left text-sm text-accent px-4 py-3 rounded-lg border border-hair bg-panel hover:bg-panel2 disabled:opacity-50"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ThreadView({
  plugins,
  selectedPlugin,
  onSelectPlugin,
  messages,
  pending,
  onAsk,
  draft,
  onDraftChange,
  ingest,
  ingestPending,
  ingestError,
  onIngestUrl,
  onIngestFile,
  onClearIngest,
}: {
  plugins: PluginSummary[];
  selectedPlugin: string | null;
  onSelectPlugin: (name: string | null) => void;
  messages: Message[];
  pending: boolean;
  onAsk: (q: string) => void;
  draft: string;
  onDraftChange: (v: string) => void;
} & IngestProps) {
  return (
    <>
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8">
          <MessageList messages={messages} pending={pending} />
        </div>
      </div>
      <div className="border-t border-hair">
        <div className="max-w-3xl mx-auto px-6 py-4">
          <AskBox
            plugins={plugins}
            selectedPlugin={selectedPlugin}
            onSelectPlugin={onSelectPlugin}
            value={draft}
            onChange={onDraftChange}
            onSubmit={onAsk}
            disabled={pending}
            ingest={ingest}
            ingestPending={ingestPending}
            ingestError={ingestError}
            onIngestUrl={onIngestUrl}
            onIngestFile={onIngestFile}
            onClearIngest={onClearIngest}
          />
        </div>
      </div>
    </>
  );
}

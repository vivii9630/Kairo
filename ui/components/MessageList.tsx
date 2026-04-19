"use client";

import type { ChartSpec, Message } from "@/lib/types";
import { ChartStack } from "@/components/ChartBlock";

interface MessageListProps {
  messages: Message[];
  pending: boolean;
  charts?: ChartSpec[];
}

export function MessageList({ messages, pending, charts }: MessageListProps) {
  if (messages.length === 0 && !pending) return null;

  // Charts attach to the most recent assistant message in the thread.
  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  })();

  return (
    <div className="space-y-5">
      {messages.map((m, i) => (
        <div key={i}>
          <div className="text-xs uppercase tracking-wide text-muted mb-1.5">
            {m.role === "user" ? "You" : "Kairo"}
          </div>
          {m.role === "assistant" && i === lastAssistantIndex && charts && charts.length > 0 && (
            <ChartStack charts={charts} />
          )}
          <div className="text-sm text-accent whitespace-pre-wrap leading-relaxed">
            {m.content}
          </div>
        </div>
      ))}
      {pending && (
        <div>
          <div className="text-xs uppercase tracking-wide text-muted mb-1.5">
            Kairo
          </div>
          <div className="text-sm text-muted animate-pulse">thinking…</div>
        </div>
      )}
    </div>
  );
}

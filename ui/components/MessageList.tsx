"use client";

import type { Message } from "@/lib/types";

interface MessageListProps {
  messages: Message[];
  pending: boolean;
}

export function MessageList({ messages, pending }: MessageListProps) {
  if (messages.length === 0 && !pending) return null;
  return (
    <div className="space-y-5">
      {messages.map((m, i) => (
        <div key={i}>
          <div className="text-xs uppercase tracking-wide text-muted mb-1.5">
            {m.role === "user" ? "You" : "Kairo"}
          </div>
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

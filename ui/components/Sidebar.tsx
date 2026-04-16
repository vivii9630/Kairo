"use client";

import type { Thread } from "@/lib/types";

interface SidebarProps {
  threads: Thread[];
  activeThreadId: string | null;
  onNewThread: () => void;
  onSelectThread: (id: string) => void;
}

const NAV_ITEMS = [
  { label: "Connections", icon: "⌘" },
  { label: "Spaces", icon: "◫" },
  { label: "Customize", icon: "◈" },
  { label: "History", icon: "◴" },
];

export function Sidebar({
  threads,
  activeThreadId,
  onNewThread,
  onSelectThread,
}: SidebarProps) {
  return (
    <aside className="w-60 shrink-0 h-full border-r border-hair bg-surface flex flex-col">
      <div className="px-4 h-14 flex items-center justify-between">
        <span className="text-lg font-semibold tracking-tight">kairo</span>
        <button
          className="text-muted hover:text-accent"
          aria-label="Toggle sidebar"
        >
          ⟨
        </button>
      </div>

      <button
        onClick={onNewThread}
        className="mx-3 mb-3 flex items-center gap-2 px-3 h-9 rounded-md bg-panel hover:bg-panel2 border border-hair text-sm"
      >
        <span className="text-base">+</span>
        <span>New</span>
      </button>

      <nav className="px-2 space-y-0.5">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.label}
            className="w-full flex items-center gap-3 px-3 h-9 rounded-md text-sm text-muted hover:text-accent hover:bg-panel"
          >
            <span className="w-4 text-center">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="mt-6 flex-1 overflow-y-auto px-2">
        <div className="px-3 text-xs uppercase tracking-wide text-muted mb-2">
          Threads
        </div>
        {threads.length === 0 ? (
          <div className="px-3 text-sm text-muted">No recent threads</div>
        ) : (
          <ul className="space-y-0.5">
            {threads.map((t) => (
              <li key={t.id}>
                <button
                  onClick={() => onSelectThread(t.id)}
                  className={`w-full text-left px-3 py-2 rounded-md text-sm truncate ${
                    t.id === activeThreadId
                      ? "bg-panel text-accent"
                      : "text-muted hover:text-accent hover:bg-panel"
                  }`}
                  title={t.title}
                >
                  {t.title}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="p-3 border-t border-hair">
        <button className="w-full flex items-center gap-2 px-3 h-9 rounded-md text-sm text-muted hover:text-accent hover:bg-panel">
          <span className="w-5 h-5 rounded-full bg-hair" />
          <span>Sign In</span>
          <span className="ml-auto">›</span>
        </button>
      </div>
    </aside>
  );
}

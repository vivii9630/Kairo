"use client";

const TABS = ["Discover", "Connections", "Threads", "Spaces", "Graph"];

export function TopNav() {
  return (
    <div className="h-14 flex items-center justify-center gap-6 border-b border-hair px-6">
      {TABS.map((tab, i) => (
        <button
          key={tab}
          className={`text-sm ${
            i === 0 ? "text-accent" : "text-muted hover:text-accent"
          }`}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}

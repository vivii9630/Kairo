import type {
  AskRequest,
  AskResponse,
  IngestResponse,
  PluginSummary,
  Thread,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_KAIRO_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await safeText(res);
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return (await res.json()) as T;
}

async function safeText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch {
    return "";
  }
}

async function postMultipart<T>(path: string, formData: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await safeText(res);
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  getPlugins: () => request<PluginSummary[]>("/plugins"),
  listThreads: () => request<Thread[]>("/threads"),
  getThread: (id: string) => request<Thread>(`/threads/${id}`),
  ask: (body: AskRequest) =>
    request<AskResponse>("/ask", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  ingestUrl: (sourceUrl: string) =>
    request<IngestResponse>("/ingest", {
      method: "POST",
      body: JSON.stringify({ source_url: sourceUrl }),
    }),
  ingestFile: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return postMultipart<IngestResponse>("/ingest/upload", fd);
  },
};

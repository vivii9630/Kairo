// Mirrors the pydantic models in ai/src/kairo_ai/models.py.
// Keep in sync — drift here silently breaks the UI.

export interface PluginSummary {
  name: string;
  label: string;
  description: string;
  uri_example: string;
  icon: string | null;
  tags: string[];
  auth_required: boolean;
  auth_methods: string[];
  auth_scopes: string[];
  auth_instructions: string;
}

export interface Citation {
  source: string;
  title: string | null;
  score: number | null;
  metadata: Record<string, unknown>;
}

export type TraceKind = "plan" | "retrieve" | "synthesize" | "tool" | "note";

export interface TraceStep {
  step: number;
  kind: TraceKind;
  summary: string;
  snapshot_id: string | null;
  metadata: Record<string, unknown>;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface Thread {
  id: string;
  title: string;
  created_at: string;
  plugin: string | null;
  messages: Message[];
}

export interface AskRequest {
  query: string;
  plugin?: string | null;
  thread_id?: string | null;
}

export interface AskResponse {
  thread_id: string;
  answer: string;
  citations: Citation[];
  trace: TraceStep[];
  stubbed: boolean;
}

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
  graph_data?: LayeredGraphData | null;
  traversal_trace?: TraversalTrace | null;
}

// ---------------------------------------------------------------------------
// 3-Layer graph types (mirrors kairo-core models)
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  kind: string;
  label: string;
  attrs: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: string;
  attrs: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface InterLayerEdge {
  source: string;
  target: string;
  source_layer: LayerKind;
  target_layer: LayerKind;
  kind: string;
  attrs: Record<string, unknown>;
}

export type LayerKind = "document" | "semantic" | "detail";

export interface LayeredGraphData {
  document: GraphData;
  semantic: GraphData;
  detail: GraphData;
  inter_layer_edges: InterLayerEdge[];
}

export type TraversalAction =
  | "visit"
  | "expand"
  | "cross_layer"
  | "cluster"
  | "score";

export interface TraversalStep {
  agent_id: string;
  node_id: string;
  layer: LayerKind;
  action: TraversalAction;
  score: number | null;
  metadata: Record<string, unknown>;
}

export interface TraversalTrace {
  query: string;
  steps: TraversalStep[];
  visited_nodes: Record<LayerKind, string[]>;
  crossed_edges: InterLayerEdge[];
  clusters: Record<string, string[]>;
}

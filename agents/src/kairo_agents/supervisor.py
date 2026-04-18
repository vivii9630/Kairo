"""Deterministic supervisor for the Kairo 3-layer engine.

The supervisor is an **algorithm**, not a prompt.  Given a query, a
3-layer graph, and a set of available agents (user-chosen models), it:

1. Analyzes the graph topology (cluster count, layer sizes, cross-layer
   density) to decide how many agents are needed and what roles to assign.
2. Produces a :class:`TaskPlan` — a deterministic, reproducible work
   allocation that maps agents to graph regions and roles.
3. The plan is executed by the :class:`EngineRunner` which delegates
   each task to the assigned agent.

**Determinism guarantee**: given the same graph state and query, the
supervisor produces the identical plan every time.  The LLM creativity
lives in the agents' ``think_fn`` calls, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kairo_core import TraversalTrace


# ---------------------------------------------------------------------------
# Task plan types
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    """A single work unit assigned to an agent."""

    agent_name: str
    role: str
    instruction: str
    layer_scope: List[str]
    cluster_ids: List[str] = field(default_factory=list)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Complete work allocation produced by the supervisor.

    Deterministic: same inputs → same plan.
    """

    query: str
    tasks: List[AgentTask] = field(default_factory=list)
    total_agents: int = 0
    scaling_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph topology analysis
# ---------------------------------------------------------------------------

@dataclass
class TopologyStats:
    """Lightweight summary of a layered graph's shape."""

    document_nodes: int = 0
    semantic_nodes: int = 0
    detail_nodes: int = 0
    inter_layer_edges: int = 0
    cluster_count: int = 0
    cluster_sizes: List[int] = field(default_factory=list)
    total_nodes: int = 0

    @property
    def density(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return self.inter_layer_edges / max(self.total_nodes, 1)


def analyze_topology(
    visited_nodes: Dict[str, List[str]],
    clusters: Dict[str, List[str]],
    inter_layer_edge_count: int,
) -> TopologyStats:
    """Analyze the visited subgraph topology for supervisor routing."""
    doc_count = len(visited_nodes.get("document", []))
    sem_count = len(visited_nodes.get("semantic", []))
    det_count = len(visited_nodes.get("detail", []))
    total = doc_count + sem_count + det_count
    cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True)

    return TopologyStats(
        document_nodes=doc_count,
        semantic_nodes=sem_count,
        detail_nodes=det_count,
        inter_layer_edges=inter_layer_edge_count,
        cluster_count=len(clusters),
        cluster_sizes=cluster_sizes,
        total_nodes=total,
    )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

# Role definitions for dynamic assignment
ROLE_EXPLORER = "explorer"
ROLE_ANALYZER = "analyzer"
ROLE_SYNTHESIZER = "synthesizer"
ROLE_VALIDATOR = "validator"
ROLE_WEB_RESEARCHER = "web_researcher"

_ROLE_INSTRUCTIONS = {
    ROLE_EXPLORER: (
        "You are an explorer agent. Your job is to examine the document-level "
        "structure and identify the key sections, themes, and high-level "
        "relationships relevant to the query. Summarize what you find."
    ),
    ROLE_ANALYZER: (
        "You are an analyzer agent. Your job is to examine the semantic "
        "concepts and detail-level data points in your assigned graph region. "
        "Identify the most relevant evidence, note connections between "
        "concepts, and report your findings with supporting data."
    ),
    ROLE_SYNTHESIZER: (
        "You are a synthesizer agent. You receive findings from explorer and "
        "analyzer agents. Your job is to combine them into a coherent, "
        "evidence-backed answer to the user's query. Cite specific data "
        "points and explain the reasoning chain."
    ),
    ROLE_VALIDATOR: (
        "You are a validator agent. Your job is to check the synthesized "
        "answer against the raw evidence. Flag any unsupported claims, "
        "missing context, or contradictions. Suggest corrections if needed."
    ),
    ROLE_WEB_RESEARCHER: (
        "You are a web research agent. Web search results have been "
        "attached to the detail layer under the 'web' cluster. Your job "
        "is to read those external sources, extract what they contribute "
        "beyond the local documents — new suggestions, alternatives, "
        "recent trends, or corroborating evidence — and produce a short "
        "report. Always include each source URL so the synthesizer can "
        "cite it."
    ),
}


class Supervisor:
    """Deterministic supervisor that allocates agents to graph regions.

    Parameters
    ----------
    available_agents : int
        Number of agent slots available (default 3 = user's chosen models).
    min_agents : int
        Minimum agents to use regardless of graph size.
    max_agents : int
        Hard ceiling on agent count.
    scale_threshold : int
        If total visited nodes exceed this, request additional agents.
    """

    def __init__(
        self,
        *,
        available_agents: int = 3,
        min_agents: int = 2,
        max_agents: int = 8,
        scale_threshold: int = 40,
    ) -> None:
        self.available_agents = available_agents
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.scale_threshold = scale_threshold

    def plan(
        self,
        query: str,
        trace: TraversalTrace,
    ) -> TaskPlan:
        """Produce a deterministic task plan from traversal results.

        The plan allocates agents to roles based on graph topology:

        - **2 agents**: explorer (L1+L2) + synthesizer (L3 + final answer)
        - **3 agents**: explorer (L1) + analyzer (L2+L3) + synthesizer
        - **4+ agents**: explorer + N analyzers (one per cluster) + synthesizer
          + optional validator if graph is dense enough

        When the traversal contains a ``web_hits`` cluster (populated by
        :class:`kairo_rag.WebEnricher`), a web-researcher task is spliced
        in between the analyzers and the synthesizer.

        Scaling is deterministic: it's a function of cluster count, node
        count, and inter-layer density.
        """
        stats = analyze_topology(
            trace.visited_nodes,
            trace.clusters,
            len(trace.crossed_edges),
        )

        has_web = self._has_web_hits(trace)

        # Determine how many agents to use
        needed = self._compute_agent_count(stats)
        total = min(max(needed, self.min_agents), self.max_agents)
        scaling_reason = self._scaling_reason(stats, needed, total)

        tasks: List[AgentTask] = []

        if total <= 2:
            tasks = self._plan_two_agents(query, trace, stats)
        elif total == 3:
            tasks = self._plan_three_agents(query, trace, stats)
        else:
            tasks = self._plan_scaled(query, trace, stats, total)

        if has_web:
            tasks = self._inject_web_researcher(tasks, trace)
            scaling_reason = (
                f"{scaling_reason}; +web_researcher (web_hits cluster)"
            )

        return TaskPlan(
            query=query,
            tasks=tasks,
            total_agents=len(tasks),
            scaling_reason=scaling_reason,
            metadata={"topology": {
                "document_nodes": stats.document_nodes,
                "semantic_nodes": stats.semantic_nodes,
                "detail_nodes": stats.detail_nodes,
                "clusters": stats.cluster_count,
                "density": round(stats.density, 3),
                "has_web_hits": has_web,
            }},
        )

    # -- web enrichment detection ------------------------------------------

    @staticmethod
    def _has_web_hits(trace: TraversalTrace) -> bool:
        """True if the enricher attached a ``web_hits`` cluster."""
        return bool(trace.clusters.get("web_hits"))

    def _inject_web_researcher(
        self, tasks: List[AgentTask], trace: TraversalTrace,
    ) -> List[AgentTask]:
        """Splice a web_researcher task before the synthesizer.

        Re-priorities adjacent tasks so the researcher runs after
        analyzers (its findings should feed the synthesizer).
        """
        web_ids = list(trace.clusters.get("web_hits", []))
        # Find the synthesizer's current priority to place the web task
        # just before it.  Fall back to appending if no synthesizer found.
        synth_priority = None
        for t in tasks:
            if t.role == ROLE_SYNTHESIZER:
                synth_priority = t.priority
                break

        # Bump the synthesizer (and anything at that priority or later) up
        # by one so the researcher slots between analyzer and synthesis.
        if synth_priority is not None:
            for t in tasks:
                if t.priority >= synth_priority:
                    t.priority = t.priority + 1
            web_priority = synth_priority
        else:
            web_priority = max((t.priority for t in tasks), default=0) + 1

        web_task = AgentTask(
            agent_name=f"agent_{len(tasks)}",
            role=ROLE_WEB_RESEARCHER,
            instruction=_ROLE_INSTRUCTIONS[ROLE_WEB_RESEARCHER],
            layer_scope=["detail"],
            cluster_ids=["web_hits"],
            priority=web_priority,
            metadata={"web_hit_ids": web_ids},
        )
        tasks.append(web_task)
        return tasks

    # -- agent count heuristics ---------------------------------------------

    def _compute_agent_count(self, stats: TopologyStats) -> int:
        """Deterministic function: topology → agent count."""
        base = self.available_agents

        # Scale up for large graphs
        if stats.total_nodes > self.scale_threshold:
            extra = (stats.total_nodes - self.scale_threshold) // 20
            base += extra

        # Scale up for multiple clusters
        if stats.cluster_count > 2:
            base = max(base, stats.cluster_count + 1)  # one per cluster + synthesizer

        # Scale up for high density (lots of cross-layer connections)
        if stats.density > 2.0:
            base += 1

        return base

    def _scaling_reason(
        self, stats: TopologyStats, needed: int, actual: int
    ) -> str:
        parts = []
        if stats.total_nodes > self.scale_threshold:
            parts.append(f"{stats.total_nodes} nodes (>{self.scale_threshold} threshold)")
        if stats.cluster_count > 2:
            parts.append(f"{stats.cluster_count} clusters")
        if stats.density > 2.0:
            parts.append(f"density={stats.density:.2f}")
        if needed > actual:
            parts.append(f"capped from {needed} to {actual}")
        return "; ".join(parts) if parts else "standard allocation"

    # -- plan templates -----------------------------------------------------

    def _plan_two_agents(
        self, query: str, trace: TraversalTrace, stats: TopologyStats
    ) -> List[AgentTask]:
        cluster_ids = sorted(trace.clusters.keys())
        return [
            AgentTask(
                agent_name="agent_0",
                role=ROLE_EXPLORER,
                instruction=_ROLE_INSTRUCTIONS[ROLE_EXPLORER],
                layer_scope=["document", "semantic"],
                cluster_ids=cluster_ids,
                priority=0,
            ),
            AgentTask(
                agent_name="agent_1",
                role=ROLE_SYNTHESIZER,
                instruction=_ROLE_INSTRUCTIONS[ROLE_SYNTHESIZER],
                layer_scope=["detail"],
                cluster_ids=cluster_ids,
                priority=1,
            ),
        ]

    def _plan_three_agents(
        self, query: str, trace: TraversalTrace, stats: TopologyStats
    ) -> List[AgentTask]:
        cluster_ids = sorted(trace.clusters.keys())
        return [
            AgentTask(
                agent_name="agent_0",
                role=ROLE_EXPLORER,
                instruction=_ROLE_INSTRUCTIONS[ROLE_EXPLORER],
                layer_scope=["document"],
                cluster_ids=cluster_ids,
                priority=0,
            ),
            AgentTask(
                agent_name="agent_1",
                role=ROLE_ANALYZER,
                instruction=_ROLE_INSTRUCTIONS[ROLE_ANALYZER],
                layer_scope=["semantic", "detail"],
                cluster_ids=cluster_ids,
                priority=0,
            ),
            AgentTask(
                agent_name="agent_2",
                role=ROLE_SYNTHESIZER,
                instruction=_ROLE_INSTRUCTIONS[ROLE_SYNTHESIZER],
                layer_scope=["document", "semantic", "detail"],
                cluster_ids=cluster_ids,
                priority=1,
            ),
        ]

    def _plan_scaled(
        self, query: str, trace: TraversalTrace, stats: TopologyStats,
        total: int,
    ) -> List[AgentTask]:
        cluster_ids = sorted(trace.clusters.keys())
        tasks: List[AgentTask] = []

        # Agent 0: explorer (always)
        tasks.append(AgentTask(
            agent_name="agent_0",
            role=ROLE_EXPLORER,
            instruction=_ROLE_INSTRUCTIONS[ROLE_EXPLORER],
            layer_scope=["document"],
            cluster_ids=cluster_ids,
            priority=0,
        ))

        # Middle agents: one analyzer per cluster (or split evenly)
        analyzer_count = total - 2  # minus explorer and synthesizer
        if stats.density > 2.0 and analyzer_count > 1:
            # Reserve one for validator
            analyzer_count -= 1

        for i in range(analyzer_count):
            # Assign clusters round-robin to analyzers
            assigned_clusters = [
                cid for j, cid in enumerate(cluster_ids)
                if j % analyzer_count == i
            ]
            tasks.append(AgentTask(
                agent_name=f"agent_{i + 1}",
                role=ROLE_ANALYZER,
                instruction=_ROLE_INSTRUCTIONS[ROLE_ANALYZER],
                layer_scope=["semantic", "detail"],
                cluster_ids=assigned_clusters,
                priority=0,
                metadata={"cluster_assignment": assigned_clusters},
            ))

        # Optional validator for dense graphs
        if stats.density > 2.0 and total >= 4:
            tasks.append(AgentTask(
                agent_name=f"agent_{total - 2}",
                role=ROLE_VALIDATOR,
                instruction=_ROLE_INSTRUCTIONS[ROLE_VALIDATOR],
                layer_scope=["document", "semantic", "detail"],
                cluster_ids=cluster_ids,
                priority=1,
            ))

        # Last agent: synthesizer (always)
        tasks.append(AgentTask(
            agent_name=f"agent_{total - 1}",
            role=ROLE_SYNTHESIZER,
            instruction=_ROLE_INSTRUCTIONS[ROLE_SYNTHESIZER],
            layer_scope=["document", "semantic", "detail"],
            cluster_ids=cluster_ids,
            priority=2,
        ))

        return tasks

"""EngineRunner — wires Supervisor + Agents + TemporalRAGEngine.

This is the top-level entry point for a full Kairo query:

1. ``TemporalRAGEngine.query()`` builds the 3-layer graph, does KNN +
   graph walk, collects evidence, and snapshots the state.
2. ``Supervisor.plan()`` analyzes the traversal trace and produces a
   deterministic task plan.
3. ``EngineRunner`` instantiates agents (using user-chosen model configs)
   and executes the plan: each agent receives its assigned evidence and
   produces findings; the synthesizer merges everything into a final answer.

The runner itself is deterministic in its orchestration (same plan →
same execution order).  Non-determinism lives only inside each agent's
``think_fn`` (the LLM call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from kairo_core import Document, Evidence, LayeredGraphData, TraversalTrace
from kairo_agents import Agent, AgentTask, MessageBus, Supervisor, TaskPlan
from kairo_agents.agent import ThinkFn, _default_think
from kairo_embeddings.provider import EmbeddingProvider

from .temporal_rag import RAGResult, TemporalRAGEngine


@dataclass
class AgentFinding:
    """Output from a single agent's work on its assigned task."""

    agent_name: str
    role: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    """Full output of an EngineRunner query."""

    answer: str
    findings: List[AgentFinding]
    plan: TaskPlan
    rag_result: RAGResult
    graph_data: Optional[LayeredGraphData] = None


class EngineRunner:
    """Top-level orchestrator: RAG engine + supervisor + agents.

    Parameters
    ----------
    embedding_provider : EmbeddingProvider
        For graph node embeddings + query embedding.
    think_fn : ThinkFn
        Default LLM callable for agents.  Override per-agent via
        ``agent_think_fns``.
    agent_think_fns : dict
        Optional per-agent overrides: ``{"agent_0": my_think_fn, ...}``.
    available_agents : int
        Number of agent slots (default 3).
    top_k : int
        KNN neighbors per layer.
    walk_depth : int
        BFS expansion depth.
    verbose : bool
        Print agent reasoning to stdout.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        think_fn: Optional[ThinkFn] = None,
        *,
        agent_think_fns: Optional[Dict[str, ThinkFn]] = None,
        available_agents: int = 3,
        top_k: int = 5,
        walk_depth: int = 2,
        verbose: bool = False,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.default_think_fn = think_fn or _default_think
        self.agent_think_fns = agent_think_fns or {}
        self.verbose = verbose

        self.engine = TemporalRAGEngine(
            embedding_provider, top_k=top_k, walk_depth=walk_depth
        )
        self.supervisor = Supervisor(available_agents=available_agents)

    def run(
        self,
        query: str,
        documents: List[Document],
        *,
        session_id: str = "default",
    ) -> EngineResult:
        """Execute a full query through the Kairo engine.

        Steps:
        1. TemporalRAGEngine builds graph, retrieves evidence, snapshots.
        2. Supervisor produces a deterministic task plan.
        3. Agents execute their assigned tasks in priority order.
        4. Synthesizer produces the final answer.
        """
        # 1. RAG engine: graph + retrieval + snapshot
        rag_result = self.engine.query(
            query, documents, session_id=session_id
        )

        # 2. Supervisor: deterministic planning
        plan = self.supervisor.plan(query, rag_result.trace)

        # 3. Create agents and execute plan
        bus = MessageBus(verbose=self.verbose)
        findings = self._execute_plan(plan, rag_result, bus)

        # 4. Extract final answer (synthesizer's output, or fallback)
        answer = self._extract_answer(findings, rag_result)

        return EngineResult(
            answer=answer,
            findings=findings,
            plan=plan,
            rag_result=rag_result,
            graph_data=rag_result.graph_data,
        )

    def _execute_plan(
        self,
        plan: TaskPlan,
        rag_result: RAGResult,
        bus: MessageBus,
    ) -> List[AgentFinding]:
        """Instantiate agents and run tasks in priority order."""
        # Sort tasks by priority (lower = earlier)
        sorted_tasks = sorted(plan.tasks, key=lambda t: t.priority)

        findings: List[AgentFinding] = []
        prior_findings_text = ""

        for task in sorted_tasks:
            # Get the think_fn for this agent
            think_fn = self.agent_think_fns.get(
                task.agent_name, self.default_think_fn
            )

            agent = Agent(
                name=task.agent_name,
                role=task.role,
                instruction=task.instruction,
                verbose=self.verbose,
                think_fn=think_fn,
                bus=bus,
            )

            # Build the agent's input
            agent_input = self._build_agent_input(
                task, rag_result, prior_findings_text, plan.query
            )

            # Run the agent
            output = agent.think(agent_input)

            finding = AgentFinding(
                agent_name=task.agent_name,
                role=task.role,
                output=output,
                metadata={"layer_scope": task.layer_scope},
            )
            findings.append(finding)

            # Accumulate findings for downstream agents
            prior_findings_text += f"\n\n--- {task.role} ({task.agent_name}) ---\n{output}"

        return findings

    def _build_agent_input(
        self,
        task: AgentTask,
        rag_result: RAGResult,
        prior_findings: str,
        query: str,
    ) -> str:
        """Assemble the input text for an agent based on its role and scope."""
        parts = [f"Query: {query}\n"]

        # Filter evidence by layer scope
        scoped_evidence = [
            ev for ev in rag_result.evidences
            if ev.metadata.get("layer") in task.layer_scope
        ]

        if scoped_evidence:
            parts.append("Relevant evidence:")
            for ev in scoped_evidence[:15]:  # cap to avoid token overflow
                parts.append(
                    f"  [{ev.document_id}] (score={ev.score:.3f}): {ev.text}"
                )

        # Cluster info
        if task.cluster_ids:
            cluster_info = []
            for cid in task.cluster_ids:
                nodes = rag_result.clusters.get(cid, [])
                cluster_info.append(f"  {cid}: {len(nodes)} nodes")
            if cluster_info:
                parts.append("\nAssigned clusters:")
                parts.extend(cluster_info)

        # Synthesizer and validator get prior findings
        if task.role in ("synthesizer", "validator") and prior_findings:
            parts.append(f"\nFindings from other agents:{prior_findings}")

        return "\n".join(parts)

    def _extract_answer(
        self,
        findings: List[AgentFinding],
        rag_result: RAGResult,
    ) -> str:
        """Get the final answer from the synthesizer, or fall back."""
        # Look for synthesizer output
        for finding in reversed(findings):
            if finding.role == "synthesizer":
                return finding.output

        # Fallback: concatenate all findings
        if findings:
            return "\n\n".join(f.output for f in findings)

        # Ultimate fallback: raw evidence context
        return rag_result.answer_context

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

from kairo_core import Message

if TYPE_CHECKING:
    from .bus import MessageBus

ThinkFn = Callable[[str, str, str], str]


def _default_think(role: str, instruction: str, user_input: str) -> str:
    return (
        f"[{role}] (no think_fn configured) "
        f"received input with instruction '{instruction[:60]}...': {user_input}"
    )


class Agent:
    """An agent defined by its role, instruction, and optional thinking function.

    Parameters
    ----------
    name:
        Unique identifier used for message routing and logs.
    role:
        What the agent is, e.g. "research analyst" or "planner".
    instruction:
        How the agent should behave — the system prompt / guidance.
    verbose:
        When True, every reasoning step, subagent delegation, and message
        send/receive is printed so callers can observe how the agent thinks.
    think_fn:
        Callable `(role, instruction, user_input) -> str` that produces the
        agent's response. Swap in an LLM call here. Defaults to a stub.
    bus:
        Optional MessageBus for inter-agent communication. If provided,
        the agent auto-subscribes under `name`.
    """

    def __init__(
        self,
        name: str,
        role: str,
        instruction: str,
        verbose: bool = False,
        think_fn: Optional[ThinkFn] = None,
        bus: Optional["MessageBus"] = None,
    ) -> None:
        self.name = name
        self.role = role
        self.instruction = instruction
        self.verbose = verbose
        self._think_fn: ThinkFn = think_fn or _default_think
        self.bus = bus
        self.subagents: List["Agent"] = []
        self.inbox: List[Message] = []

        if bus is not None:
            bus.subscribe(name, self._on_message)

    def _log(self, label: str, text: str) -> None:
        if self.verbose:
            print(f"[{self.name}:{label}] {text}")

    def think(self, user_input: str) -> str:
        """Run one reasoning step and return the agent's response."""
        self._log("thinking", f"role={self.role!r} input={user_input!r}")
        output = self._think_fn(self.role, self.instruction, user_input)
        self._log("thought", output)
        return output

    def add_subagent(self, agent: "Agent") -> "Agent":
        """Attach a subagent. Subagents inherit the parent's bus if not set."""
        if agent.bus is None and self.bus is not None:
            agent.bus = self.bus
            self.bus.subscribe(agent.name, agent._on_message)
        self.subagents.append(agent)
        self._log("subagent", f"attached '{agent.name}' ({agent.role})")
        return agent

    def delegate(self, subagent_name: str, task: str) -> str:
        """Hand a task to a named subagent and return its response."""
        for sub in self.subagents:
            if sub.name == subagent_name:
                self._log("delegate", f"-> {subagent_name}: {task!r}")
                result = sub.think(task)
                self._log("delegate-result", f"<- {subagent_name}: {result!r}")
                return result
        raise ValueError(
            f"subagent '{subagent_name}' not found under '{self.name}'. "
            f"Known subagents: {[s.name for s in self.subagents]}"
        )

    def send(self, recipient: str, content: str, **metadata) -> Message:
        """Publish a message to another agent via the shared MessageBus."""
        if self.bus is None:
            raise RuntimeError(
                f"agent '{self.name}' has no MessageBus attached; "
                f"pass bus=... when constructing it."
            )
        msg = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            metadata=dict(metadata),
        )
        self._log("send", f"-> {recipient}: {content!r}")
        self.bus.publish(msg)
        return msg

    def _on_message(self, message: Message) -> None:
        self._log("recv", f"<- {message.sender}: {message.content!r}")
        self.inbox.append(message)

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, role={self.role!r}, "
            f"subagents={[s.name for s in self.subagents]})"
        )

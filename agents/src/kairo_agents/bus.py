from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional

from kairo_core import Message

Handler = Callable[[Message], None]


class MessageBus:
    """Shared channel that routes messages between agents.

    Agents subscribe by name; publish() delivers a message to every handler
    registered under the recipient name. History is retained so conversations
    can be inspected after the fact.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._subscribers: Dict[str, List[Handler]] = defaultdict(list)
        self._history: List[Message] = []
        self.verbose = verbose

    def subscribe(self, agent_name: str, handler: Handler) -> None:
        self._subscribers[agent_name].append(handler)
        if self.verbose:
            print(f"[bus] subscribed '{agent_name}'")

    def publish(self, message: Message) -> None:
        self._history.append(message)
        if self.verbose:
            print(f"[bus] {message.pretty()}")
        for handler in self._subscribers.get(message.recipient, []):
            handler(message)

    def history(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
    ) -> List[Message]:
        msgs = self._history
        if sender is not None:
            msgs = [m for m in msgs if m.sender == sender]
        if recipient is not None:
            msgs = [m for m in msgs if m.recipient == recipient]
        return list(msgs)

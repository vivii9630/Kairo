from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class Message:
    """Generic directed message used for inter-agent and inter-node communication."""

    sender: str
    recipient: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def pretty(self) -> str:
        return f"{self.sender} -> {self.recipient}: {self.content}"

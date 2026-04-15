from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Credentials:
    """Authentication bundle for provider backends.

    Designed as the seam for the future ``kairo-auth`` package. Any
    provider that needs credentials accepts one of these via
    ``get_provider(name, credentials=...)``. For now, construct one by
    hand or from environment variables with :meth:`from_env`. When
    ``kairo-auth`` ships, it will hand back a ``Credentials`` object
    with the same shape and no existing provider code needs to change.
    """

    api_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, *, api_key_var: str, **extra_vars: str) -> "Credentials":
        """Resolve credentials from environment variables.

        Example:
            Credentials.from_env(api_key_var="OPENAI_API_KEY", org="OPENAI_ORG")
        """
        return cls(
            api_key=os.environ.get(api_key_var),
            extra={k: os.environ.get(v, "") for k, v in extra_vars.items()},
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from kairo_core import Document


class BaseConnector(ABC):
    """Base interface for all Kairo connectors.

    A connector pulls data from an external source and yields `Document`s.
    Subclasses live in per-source modules (e.g. `s3.py`, `notion.py`) so their
    heavy dependencies stay optional.
    """

    name: str = "base"

    @abstractmethod
    def fetch(self) -> Iterable[Document]:
        """Yield documents from the external source."""
        raise NotImplementedError

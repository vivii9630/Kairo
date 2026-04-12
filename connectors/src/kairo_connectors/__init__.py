"""Kairo connectors: adapters for external data sources.

Each connector is an optional extra — install only what you need:

    pip install -e "./connectors[s3]"
    pip install -e "./connectors[notion]"
    pip install -e "./connectors[web]"

Connectors produce `kairo_core.Document` instances that can be fed into any
retrieval pipeline.
"""

from .base import BaseConnector

__all__ = ["BaseConnector"]

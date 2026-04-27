"""Kairo edge: ARM/mobile-minimal build.

The public Python API prefers the Rust/PyO3 backend when ``kairo_edge_py``
is installed, then falls back to the pure-Python store. No pandas, pypdf,
httpx, soundfile, or heavyweight desktop dependencies are required.
"""

from .store import EdgeStore

__all__ = ["EdgeStore"]

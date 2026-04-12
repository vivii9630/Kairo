"""Kairo edge: ARM/mobile-minimal build.

Pure-Python only. No pandas, pypdf, httpx, soundfile, or anything with binary
wheels that complicates mobile/ARM deployment. The goal is that you can
`pip install kairo-edge` on a Raspberry Pi, a Termux shell, or a BeeWare
iOS/Android app and get a working retrieval + agent loop.
"""

from .store import EdgeStore

__all__ = ["EdgeStore"]

"""Kairo edge emulator.

Runs Kairo scenarios against a named :class:`DeviceProfile`,
enforcing per-profile memory / wall-clock budgets and reporting
structured pass/fail results. Designed to gate every PR that touches
``edge/`` — a change that makes ``pi-zero`` go over its 256 MB budget
fails here, not on real hardware.

See ``edge/docs/EDGE_ARCHITECTURE.md`` §7 for the full test strategy.
"""

from .profile import DeviceProfile, PROFILES
from .runner import EmulatorResult, run_scenario

__all__ = [
    "DeviceProfile",
    "PROFILES",
    "EmulatorResult",
    "run_scenario",
]

"""Device profiles for the edge emulator.

Each :class:`DeviceProfile` captures a real-world target in enough
detail for the emulator to enforce a budget. Numbers match the
architecture doc §4 and §8.

Profiles are intentionally a small, named set — adding one is a
deliberate act, not something a scenario can do ad hoc. This keeps
CI budgets comparable across PRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional


@dataclass(frozen=True)
class DeviceProfile:
    """A named edge target with hard budgets the emulator enforces.

    ``peak_ram_mb`` — the emulator fails a scenario whose peak RSS
    during execution exceeds this.

    ``ingest_ms`` / ``retrieve_ms`` / ``generate_ms`` — wall-clock
    budgets per phase. A missing value (``None``) means "no budget
    enforced" (e.g. ``generate_ms=None`` on extractive-only profiles).

    ``cpu_cores`` / ``cpu_ghz`` — reference hardware; emulator does
    not simulate these, but ``select_provider`` reads them to gate
    providers whose capability manifest requires more compute than
    the profile advertises.

    ``allowed_imports`` — whitelist of top-level modules the scenario
    is allowed to import. An import not in this set fails the scenario
    — this is how we enforce "no pandas on pi-zero" statically rather
    than waiting for OOM. ``None`` means no restriction.

    ``has_gpu`` / ``has_webgpu`` — capability gates for provider
    selection. The emulator does not actually disable GPU; it fails
    any provider that declares ``needs_gpu=True`` on a profile with
    ``has_gpu=False``.
    """

    name: str
    peak_ram_mb: int
    cpu_cores: int
    cpu_ghz: float
    ingest_ms: Optional[int]
    retrieve_ms: Optional[int]
    generate_ms: Optional[int]
    has_gpu: bool = False
    has_webgpu: bool = False
    allowed_imports: Optional[FrozenSet[str]] = None

    def budget(self, phase: str) -> Optional[int]:
        """Return the wall-clock budget for *phase* in milliseconds."""
        return {
            "ingest": self.ingest_ms,
            "retrieve": self.retrieve_ms,
            "generate": self.generate_ms,
        }.get(phase)


# Core edge-safe module allowlist used by the smallest profiles.
_EDGE_SAFE_IMPORTS: FrozenSet[str] = frozenset({
    "kairo_core",
    "kairo_edge",
    "json",
    "math",
    "re",
    "typing",
    "collections",
    "dataclasses",
    "pathlib",
    "os",
    "sys",
    "time",
    "gc",
    "pydantic",
    "pydantic_core",
    "annotated_types",
    "typing_extensions",
})


# Profile catalog — stable contracts, edit here deliberately.
PROFILES: Dict[str, DeviceProfile] = {
    "pi-zero": DeviceProfile(
        name="pi-zero",
        peak_ram_mb=256,
        cpu_cores=1,
        cpu_ghz=1.0,
        ingest_ms=20_000,
        retrieve_ms=500,
        generate_ms=None,  # extractive only
        allowed_imports=_EDGE_SAFE_IMPORTS,
    ),
    "pi-4": DeviceProfile(
        name="pi-4",
        peak_ram_mb=1024,
        cpu_cores=4,
        cpu_ghz=1.5,
        ingest_ms=10_000,
        retrieve_ms=200,
        generate_ms=15_000,
        allowed_imports=_EDGE_SAFE_IMPORTS | {"numpy", "onnxruntime", "llama_cpp"},
    ),
    "pi-5": DeviceProfile(
        name="pi-5",
        peak_ram_mb=2048,
        cpu_cores=4,
        cpu_ghz=2.4,
        ingest_ms=5_000,
        retrieve_ms=100,
        generate_ms=10_000,
        allowed_imports=_EDGE_SAFE_IMPORTS | {"numpy", "onnxruntime", "llama_cpp"},
    ),
    "phone-mid": DeviceProfile(
        name="phone-mid",
        peak_ram_mb=1024,
        cpu_cores=4,
        cpu_ghz=2.0,
        ingest_ms=8_000,
        retrieve_ms=150,
        generate_ms=12_000,
        allowed_imports=_EDGE_SAFE_IMPORTS | {"numpy", "onnxruntime"},
    ),
    "browser-lite": DeviceProfile(
        name="browser-lite",
        peak_ram_mb=400,
        cpu_cores=1,
        cpu_ghz=0.0,  # n/a in browser
        ingest_ms=15_000,
        retrieve_ms=300,
        generate_ms=None,
        allowed_imports=_EDGE_SAFE_IMPORTS,
    ),
    "browser-gpu": DeviceProfile(
        name="browser-gpu",
        peak_ram_mb=1024,
        cpu_cores=1,
        cpu_ghz=0.0,
        ingest_ms=12_000,
        retrieve_ms=200,
        generate_ms=8_000,
        has_webgpu=True,
        allowed_imports=_EDGE_SAFE_IMPORTS | {"numpy"},
    ),
}


def get_profile(name: str) -> DeviceProfile:
    """Look up a profile by name. Raises ValueError on unknown names."""
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile: {name!r}. "
            f"Known: {sorted(PROFILES)}."
        )
    return PROFILES[name]

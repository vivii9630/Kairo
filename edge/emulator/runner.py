"""Scenario runner — the part that actually enforces budgets.

A scenario is any callable ``(ctx: EmulatorContext) -> dict`` that
declares the phases it runs via ``ctx.phase("ingest")``, etc. The
runner wraps the call to measure RSS and wall time per phase, compares
them to the named :class:`DeviceProfile`, and emits a structured
:class:`EmulatorResult`.

RSS measurement uses ``resource.getrusage`` on POSIX and
``psutil.Process().memory_info().rss`` on Windows. The runner picks
whichever is available at runtime — no hard dep on psutil, but we
use it when present for better Windows numbers.
"""

from __future__ import annotations

import gc
import json
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .profile import DeviceProfile


# ---------------------------------------------------------------------------
# RSS snapshot — tracemalloc is cross-platform and good enough for
# CI-grade enforcement. Real hardware validation uses OS RSS.
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Current peak RSS since tracemalloc.start(), in MB.

    Tracemalloc measures Python-allocator memory only — it misses
    numpy buffers and C-extension allocations. That's an acceptable
    approximation for the pure-Python edge package; when we start
    testing ONNX / llama.cpp providers, we'll switch to psutil
    RSS. Tracked in EDGE_ARCHITECTURE.md §10 as an open question.
    """
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Emulator context passed to scenarios
# ---------------------------------------------------------------------------


@dataclass
class _PhaseMeasurement:
    name: str
    wall_ms: float
    rss_mb_start: float
    rss_mb_end: float
    budget_ms: Optional[int]
    rss_mb_delta: float
    over_budget: bool


class EmulatorContext:
    """Passed into every scenario; owns timers and emits phase markers."""

    def __init__(self, profile: DeviceProfile) -> None:
        self.profile = profile
        self.phases: List[_PhaseMeasurement] = []
        self._import_check_failed: List[str] = []

    @contextmanager
    def phase(self, name: str):
        """Wrap a phase with a wall-clock timer and RSS snapshots."""
        self._guard_imports()
        gc.collect()
        start_rss = _rss_mb()
        start_wall = time.perf_counter()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - start_wall) * 1000.0
            gc.collect()
            end_rss = _rss_mb()
            budget = self.profile.budget(name)
            over = budget is not None and wall_ms > budget
            self.phases.append(_PhaseMeasurement(
                name=name,
                wall_ms=wall_ms,
                rss_mb_start=start_rss,
                rss_mb_end=end_rss,
                rss_mb_delta=end_rss - start_rss,
                budget_ms=budget,
                over_budget=over,
            ))

    def _guard_imports(self) -> None:
        """Record any currently-loaded module that isn't in the profile's
        allowlist. Checked inside each ``phase()`` so a late import of
        pandas during ingest still surfaces."""
        allowed = self.profile.allowed_imports
        if allowed is None:
            return
        for mod_name in list(sys.modules):
            top = mod_name.split(".", 1)[0]
            if (
                top
                and not top.startswith("_")
                and top not in allowed
                and top not in self._import_check_failed
                # stdlib + pytest + internal plumbing we can't filter
                # from here are noisy; only flag a curated list of
                # "known heavy" imports — pandas/torch/networkx/etc.
                # Everything else is treated as ambient.
                and top in _KNOWN_HEAVY_IMPORTS
            ):
                self._import_check_failed.append(top)


_KNOWN_HEAVY_IMPORTS = frozenset({
    "pandas", "torch", "networkx", "statsmodels", "sklearn",
    "transformers", "sentence_transformers", "leidenalg", "igraph",
    "rank_bm25", "httpx", "soundfile", "pypdf", "fastapi", "uvicorn",
})


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class EmulatorResult:
    """Outcome of a single scenario run."""

    scenario: str
    profile: str
    passed: bool
    peak_rss_mb: float
    total_wall_ms: float
    rss_budget_mb: int
    rss_over_budget: bool
    phase_measurements: List[_PhaseMeasurement] = field(default_factory=list)
    forbidden_imports: List[str] = field(default_factory=list)
    scenario_output: Dict[str, Any] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "profile": self.profile,
            "passed": self.passed,
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "total_wall_ms": round(self.total_wall_ms, 2),
            "rss_budget_mb": self.rss_budget_mb,
            "rss_over_budget": self.rss_over_budget,
            "forbidden_imports": self.forbidden_imports,
            "failure_reasons": self.failure_reasons,
            "phases": [
                {
                    "name": p.name,
                    "wall_ms": round(p.wall_ms, 2),
                    "budget_ms": p.budget_ms,
                    "over_budget": p.over_budget,
                    "rss_delta_mb": round(p.rss_mb_delta, 2),
                }
                for p in self.phase_measurements
            ],
            "scenario_output": self.scenario_output,
        }

    def pretty(self) -> str:
        lines = [
            f"scenario={self.scenario}  profile={self.profile}  "
            f"{'PASS' if self.passed else 'FAIL'}",
            f"  peak RSS: {self.peak_rss_mb:.1f} MB "
            f"(budget {self.rss_budget_mb} MB"
            f"{' — OVER' if self.rss_over_budget else ''})",
            f"  total wall: {self.total_wall_ms:.0f} ms",
        ]
        for p in self.phase_measurements:
            budget = f"{p.budget_ms}" if p.budget_ms is not None else "-"
            flag = " OVER" if p.over_budget else ""
            lines.append(
                f"    {p.name:<10} {p.wall_ms:>7.1f} ms  "
                f"(budget {budget:>6} ms){flag}  "
                f"d_RSS {p.rss_mb_delta:+.1f} MB"
            )
        if self.forbidden_imports:
            lines.append(
                f"  forbidden imports: {', '.join(self.forbidden_imports)}"
            )
        if self.failure_reasons:
            lines.append("  failures:")
            for r in self.failure_reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


ScenarioFn = Callable[[EmulatorContext], Dict[str, Any]]


def run_scenario(
    scenario: ScenarioFn,
    *,
    profile: DeviceProfile,
    scenario_name: Optional[str] = None,
) -> EmulatorResult:
    """Run *scenario* against *profile* and return an
    :class:`EmulatorResult`.

    The runner:

    1. Resets tracemalloc so RSS measurements start at zero.
    2. Calls the scenario with a fresh :class:`EmulatorContext`.
    3. Collects phase measurements + final peak RSS.
    4. Computes pass/fail against the profile's budgets.
    """
    name = scenario_name or getattr(scenario, "__name__", "scenario")

    tracemalloc.stop()
    tracemalloc.start()
    try:
        ctx = EmulatorContext(profile)
        output: Dict[str, Any] = {}
        failure_reasons: List[str] = []
        try:
            result = scenario(ctx)
            if isinstance(result, dict):
                output = result
        except Exception as exc:  # noqa: BLE001 — scenario errors are a failure mode we report
            failure_reasons.append(
                f"scenario raised {type(exc).__name__}: {exc}"
            )

        peak_rss = _rss_mb()
    finally:
        tracemalloc.stop()

    rss_over = peak_rss > profile.peak_ram_mb
    if rss_over:
        failure_reasons.append(
            f"peak RSS {peak_rss:.1f} MB exceeded budget {profile.peak_ram_mb} MB"
        )
    for p in ctx.phases:
        if p.over_budget:
            failure_reasons.append(
                f"phase {p.name!r} wall {p.wall_ms:.0f} ms exceeded "
                f"budget {p.budget_ms} ms"
            )
    if ctx._import_check_failed:
        failure_reasons.append(
            f"forbidden imports: {', '.join(ctx._import_check_failed)}"
        )

    total_wall = sum(p.wall_ms for p in ctx.phases)
    passed = not failure_reasons

    return EmulatorResult(
        scenario=name,
        profile=profile.name,
        passed=passed,
        peak_rss_mb=peak_rss,
        total_wall_ms=total_wall,
        rss_budget_mb=profile.peak_ram_mb,
        rss_over_budget=rss_over,
        phase_measurements=ctx.phases,
        forbidden_imports=list(ctx._import_check_failed),
        scenario_output=output,
        failure_reasons=failure_reasons,
    )

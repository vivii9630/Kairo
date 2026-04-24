"""Rust smoke scenario — same workload as ``smoke``, but executed by
the ``kairo-edge-cli`` binary over the Rust ``kairo-edge-core`` crate.

Why this exists: the emulator has to exercise Rust — the canonical
edge runtime going forward — on the same profile budgets as the
Python scenarios. Running the Rust binary as a subprocess is a
faithful simulation of how a real edge deployment will invoke it
(from another language's shell, not in-process Python).

Measurement notes:

- **Wall time** is the subprocess round-trip plus whatever the CLI
  itself reports internally. We use the CLI's self-reported
  ``ingest_ns`` / ``query_ns`` for per-phase budgets because they
  isolate Rust's time from Python's subprocess overhead.
- **RSS via tracemalloc in the Python parent is ~meaningless** for a
  subprocess — the parent only allocates for the json payload + the
  subprocess handle, a few hundred KB. Peak RSS in the emulator
  output therefore reflects ``ingest_id`` + Python plumbing, not the
  Rust runtime's actual memory use. To capture native memory properly
  we'll switch to ``psutil.Process(child_pid).memory_info().peak_wss``
  when we add native inference providers — logged as open question in
  EDGE_ARCHITECTURE.md §10.
- **Build policy**: the scenario assumes the release binary already
  exists at ``edge/rust/target/release/kairo-edge-cli``. If missing,
  it fails fast with an actionable error rather than kicking off a
  cargo build (which would dominate the wall time and defeat the
  budget measurement).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from ..runner import EmulatorContext
from . import register_scenario
from .smoke import _CORPUS  # reuse identical corpus for apples-to-apples comparison


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI_BIN_REL = Path("edge/rust/target/release/kairo-edge-cli")
_CLI_BIN_EXE = _CLI_BIN_REL.with_suffix(".exe")


def _resolve_cli_binary() -> Path:
    for candidate in (_REPO_ROOT / _CLI_BIN_EXE, _REPO_ROOT / _CLI_BIN_REL):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"kairo-edge-cli binary not found. Build it first:\n"
        f"  cd {_REPO_ROOT / 'edge/rust'} && cargo build --release --bin kairo-edge-cli"
    )


def _scenario(ctx: EmulatorContext) -> Dict[str, Any]:
    cli = _resolve_cli_binary()

    payload = {
        "action": "smoke",
        "corpus": [{"id": d.id, "text": d.text, "metadata": d.metadata} for d in _CORPUS],
        "query": "felines hunt at night",
        "top_k": 3,
    }
    payload_json = json.dumps(payload)

    with ctx.phase("rust_subprocess"):
        result = subprocess.run(
            [str(cli)],
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=60,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"CLI exited {result.returncode}: "
            f"stdout={result.stdout[:200]!r} stderr={result.stderr[:200]!r}"
        )

    try:
        parsed = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"CLI stdout was not valid JSON: {result.stdout[:200]!r}"
        ) from exc

    if not parsed.get("ok"):
        raise RuntimeError(f"CLI reported error: {parsed.get('error')}")

    ingest_us = parsed["ingest_ns"] / 1000.0
    query_us = parsed["query_ns"] / 1000.0

    return {
        "binary": str(cli.relative_to(_REPO_ROOT)),
        "binary_size_kb": round(cli.stat().st_size / 1024, 1),
        "rust_ingest_us": round(ingest_us, 1),
        "rust_query_us": round(query_us, 1),
        "retrieved_ids": parsed["retrieved_ids"],
        "evidence_count": parsed["evidence_count"],
        "answer_preview": parsed["answer_preview"][:80],
        "corpus_size": parsed["corpus_size"],
    }


register_scenario("rust_smoke", _scenario)

# Kairo edge emulator

A scenario runner that enforces per-profile memory, wall-clock, and
import budgets on Kairo edge code. Runs on any developer machine — no
ARM hardware, Docker, or QEMU required for the gate.

The full architectural rationale is in [`../docs/EDGE_ARCHITECTURE.md`](../docs/EDGE_ARCHITECTURE.md);
this README is the how-to.

## Quickstart

```bash
# From repo root, run the default smoke scenario on pi-4
python -m edge.emulator --profile pi-4 --scenario smoke

# Run against every profile in one pass
python -m edge.emulator --all-profiles

# Machine-readable output
python -m edge.emulator --profile pi-zero --json
```

Exit code is non-zero when any profile fails, so CI can use the
emulator directly as a test gate.

## What a run looks like

```
scenario=smoke  profile=pi-4  PASS
  peak RSS: 4.1 MB (budget 1024 MB)
  total wall: 6 ms
    ingest         2.3 ms  (budget  10000 ms)  ΔRSS +0.8 MB
    retrieve       3.7 ms  (budget    200 ms)  ΔRSS +0.0 MB
```

## Profiles

Six named profiles live in [`profile.py`](./profile.py). Each carries
peak RAM, per-phase wall-clock budgets, allowed imports, and GPU /
WebGPU flags. Adding a profile is deliberate — we keep the set small
so CI budgets stay comparable across PRs.

| Profile | RAM budget | Use for |
|---|---|---|
| `pi-zero` | 256 MB | BM25-only, no semantic, extractive answers |
| `pi-4` | 1 GB | ONNX MiniLM semantic + 1B Q4 LLM |
| `pi-5` | 2 GB | Full retrieval + 3B Q4 LLM |
| `phone-mid` | 1 GB | LiteRT-LM Gemma 2B int4 |
| `browser-lite` | 400 MB virt | TS shell, BM25 only |
| `browser-gpu` | 1 GB virt | TS shell + WebLLM/LiteRT Web |

## Writing a scenario

A scenario is `(ctx: EmulatorContext) -> dict`:

```python
# edge/emulator/scenarios/myscenario.py
from . import register_scenario
from ..runner import EmulatorContext

def _scenario(ctx: EmulatorContext) -> dict:
    with ctx.phase("ingest"):
        ...
    with ctx.phase("retrieve"):
        ...
    return {"ok": True}

register_scenario("myscenario", _scenario)
```

The runner wraps each `ctx.phase(...)` in wall-clock + RSS timers and
compares them to the active profile's budget. Any phase over budget,
any import from the heavy-imports blocklist not in the profile's
allowlist, or any scenario-raised exception becomes a failure reason
— the final `EmulatorResult.passed` collapses them into a single bit
the CI can gate on.

Add the scenario to [`scenarios/__init__.py`](./scenarios/__init__.py)
so `from . import myscenario` fires during registry import.

## Measurement caveats

Memory is tracked via `tracemalloc` (Python-allocator-only). Good
enough for pure-Python edge code; once the ONNX / llama.cpp providers
land, the runner will switch to `psutil.Process().memory_info().rss`
so native allocations count. This gap is documented in
[`../docs/EDGE_ARCHITECTURE.md`](../docs/EDGE_ARCHITECTURE.md) §7 and
§10.

The emulator does **not** simulate ARM instructions — wall-clock
budgets are generous enough that passing on a modern x86 dev machine
correlates with passing on real Pi hardware. Release-candidate
validation still runs the full scenario set on actual devices.

## CI integration

```yaml
# suggested snippet
- name: edge emulator
  run: python -m edge.emulator --all-profiles
```

One step, one exit code, every profile covered.

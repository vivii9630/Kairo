"""CLI entry point: ``python -m kairo_edge.emulator`` is a no-go because
the emulator lives outside ``kairo_edge`` (it's dev-tooling, not a
runtime dep). Invoke as:

    python -m edge.emulator --profile pi-4 --scenario smoke

The CLI is intentionally tiny — all logic lives in :mod:`runner` and
the scenarios themselves. This file just wires argparse to them.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .profile import PROFILES, get_profile
from .runner import run_scenario
from .scenarios import get_scenario, list_scenarios


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="pi-4",
        choices=sorted(PROFILES),
        help="device profile to emulate (default: pi-4)",
    )
    parser.add_argument(
        "--scenario",
        default="smoke",
        help=f"scenario to run (known: {', '.join(list_scenarios())})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit result as JSON instead of the human-readable table",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="run the scenario against every profile; ignores --profile",
    )

    args = parser.parse_args(argv)
    scenario_fn = get_scenario(args.scenario)
    profile_names = (
        sorted(PROFILES) if args.all_profiles else [args.profile]
    )

    failed_any = False
    results = []
    for name in profile_names:
        profile = get_profile(name)
        result = run_scenario(
            scenario_fn, profile=profile, scenario_name=args.scenario
        )
        results.append(result)
        if not result.passed:
            failed_any = True

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            print(r.pretty())
            print()

    return 1 if failed_any else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

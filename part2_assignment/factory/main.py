"""Command-line entry point for the factory solver."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from factory.core import solve_from_json
else:  # pragma: no cover - exercised via module execution
    from .core import solve_from_json


def main() -> None:
    data = sys.stdin.read()
    result = solve_from_json(data)
    if result.status == "ok":
        output = {
            "status": "ok",
            "per_recipe_crafts_per_min": result.crafts_per_min,
            "per_machine_counts": result.machine_usage,
            "raw_consumption_per_min": result.raw_consumption,
        }
    else:
        output = {
            "status": "infeasible",
            "max_feasible_target_per_min": result.max_feasible_rate,
            "bottleneck_hint": result.bottlenecks or [],
        }
    json.dump(output, sys.stdout, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()

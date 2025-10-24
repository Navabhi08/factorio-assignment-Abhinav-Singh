"""Command-line interface for the belts solver."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from belts.core import solve_from_json
else:  # pragma: no cover - exercised via module execution
    from .core import solve_from_json


def main() -> None:
    data = sys.stdin.read()
    result = solve_from_json(data)
    if result.status == "ok":
        output = {
            "status": "ok",
            "max_flow_per_min": result.max_flow,
            "flows": result.flows,
        }
    else:
        output = {
            "status": "infeasible",
            "cut_reachable": result.cut_reachable or [],
            "deficit": result.deficit or {},
        }
    json.dump(output, sys.stdout, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()

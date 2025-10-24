"""Optional helper to sanity-check belt flow outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from belts.core import BeltsModel


def _accumulate_flows(flows: list[dict[str, object]]) -> Dict[Tuple[str, str], float]:
    totals: Dict[Tuple[str, str], float] = {}
    for entry in flows:
        edge = (str(entry["from"]), str(entry["to"]))
        totals[edge] = totals.get(edge, 0.0) + float(entry["flow"])
    return totals


def verify(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    model = BeltsModel(payload)
    result = model.solve()
    issues: list[str] = []
    if result.status != "ok":
        issues.append("solver reported infeasible instance")
        return issues
    totals = _accumulate_flows(result.flows)
    for edge in payload.get("edges", []):
        frm = str(edge["from"])
        to = str(edge["to"])
        capacity = float(edge.get("upper", float("inf")))
        lower = float(edge.get("lower", 0.0))
        value = totals.get((frm, to), 0.0)
        if value + 1e-6 < lower:
            issues.append(f"edge {frm}->{to} violates lower bound {value} < {lower}")
        if value - capacity > 1e-6:
            issues.append(f"edge {frm}->{to} exceeds capacity {value} > {capacity}")
    return issues


def main(path: str) -> None:
    problems = verify(Path(path))
    if problems:
        print("FAIL")
        for entry in problems:
            print(f" - {entry}")
    else:
        print("OK")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_belts.py <input.json>")
    main(sys.argv[1])

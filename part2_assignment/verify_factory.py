"""Optional helper to verify factory solutions against the JSON contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factory.core import FactoryModel, solve_from_json


def _check_balance(model: FactoryModel, crafts: dict[str, float]) -> Iterable[str]:
    for item in model.item_set:
        net = sum(n * crafts[name] for n, name in zip(model._net_vector(item), model.recipe_order))
        if item == model.target_item:
            if abs(net - model.target_rate) > 1e-6:
                yield f"target mismatch: expected {model.target_rate}, got {net}"
        elif item in model.raw_items:
            if net > 1e-6:
                yield f"raw item {item} has surplus {net}"
        else:
            if abs(net) > 1e-6:
                yield f"intermediate {item} not balanced (net={net})"


def verify(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    result = solve_from_json(json.dumps(payload))
    issues: list[str] = []
    if result.status != "ok":
        issues.append("solver reported infeasible instance")
        return issues
    model = FactoryModel(payload)
    issues.extend(_check_balance(model, result.crafts_per_min))
    for machine, cap in model.machine_caps.items():
        usage = result.machine_usage.get(machine, 0.0)
        if usage - cap > 1e-6:
            issues.append(f"machine {machine} exceeds cap: {usage} > {cap}")
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
        raise SystemExit("Usage: python verify_factory.py <input.json>")
    main(sys.argv[1])

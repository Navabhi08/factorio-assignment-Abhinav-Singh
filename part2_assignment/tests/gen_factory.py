"""Optional deterministic generator for factory JSON payloads."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class FactoryConfig:
    machines: int = 2
    depth: int = 3


def generate(config: FactoryConfig, seed: int = 0) -> Dict[str, object]:
    rng = random.Random(seed)
    machines = {
        f"assembler_{idx}": {"crafts_per_min": 30 + 10 * idx}
        for idx in range(1, config.machines + 1)
    }
    recipes: Dict[str, Dict[str, object]] = {}
    previous_output = "ore"
    for level in range(1, config.depth + 1):
        name = f"recipe_{level}"
        machine = rng.choice(list(machines))
        recipes[name] = {
            "machine": machine,
            "time_s": 1.5 + level * 0.25,
            "in": {previous_output: 1},
            "out": {f"item_{level}": 1},
        }
        previous_output = f"item_{level}"
    return {
        "machines": machines,
        "recipes": recipes,
        "target": {"item": previous_output, "rate_per_min": 60},
    }


def main() -> None:
    payload = generate(FactoryConfig())
    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    import sys

    main()

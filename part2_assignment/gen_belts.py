"""Optional deterministic generator for belt flow JSON payloads."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BeltConfig:
    layers: int = 2
    capacity: float = 200.0


def generate(config: BeltConfig, seed: int = 0) -> Dict[str, object]:
    rng = random.Random(seed)
    mids = [f"mid_{i}" for i in range(config.layers)]
    nodes = {name: {"cap": config.capacity} for name in mids}
    edges: List[Dict[str, object]] = []
    chain = ["source"] + mids + ["sink"]
    for idx in range(len(chain) - 1):
        lower = rng.uniform(0, config.capacity * 0.25)
        edges.append(
            {
                "from": chain[idx],
                "to": chain[idx + 1],
                "lower": round(lower, 2),
                "upper": config.capacity,
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "sources": {"source": config.capacity},
        "sink": "sink",
    }


def main() -> None:
    payload = generate(BeltConfig())
    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    import sys

    main()

"""Run lightweight smoke tests for the factory and belts CLIs."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from typing import Iterable, Tuple


def _run_command(command: str, payload: dict) -> Tuple[int, str]:
    process = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return process.returncode, process.stdout.decode()


def run(factory_cmd: str, belts_cmd: str) -> None:
    factory_payload = {
        "machines": {"assembler": {"crafts_per_min": 60}},
        "recipes": {
            "iron_plate": {
                "machine": "assembler",
                "time_s": 1.0,
                "in": {"iron_ore": 1},
                "out": {"iron_plate": 1},
            }
        },
        "target": {"item": "iron_plate", "rate_per_min": 60},
    }
    belts_payload = {
        "nodes": {"mid": {"cap": 300}},
        "sources": {"source": 120},
        "sink": "sink",
        "edges": [
            {"from": "source", "to": "mid", "upper": 120},
            {"from": "mid", "to": "sink", "upper": 120},
        ],
    }

    scenarios: Iterable[Tuple[str, str, dict]] = (
        ("factory", factory_cmd, factory_payload),
        ("belts", belts_cmd, belts_payload),
    )

    for label, command, payload in scenarios:
        code, stdout = _run_command(command, payload)
        if code != 0:
            raise SystemExit(f"{label} command '{command}' exited with code {code}")
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
            raise SystemExit(f"{label} output was not valid JSON: {stdout}") from exc
        if not isinstance(data, dict) or "status" not in data:
            raise SystemExit(f"{label} output missing required fields: {data}")
        print(f"{label} ✓ status={data['status']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python run_samples.py \"<factory command>\" \"<belts command>\""
        )
    run(sys.argv[1], sys.argv[2])

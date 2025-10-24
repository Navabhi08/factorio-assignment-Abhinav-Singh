import json

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from belts.core import solve_from_json


def test_belts_feasible_flow():
    payload = {
        "nodes": {"a": {"cap": 2000}},
        "sources": {"s1": 900, "s2": 600},
        "sink": "sink",
        "edges": [
            {"from": "s1", "to": "a", "upper": 900},
            {"from": "s2", "to": "a", "upper": 600},
            {"from": "a", "to": "b", "upper": 900},
            {"from": "a", "to": "c", "upper": 600},
            {"from": "b", "to": "sink", "upper": 900},
            {"from": "c", "to": "sink", "upper": 600},
        ],
    }
    result = solve_from_json(json.dumps(payload))
    assert result.status == "ok"
    assert abs(result.max_flow - 1500.0) <= 1e-6
    sink_flow = sum(edge["flow"] for edge in result.flows if edge["to"] == "sink")
    assert abs(sink_flow - 1500.0) <= 1e-6


def test_belts_infeasible_certificate():
    payload = {
        "nodes": {"a": {"cap": 1100}},
        "sources": {"s1": 900, "s2": 600},
        "sink": "sink",
        "edges": [
            {"from": "s1", "to": "a", "upper": 900},
            {"from": "s2", "to": "a", "upper": 600},
            {"from": "a", "to": "b", "upper": 700},
            {"from": "a", "to": "c", "upper": 600},
            {"from": "b", "to": "sink", "upper": 600},
            {"from": "c", "to": "sink", "upper": 600},
        ],
    }
    result = solve_from_json(json.dumps(payload))
    assert result.status == "infeasible"
    assert result.deficit["demand_balance"] >= 300 - 1e-6
    assert "a" in result.deficit["tight_nodes"] or result.deficit["tight_nodes"] is None

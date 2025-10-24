import json

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factory.core import FactoryModel, solve_from_json


def test_sample_factory_conservation():
    payload = {
        "machines": {
            "assembler_1": {"crafts_per_min": 30},
            "chemical": {"crafts_per_min": 60},
        },
        "recipes": {
            "iron_plate": {
                "machine": "chemical",
                "time_s": 3.2,
                "in": {"iron_ore": 1},
                "out": {"iron_plate": 1},
            },
            "copper_plate": {
                "machine": "chemical",
                "time_s": 3.2,
                "in": {"copper_ore": 1},
                "out": {"copper_plate": 1},
            },
            "green_circuit": {
                "machine": "assembler_1",
                "time_s": 0.5,
                "in": {"iron_plate": 1, "copper_plate": 3},
                "out": {"green_circuit": 1},
            },
        },
        "modules": {
            "assembler_1": {"prod": 0.1, "speed": 0.15},
            "chemical": {"prod": 0.2, "speed": 0.1},
        },
        "limits": {
            "raw_supply_per_min": {"iron_ore": 5000, "copper_ore": 5000},
            "max_machines": {"assembler_1": 300, "chemical": 300},
        },
        "target": {"item": "green_circuit", "rate_per_min": 1800},
    }
    result = solve_from_json(json.dumps(payload))
    assert result.status == "ok"
    model = FactoryModel(payload)
    crafts = result.crafts_per_min
    target_net = sum(n * crafts[name] for n, name in zip(model._net_vector(model.target_item), model.recipe_order))
    assert abs(target_net - model.target_rate) <= 1e-3
    for item in (model.item_set - model.raw_items - {model.target_item}):
        net = sum(n * crafts[name] for n, name in zip(model._net_vector(item), model.recipe_order))
        assert abs(net) <= 1e-6
    for machine, cap in model.machine_caps.items():
        assert result.machine_usage[machine] <= cap + 1e-6


def test_factory_infeasible_when_caps_tight():
    payload = {
        "machines": {"assembler": {"crafts_per_min": 2}},
        "recipes": {
            "target": {
                "machine": "assembler",
                "time_s": 60.0,
                "in": {"ore": 1},
                "out": {"widget": 1},
            }
        },
        "limits": {
            "raw_supply_per_min": {"ore": 5},
            "max_machines": {"assembler": 1},
        },
        "target": {"item": "widget", "rate_per_min": 20},
    }
    result = solve_from_json(json.dumps(payload))
    assert result.status == "infeasible"
    assert result.max_feasible_rate < 20


def test_factory_tie_break_prefers_faster_machine():
    payload = {
        "machines": {
            "fast": {"crafts_per_min": 10},
            "slow": {"crafts_per_min": 5},
        },
        "recipes": {
            "fast_recipe": {
                "machine": "fast",
                "time_s": 60.0,
                "in": {},
                "out": {"widget": 1},
            },
            "slow_recipe": {
                "machine": "slow",
                "time_s": 60.0,
                "in": {},
                "out": {"widget": 1},
            },
        },
        "target": {"item": "widget", "rate_per_min": 50},
    }
    result = solve_from_json(json.dumps(payload))
    assert result.status == "ok"
    assert result.crafts_per_min["slow_recipe"] == 0.0
    assert result.crafts_per_min["fast_recipe"] > 0.0

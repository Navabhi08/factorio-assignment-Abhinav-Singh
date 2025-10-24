"""Core logic for the factory steady-state solver."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from common.lp import LinearProgramResult, solve_linear_program


@dataclass
class Machine:
    name: str
    crafts_per_min: float
    speed_bonus: float = 0.0
    prod_bonus: float = 0.0

    def effective_crafts_per_min(self, recipe_time_s: float) -> float:
        speed = 1.0 + self.speed_bonus
        return self.crafts_per_min * speed * 60.0 / recipe_time_s


@dataclass
class Recipe:
    name: str
    machine: str
    time_s: float
    inputs: Dict[str, float]
    outputs: Dict[str, float]


@dataclass
class FactoryResult:
    status: str
    crafts_per_min: Dict[str, float]
    machine_usage: Dict[str, float]
    raw_consumption: Dict[str, float]
    max_feasible_rate: Optional[float] = None
    bottlenecks: Optional[List[str]] = None


class FactoryModel:
    def __init__(self, payload: Dict[str, object]) -> None:
        self.machines = self._parse_machines(payload.get("machines", {}))
        self.recipes = self._parse_recipes(payload.get("recipes", {}))
        self.modules = self._parse_modules(payload.get("modules", {}))
        self.raw_caps = {
            item: float(value)
            for item, value in payload.get("limits", {}).get("raw_supply_per_min", {}).items()
        }
        self.machine_caps = {
            name: float(value)
            for name, value in payload.get("limits", {}).get("max_machines", {}).items()
        }
        target = payload.get("target", {})
        self.target_item = str(target.get("item"))
        self.target_rate = float(target.get("rate_per_min", 0.0))
        self._apply_modules()
        self.recipe_order = list(self.recipes.keys())
        self.item_set = self._collect_items()
        self.produced_items = self._collect_produced_items()
        self.raw_items = self._identify_raw_items()

    def _parse_machines(self, raw: Dict[str, Dict[str, float]]) -> Dict[str, Machine]:
        machines: Dict[str, Machine] = {}
        for name, cfg in raw.items():
            machines[name] = Machine(name=name, crafts_per_min=float(cfg["crafts_per_min"]))
        return machines

    def _parse_recipes(self, raw: Dict[str, Dict[str, object]]) -> Dict[str, Recipe]:
        recipes: Dict[str, Recipe] = {}
        for name, cfg in raw.items():
            recipes[name] = Recipe(
                name=name,
                machine=str(cfg["machine"]),
                time_s=float(cfg["time_s"]),
                inputs={item: float(qty) for item, qty in cfg.get("in", {}).items()},
                outputs={item: float(qty) for item, qty in cfg.get("out", {}).items()},
            )
        return recipes

    def _parse_modules(self, raw: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        modules: Dict[str, Tuple[float, float]] = {}
        for machine, cfg in raw.items():
            modules[machine] = (float(cfg.get("speed", 0.0)), float(cfg.get("prod", 0.0)))
        return modules

    def _apply_modules(self) -> None:
        for name, machine in self.machines.items():
            speed, prod = self.modules.get(name, (0.0, 0.0))
            machine.speed_bonus = speed
            machine.prod_bonus = prod

    def _collect_items(self) -> set:
        items = set()
        for recipe in self.recipes.values():
            items.update(recipe.inputs)
            items.update(recipe.outputs)
        items.add(self.target_item)
        items.update(self.raw_caps.keys())
        return items

    def _collect_produced_items(self) -> set:
        produced = set()
        for recipe in self.recipes.values():
            produced.update(recipe.outputs)
        return produced

    def _identify_raw_items(self) -> set:
        raw = set(self.raw_caps.keys())
        for item in self.item_set:
            if item not in self.produced_items:
                raw.add(item)
        return raw

    def _net_vector(self, item: str) -> List[float]:
        vector: List[float] = []
        for recipe_name in self.recipe_order:
            recipe = self.recipes[recipe_name]
            machine = self.machines[recipe.machine]
            prod_multiplier = 1.0 + machine.prod_bonus
            produced = prod_multiplier * recipe.outputs.get(item, 0.0)
            consumed = recipe.inputs.get(item, 0.0)
            vector.append(produced - consumed)
        return vector

    def _effective_rates(self) -> Dict[str, float]:
        rates: Dict[str, float] = {}
        for recipe_name in self.recipe_order:
            recipe = self.recipes[recipe_name]
            machine = self.machines[recipe.machine]
            rates[recipe_name] = machine.effective_crafts_per_min(recipe.time_s)
        return rates

    # Constraint builders -------------------------------------------------

    def _build_equalities(
        self, *, include_target: bool, include_z: bool
    ) -> Tuple[List[List[float]], List[float]]:
        rows: List[List[float]] = []
        rhs: List[float] = []
        if include_target:
            row = self._net_vector(self.target_item)
            if include_z:
                row.append(-1.0)
                rhs.append(0.0)
            else:
                rhs.append(self.target_rate)
            rows.append(row)
        else:
            if include_z:
                row = self._net_vector(self.target_item)
                row.append(-1.0)
                rows.append(row)
                rhs.append(0.0)
        for item in sorted(self.item_set - self.raw_items - {self.target_item}):
            row = self._net_vector(item)
            if include_z:
                row.append(0.0)
            rows.append(row)
            rhs.append(0.0)
        return rows, rhs

    def _build_inequalities(self, *, include_z: bool) -> Tuple[List[List[float]], List[float]]:
        rows: List[List[float]] = []
        rhs: List[float] = []
        for item in sorted(self.raw_items):
            net = self._net_vector(item)
            row = net.copy()
            if include_z:
                row.append(0.0)
            rows.append(row)
            rhs.append(0.0)
            cap = self.raw_caps.get(item)
            if cap is not None:
                cap_row = [-value for value in net]
                if include_z:
                    cap_row.append(0.0)
                rows.append(cap_row)
                rhs.append(cap)
        rates = self._effective_rates()
        for machine_name, cap in sorted(self.machine_caps.items()):
            row: List[float] = []
            for recipe_name in self.recipe_order:
                if self.recipes[recipe_name].machine == machine_name:
                    eff = rates[recipe_name]
                    row.append(1.0 / eff if eff > 0 else 0.0)
                else:
                    row.append(0.0)
            if include_z:
                row.append(0.0)
            rows.append(row)
            rhs.append(cap)
        return rows, rhs

    # Cost vectors --------------------------------------------------------

    def _cost_for_minimum_machines(self) -> List[float]:
        rates = self._effective_rates()
        return [1.0 / rates[name] if rates[name] > 0 else 0.0 for name in self.recipe_order]

    def _cost_for_max_rate(self) -> List[float]:
        return [0.0 for _ in self.recipe_order] + [-1.0]

    # Public API ----------------------------------------------------------

    def solve(self) -> FactoryResult:
        if not self.recipe_order:
            return FactoryResult("ok", {}, {}, {})
        eq_rows, eq_rhs = self._build_equalities(include_target=True, include_z=False)
        ub_rows, ub_rhs = self._build_inequalities(include_z=False)
        cost = self._cost_for_minimum_machines()
        result = solve_linear_program(cost, eq_rows, eq_rhs, ub_rows, ub_rhs)
        if result.status != "optimal":
            max_rate, crafts = self._solve_max_rate()
            hints = self._identify_bottlenecks(crafts)
            return FactoryResult(
                status="infeasible",
                crafts_per_min={},
                machine_usage={},
                raw_consumption={},
                max_feasible_rate=min(max_rate, self.target_rate),
                bottlenecks=hints,
            )
        crafts = self._craft_mapping(result.solution)
        machines = self._machine_usage(crafts)
        raw = self._raw_consumption(crafts)
        return FactoryResult("ok", crafts, machines, raw)

    def _solve_max_rate(self) -> Tuple[float, Dict[str, float]]:
        eq_rows, eq_rhs = self._build_equalities(include_target=False, include_z=True)
        ub_rows, ub_rhs = self._build_inequalities(include_z=True)
        cost = self._cost_for_max_rate()
        result = solve_linear_program(cost, eq_rows, eq_rhs, ub_rows, ub_rhs)
        if result.status != "optimal":
            return 0.0, {name: 0.0 for name in self.recipe_order}
        crafts = self._craft_mapping(result.solution[:-1])
        max_rate = result.solution[-1] if result.solution else 0.0
        return max_rate, crafts

    # Derived metrics -----------------------------------------------------

    def _craft_mapping(self, solution: Sequence[float]) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        for name, value in zip(self.recipe_order, solution):
            mapping[name] = max(0.0, float(value))
        return mapping

    def _machine_usage(self, crafts: Dict[str, float]) -> Dict[str, float]:
        usage: Dict[str, float] = {}
        rates = self._effective_rates()
        for recipe_name, recipe in self.recipes.items():
            rate = crafts.get(recipe_name, 0.0)
            if rate <= 0:
                continue
            eff = rates[recipe_name]
            machines_needed = rate / eff if eff > 0 else 0.0
            usage[recipe.machine] = usage.get(recipe.machine, 0.0) + machines_needed
        for machine in self.machines.keys():
            usage.setdefault(machine, 0.0)
        return dict(sorted(usage.items()))

    def _raw_consumption(self, crafts: Dict[str, float]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for recipe_name, recipe in self.recipes.items():
            rate = crafts.get(recipe_name, 0.0)
            if rate <= 0:
                continue
            machine = self.machines[recipe.machine]
            prod_multiplier = 1.0 + machine.prod_bonus
            for item, qty in recipe.inputs.items():
                totals[item] = totals.get(item, 0.0) + qty * rate
            for item, qty in recipe.outputs.items():
                totals[item] = totals.get(item, 0.0) - prod_multiplier * qty * rate
        consumption: Dict[str, float] = {}
        tol = 1e-9
        for item in sorted(self.raw_items):
            net = totals.get(item, 0.0)
            value = net if net > tol else 0.0
            if value > 0.0 or item in self.raw_caps:
                consumption[item] = value
        return consumption

    def _identify_bottlenecks(self, crafts: Dict[str, float]) -> List[str]:
        hints: List[str] = []
        usage = self._machine_usage(crafts)
        raw = self._raw_consumption(crafts)
        tol = 1e-6
        for machine, cap in sorted(self.machine_caps.items()):
            if usage.get(machine, 0.0) >= cap - tol:
                hints.append(f"{machine} cap")
        for item, cap in sorted(self.raw_caps.items()):
            if raw.get(item, 0.0) >= cap - tol:
                hints.append(f"{item} supply")
        if not hints:
            hints.append("target limited")
        return hints


def solve_from_json(data: str) -> FactoryResult:
    payload = json.loads(data)
    model = FactoryModel(payload)
    return model.solve()

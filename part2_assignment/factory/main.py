"""Factory solver and command-line entry point.

This module now houses the entire steady-state optimisation logic, including the
linear-programming utilities that used to live under ``common/lp.py``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Linear programming helpers (moved from the former ``common/lp`` module).


@dataclass
class LinearProgramResult:
    status: str
    objective: float
    solution: List[float]


class _Tableau:
    def __init__(self, num_vars: int) -> None:
        self.col_types: List[str] = ["orig"] * num_vars
        self.rows: List[List[float]] = []
        self.rhs: List[float] = []
        self.basis: List[int] = []
        self.tol = 1e-9

    def _extend_rows(self) -> None:
        for row in self.rows:
            row.append(0.0)

    def add_constraint(self, coeffs: Sequence[float], rhs: float, *, kind: str) -> None:
        if kind not in {"eq", "le"}:
            raise ValueError(f"Unsupported constraint type: {kind}")
        row = [0.0] * len(self.col_types)
        for idx, value in enumerate(coeffs):
            row[idx] = float(value)
        rhs_val = float(rhs)
        if kind == "le":
            if rhs_val < 0:
                row = [-v for v in row]
                rhs_val = -rhs_val
            slack_idx = self._add_column("slack")
            row.append(0.0)
            row[slack_idx] = 1.0
            basis_var = slack_idx
        else:  # equality
            if rhs_val < 0:
                row = [-v for v in row]
                rhs_val = -rhs_val
            art_idx = self._add_column("artificial")
            row.append(0.0)
            row[art_idx] = 1.0
            basis_var = art_idx
        self.rows.append(row)
        self.rhs.append(rhs_val)
        self.basis.append(basis_var)

    def _add_column(self, kind: str) -> int:
        self.col_types.append(kind)
        self._extend_rows()
        return len(self.col_types) - 1

    def build_tableau(self) -> List[List[float]]:
        tableau: List[List[float]] = []
        for row, rhs in zip(self.rows, self.rhs):
            tableau.append(row + [rhs])
        num_cols = len(self.col_types)
        tableau.append([0.0] * (num_cols + 1))
        return tableau


def _pivot(tableau: List[List[float]], pivot_row: int, pivot_col: int, tol: float) -> None:
    pivot_val = tableau[pivot_row][pivot_col]
    if abs(pivot_val) < tol:
        raise ValueError("Pivot value too small")
    num_rows = len(tableau)
    num_cols = len(tableau[0])
    inv = 1.0 / pivot_val
    for j in range(num_cols):
        tableau[pivot_row][j] *= inv
    for i in range(num_rows):
        if i == pivot_row:
            continue
        factor = tableau[i][pivot_col]
        if abs(factor) <= tol:
            continue
        for j in range(num_cols):
            tableau[i][j] -= factor * tableau[pivot_row][j]


def _choose_entering(tableau: List[List[float]], col_types: Sequence[str], tol: float) -> Optional[int]:
    objective = tableau[-1]
    num_cols = len(objective) - 1
    for idx in range(num_cols):
        if objective[idx] > tol:
            if col_types[idx] == "artificial":
                continue
            return idx
    return None


def _choose_leaving(tableau: List[List[float]], pivot_col: int, basis: Sequence[int], tol: float) -> Optional[int]:
    best_row: Optional[int] = None
    best_ratio: float = float("inf")
    for row_idx, row in enumerate(tableau[:-1]):
        coeff = row[pivot_col]
        if coeff > tol:
            ratio = row[-1] / coeff
            if ratio < best_ratio - tol:
                best_ratio = ratio
                best_row = row_idx
            elif abs(ratio - best_ratio) <= tol:
                if best_row is None or basis[row_idx] < basis[best_row]:
                    best_row = row_idx
    return best_row


def _simplex_phase(tableau: List[List[float]], col_types: List[str], basis: List[int], tol: float) -> str:
    while True:
        entering = _choose_entering(tableau, col_types, tol)
        if entering is None:
            return "optimal"
        leaving = _choose_leaving(tableau, entering, basis, tol)
        if leaving is None:
            return "unbounded"
        _pivot(tableau, leaving, entering, tol)
        basis[leaving] = entering


def _prepare_phase_one(table: _Tableau) -> List[List[float]]:
    tableau = table.build_tableau()
    objective = tableau[-1]
    num_cols = len(objective) - 1
    for idx, kind in enumerate(table.col_types):
        if kind == "artificial":
            objective[idx] = 1.0
        else:
            objective[idx] = 0.0
    objective[-1] = 0.0
    for row_idx, basic_var in enumerate(table.basis):
        if table.col_types[basic_var] == "artificial":
            row = tableau[row_idx]
            for col in range(num_cols + 1):
                objective[col] += row[col]
    return tableau


def _remove_artificial(tableau: List[List[float]], table: _Tableau) -> None:
    num_cols = len(table.col_types)
    remove_indices = [i for i, kind in enumerate(table.col_types) if kind == "artificial"]
    for idx in sorted(remove_indices, reverse=True):
        for row_idx, basic in enumerate(table.basis):
            if basic == idx:
                pivot_col = None
                for candidate, kind in enumerate(table.col_types):
                    if kind != "artificial" and abs(tableau[row_idx][candidate]) > table.tol:
                        pivot_col = candidate
                        break
                if pivot_col is None:
                    del tableau[row_idx]
                    del table.basis[row_idx]
                    del table.rows[row_idx]
                    del table.rhs[row_idx]
                    break
                _pivot(tableau, row_idx, pivot_col, table.tol)
                table.basis[row_idx] = pivot_col
        for row in tableau:
            del row[idx]
        del table.col_types[idx]
        for i, basic in enumerate(table.basis):
            if basic > idx:
                table.basis[i] -= 1


def _set_phase_two_objective(tableau: List[List[float]], table: _Tableau, cost: Sequence[float]) -> None:
    objective = tableau[-1]
    num_cols = len(objective) - 1
    for j in range(num_cols):
        objective[j] = 0.0
    for idx, kind in enumerate(table.col_types):
        if kind == "orig":
            objective[idx] = -float(cost[idx])
    objective[-1] = 0.0
    for row_idx, basic in enumerate(table.basis):
        coeff = objective[basic]
        if abs(coeff) > table.tol:
            row = tableau[row_idx]
            for col in range(num_cols + 1):
                objective[col] -= coeff * row[col]


def solve_linear_program(
    cost: Sequence[float],
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    *,
    tol: float = 1e-9,
) -> LinearProgramResult:
    num_vars = len(cost)
    table = _Tableau(num_vars)
    if A_eq and b_eq:
        for row, rhs in zip(A_eq, b_eq):
            table.add_constraint(row, rhs, kind="eq")
    if A_ub and b_ub:
        for row, rhs in zip(A_ub, b_ub):
            table.add_constraint(row, rhs, kind="le")
    tableau = _prepare_phase_one(table)
    table.tol = tol
    status = _simplex_phase(tableau, table.col_types, table.basis, table.tol)
    if status == "unbounded":
        return LinearProgramResult("unbounded", float("nan"), [0.0] * num_vars)
    phase_one_value = tableau[-1][-1]
    if phase_one_value > tol:
        return LinearProgramResult("infeasible", float("nan"), [0.0] * num_vars)
    _remove_artificial(tableau, table)
    _set_phase_two_objective(tableau, table, cost)
    status = _simplex_phase(tableau, table.col_types, table.basis, table.tol)
    if status != "optimal":
        return LinearProgramResult(status, float("nan"), [0.0] * num_vars)
    solution = [0.0] * len(table.col_types)
    for row_idx, basic in enumerate(table.basis):
        solution[basic] = tableau[row_idx][-1]
    result = solution[:num_vars]
    objective = sum(cost[i] * result[i] for i in range(num_vars))
    for idx, value in enumerate(result):
        if abs(value) < tol:
            result[idx] = 0.0
    return LinearProgramResult("optimal", objective, result)


# ---------------------------------------------------------------------------
# Factory domain model and solver.


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

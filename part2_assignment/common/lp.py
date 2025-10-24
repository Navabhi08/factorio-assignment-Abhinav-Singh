"""Linear programming utilities for the Factorio assessment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


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
                    # remove redundant row
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

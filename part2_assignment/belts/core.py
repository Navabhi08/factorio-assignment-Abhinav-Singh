"""Core solver for the bounded belt flow problem."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple


class Edge:
    __slots__ = ("to", "rev", "capacity", "initial")

    def __init__(self, to: int, rev: int, capacity: float) -> None:
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.initial = capacity


class Dinic:
    def __init__(self) -> None:
        self.graph: List[List[Edge]] = []

    def add_node(self) -> int:
        self.graph.append([])
        return len(self.graph) - 1

    def ensure_node(self, idx: int) -> None:
        while idx >= len(self.graph):
            self.add_node()

    def add_edge(self, u: int, v: int, capacity: float) -> Edge:
        self.ensure_node(max(u, v))
        forward = Edge(v, len(self.graph[v]), capacity)
        backward = Edge(u, len(self.graph[u]), 0.0)
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        return forward

    def max_flow(self, source: int, sink: int) -> float:
        flow = 0.0
        n = len(self.graph)
        level = [0] * n
        while True:
            for i in range(n):
                level[i] = -1
            queue: Deque[int] = deque([source])
            level[source] = 0
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 1e-12 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                break
            it = [0] * n

            def dfs(v: int, f: float) -> float:
                if v == sink:
                    return f
                for i in range(it[v], len(self.graph[v])):
                    edge = self.graph[v][i]
                    if edge.capacity <= 1e-12 or level[edge.to] != level[v] + 1:
                        it[v] += 1
                        continue
                    pushed = dfs(edge.to, min(f, edge.capacity))
                    if pushed > 0:
                        edge.capacity -= pushed
                        rev = self.graph[edge.to][edge.rev]
                        rev.capacity += pushed
                        return pushed
                    it[v] += 1
                return 0.0

            while True:
                pushed = dfs(source, float("inf"))
                if pushed <= 1e-12:
                    break
                flow += pushed
        return flow


@dataclass
class BeltResult:
    status: str
    flows: List[Dict[str, object]]
    max_flow: Optional[float] = None
    cut_reachable: Optional[List[str]] = None
    deficit: Optional[Dict[str, object]] = None


class BeltsModel:
    def __init__(self, payload: Dict[str, object]) -> None:
        self.payload = payload
        self.graph = Dinic()
        self.node_index: Dict[str, int] = {}
        self.balance: Dict[int, float] = {}
        self.original_edges: List[Tuple[str, str, float, float, Edge]] = []
        self.node_caps: Dict[str, Tuple[str, str, Edge, float]] = {}
        self.base_node: Dict[str, str] = {}

    def solve(self) -> BeltResult:
        if not self.payload.get("edges"):
            return BeltResult("ok", [], 0.0)
        sources = {str(k): float(v) for k, v in self.payload.get("sources", {}).items()}
        sink = str(self.payload.get("sink"))
        node_caps = {
            name: float(cfg["cap"])
            for name, cfg in self.payload.get("nodes", {}).items()
            if "cap" in cfg
        }
        total_supply = sum(sources.values())
        self._build_nodes(node_caps, sources, sink)
        self._add_original_edges()
        self._add_supply_edges(sources, sink, total_supply)
        feasible, deficit = self._run_feasibility_check()
        if not feasible:
            cut_info = self._build_cut_certificate(deficit)
            return BeltResult("infeasible", [], None, cut_info[0], cut_info[1])
        flow_value = self._push_main_flow()
        flows = self._extract_flows()
        return BeltResult("ok", flows, flow_value)

    # Node and edge setup -------------------------------------------------

    def _node_id(self, name: str) -> int:
        if name not in self.node_index:
            idx = self.graph.add_node()
            self.node_index[name] = idx
            self.balance[idx] = 0.0
        return self.node_index[name]

    def _add_balance(self, node: str, delta: float) -> None:
        idx = self._node_id(node)
        self.balance[idx] = self.balance.get(idx, 0.0) + delta

    def _build_nodes(self, node_caps: Dict[str, float], sources: Dict[str, float], sink: str) -> None:
        for node, cap in node_caps.items():
            if node in sources or node == sink:
                self._node_id(node)
                continue
            vin = f"{node}__in"
            vout = f"{node}__out"
            self.base_node[vin] = node
            self.base_node[vout] = node
            u = self._node_id(vin)
            v = self._node_id(vout)
            edge = self.graph.add_edge(u, v, cap)
            self.graph.graph[v][edge.rev].capacity = 0.0
            self.node_caps[node] = (vin, vout, edge, cap)
        for node in set(list(sources.keys()) + [sink]):
            self._node_id(node)

    def _adjust_endpoint(self, node: str, is_incoming: bool) -> str:
        if node in self.node_caps:
            vin, vout, _, _ = self.node_caps[node]
            return vin if is_incoming else vout
        return node

    def _add_original_edges(self) -> None:
        for cfg in self.payload.get("edges", []):
            origin = self._adjust_endpoint(str(cfg["from"]), is_incoming=False)
            dest = self._adjust_endpoint(str(cfg["to"]), is_incoming=True)
            lower = float(cfg.get("lower", 0.0))
            upper = float(cfg.get("upper"))
            if upper < lower:
                raise ValueError("Edge upper bound must be >= lower bound")
            edge = self._add_lower_bounded_edge(origin, dest, lower, upper)
            self.original_edges.append((origin, dest, lower, upper, edge))

    def _add_supply_edges(self, sources: Dict[str, float], sink: str, total_supply: float) -> None:
        self.src_node = "__super_src__"
        self.sink_node = "__super_sink__"
        src_id = self._node_id(self.src_node)
        sink_id = self._node_id(self.sink_node)
        for name, supply in sorted(sources.items()):
            origin = self._adjust_endpoint(name, is_incoming=False)
            edge = self._add_lower_bounded_edge(self.src_node, origin, supply, supply)
            self.original_edges.append((self.src_node, origin, supply, supply, edge))
        sink_entry = self._adjust_endpoint(sink, is_incoming=True)
        edge = self._add_lower_bounded_edge(sink_entry, self.sink_node, total_supply, total_supply)
        self.original_edges.append((sink_entry, self.sink_node, total_supply, total_supply, edge))
        cycle_upper = total_supply + sum(upper - lower for _, _, lower, upper, _ in self.original_edges)
        self._add_lower_bounded_edge(self.sink_node, self.src_node, 0.0, cycle_upper)

    def _add_lower_bounded_edge(self, u_name: str, v_name: str, lower: float, upper: float) -> Edge:
        u = self._node_id(u_name)
        v = self._node_id(v_name)
        capacity = upper - lower
        edge = self.graph.add_edge(u, v, capacity)
        self.graph.graph[v][edge.rev].initial = 0.0
        self._add_balance(u_name, -lower)
        self._add_balance(v_name, lower)
        return edge

    # Flow execution ------------------------------------------------------

    def _run_feasibility_check(self) -> Tuple[bool, float]:
        super_source = self._node_id("__balance_src__")
        super_sink = self._node_id("__balance_sink__")
        total_demand = 0.0
        for idx in range(len(self.graph.graph)):
            balance = self.balance.get(idx, 0.0)
            if balance > 1e-9:
                self.graph.add_edge(super_source, idx, balance)
                total_demand += balance
            elif balance < -1e-9:
                self.graph.add_edge(idx, super_sink, -balance)
        flow = self.graph.max_flow(super_source, super_sink)
        deficit = max(0.0, total_demand - flow)
        return flow + 1e-6 >= total_demand, deficit

    def _push_main_flow(self) -> float:
        src = self.node_index[self.src_node]
        sink = self.node_index[self.sink_node]
        flow = self.graph.max_flow(src, sink)
        return flow

    # Extraction ----------------------------------------------------------

    def _edge_flow(self, edge: Edge, lower: float) -> float:
        return lower + (edge.initial - edge.capacity)

    def _extract_flows(self) -> List[Dict[str, object]]:
        flows: List[Dict[str, object]] = []
        for origin, dest, lower, upper, edge in self.original_edges:
            if origin.startswith("__"):
                continue
            if dest == self.sink_node:
                continue
            flow_value = self._edge_flow(edge, lower)
            if origin in self.base_node:
                origin = self.base_node[origin]
            if dest in self.base_node:
                dest = self.base_node[dest]
            flows.append({"from": origin, "to": dest, "flow": flow_value})
        return flows

    def _build_cut_certificate(self, deficit: float) -> Tuple[List[str], Dict[str, object]]:
        src = self.node_index[self.src_node]
        visited = [False] * len(self.graph.graph)
        queue = deque([src])
        visited[src] = True
        while queue:
            node = queue.popleft()
            for edge in self.graph.graph[node]:
                if edge.capacity > 1e-9 and not visited[edge.to]:
                    visited[edge.to] = True
                    queue.append(edge.to)
        reachable: List[str] = []
        for name, idx in self.node_index.items():
            if visited[idx]:
                base = self.base_node.get(name, name)
                if base not in reachable and not base.startswith("__"):
                    reachable.append(base)
        reachable.sort()
        tight_edges: List[Dict[str, object]] = []
        for origin, dest, lower, upper, edge in self.original_edges:
            if origin.startswith("__") or dest.startswith("__"):
                continue
            o_idx = self.node_index.get(origin, -1)
            d_idx = self.node_index.get(dest, -1)
            if o_idx == -1 or d_idx == -1:
                continue
            if visited[o_idx] and not visited[d_idx]:
                flow_value = self._edge_flow(edge, lower)
                remaining = max(0.0, upper - flow_value)
                if origin in self.base_node:
                    origin = self.base_node[origin]
                if dest in self.base_node:
                    dest = self.base_node[dest]
                tight_edges.append({"from": origin, "to": dest, "flow_needed": deficit})
        certificate = {
            "demand_balance": deficit,
            "tight_nodes": [node for node in reachable if node in self.node_caps],
            "tight_edges": tight_edges or None,
        }
        return reachable, certificate


def solve_from_json(data: str) -> BeltResult:
    payload = json.loads(data)
    model = BeltsModel(payload)
    return model.solve()

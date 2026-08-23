"""Graph query engine: SPARQL-like queries, path finding, subgraph extraction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from graph.knowledge import KnowledgeGraph


@dataclass
class PathResult:
    """An ordered walk through the graph connecting two entities."""

    nodes: list[str]
    edges: list[str]

    def length(self) -> int:
        return len(self.edges)


class GraphQueryEngine:
    """Read-side API over the knowledge graph for search integration."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def find_by_type(self, etype: str, limit: int = 50) -> list[str]:
        """List entity names filtered by their type label."""
        return [name for name, ent in self.graph.entities.items()
                if ent.type == etype][:limit]

    def neighbors(self, entity: str, depth: int = 1) -> dict[str, list[str]]:
        """Return an adjacency map up to ``depth`` hops from ``entity``."""
        frontier = {entity}
        seen: dict[str, list[str]] = {}
        for _ in range(depth):
            nxt: set[str] = set()
            for node in frontier:
                links = [r.obj for r in self.graph.relations if r.subject == node]
                inbound = [r.subject for r in self.graph.relations if r.obj == node]
                seen[node] = links + inbound
                nxt.update(seen[node])
            frontier = nxt - set(seen)
            if not frontier:
                break
        return seen

    def shortest_path(self, source: str, target: str, max_depth: int = 5) -> PathResult | None:
        """BFS shortest path between two entities, or None when disconnected."""
        queue: deque[tuple[str, list[str], list[str]]] = deque([(source, [source], [])])
        visited: set[str] = {source}
        while queue:
            node, nodes, edges = queue.popleft()
            if node == target:
                return PathResult(nodes=nodes, edges=edges)
            if len(edges) >= max_depth:
                continue
            outgoing = [(r.obj, r.predicate) for r in self.graph.relations if r.subject == node]
            incoming = [(r.subject, r.predicate + "^") for r in self.graph.relations if r.obj == node]
            for neighbor, edge in outgoing + incoming:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, nodes + [neighbor], edges + [edge]))
        return None

    def subgraph(self, seed_entities: list[str], max_edges: int = 100) -> dict:
        """Extract the ego-subgraph around seeds in a JSON-friendly layout."""
        keep = set(seed_entities)
        edges = []
        for rel in self.graph.relations:
            if rel.subject in keep and rel.obj in keep:
                edges.append({"source": rel.subject,
                              "label": rel.predicate, "target": rel.obj})
            if len(keep) < len(seed_entities) * 3 and len(edges) < max_edges:
                keep.update({rel.subject, rel.obj} & (keep | seed_entities))
        nodes = [{"id": n, "type": self.graph.entities[n].type}
                 for n in keep if n in self.graph.entities]
        return {"nodes": nodes[:max_edges], "edges": edges[:max_edges]}

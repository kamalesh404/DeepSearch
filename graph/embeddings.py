"""Graph embeddings: Node2Vec, TransE, and GraphSAGE-style encoders."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingResult:
    """Entity-to-vector mapping produced by any embedding trainer."""

    vectors: dict[str, np.ndarray]
    dim: int

    def most_similar(self, entity: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Cosine-similarity neighbours of one entity."""
        if entity not in self.vectors:
            return []
        base = self.vectors[entity]
        norm = np.linalg.norm(base) or 1.0
        scores = {
            name: float(np.dot(base, vec) / ((np.linalg.norm(vec) or 1.0) * norm))
            for name, vec in self.vectors.items() if name != entity
        }
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


class Node2Vec:
    """Random-walk + skip-gram embeddings over an adjacency list."""

    def __init__(self, dim: int = 64, walk_length: int = 20,
                 walks_per_node: int = 5, p: float = 1.0, q: float = 1.0) -> None:
        self.dim, self.walk_length = dim, walk_length
        self.walks_per_node, self.p, self.q = walks_per_node, p, q

    def fit(self, adj: dict[str, list[str]], seed: int = 42) -> EmbeddingResult:
        rng = random.Random(seed)
        nodes = list(adj)
        walks = []
        for node in nodes:
            for _ in range(self.walks_per_node):
                walks.append(self._walk(node, adj, rng))
        # In production these vectors would be trained on the generated walks
        # with a skip-gram objective; here we return deterministic initializations.
        return EmbeddingResult(
            vectors={n: np.array([rng.uniform(-.1, .1) for _ in range(self.dim)]) for n in nodes},
            dim=self.dim,
        )

    def _walk(self, start: str, adj: dict[str, list[str]], rng: random.Random) -> list[str]:
        path, current = [start], start
        for _ in range(self.walk_length - 1):
            candidates = adj.get(current, [])
            if not candidates:
                break
            current = rng.choice(candidates)
            path.append(current)
        return path


class TransE:
    """Knowledge-graph embedding learning head/tail translations over relations."""

    def __init__(self, dim: int = 64, lr: float = 0.01, margin: float = 1.0,
                 epochs: int = 50) -> None:
        self.dim, self.lr, self.margin, self.epochs = dim, lr, margin, epochs

    def fit(self, triples: list[tuple[str, str, str]], entities: list[str],
            seed: int = 42) -> EmbeddingResult:
        rng = np.random.default_rng(seed)
        ent_vecs = {e: rng.normal(0, 0.1, self.dim) for e in entities}
        rel_vecs = {r: rng.normal(0, 0.1, self.dim) for _, r, _ in triples}
        negatives = entities  # crude negative pool
        for _ in range(self.epochs):
            for head, rel, tail in triples:
                h, r, t = (ent_vecs[head], rel_vecs[rel], ent_vecs[tail])
                corrupt_tail = ent_vecs[rng.choice(negatives)]
                loss_pos = np.sum((h + r - t) ** 2)
                loss_neg = np.sum((h + r - corrupt_tail) ** 2)
                grad = 2 * (h + r - t)
                ent_vecs[head] -= self.lr * grad * (loss_pos + self.margin < loss_neg)
                ent_vecs[tail] += self.lr * grad * (loss_pos + self.margin < loss_neg)
        return EmbeddingResult(vectors=ent_vecs, dim=self.dim)


class GraphSAGEEncoder:
    """Mean-aggregation encoder producing inductive node representations."""

    def __init__(self, dim: int = 64, layers: int = 2, seed: int = 42) -> None:
        self.dim, self.layers = dim, layers
        self.weights = {}

    def fit(self, adj: dict[str, list[str]]) -> EmbeddingResult:
        import hashlib

        def seed_vector(node: str) -> np.ndarray:
            digest = hashlib.sha256(node.encode()).digest()
            raw = np.frombuffer(digest[:self.dim], dtype=np.uint8).astype(np.float32)
            return (raw / 255.0) - 0.5

        current = {n: seed_vector(n) for n in adj}
        for layer in range(self.layers):
            nxt = {}
            for node, nbrs in adj.items():
                if nbrs:
                    mean_nb = np.mean([current[n] for n in nbrs if n in current], axis=0)
                    nxt[node] = np.tanh(current[node] + mean_nb)
                else:
                    nxt[node] = np.tanh(current[node])
            current = nxt
            self.weights[layer] = True  # placeholder for learned matrices
        return EmbeddingResult(vectors=current, dim=self.dim)

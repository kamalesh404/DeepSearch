"""Knowledge graph construction: entity extraction, relation mapping, storage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ENTITY_PATTERNS = [
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b"), "PERSON_OR_ORG"),
    (re.compile(r"\b([A-Z][A-Za-z]*Search)\b"), "PRODUCT"),
    (re.compile(r"\b(\d{4})\b"), "YEAR"),
]

RELATION_VERBS = re.compile(
    r"\b(is|was|are|uses|created|founded|acquired|partners with|part of|based in)\b", re.I
)


@dataclass
class Entity:
    """A named node in the knowledge graph."""

    name: str
    type: str
    mentions: int = 1


@dataclass
class Relation:
    """A typed directed edge between two entities."""

    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0


@dataclass
class KnowledgeGraph:
    """In-memory triple store backed by a JSON file on disk."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    def add_entity(self, name: str, etype: str) -> None:
        if name in self.entities:
            self.entities[name].mentions += 1
        else:
            self.entities[name] = Entity(name=name, type=etype)

    def add_relation(self, rel: Relation) -> None:
        self.relations.append(rel)

    def save(self, path: str | Path) -> None:
        payload = {
            "entities": [vars(e) for e in self.entities.values()],
            "relations": [vars(r) for r in self.relations],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        for item in raw.get("entities", []):
            graph.entities[item["name"]] = Entity(**item)
        for item in raw.get("relations", []):
            graph.relations.append(Relation(**item))
        return graph


class KnowledgeGraphBuilder:
    """Pipeline turning raw documents into entities and subject-verb-object triples."""

    def build(self, documents: list[str]) -> KnowledgeGraph:
        """Extract entities and co-occurrence relations from every document."""
        graph = KnowledgeGraph()
        for text in documents:
            found = self._extract_entities(text)
            for name, etype in found.items():
                graph.add_entity(name, etype)
            graph.relations.extend(self._extract_relations(text, list(found)))
        return graph

    def _extract_entities(self, text: str) -> dict[str, str]:
        """Apply regex patterns to surface typed entity candidates."""
        found: dict[str, str] = {}
        for pattern, label in ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                found.setdefault(match.group(1), label)
        return found

    def _extract_relations(self, text: str, entities: list[str], max_rels: int = 20) -> list[Relation]:
        """Pair entities that appear in the same sentence as a relation verb."""
        relations: list[Relation] = []
        for sentence in re.split(r"[.!?]\s+", text):
            if not RELATION_VERBS.search(sentence):
                continue
            present = [e for e in entities if e in sentence]
            for i in range(len(present) - 1):
                verb = RELATION_VERBS.search(sentence)
                relations.append(Relation(
                    subject=present[i],
                    predicate=verb.group(1).lower() if verb else "related_to",
                    obj=present[i + 1],
                    confidence=0.6,
                ))
                if len(relations) >= max_rels:
                    return relations
        return relations

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


@dataclass
class EntityCandidate:
    name: str
    node_type: str
    confidence: float = 0.5


class GraphMemory:
    def __init__(self, similarity_threshold: float = 0.88) -> None:
        self.graph = nx.MultiDiGraph()
        self.similarity_threshold = similarity_threshold
        self._index_by_type: dict[str, list[str]] = {}
        self._node_counter = 0

    def ensure_user_node(self, user_id: str) -> str:
        node_id = f"user::{_normalize_name(user_id)}"
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                node_type="User",
                content=user_id,
                created_at=_now_iso(),
                last_seen=_now_iso(),
            )
        return node_id

    def _new_node_id(self, node_type: str) -> str:
        self._node_counter += 1
        prefix = _normalize_name(node_type).replace(" ", "_")
        return f"{prefix}::{self._node_counter}"

    def _find_existing_node(self, candidate: EntityCandidate) -> str | None:
        key = candidate.node_type.strip().lower()
        normalized_target = _normalize_name(candidate.name)
        for node_id in self._index_by_type.get(key, []):
            node_name = str(self.graph.nodes[node_id].get("content", ""))
            score = SequenceMatcher(None, normalized_target, _normalize_name(node_name)).ratio()
            if score >= self.similarity_threshold:
                return node_id
        return None

    def link_or_create_entity(self, candidate: EntityCandidate) -> str:
        existing_node = self._find_existing_node(candidate)
        if existing_node:
            self.graph.nodes[existing_node]["last_seen"] = _now_iso()
            self.graph.nodes[existing_node]["mention_count"] = int(
                self.graph.nodes[existing_node].get("mention_count", 1)
            ) + 1
            return existing_node

        node_id = self._new_node_id(candidate.node_type)
        self.graph.add_node(
            node_id,
            node_type=candidate.node_type,
            content=candidate.name,
            confidence=candidate.confidence,
            created_at=_now_iso(),
            last_seen=_now_iso(),
            mention_count=1,
        )
        key = candidate.node_type.strip().lower()
        self._index_by_type.setdefault(key, []).append(node_id)
        return node_id

    def add_relation(
        self,
        source_node: str,
        target_node: str,
        relation_type: str,
        evidence_turn_id: str,
    ) -> None:
        self.graph.add_edge(
            source_node,
            target_node,
            relation_type=relation_type,
            evidence_turn_id=evidence_turn_id,
            timestamp_utc=_now_iso(),
        )

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes.append({"node_id": node_id, **attrs})

        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, **attrs})

        payload = {"nodes": nodes, "edges": edges}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def summary(self) -> dict[str, int]:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}


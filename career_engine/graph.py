from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


@dataclass
class CareerSignal:
    content: str
    node_type: str
    confidence: float = 0.7


class CareerGraphMemory:
    def __init__(self, similarity_threshold: float = 0.86) -> None:
        self.graph = nx.MultiDiGraph()
        self.similarity_threshold = similarity_threshold
        self._index_by_type: dict[str, list[str]] = {}
        self._node_counter = 0

    def ensure_user_node(self, user_id: str) -> str:
        node_id = f"user::{_normalize(user_id)}"
        now = _now_iso()
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                node_type="user",
                content=user_id,
                confidence=1.0,
                mention_count=1,
                recency_weight=1.0,
                source_turns=[],
                created_at=now,
                last_seen=now,
            )
        return node_id

    def link_or_create_signal(self, signal: CareerSignal, source_turn: str) -> str:
        existing = self._find_existing(signal)
        now = _now_iso()
        if existing:
            attrs = self.graph.nodes[existing]
            attrs["last_seen"] = now
            attrs["mention_count"] = int(attrs.get("mention_count", 1)) + 1
            attrs["confidence"] = max(float(attrs.get("confidence", 0.7)), signal.confidence)
            turns = list(attrs.get("source_turns", []))
            if source_turn not in turns:
                turns.append(source_turn)
            attrs["source_turns"] = turns
            attrs["recency_weight"] = 1.0
            return existing

        self._node_counter += 1
        node_type = signal.node_type.strip().lower()
        node_id = f"{node_type}::{self._node_counter}"
        self.graph.add_node(
            node_id,
            node_type=node_type,
            content=signal.content,
            confidence=max(0.0, min(1.0, signal.confidence)),
            mention_count=1,
            recency_weight=1.0,
            source_turns=[source_turn],
            created_at=now,
            last_seen=now,
        )
        self._index_by_type.setdefault(node_type, []).append(node_id)
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
            relation_type=relation_type.upper(),
            evidence_turn_id=evidence_turn_id,
            timestamp_utc=_now_iso(),
        )

    def _find_existing(self, signal: CareerSignal) -> str | None:
        node_type = signal.node_type.strip().lower()
        target = _normalize(signal.content)
        for node_id in self._index_by_type.get(node_type, []):
            content = str(self.graph.nodes[node_id].get("content", ""))
            score = SequenceMatcher(None, target, _normalize(content)).ratio()
            if score >= self.similarity_threshold:
                return node_id
        return None

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"node_id": node_id, **attrs} for node_id, attrs in self.graph.nodes(data=True)],
            "edges": [
                {"source": source, "target": target, **attrs}
                for source, target, attrs in self.graph.edges(data=True)
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def summary(self) -> dict[str, int | dict[str, int]]:
        by_type: dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            node_type = str(attrs.get("node_type", "unknown"))
            by_type[node_type] = by_type.get(node_type, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "nodes_by_type": by_type,
        }

    def signal_nodes(self) -> list[dict]:
        return [
            {"node_id": node_id, **attrs}
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") != "user"
        ]


from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from .graph_memory import GraphMemory


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", _normalize(text))
    return [tok for tok in cleaned.split() if tok]


def _keyword_overlap_score(query_tokens: list[str], candidate: str) -> float:
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokenize(candidate))
    if not candidate_tokens:
        return 0.0
    overlap = len(set(query_tokens) & candidate_tokens) / max(len(query_tokens), 1)
    return overlap


def _similarity_score(a: str, b: str) -> float:
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _recency_boost(last_seen: str, halflife_days: float = 30.0) -> float:
    try:
        dt = datetime.fromisoformat(last_seen)
    except Exception:
        return 1.0
    now = datetime.now(timezone.utc)
    delta_days = max((now - dt).total_seconds() / 86400.0, 0.0)
    decay = 0.5 ** (delta_days / max(halflife_days, 0.1))
    return 1.0 + decay  # recent nodes get a small bump between 1.0 and 2.0


@dataclass
class RetrievedNode:
    node_id: str
    content: str
    node_type: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    nodes: list[RetrievedNode]
    edges: list[dict[str, Any]]
    justification: str
    latency_ms: float


class RetrievalEngine:
    """Lightweight keyword/similarity retriever over GraphMemory."""

    def __init__(
        self,
        graph_memory: GraphMemory,
        recency_halflife_days: float = 30.0,
        default_top_k: int = 5,
    ) -> None:
        self.graph_memory = graph_memory
        self.recency_halflife_days = recency_halflife_days
        self.default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        allowed_types: set[str] | None = None,
        include_edges: bool = True,
    ) -> RetrievalResult:
        started = time.time()
        q_tokens = _tokenize(query)
        scores: list[RetrievedNode] = []
        for node_id, attrs in self.graph_memory.graph.nodes(data=True):
            node_type = str(attrs.get("node_type", "Entity"))
            if allowed_types and node_type not in allowed_types:
                continue
            content = str(attrs.get("content", ""))
            if not content:
                continue
            kw_score = _keyword_overlap_score(q_tokens, content)
            sim_score = _similarity_score(query, content)
            recency = _recency_boost(str(attrs.get("last_seen", "")), self.recency_halflife_days)
            mention_count = float(attrs.get("mention_count", 1))
            score = (0.6 * kw_score + 0.4 * sim_score) * recency + math.log1p(mention_count)
            scores.append(
                RetrievedNode(
                    node_id=node_id,
                    content=content,
                    node_type=node_type,
                    score=score,
                    metadata={
                        "last_seen": attrs.get("last_seen"),
                        "created_at": attrs.get("created_at"),
                        "mention_count": mention_count,
                        "confidence": attrs.get("confidence"),
                    },
                )
            )

        final_k = top_k or self.default_top_k
        ranked = sorted(scores, key=lambda n: n.score, reverse=True)[:final_k]
        edge_payload: list[dict[str, Any]] = []
        if include_edges and ranked:
            seed_ids = [node.node_id for node in ranked]
            edge_payload = self.graph_memory.edges_for_nodes(seed_ids)

        latency_ms = (time.time() - started) * 1000.0
        justification = (
            f"Keyword/semantic match with recency and mention_count; query='{query.strip()}'"
        )
        return RetrievalResult(nodes=ranked, edges=edge_payload, justification=justification, latency_ms=latency_ms)

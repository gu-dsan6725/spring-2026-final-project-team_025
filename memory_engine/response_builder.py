from __future__ import annotations

import os
from typing import Any

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

from .retrieval import RetrievalResult


class ResponseBuilder:
    """Turns retrieval results into a concise textual summary.

    This is a deterministic fallback for demoing personalization without an LLM call.
    """

    def build_summary(self, retrieval: RetrievalResult) -> str:
        lines: list[str] = []
        lines.append("Memory summary based on your query:")
        for idx, node in enumerate(retrieval.nodes, start=1):
            meta = node.metadata or {}
            mention = int(meta.get("mention_count", 1))
            lines.append(f"{idx}. [{node.node_type}] {node.content} (score={node.score:.2f}, mentions={mention})")
        if retrieval.edges:
            lines.append("Related links:")
            for edge in retrieval.edges:
                rel = edge.get("relation_type", "RELATED_TO")
                lines.append(f"- {edge['source']} --{rel}--> {edge['target']}")
        lines.append(f"Justification: {retrieval.justification}; latency={retrieval.latency_ms:.1f}ms")
        return "\n".join(lines)


class LLMResponseBuilder:
    """Optional LLM-backed responder using retrieved memory as context."""

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key and Groq else None

    def can_run(self) -> bool:
        return self.client is not None

    def build_response(self, query: str, retrieval: RetrievalResult) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("Groq client unavailable; set GROQ_API_KEY or install groq package.")
        context_lines: list[str] = []
        for node in retrieval.nodes:
            context_lines.append(f"- {node.node_type}: {node.content} (score={node.score:.2f})")
        for edge in retrieval.edges:
            rel = edge.get("relation_type", "RELATED_TO")
            context_lines.append(f"- LINK: {edge['source']} --{rel}--> {edge['target']}")
        context = "\n".join(context_lines) if context_lines else "No memory retrieved."
        prompt = (
            "You are a helpful assistant that must ground answers in the provided personal memory context. "
            "If context is empty or irrelevant, say you don't have enough personal info. Keep response concise.\n\n"
            f"User query:\n{query}\n\n"
            f"Memory context:\n{context}\n\n"
            "Answer the user using only the memory above. If nothing relevant, say so briefly."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[{"role": "system", "content": prompt}],
            )
            text = (completion.choices[0].message.content or "").strip()
            return {"model": self.model, "response": text}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {"model": self.model, "error": str(exc), "response": None}

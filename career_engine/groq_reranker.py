"""Groq-powered re-ranker (Stage 2) and learning roadmap generator (Stage 3).

Stage 2 — Re-ranking:
  O*NET cosine similarity produces a top-20 shortlist. This module asks Groq to
  re-order those candidates using full conversation context (skills, goals,
  interests, work styles extracted from all turns) and to emit a reasoning trace
  for every recommendation.

Stage 3 — Personalized Learning Roadmap:
  After gap analysis identifies missing O*NET elements, Groq generates a
  phased, personalized learning plan that references the user's existing
  strengths to build on what they already know.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore[assignment,misc]


class GroqCareerReranker:
    """Groq-based re-ranker and roadmap generator for the career pipeline."""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def _get_client(self) -> "Groq | None":
        """Lazily create Groq client so key added after import is picked up."""
        if not Groq:
            return None
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return Groq(api_key=api_key, timeout=30.0, max_retries=0)

    # ------------------------------------------------------------------
    # Stage 2: Re-ranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        extractions: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank cosine-similarity candidates using Groq and return top_k.

        Each returned item keeps all original fields (onet_code, title, score,
        component_scores) and gains:
          - ``reasoning``   — Groq's explanation referencing conversation signals
          - ``groq_rank``   — position assigned by Groq (1 = best)
        """
        client = self._get_client()
        if not client:
            for c in candidates[:top_k]:
                c.setdefault("reasoning", "Ranked by O*NET cosine similarity (Groq unavailable).")
                c.setdefault("groq_rank", candidates.index(c) + 1)
            return candidates[:top_k]

        context_summary = _build_context_summary(extractions)
        try:
            return self._rerank_with_groq(client, candidates, context_summary, top_k)
        except Exception:  # pylint: disable=broad-exception-caught
            for i, c in enumerate(candidates[:top_k]):
                c.setdefault("reasoning", "Ranked by O*NET cosine similarity (Groq re-ranking failed).")
                c.setdefault("groq_rank", i + 1)
            return candidates[:top_k]

    def _rerank_with_groq(
        self,
        client: Any,
        candidates: list[dict[str, Any]],
        context_summary: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidate_list = "\n".join(
            f"{i + 1}. {c['title']} (O*NET: {c['onet_code']}, cosine score: {c['score']:.4f})"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "You are a career counselor AI. You have two inputs:\n"
            "1. A summary of a user's career signals (skills, goals, interests, work styles)\n"
            "2. A shortlist of O*NET occupations ranked by vector cosine similarity\n\n"
            "Your job: re-rank the occupations based on the user's actual context. "
            "Choose the best matches considering not just skills but also stated goals, "
            "interests, constraints, and work style preferences.\n\n"
            f"USER CONTEXT:\n{context_summary}\n\n"
            f"CANDIDATE OCCUPATIONS (cosine similarity order):\n{candidate_list}\n\n"
            f"Return a JSON array of exactly the top {top_k} occupations in your recommended order. "
            "For each entry include the onet_code, title, and a 'reasoning' field (2-4 sentences) "
            "that explicitly references the user's skills, goals or interests from the context above "
            "to justify the ranking.\n\n"
            "Return JSON only — no markdown, no explanation outside the array:\n"
            "[\n"
            '  {"rank": 1, "title": "...", "onet_code": "...", "reasoning": "..."},\n'
            "  ...\n"
            "]"
        )
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": prompt}],
        )
        raw = (completion.choices[0].message.content or "").strip()
        reranked_raw = _parse_json_list(raw)

        # Merge Groq ordering with original cosine data
        code_to_candidate = {c["onet_code"]: c for c in candidates}
        result: list[dict[str, Any]] = []
        seen_codes: set[str] = set()

        for item in reranked_raw:
            code = str(item.get("onet_code", "")).strip()
            if code in code_to_candidate and code not in seen_codes:
                merged = dict(code_to_candidate[code])
                merged["reasoning"] = str(item.get("reasoning", ""))
                merged["groq_rank"] = int(item.get("rank", len(result) + 1))
                result.append(merged)
                seen_codes.add(code)
            if len(result) >= top_k:
                break

        # Fill remaining slots from cosine order if Groq returned too few
        for i, c in enumerate(candidates):
            if len(result) >= top_k:
                break
            if c["onet_code"] not in seen_codes:
                merged = dict(c)
                merged["reasoning"] = "Included from cosine similarity ranking (not explicitly re-ranked by Groq)."
                merged["groq_rank"] = len(result) + 1
                result.append(merged)
                seen_codes.add(c["onet_code"])

        return result

    # ------------------------------------------------------------------
    # Stage 3: Personalized Learning Roadmap
    # ------------------------------------------------------------------

    def generate_learning_roadmap(
        self,
        gap_analysis: dict[str, Any],
        extractions: list[dict[str, Any]],
        target_career: str,
    ) -> dict[str, Any]:
        """Generate a phased learning roadmap from gap analysis using Groq.

        The roadmap is personalized: Groq is given both the gap data *and* the
        user's existing strengths, so it builds on what they already know rather
        than starting from zero.
        """
        client = self._get_client()
        if not client:
            return _fallback_roadmap(gap_analysis, target_career)

        context_summary = _build_context_summary(extractions)
        try:
            return self._generate_roadmap_with_groq(client, gap_analysis, context_summary, target_career)
        except Exception:  # pylint: disable=broad-exception-caught
            return _fallback_roadmap(gap_analysis, target_career)

    def _generate_roadmap_with_groq(
        self,
        client: Any,
        gap_analysis: dict[str, Any],
        context_summary: str,
        target_career: str,
    ) -> dict[str, Any]:
        knowledge_gaps = gap_analysis.get("knowledge_gaps", [])
        skill_gaps = gap_analysis.get("skill_gaps", [])
        work_style_gaps = gap_analysis.get("work_style_gaps", [])
        gap_text = _format_gaps(knowledge_gaps, skill_gaps, work_style_gaps)

        prompt = (
            "You are a career development coach AI. Based on a user's profile and their skill gaps "
            "relative to their target career, generate a personalized learning roadmap.\n\n"
            f"TARGET CAREER: {target_career}\n\n"
            f"USER'S EXISTING STRENGTHS AND CONTEXT:\n{context_summary}\n\n"
            f"IDENTIFIED GAPS (O*NET element — gap magnitude):\n{gap_text}\n\n"
            "Generate a personalized roadmap with 3-5 concrete phases. "
            "Build on what the user already knows — explicitly reference their existing skills or "
            "projects from the context to show how each phase extends their current foundation.\n\n"
            "For each phase include:\n"
            "- phase number and name\n"
            "- estimated duration\n"
            "- which gaps it addresses (focus_gaps)\n"
            "- 2-3 specific actions (a course, a project, or a practice activity)\n"
            "  Each action needs: action description, type (course/project/practice), "
            "  and 'why' referencing the user's specific background\n"
            "- a measurable milestone\n\n"
            "Return JSON only — no markdown, no explanation outside the object:\n"
            "{\n"
            '  "target_career": "...",\n'
            '  "summary": "...",\n'
            '  "phases": [\n'
            '    {\n'
            '      "phase": 1,\n'
            '      "name": "...",\n'
            '      "duration": "...",\n'
            '      "focus_gaps": ["..."],\n'
            '      "actions": [\n'
            '        {"action": "...", "type": "course|project|practice", "why": "..."}\n'
            '      ],\n'
            '      "milestone": "..."\n'
            '    }\n'
            '  ]\n'
            "}"
        )
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": prompt}],
        )
        raw = (completion.choices[0].message.content or "").strip()
        parsed = _parse_json_object(raw)
        if not parsed.get("phases"):
            return _fallback_roadmap(gap_analysis, target_career)
        return parsed


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_context_summary(extractions: list[dict[str, Any]]) -> str:
    """Summarise all extracted career signals as a short text block for the LLM."""
    sections = [
        ("Skills", _collect_field(extractions, "skill"), 10),
        ("Tools", _collect_field(extractions, "tool"), 10),
        ("Knowledge areas", _collect_field(extractions, "knowledge"), 8),
        ("Career goals", _collect_field(extractions, "career_goal"), 5),
        ("Interests", _collect_field(extractions, "interest"), 5),
        ("Work styles", _collect_field(extractions, "behavioral_trait"), 6),
        ("Projects", _collect_field(extractions, "project"), 5),
    ]
    lines = [
        f"{label}: {', '.join(items[:limit])}"
        for label, items, limit in sections
        if items
    ]
    return "\n".join(lines) if lines else "No career signals extracted."


def _collect_field(extractions: list[dict[str, Any]], field: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for row in extractions:
        for item in row.get(field, []):
            name = str(item.get("name", "") if isinstance(item, dict) else item).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                result.append(name)
    return result


def _format_gaps(
    knowledge_gaps: list[dict[str, Any]],
    skill_gaps: list[dict[str, Any]],
    work_style_gaps: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    if knowledge_gaps:
        lines.append("Knowledge gaps:")
        for g in knowledge_gaps:
            lines.append(f"  - {g['element']} (gap: {g['gap']:.2f})")
    if skill_gaps:
        lines.append("Skill gaps:")
        for g in skill_gaps:
            lines.append(f"  - {g['element']} (gap: {g['gap']:.2f})")
    if work_style_gaps:
        lines.append("Work style gaps:")
        for g in work_style_gaps:
            lines.append(f"  - {g['element']} (gap: {g['gap']:.2f})")
    return "\n".join(lines) if lines else "No significant gaps identified."


def _fallback_roadmap(gap_analysis: dict[str, Any], target_career: str) -> dict[str, Any]:
    priority_gaps = (
        gap_analysis.get("skill_gaps", [])[:3]
        + gap_analysis.get("knowledge_gaps", [])[:2]
    )
    phases = [
        {
            "phase": i + 1,
            "name": f"Build {g['element']}",
            "duration": "4-6 weeks",
            "focus_gaps": [g["element"]],
            "actions": [
                {
                    "action": f"Study {g['element']} fundamentals",
                    "type": "course",
                    "why": f"Closes a gap of {g['gap']:.2f} points toward {target_career}",
                }
            ],
            "milestone": f"Demonstrate proficiency in {g['element']}",
        }
        for i, g in enumerate(priority_gaps[:3])
    ]
    return {
        "target_career": target_career,
        "summary": (
            f"Roadmap to {target_career} based on identified skill gaps "
            "(Groq unavailable; generated from gap analysis only)."
        ),
        "phases": phases,
    }


def _parse_json_list(raw: str) -> list[Any]:
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

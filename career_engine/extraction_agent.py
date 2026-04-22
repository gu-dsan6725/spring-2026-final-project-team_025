from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .graph import CareerGraphMemory, CareerSignal

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


@dataclass
class CareerExtractionOutput:
    knowledge_areas: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    projects: list[dict[str, Any]]
    courses: list[dict[str, Any]]
    career_goals: list[dict[str, Any]]
    interests: list[dict[str, Any]]
    work_styles: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    implicit_signals: list[dict[str, Any]]
    relations: list[dict[str, Any]]


class CareerExtractionAgent:
    def __init__(
        self,
        graph_memory: CareerGraphMemory,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        use_llm: bool = True,
    ) -> None:
        self.graph_memory = graph_memory
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = (
            Groq(api_key=self.api_key, timeout=10.0, max_retries=0)
            if use_llm and self.api_key and Groq
            else None
        )
        self._groq_failures = 0
        self._max_groq_failures = 1000

    def process_turn(self, turn: dict[str, Any]) -> CareerExtractionOutput:
        extracted = self.extract(turn["text"])
        if has_career_output(extracted):
            self._write_to_graph(turn, extracted)
        return extracted

    def extract(self, text: str) -> CareerExtractionOutput:
        if not _looks_career_relevant(text):
            return empty_output()
        if self.client:
            try:
                return sanitize_output(self._extract_with_groq(text))
            except Exception:  # pylint: disable=broad-exception-caught
                self._groq_failures += 1
                if self._groq_failures >= self._max_groq_failures:
                    self.client = None
        return sanitize_output(_extract_with_rules(text))

    def _extract_with_groq(self, text: str) -> CareerExtractionOutput:
        prompt = (
            "You extract career-development signals from natural user conversation. "
            "Extract explicit and implicit evidence about the user's coursework, projects, skills, tools, "
            "knowledge areas, interests, career goals, constraints, and behavioral work styles.\n\n"
            f'USER message: "{text}"\n\n'
            "Rules:\n"
            "- Capture both explicit signals and strongly implied latent signals.\n"
            "- Work styles are behavioral traits such as attention to detail, analytical thinking, persistence, independence.\n"
            "- Tools are concrete technologies such as Python, SQL, PyTorch, Excel, FastAPI.\n"
            "- Knowledge areas are broader domains such as data analysis, machine learning, finance, statistics.\n"
            "- Do not fabricate career goals that are not stated or strongly implied.\n"
            "- Return JSON only, with confidence from 0.0 to 1.0.\n\n"
            "{\n"
            '  "knowledge_areas": [{"name": str, "confidence": float}],\n'
            '  "skills": [{"name": str, "confidence": float}],\n'
            '  "tools": [{"name": str, "confidence": float}],\n'
            '  "projects": [{"name": str, "confidence": float}],\n'
            '  "courses": [{"name": str, "confidence": float}],\n'
            '  "career_goals": [{"name": str, "confidence": float}],\n'
            '  "interests": [{"name": str, "confidence": float}],\n'
            '  "work_styles": [{"name": str, "confidence": float}],\n'
            '  "constraints": [{"name": str, "confidence": float}],\n'
            '  "implicit_signals": [{"name": str, "confidence": float}],\n'
            '  "relations": [{"source": "USER", "target": str, "relation_type": str}]\n'
            "}"
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": prompt}],
        )
        parsed = _parse_json_response((completion.choices[0].message.content or "").strip())
        self._groq_failures = 0
        return CareerExtractionOutput(
            knowledge_areas=parsed.get("knowledge_areas", []),
            skills=parsed.get("skills", []),
            tools=parsed.get("tools", []),
            projects=parsed.get("projects", []),
            courses=parsed.get("courses", []),
            career_goals=parsed.get("career_goals", []),
            interests=parsed.get("interests", []),
            work_styles=parsed.get("work_styles", []),
            constraints=parsed.get("constraints", []),
            implicit_signals=parsed.get("implicit_signals", []),
            relations=parsed.get("relations", []),
        )

    def _write_to_graph(self, turn: dict[str, Any], extraction: CareerExtractionOutput) -> None:
        user_node = self.graph_memory.ensure_user_node(turn["dialogue_id"])
        node_lookup: dict[str, str] = {}
        for node_type, items in extraction_sections(extraction).items():
            for item in items:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                signal = CareerSignal(
                    content=name,
                    node_type=node_type,
                    confidence=float(item.get("confidence", 0.7)),
                )
                node_id = self.graph_memory.link_or_create_signal(signal, turn["turn_id"])
                node_lookup[name.lower()] = node_id
                self.graph_memory.add_relation(
                    source_node=user_node,
                    target_node=node_id,
                    relation_type=_default_relation_for(node_type),
                    evidence_turn_id=turn["turn_id"],
                )

        for relation in extraction.relations:
            target = str(relation.get("target", "")).strip().lower()
            target_node = node_lookup.get(target)
            if not target_node:
                continue
            self.graph_memory.add_relation(
                source_node=user_node,
                target_node=target_node,
                relation_type=str(relation.get("relation_type", "RELATED_TO")).upper(),
                evidence_turn_id=turn["turn_id"],
            )


def extraction_sections(output: CareerExtractionOutput) -> dict[str, list[dict[str, Any]]]:
    return {
        "knowledge": output.knowledge_areas,
        "skill": output.skills,
        "tool": output.tools,
        "project": output.projects,
        "course": output.courses,
        "career_goal": output.career_goals,
        "interest": output.interests,
        "behavioral_trait": output.work_styles,
        "constraint": output.constraints,
        "implicit_signal": output.implicit_signals,
    }


def has_career_output(output: CareerExtractionOutput) -> bool:
    return any(bool(items) for items in extraction_sections(output).values()) or bool(output.relations)


def empty_output() -> CareerExtractionOutput:
    return CareerExtractionOutput([], [], [], [], [], [], [], [], [], [], [])


def sanitize_output(output: CareerExtractionOutput) -> CareerExtractionOutput:
    for _, items in extraction_sections(output).items():
        _sanitize_items_in_place(items)
    output.relations = [
        {
            "source": str(rel.get("source", "USER") or "USER"),
            "target": str(rel.get("target", "")).strip(),
            "relation_type": str(rel.get("relation_type", "RELATED_TO")).upper(),
        }
        for rel in output.relations
        if str(rel.get("target", "")).strip()
    ]
    return output


def _sanitize_items_in_place(items: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            name = item
            confidence = 0.7
        else:
            name = str(item.get("name", "") or item.get("description", "")).strip()
            confidence = float(item.get("confidence", 0.7))
        name = _clean_span(name)
        key = name.lower()
        if not name or key in seen or len(name) < 2 or len(name) > 90:
            continue
        seen.add(key)
        cleaned.append({"name": name, "confidence": max(0.0, min(1.0, confidence))})
    items[:] = cleaned


def _extract_with_rules(text: str) -> CareerExtractionOutput:
    normalized = _normalize_text(text)
    lowered = normalized.lower()

    tools = _keyword_hits(
        lowered,
        [
            "python",
            "sql",
            "excel",
            "tableau",
            "power bi",
            "pytorch",
            "tensorflow",
            "fastapi",
            "docker",
            "git",
            "r",
            "spark",
            "pandas",
            "scikit-learn",
            "aws",
        ],
    )
    skills = _keyword_hits(
        lowered,
        [
            "programming",
            "data cleaning",
            "model evaluation",
            "debugging",
            "optimization",
            "data visualization",
            "statistical analysis",
            "machine learning",
            "deep learning",
            "system design",
            "distributed systems",
            "api development",
            "quality control",
        ],
    )
    knowledge = _keyword_hits(
        lowered,
        [
            "computer vision",
            "data analysis",
            "machine learning",
            "deep learning",
            "statistics",
            "finance",
            "economics",
            "mathematics",
            "software engineering",
            "databases",
            "cloud computing",
        ],
    )
    work_styles = _work_style_hits(lowered)
    projects = _extract_patterns(
        normalized,
        [
            r"(?:working on|building|built|developing|developed|project is|project:)\s+([^.;!\n]+)",
            r"(?:my|our)\s+([^.;!\n]{4,70}?\s+(?:project|classifier|dashboard|app|api|model|website))",
        ],
    )
    courses = _extract_patterns(
        normalized,
        [
            r"(?:coursework|course|class)\s+(?:in|on|about)?\s*([^.;!\n]+)",
            r"(?:taking|took|studying)\s+([^.;!\n]+)",
        ],
    )
    goals = _extract_patterns(
        normalized,
        [
            r"(?:i want|i'd like|i would like|long term i want|my goal is)\s+([^.;!\n]+)",
            r"(?:become|becoming|move toward|moving toward)\s+([^.;!\n]+)",
        ],
    )
    interests = _extract_patterns(
        normalized,
        [
            r"(?:interested in|curious about|drawn to|appealing)\s+([^.;!\n]+)",
            r"(?:i like|i enjoy|i'm enjoying|i am enjoying)\s+([^.;!\n]+)",
        ],
    )
    constraints = _extract_patterns(
        normalized,
        [
            r"(?:concern is|worried about|my concern is|gap(?:s)? in)\s+([^.;!\n]+)",
            r"(?:not sure whether|unsure whether)\s+([^.;!\n]+)",
        ],
    )

    implicit_signals = sorted(set(work_styles + _implicit_from_text(lowered)))
    relations = []
    for item in goals:
        relations.append({"source": "USER", "target": item, "relation_type": "HAS_CAREER_GOAL"})
    for item in projects:
        relations.append({"source": "USER", "target": item, "relation_type": "WORKS_ON"})
    for item in tools:
        relations.append({"source": "USER", "target": item, "relation_type": "USES_TOOL"})
    for item in skills:
        relations.append({"source": "USER", "target": item, "relation_type": "HAS_SKILL_SIGNAL"})

    return CareerExtractionOutput(
        knowledge_areas=_items(knowledge),
        skills=_items(skills),
        tools=_items(tools),
        projects=_items(projects),
        courses=_items(courses),
        career_goals=_items(goals),
        interests=_items(interests),
        work_styles=_items(work_styles),
        constraints=_items(constraints),
        implicit_signals=_items(implicit_signals),
        relations=relations,
    )


def _looks_career_relevant(text: str) -> bool:
    lowered = _normalize_text(text).lower()
    signals = [
        "career",
        "job",
        "role",
        "project",
        "course",
        "class",
        "skill",
        "tool",
        "python",
        "sql",
        "data",
        "model",
        "engineer",
        "analyst",
        "scientist",
        "learn",
        "study",
        "work on",
        "working on",
        "interested in",
        "i want",
        "my goal",
        "gap",
        "concern",
    ]
    return any(signal in lowered for signal in signals)


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    hits = []
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            hits.append(keyword)
    return hits


def _work_style_hits(text: str) -> list[str]:
    mapping = {
        "attention to detail": ["attention to detail", "detail-oriented", "edge cases", "checking"],
        "analytical thinking": ["analytical", "analyze", "analysis", "problem solving", "reasoning"],
        "persistence": ["persistent", "keep trying", "keep seeing", "going back", "refine"],
        "dependability": ["reliable", "make it more reliable", "responsible"],
        "innovation": ["creative", "new idea", "prototype", "experiment"],
        "independence": ["independent", "on my own", "self-directed"],
    }
    hits = []
    for label, phrases in mapping.items():
        if any(phrase in text for phrase in phrases):
            hits.append(label)
    return hits


def _implicit_from_text(text: str) -> list[str]:
    signals = []
    if any(word in text for word in ["debug", "edge cases", "quality", "reliable"]):
        signals.append("quality-focused")
    if any(word in text for word in ["optimize", "optimization", "refine"]):
        signals.append("optimization-oriented")
    if any(word in text for word in ["deploy", "api", "production"]):
        signals.append("production-oriented")
    return signals


def _extract_patterns(text: str, patterns: list[str]) -> list[str]:
    out = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            out.append(_clean_span(match))
    return _dedupe([item for item in out if item])


def _items(names: list[str], confidence: float = 0.75) -> list[dict[str, Any]]:
    return [{"name": item, "confidence": confidence} for item in _dedupe(names)]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _default_relation_for(node_type: str) -> str:
    return {
        "knowledge": "HAS_KNOWLEDGE_SIGNAL",
        "skill": "HAS_SKILL_SIGNAL",
        "tool": "USES_TOOL",
        "project": "WORKS_ON",
        "course": "TOOK_OR_DISCUSS_COURSE",
        "career_goal": "HAS_CAREER_GOAL",
        "interest": "HAS_INTEREST",
        "behavioral_trait": "SHOWS_WORK_STYLE",
        "constraint": "HAS_CONSTRAINT",
        "implicit_signal": "HAS_IMPLICIT_SIGNAL",
    }.get(node_type, "RELATED_TO")


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("’", "'").split())


def _clean_span(text: str) -> str:
    text = _normalize_text(str(text))
    text = re.sub(r"^(to|in|on|about|that|whether)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" .,:;!?\"'")


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(text)
        return _coerce_json_object(parsed)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed, _ = decoder.raw_decode(text[start : end + 1])
            return _coerce_json_object(parsed)
        raise


def _coerce_json_object(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"implicit_signals": parsed}
    return {}

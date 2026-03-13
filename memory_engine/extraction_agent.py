from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .graph_memory import EntityCandidate, GraphMemory

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


@dataclass
class ExtractionOutput:
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    preferences: list[str]
    constraints: list[str]
    goals: list[str]
    projects: list[str]
    tools: list[str]


class MemoryExtractionAgent:
    def __init__(self, graph_memory: GraphMemory, model: str = "llama-3.1-8b-instant") -> None:
        self.graph_memory = graph_memory
        self.model = model
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = (
            Groq(api_key=self.api_key, timeout=8.0, max_retries=0) if self.api_key and Groq else None
        )
        self._groq_failures = 0
        self._max_groq_failures = 3

    def process_turn(self, turn: dict[str, Any]) -> ExtractionOutput:
        extracted = self._extract(turn["text"])
        self._write_to_graph(turn, extracted)
        return extracted

    def _extract(self, text: str) -> ExtractionOutput:
        if self.client:
            try:
                return self._extract_with_groq(text)
            except Exception:
                self._groq_failures += 1
                if self._groq_failures >= self._max_groq_failures:
                    self.client = None
                return self._extract_with_rules(text)
        return self._extract_with_rules(text)

    def _extract_with_groq(self, text: str) -> ExtractionOutput:
        prompt = (
            "Extract memory facts from one user utterance.\n"
            "Return strict JSON with keys: entities, relations, preferences, constraints, goals, projects, tools.\n"
            "entities item: {name, type, confidence}.\n"
            "relations item: {source, target, relation_type}.\n"
            "Use concise spans copied from text.\n"
            "Return JSON only. No markdown."
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        parsed = _parse_json_response(raw)
        self._groq_failures = 0
        return ExtractionOutput(
            entities=parsed.get("entities", []),
            relations=parsed.get("relations", []),
            preferences=parsed.get("preferences", []),
            constraints=parsed.get("constraints", []),
            goals=parsed.get("goals", []),
            projects=parsed.get("projects", []),
            tools=parsed.get("tools", []),
        )

    def _extract_with_rules(self, text: str) -> ExtractionOutput:
        lowered = text.lower()
        preferences = _find_by_patterns(
            text,
            [
                r"\bi prefer ([^.;!\n]+)",
                r"\bi like ([^.;!\n]+)",
                r"\bplease (?:be|keep) ([^.;!\n]+)",
            ],
        )
        constraints = _find_by_patterns(
            text,
            [
                r"\bi can(?:not|'t) ([^.;!\n]+)",
                r"\bi have to ([^.;!\n]+)",
                r"\bdeadline(?: is|:)? ([^.;!\n]+)",
            ],
        )
        goals = _find_by_patterns(
            text,
            [
                r"\bmy goal is to ([^.;!\n]+)",
                r"\bi want to ([^.;!\n]+)",
                r"\bi need to ([^.;!\n]+)",
            ],
        )
        projects = _find_by_patterns(
            text,
            [
                r"\bi(?:'m| am) working on ([^.;!\n]+)",
                r"\bmy project(?: is|:)? ([^.;!\n]+)",
                r"\bfor my (?:class|course|research) ([^.;!\n]+)",
            ],
        )

        tool_keywords = [
            "python",
            "neo4j",
            "networkx",
            "pandas",
            "sql",
            "excel",
            "docker",
            "langchain",
            "llamaindex",
            "notion",
        ]
        tools = [tool for tool in tool_keywords if tool in lowered]

        entities: list[dict[str, Any]] = []
        for pref in preferences:
            entities.append({"name": pref, "type": "Preference", "confidence": 0.7})
        for cst in constraints:
            entities.append({"name": cst, "type": "Constraint", "confidence": 0.7})
        for goal in goals:
            entities.append({"name": goal, "type": "Goal", "confidence": 0.75})
        for proj in projects:
            entities.append({"name": proj, "type": "Project", "confidence": 0.75})
        for tool in tools:
            entities.append({"name": tool, "type": "Tool", "confidence": 0.8})

        for phrase in _capitalized_phrases(text):
            entities.append({"name": phrase, "type": "Entity", "confidence": 0.55})

        relations: list[dict[str, Any]] = []
        for pref in preferences:
            relations.append({"source": "user", "target": pref, "relation_type": "PREFERS"})
        for cst in constraints:
            relations.append({"source": "user", "target": cst, "relation_type": "CONSTRAINED_BY"})
        for goal in goals:
            relations.append({"source": "user", "target": goal, "relation_type": "HAS_GOAL"})
        for proj in projects:
            relations.append({"source": "user", "target": proj, "relation_type": "WORKS_ON"})
        for tool in tools:
            relations.append({"source": "user", "target": tool, "relation_type": "USES_TOOL"})

        return ExtractionOutput(
            entities=_dedupe_entities(entities),
            relations=relations,
            preferences=preferences,
            constraints=constraints,
            goals=goals,
            projects=projects,
            tools=tools,
        )

    def _write_to_graph(self, turn: dict[str, Any], extraction: ExtractionOutput) -> None:
        user_node = self.graph_memory.ensure_user_node(turn["dialogue_id"])
        entity_nodes: dict[str, str] = {}

        for ent in extraction.entities:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            node_type = str(ent.get("type", "Entity"))
            candidate = EntityCandidate(
                name=name,
                node_type=node_type,
                confidence=float(ent.get("confidence", 0.5)),
            )
            node_id = self.graph_memory.link_or_create_entity(candidate)
            entity_nodes[name.lower()] = node_id
            self.graph_memory.add_relation(
                source_node=user_node,
                target_node=node_id,
                relation_type="MENTIONS",
                evidence_turn_id=turn["turn_id"],
            )

        for rel in extraction.relations:
            target_name = str(rel.get("target", "")).strip().lower()
            relation_type = str(rel.get("relation_type", "RELATED_TO")).upper()
            target_node = entity_nodes.get(target_name)
            if not target_node:
                continue
            self.graph_memory.add_relation(
                source_node=user_node,
                target_node=target_node,
                relation_type=relation_type,
                evidence_turn_id=turn["turn_id"],
            )


def _find_by_patterns(text: str, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for m in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = " ".join(m.strip().split())
            if cleaned:
                matches.append(cleaned)
    return _dedupe_text(matches)


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for ent in entities:
        name = str(ent.get("name", "")).strip()
        etype = str(ent.get("type", "Entity")).strip()
        key = (name.lower(), etype.lower())
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


def _capitalized_phrases(text: str) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    blacklist = {"I", "The", "This", "That", "And"}
    return [x for x in _dedupe_text(candidates) if x not in blacklist]


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


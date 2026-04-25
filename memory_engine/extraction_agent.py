from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
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
    preferences: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    goals: list[dict[str, Any]]
    projects: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class MemoryExtractionAgent:
    def __init__(
        self,
        graph_memory: GraphMemory,
        model: str = "llama-3.1-8b-instant",
        use_relevance_filter: bool = True,
        temperature: float = 0.0,
    ) -> None:
        self.graph_memory = graph_memory
        self.model = model
        self.use_relevance_filter = use_relevance_filter
        self.temperature = temperature
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = (
            Groq(api_key=self.api_key, timeout=8.0, max_retries=0) if self.api_key and Groq else None
        )
        self._groq_failures = 0
        # Allow more transient failures before disabling Groq; prevents early fallback to rules
        self._max_groq_failures = 10

    def process_turn(self, turn: dict[str, Any]) -> ExtractionOutput:
        extracted = self._extract(turn["text"])
        if has_memory_output(extracted):
            self._write_to_graph(turn, extracted)
        return extracted

    def _extract(self, text: str) -> ExtractionOutput:
        if self.use_relevance_filter and not _is_memory_relevant_text(text):
            return _empty_output()
        if self.client:
            try:
                return _sanitize_output(self._extract_with_groq(text))
            except Exception:  # pylint: disable=broad-exception-caught
                self._groq_failures += 1
                if self._groq_failures >= self._max_groq_failures:
                    self.client = None
                return _sanitize_output(self._extract_with_rules(text))
        return _sanitize_output(self._extract_with_rules(text))

    def _extract_with_groq(self, text: str) -> ExtractionOutput:
        prompt = (
            "You are a personal memory extractor. Your job is to extract the user's tasks/goals/projects, preferences,"
            " constraints, tools, and skills from their message. If the user gives instructions (write/make/build/create)"
            " without saying 'I', treat it as the user's project/goal.\n\n"
            f'USER message: "{text}"\n\n'
            "Rules:\n"
            "- Extract facts that reveal the user's goals, preferences, constraints, active projects, "
            "tools they use/plan to use, or skills they have/are learning.\n"
            "- Instructions like \"write/make/create/build\" count as user goals/projects even without pronouns.\n"
            "- If the user merely mentions general knowledge with no task or personal implication, return empty.\n"
            "- List EVERY named tool, service, library, language, framework, or technology individually — "
            "do not collapse multiple tools into one entity.\n"
            "- Classify entities: PERSON | ORGANIZATION | PROJECT | TOOL | SKILL | TOPIC.\n"
            "- Relations: include edges where one node is USER (e.g., USER->WORKS_ON project, USER->USES_TOOL, USER->HAS_GOAL).\n"
            "- Confidence: 1.0 explicit, 0.7 strongly implied, 0.4 weakly implied.\n"
            "- If truly no task/personal signal, return all empty arrays. DO NOT fabricate unrelated facts.\n\n"
            "Return strict JSON only, no markdown, no explanation:\n"
            "{\n"
            '  "entities": [{"name": str, "type": str, "confidence": float}],\n'
            '  "relations": [{"source": str, "target": str, "relation_type": str}],\n'
            '  "preferences": [{"description": str, "confidence": float}],\n'
            '  "constraints": [{"description": str, "confidence": float}],\n'
            '  "goals": [{"description": str, "confidence": float}],\n'
            '  "projects": [{"name": str, "description": str, "confidence": float}],\n'
            '  "tools": [{"name": str, "use_case": str, "confidence": float}]\n'
            "}"
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": prompt},
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
        normalized_text = _normalize_rule_text(text)
        lowered = normalized_text.lower()
        if _looks_like_reference_or_content_dump(text) and not _has_strong_personal_memory_signal(normalized_text):
            return _empty_output()
        preferences = _find_by_patterns(
            normalized_text,
            [
                r"\bi prefer ([^.;!\n]+)",
                r"\bi like ([^.;!\n]+)",
                r"\bplease (?:be|keep) ([^.;!\n]+)",
            ],
        )
        constraints = _find_by_patterns(
            normalized_text,
            [
                r"\bi can(?:not|'t) ([^.;!\n]+)",
                r"\bi have to ([^.;!\n]+)",
                r"\bdeadline(?: is|:)? ([^.;!\n]+)",
            ],
        )
        goals = _find_by_patterns(
            normalized_text,
            [
                r"\bmy goal is to ([^.;!\n]+)",
                r"\bi want to ([^.;!\n]+)",
                # "I want a/an X" — career goals often use this form
                r"\bi want (?:a|an|to be|to become) ([^.;!\n]+)",
                r"\bi need to ([^.;!\n]+)",
                r"\bi(?:'m| am) (?:hoping|planning|trying) to ([^.;!\n]+)",
                r"\bi(?:'m| am) learning ([^.;!\n]+)",
                r"\blong[- ]term (?:I |my )?(?:goal|plan|aim)[^,]*[,:]? ([^.;!\n]+)",
                r"\bi(?:'m| am) interested in (?:moving into|transitioning to|becoming) ([^.;!\n]+)",
            ],
        )
        projects = _find_by_patterns(
            normalized_text,
            [
                # Allow optional adverbs between "I'm" and the verb ("I'm currently working on")
                r"\bi(?:'m| am) (?:\w+ )?working on ([^.;!\n]+)",
                r"\bi(?:'m| am) (?:\w+ )?building ([^.;!\n]+)",
                r"\bi(?:'m| am) (?:\w+ )?creating ([^.;!\n]+)",
                r"\bi(?:'m| am) (?:\w+ )?writing ([^.;!\n]+)",
                r"\bi(?:'m| am) (?:\w+ )?developing ([^.;!\n]+)",
                r"\bi(?:'m| am) (?:\w+ )?learning ([^.;!\n]+)",
                r"\bworking on (?:a |an |the )?([^.;!\n]+)",
                r"\bmy project(?: is|:)? ([^.;!\n]+)",
                r"\bfor my (?:class|course|research) ([^.;!\n]+)",
                r"\bi (?:built|developed|created|implemented) ([^.;!\n]+)",
            ],
        )

        tool_keywords = [
            # Languages
            "python", "javascript", "typescript", "java", "c++", "rust", "go", "scala", "r",
            # ML / AI
            "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "huggingface",
            "transformers", "langchain", "llamaindex", "openai", "groq",
            # Data
            "pandas", "numpy", "spark", "hadoop", "dbt", "airflow",
            # Databases
            "sql", "postgresql", "mysql", "mongodb", "neo4j", "redis", "elasticsearch",
            # DevOps / infra
            "docker", "kubernetes", "aws", "gcp", "azure", "terraform",
            # Web / APIs
            "fastapi", "flask", "django", "node", "react", "vue", "nextjs",
            # Tools
            "git", "networkx", "excel", "notion", "tableau", "powerbi",
        ]
        tools = [tool for tool in tool_keywords if re.search(rf"\b{re.escape(tool)}\b", lowered)]

        preference_objs = [{"description": pref, "confidence": 0.7} for pref in preferences]
        constraint_objs = [{"description": cst, "confidence": 0.7} for cst in constraints]
        goal_objs = [{"description": goal, "confidence": 0.7} for goal in goals]
        project_objs = [
            {"name": proj, "description": proj, "confidence": 0.7}
            for proj in projects
        ]
        tool_objs = [
            {"name": tool, "use_case": "mentioned in user message", "confidence": 0.7}
            for tool in tools
        ]

        entities: list[dict[str, Any]] = []
        for proj in project_objs:
            entities.append({"name": proj["name"], "type": "PROJECT", "confidence": proj["confidence"]})
        for tool in tool_objs:
            entities.append({"name": tool["name"], "type": "TOOL", "confidence": tool["confidence"]})

        relations: list[dict[str, Any]] = []
        for pref in preference_objs:
            relations.append(
                {"source": "USER", "target": pref["description"], "relation_type": "PREFERS"}
            )
        for cst in constraint_objs:
            relations.append(
                {"source": "USER", "target": cst["description"], "relation_type": "CONSTRAINED_BY"}
            )
        for goal in goal_objs:
            relations.append({"source": "USER", "target": goal["description"], "relation_type": "HAS_GOAL"})
        for proj in project_objs:
            relations.append({"source": "USER", "target": proj["name"], "relation_type": "WORKS_ON"})
        for tool in tool_objs:
            relations.append({"source": "USER", "target": tool["name"], "relation_type": "USES_TOOL"})

        return ExtractionOutput(
            entities=_dedupe_entities(entities),
            relations=relations,
            preferences=preference_objs,
            constraints=constraint_objs,
            goals=goal_objs,
            projects=project_objs,
            tools=tool_objs,
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
    blacklist = {
        "I",
        "The",
        "This",
        "That",
        "And",
        "What",
        "How",
        "Why",
        "Summarize",
        "Explain",
        "Write",
        "Create",
        "Generate",
    }
    return [x for x in _dedupe_text(candidates) if x not in blacklist]


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _empty_output() -> ExtractionOutput:
    return ExtractionOutput(
        entities=[],
        relations=[],
        preferences=[],
        constraints=[],
        goals=[],
        projects=[],
        tools=[],
    )


def has_memory_output(output: ExtractionOutput) -> bool:
    return any(
        [
            bool(output.entities),
            bool(output.relations),
            bool(output.preferences),
            bool(output.constraints),
            bool(output.goals),
            bool(output.projects),
            bool(output.tools),
        ]
    )


def _is_memory_relevant_text(text: str) -> bool:
    normalized_text = _normalize_rule_text(text)
    if _looks_like_reference_or_content_dump(text) and not _has_strong_personal_memory_signal(normalized_text):
        return False
    return _has_strong_personal_memory_signal(normalized_text)


def _sanitize_output(output: ExtractionOutput) -> ExtractionOutput:
    output.preferences = _sanitize_desc_items(output.preferences)
    output.constraints = _sanitize_desc_items(output.constraints)
    output.goals = _sanitize_desc_items(output.goals)
    output.projects = _sanitize_projects(output.projects)
    output.tools = _sanitize_tools(output.tools)

    allowed_types = {"PERSON", "ORGANIZATION", "PROJECT", "TOOL", "SKILL", "TOPIC"}
    cleaned_entities: list[dict[str, Any]] = []
    for ent in output.entities:
        name = str(ent.get("name", "")).strip()
        if not _is_good_memory_span(name):
            continue
        etype = str(ent.get("type", "TOPIC")).strip().upper()
        if etype not in allowed_types:
            etype = "TOPIC"
        cleaned_entities.append(
            {
                "name": name,
                "type": etype,
                "confidence": float(ent.get("confidence", 0.5)),
            }
        )
    output.entities = _dedupe_entities(cleaned_entities)
    output.entities.extend(_entities_from_structured_sections(output))
    output.entities = _dedupe_entities(output.entities)
    output.entities = _semantic_dedupe_entities(output.entities)

    valid_targets = {str(e["name"]).lower() for e in output.entities}
    cleaned_relations: list[dict[str, Any]] = []
    for rel in output.relations:
        source = str(rel.get("source", "USER")).strip()
        target = str(rel.get("target", "")).strip()
        if not source:
            source = "USER"
        source_lower = source.lower()
        target_lower = target.lower()
        if source_lower != "user" and target_lower != "user":
            continue
        if not target or target.lower() not in valid_targets:
            continue
        cleaned_relations.append(
            {
                "source": source,
                "target": target,
                "relation_type": str(rel.get("relation_type", "RELATED_TO")).upper(),
            }
        )
    output.relations = cleaned_relations
    return output


def _sanitize_desc_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            desc = item.strip()
            conf = 0.7
        else:
            desc = str(item.get("description", "")).strip()
            conf = float(item.get("confidence", 0.7))
        if _is_good_memory_span(desc):
            cleaned.append({"description": desc, "confidence": conf})
    return _semantic_dedupe_items(cleaned, "description")


def _sanitize_projects(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
            desc = item.strip()
            conf = 0.7
        else:
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", name)).strip()
            conf = float(item.get("confidence", 0.7))
        if _is_good_memory_span(name):
            cleaned.append({"name": name, "description": desc, "confidence": conf})
    return _semantic_dedupe_items(cleaned, "name")


def _sanitize_tools(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
            use_case = "mentioned in user message"
            conf = 0.7
        else:
            name = str(item.get("name", "")).strip()
            use_case = str(item.get("use_case", "mentioned in user message")).strip()
            conf = float(item.get("confidence", 0.7))
        if _is_good_memory_span(name):
            cleaned.append({"name": name, "use_case": use_case, "confidence": conf})
    return _semantic_dedupe_items(cleaned, "name")


def _entities_from_structured_sections(output: ExtractionOutput) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for proj in output.projects:
        entities.append({"name": proj["name"], "type": "PROJECT", "confidence": proj["confidence"]})
    for tool in output.tools:
        entities.append({"name": tool["name"], "type": "TOOL", "confidence": tool["confidence"]})
    for goal in output.goals:
        entities.append({"name": goal["description"], "type": "TOPIC", "confidence": goal["confidence"]})
    for pref in output.preferences:
        entities.append({"name": pref["description"], "type": "TOPIC", "confidence": pref["confidence"]})
    for cst in output.constraints:
        entities.append({"name": cst["description"], "type": "TOPIC", "confidence": cst["confidence"]})
    return entities


def _dedupe_by_key(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        value = str(item.get(key, "")).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(item)
    return out


def _semantic_dedupe_items(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in items:
        raw_value = str(item.get(key, "")).strip()
        if not raw_value:
            continue
        replaced_idx = None
        for idx, kept in enumerate(selected):
            kept_value = str(kept.get(key, "")).strip()
            if _is_semantic_duplicate(raw_value, kept_value):
                replaced_idx = idx
                break
        if replaced_idx is None:
            selected.append(item)
            continue
        if _item_quality_score(item, key) > _item_quality_score(selected[replaced_idx], key):
            selected[replaced_idx] = item
    return selected


def _semantic_dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for ent in entities:
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        replaced_idx = None
        for idx, kept in enumerate(selected):
            kept_name = str(kept.get("name", "")).strip()
            if _is_semantic_duplicate(name, kept_name):
                replaced_idx = idx
                break
        if replaced_idx is None:
            selected.append(ent)
            continue
        if _entity_quality_score(ent) > _entity_quality_score(selected[replaced_idx]):
            selected[replaced_idx] = ent
    return selected


def _entity_quality_score(entity: dict[str, Any]) -> float:
    type_priority = {
        "PROJECT": 4.0,
        "TOOL": 4.0,
        "SKILL": 3.0,
        "PERSON": 2.0,
        "ORGANIZATION": 2.0,
        "TOPIC": 1.0,
    }
    entity_type = str(entity.get("type", "TOPIC")).upper()
    confidence = float(entity.get("confidence", 0.0))
    name_len = len(str(entity.get("name", "")).strip())
    return type_priority.get(entity_type, 0.5) + confidence + min(name_len / 100.0, 0.5)


def _item_quality_score(item: dict[str, Any], key: str) -> float:
    confidence = float(item.get("confidence", 0.0))
    value_len = len(str(item.get(key, "")).strip())
    return confidence + min(value_len / 100.0, 0.5)


def _is_semantic_duplicate(a: str, b: str) -> bool:
    a_norm = _normalize_semantic_text(a)
    b_norm = _normalize_semantic_text(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if not a_tokens or not b_tokens:
        return False

    overlap = len(a_tokens & b_tokens) / max(min(len(a_tokens), len(b_tokens)), 1)
    if overlap >= 0.8:
        return True

    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= 0.84


def _normalize_semantic_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"^(i|we|my|our)\s+(need to|want to|prefer|like|have to|am|m|would like to)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^(need to|want to|prefer|like|have to|please|must)\s+",
        "",
        cleaned,
    )
    replacements = {
        "replacement": "replace",
        "replacing": "replace",
        "uses": "use",
        "using": "use",
        "worked": "work",
        "working": "work",
    }
    tokens = []
    stopwords = {
        "i",
        "my",
        "we",
        "our",
        "to",
        "the",
        "a",
        "an",
        "please",
        "that",
        "this",
        "it",
    }
    for tok in cleaned.split():
        tok = replacements.get(tok, tok)
        if tok in stopwords:
            continue
        tokens.append(tok)
    return " ".join(tokens)


def _is_good_memory_span(text: str) -> bool:
    span = " ".join(text.strip().split())
    if len(span) < 3:
        return False
    if len(span) > 120:
        return False
    generic = {
        "what",
        "how",
        "why",
        "help",
        "summarize",
        "summary",
        "explain",
        "write",
        "create",
        "generate",
        "answer",
        "question",
    }
    lowered = span.lower()
    if lowered in generic:
        return False
    if re.fullmatch(r"[A-Za-z]+", span) and lowered in generic:
        return False
    if any(marker in lowered for marker in ("url:", "abstract:", "introduction:", "web search results:")):
        return False
    if re.search(r"\b(mov|jmp|push|pop|call)\b", lowered) and any(ch in span for ch in "[]*"):
        return False
    if len(span.split()) > 12 and not re.search(r"\b(my|i|we|our)\b", lowered):
        return False
    return True


def _has_strong_personal_memory_signal(text: str) -> bool:
    lowered = _normalize_rule_text(text).lower()
    strong_patterns = [
        r"\bi(?:'m| am)?\s+working on\b",
        r"\bi(?:'m| am)?\s+learning\b",
        r"\bi(?:'m| am)?\s+building\b",
        r"\bi(?:'m| am)?\s+creating\b",
        r"\bi(?:'m| am)?\s+writing\b",
        r"\bmy project\b",
        r"\bfor my (?:class|course|research|job|team)\b",
        r"\bi want to\b",
        r"\bi need to\b",
        r"\bi have to\b",
        r"\bi prefer\b",
        r"\bi like\b",
        r"\bi can(?:not|'t)\b",
        r"\bdeadline(?: is|:)?\b",
        r"\bplease (?:be|keep)\b",
        r"\bour project\b",
        r"\bwe are working on\b",
    ]
    return any(re.search(pattern, lowered) for pattern in strong_patterns)


def _looks_like_reference_or_content_dump(text: str) -> bool:
    lowered = text.lower()
    if "web search results:" in lowered:
        return True
    if any(marker in lowered for marker in ("abstract:", "introduction:", "paper name:")):
        return True
    if lowered.count("url:") >= 1:
        return True
    if re.search(r"\[[0-9]+\]\s+\"", text):
        return True
    code_lines = re.findall(r"^(?:[A-Z]{2,6}\b.*|\s*[A-Za-z_]+\s*[\(\[].*)$", text, flags=re.MULTILINE)
    if len(code_lines) >= 4:
        return True
    return False


def _normalize_rule_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("…", " ")
    normalized = re.sub(r"\.{2,}", " ", normalized)
    normalized = re.sub(r"[?!]{2,}", " ", normalized)
    normalized = re.sub(r"\bim\b", "I am", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bi've been kinda learning\b", "I am learning", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bi have been kinda learning\b", "I am learning", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bi'm kind of dealing with this:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\buh,\s*basically,\s*recently,\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bkinda\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bpyhton\b", "python", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\banalyss\b", "analysis", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\banalyis\b", "analysis", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

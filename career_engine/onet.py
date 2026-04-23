from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from .schema import KNOWLEDGE_ELEMENTS, SKILL_ELEMENTS, WORK_STYLE_ELEMENTS

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    import numpy as _np
    _SBERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SBERT_AVAILABLE = False

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

_ALL_CANONICAL: list[str] = KNOWLEDGE_ELEMENTS + SKILL_ELEMENTS + WORK_STYLE_ELEMENTS
_CANONICAL_LOWER: dict[str, str] = {e.lower(): e for e in _ALL_CANONICAL}


def _canonicalize_element(name: str) -> str:
    """Map an xlsx element name to the canonical form used in schema lists (case-insensitive)."""
    return _CANONICAL_LOWER.get(name.strip().lower(), name.strip())


class _EmbeddingMapper:
    """Lazy singleton: maps free-text node content to O*NET elements via sentence embeddings."""

    _instance: "_EmbeddingMapper | None" = None

    def __init__(self) -> None:
        model = _SentenceTransformer("all-MiniLM-L6-v2")
        self._model = model
        self._categories: dict[str, tuple[list[str], Any]] = {}
        for cat, elements in [
            ("knowledge", KNOWLEDGE_ELEMENTS),
            ("skills", SKILL_ELEMENTS),
            ("work_styles", WORK_STYLE_ELEMENTS),
        ]:
            embs = model.encode(elements, normalize_embeddings=True)
            self._categories[cat] = (list(elements), embs)

    @classmethod
    def get(cls) -> "_EmbeddingMapper":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def map(self, content: str, category: str, top_k: int = 2) -> list[dict[str, Any]]:
        if category not in self._categories:
            return []
        elements, embs = self._categories[category]
        query_emb = self._model.encode([content], normalize_embeddings=True)[0]
        scores = (_np.array(embs) @ query_emb).tolist()
        ranked = sorted(zip(elements, scores), key=lambda x: -x[1])
        return [
            {"element_name": elem, "confidence": round(float(max(0.0, score)), 4)}
            for elem, score in ranked[:top_k]
        ]


@dataclass
class OccupationProfile:
    onet_code: str
    title: str
    knowledge_vector: dict[str, float]
    skills_vector: dict[str, float]
    work_styles_vector: dict[str, float]


def load_builtin_occupation_profiles() -> list[OccupationProfile]:
    """Small O*NET-style profile set used until raw O*NET xlsx files are added."""
    return [
        OccupationProfile(
            onet_code="15-2051.00",
            title="Data Scientists",
            knowledge_vector=_vector(
                KNOWLEDGE_ELEMENTS,
                {
                    "Computers and Electronics": 5.8,
                    "Mathematics": 5.1,
                    "Engineering and Technology": 4.8,
                    "Economics and Accounting": 3.4,
                    "Communications and Media": 3.1,
                },
            ),
            skills_vector=_vector(
                SKILL_ELEMENTS,
                {
                    "Programming": 5.5,
                    "Complex Problem Solving": 5.2,
                    "Critical Thinking": 5.0,
                    "Mathematics": 4.8,
                    "Systems Evaluation": 4.3,
                    "Quality Control Analysis": 3.8,
                },
            ),
            work_styles_vector=_vector(
                WORK_STYLE_ELEMENTS,
                {
                    "Analytical Thinking": 5.7,
                    "Attention to Detail": 5.2,
                    "Persistence": 4.7,
                    "Innovation": 4.6,
                    "Independence": 4.5,
                },
            ),
        ),
        OccupationProfile(
            onet_code="13-2099.01",
            title="Quantitative Analysts",
            knowledge_vector=_vector(
                KNOWLEDGE_ELEMENTS,
                {
                    "Mathematics": 6.1,
                    "Economics and Accounting": 5.9,
                    "Computers and Electronics": 5.2,
                    "Administration and Management": 3.0,
                    "English Language": 3.0,
                },
            ),
            skills_vector=_vector(
                SKILL_ELEMENTS,
                {
                    "Mathematics": 5.8,
                    "Critical Thinking": 5.4,
                    "Programming": 4.8,
                    "Judgment and Decision Making": 4.7,
                    "Complex Problem Solving": 4.6,
                    "Systems Analysis": 4.3,
                },
            ),
            work_styles_vector=_vector(
                WORK_STYLE_ELEMENTS,
                {
                    "Analytical Thinking": 5.8,
                    "Attention to Detail": 5.5,
                    "Dependability": 4.6,
                    "Integrity": 4.5,
                    "Persistence": 4.4,
                },
            ),
        ),
        OccupationProfile(
            onet_code="15-1252.00",
            title="Software Developers",
            knowledge_vector=_vector(
                KNOWLEDGE_ELEMENTS,
                {
                    "Computers and Electronics": 5.9,
                    "Engineering and Technology": 5.2,
                    "Mathematics": 4.0,
                    "Design": 3.8,
                    "Telecommunications": 3.0,
                },
            ),
            skills_vector=_vector(
                SKILL_ELEMENTS,
                {
                    "Programming": 5.8,
                    "Systems Analysis": 5.1,
                    "Complex Problem Solving": 5.0,
                    "Critical Thinking": 4.8,
                    "Quality Control Analysis": 4.4,
                    "Technology Design": 4.3,
                },
            ),
            work_styles_vector=_vector(
                WORK_STYLE_ELEMENTS,
                {
                    "Analytical Thinking": 5.4,
                    "Attention to Detail": 5.1,
                    "Dependability": 4.6,
                    "Persistence": 4.5,
                    "Innovation": 4.3,
                },
            ),
        ),
        OccupationProfile(
            onet_code="15-2051.00-ML",
            title="Machine Learning Engineer",
            knowledge_vector=_vector(
                KNOWLEDGE_ELEMENTS,
                {
                    "Computers and Electronics": 6.0,
                    "Mathematics": 5.5,
                    "Engineering and Technology": 5.4,
                    "Physics": 3.1,
                    "Communications and Media": 2.8,
                },
            ),
            skills_vector=_vector(
                SKILL_ELEMENTS,
                {
                    "Programming": 5.9,
                    "Complex Problem Solving": 5.4,
                    "Systems Analysis": 5.0,
                    "Systems Evaluation": 4.8,
                    "Quality Control Analysis": 4.6,
                    "Mathematics": 4.6,
                },
            ),
            work_styles_vector=_vector(
                WORK_STYLE_ELEMENTS,
                {
                    "Analytical Thinking": 5.7,
                    "Attention to Detail": 5.3,
                    "Persistence": 4.9,
                    "Innovation": 4.7,
                    "Dependability": 4.5,
                },
            ),
        ),
    ]


def load_occupation_profiles(data_dir: Path | None = None) -> list[OccupationProfile]:
    """Load real O*NET xlsx files when present; otherwise use the small bundled scaffold."""
    if data_dir is None:
        data_dir = Path("data")
    knowledge_path = data_dir / "Knowledge.xlsx"
    skills_path = data_dir / "Skills.xlsx"
    work_styles_path = data_dir / "Work Styles.xlsx"
    if not knowledge_path.exists() or not skills_path.exists():
        return load_builtin_occupation_profiles()

    knowledge = _load_onet_xlsx(knowledge_path, scale_id="IM")
    skills = _load_onet_xlsx(skills_path, scale_id="IM")
    work_styles = _load_onet_xlsx(work_styles_path, scale_id="WI") if work_styles_path.exists() else {}
    occupation_keys = sorted(set(knowledge) | set(skills) | set(work_styles))
    profiles: list[OccupationProfile] = []
    for code in occupation_keys:
        title = (
            knowledge.get(code, {}).get("title")
            or skills.get(code, {}).get("title")
            or work_styles.get(code, {}).get("title")
            or code
        )
        profiles.append(
            OccupationProfile(
                onet_code=code,
                title=title,
                knowledge_vector=dict(knowledge.get(code, {}).get("vector", {})),
                skills_vector=dict(skills.get(code, {}).get("vector", {})),
                work_styles_vector=dict(work_styles.get(code, {}).get("vector", {})),
            )
        )
    return profiles


def align_nodes_to_onet(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**node, "onet_mappings": _map_node(node)} for node in nodes]


def build_user_profile(mapped_nodes: list[dict[str, Any]], user_id: str = "user_001") -> dict[str, Any]:
    profile = {
        "user_id": user_id,
        "knowledge_vector": {element: 0.0 for element in KNOWLEDGE_ELEMENTS},
        "skills_vector": {element: 0.0 for element in SKILL_ELEMENTS},
        "work_styles_vector": {element: 0.0 for element in WORK_STYLE_ELEMENTS},
    }
    for node in mapped_nodes:
        dataset = _dataset_for_node(str(node.get("node_type", "")))
        if not dataset:
            continue
        vector_name = f"{dataset}_vector"
        node_conf = float(node.get("confidence", 0.7))
        mentions = max(1, int(node.get("mention_count", 1)))
        recency = float(node.get("recency_weight", 1.0))
        for mapping in node.get("onet_mappings", []):
            element = mapping["element_name"]
            if element not in profile[vector_name]:
                continue
            contribution = node_conf * float(mapping["confidence"]) * mentions * recency
            profile[vector_name][element] = min(7.0, profile[vector_name][element] + contribution)
    _apply_cognitive_inference(profile, mapped_nodes)
    return profile


def _apply_cognitive_inference(profile: dict[str, Any], mapped_nodes: list[dict[str, Any]]) -> None:
    """Infer baseline cognitive skill scores that the LLM never extracts explicitly.

    A user with multiple technical skills/tools and projects demonstrably has
    Critical Thinking and Complex Problem Solving — these should not be 0.
    """
    skill_tool_count = sum(
        1 for n in mapped_nodes if str(n.get("node_type", "")) in {"skill", "tool"}
    )
    project_count = sum(
        1 for n in mapped_nodes if str(n.get("node_type", "")) == "project"
    )
    sv = profile["skills_vector"]
    if skill_tool_count >= 3:
        sv["Critical Thinking"] = max(sv.get("Critical Thinking", 0.0), 0.8)
        sv["Complex Problem Solving"] = max(sv.get("Complex Problem Solving", 0.0), 0.7)
        sv["Active Learning"] = max(sv.get("Active Learning", 0.0), 0.6)
    if project_count >= 2:
        sv["Judgment and Decision Making"] = max(sv.get("Judgment and Decision Making", 0.0), 0.6)
        sv["Reading Comprehension"] = max(sv.get("Reading Comprehension", 0.0), 0.5)
    # Writing: anyone who codes writes documentation, comments, READMEs
    if skill_tool_count >= 3:
        sv["Writing"] = max(sv.get("Writing", 0.0), 0.4)
    # Stress Tolerance: projects involve deadlines and technical pressure
    wv = profile["work_styles_vector"]
    if project_count >= 2:
        wv["Stress Tolerance"] = max(wv.get("Stress Tolerance", 0.0), 0.4)


def recommend_careers(
    user_profile: dict[str, Any],
    occupations: list[OccupationProfile],
    top_k: int = 5,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if weights is None:
        has_work_styles = any(any(value > 0 for value in occ.work_styles_vector.values()) for occ in occupations)
        weights = (
            {"knowledge": 0.35, "skills": 0.45, "work_styles": 0.20}
            if has_work_styles
            else {"knowledge": 0.45, "skills": 0.55, "work_styles": 0.0}
        )
    recommendations = []
    for occ in occupations:
        knowledge = cosine_similarity(user_profile["knowledge_vector"], occ.knowledge_vector)
        skills = cosine_similarity(user_profile["skills_vector"], occ.skills_vector)
        styles = cosine_similarity(user_profile["work_styles_vector"], occ.work_styles_vector)
        score = weights["knowledge"] * knowledge + weights["skills"] * skills + weights["work_styles"] * styles
        recommendations.append(
            {
                "onet_code": occ.onet_code,
                "title": occ.title,
                "score": round(score, 4),
                "component_scores": {
                    "knowledge": round(knowledge, 4),
                    "skills": round(skills, 4),
                    "work_styles": round(styles, 4),
                },
            }
        )
    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return {"user_id": user_profile["user_id"], "recommendations": recommendations[:top_k]}


def analyze_skill_gaps(
    user_profile: dict[str, Any],
    occupation: OccupationProfile,
    top_k: int = 5,
) -> dict[str, Any]:
    return {
        "user_id": user_profile["user_id"],
        "target": f"{occupation.title} ({occupation.onet_code})",
        "knowledge_gaps": _top_gaps(user_profile["knowledge_vector"], occupation.knowledge_vector, top_k),
        "skill_gaps": _top_gaps(user_profile["skills_vector"], occupation.skills_vector, top_k),
        "work_style_gaps": _top_gaps(user_profile["work_styles_vector"], occupation.work_styles_vector, top_k),
    }


def save_occupation_profiles(path: Path, occupations: list[OccupationProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "onet_code": occ.onet_code,
            "title": occ.title,
            "knowledge_vector": occ.knowledge_vector,
            "skills_vector": occ.skills_vector,
            "work_styles_vector": occ.work_styles_vector,
        }
        for occ in occupations
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    left_norm = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    right_norm = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _vector(elements: list[str], values: dict[str, float]) -> dict[str, float]:
    return {element: float(values.get(element, 0.0)) for element in elements}


def _load_onet_xlsx(path: Path, scale_id: str = "IM") -> dict[str, dict[str, Any]]:
    rows = _read_xlsx_rows(path)
    if not rows:
        return {}
    headers = {str(name): idx for idx, name in enumerate(rows[0])}
    required = ["O*NET-SOC Code", "Title", "Element Name", "Scale ID", "Data Value"]
    missing = [name for name in required if name not in headers]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    by_code: dict[str, dict[str, Any]] = defaultdict(lambda: {"title": "", "vector": {}})
    for row in rows[1:]:
        if len(row) <= max(headers[name] for name in required):
            continue
        if str(row[headers["Scale ID"]]).strip() != scale_id:
            continue
        code = str(row[headers["O*NET-SOC Code"]]).strip()
        title = str(row[headers["Title"]]).strip()
        element = _canonicalize_element(str(row[headers["Element Name"]]).strip())
        value_raw = str(row[headers["Data Value"]]).strip()
        if not code or not element or not value_raw:
            continue
        try:
            value = float(value_raw)
        except ValueError:
            continue
        by_code[code]["title"] = title
        by_code[code]["vector"][element] = value
    return dict(by_code)


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as xlsx:
        shared_strings = _read_shared_strings(xlsx)
        sheet_name = _first_sheet_name(xlsx)
        root = ET.fromstring(xlsx.read(sheet_name))
    rows: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values: list[str] = []
        for cell in row.findall("a:c", XLSX_NS):
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", XLSX_NS)
            if value_node is None or value_node.text is None:
                values.append("")
            elif cell_type == "s":
                values.append(shared_strings[int(value_node.text)])
            else:
                values.append(value_node.text)
        rows.append(values)
    return rows


def _first_sheet_name(xlsx: ZipFile) -> str:
    names = [name for name in xlsx.namelist() if name.startswith("xl/worksheets/sheet")]
    if not names:
        raise ValueError("xlsx file has no worksheets")
    return sorted(names)[0]


def _read_shared_strings(xlsx: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []
    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    text_tag = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
    for item in root.findall("a:si", XLSX_NS):
        strings.append("".join(node.text or "" for node in item.iter(text_tag)))
    return strings


def _dataset_for_node(node_type: str) -> str | None:
    normalized = node_type.lower()
    if normalized == "knowledge":
        return "knowledge"
    if normalized in {"skill", "tool"}:
        return "skills"
    if normalized in {"behavioral_trait", "implicit_signal"}:
        return "work_styles"
    return None


def _map_node(node: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(node.get("content", "")).lower()
    node_type = str(node.get("node_type", "")).lower()

    if node_type == "knowledge":
        category = "knowledge"
    elif node_type in {"skill", "tool"}:
        category = "skills"
    elif node_type in {"behavioral_trait", "implicit_signal"}:
        category = "work_styles"
    else:
        return []

    if _SBERT_AVAILABLE:
        try:
            return _EmbeddingMapper.get().map(content, category, top_k=2)
        except Exception:  # pragma: no cover
            pass

    # fallback: keyword rules
    if category == "knowledge":
        mappings = _knowledge_mappings(content)
    elif category == "skills":
        mappings = _skill_mappings(content)
    else:
        mappings = _work_style_mappings(content)
    mappings.sort(key=lambda item: item["confidence"], reverse=True)
    return mappings[:3]


def _knowledge_mappings(content: str) -> list[dict[str, Any]]:
    rules = [
        ("Computers and Electronics", ["computer", "software", "database", "cloud", "api", "python", "sql"]),
        ("Mathematics", ["math", "statistics", "statistical", "model", "quantitative"]),
        ("Economics and Accounting", ["finance", "financial", "economics", "accounting", "quant"]),
        ("Engineering and Technology", ["engineering", "machine learning", "deep learning", "deployment"]),
        ("Design", ["design", "dashboard", "visualization", "interface"]),
        ("Communications and Media", ["writing", "presentation", "communication", "media"]),
    ]
    return _rules_to_mappings(content, rules)


def _skill_mappings(content: str) -> list[dict[str, Any]]:
    rules = [
        ("Programming", ["python", "sql", "r", "pytorch", "tensorflow", "fastapi", "programming", "docker"]),
        ("Mathematics", ["math", "statistics", "quantitative", "statistical"]),
        ("Complex Problem Solving", ["problem solving", "debugging", "optimization", "troubleshooting"]),
        ("Quality Control Analysis", ["quality", "edge cases", "evaluation", "testing", "reliable"]),
        ("Systems Analysis", ["system design", "distributed systems", "api", "deployment"]),
        ("Systems Evaluation", ["model evaluation", "evaluation", "monitoring"]),
        ("Technology Design", ["design", "build", "develop", "architecture"]),
        ("Critical Thinking", ["analysis", "analytical", "reasoning", "decision"]),
    ]
    return _rules_to_mappings(content, rules)


def _work_style_mappings(content: str) -> list[dict[str, Any]]:
    rules = [
        ("Attention to Detail", ["detail", "edge cases", "quality", "checking", "thorough"]),
        ("Intellectual Curiosity", ["analytical", "analysis", "problem solving", "reasoning", "curious", "explore"]),
        ("Perseverance", ["persistence", "persistent", "refine", "keep trying", "going back", "determined"]),
        ("Dependability", ["reliable", "dependable", "responsible", "consistent"]),
        ("Innovation", ["creative", "innovation", "prototype", "experiment", "novel"]),
        ("Initiative", ["independent", "self-directed", "on my own", "proactive", "self-starter"]),
        ("Adaptability", ["adapt", "flexible", "change", "pivot", "adjust"]),
        ("Achievement Orientation", ["achieve", "accomplish", "goal", "driven", "motivated"]),
        ("Cooperation", ["team", "collaborate", "work with", "together"]),
        ("Stress Tolerance", ["stress", "pressure", "deadline", "calm", "composed"]),
        ("Integrity", ["honest", "ethics", "transparent", "integrity"]),
        ("Self-Control", ["patient", "self-control", "disciplined", "composed"]),
    ]
    return _rules_to_mappings(content, rules)


def _rules_to_mappings(content: str, rules: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    out = []
    for element, keywords in rules:
        if any(keyword in content for keyword in keywords):
            out.append({"element_name": element, "confidence": 0.9})
    if not out:
        fallback = rules[0][0] if rules else None
        if fallback:
            out.append({"element_name": fallback, "confidence": 0.35})
    return out


def _top_gaps(user_vector: dict[str, float], occupation_vector: dict[str, float], top_k: int) -> list[dict[str, Any]]:
    gaps = []
    for element, required in occupation_vector.items():
        if required <= 0:
            continue
        user_score = float(user_vector.get(element, 0.0))
        gap = max(0.0, float(required) - user_score)
        if gap <= 0:
            continue
        gaps.append(
            {
                "element": element,
                "required_score": round(float(required), 3),
                "gap": round(gap, 3),
            }
        )
    gaps.sort(key=lambda item: item["gap"], reverse=True)
    for idx, item in enumerate(gaps[:top_k], start=1):
        item["priority"] = idx
    return gaps[:top_k]

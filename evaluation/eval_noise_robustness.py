from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_engine.extraction_agent import MemoryExtractionAgent
from memory_engine.graph_memory import GraphMemory


FIELDS = ("entities", "relations", "preferences", "constraints", "goals", "projects", "tools")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def normalize_item(field: str, item: Any) -> str:
    if field == "relations" and isinstance(item, dict):
        return "|".join(
            [
                normalize_text(str(item.get("source", "USER"))),
                normalize_text(str(item.get("relation_type", "RELATED_TO"))),
                normalize_text(str(item.get("target", ""))),
            ]
        )
    if isinstance(item, dict):
        if "name" in item:
            return normalize_text(str(item.get("name", "")))
        return normalize_text(str(item.get("description", "")))
    return normalize_text(str(item))


def extraction_record(extraction: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        "entities": extraction.entities,
        "relations": extraction.relations,
        "preferences": extraction.preferences,
        "constraints": extraction.constraints,
        "goals": extraction.goals,
        "projects": extraction.projects,
        "tools": extraction.tools,
    }


def flatten_set(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in FIELDS:
        for item in record.get(field, []):
            normalized = normalize_item(field, item)
            if normalized:
                values.add(f"{field}:{normalized}")
    return values


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def typo_variant(text: str) -> str:
    out = text.replace("Python", "pyhton")
    out = out.replace("analysis", "analyss")
    out = out.replace("I am", "Im")
    return out


def filler_variant(text: str) -> str:
    return f"Uh, basically, recently, I'm kind of dealing with this: {text}"


def colloquial_variant(text: str) -> str:
    return text.replace("I am learning", "Lately I've been kinda learning")


def punctuation_variant(text: str) -> str:
    return re.sub(r"\s+", " ... ", text).strip() + " ???"


def build_variants(clean_text: str) -> list[dict[str, str]]:
    return [
        {"label": "clean", "text": clean_text},
        {"label": "typo", "text": typo_variant(clean_text)},
        {"label": "filler", "text": filler_variant(clean_text)},
        {"label": "colloquial", "text": colloquial_variant(clean_text)},
        {"label": "punctuation", "text": punctuation_variant(clean_text)},
    ]


def graph_snapshot(graph: GraphMemory) -> dict[str, Any]:
    nodes = []
    for node_id, attrs in graph.graph.nodes(data=True):
        nodes.append({"node_id": node_id, **dict(attrs)})
    edges = []
    for source, target, attrs in graph.graph.edges(data=True):
        edges.append({"source": source, "target": target, **dict(attrs)})
    return {"nodes": nodes, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extractor robustness under noisy user phrasing.")
    parser.add_argument(
        "--text",
        type=str,
        default="I am learning Python data analysis.",
        help="Clean source utterance to perturb",
    )
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Extractor model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Extractor temperature")
    parser.add_argument(
        "--disable-relevance-filter",
        action="store_true",
        help="Disable relevance filtering during extraction",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/noise_robustness.json"),
        help="Where to write the robustness report",
    )
    args = parser.parse_args()

    graph = GraphMemory()
    agent = MemoryExtractionAgent(
        graph_memory=graph,
        model=args.model,
        use_relevance_filter=not args.disable_relevance_filter,
        temperature=args.temperature,
    )
    variants = build_variants(args.text)
    baseline_set: set[str] | None = None
    results: list[dict[str, Any]] = []

    for idx, variant in enumerate(variants):
        turn = {
            "dialogue_id": "robustness_eval_dialogue",
            "turn_id": f"robustness::{idx}",
            "speaker": "human",
            "text": variant["text"],
        }
        extraction = agent.process_turn(turn)
        record = extraction_record(extraction)
        item_set = flatten_set(record)
        if baseline_set is None:
            baseline_set = item_set
        non_user_nodes = [
            {"node_id": node_id, **dict(attrs)}
            for node_id, attrs in graph.graph.nodes(data=True)
            if attrs.get("node_type") != "User"
        ]
        results.append(
            {
                "label": variant["label"],
                "text": variant["text"],
                "extraction": record,
                "jaccard_vs_clean": jaccard(baseline_set or set(), item_set),
                "graph_non_user_nodes_after_turn": non_user_nodes,
            }
        )

    non_user_nodes = [
        {"node_id": node_id, **dict(attrs)}
        for node_id, attrs in graph.graph.nodes(data=True)
        if attrs.get("node_type") != "User"
    ]
    mention_counts = [int(node.get("mention_count", 0)) for node in non_user_nodes]
    payload = {
        "summary": {
            "variants_tested": len(variants),
            "avg_jaccard_vs_clean": sum(item["jaccard_vs_clean"] for item in results[1:]) / max(len(results) - 1, 1),
            "linked_non_user_node_count": len(non_user_nodes),
            "max_mention_count": max(mention_counts) if mention_counts else 0,
        },
        "variants": results,
        "graph": graph_snapshot(graph),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_engine.extraction_agent import MemoryExtractionAgent, has_memory_output
from memory_engine.graph_memory import GraphMemory


FIELDS = ("entities", "relations", "preferences", "constraints", "goals", "projects", "tools")


def load_turns(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            turns.append(json.loads(line))
    return turns


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def normalize_item(field: str, item: Any) -> str:
    if field == "relations" and isinstance(item, dict):
        source = normalize_text(str(item.get("source", "USER")))
        target = normalize_text(str(item.get("target", "")))
        rel = normalize_text(str(item.get("relation_type", "RELATED_TO")))
        return f"{source}|{rel}|{target}"
    if isinstance(item, dict):
        if "name" in item:
            return normalize_text(str(item.get("name", "")))
        return normalize_text(str(item.get("description", "")))
    return normalize_text(str(item))


def extraction_to_sets(record: dict[str, Any]) -> dict[str, set[str]]:
    field_sets: dict[str, set[str]] = {}
    for field in FIELDS:
        field_sets[field] = {
            normalized
            for normalized in (normalize_item(field, item) for item in record.get(field, []))
            if normalized
        }
    field_sets["all"] = set().union(*(field_sets[field] for field in FIELDS))
    return field_sets


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def run_extraction(
    turns: list[dict[str, Any]],
    model: str,
    temperature: float,
    use_relevance_filter: bool,
) -> dict[str, dict[str, set[str]]]:
    agent = MemoryExtractionAgent(
        graph_memory=GraphMemory(),
        model=model,
        use_relevance_filter=use_relevance_filter,
        temperature=temperature,
    )
    outputs: dict[str, dict[str, set[str]]] = {}
    for turn in turns:
        extraction = agent.process_turn(turn)
        if has_memory_output(extraction):
            record = {
                "entities": extraction.entities,
                "relations": extraction.relations,
                "preferences": extraction.preferences,
                "constraints": extraction.constraints,
                "goals": extraction.goals,
                "projects": extraction.projects,
                "tools": extraction.tools,
            }
        else:
            record = {field: [] for field in FIELDS}
        outputs[turn["turn_id"]] = extraction_to_sets(record)
    return outputs


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure extractor self-consistency across repeated runs.")
    parser.add_argument("--turns", type=Path, required=True, help="Path to sample_turns.jsonl")
    parser.add_argument("--sample", type=int, default=10, help="How many human turns to evaluate")
    parser.add_argument("--runs", type=int, default=5, help="How many repeated extraction runs")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
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
        default=Path("outputs/self_consistency.json"),
        help="Where to write the consistency report",
    )
    args = parser.parse_args()

    all_turns = [turn for turn in load_turns(args.turns) if turn.get("speaker") == "human"]
    rng = random.Random(args.seed)
    rng.shuffle(all_turns)
    selected_turns = all_turns[: args.sample] if args.sample > 0 else all_turns

    run_outputs = [
        run_extraction(
            turns=selected_turns,
            model=args.model,
            temperature=args.temperature,
            use_relevance_filter=not args.disable_relevance_filter,
        )
        for _ in range(args.runs)
    ]

    pair_metrics: list[dict[str, Any]] = []
    per_field_scores: dict[str, list[float]] = {field: [] for field in (*FIELDS, "all")}
    identical_turn_count = 0

    for turn in selected_turns:
        turn_id = turn["turn_id"]
        turn_pair_scores: dict[str, list[float]] = {field: [] for field in (*FIELDS, "all")}
        all_identical = True
        for left_idx, right_idx in combinations(range(args.runs), 2):
            left = run_outputs[left_idx][turn_id]
            right = run_outputs[right_idx][turn_id]
            for field in (*FIELDS, "all"):
                score = jaccard(left[field], right[field])
                turn_pair_scores[field].append(score)
                per_field_scores[field].append(score)
                if field == "all":
                    pair_metrics.append(
                        {
                            "turn_id": turn_id,
                            "run_a": left_idx + 1,
                            "run_b": right_idx + 1,
                            "jaccard_all": score,
                        }
                    )
                    if score < 1.0:
                        all_identical = False
        if all_identical:
            identical_turn_count += 1

    summary = {
        "sampled_turns": len(selected_turns),
        "runs": args.runs,
        "model": args.model,
        "temperature": args.temperature,
        "avg_jaccard_all": mean(per_field_scores["all"]),
        "avg_jaccard_entities": mean(per_field_scores["entities"]),
        "avg_jaccard_relations": mean(per_field_scores["relations"]),
        "avg_jaccard_preferences": mean(per_field_scores["preferences"]),
        "avg_jaccard_constraints": mean(per_field_scores["constraints"]),
        "avg_jaccard_goals": mean(per_field_scores["goals"]),
        "avg_jaccard_projects": mean(per_field_scores["projects"]),
        "avg_jaccard_tools": mean(per_field_scores["tools"]),
        "exact_match_turn_ratio": identical_turn_count / len(selected_turns) if selected_turns else None,
    }
    payload = {
        "summary": summary,
        "sample_turn_ids": [turn["turn_id"] for turn in selected_turns],
        "pairwise_scores": pair_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from pathlib import Path

from memory_engine.data_pipeline import build_sample_turns, load_sharegpt, write_json, write_jsonl

from .extraction_agent import CareerExtractionAgent, extraction_sections, has_career_output
from .graph import CareerGraphMemory
from .onet import (
    align_nodes_to_onet,
    analyze_skill_gaps,
    build_user_profile,
    load_occupation_profiles,
    recommend_careers,
    save_occupation_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run career development graph pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/career_dialogues_all_180.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/career"))
    parser.add_argument("--sample-size", type=int, default=180)
    parser.add_argument("--max-turns", type=int, default=10000)
    parser.add_argument("--min-turn-chars", type=int, default=8)
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--onet-dir", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-career", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_dialogues = load_sharegpt(args.input)
    turns, stats = build_sample_turns(
        raw_dialogues=raw_dialogues,
        sample_size=args.sample_size,
        min_turn_chars=args.min_turn_chars,
    )
    write_jsonl(args.output_dir / "career_sample_turns.jsonl", turns)
    write_json(args.output_dir / "career_pipeline_stats.json", stats.__dict__)

    graph = CareerGraphMemory()
    agent = CareerExtractionAgent(
        graph_memory=graph,
        model=args.model,
        temperature=args.temperature,
        use_llm=not args.disable_llm,
    )
    print(f"Career extraction mode: {'Groq + rules fallback' if agent.client else 'Rules only'}")

    extractions = []
    human_processed = 0
    started_at = time.time()
    for turn in turns[: args.max_turns]:
        if turn["speaker"] != "human":
            continue
        human_processed += 1
        extracted = agent.process_turn(turn)
        if not has_career_output(extracted):
            continue
        row = {"turn_id": turn["turn_id"], "text": turn["text"], "relations": extracted.relations}
        row.update(extraction_sections(extracted))
        extractions.append(row)
        if args.progress_every > 0 and human_processed % args.progress_every == 0:
            print(f"Processed {human_processed} human turns in {time.time() - started_at:.1f}s")

    write_json(args.output_dir / "career_extractions.json", extractions)
    graph.export_json(args.output_dir / "career_graph_memory.json")
    write_json(args.output_dir / "career_graph_stats.json", graph.summary())

    occupations = load_occupation_profiles(args.onet_dir)
    save_occupation_profiles(args.output_dir / "career_occupation_profiles.json", occupations)

    mapped_nodes = align_nodes_to_onet(graph.signal_nodes())
    write_json(args.output_dir / "career_onet_mappings.json", mapped_nodes)

    profile = build_user_profile(mapped_nodes)
    write_json(args.output_dir / "career_user_profile.json", profile)

    recommendations = recommend_careers(profile, occupations, top_k=args.top_k)
    write_json(args.output_dir / "career_recommendations.json", recommendations)

    target = _select_target_occupation(args.target_career, recommendations, occupations)
    gap_analysis = analyze_skill_gaps(profile, target, top_k=args.top_k)
    write_json(args.output_dir / "career_gap_analysis.json", gap_analysis)

    summary = _build_summary(
        stats.__dict__,
        graph.summary(),
        extractions,
        recommendations,
        gap_analysis,
        occupation_count=len(occupations),
        onet_dir=args.onet_dir,
    )
    (args.output_dir / "career_evaluation_summary.md").write_text(summary, encoding="utf-8")

    print("Career pipeline finished.")
    print(f"- output dir: {args.output_dir}")
    print(f"- career graph: {args.output_dir / 'career_graph_memory.json'}")
    print(f"- recommendations: {args.output_dir / 'career_recommendations.json'}")
    print(f"- gap analysis: {args.output_dir / 'career_gap_analysis.json'}")


def _select_target_occupation(
    target: str | None,
    recommendations: dict,
    occupations: list,
):
    if target:
        lowered = target.lower()
        for occupation in occupations:
            if lowered in occupation.title.lower() or lowered == occupation.onet_code.lower():
                return occupation
    top_code = recommendations["recommendations"][0]["onet_code"]
    return next(occupation for occupation in occupations if occupation.onet_code == top_code)


def _build_summary(
    stats: dict,
    graph_stats: dict,
    extractions: list[dict],
    recommendations: dict,
    gap_analysis: dict,
    occupation_count: int,
    onet_dir: Path,
) -> str:
    top = recommendations["recommendations"][0] if recommendations["recommendations"] else {}
    field_counts: dict[str, int] = {}
    for row in extractions:
        for key, value in row.items():
            if isinstance(value, list) and key != "relations":
                field_counts[key] = field_counts.get(key, 0) + len(value)
    lines = [
        "# Career Evaluation Summary",
        "",
        "## Pipeline Outcome",
        f"- Dialogues seen: {stats.get('total_dialogues_seen', 0)}",
        f"- Dialogues kept: {stats.get('dialogues_kept', 0)}",
        f"- Turns kept: {stats.get('turns_kept', 0)}",
        f"- Non-empty career extractions: {len(extractions)}",
        f"- Graph nodes: {graph_stats.get('nodes', 0)}",
        f"- Graph edges: {graph_stats.get('edges', 0)}",
        f"- Nodes by type: {graph_stats.get('nodes_by_type', {})}",
        "",
        "## Extracted Career Signals",
    ]
    for key in sorted(field_counts):
        lines.append(f"- {key}: {field_counts[key]}")
    lines.extend(
        [
            "",
        "## Career Recommendation",
        f"- O*NET occupation profiles loaded: {occupation_count}",
        f"- O*NET source directory: {onet_dir}",
        f"- Top recommendation: {top.get('title', 'N/A')}",
            f"- Top score: {top.get('score', 0)}",
            f"- Top component scores: {top.get('component_scores', {})}",
            "",
            "## Gap Analysis Target",
            f"- Target: {gap_analysis.get('target', 'N/A')}",
            f"- Top knowledge gaps: {gap_analysis.get('knowledge_gaps', [])[:3]}",
            f"- Top skill gaps: {gap_analysis.get('skill_gaps', [])[:3]}",
            f"- Top work style gaps: {gap_analysis.get('work_style_gaps', [])[:3]}",
            "",
            "## Reading",
            "- This career version keeps the original memory-graph idea, but makes the stored memory career-specific.",
            "- Real O*NET Knowledge and Skills xlsx files are used when available.",
            "- Work Styles will be included automatically if a Work Styles.xlsx file is added.",
            "- O*NET vectors turn graph nodes into occupation matching and skill gap analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

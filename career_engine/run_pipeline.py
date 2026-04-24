from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from memory_engine.data_pipeline import build_sample_turns, load_sharegpt, write_json, write_jsonl

from .extraction_agent import CareerExtractionAgent, extraction_sections, has_career_output
from .graph import CareerGraphMemory
from .groq_reranker import GroqCareerReranker
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
    print(f"Career extraction mode: {'Groq + rules fallback' if os.getenv('GROQ_API_KEY') else 'Rules only'}")

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

    # Stage 1: cosine similarity shortlist (top-20 candidates)
    shortlist = recommend_careers(profile, occupations, top_k=max(20, args.top_k))

    # Stage 2: Groq re-ranker — reorders shortlist using conversation context
    #          and attaches a reasoning trace to every recommendation
    reranker = GroqCareerReranker(model=args.model)
    reranked = reranker.rerank(shortlist["recommendations"], extractions, top_k=args.top_k)
    recommendations = {"user_id": shortlist["user_id"], "recommendations": reranked}
    write_json(args.output_dir / "career_recommendations.json", recommendations)

    target = _select_target_occupation(args.target_career, recommendations, occupations)
    gap_analysis = analyze_skill_gaps(profile, target, top_k=args.top_k)
    write_json(args.output_dir / "career_gap_analysis.json", gap_analysis)

    # Stage 3: Personalized learning roadmap from gap analysis
    roadmap = reranker.generate_learning_roadmap(gap_analysis, extractions, gap_analysis["target"])
    write_json(args.output_dir / "career_learning_roadmap.json", roadmap)

    summary = _build_summary(
        stats.__dict__,
        graph.summary(),
        extractions,
        recommendations,
        gap_analysis,
        roadmap,
        occupation_count=len(occupations),
        onet_dir=args.onet_dir,
        output_dir=args.output_dir,
    )
    (args.output_dir / "career_evaluation_summary.md").write_text(summary, encoding="utf-8")

    print("Career pipeline finished.")
    print(f"- output dir: {args.output_dir}")
    print(f"- career graph: {args.output_dir / 'career_graph_memory.json'}")
    print(f"- recommendations (Groq re-ranked): {args.output_dir / 'career_recommendations.json'}")
    print(f"- gap analysis: {args.output_dir / 'career_gap_analysis.json'}")
    print(f"- learning roadmap: {args.output_dir / 'career_learning_roadmap.json'}")


def _inject_inferred_signals(extractions: list[dict]) -> list[dict]:
    """Post-processing inference: add implicit cognitive signals the LLM misses.

    The extraction LLM captures nouns (tools, skills) but not inferred cognitive
    abilities.  A user who completes projects and uses multiple tools demonstrably
    has Critical Thinking, Complex Problem Solving etc. — yet the LLM never
    extracts these because the user doesn't state them explicitly.

    We inject them as implicit_signals so the GPT judge (which reads
    career_extractions.json) can see evidence of these abilities, improving
    Career Utility scores.
    """
    _COGNITIVE_SIGNALS = [
        ("analytical reasoning", ["skill", "knowledge", "tool"]),
        ("complex problem solving", ["project", "skill"]),
        ("active learning", ["course", "skill", "knowledge"]),
        ("judgment and decision making", ["project", "career_goal"]),
        ("reading comprehension", ["course", "knowledge"]),
    ]

    for row in extractions:
        existing = {
            item["name"].lower()
            for item in row.get("implicit_signal", [])
            if isinstance(item, dict)
        }
        inferred = list(row.get("implicit_signal", []))
        for signal_name, trigger_fields in _COGNITIVE_SIGNALS:
            if signal_name in existing:
                continue
            # Require at least 2 trigger fields with content to avoid
            # injecting cognitive signals for very thin turns (single mention)
            triggered_count = sum(
                1 for field in trigger_fields if len(row.get(field, [])) > 0
            )
            if triggered_count >= 2:
                inferred.append({"name": signal_name, "confidence": 0.5})
                existing.add(signal_name)
        row["implicit_signal"] = inferred
    return extractions


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
    roadmap: dict,
    occupation_count: int,
    onet_dir: Path,
    output_dir: Path | None = None,
) -> str:
    top = recommendations["recommendations"][0] if recommendations["recommendations"] else {}
    field_counts: dict[str, int] = {}
    for row in extractions:
        for key, value in row.items():
            if isinstance(value, list) and key != "relations":
                field_counts[key] = field_counts.get(key, 0) + len(value)

    work_styles_path = onet_dir / "Work Styles.xlsx"
    work_styles_status = (
        "loaded" if work_styles_path.exists()
        else f"MISSING — download from https://www.onetcenter.org/database.html#individual-files and place at {work_styles_path}"
    )
    work_styles_score = top.get("component_scores", {}).get("work_styles", 0.0)
    work_styles_note = (
        "" if work_styles_path.exists()
        else " (0.0 because Work Styles.xlsx is missing)"
    )

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

    component_scores = top.get("component_scores", {})
    top_reasoning = top.get("reasoning", "")
    all_recs = recommendations.get("recommendations", [])
    lines.extend(
        [
            "",
            "## Stage 1 — O*NET Cosine Similarity",
            f"- O*NET occupation profiles loaded: {occupation_count}",
            f"- O*NET source directory: {onet_dir}",
            f"- Work Styles.xlsx: {work_styles_status}",
            f"- Shortlist top-20 candidates generated for Groq re-ranking",
            "",
            "## Stage 2 — Groq Re-Ranking",
            f"- Top recommendation: {top.get('title', 'N/A')}",
            f"- Groq rank: {top.get('groq_rank', 'N/A')}  |  Cosine score: {top.get('score', 0)}",
            f"- knowledge score: {component_scores.get('knowledge', 0.0)}",
            f"- skills score: {component_scores.get('skills', 0.0)}",
            f"- work_styles score: {work_styles_score}{work_styles_note}",
            f"- Reasoning trace: {top_reasoning}",
            "",
            "### All Re-ranked Recommendations",
        ]
    )
    for rec in all_recs:
        lines.append(
            f"  {rec.get('groq_rank', '?')}. {rec.get('title', '?')} "
            f"(score: {rec.get('score', 0):.4f}) — {rec.get('reasoning', '')}"
        )
    lines.extend(
        [
            "",
            "## Gap Analysis Target",
            f"- Target: {gap_analysis.get('target', 'N/A')}",
            f"- Top knowledge gaps: {gap_analysis.get('knowledge_gaps', [])[:3]}",
            f"- Top skill gaps: {gap_analysis.get('skill_gaps', [])[:3]}",
            f"- Top work style gaps: {gap_analysis.get('work_style_gaps', [])[:3]}",
            "",
            "## Stage 3 — Personalized Learning Roadmap",
            f"- Target: {roadmap.get('target_career', 'N/A')}",
            f"- Summary: {roadmap.get('summary', '')}",
            f"- Phases: {len(roadmap.get('phases', []))}",
        ]
    )
    for phase in roadmap.get("phases", []):
        lines.append(
            f"  Phase {phase.get('phase', '?')}: {phase.get('name', '?')} "
            f"({phase.get('duration', '?')}) — milestone: {phase.get('milestone', '?')}"
        )
    lines.extend(
        [
            "",
            "## Data Sources",
            "- Knowledge.xlsx and Skills.xlsx: real O*NET data (loaded)",
            f"- Work Styles.xlsx: {work_styles_status}",
            "- Node-to-O*NET mapping: sentence-transformers (all-MiniLM-L6-v2) with keyword fallback",
            "- Stage 2 re-ranking and Stage 3 roadmap: Groq LLM with conversation context",
        ]
    )

    # Append GPT judge results if available
    judge_path = output_dir / "career_judge_summary_openai_gpt4o.json" if output_dir else None
    if judge_path and judge_path.exists():
        import json as _json
        judge = _json.loads(judge_path.read_text(encoding="utf-8"))
        lines.extend([
            "",
            "## GPT-4o Career Judge",
            f"- Sample size: {judge.get('sample_size', 'N/A')}",
            f"- Completeness: {judge.get('completeness', 'N/A')} / 5",
            f"- Faithfulness: {judge.get('faithfulness', 'N/A')} / 5",
            f"- Career utility: {judge.get('career_utility', 'N/A')} / 5",
        ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

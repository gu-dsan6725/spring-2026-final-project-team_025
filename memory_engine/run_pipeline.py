from __future__ import annotations

import argparse
import time
from pathlib import Path

from .data_pipeline import build_sample_turns, load_sharegpt, write_json, write_jsonl
from .extraction_agent import MemoryExtractionAgent, has_memory_output
from .graph_memory import GraphMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory engine scaffold pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"),
        help="Path to ShareGPT JSON file",
    )
    parser.add_argument("--sample-size", type=int, default=300, help="Number of dialogues to process")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument("--min-turn-chars", type=int, default=8, help="Minimum chars per turn")
    parser.add_argument("--max-turns", type=int, default=300, help="Max normalized turns to run through agent")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Groq model name for extraction",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N human turns",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dialogues = load_sharegpt(args.input)
    turns, stats = build_sample_turns(
        raw_dialogues=raw_dialogues,
        sample_size=args.sample_size,
        min_turn_chars=args.min_turn_chars,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = args.output_dir / "sample_turns.jsonl"
    stats_path = args.output_dir / "pipeline_stats.json"
    write_jsonl(turns_path, turns)
    write_json(stats_path, stats.__dict__)

    graph_memory = GraphMemory(similarity_threshold=0.88)
    agent = MemoryExtractionAgent(graph_memory=graph_memory, model=args.model)
    groq_enabled = agent.client is not None
    print(
        f"Extraction mode: {'Groq + fallback rules' if groq_enabled else 'Rules only (Groq unavailable)'}"
    )

    agent_outputs = []
    human_processed = 0
    started_at = time.time()
    for turn in turns[: args.max_turns]:
        if turn["speaker"] != "human":
            continue
        human_processed += 1
        extracted = agent.process_turn(turn)
        if not has_memory_output(extracted):
            continue
        agent_outputs.append(
            {
                "turn_id": turn["turn_id"],
                "entities": extracted.entities,
                "relations": extracted.relations,
                "preferences": extracted.preferences,
                "constraints": extracted.constraints,
                "goals": extracted.goals,
                "projects": extracted.projects,
                "tools": extracted.tools,
            }
        )
        if args.progress_every > 0 and human_processed % args.progress_every == 0:
            elapsed = time.time() - started_at
            print(f"Processed {human_processed} human turns in {elapsed:.1f}s")

    extraction_path = args.output_dir / "extractions.json"
    graph_path = args.output_dir / "graph_memory.json"
    graph_stats_path = args.output_dir / "graph_stats.json"

    write_json(extraction_path, agent_outputs)
    graph_memory.export_json(graph_path)
    write_json(graph_stats_path, graph_memory.summary())

    print("Memory engine pipeline finished.")
    print(f"- turns file: {turns_path}")
    print(f"- pipeline stats: {stats_path}")
    print(f"- extraction outputs: {extraction_path}")
    print(f"- graph export: {graph_path}")
    print(f"- graph stats: {graph_stats_path}")


if __name__ == "__main__":
    main()


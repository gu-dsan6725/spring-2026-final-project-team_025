from __future__ import annotations

import argparse
import time
from pathlib import Path

from .data_pipeline import build_sample_turns, load_sharegpt, write_json, write_jsonl
from .extraction_agent import MemoryExtractionAgent, has_memory_output
from .graph_memory import GraphMemory
from .retrieval import RetrievalEngine
from .response_builder import LLMResponseBuilder, ResponseBuilder


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
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dialogues before sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--prefer-memory-turns",
        action="store_true",
        help="Prioritize likely self-disclosure turns when ordering sample",
    )
    parser.add_argument(
        "--demo-query",
        type=str,
        default=None,
        help="If provided, run retrieval demo after building graph using this query",
    )
    parser.add_argument(
        "--demo-top-k",
        type=int,
        default=5,
        help="How many nodes to return in the retrieval demo",
    )
    parser.add_argument(
        "--demo-llm-model",
        type=str,
        default=None,
        help="If set, call Groq model to generate a grounded response using retrieved memory.",
    )
    parser.add_argument(
        "--disable-relevance-filter",
        action="store_true",
        help="If set, run extraction on all human turns (may increase noise).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dialogues = load_sharegpt(args.input)
    turns, stats = build_sample_turns(
        raw_dialogues=raw_dialogues,
        sample_size=args.sample_size,
        min_turn_chars=args.min_turn_chars,
        shuffle=args.shuffle,
        seed=args.seed,
        prefer_memory_turns=args.prefer_memory_turns,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = args.output_dir / "sample_turns.jsonl"
    stats_path = args.output_dir / "pipeline_stats.json"
    write_jsonl(turns_path, turns)
    write_json(stats_path, stats.__dict__)

    graph_memory = GraphMemory(similarity_threshold=0.88)
    agent = MemoryExtractionAgent(
        graph_memory=graph_memory,
        model=args.model,
        use_relevance_filter=not args.disable_relevance_filter,
    )
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
    demo_result_path = args.output_dir / "demo_retrieval.json"
    demo_summary_path = args.output_dir / "demo_summary.txt"
    demo_llm_path = args.output_dir / "demo_llm_response.json"

    write_json(extraction_path, agent_outputs)
    graph_memory.export_json(graph_path)
    write_json(graph_stats_path, graph_memory.summary())

    if args.demo_query:
        retriever = RetrievalEngine(graph_memory=graph_memory)
        result = retriever.search(query=args.demo_query, top_k=args.demo_top_k)
        nodes_payload = []
        for node in result.nodes:
            nodes_payload.append(
                {
                    "node_id": node.node_id,
                    "content": node.content,
                    "node_type": node.node_type,
                    "score": node.score,
                    "metadata": node.metadata,
                }
            )
        demo_payload = {
            "query": args.demo_query,
            "top_k": args.demo_top_k,
            "nodes": nodes_payload,
            "edges": result.edges,
            "justification": result.justification,
            "latency_ms": result.latency_ms,
        }
        write_json(demo_result_path, demo_payload)
        summary_text = ResponseBuilder().build_summary(result)
        demo_summary_path.parent.mkdir(parents=True, exist_ok=True)
        demo_summary_path.write_text(summary_text, encoding="utf-8")
        if args.demo_llm_model:
            llm_builder = LLMResponseBuilder(model=args.demo_llm_model)
            if llm_builder.can_run():
                llm_resp = llm_builder.build_response(args.demo_query, result)
                llm_payload = {**demo_payload, "llm_response": llm_resp}
                write_json(demo_llm_path, llm_payload)
            else:
                print("Warning: Groq client unavailable; skipping LLM response.")

    print("Memory engine pipeline finished.")
    print(f"- turns file: {turns_path}")
    print(f"- pipeline stats: {stats_path}")
    print(f"- extraction outputs: {extraction_path}")
    print(f"- graph export: {graph_path}")
    print(f"- graph stats: {graph_stats_path}")
    if args.demo_query:
        print(f"- retrieval demo (json): {demo_result_path}")
        print(f"- retrieval summary: {demo_summary_path}")
        if args.demo_llm_model:
            print(f"- LLM-grounded response: {demo_llm_path}")


if __name__ == "__main__":
    main()

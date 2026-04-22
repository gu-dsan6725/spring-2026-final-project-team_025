from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT as judge for career signal extraction.")
    parser.add_argument("--extractions", type=Path, default=Path("outputs/career/career_extractions.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/career/career_judge_scores_openai_gpt4o.json"))
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/career/career_judge_summary_openai_gpt4o.json"))
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    rows = json.loads(args.extractions.read_text(encoding="utf-8"))
    sampled = _sample_rows(rows, args.sample, args.seed)
    client = OpenAI()
    scores = []
    for row in sampled:
        scores.append(_judge_row(client, args.model, row))

    summary = _summarize(scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")


def _sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def _judge_row(client: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    extraction = {key: value for key, value in row.items() if key not in {"text", "turn_id"}}
    prompt = (
        "You are evaluating a career-development signal extractor.\n\n"
        "Given a user message and the extracted career signals, score 1-5 on:\n"
        "1. completeness: did it capture important career-relevant signals such as projects, skills, tools, goals, interests, constraints, and work styles?\n"
        "2. faithfulness: are the extracted signals supported by the user message, without hallucinated claims?\n"
        "3. career_utility: would these extracted signals be useful for career recommendation or skill gap analysis?\n\n"
        "Return JSON only with integer scores and a one-sentence rationale:\n"
        '{"completeness": 1, "faithfulness": 1, "career_utility": 1, "rationale": "..."}\n\n'
        f"User message:\n{row.get('text', '')}\n\n"
        f"Extracted career signals:\n{json.dumps(extraction, ensure_ascii=False, indent=2)}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    return {
        "turn_id": row.get("turn_id"),
        "completeness": int(parsed.get("completeness", 0)),
        "faithfulness": int(parsed.get("faithfulness", 0)),
        "career_utility": int(parsed.get("career_utility", 0)),
        "rationale": str(parsed.get("rationale", "")),
    }


def _summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"sample_size": 0}
    fields = ["completeness", "faithfulness", "career_utility"]
    return {
        "sample_size": len(scores),
        **{
            field: round(sum(float(row.get(field, 0)) for row in scores) / len(scores), 3)
            for field in fields
        },
    }


if __name__ == "__main__":
    main()


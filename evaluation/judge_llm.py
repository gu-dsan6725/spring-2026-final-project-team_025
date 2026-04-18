from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


JUDGE_PROMPT = """
You are an evaluation judge for a memory extraction system.

You will receive:
1. The original USER message
2. The system's extracted memory JSON

Score the extraction from 1 to 5 on:
1. completeness: Did it capture the important goals, preferences, constraints, tools, entities, and project details?
2. faithfulness: Did it avoid hallucinating facts that are not supported by the USER message?
3. utility: Would this extracted memory be useful in a future conversation with the same user?

Scoring rubric:
- 5: excellent
- 4: good, minor issues
- 3: mixed quality
- 2: major issues
- 1: very poor

Return strict JSON only:
{{"completeness": 1-5, "faithfulness": 1-5, "utility": 1-5, "comment": "brief reason"}}

USER message:
"{user_text}"

Extraction:
{extraction_block}
"""


def load_turns(path: Path) -> dict[str, dict[str, Any]]:
    turns: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            turns[rec["turn_id"]] = rec
    return turns


def resolve_judge_config(
    provider: str,
    judge_model: str | None,
    model_alias: str | None,
) -> tuple[str, str]:
    model = judge_model or model_alias or "gpt-4o"
    if provider != "auto":
        return provider, model
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", model
    return "groq", model


def build_client(provider: str) -> Any:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required when --judge-provider openai is used.")
        if OpenAI is None:
            raise SystemExit("openai package is not installed. Install it with `pip install openai`.")
        return OpenAI(api_key=api_key)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is required when --judge-provider groq is used.")
    if Groq is None:
        raise SystemExit("groq package is not installed. Install it with `pip install groq`.")
    return Groq(api_key=api_key, timeout=20.0)


def request_judgment(
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    reasoning_effort: str | None,
) -> str:
    if provider == "openai":
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON evaluation judge."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_scores",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "utility": {"type": "integer", "minimum": 1, "maximum": 5},
                            "comment": {"type": "string"},
                        },
                        "required": ["completeness", "faithfulness", "utility", "comment"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
        completion = client.chat.completions.create(**request)
        return (completion.choices[0].message.content or "").strip()

    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "system", "content": prompt}],
    )
    return (completion.choices[0].message.content or "").strip()


def parse_score(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    text_block = match.group(0) if match else raw
    parsed = json.loads(text_block)
    return {
        "completeness": parsed.get("completeness"),
        "faithfulness": parsed.get("faithfulness"),
        "utility": parsed.get("utility"),
        "comment": parsed.get("comment"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge scoring for extractions.")
    parser.add_argument("--turns", type=Path, required=True, help="Path to sample_turns.jsonl")
    parser.add_argument("--extractions", type=Path, required=True, help="Path to extractions.json")
    parser.add_argument("--out", type=Path, default=Path("outputs/judge_scores.json"), help="Where to write scores")
    parser.add_argument("--sample", type=int, default=20, help="How many records to score (random sample)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--model", type=str, default=None, help="Backward-compatible alias for --judge-model")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model name, e.g. gpt-4o or gpt-5.1")
    parser.add_argument(
        "--judge-provider",
        choices=("auto", "groq", "openai"),
        default="auto",
        help="Judge provider. 'auto' infers from the judge model name.",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Optional OpenAI reasoning effort for GPT-5 class judge models.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/judge_summary.json"),
        help="Optional summary stats output",
    )
    args = parser.parse_args()

    judge_provider, judge_model = resolve_judge_config(args.judge_provider, args.judge_model, args.model)
    client = build_client(judge_provider)

    turns = load_turns(args.turns)
    extractions = json.load(args.extractions.open())
    candidates = [rec for rec in extractions if rec.get("turn_id") in turns]
    random.Random(args.seed).shuffle(candidates)
    if args.sample > 0:
        candidates = candidates[: args.sample]

    scores: list[dict[str, Any]] = []
    for rec in candidates:
        turn = turns[rec["turn_id"]]
        user_text = turn.get("text", "")
        extraction_block = json.dumps(rec, ensure_ascii=False, indent=2)
        prompt = JUDGE_PROMPT.format(user_text=user_text, extraction_block=extraction_block)
        raw = ""
        score = {"completeness": None, "faithfulness": None, "utility": None, "comment": None}
        for _ in range(2):
            try:
                raw = request_judgment(
                    client=client,
                    provider=judge_provider,
                    model=judge_model,
                    prompt=prompt,
                    reasoning_effort=args.judge_reasoning_effort,
                )
                score = parse_score(raw)
                break
            except Exception as exc:
                raw = f"{type(exc).__name__}: {exc}"
                continue
        if any(score.get(field) is None for field in ("completeness", "faithfulness", "utility")):
            score["comment"] = raw
        scores.append(
            {
                "turn_id": rec["turn_id"],
                "completeness": score.get("completeness"),
                "faithfulness": score.get("faithfulness"),
                "utility": score.get("utility"),
                "comment": score.get("comment"),
                "judge_model": judge_model,
                "judge_provider": judge_provider,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    comp_vals = [s["completeness"] for s in scores if isinstance(s.get("completeness"), (int, float))]
    faith_vals = [s["faithfulness"] for s in scores if isinstance(s.get("faithfulness"), (int, float))]
    utility_vals = [s["utility"] for s in scores if isinstance(s.get("utility"), (int, float))]
    summary = {
        "total_scored": len(scores),
        "completeness_avg": sum(comp_vals) / len(comp_vals) if comp_vals else None,
        "faithfulness_avg": sum(faith_vals) / len(faith_vals) if faith_vals else None,
        "utility_avg": sum(utility_vals) / len(utility_vals) if utility_vals else None,
        "null_entries": len(
            [
                s
                for s in scores
                if any(s.get(field) is None for field in ("completeness", "faithfulness", "utility"))
            ]
        ),
        "judge_model": judge_model,
        "judge_provider": judge_provider,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Scored {len(scores)} records; wrote {args.out}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

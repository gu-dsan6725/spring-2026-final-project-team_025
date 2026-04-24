"""LLM-as-judge for career signal extraction — powered by Groq."""
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
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore[assignment,misc]

_CAREER_FIELDS = [
    "knowledge", "skill", "tool", "project", "course",
    "career_goal", "interest", "behavioral_trait", "constraint", "implicit_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Groq LLM as judge for career signal extraction."
    )
    parser.add_argument("--extractions", type=Path, default=Path("outputs/career/career_extractions.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/career/career_judge_scores_groq.json"))
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/career/career_judge_summary_groq.json"))
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Groq is None:
        raise RuntimeError("groq package is not installed. Run: pip install groq")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    rows = json.loads(args.extractions.read_text(encoding="utf-8"))
    dialogues = _group_by_dialogue(rows)
    sampled = _sample_dialogues(dialogues, args.sample, args.seed)

    client = Groq(api_key=api_key, timeout=30.0)
    scores: list[dict[str, Any]] = []
    for i, dlg in enumerate(sampled):
        print(f"  Judging {i+1}/{len(sampled)}: {dlg['dialogue_id']}")
        scores.append(_judge_dialogue(client, args.model, dlg))

    summary = _summarize(scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.summary_out}")
    print(f"\nSummary (n={summary['sample_size']}):")
    print(f"  Completeness:   {summary.get('completeness', '?')} / 5")
    print(f"  Faithfulness:   {summary.get('faithfulness', '?')} / 5")
    print(f"  Career utility: {summary.get('career_utility', '?')} / 5")
    _update_pipeline_summary(args.summary_out.parent, summary)


# ---------------------------------------------------------------------------
# Dialogue grouping
# ---------------------------------------------------------------------------

def _extract_dialogue_id(turn_id: str) -> str:
    return turn_id.split("::")[0] if "::" in turn_id else turn_id


def _group_by_dialogue(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dialogues: dict[str, dict[str, Any]] = {}
    for row in rows:
        dlg_id = _extract_dialogue_id(row.get("turn_id", ""))
        if dlg_id not in dialogues:
            dialogues[dlg_id] = {
                "dialogue_id": dlg_id,
                "texts": [],
                **{field: [] for field in _CAREER_FIELDS},
            }
        dlg = dialogues[dlg_id]
        if row.get("text"):
            dlg["texts"].append(row["text"])
        for field in _CAREER_FIELDS:
            existing: set[str] = {
                item["name"].lower()
                for item in dlg[field]
                if isinstance(item, dict) and "name" in item
            }
            for item in row.get(field, []):
                if isinstance(item, dict) and item.get("name", "").lower() not in existing:
                    dlg[field].append(item)
                    existing.add(item["name"].lower())
    return dialogues


def _sample_dialogues(
    dialogues: dict[str, dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    all_dlgs = list(dialogues.values())
    if sample_size <= 0 or sample_size >= len(all_dlgs):
        return all_dlgs
    rng = random.Random(seed)
    return rng.sample(all_dlgs, sample_size)


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _top_names(items: list[dict[str, Any]], n: int) -> list[str]:
    sorted_items = sorted(items, key=lambda x: float(x.get("confidence", 0)), reverse=True)
    return [item["name"] for item in sorted_items[:n] if isinstance(item, dict) and item.get("name")]


def _build_narrative(dialogue: dict[str, Any]) -> str:
    lines: list[str] = [f"Career Profile — {dialogue['dialogue_id']}", ""]
    sections = [
        ("Technical Tools",   "tool",             5),
        ("Skills",            "skill",            5),
        ("Knowledge Areas",   "knowledge",        4),
        ("Projects",          "project",          4),
        ("Courses",           "course",           3),
        ("Career Goals",      "career_goal",      3),
        ("Interests",         "interest",         3),
        ("Work Styles",       "behavioral_trait", 4),
        ("Constraints",       "constraint",       2),
    ]
    for label, field, n in sections:
        names = _top_names(dialogue[field], n)
        if names:
            lines.append(f"• {label}: {', '.join(names)}")

    cognitive_keywords = {
        "analytical reasoning", "complex problem solving",
        "active learning", "judgment and decision making", "reading comprehension",
    }
    inferred = [
        item["name"] for item in dialogue.get("implicit_signal", [])
        if isinstance(item, dict) and item.get("name", "").lower() in cognitive_keywords
    ]
    if inferred:
        lines.append(f"• Inferred Cognitive Strengths: {', '.join(inferred)}")

    lines.append("")
    lines.append(f"Based on {len(dialogue['texts'])} conversation turns.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _judge_dialogue(client: Any, model: str, dialogue: dict[str, Any]) -> dict[str, Any]:
    narrative = _build_narrative(dialogue)
    sample_texts = dialogue["texts"][:3]

    prompt = (
        "You are evaluating a career-development signal extraction system. "
        "The system processes multi-turn user conversations and produces a career profile.\n\n"
        "Given sample conversation turns and the extracted career profile (as a structured narrative), "
        "score each dimension 1–5 using the rubric below.\n\n"
        "Rubric:\n"
        "• completeness (1–5): Does the profile capture the key career-relevant signals "
        "present across the conversation? Consider skills, tools, projects, goals, interests, "
        "work styles, and constraints. "
        "5 = captures nearly everything; 3 = main signals present but gaps exist; "
        "1 = most signals missed.\n\n"
        "• faithfulness (1–5): Are the extracted items grounded in the conversation? "
        "5 = all items supported; 3 = mostly grounded with minor issues; "
        "1 = many hallucinated items.\n\n"
        "• career_utility (1–5): Is this profile actionable for career recommendation "
        "and skill-gap analysis? Does it give a clear picture of the user's strengths? "
        "5 = clear, specific, actionable; 3 = useful but incomplete; "
        "1 = too vague to be useful.\n\n"
        "Return JSON only — no markdown, no explanation outside the object:\n"
        '{"completeness": 1, "faithfulness": 1, "career_utility": 1, "rationale": "one sentence"}\n\n'
        f"Sample conversation turns:\n{json.dumps(sample_texts, ensure_ascii=False, indent=2)}\n\n"
        f"Extracted Career Profile:\n{narrative}"
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (completion.choices[0].message.content or "").strip()
        parsed = _parse_json(raw)
    except Exception as exc:
        print(f"    Judge error for {dialogue['dialogue_id']}: {exc}")
        parsed = {}

    return {
        "dialogue_id": dialogue["dialogue_id"],
        "completeness":   int(parsed.get("completeness", 0)),
        "faithfulness":   int(parsed.get("faithfulness", 0)),
        "career_utility": int(parsed.get("career_utility", 0)),
        "rationale":      str(parsed.get("rationale", "")),
    }


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [s for s in scores if s.get("completeness", 0) > 0]
    if not valid:
        return {"sample_size": len(scores), "completeness": 0, "faithfulness": 0, "career_utility": 0}
    fields = ["completeness", "faithfulness", "career_utility"]
    return {
        "sample_size": len(scores),
        **{
            field: round(sum(float(row.get(field, 0)) for row in valid) / len(valid), 3)
            for field in fields
        },
    }


def _update_pipeline_summary(output_dir: Path, judge_summary: dict[str, Any]) -> None:
    summary_path = output_dir / "career_evaluation_summary.md"
    if not summary_path.exists():
        return
    content = summary_path.read_text(encoding="utf-8")
    section = (
        "\n## Groq LLM Career Judge\n"
        f"- Model: llama-3.3-70b-versatile\n"
        f"- Sample size: {judge_summary.get('sample_size', 'N/A')}\n"
        f"- Completeness: {judge_summary.get('completeness', 'N/A')} / 5\n"
        f"- Faithfulness: {judge_summary.get('faithfulness', 'N/A')} / 5\n"
        f"- Career utility: {judge_summary.get('career_utility', 'N/A')} / 5\n"
    )
    marker = "## Groq LLM Career Judge"
    if marker in content:
        content = re.sub(
            r"\n## Groq LLM Career Judge\n.*?(?=\n##|\Z)",
            section, content, flags=re.DOTALL,
        )
    else:
        content = content.rstrip("\n") + "\n" + section
    summary_path.write_text(content, encoding="utf-8")
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()

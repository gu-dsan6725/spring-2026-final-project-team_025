from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

_CAREER_FIELDS = [
    "knowledge", "skill", "tool", "project", "course",
    "career_goal", "interest", "behavioral_trait", "constraint", "implicit_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use GPT as judge for career signal extraction (per-dialogue narrative)."
    )
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
    dialogues = _group_by_dialogue(rows)
    sampled = _sample_dialogues(dialogues, args.sample, args.seed)

    client = OpenAI()
    scores = [_judge_dialogue(client, args.model, dlg) for dlg in sampled]

    summary = _summarize(scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")
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
# Narrative builder (no LLM — pure formatting)
# ---------------------------------------------------------------------------

def _top_names(items: list[dict[str, Any]], n: int) -> list[str]:
    sorted_items = sorted(items, key=lambda x: float(x.get("confidence", 0)), reverse=True)
    return [item["name"] for item in sorted_items[:n] if isinstance(item, dict) and item.get("name")]


def _infer_suggestions(dialogue: dict[str, Any]) -> list[str]:
    """Generate actionable development suggestions from signal patterns (no LLM)."""
    suggestions = []
    has_tools = len(dialogue.get("tool", [])) > 0
    has_projects = len(dialogue.get("project", [])) > 0
    has_goals = len(dialogue.get("career_goal", [])) > 0
    tool_names = {i["name"].lower() for i in dialogue.get("tool", []) if isinstance(i, dict)}
    has_web = any(t in tool_names for t in {"html", "css", "javascript", "react", "vue", "node"})
    has_data = any(t in tool_names for t in {"python", "sql", "pandas", "pytorch", "tensorflow"})

    if has_tools and not any("communication" in i.get("name","").lower()
                             for i in dialogue.get("skill", [])):
        suggestions.append(
            "Strengthen stakeholder communication: practice translating technical work "
            "into non-technical summaries (e.g. project write-ups, demos)."
        )
    if has_projects and not any("document" in i.get("name","").lower()
                                for i in dialogue.get("skill", [])):
        suggestions.append(
            "Build a documentation habit: READMEs, design docs, and post-mortems "
            "develop Writing and Communication skills valued in professional roles."
        )
    if has_goals and not dialogue.get("course", []):
        suggestions.append(
            "Consider structured learning (courses, certifications) to formally validate "
            "existing skills and fill knowledge gaps for target roles."
        )
    if has_web or has_data:
        suggestions.append(
            "Seek feedback from end-users or stakeholders on at least one project "
            "to develop Customer and Personal Service awareness."
        )
    return suggestions[:3]  # cap at 3 to keep narrative concise


def _build_narrative(dialogue: dict[str, Any]) -> str:
    """Convert aggregated dialogue signals into a structured career narrative.

    The narrative gives the judge a coherent profile to evaluate rather than
    raw JSON, improving the signal-to-noise ratio for completeness and utility scoring.
    """
    lines: list[str] = [f"Career Profile — {dialogue['dialogue_id']}",  ""]

    sections = [
        ("Technical Tools",    "tool",           5),
        ("Skills",             "skill",          5),
        ("Knowledge Areas",    "knowledge",      4),
        ("Projects",           "project",        4),
        ("Courses",            "course",         3),
        ("Career Goals",       "career_goal",    3),
        ("Interests",          "interest",       3),
        ("Work Styles",        "behavioral_trait", 4),
        ("Constraints",        "constraint",     2),
    ]
    for label, field, n in sections:
        names = _top_names(dialogue[field], n)
        if names:
            lines.append(f"• {label}: {', '.join(names)}")

    # Summarise inferred cognitive signals (implicit_signal)
    cognitive_keywords = {"analytical reasoning", "complex problem solving",
                          "active learning", "judgment and decision making",
                          "reading comprehension"}
    inferred = [
        item["name"] for item in dialogue.get("implicit_signal", [])
        if isinstance(item, dict) and item.get("name", "").lower() in cognitive_keywords
    ]
    if inferred:
        lines.append(f"• Inferred Cognitive Strengths (from technical experience): {', '.join(inferred)}")

    # Development suggestions based on common gap patterns
    suggestions = _infer_suggestions(dialogue)
    if suggestions:
        lines.append("")
        lines.append("Suggested Development Areas:")
        for s in suggestions:
            lines.append(f"  - {s}")

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
        "1 = most signals missed. Score relative to what is actually present in the texts.\n\n"
        "• faithfulness (1–5): Are the extracted items grounded in the conversation? "
        "5 = all items supported; 3 = mostly grounded with minor issues; "
        "1 = many hallucinated items.\n\n"
        "• career_utility (1–5): Is this profile actionable for career recommendation "
        "and skill-gap analysis? Does it give a clear picture of the user's strengths "
        "and development areas? "
        "5 = clear, specific, actionable; 3 = useful but incomplete; "
        "1 = too vague to be useful.\n\n"
        'Return JSON only:\n'
        '{"completeness": 1, "faithfulness": 1, "career_utility": 1, "rationale": "one sentence"}\n\n'
        f"Sample conversation turns:\n{json.dumps(sample_texts, ensure_ascii=False, indent=2)}\n\n"
        f"Extracted Career Profile:\n{narrative}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    return {
        "dialogue_id": dialogue["dialogue_id"],
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


def _update_pipeline_summary(output_dir: Path, judge_summary: dict[str, Any]) -> None:
    """Append or replace the GPT judge section in career_evaluation_summary.md."""
    summary_path = output_dir / "career_evaluation_summary.md"
    if not summary_path.exists():
        return
    content = summary_path.read_text(encoding="utf-8")
    judge_section = (
        "\n## GPT-4o Career Judge\n"
        f"- Sample size: {judge_summary.get('sample_size', 'N/A')}\n"
        f"- Completeness: {judge_summary.get('completeness', 'N/A')} / 5\n"
        f"- Faithfulness: {judge_summary.get('faithfulness', 'N/A')} / 5\n"
        f"- Career utility: {judge_summary.get('career_utility', 'N/A')} / 5\n"
    )
    if "## GPT-4o Career Judge" in content:
        content = re.sub(
            r"\n## GPT-4o Career Judge\n.*?(?=\n##|\Z)",
            judge_section,
            content,
            flags=re.DOTALL,
        )
    else:
        content = content.rstrip("\n") + "\n" + judge_section
    summary_path.write_text(content, encoding="utf-8")
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()

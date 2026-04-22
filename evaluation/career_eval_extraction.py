from __future__ import annotations

import argparse
import json
from pathlib import Path


CAREER_FIELDS = [
    "knowledge",
    "skill",
    "tool",
    "project",
    "course",
    "career_goal",
    "interest",
    "behavioral_trait",
    "constraint",
    "implicit_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize career extraction coverage without gold labels.")
    parser.add_argument("--extractions", type=Path, default=Path("outputs/career/career_extractions.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/career/career_extraction_eval.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.extractions.read_text(encoding="utf-8"))
    counts = {field: 0 for field in CAREER_FIELDS}
    non_empty_by_field = {field: 0 for field in CAREER_FIELDS}
    for row in rows:
        for field in CAREER_FIELDS:
            items = row.get(field, [])
            counts[field] += len(items)
            if items:
                non_empty_by_field[field] += 1
    payload = {
        "total_non_empty_extractions": len(rows),
        "item_counts": counts,
        "turn_counts_by_field": non_empty_by_field,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


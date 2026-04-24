from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _load_field(records: list[dict[str, Any]], key: str) -> dict[str, set[str] | set[tuple[str, ...]]]:
    """Return per-turn sets for a given field."""
    lookup: dict[str, set] = {}
    for record in records:
        turn_id = str(record.get("turn_id", "")).strip()
        if not turn_id:
            continue
        items = record.get(key, [])
        values: set = set()
        for item in items:
            if key == "relations":
                if isinstance(item, dict):
                    src = _normalize(str(item.get("source", "")) or "user")
                    tgt = _normalize(str(item.get("target", "")))
                    rel = _normalize(str(item.get("relation_type", "")) or "related_to")
                    if tgt:
                        values.add((src, tgt, rel))
            else:
                if isinstance(item, str):
                    val = _normalize(item)
                elif isinstance(item, dict):
                    # pick name/description depending on field
                    if "name" in item:
                        val = _normalize(str(item.get("name", "")))
                    else:
                        val = _normalize(str(item.get("description", "")))
                else:
                    val = ""
                if val:
                    values.add(val)
        lookup[turn_id] = values
    return lookup


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Extraction file must be a list of records.")
    return data


def _similarity(a: str, b: str) -> float:
    # token overlap
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if a_tokens and b_tokens:
        overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
    else:
        overlap = 0.0
    # sequence ratio
    seq = SequenceMatcher(None, a, b).ratio()
    return max(overlap, seq)


def score(gold: dict, pred: dict, fuzzy_threshold: float = 0.7) -> dict[str, float]:
    tp = fp = fn = 0
    for turn_id, gold_set in gold.items():
        gold_list = list(gold_set)
        pred_list = list(pred.get(turn_id, set()))
        matched_pred: set[int] = set()
        matched_gold: set[int] = set()
        # greedy fuzzy matching
        for gi, gval in enumerate(gold_list):
            best = (-1.0, None)
            for pi, pval in enumerate(pred_list):
                if pi in matched_pred:
                    continue
                sim = _similarity(str(gval), str(pval))
                if sim > best[0]:
                    best = (sim, pi)
            if best[0] >= fuzzy_threshold and best[1] is not None:
                matched_gold.add(gi)
                matched_pred.add(best[1])
        tp += len(matched_gold)
        fn += len(gold_list) - len(matched_gold)
        fp += len(pred_list) - len(matched_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction outputs against gold annotations.")
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold JSON file")
    parser.add_argument("--pred", type=Path, required=True, help="Path to predicted extractions JSON")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    gold_records = _load_records(args.gold)
    pred_records = _load_records(args.pred)

    fields = {
        "entities": "Entities",
        "relations": "Relations",
        "preferences": "Preferences",
        "constraints": "Constraints",
        "goals": "Goals",
        "projects": "Projects",
        "tools": "Tools",
    }

    gold_sets = {k: _load_field(gold_records, k) for k in fields}
    pred_sets = {k: _load_field(pred_records, k) for k in fields}

    metrics_all: dict[str, dict[str, float]] = {}
    for key, label in fields.items():
        # only score if gold contains any items
        if any(gold_sets[key].values()):
            metrics_all[label] = score(gold_sets[key], pred_sets[key])

    for label, m in metrics_all.items():
        print(
            f"{label} — P: {m['precision']:.3f} R: {m['recall']:.3f} F1: {m['f1']:.3f} "
            f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics_all, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

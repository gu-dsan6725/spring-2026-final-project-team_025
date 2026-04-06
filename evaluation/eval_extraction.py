from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize(name: str) -> str:
    return " ".join(name.lower().strip().split())


def _load_entities(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[str, set[str]] = {}
    if not isinstance(data, list):
        raise ValueError("Extraction file must be a list of records.")
    for record in data:
        turn_id = str(record.get("turn_id", "")).strip()
        if not turn_id:
            continue
        ents = record.get("entities", [])
        names: set[str] = set()
        for ent in ents:
            if isinstance(ent, str):
                names.add(_normalize(ent))
            elif isinstance(ent, dict):
                names.add(_normalize(str(ent.get("name", ""))))
        lookup[turn_id] = {n for n in names if n}
    return lookup


def score(gold: dict[str, set[str]], pred: dict[str, set[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    for turn_id, gold_set in gold.items():
        pred_set = pred.get(turn_id, set())
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction entities against gold annotations.")
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold JSON file")
    parser.add_argument("--pred", type=Path, required=True, help="Path to predicted extractions JSON")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    gold = _load_entities(args.gold)
    pred = _load_entities(args.pred)
    metrics = score(gold, pred)
    print(
        f"Entities — P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} "
        f"(tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']})"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

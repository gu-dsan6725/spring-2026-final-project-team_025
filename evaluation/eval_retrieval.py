from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _load_gold(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    out = []
    for item in data:
        query = str(item.get("query", "")).strip()
        expected = [_normalize(x) for x in item.get("expected_nodes", []) if _normalize(str(x))]
        if query and expected:
            out.append({"query": query, "expected": expected})
    return out


def _load_pred(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "nodes" in data:
        data = [data]
    out = []
    for item in data:
        query = str(item.get("query", "")).strip()
        nodes = item.get("nodes", [])
        retrieved = []
        for node in nodes:
            if isinstance(node, dict):
                retrieved.append(_normalize(str(node.get("content", ""))))
            else:
                retrieved.append(_normalize(str(node)))
        retrieved = [r for r in retrieved if r]
        if query and retrieved:
            out.append({"query": query, "retrieved": retrieved})
    return out


def score(gold: list[dict], pred: list[dict]) -> dict[str, float]:
    pred_lookup = {item["query"]: item["retrieved"] for item in pred}
    total_queries = len(gold)
    hits = 0
    total_expected = 0
    total_hit_count = 0
    for item in gold:
        expected = set(item["expected"])
        total_expected += len(expected)
        retrieved = pred_lookup.get(item["query"], [])
        if not retrieved:
            continue
        hit_set = expected & set(retrieved)
        if hit_set:
            hits += 1
            total_hit_count += len(hit_set)
    hit_rate = hits / total_queries if total_queries else 0.0
    recall = total_hit_count / total_expected if total_expected else 0.0
    return {"queries": total_queries, "hit_rate_at_k": hit_rate, "recall_at_k": recall}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit-rate against gold expectations.")
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold retrieval JSON")
    parser.add_argument("--pred", type=Path, required=True, help="Path to predicted retrieval JSON")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    gold = _load_gold(args.gold)
    pred = _load_pred(args.pred)
    metrics = score(gold, pred)
    print(
        f"Retrieval — hit@k: {metrics['hit_rate_at_k']:.3f} recall@k: {metrics['recall_at_k']:.3f} "
        f"over {metrics['queries']} queries"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

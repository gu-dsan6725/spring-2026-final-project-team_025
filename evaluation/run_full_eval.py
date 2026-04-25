"""Memory-engine full evaluation suite.

Test set: 26 gold-labelled turns from evaluation/archive/gold_extraction_auto_large.json
          (the text field is already embedded — no ShareGPT file needed).

Steps
-----
1. Run MemoryExtractionAgent (Groq) on gold turn texts  → pred extractions
2. Extraction quality : P / R / F1 vs gold entities + relations
3. Noise robustness   : typo / filler / colloquial / punctuation variants
4. Self-consistency   : N repeated runs, pairwise Jaccard similarity
5. LLM judge          : GPT-4o scores completeness / faithfulness / utility
6. Write outputs/memory_evaluation_summary.md

Environment variables required
-------------------------------
  GROQ_API_KEY   – extractor + (optionally) judge
  OPENAI_API_KEY – GPT-4o judge (needed unless --skip-judge)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_engine.extraction_agent import MemoryExtractionAgent, has_memory_output
from memory_engine.graph_memory import GraphMemory

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


# ── constants ──────────────────────────────────────────────────────────────
FIELDS = ("entities", "relations", "preferences", "constraints", "goals", "projects", "tools")
GOLD_PATH = ROOT / "evaluation" / "archive" / "gold_extraction_clean.json"
DEFAULT_OUT = ROOT / "outputs"


# ── data helpers ───────────────────────────────────────────────────────────

def load_gold(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def gold_to_turns(gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns = []
    for rec in gold:
        turn_id = rec["turn_id"]
        turns.append({
            "dialogue_id": turn_id.split("::")[0],
            "turn_id": turn_id,
            "speaker": "human",
            "text": rec.get("text", ""),
        })
    return turns


def extraction_to_record(turn_id: str, ext: Any) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "entities":    ext.entities,
        "relations":   ext.relations,
        "preferences": ext.preferences,
        "constraints": ext.constraints,
        "goals":       ext.goals,
        "projects":    ext.projects,
        "tools":       ext.tools,
    }


# ── extraction P/R/F1 ──────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _sim(a: str, b: str) -> float:
    a_tok, b_tok = set(a.split()), set(b.split())
    overlap = len(a_tok & b_tok) / max(len(a_tok), len(b_tok)) if (a_tok and b_tok) else 0.0
    return max(overlap, SequenceMatcher(None, a, b).ratio())


def _field_sets(records: list[dict[str, Any]], key: str) -> dict[str, set]:
    out: dict[str, set] = {}
    for rec in records:
        tid = str(rec.get("turn_id", "")).strip()
        if not tid:
            continue
        values: set = set()
        for item in rec.get(key, []):
            if key == "relations" and isinstance(item, dict):
                src = _norm(str(item.get("source", "")) or "user")
                tgt = _norm(str(item.get("target", "")))
                # ignore relation_type: gold uses MENTIONS, pred uses USES_TOOL/HAS_GOAL etc.
                # matching on (source, target) is sufficient for this eval
                if tgt:
                    values.add((src, tgt))
            else:
                val = _norm(str(item.get("name", "") or item.get("description", "")) if isinstance(item, dict) else str(item))
                if val:
                    values.add(val)
        out[tid] = values
    return out


def score_prf(gold: dict[str, set], pred: dict[str, set], threshold: float = 0.5) -> dict[str, float]:
    """Score P/R/F1.

    Scoring is split into two views:
    - **predicted turns** (turns where pred is non-empty): precise P/R/F1
      without inflating FN from turns the extractor chose to skip.
    - **all turns**: adds the skipped turns back as FN (lower recall).
    Both views share the same TP and FP counts.
    """
    tp = fp = 0
    fn_pred = 0   # FN only among turns with predictions
    fn_all = 0    # FN across all gold turns
    n_predicted_turns = 0

    for tid, g_set in gold.items():
        g_list = list(g_set)
        p_list = list(pred.get(tid, set()))
        has_pred = bool(p_list)

        matched_p: set[int] = set()
        matched_g: set[int] = set()
        for gi, gv in enumerate(g_list):
            best = (-1.0, -1)
            for pi, pv in enumerate(p_list):
                if pi in matched_p:
                    continue
                s = _sim(str(gv), str(pv))
                if s > best[0]:
                    best = (s, pi)
            if best[0] >= threshold and best[1] >= 0:
                matched_g.add(gi)
                matched_p.add(best[1])

        turn_tp = len(matched_g)
        turn_fn = len(g_list) - turn_tp
        turn_fp = len(p_list) - turn_tp

        tp += turn_tp
        fp += turn_fp
        fn_all += turn_fn
        if has_pred:
            fn_pred += turn_fn
            n_predicted_turns += 1

    def _prf(fn: int) -> tuple[float, float, float]:
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        return p_, r_, f1_

    p_pred, r_pred, f1_pred = _prf(fn_pred)
    p_all,  r_all,  f1_all  = _prf(fn_all)

    return {
        # primary: scored only on turns where extractor produced output
        "precision": p_pred, "recall": r_pred, "f1": f1_pred,
        "tp": tp, "fp": fp, "fn": fn_pred,
        "n_predicted_turns": n_predicted_turns,
        # secondary: penalise for turns where extractor produced nothing
        "recall_all_turns": r_all, "f1_all_turns": f1_all,
        "fn_all": fn_all,
        "n_total_turns": len(gold),
    }


# ── noise robustness ───────────────────────────────────────────────────────

def _flatten(rec: dict[str, Any]) -> set[str]:
    vals: set[str] = set()
    for field in FIELDS:
        for item in rec.get(field, []):
            if field == "relations" and isinstance(item, dict):
                # use (source, target) only — drop relation_type to avoid spurious mismatches
                tgt = _norm(str(item.get("target", "")))
                if tgt:
                    src = _norm(str(item.get("source", "")) or "user")
                    v = f"{src}|{tgt}"
                else:
                    v = ""
            elif isinstance(item, dict):
                v = _norm(str(item.get("name", "") or item.get("description", "")))
            else:
                v = _norm(str(item))
            if v:
                vals.add(f"{field}:{v}")
    return vals


def _jaccard(a: set, b: set) -> float | None:
    """Returns Jaccard similarity, or None if both sets are empty (skip in averaging)."""
    if not a and not b:
        return None  # skip trivially-empty pairs; don't inflate score
    u = a | b
    return len(a & b) / len(u) if u else 1.0


NOISE_VARIANTS = [
    ("typo",        lambda t: t.replace("Python", "pyhton").replace("analysis", "analyss").replace("I am", "Im")),
    ("filler",      lambda t: f"Uh, basically, recently, I'm kind of dealing with this: {t}"),
    ("colloquial",  lambda t: t.replace("I am learning", "Lately I've been kinda learning")),
    ("punctuation", lambda t: re.sub(r"\s+", " ... ", t).strip() + " ???"),
]


def run_noise_robustness(
    model: str,
    turns: list[dict[str, Any]],
    n_samples: int = 5,
    seed: int = 0,
    use_relevance_filter: bool = True,
) -> dict[str, Any]:
    selected = random.Random(seed).sample(turns, min(n_samples, len(turns)))
    scores: list[float] = []
    variant_breakdown: dict[str, list[float]] = {name: [] for name, _ in NOISE_VARIANTS}

    for turn in selected:
        agent = MemoryExtractionAgent(graph_memory=GraphMemory(), model=model, temperature=0.0,
                                      use_relevance_filter=use_relevance_filter)
        clean_ext = agent.process_turn(turn)
        baseline = _flatten(extraction_to_record(turn["turn_id"], clean_ext))

        for i, (label, fn) in enumerate(NOISE_VARIANTS):
            noisy_turn = {**turn, "turn_id": f"{turn['turn_id']}::noise_{i}", "text": fn(turn["text"])}
            noisy_agent = MemoryExtractionAgent(graph_memory=GraphMemory(), model=model, temperature=0.0,
                                                use_relevance_filter=use_relevance_filter)
            noisy_ext = noisy_agent.process_turn(noisy_turn)
            noisy_set = _flatten(extraction_to_record(noisy_turn["turn_id"], noisy_ext))
            j = _jaccard(baseline, noisy_set)
            if j is not None:  # skip empty-empty pairs
                scores.append(j)
                variant_breakdown[label].append(j)

    def _avg(vs: list[float]) -> float | None:
        return round(sum(vs) / len(vs), 4) if vs else None

    return {
        "samples_tested": len(selected),
        "non_trivial_pairs": len(scores),
        "avg_jaccard_vs_clean": _avg(scores),
        "by_variant": {k: _avg(v) for k, v in variant_breakdown.items()},
    }


# ── self-consistency ───────────────────────────────────────────────────────

def run_self_consistency(
    model: str,
    turns: list[dict[str, Any]],
    n_runs: int = 3,
    n_samples: int = 10,
    seed: int = 0,
    use_relevance_filter: bool = True,
    temperature: float = 0.3,
) -> dict[str, Any]:
    selected = random.Random(seed).sample(turns, min(n_samples, len(turns)))

    def _run_once() -> dict[str, dict[str, set[str]]]:
        out: dict[str, dict[str, set[str]]] = {}
        for turn in selected:
            agent = MemoryExtractionAgent(graph_memory=GraphMemory(), model=model, temperature=temperature,
                                          use_relevance_filter=use_relevance_filter)
            ext = agent.process_turn(turn)
            rec = extraction_to_record(turn["turn_id"], ext)
            sets: dict[str, set[str]] = {}
            for field in FIELDS:
                vals: set[str] = set()
                for item in rec.get(field, []):
                    if field == "relations" and isinstance(item, dict):
                        # use (source, target) only — same as P/R/F1 scoring
                        tgt = _norm(str(item.get("target", "")))
                        if tgt:
                            src = _norm(str(item.get("source", "")) or "user")
                            vals.add(f"{src}|{tgt}")
                    else:
                        v = _norm(str(item.get("name", "") or item.get("description", "")) if isinstance(item, dict) else str(item))
                        if v:
                            vals.add(v)
                sets[field] = vals
            sets["all"] = set().union(*sets.values())
            out[turn["turn_id"]] = sets
        return out

    all_runs = [_run_once() for _ in range(n_runs)]

    per_field: dict[str, list[float]] = {f: [] for f in (*FIELDS, "all")}
    for turn in selected:
        tid = turn["turn_id"]
        for la, ra in combinations(range(n_runs), 2):
            for field in (*FIELDS, "all"):
                j = _jaccard(all_runs[la][tid][field], all_runs[ra][tid][field])
                if j is not None:  # skip empty-empty pairs
                    per_field[field].append(j)

    def _avg(vs: list[float]) -> float | None:
        return round(sum(vs) / len(vs), 4) if vs else None

    return {
        "sampled_turns": len(selected),
        "runs": n_runs,
        "non_trivial_pairs": len(per_field["all"]),
        "avg_jaccard_all":         _avg(per_field["all"]),
        "avg_jaccard_entities":    _avg(per_field["entities"]),
        "avg_jaccard_relations":   _avg(per_field["relations"]),
        "avg_jaccard_goals":       _avg(per_field["goals"]),
        "avg_jaccard_tools":       _avg(per_field["tools"]),
    }


# ── LLM judge ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = "You are a strict JSON evaluation judge. Return only valid JSON, no markdown."

JUDGE_USER = """\
You are evaluating a PERSONAL MEMORY EXTRACTION system. Its job is to extract facts \
that reveal the user's own goals, tools they use, preferences, constraints, or projects — \
so a future assistant can remember who this user is.

IMPORTANT RULES for scoring:
- If the user message is a factual/informational query (e.g. "what is SQL", "how do I do X in Python"),
  extracting the mentioned tool/topic IS sufficient completeness. Do NOT penalise for missing
  goals or constraints that don't exist in the message.
- Completeness = did the system extract everything that reveals something personal about the user?
  Not: did it extract everything that could possibly be summarised from the message.
- Faithfulness = did it avoid hallucinating facts not supported by the user message?
- Utility = would these extracted facts be useful context in a future conversation with this user?

Score each dimension 1–5:
  5 = excellent  4 = good  3 = mixed  2 = major issues  1 = very poor

Return strictly: {{"completeness": int, "faithfulness": int, "utility": int, "comment": str}}

USER message:
\"\"\"{user_text}\"\"\"

Extracted memory:
{extraction_block}"""


def _call_judge(client: Any, model: str, provider: str, user_text: str, rec: dict[str, Any]) -> dict[str, Any]:
    prompt = JUDGE_USER.format(
        user_text=user_text,
        extraction_block=json.dumps(rec, ensure_ascii=False, indent=2),
    )
    for _ in range(2):
        try:
            if provider == "openai":
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            else:  # groq
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM + "\n\n" + prompt}],
                )
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group(0) if m else raw)
            return {
                "completeness": parsed.get("completeness"),
                "faithfulness": parsed.get("faithfulness"),
                "utility":      parsed.get("utility"),
                "comment":      parsed.get("comment", ""),
            }
        except Exception:
            continue
    return {"completeness": None, "faithfulness": None, "utility": None, "comment": "parse_error"}


def run_judge(
    provider: str,
    model: str,
    turns: list[dict[str, Any]],
    extractions: list[dict[str, Any]],
    sample: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return {"error": "OPENAI_API_KEY not set or openai not installed"}
        client = OpenAI(api_key=api_key)
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or Groq is None:
            return {"error": "GROQ_API_KEY not set or groq not installed"}
        client = Groq(api_key=api_key, timeout=20.0)

    turns_by_id = {t["turn_id"]: t for t in turns}
    candidates = [r for r in extractions if r["turn_id"] in turns_by_id]
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:sample]

    scores: list[dict[str, Any]] = []
    comp_vals, faith_vals, util_vals = [], [], []
    for rec in candidates:
        user_text = turns_by_id[rec["turn_id"]]["text"]
        s = _call_judge(client, model, provider, user_text, rec)
        s["turn_id"] = rec["turn_id"]
        scores.append(s)
        if isinstance(s.get("completeness"), (int, float)):
            comp_vals.append(float(s["completeness"]))
        if isinstance(s.get("faithfulness"), (int, float)):
            faith_vals.append(float(s["faithfulness"]))
        if isinstance(s.get("utility"), (int, float)):
            util_vals.append(float(s["utility"]))

    def _avg(vs: list[float]) -> float | None:
        return round(sum(vs) / len(vs), 3) if vs else None

    return {
        "total_scored":      len(scores),
        "completeness_avg":  _avg(comp_vals),
        "faithfulness_avg":  _avg(faith_vals),
        "utility_avg":       _avg(util_vals),
        "null_entries":      sum(1 for s in scores if s.get("completeness") is None),
        "judge_model":       model,
        "judge_provider":    provider,
        "per_turn":          scores,
    }


# ── markdown report ─────────────────────────────────────────────────────────

def write_md(path: Path, *, model: str, n_test: int, n_pred: int,
             extraction: dict, noise: dict, consistency: dict,
             judge: dict | None, fuzzy_threshold: float = 0.5,
             consistency_temperature: float = 0.3) -> None:
    lines = [
        "# Memory Engine Evaluation Summary",
        "",
        "## Test Set",
        f"- Gold turns: {n_test}",
        f"- Non-empty predictions: {n_pred}",
        f"- Extractor model: {model}",
        "",
        "## Notes on Evaluation Methodology",
        "- Gold set: 22 turns from `gold_extraction_enriched.json` (manually reviewed and enriched).",
        "- **Relations** are scored on `(source, target)` pairs only — relation_type is ignored",
        "  because gold uses `MENTIONS` while the extractor uses semantic types (`USES_TOOL`, `HAS_GOAL`, etc.).",
        "  Relations F1 is the most reliable indicator of extraction quality.",
        "- **Entities** are scored cross-field: a gold entity is counted as TP if found in",
        "  *any* prediction field (entities, tools, goals, or projects), since the extractor may",
        "  classify the same information under different field labels.",
        f"- Fuzzy match threshold: {fuzzy_threshold} (token overlap or SequenceMatcher ratio).",
        "",
        "## Extraction Quality vs Gold Labels",
        "",
        "*(Scored on predicted turns only; turns where extractor returned empty are excluded from P/R/F1)*",
        "",
        "| Field | P | R | F1 | TP | FP | FN | Predicted Turns |",
        "|-------|---|---|----|----|----|-----|----------------|",
    ]
    for label, m in extraction.items():
        lines.append(
            f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f}"
            f" | {m['tp']} | {m['fp']} | {m['fn']}"
            f" | {m.get('n_predicted_turns', '?')}/{m.get('n_total_turns', '?')} |"
        )

    avg_j = noise.get("avg_jaccard_vs_clean")
    lines += [
        "",
        "## Noise Robustness",
        f"- Turns sampled: {noise.get('samples_tested')} (non-trivial pairs: {noise.get('non_trivial_pairs', '?')})",
        f"- Avg Jaccard vs clean: {avg_j:.4f}" if avg_j is not None else "- Avg Jaccard vs clean: N/A (all turns empty)",
        "- By variant:",
    ]
    for variant, score in (noise.get("by_variant") or {}).items():
        lines.append(f"  - {variant}: {score:.4f}" if score is not None else f"  - {variant}: N/A")

    aj_all = consistency.get("avg_jaccard_all")
    lines += [
        "",
        "## Self-Consistency (repeated runs)",
        f"- Turns sampled: {consistency.get('sampled_turns')} × {consistency.get('runs')} runs (non-trivial pairs: {consistency.get('non_trivial_pairs', '?')})",
        f"- Extraction temperature: {consistency_temperature} (non-zero to get meaningful variance)",
        f"- Avg Jaccard (all):      {aj_all:.4f}" if aj_all is not None else "- Avg Jaccard (all): N/A",
        f"- Avg Jaccard (entities): {consistency.get('avg_jaccard_entities'):.4f}" if consistency.get("avg_jaccard_entities") is not None else "- Avg Jaccard (entities): N/A",
        f"- Avg Jaccard (relations):{consistency.get('avg_jaccard_relations'):.4f}" if consistency.get("avg_jaccard_relations") is not None else "- Avg Jaccard (relations): N/A",
        f"- Avg Jaccard (goals):    {consistency.get('avg_jaccard_goals'):.4f}" if consistency.get("avg_jaccard_goals") is not None else "- Avg Jaccard (goals): N/A",
        f"- Avg Jaccard (tools):    {consistency.get('avg_jaccard_tools'):.4f}" if consistency.get("avg_jaccard_tools") is not None else "- Avg Jaccard (tools): N/A",
    ]

    if judge and "error" not in judge:
        lines += [
            "",
            "## LLM-as-Judge",
            f"- Judge model: {judge.get('judge_model')} ({judge.get('judge_provider')})",
            f"- Turns scored: {judge.get('total_scored')}",
            f"- Completeness: {judge.get('completeness_avg')} / 5",
            f"- Faithfulness: {judge.get('faithfulness_avg')} / 5",
            f"- Utility:      {judge.get('utility_avg')} / 5",
        ]
    elif judge:
        lines += ["", "## LLM-as-Judge", f"- Skipped: {judge.get('error')}"]
    else:
        lines += ["", "## LLM-as-Judge", "- Skipped (--skip-judge)"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full memory-engine eval on gold test set.")
    p.add_argument("--gold",              type=Path, default=GOLD_PATH)
    p.add_argument("--model",             type=str,  default="llama-3.1-8b-instant",
                   help="Groq model for extraction")
    p.add_argument("--noise-samples",     type=int,  default=5)
    p.add_argument("--consistency-samples", type=int, default=10)
    p.add_argument("--consistency-runs",  type=int,  default=3)
    p.add_argument("--judge-sample",      type=int,  default=20)
    p.add_argument("--judge-model",       type=str,  default="gpt-4o",
                   help="Judge model (gpt-4o uses OPENAI_API_KEY; groq model uses GROQ_API_KEY)")
    p.add_argument("--fuzzy-threshold",          type=float, default=0.5,
                   help="Fuzzy match threshold for extraction eval (default 0.5, was 0.7)")
    p.add_argument("--consistency-temperature",  type=float, default=0.3,
                   help="Temperature for self-consistency runs (0.0=always 1.0, use 0.3 for real signal)")
    p.add_argument("--skip-judge",              action="store_true")
    p.add_argument("--disable-relevance-filter", action="store_true",
                   help="Skip relevance pre-filter so Groq runs on ALL gold turns (improves recall)")
    p.add_argument("--out-dir",           type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed",              type=int,  default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load gold
    print("Loading gold test turns...")
    gold = load_gold(args.gold)
    turns = gold_to_turns(gold)
    print(f"  {len(turns)} test turns")

    # 2. extraction
    use_filter = not args.disable_relevance_filter
    print(f"\nRunning extraction (model={args.model}, relevance_filter={use_filter})...")
    graph = GraphMemory(similarity_threshold=0.88)
    agent = MemoryExtractionAgent(graph_memory=graph, model=args.model, temperature=0.0,
                                  use_relevance_filter=use_filter)
    print(f"  Groq available: {agent.client is not None}")

    preds: list[dict[str, Any]] = []
    for turn in turns:
        ext = agent.process_turn(turn)
        rec = extraction_to_record(turn["turn_id"], ext)
        if has_memory_output(ext):
            preds.append(rec)
    print(f"  {len(preds)} non-empty predictions")
    (args.out_dir / "test_extractions.json").write_text(
        json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3. P/R/F1
    print("\nScoring extraction quality...")
    field_labels = {
        "entities": "Entities", "relations": "Relations", "preferences": "Preferences",
        "constraints": "Constraints", "goals": "Goals", "projects": "Projects", "tools": "Tools",
    }
    gold_sets = {k: _field_sets(gold,  k) for k in FIELDS}
    pred_sets = {k: _field_sets(preds, k) for k in FIELDS}

    # Build a merged pred set: any entity-like item from entities+tools+goals+projects
    # counts toward matching gold entities (cross-field scoring).
    ENTITY_LIKE_FIELDS = ("entities", "tools", "goals", "projects")
    pred_merged: dict[str, set] = {}
    for tid in {r["turn_id"] for r in preds}:
        merged: set = set()
        for f in ENTITY_LIKE_FIELDS:
            merged |= pred_sets[f].get(tid, set())
        pred_merged[tid] = merged

    extraction_metrics: dict[str, dict[str, float]] = {}
    for key, label in field_labels.items():
        if any(gold_sets[key].values()):
            # For entities: score gold entities against merged pred (cross-field)
            p_sets = pred_merged if key == "entities" else pred_sets[key]
            m = score_prf(gold_sets[key], p_sets, threshold=args.fuzzy_threshold)
            extraction_metrics[label] = m
            print(f"  {label:12s} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (on {m.get('n_predicted_turns','?')}/{m.get('n_total_turns','?')} predicted turns)")
    (args.out_dir / "metrics_extraction_test.json").write_text(
        json.dumps(extraction_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. noise robustness
    print(f"\nNoise robustness ({args.noise_samples} samples)...")
    noise = run_noise_robustness(args.model, turns, n_samples=args.noise_samples, seed=args.seed,
                                 use_relevance_filter=use_filter)
    print(f"  Avg Jaccard vs clean: {noise.get('avg_jaccard_vs_clean')}")
    (args.out_dir / "noise_robustness_test.json").write_text(
        json.dumps(noise, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5. self-consistency
    print(f"\nSelf-consistency ({args.consistency_runs} runs × {args.consistency_samples} turns)...")
    consistency = run_self_consistency(
        args.model, turns,
        n_runs=args.consistency_runs,
        n_samples=args.consistency_samples,
        seed=args.seed,
        use_relevance_filter=use_filter,
        temperature=args.consistency_temperature,
    )
    print(f"  Avg Jaccard all: {consistency.get('avg_jaccard_all')}")
    (args.out_dir / "self_consistency_test.json").write_text(
        json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 6. LLM judge
    judge_result: dict[str, Any] | None = None
    if not args.skip_judge:
        provider = "groq" if not args.judge_model.startswith(("gpt-", "o1", "o3", "o4")) else "openai"
        print(f"\nLLM judge ({args.judge_model} via {provider}, sample={args.judge_sample})...")
        judge_result = run_judge(
            provider=provider,
            model=args.judge_model,
            turns=turns,
            extractions=preds,
            sample=args.judge_sample,
            seed=args.seed,
        )
        if "error" in judge_result:
            print(f"  Skipped: {judge_result['error']}")
        else:
            print(f"  Completeness={judge_result['completeness_avg']}  "
                  f"Faithfulness={judge_result['faithfulness_avg']}  "
                  f"Utility={judge_result['utility_avg']}")
            judge_out = {k: v for k, v in judge_result.items() if k != "per_turn"}
            (args.out_dir / "judge_scores_test.json").write_text(
                json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (args.out_dir / "judge_summary_test.json").write_text(
                json.dumps(judge_out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # 7. markdown
    write_md(
        args.out_dir / "memory_evaluation_summary.md",
        model=args.model,
        n_test=len(turns),
        n_pred=len(preds),
        extraction=extraction_metrics,
        noise=noise,
        consistency=consistency,
        judge=judge_result,
        fuzzy_threshold=args.fuzzy_threshold,
        consistency_temperature=args.consistency_temperature,
    )
    print("\nDone. Outputs in", args.out_dir)


if __name__ == "__main__":
    main()

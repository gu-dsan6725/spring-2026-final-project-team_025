# Evaluation scaffolding (Milestone 3)

This folder provides lightweight scripts to score extraction and retrieval without needing full benchmarks. Bring a small hand-labeled set (20–30 examples) to get preliminary numbers.

## Extraction evaluation

Inputs:
- Gold file: list of records with `turn_id` and fields: `entities`, `relations`, `preferences`, `constraints`, `goals`, `projects`, `tools` (any subset is fine).
  - Samples: `gold_extraction_sample.json` (small) or `gold_extraction_auto_large.json` (auto-labeled from latest run).
- Pred file: output from `run_pipeline.py` (`outputs/extractions.json`).

Run:
```
python evaluation/eval_extraction.py --gold path/to/gold.json --pred outputs/extractions.json --out outputs/eval_extraction.json
```

Metrics (for every field present in gold): precision/recall/F1 over normalized strings; relations are matched on (source, target, relation_type).

## Retrieval evaluation

Inputs:
- Gold file: list of `{ "query": str, "expected_nodes": [str] }`.
  - Samples: `gold_retrieval_sample.json` (tiny) or `gold_retrieval_sample_rich.json` (2 queries).
- Pred file: either the retrieval output from `run_pipeline.py` (`demo_retrieval.json`) or a list of retrieval outputs with the same schema.

Run:
```
python evaluation/eval_retrieval.py --gold path/to/gold_retrieval.json --pred outputs/demo_retrieval.json --out outputs/eval_retrieval.json
```

Metrics: hit-rate@k and recall@k over expected nodes; reports average across queries.

## LLM-as-a-judge

This project also includes `judge_llm.py` for automatic scoring of extracted memory with an LLM judge.

OpenAI judge example:
```bash
python evaluation/judge_llm.py \
  --turns outputs/sample_turns.jsonl \
  --extractions outputs/extractions.json \
  --out outputs/judge_scores.json \
  --summary-out outputs/judge_summary.json \
  --sample 15 \
  --judge-provider openai \
  --judge-model gpt-4o
```

Stronger OpenAI judge example:
```bash
python evaluation/judge_llm.py \
  --turns outputs/sample_turns.jsonl \
  --extractions outputs/extractions.json \
  --out outputs/judge_scores.json \
  --summary-out outputs/judge_summary.json \
  --sample 15 \
  --judge-provider openai \
  --judge-model gpt-5.1 \
  --judge-reasoning-effort low
```

The script now scores:
- completeness
- faithfulness
- utility

Environment variables:
- `OPENAI_API_KEY` for OpenAI judge models
- `GROQ_API_KEY` for Groq judge models

## Self-Consistency

`eval_self_consistency.py` repeats extraction on the same sampled turns and measures pairwise Jaccard overlap across runs.

Example:
```bash
python evaluation/eval_self_consistency.py \
  --turns outputs/sample_turns.jsonl \
  --sample 10 \
  --runs 5 \
  --seed 0 \
  --model llama-3.1-8b-instant \
  --temperature 0.0 \
  --out outputs/self_consistency.json
```

Key outputs:
- `avg_jaccard_all`
- per-field average Jaccard scores
- `exact_match_turn_ratio`

## Noise Robustness

`eval_noise_robustness.py` perturbs a clean utterance with typos, filler words, colloquial phrasing, and punctuation noise, then checks extraction overlap and whether entity linking merges variants into the same graph node.

Example:
```bash
python evaluation/eval_noise_robustness.py \
  --text "I am learning Python data analysis." \
  --model llama-3.1-8b-instant \
  --temperature 0.0 \
  --out outputs/noise_robustness.json
```

Key outputs:
- `avg_jaccard_vs_clean`
- `linked_non_user_node_count`
- `max_mention_count`

Notes:
- These are minimal scripts to unblock Milestone 3. Expand to relation-level metrics or latency histograms as needed.
- Keep gold sets small and targeted to tune thresholds and linking behavior quickly.

## Recommended Gold Files

Use these as the primary human-curated reference sets:
- `gold_extraction_hand_labeled.json`
- `gold_extraction_multiturn.json`
- `gold_retrieval_hand_labeled.json`
- `gold_retrieval_multiturn.json`

## Archived Legacy Auto References

The following files were generated automatically in earlier experiments and should not be treated as authoritative gold labels for the current extractor:
- `archive/gold_extraction_auto.json`
- `archive/gold_extraction_auto_large.json`
- `archive/gold_extraction_llm_full.json`
- `archive/gold_extraction_sample.json`
- `archive/gold_extraction_sample_llm.json`
- `archive/gold_retrieval_sample.json`
- `archive/gold_retrieval_sample_rich.json`

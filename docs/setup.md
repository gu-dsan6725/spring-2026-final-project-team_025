# Memory Engine Runbook (M3)

## Quickstart (batch + demo)
```
GROQ_API_KEY=... python -m memory_engine.run_pipeline \
  --input data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --sample-size 400 --max-turns 400 --min-turn-chars 6 \
  --prefer-memory-turns --shuffle --seed 9 --disable-relevance-filter \
  --demo-query "i am learning python for a project" --demo-top-k 5 \
  --demo-llm-model llama-3.1-8b-instant
```
Outputs (all under `outputs/`):
- `sample_turns.jsonl`, `pipeline_stats.json` — normalized data + stats
- `extractions.json` — agent outputs
- `graph_memory.json`, `graph_stats.json` — graph export
- `demo_retrieval.json`, `demo_summary.txt` — retrieval result + human-readable summary
- `demo_llm_response.json` — LLM response (if network allows)

Notes:
- `--disable-relevance-filter` loosens extraction filtering to improve recall; remove it to reduce noise.
- `--demo-llm-model` requires network access to Groq; in restricted networks an error field may be present.

## Evaluation (gold sets)
Gold examples:
- Small: `evaluation/gold_extraction_sample.json`, `evaluation/gold_retrieval_sample.json`
- Auto-expanded: `evaluation/gold_extraction_auto_large.json` (from latest run), `evaluation/gold_retrieval_sample_rich.json`

Feel free to replace with your own annotations and rerun evaluation.

Run:
```
python evaluation/eval_extraction.py --gold evaluation/gold_extraction_sample.json \
  --pred outputs/extractions.json --out outputs/eval_extraction.json

python evaluation/eval_retrieval.py --gold evaluation/gold_retrieval_sample.json \
  --pred outputs/demo_retrieval.json --out outputs/eval_retrieval.json
```
Metrics are written to `outputs/eval_extraction.json` and `outputs/eval_retrieval.json`.

## Config tips
- Extraction model: `--model` defaults to `llama-3.1-8b-instant`. Without a key it falls back to rules.
- Retrieval: keyword + semantic similarity (SequenceMatcher) + recency + mention_count; Top-k controlled by `--demo-top-k`.
- For higher recall, increase `--sample-size/--max-turns` or disable relevance filtering; for higher precision, enable filtering and raise `min-turn-chars`.

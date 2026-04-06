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

Notes:
- These are minimal scripts to unblock Milestone 3. Expand to relation-level metrics or latency histograms as needed.
- Keep gold sets small and targeted to tune thresholds and linking behavior quickly.

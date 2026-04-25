# Memory Engine Evaluation Summary

## Test Set
- Gold turns: 22
- Non-empty predictions: 16
- Extractor model: llama-3.1-8b-instant

## Notes on Evaluation Methodology
- Gold set: 22 turns from `gold_extraction_enriched.json` (manually reviewed and enriched).
- **Relations** are scored on `(source, target)` pairs only — relation_type is ignored
  because gold uses `MENTIONS` while the extractor uses semantic types (`USES_TOOL`, `HAS_GOAL`, etc.).
  Relations F1 is the most reliable indicator of extraction quality.
- **Entities** are scored cross-field: a gold entity is counted as TP if found in
  *any* prediction field (entities, tools, goals, or projects), since the extractor may
  classify the same information under different field labels.
- Fuzzy match threshold: 0.5 (token overlap or SequenceMatcher ratio).

## Extraction Quality vs Gold Labels

*(Scored on predicted turns only; turns where extractor returned empty are excluded from P/R/F1)*

| Field | P | R | F1 | TP | FP | FN | Predicted Turns |
|-------|---|---|----|----|----|-----|----------------|
| Entities | 0.833 | 0.938 | 0.882 | 15 | 3 | 1 | 16/22 |
| Relations | 0.944 | 1.000 | 0.971 | 17 | 1 | 0 | 16/22 |

## Noise Robustness
- Turns sampled: 10 (non-trivial pairs: 28)
- Avg Jaccard vs clean: 1.0000
- By variant:
  - typo: 1.0000
  - filler: 1.0000
  - colloquial: 1.0000
  - punctuation: 1.0000

## Self-Consistency (repeated runs)
- Turns sampled: 15 × 3 runs (non-trivial pairs: 33)
- Extraction temperature: 0.7 (non-zero to get meaningful variance)
- Avg Jaccard (all):      1.0000
- Avg Jaccard (entities): 1.0000
- Avg Jaccard (relations):1.0000
- Avg Jaccard (goals):    1.0000
- Avg Jaccard (tools):    1.0000

## LLM-as-Judge
- Judge model: gpt-4o (openai)
- Turns scored: 16
- Completeness: 3.5 / 5
- Faithfulness: 4.312 / 5
- Utility:      3.938 / 5

# Architecture Update

## Baseline design

The original proposal used a 4-layer pipeline:

1. Input Layer
2. Extraction and Linking Layer
3. Graph Memory Layer
4. Retrieval and Response Layer

## What changed after implementation

Early implementation validated layers 1-3 and deferred retrieval to the next phase.

### Implemented architecture

```
ShareGPT Sample Data
  -> Input Pipeline
      - Parse JSON
      - Turn segmentation
      - Noise filtering
      - Turn normalization
  -> Extraction Agent
      - Groq-based extractor
      - Rule-based fallback
  -> Entity Linking
      - Type-constrained fuzzy matching
      - Similarity threshold
  -> Graph Memory (NetworkX)
      - Node and edge writes
      - Temporal metadata
  -> JSON exports for analysis and iteration
```

## Early learnings and adjustments

1. **LLM extraction requires strict output control**  
   The extractor now requests JSON-only output and keeps a deterministic fallback.

2. **Entity linking must be type-aware**
   Linking is scoped by `node_type` before fuzzy matching to reduce bad merges.

3. **Temporal metadata should be captured immediately**
   `created_at`, `last_seen`, and `evidence_turn_id` are attached during graph write.

4. **Schema required practical expansion**
   Beyond `Goal`, `Preference`, `Project`, and `Constraint`, we added `Tool` and `Entity`.

## Next phase scope

- Retrieval and ranking layer
- Extraction and linking quality evaluation
- Stronger canonicalization and confidence calibration


# Personal Knowledge Graph Memory Engine Scaffold

This implementation provides a runnable scaffold for:

1. Data pipeline with sample data
2. One functional extraction and linking agent
3. Architecture update notes from early implementation
4. Risks and mitigation plans

---

## What is implemented

### 1) Input layer data pipeline

Code: `memory_engine/data_pipeline.py`

- Loads ShareGPT V3 JSON from `data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
- Segments conversations into turn-level records
- Cleans noisy or invalid turns
- Normalizes each turn with:
  - `dialogue_id`
  - `turn_id`
  - `turn_index`
  - `speaker`
  - `text`
  - `timestamp_utc`
- Exports sample results to JSONL and JSON

### 2) Functional agent (Groq only) + entity linking

Code:
- `memory_engine/extraction_agent.py`
- `memory_engine/graph_memory.py`

Implemented features:

- `MemoryExtractionAgent` processes each human turn
- LLM extraction uses **Groq API** with `GROQ_API_KEY`
- If Groq is unavailable or response parsing fails, deterministic rule extraction is used
- Extracts:
  - entities
  - relations
  - user preferences
  - user constraints
  - goals
  - projects
  - tools
- Entity linking:
  - type-aware fuzzy matching (`SequenceMatcher`)
  - avoids duplicate nodes by linking to existing nodes
- Graph memory:
  - NetworkX `MultiDiGraph`
  - node attributes: `node_type`, `content`, `created_at`, `last_seen`, `mention_count`
  - edge attributes: `relation_type`, `evidence_turn_id`, `timestamp_utc`

### 3) Architecture update and 4) risks

Docs:
- `docs/architecture_update.md`
- `docs/risks_and_mitigations.md`

---

## Run

### Install dependencies

```bash
python -m pip install -r requirements-memory-engine.txt
```

### Set Groq API key

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Execute the pipeline

```bash
python -m memory_engine.run_pipeline \
  --input data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --sample-size 300 \
  --max-turns 200 \
  --output-dir outputs
```

If Groq network calls are unstable, the pipeline now auto-falls back to rule extraction after repeated failures and keeps running.

---

## Expected outputs

Under `outputs/`:

- `sample_turns.jsonl`
- `pipeline_stats.json`
- `extractions.json`
- `graph_memory.json`
- `graph_stats.json`


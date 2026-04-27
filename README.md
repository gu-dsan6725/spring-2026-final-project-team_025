# Personal Knowledge Graph Memory Engine and Career Development Assistant

**Turning conversations into structured long-term memory for personalized and evolving career guidance**


---

## Overview

This project builds a two-part system on top of conversational AI:

1. **Memory Engine** — extracts user-relevant information from conversations (goals, preferences, projects, constraints) and stores them in a persistent knowledge graph for long-term retrieval.
2. **Career Engine** — uses the extracted signals to align with O\*NET occupation profiles, recommend careers via a two-stage pipeline, and generate personalized gap analysis and learning roadmaps.

```
Conversation Turns
  → Signal Extraction (Groq LLM + rule-based fallback)
  → Knowledge Graph (NetworkX)
  → O*NET Profile Alignment (Sentence Transformers + keyword rules)
  → Cognitive Inference (implicit trait inference)
  → Stage 1: Cosine Similarity Shortlist (Top-20)
  → Stage 2: Groq Re-ranking (Top-5 + reasoning)
  → Gap Analysis & Learning Roadmap
```

---

## Memory Engine

**Code:** `memory_engine/`

Extracts entities, relations, preferences, constraints, goals, projects, and tools from raw conversation turns and stores them in a `NetworkX MultiDiGraph` with temporal and confidence metadata. A hybrid retrieval layer surfaces relevant memory subgraphs at inference time.

### Run

```bash
pip install -r requirements-memory-engine.txt
export GROQ_API_KEY="your_key"

python -m memory_engine.run_pipeline \
  --input data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --sample-size 300 \
  --max-turns 200 \
  --output-dir outputs
```

### Evaluation

- **LLM-as-Judge** (`evaluation/judge_llm.py`): scores extraction quality on Completeness, Faithfulness, and Utility (1–5) using Groq or GPT-4o as judge.
- **Retrieval metrics** (`evaluation/eval_retrieval.py`): Precision@k, Recall@k, and NDCG@k against gold-standard queries.
- Additional probes: noise robustness (`eval_noise_robustness.py`) and self-consistency (`eval_self_consistency.py`).

---

## Career Engine

**Code:** `career_engine/`

Builds on extracted signals to provide career guidance:

- **O\*NET Alignment**: maps extracted nodes to O\*NET Knowledge, Skills, and Work Styles elements via `all-MiniLM-L6-v2` sentence embeddings and keyword rules.
- **Cognitive Inference**: infers implicit traits (e.g., many technical projects → Complex Problem Solving, repeated debugging → Attention to Detail).
- **Two-Stage Recommendation**: Stage 1 uses weighted cosine similarity to produce a top-20 shortlist; Stage 2 uses Groq to re-rank with full conversation context and attach reasoning.
- **Gap Analysis**: compares the user profile against a target occupation to surface missing knowledge, skills, and work styles.
- **Learning Roadmap**: Groq generates a phased, personalized development plan grounded in the user's existing strengths.

### Run

```bash
pip install -r requirements-memory-engine.txt
export GROQ_API_KEY="your_key"

python -m career_engine.run_pipeline \
  --input data/career_dialogues_all_180.json \
  --output-dir outputs/career
```

### Evaluation

- **Extraction coverage** (`evaluation/career_eval_extraction.py`): measures signal extraction counts and per-field coverage across career dialogues.
- **LLM-as-Judge** (`evaluation/career_judge_llm.py`): qualitative scoring on Completeness, Faithfulness, and Career Utility (1–5), plus narrative-based profile validation where the judge generates a "Career Profile Narrative" to verify the agent's understanding matches user intent.

---

## Data Sources

- **ShareGPT V3**: large-scale dialogues for memory engine evaluation.
- **Career Dialogues** (`data/career_dialogues_all_180.json`): self-collected career conversations for the career engine.
- **O\*NET Database** (`data/Knowledge.xlsx`, `Skills.xlsx`, `Work Styles.xlsx`): occupation profiles for alignment and gap analysis.

---

## Demo

[Demo](https://drive.google.com/file/d/1wpz_UbatGvZyUmTuQR73VcfOxA4nQjRR/view?usp=drive_link)

---

## Team

- Jiayi Peng
- Kexin Lyu
- Yiran Tao
- Chenxi Guo

---

## License

MIT License

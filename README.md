# Personal Knowledge Graph Memory Engine and Career Development Assistant

**Turning conversations into structured long-term memory for personalized and evolving career guidance**

## Overview

Large language model assistants typically rely on short-term context windows or flat retrieval mechanisms (e.g., vector similarity over past messages). While effective for semantic recall, these approaches struggle to preserve **relational context**, maintain **long-term user consistency**, and provide **interpretable personalization**.

This project proposes a **Personal Knowledge Graph Memory Engine and Career Development Assistant** that converts conversational interactions into a structured, persistent, and queryable memory representation. Instead of storing conversations as unstructured logs, the system automatically extracts user-relevant information — such as goals, preferences, active projects, and constraints — and organizes them into a graph-based memory.

At inference time, a hybrid retrieval-ranking layer surfaces the most relevant memory subgraph and injects it into the LLM context to produce personalized, context-aware responses, specifically tailored for evolving career guidance.

---

## Abstract

We present a Personal Knowledge Graph Memory Engine and Career Development Assistant that transforms conversational interactions into structured, queryable knowledge representations. Rather than relying on flat retrieval mechanisms, our system automatically identifies and extracts semantically meaningful elements from user dialogues — including goals, preferences, active projects, and constraints — and organizes them within a graph-based memory architecture.

The core pipeline integrates information extraction and entity linking to surface relevant concepts from unstructured conversation, followed by graph storage to preserve relational context between entities. A memory retrieval ranking layer then surfaces the most contextually relevant nodes during subsequent interactions, enabling the system to reason over accumulated user knowledge rather than isolated exchanges. This architecture allows the assistant to provide evolving career guidance based on a deep understanding of the user's professional journey.

We explore two primary experimental directions: first, the tradeoff between memory size and retrieval quality, examining how graph density affects precision and recall; second, a comparative evaluation between graph-structured memory and pure vector embedding approaches, assessing which better captures relational semantics over long interaction histories.

---

## Agent Architecture

### Agent Flow

```
Conversation Turns
  → [1] Input Layer
  → [2] Extraction & Linking Layer
        - Information Extraction
        - Entity Linking
  → [3] Graph Memory Layer
        - Graph Storage
        - Temporal Updates
  → [4] Retrieval & Response Layer
        - Hybrid Retrieval
        - Ranking
  → LLM Context Injection
  → Personalized Career Response
```

---

## System Pipeline

| Step | Module | Description | Output |
|------|--------|------------|--------|
| 1 | Input Layer | Ingests and segments conversation turns; normalization and timestamping | Clean conversation chunks |
| 2 | Information Extraction | Extracts goals, preferences, projects, and constraints | Candidate entities & relations |
| 3 | Entity Linking | Resolves extracted items to existing graph nodes | Canonical entities |
| 4 | Graph Storage | Stores nodes and edges with temporal and confidence metadata | Persistent knowledge graph |
| 5 | Retrieval Ranking | Retrieves and ranks relevant memory subgraphs | Ranked memory context |
| 6 | Context Injection | Converts subgraph into structured LLM grounding | Memory-aware prompt |
| 7 | Response | Generates personalized career guidance | Final output |

---

## Memory Schema

### Node Types

- **Goal** — long-term or short-term user objectives
- **Preference** — behavioral or stylistic choices
- **Project** — ongoing tasks or research topics
- **Constraint** — time, resource, or contextual limitations
- **Career Signal** — specific professional interests, skills, or paths

### Edge Types

- `HAS_GOAL`
- `PREFERS`
- `WORKS_ON`
- `CONSTRAINED_BY`
- `RELATED_TO`
- `INTERESTED_IN`

---

## Implementation Details

### What is implemented

#### 1) Input layer data pipeline
Code: `memory_engine/data_pipeline.py`
- Loads ShareGPT V3 JSON
- Segments conversations into turn-level records
- Normalizes turns with metadata (dialogue_id, speaker, text, etc.)

#### 2) Functional agent & entity linking
Code: `memory_engine/extraction_agent.py`, `memory_engine/graph_memory.py`
- `MemoryExtractionAgent` processes human turns using **Groq API**
- Extracts entities, relations, goals, projects, and tools
- Entity linking with type-aware fuzzy matching to avoid duplicates
- Graph memory using NetworkX `MultiDiGraph`

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

---

## Expected Outputs

Under `outputs/`:
- `sample_turns.jsonl`
- `pipeline_stats.json`
- `extractions.json`
- `graph_memory.json`
- `graph_stats.json`

---

## Data Sources

1. **ShareGPT V3 (Primary)**: Large-scale dialogues for memory accumulation.
2. **Self-Collected Career Conversations**: Evaluates long-term persona consistency for career guidance.

---

## Experimental Plan

### Experiment 1 — Memory Size vs Retrieval Quality
- Precision@k / Recall@k
- NDCG@k
- Retrieval latency vs memory size

### Experiment 2 — Graph Memory vs Vector Memory
- Relational reasoning capability
- Long-term consistency
- Personalization quality

---

## Team

- Jiayi Peng
- Kexin Lyu  
- Yiran Tao
- Chenxi Guo  

---

## License

MIT License

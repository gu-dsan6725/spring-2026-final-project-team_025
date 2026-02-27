# Personal Knowledge Graph Memory Engine

## Overview

Large language model assistants typically rely on short-term context windows or flat retrieval mechanisms (e.g., vector similarity over past messages). While effective for semantic recall, these approaches struggle to preserve **relational context**, maintain **long-term user consistency**, and provide **interpretable personalization**.

This project proposes a **Personal Knowledge Graph Memory Engine** that converts conversational interactions into a structured, persistent, and queryable memory representation. Instead of storing conversations as unstructured logs, the system automatically extracts user-relevant information — such as goals, preferences, active projects, and constraints — and organizes them into a graph-based memory.

At inference time, a hybrid retrieval-ranking layer surfaces the most relevant memory subgraph and injects it into the LLM context to produce personalized, context-aware responses.

---

## Abstract

We present a Personal Knowledge Graph Memory Engine that transforms conversational interactions into structured, queryable knowledge representations. Rather than relying on flat retrieval mechanisms, our system automatically identifies and extracts semantically meaningful elements from user dialogues — including goals, preferences, active projects, and constraints — and organizes them within a graph-based memory architecture.

The core pipeline integrates information extraction and entity linking to surface relevant concepts from unstructured conversation, followed by graph storage to preserve relational context between entities. A memory retrieval ranking layer then surfaces the most contextually relevant nodes during subsequent interactions, enabling the system to reason over accumulated user knowledge rather than isolated exchanges.

We explore two primary experimental directions: first, the tradeoff between memory size and retrieval quality, examining how graph density affects precision and recall; second, a comparative evaluation between graph-structured memory and pure vector embedding approaches, assessing which better captures relational semantics over long interaction histories.

This work sits at the intersection of knowledge representation and cognitive modeling, drawing inspiration from how humans structure personal memory around relationships and context rather than raw similarity. The proposed system offers a principled foundation for building AI assistants with persistent, structured, and interpretable long-term memory.

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
  → Personalized Response
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
| 7 | Response | Generates personalized response | Final output |

---

## Memory Schema

### Node Types

- **Goal** — long-term or short-term user objectives
- **Preference** — behavioral or stylistic choices
- **Project** — ongoing tasks or research topics
- **Constraint** — time, resource, or contextual limitations

### Edge Types

- `HAS_GOAL`
- `PREFERS`
- `WORKS_ON`
- `CONSTRAINED_BY`
- `RELATED_TO`

### Metadata

- Timestamp
- Source utterance
- Confidence score
- Recency weight

---

## Data Sources

### 1. ShareGPT V3 (Primary Conversational Source)

Large-scale, diverse human–AI dialogues used to simulate memory accumulation across heterogeneous conversations.

https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json

### 2. Self-Collected Long-Form Conversations (Optional)

Used to evaluate long-term persona consistency and memory persistence across extended interaction histories.

---

## Experimental Plan

### Experiment 1 — Memory Size vs Retrieval Quality

We analyze how retrieval performance scales as the graph grows:

- Precision@k / Recall@k
- NDCG@k
- Retrieval latency vs memory size
- Effect of recency weighting

Key questions:

- Does more memory always improve relevance?
- When does graph density introduce noise?

---

### Experiment 2 — Graph Memory vs Vector Memory

We compare:

- Graph-based retrieval
- Pure vector similarity retrieval

Evaluation focus:

- Relational reasoning capability
- Long-term consistency
- Personalization quality

---

## Why Graph Memory?

Vector memory is effective for semantic similarity but limited in:

- Multi-hop reasoning
- Explicit relationship modeling
- Interpretable structure
- Tracking evolving user state

Graph memory enables structured, queryable long-term memory grounded in relational context — closer to human cognitive organization.

---

## Team

- Kexin Lyu  
- Jiayi Peng  
- Chenxi Guo  
- Yiran Tao

---

## License

MIT License


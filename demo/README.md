# Career & Memory Engine — Demo

An interactive web demo that lets you chat with the Career & Memory Engine in real time.
As you describe your background and goals, the system:

- **Extracts career signals** (skills, tools, projects, knowledge areas, work styles…) using an LLM
- **Builds a personal memory graph** that accumulates across turns in the session
- **Matches O\*NET occupations** via semantic embedding of your profile
- **Responds conversationally** as a career advisor powered by Groq Llama 3

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| uv (recommended) | any recent |
| GROQ_API_KEY | free at [console.groq.com](https://console.groq.com) |

> **Without a GROQ_API_KEY** the system falls back to rule-based extraction and a templated reply.  
> Career recommendations and the memory graph still work — they just won't be LLM-powered.

---

## Setup

### 1. Install dependencies

From the **repo root**:

```bash
# Install all project + demo deps via uv (respects uv.lock for pinned versions)
uv sync --extra demo

# Or with plain pip:
pip install -r requirements-memory-engine.txt
pip install -r demo/requirements.txt
```

### 2. Set your API key

```bash
cp demo/.env.example demo/.env
# Edit demo/.env and fill in:
#   GROQ_API_KEY=gsk_...
```

The server automatically loads `demo/.env` (or a `.env` in the repo root) on startup.

### 3. Download O\*NET data *(optional but recommended)*

The demo bundles a small built-in profile set.  
For full 900+ occupation matching, place the following files in `data/`:

```
data/Knowledge.xlsx
data/Skills.xlsx
data/Work Styles.xlsx
```

Download from [O\*NET Resource Center → Database](https://www.onetcenter.org/database.html#individual-files).  
The server detects them automatically at startup.

---

## Running the demo

From the **repo root**:

```bash
uvicorn demo.app:app --reload --port 8000
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

> ⚠️ Use `localhost`, not `127.0.0.1` — some environments resolve them differently and `127.0.0.1` may not respond correctly.

Expected terminal output on a successful start:

```
INFO:     Will watch for changes in these directories: ['/path/to/repo']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxx] using WatchFiles
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
Loading O*NET occupation profiles…
  Loaded 923 occupation profiles.
INFO:     Application startup complete.
```

---

## Using the demo

The UI has two panels:

**Left — Chat**
- Type your background, skills, projects, or career questions and press **Enter** or **Send**
- Click any example prompt on the welcome screen to prefill the input
- The AI responds as a career advisor, drawing on your accumulated memory

**Right — Signals (three tabs)**

| Tab | What it shows |
|---|---|
| **Extracted Signals** | Career signals from the latest message: knowledge areas, skills, tools, projects, goals, work styles, implicit signals |
| **Career Matches** | Top-5 O\*NET occupations matched to your cumulative profile, with component scores (knowledge / skills / work style) |
| **Memory** | Nodes recalled from the personal memory graph that are relevant to the current message |

**Header stats** update after each message:
- **Career graph nodes** — total entities accumulated in the career knowledge graph this session
- **Memory nodes** — total nodes in the personal memory graph this session

---

## Architecture overview

```
Browser (index.html)
       │  POST /api/chat
       ▼
FastAPI (demo/app.py)
       ├─ CareerExtractionAgent  ──► CareerGraphMemory
       │       └─ Groq LLM / rule fallback
       ├─ MemoryExtractionAgent  ──► GraphMemory
       │       └─ Groq LLM / rule fallback
       ├─ align_nodes_to_onet + recommend_careers  (sentence-transformers)
       ├─ RetrievalEngine.search()  (keyword + semantic, recency boost)
       └─ _generate_reply()  ──► Groq LLM (career advisor persona)
```

Each browser tab gets a fresh **session** (8-character UUID).  
State is in-memory only — refreshing the page starts a new session.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/session` | Create a new session → `{"session_id": "..."}` |
| `POST` | `/api/chat` | Send a message, get reply + extraction + recommendations |
| `GET` | `/api/graph/{session_id}` | Raw career graph nodes + edges (JSON) |
| `GET` | `/health` | Server health check + occupation count |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Page blank / stuck loading | JS from external CDN blocked | All styles are now inline — should not occur |
| `Session not found` error | Session expired (server restarted) | Refresh the page to create a new session |
| Memory tab always empty | GROQ_API_KEY not set + rule fallback didn't match | Set `GROQ_API_KEY` in `demo/.env`; rule fallback now covers common ML tools |
| Startup slow (30–60 s) | sentence-transformers model downloading (~23 MB) on first run | Wait once; model is cached in `~/.cache/torch/` afterward |
| `curl http://127.0.0.1:8000` hangs | Environment-specific loopback routing | Use `http://localhost:8000` instead |

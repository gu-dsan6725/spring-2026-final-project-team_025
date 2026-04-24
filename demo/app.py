"""
Career & Memory Engine — Web Demo Backend

Run from the repo root:
    uvicorn demo.app:app --reload --port 8000

Then open http://localhost:8000  (use localhost, NOT 127.0.0.1)
"""
from __future__ import annotations

import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    # Always load from demo/.env regardless of where uvicorn is launched from
    _DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(_DEMO_DIR, ".env"), override=True)
except ImportError:
    pass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

from career_engine.extraction_agent import CareerExtractionAgent, extraction_sections
from career_engine.graph import CareerGraphMemory
from career_engine.groq_reranker import GroqCareerReranker
from career_engine.onet import (
    align_nodes_to_onet,
    build_user_profile,
    load_occupation_profiles,
    recommend_careers,
)
from memory_engine.extraction_agent import MemoryExtractionAgent
from memory_engine.graph_memory import GraphMemory
from memory_engine.retrieval import RetrievalEngine

try:
    from groq import Groq as _Groq
except ImportError:
    _Groq = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
_occupations: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _occupations
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"  Groq: configured (key={groq_key[:8]}…) — LLM extraction, re-ranking, and replies enabled.")
    else:
        print("  Groq: NOT configured — rule-based fallbacks only. Set GROQ_API_KEY in demo/.env")
    print("Loading O*NET occupation profiles…")
    try:
        from pathlib import Path
        _occupations = load_occupation_profiles(data_dir=Path(_REPO_ROOT) / "data")
        print(f"  Loaded {len(_occupations)} occupation profiles.")
    except Exception as exc:
        print(f"  Warning: could not load O*NET profiles: {exc}")
    yield


app = FastAPI(title="Career & Memory Engine Demo", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
_sessions: dict[str, dict[str, Any]] = {}


def _new_session() -> str:
    sid = str(uuid.uuid4())[:8]
    cg = CareerGraphMemory()
    mg = GraphMemory()
    _sessions[sid] = {
        "career_graph": cg,
        "memory_graph": mg,
        "career_agent": CareerExtractionAgent(graph_memory=cg),
        # Disable relevance filter so all user messages are stored in memory
        # (the filter is designed for noisy batch data, not interactive demos)
        "memory_agent": MemoryExtractionAgent(graph_memory=mg, use_relevance_filter=False),
        "reranker": GroqCareerReranker(),
        "turn_counter": 0,
        "history": [],          # [{role, content}]  for multi-turn context
        "extractions": [],      # accumulated extraction sections (for re-ranking context)
    }
    return sid


# ---------------------------------------------------------------------------
# LLM response generation (career counselor)
# ---------------------------------------------------------------------------

def _generate_reply(
    message: str,
    history: list[dict],
    extraction: dict,
    recommendations: list[dict],
    memory_context: list[dict],
) -> str | None:
    """Call Groq to produce a career-counselor reply. Returns None if unavailable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not _Groq:
        return None

    # Build a rich system prompt from everything we know
    ctx_lines: list[str] = []

    if memory_context:
        mem_str = "; ".join(n["content"] for n in memory_context[:4])
        ctx_lines.append(f"What you already know about the user: {mem_str}.")

    sig_parts: list[str] = []
    for key, items in extraction.items():
        if items:
            names = ", ".join(str(i.get("name", "")) for i in items[:4] if i.get("name"))
            if names:
                sig_parts.append(f"{key.replace('_', ' ')}: {names}")
    if sig_parts:
        ctx_lines.append("Signals extracted from this message — " + "; ".join(sig_parts) + ".")

    if recommendations:
        top3 = ", ".join(r["title"] for r in recommendations[:3])
        ctx_lines.append(f"Current top O*NET career matches: {top3}.")

    system = (
        "You are a warm, insightful career advisor having a natural conversation. "
        "Respond to what the user just said: acknowledge their experience, "
        "reflect their strengths back to them, and ask one thoughtful follow-up question "
        "to learn more. Keep it to 2–3 sentences. "
        "Do NOT mention specific job titles or career recommendations — "
        "focus on the person's skills, interests, and growth. "
        "Do not use bullet points.\n\n"
        + ("\n".join(ctx_lines) if ctx_lines else "")
    ).strip()

    # Keep last 4 turns of history for context (avoid token bloat)
    messages = [{"role": "system", "content": system}]
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    try:
        client = _Groq(api_key=api_key, timeout=15.0, max_retries=0)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=250,
            messages=messages,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"LLM reply error: {exc}")
        return None


def _fallback_reply(extraction: dict, recommendations: list[dict]) -> str:
    """Rule-based reply when Groq is unavailable."""
    parts: list[str] = []
    tools  = [i["name"] for i in extraction.get("tool", [])[:3] if i.get("name")]
    skills = [i["name"] for i in extraction.get("skill", [])[:2] if i.get("name")]
    goals  = [i["name"] for i in extraction.get("career_goal", [])[:1] if i.get("name")]

    if tools:
        parts.append(f"I can see you work with {', '.join(tools)} — that's a solid technical foundation.")
    if skills:
        parts.append(f"Your experience in {', '.join(skills)} is particularly valuable.")
    if goals:
        parts.append(f"Your interest in {goals[0]} gives a clear direction to work toward.")
    if not parts:
        parts.append("Thanks for sharing. What aspect of your career are you most focused on right now?")
    else:
        parts.append("What kind of work environment or problems excite you most?")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "occupations_loaded": len(_occupations)}


@app.post("/api/session")
async def create_session():
    return {"session_id": _new_session()}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please refresh.")

    sess = _sessions[req.session_id]
    sess["turn_counter"] += 1
    turn_id = f"turn_{sess['turn_counter']}"
    turn = {
        "turn_id": turn_id,
        "dialogue_id": f"session_{req.session_id}",
        "text": req.message,
        "speaker": "human",
    }

    # 1. Extraction + graph update
    career_out = sess["career_agent"].process_turn(turn)
    memory_out = sess["memory_agent"].process_turn(turn)

    # Accumulate extraction sections for re-ranking context
    ext_for_rerank = dict(extraction_sections(career_out))
    if any(ext_for_rerank.values()):
        sess["extractions"].append(ext_for_rerank)

    # 2. Career recommendations (Stage 1: cosine shortlist → Stage 2: Groq re-rank)
    career_graph: CareerGraphMemory = sess["career_graph"]
    recommendations: list[dict] = []
    if _occupations:
        nodes = career_graph.signal_nodes()
        if nodes:
            try:
                mapped    = align_nodes_to_onet(nodes)
                profile   = build_user_profile(mapped)
                shortlist = recommend_careers(profile, _occupations, top_k=20)
                reranker: GroqCareerReranker = sess["reranker"]
                recommendations = reranker.rerank(
                    shortlist["recommendations"], sess["extractions"], top_k=5
                )
            except Exception as exc:
                print(f"Recommendation error: {exc}")

    # 3. Memory retrieval
    memory_graph: GraphMemory = sess["memory_graph"]
    # Top-5 relevant nodes for LLM reply context (changes per message)
    memory_context: list[dict] = []
    # All accumulated nodes for UI display (grows over the session)
    memory_all: list[dict] = []
    if memory_graph.graph.number_of_nodes() > 1:
        try:
            result = RetrievalEngine(memory_graph, default_top_k=5).search(req.message, top_k=5)
            memory_context = [
                {
                    "content": n.content,
                    "node_type": n.node_type,
                    "score": round(n.score, 3),
                    "mention_count": int(n.metadata.get("mention_count", 1)),
                }
                for n in result.nodes
                if n.node_type != "User"
            ]
        except Exception as exc:
            print(f"Retrieval error: {exc}")

        # Collect every non-User node in the graph for persistent UI display
        for nid, attrs in memory_graph.graph.nodes(data=True):
            if attrs.get("node_type") == "User":
                continue
            memory_all.append({
                "content": attrs.get("content", nid),
                "node_type": attrs.get("node_type", "TOPIC"),
                "score": float(attrs.get("confidence", 0.5)),
                "mention_count": int(attrs.get("mention_count", 1)),
            })
        # Sort by mention count desc, then confidence desc
        memory_all.sort(key=lambda x: (-x["mention_count"], -x["score"]))

    # 4. LLM reply
    ext_sections = extraction_sections(career_out)
    reply = _generate_reply(
        message=req.message,
        history=sess["history"],
        extraction=ext_sections,
        recommendations=recommendations,
        memory_context=memory_context,
    ) or _fallback_reply(ext_sections, recommendations)

    # Update history
    sess["history"].append({"role": "user",      "content": req.message})
    sess["history"].append({"role": "assistant",  "content": reply})

    # Build accumulated extraction view from career graph (all turns so far)
    accumulated_extraction: dict[str, list[dict]] = {}
    for nid, attrs in career_graph.graph.nodes(data=True):
        ntype = attrs.get("node_type", "")
        if not ntype or ntype == "User":
            continue
        accumulated_extraction.setdefault(ntype, []).append({
            "name": attrs.get("content", nid),
            "confidence": round(float(attrs.get("confidence", 0.7)), 3),
            "_mentions": int(attrs.get("mention_count", 1)),
        })
    for ntype in accumulated_extraction:
        accumulated_extraction[ntype].sort(key=lambda x: (-x["_mentions"], -x["confidence"]))
        for item in accumulated_extraction[ntype]:
            item.pop("_mentions", None)

    return {
        "reply": reply,
        "extraction": accumulated_extraction,
        "memory_extraction": {
            "goals": memory_out.goals,
            "preferences": memory_out.preferences,
            "constraints": memory_out.constraints,
            "projects": memory_out.projects,
            "tools": memory_out.tools,
        },
        "recommendations": recommendations,
        "memory_context": memory_all,
        "graph_stats": {
            "career_nodes": career_graph.graph.number_of_nodes(),
            "career_edges": career_graph.graph.number_of_edges(),
            "memory_nodes": memory_graph.graph.number_of_nodes(),
        },
    }


@app.get("/api/graph/{session_id}")
async def get_graph(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    cg: CareerGraphMemory = _sessions[session_id]["career_graph"]
    return {
        "nodes": [
            {"id": nid, "type": a.get("node_type"), "label": a.get("content", nid),
             "confidence": round(float(a.get("confidence", 0.7)), 3),
             "mentions": int(a.get("mention_count", 1))}
            for nid, a in cg.graph.nodes(data=True)
        ],
        "edges": [
            {"source": s, "target": d, "relation": a.get("relation_type", "")}
            for s, d, a in cg.graph.edges(data=True)
        ],
    }


# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"), media_type="text/html")


@app.get("/{filename:path}")
async def serve_static(filename: str):
    path = os.path.join(_STATIC_DIR, filename)
    if os.path.isfile(path):
        return FileResponse(path)
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"), media_type="text/html")

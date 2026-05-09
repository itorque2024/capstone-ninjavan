"""
Multi-Agent Chatbot Orchestrator — LangGraph-based with multi-question decomposition.

Flow:
  decomposer → processor → synthesizer → END

  decomposer : Gemini splits message into N sub-questions with intents (JSON)
  processor  : for each sub-question → ChromaDB first → DuckDuckGo + Groq/Gemini fallback
  synthesizer: merges N answers; single question returns directly

Returns:
  answer, intent, agent_name, agent_emoji, agents_involved, debug_log,
  escalated, parcel_id, sources
"""
import json
import os
import re
from pathlib import Path
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.utils.chroma_setup import query_collection
from ._llm import llm_call as _gemini, get_last_llm

load_dotenv()

_CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "mock_parcels.csv"

_parcels_df: pd.DataFrame | None = None


def _load_parcels() -> pd.DataFrame:
    global _parcels_df
    if _parcels_df is None:
        _parcels_df = pd.read_csv(_CSV_PATH)
    return _parcels_df


def _extract_parcel_id(text: str) -> str | None:
    match = re.search(r"NV-\d{6}", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _history_text(history: list[dict], limit: int = 6) -> str:
    if not history:
        return ""
    lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-limit:]]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _rag_context(collection: str, query: str, n: int = 3, max_chars: int = 3000) -> tuple[str, int]:
    """Query a ChromaDB collection; returns (context_text, doc_count)."""
    docs = query_collection(collection, query, n_results=n)
    if not docs:
        return "", 0
    combined = "\n\n---\n\n".join(docs)
    return combined[:max_chars], len(docs)


def _duckduckgo_search(query: str, max_results: int = 3) -> str:
    """Web search via DuckDuckGo; returns formatted results or empty string."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n\n".join(f"[{r['title']}]\n{r['body']}" for r in results)
    except Exception:
        return ""


def _groq_answer(question: str, web_ctx: str, agent_role: str) -> str:
    """Answer via Groq (Llama-3.3-70B); falls back to Gemini on any error."""
    try:
        import groq as _groq_lib
        client = _groq_lib.Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",
                 "content": (f"You are {agent_role}. Answer concisely using the web search results "
                             "provided. If they aren't relevant, use your general knowledge.")},
                {"role": "user",
                 "content": f"Web search results:\n{web_ctx}\n\nCustomer question: {question}"},
            ],
            max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception:
        prompt = (
            f"You are {agent_role}. Use these web search results if relevant:\n\n"
            f"{web_ctx}\n\nCustomer question: {question}"
        )
        return _gemini(prompt)


# ── Intent → agent config ─────────────────────────────────────────────────────

_INTENT_CONFIG: dict[str, dict] = {
    "tracking":   {"collection": "nv_tracking",  "name": "Tracking Agent",   "emoji": "📦"},
    "delivery":   {"collection": "nv_delivery",  "name": "Delivery Agent",   "emoji": "🚚"},
    "claims":     {"collection": "nv_claims",    "name": "Claims Agent",     "emoji": "📋"},
    "policy":     {"collection": "nv_policy",    "name": "Policy Agent",     "emoji": "📜"},
    "ops":        {"collection": "nv_delivery",  "name": "Ops Agent",        "emoji": "🏭"},
    "escalation": {"collection": None,           "name": "Escalation Agent", "emoji": "🚨"},
}


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    message:          str
    history:          list[dict]
    ops_context:      str
    subqueries:       list   # [{"question": str, "intent": str}]
    sub_answers:      list   # [{"question", "agent_name", "agent_emoji", "answer", ...}]
    intent:           str
    agent_name:       str
    agent_emoji:      str
    agents_involved:  list   # ["📦 Tracking Agent", "📋 Claims Agent", ...]
    debug_log:        str
    answer:           str
    sources:          int
    escalated:        bool
    parcel_id:        str | None
    llm_source:       str    # "gemini" | "groq" | "web+groq" | "web+gemini" etc.


# ── Specialist sub-query handlers ─────────────────────────────────────────────

def _answer_tracking_subquery(question: str, history: list[dict], pid: str | None) -> dict:
    parcel_id = _extract_parcel_id(question) or pid
    if not parcel_id:
        for m in reversed(history[-6:]):
            parcel_id = _extract_parcel_id(m["content"])
            if parcel_id:
                break

    if not parcel_id:
        return {"question": question, "agent_name": "Tracking Agent", "agent_emoji": "📦",
                "answer": ("I'd love to help track your parcel! Please share your parcel ID — "
                           "it looks like **NV-XXXXXX** and can be found in your shipping "
                           "confirmation email or on the airway bill."),
                "source": "no_id", "n_docs": 0, "escalated": False, "parcel_id": None}

    df  = _load_parcels()
    row = df[df["parcel_id"].str.upper() == parcel_id]

    if row.empty:
        return {"question": question, "agent_name": "Tracking Agent", "agent_emoji": "📦",
                "answer": (f"I couldn't find parcel **{parcel_id}** in our system. "
                           "Please double-check the ID (format: NV-XXXXXX). "
                           "If the problem persists, call **1800-NJV-CARE** for immediate help."),
                "source": "not_found", "n_docs": 0, "escalated": True, "parcel_id": parcel_id}

    r = row.iloc[0]
    status_emojis = {
        "Delivered": "✅", "Out for Delivery": "🚚", "In Transit": "📦",
        "Pending Pickup": "🕐", "Failed Delivery - Attempted": "⚠️",
    }
    emoji     = status_emojis.get(r["status"], "📦")
    rag_ctx, n_docs = _rag_context("nv_tracking", question)
    parcel_info = (
        f"Parcel ID: {r['parcel_id']}\n"
        f"Product: {r['product_description']}\n"
        f"Status: {emoji} {r['status']}\n"
        f"Route: {r['origin']} → {r['destination']}\n"
        f"Estimated Delivery: {r['estimated_delivery']}\n"
        f"Last Update: {r['last_update']}\n"
        f"Assigned Rider: {r['rider_name']}\n"
    )
    prompt = (
        "You are the NinjaVan Tracking Agent. Answer using the parcel data below. "
        "Be concise, warm, and include the status emoji. "
        "If the status is 'Failed Delivery', proactively explain the re-attempt process.\n\n"
        f"PARCEL DATA:\n{parcel_info}\n\n"
        f"TRACKING POLICY:\n{rag_ctx}\n\n"
        f"{_history_text(history)}"
        f"Customer question: {question}"
    )
    answer    = _gemini(prompt)
    escalated = r["status"] == "Failed Delivery - Attempted"
    return {"question": question, "agent_name": "Tracking Agent", "agent_emoji": "📦",
            "answer": answer, "source": f"csv+chromadb+{get_last_llm()}", "n_docs": n_docs,
            "escalated": escalated, "parcel_id": parcel_id}


def _answer_subquery(question: str, intent: str, history: list[dict],
                     ops_context: str, pid: str | None) -> dict:
    """Route one sub-question to the correct specialist; web fallback if ChromaDB is empty."""
    cfg        = _INTENT_CONFIG.get(intent, _INTENT_CONFIG["policy"])
    name       = cfg["name"]
    emoji      = cfg["emoji"]
    collection = cfg["collection"]

    if intent == "tracking":
        return _answer_tracking_subquery(question, history, pid)

    if intent == "escalation":
        prompt = (
            "You are the NinjaVan Senior Customer Experience Agent handling a sensitive situation. "
            "Structure your response:\n"
            "1. Acknowledge their frustration sincerely and apologise.\n"
            "2. Briefly summarise the issue based on their message.\n"
            "3. Confirm a Senior Support Agent will contact within 24 hours.\n"
            "4. Provide: call 1800-NJV-CARE or email escalation@ninjavan.co\n\n"
            f"{_history_text(history)}"
            f"Customer message: {question}"
        )
        ans = _gemini(prompt)
        return {"question": question, "agent_name": name, "agent_emoji": emoji,
                "answer": "🚨 " + ans,
                "source": get_last_llm(), "n_docs": 0, "escalated": True, "parcel_id": None}

    # ── ChromaDB lookup ───────────────────────────────────────────────────────
    ctx, n_docs = _rag_context(collection, question) if collection else ("", 0)
    extra_ctx, extra_n = "", 0
    if intent == "policy":
        extra_ctx, extra_n = _rag_context("nv_general", question, n=2)
        n_docs += extra_n

    if n_docs > 0:
        ops_sec = (f"LIVE OPS:\n{ops_context}\n\n"
                   if ops_context and intent in ("delivery", "ops") else "")
        prompt = (
            f"You are the {name}, a NinjaVan customer service specialist. "
            "Answer the customer question using the knowledge base below. "
            "Be concise and helpful. "
            "IMPORTANT: Never mention internal document names, filenames, or file paths. "
            "Speak naturally as a customer service representative.\n\n"
            f"KNOWLEDGE BASE:\n{ctx}\n\n"
            + (f"ADDITIONAL INFO:\n{extra_ctx}\n\n" if extra_ctx else "")
            + ops_sec
            + f"{_history_text(history)}"
            + f"Customer question: {question}"
        )
        ans = _gemini(prompt)
        return {"question": question, "agent_name": name, "agent_emoji": emoji,
                "answer": ans, "source": f"chromadb+{get_last_llm()}",
                "n_docs": n_docs, "escalated": False, "parcel_id": None}

    # ── Web fallback: DuckDuckGo → Gemini (Groq if rate-limited) ────────────
    web_ctx = _duckduckgo_search(f"NinjaVan {question}")
    web_prompt = (
        f"You are the {name}, a NinjaVan customer service specialist. "
        "Answer the customer question using the web search results below. "
        "Do not mention internal document names or filenames. "
        "Be concise and helpful.\n\n"
        f"WEB RESULTS:\n{web_ctx}\n\n"
        f"Customer question: {question}"
    ) if web_ctx else (
        f"You are the {name}, a NinjaVan customer service specialist. "
        "Answer this NinjaVan customer service question based on your knowledge.\n\n"
        f"Customer question: {question}"
    )
    answer = _gemini(web_prompt)
    source = ("web+" if web_ctx else "") + get_last_llm()

    return {"question": question, "agent_name": name, "agent_emoji": emoji,
            "answer": answer, "source": source, "n_docs": 0,
            "escalated": False, "parcel_id": None}


# ── Graph nodes ───────────────────────────────────────────────────────────────

_DECOMPOSER_SYSTEM = """You are a query decomposer for a NinjaVan customer service chatbot.

Split the customer message into individual questions. For each, assign the best intent.
Valid intents: tracking, delivery, claims, policy, ops, escalation

Rules:
- If there is only one question, return a single-element array.
- Never return more than 5 elements.
- Return ONLY valid JSON — no markdown fences, no explanation.

Format: [{"question": "...", "intent": "..."}]"""


def decomposer_node(state: ChatState) -> ChatState:
    msg  = state["message"]
    hist = _history_text(state["history"])
    prompt = f"{_DECOMPOSER_SYSTEM}\n\n{hist}Customer message: {msg}"
    try:
        raw = _gemini(prompt).strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        subqueries = json.loads(raw)
        if not isinstance(subqueries, list) or not subqueries:
            raise ValueError("empty or non-list")
        subqueries = [
            {
                "question": str(q.get("question", msg)),
                "intent":   q.get("intent") if q.get("intent") in _INTENT_CONFIG else "policy",
            }
            for q in subqueries[:5]
        ]
    except Exception:
        subqueries = [{"question": msg, "intent": "policy"}]

    return {**state, "subqueries": subqueries,
            "debug_log": f"🔀 Decomposer → {len(subqueries)} sub-question(s)"}


def processor_node(state: ChatState) -> ChatState:
    subqueries  = state["subqueries"]
    history     = state["history"]
    ops_context = state.get("ops_context", "")
    msg         = state["message"]

    pid = _extract_parcel_id(msg)
    if not pid:
        for m in reversed(history[-6:]):
            pid = _extract_parcel_id(m["content"])
            if pid:
                break

    sub_answers = []
    debug_parts = [state["debug_log"]]

    for sq in subqueries:
        result = _answer_subquery(sq["question"], sq["intent"], history, ops_context, pid)
        sub_answers.append(result)
        src = result.get("source", "?")
        n   = result.get("n_docs", 0)
        debug_parts.append(
            f"  {result['agent_emoji']} {result['agent_name']}: "
            f"[{src}, {n} docs] → \"{sq['question'][:50]}\""
        )
        if result.get("parcel_id") and not pid:
            pid = result["parcel_id"]

    return {**state, "sub_answers": sub_answers, "parcel_id": pid,
            "debug_log": "\n".join(debug_parts)}


def synthesizer_node(state: ChatState) -> ChatState:
    sub_answers = state["sub_answers"]

    if not sub_answers:
        return {**state,
                "answer": "I'm sorry, I couldn't process your question. Please try again.",
                "intent": "policy", "agent_name": "Policy Agent", "agent_emoji": "📜",
                "agents_involved": [], "sources": 0, "escalated": False, "llm_source": "gemini",
                "debug_log": state["debug_log"] + "\n⚠️ Synthesizer: no answers."}

    agents_involved = [f"{sa['agent_emoji']} {sa['agent_name']}" for sa in sub_answers]
    total_sources   = sum(sa.get("n_docs", 0) for sa in sub_answers)
    any_escalated   = any(sa.get("escalated", False) for sa in sub_answers)
    primary         = sub_answers[0]
    primary_intent  = state["subqueries"][0]["intent"] if state.get("subqueries") else "policy"

    all_sources = [sa.get("source", "") for sa in sub_answers]
    llm_label   = "groq" if any("groq" in s for s in all_sources) else "gemini"

    if len(sub_answers) == 1:
        return {**state,
                "answer":          primary["answer"],
                "intent":          primary_intent,
                "agent_name":      primary["agent_name"],
                "agent_emoji":     primary["agent_emoji"],
                "agents_involved": agents_involved,
                "sources":         total_sources,
                "escalated":       any_escalated,
                "llm_source":      llm_label,
                "debug_log":       state["debug_log"] + "\n✅ Synthesizer: single answer."}

    sections = [f"**{sa['agent_emoji']} {sa['agent_name']}**\n{sa['answer']}" for sa in sub_answers]
    combined = "\n\n---\n\n".join(sections)

    return {**state,
            "answer":          combined,
            "intent":          primary_intent,
            "agent_name":      primary["agent_name"],
            "agent_emoji":     primary["agent_emoji"],
            "agents_involved": agents_involved,
            "sources":         total_sources,
            "escalated":       any_escalated,
            "llm_source":      llm_label,
            "debug_log":       state["debug_log"] + f"\n✅ Synthesizer: merged {len(sub_answers)} answers."}


# ── Build LangGraph ───────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(ChatState)
    g.add_node("decomposer",  decomposer_node)
    g.add_node("processor",   processor_node)
    g.add_node("synthesizer", synthesizer_node)
    g.set_entry_point("decomposer")
    g.add_edge("decomposer",  "processor")
    g.add_edge("processor",   "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


_graph = _build_graph()

HISTORY_LIMIT = 20


def chat(message: str, history: list[dict], ops_context: str = "") -> dict:
    """
    Process one customer message through the multi-agent graph.
    history: list of {"role": "user"/"assistant", "content": "..."} dicts.
    """
    result = _graph.invoke({
        "message":         message,
        "history":         history,
        "ops_context":     ops_context,
        "subqueries":      [],
        "sub_answers":     [],
        "intent":          "",
        "agent_name":      "",
        "agent_emoji":     "",
        "agents_involved": [],
        "debug_log":       "",
        "answer":          "",
        "sources":         0,
        "escalated":       False,
        "parcel_id":       None,
        "llm_source":      "gemini",
    })
    return {
        "answer":          result["answer"],
        "intent":          result["intent"],
        "agent_name":      result["agent_name"],
        "agent_emoji":     result["agent_emoji"],
        "agents_involved": result["agents_involved"],
        "debug_log":       result["debug_log"],
        "escalated":       result["escalated"],
        "parcel_id":       result["parcel_id"],
        "sources":         result["sources"],
        "llm_source":      result.get("llm_source", "gemini"),
    }


def trim_history(history: list[dict]) -> list[dict]:
    return history[-HISTORY_LIMIT:]

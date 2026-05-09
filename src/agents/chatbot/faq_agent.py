"""
FAQ Agent — answers general NinjaVan questions using ChromaDB RAG + Gemini/Groq.
"""
from pathlib import Path
import chromadb
from ._llm import llm_call

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db")
_COLLECTION = "ninjavan_kb"

_SYSTEM_PROMPT = (
    "You are a NinjaVan customer service assistant. "
    "Answer using ONLY the context provided. Be concise and friendly. "
    "If the answer is not in the context, say you don't have that information "
    "and suggest the customer contact NinjaVan support."
)


def _get_collection():
    chroma = chromadb.PersistentClient(path=_CHROMA_PATH)
    return chroma.get_or_create_collection(_COLLECTION)


def answer_faq(message: str, history: list[dict]) -> dict:
    """Returns {'answer': str, 'escalated': bool, 'sources': int}."""
    collection = _get_collection()
    results = collection.query(query_texts=[message], n_results=3)
    docs = results["documents"][0] if results["documents"] else []

    if not docs:
        return {
            "answer": "I'm sorry, I couldn't find information on that topic. Let me connect you with a human agent.",
            "escalated": True,
            "sources": 0,
        }

    history_text = ""
    if history:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    context = "\n\n".join(docs)
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"{history_text}"
        f"Customer question: {message}"
    )

    answer = llm_call(prompt)
    escalated = "escalate" in answer.lower() or "human agent" in answer.lower()

    return {"answer": answer, "escalated": escalated, "sources": len(docs)}

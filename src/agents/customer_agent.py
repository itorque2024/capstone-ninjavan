"""
Customer Service Agent — RAG-powered chatbot using Gemini + ChromaDB.
Answers customer queries from the NinjaVan knowledge base.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from google import genai

load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-2.5-flash"

COLLECTION_NAME = "ninjavan_kb"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db")

_SYSTEM_PROMPT = (
    "You are a NinjaVan customer service assistant. "
    "Answer the customer's question using ONLY the context provided. "
    "Be concise, friendly, and helpful. "
    "If the answer is not in the context, say: "
    "'I don't have enough information to answer that. I will escalate this to a human agent.' "
    "Do not make up information not found in the context."
)


def _get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_or_create_collection(COLLECTION_NAME)


def run_customer_agent(state: dict) -> dict:
    """
    LangGraph-compatible node. Expects 'customer_query' in state.
    Returns a grounded answer from the knowledge base via Gemini.
    """
    query = state.get("customer_query", "")

    if not query:
        return {**state, "customer_result": {"error": "No customer query provided."}}

    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0] if results["documents"] else []

    if not docs:
        return {
            **state,
            "customer_result": {
                "answer": "I'm sorry, I couldn't find relevant information. A human agent will follow up.",
                "escalated": True,
            },
        }

    context = "\n\n".join(docs)
    full_prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nCustomer question: {query}"
    response = _client.models.generate_content(model=_MODEL, contents=full_prompt)
    answer = response.text
    escalated = "escalate" in answer.lower() or "human agent" in answer.lower()

    return {
        **state,
        "customer_result": {
            "answer": answer,
            "escalated": escalated,
            "sources_used": len(docs),
        },
    }

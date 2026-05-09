"""
Intent Router — classifies each customer message into one of three intents.
"""
from ._llm import llm_call

_ROUTER_PROMPT = """You are an intent classifier for NinjaVan customer service.

Given the conversation history and the latest user message, classify the intent into EXACTLY one of:
- faq        — general questions: delivery hours, fees, returns, rescheduling, prohibited items, packaging, etc.
- tracking   — questions about a specific parcel status, location, or delivery (may contain a parcel ID like NV-XXXXXX)
- escalation — complaints, refund/compensation disputes, repeated failures, or the customer explicitly asks to speak to a human

Reply with ONLY the single word: faq, tracking, or escalation. Nothing else."""


def classify_intent(message: str, history: list[dict]) -> str:
    """Returns 'faq', 'tracking', or 'escalation'."""
    history_text = ""
    if history:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    prompt = f"{_ROUTER_PROMPT}\n\n{history_text}Latest message: {message}"
    try:
        intent = llm_call(prompt).strip().lower().split()[0]
        if intent not in ("faq", "tracking", "escalation"):
            return "faq"
        return intent
    except Exception:
        return "faq"

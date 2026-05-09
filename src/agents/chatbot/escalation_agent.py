"""
Escalation Agent — handles complaints, refund disputes, and requests to speak to a human.
"""
from ._llm import llm_call

_ESCALATION_PROMPT = (
    "You are a NinjaVan customer service assistant handling a sensitive situation. "
    "The customer is frustrated or needs human assistance. "
    "Acknowledge their concern empathetically, apologise for any inconvenience, "
    "and inform them that you are connecting them with a human agent who will follow up within 24 hours. "
    "Give a short, warm, professional response. Do not promise specific outcomes."
)


def answer_escalation(message: str, history: list[dict]) -> dict:
    """Returns {'answer': str, 'escalated': bool}."""
    history_text = ""
    if history:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        f"{_ESCALATION_PROMPT}\n\n"
        f"{history_text}"
        f"Customer message: {message}"
    )

    try:
        answer = "🚨 " + llm_call(prompt)
    except Exception:
        answer = (
            "🚨 I'm sorry to hear about your experience. I'm connecting you with a human agent "
            "who will follow up within 24 hours. Thank you for your patience."
        )

    return {"answer": answer, "escalated": True}

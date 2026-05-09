"""
Tracking Agent — looks up parcel status from mock_parcels.csv and generates
a natural-language response via Gemini.
Gemini responses are cached on disk via joblib so the same query returns instantly.
"""
import os
import re
from pathlib import Path

import pandas as pd
from ._llm import llm_call
_CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "mock_parcels.csv"

_df: pd.DataFrame | None = None


def _load_parcels() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_CSV_PATH)
    return _df


def _extract_parcel_id(text: str) -> str | None:
    match = re.search(r"NV-\d{6}", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _status_emoji(status: str) -> str:
    mapping = {
        "Delivered": "✅",
        "Out for Delivery": "🚚",
        "In Transit": "📦",
        "Pending Pickup": "🕐",
        "Failed Delivery - Attempted": "⚠️",
    }
    return mapping.get(status, "📦")


def answer_tracking(message: str, history: list[dict]) -> dict:
    """Returns {'answer': str, 'escalated': bool, 'parcel_id': str|None}."""
    # Try to find a parcel ID in the current message or recent history
    parcel_id = _extract_parcel_id(message)
    if not parcel_id:
        for m in reversed(history[-6:]):
            parcel_id = _extract_parcel_id(m["content"])
            if parcel_id:
                break

    if not parcel_id:
        return {
            "answer": (
                "I'd love to help track your parcel! Could you please provide your parcel ID? "
                "It looks like **NV-XXXXXX** and can be found in your order confirmation email."
            ),
            "escalated": False,
            "parcel_id": None,
        }

    df = _load_parcels()
    row = df[df["parcel_id"].str.upper() == parcel_id]

    if row.empty:
        return {
            "answer": (
                f"I couldn't find parcel **{parcel_id}** in our system. "
                "Please double-check the ID. If the issue persists, I'll escalate to a human agent."
            ),
            "escalated": True,
            "parcel_id": parcel_id,
        }

    r = row.iloc[0]
    emoji = _status_emoji(r["status"])
    parcel_context = (
        f"Parcel: {r['parcel_id']}\n"
        f"Product: {r['product_description']}\n"
        f"Status: {r['status']}\n"
        f"From: {r['origin']}  →  To: {r['destination']}\n"
        f"Estimated Delivery: {r['estimated_delivery']}\n"
        f"Last Update: {r['last_update']}\n"
        f"Rider: {r['rider_name']}\n"
    )

    history_text = ""
    if history:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        "You are a NinjaVan customer service assistant. "
        "Use the parcel information below to answer the customer's question naturally and helpfully. "
        "Include the status emoji and key details. Be concise.\n\n"
        f"Parcel information:\n{parcel_context}\n\n"
        f"{history_text}"
        f"Customer question: {message}"
    )

    # ── Blockchain audit trail (optional — silent if not configured) ───────────
    from src.agents.chatbot.blockchain_logger import get_history, get_etherscan_url, is_available
    blockchain_history = []
    etherscan_url = None
    if is_available():
        blockchain_history = get_history(parcel_id)
        etherscan_url = get_etherscan_url(parcel_id)

    if blockchain_history:
        chain_text = "\n".join(
            f"  [{entry['timestamp']}] {entry['status']}"
            for entry in blockchain_history
        )
        full_prompt = prompt + f"\n\nBlockchain audit trail (tamper-proof):\n{chain_text}"
        answer = f"{emoji} " + llm_call(full_prompt)
    else:
        answer = f"{emoji} " + llm_call(prompt)

    escalated = r["status"] == "Failed Delivery - Attempted"

    return {
        "answer": answer,
        "escalated": escalated,
        "parcel_id": parcel_id,
        "blockchain_history": blockchain_history,
        "etherscan_url": etherscan_url,
    }

"""
Telegram Notifier Utility

Sends simple text alerts to a specific Telegram Chat ID using the bot token.
"""
import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

def send_telegram_alert(message: str) -> bool:
    """
    Sends a message to the configured TELEGRAM_OPS_CHAT_ID.
    Returns True if successful (or if simulated), False otherwise.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_OPS_CHAT_ID")

    if not bot_token or not chat_id:
        logging.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_OPS_CHAT_ID not set. Simulating alert.")
        return True # Simulate success for UI purposes if not configured

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")
        return False

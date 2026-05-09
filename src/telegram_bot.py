"""
NinjaVan Telegram Bot — long-polling interface for the multi-agent chatbot.

Run:
    conda run -n ninjavan python src/telegram_bot.py

Commands:
    /start — greeting
    /reset — clear conversation history for this chat
"""
import asyncio
import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# Per-chat conversation history: {chat_id: [{"role": ..., "content": ...}]}
_histories: dict[int, list[dict]] = {}

INTENT_EMOJI = {
    "faq": "🔍",
    "tracking": "📦",
    "escalation": "🚨",
}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi! I'm the NinjaVan AI assistant.\n\n"
        "I can help you with:\n"
        "📦 *Parcel tracking* — just share your parcel ID (e.g. NV-100125)\n"
        "🔍 *FAQs* — delivery hours, returns, rescheduling, and more\n"
        "🚨 *Escalations* — complaints or requests to speak with a human\n\n"
        "Type /reset to start a new conversation.",
        parse_mode="Markdown",
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    _histories[chat_id] = []
    await update.message.reply_text("Conversation cleared. How can I help you?")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_message = update.message.text

    if chat_id not in _histories:
        _histories[chat_id] = []

    history = _histories[chat_id]

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        from src.agents.chatbot.orchestrator import chat, trim_history
        result = chat(user_message, history)

        intent_tag = INTENT_EMOJI.get(result["intent"], "💬")
        answer = result["answer"]
        reply = f"{intent_tag} {answer}"

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
        _histories[chat_id] = trim_history(history)

    except Exception as e:
        logging.exception("Orchestrator error")
        reply = f"Sorry, something went wrong: {e}"

    await update.message.reply_text(reply, parse_mode="Markdown")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Bot started — polling for messages…")
    app.run_polling()


if __name__ == "__main__":
    main()

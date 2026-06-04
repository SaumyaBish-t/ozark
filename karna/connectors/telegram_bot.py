"""Telegram interface.

    python main.py telegram

Requires TELEGRAM_BOT_TOKEN in .env (get one from @BotFather).
Optionally set TELEGRAM_ALLOWED_USER_IDS to restrict who can talk to it.

One Telegram chat == one Karna session. We key sessions by chat_id so
group chats (later — for org-scoped memory) get their own thread.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from karna.agent.core import Agent, AgentReply
from karna.config import settings
from karna.memory.memory import build_memory
from karna.memory.subscriptions import Subscriptions


log = logging.getLogger(__name__)


def _try_graph_store():
    try:
        from karna.memory.graph_store import GraphStore
        gs = GraphStore()
        with gs._driver.session() as s:  # noqa: SLF001
            s.run("RETURN 1").consume()
        return gs
    except Exception as e:
        log.warning("Neo4j unavailable, running without concept graph: %s", e)
        return None


# ---------- bot wiring ----------

def _build_agent() -> Agent:
    memory = build_memory()
    graph = _try_graph_store()
    return Agent(memory=memory, graph_store=graph)


def _is_allowed(user_id: int) -> bool:
    allowed = settings.allowed_user_ids
    return not allowed or user_id in allowed


def _session_id_for_chat(chat_id: int) -> str:
    return f"tg-{chat_id}"


# ---------- handlers ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not _is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "Hi — I'm Karna. Ask me anything about trading and I'll teach you, "
        "starting from whatever foundations you need.\n\n"
        "Try: *what is RSI?* or *teach me support and resistance*",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not _is_allowed(update.effective_user.id):
        return
    agent: Agent = context.application.bot_data["agent"]
    n_turns = agent.memory.count()
    sem_count = agent.memory.semantic.count() if agent.memory.semantic else 0
    await update.message.reply_text(
        f"Stored turns: {n_turns}\nSemantic-indexed: {sem_count}\nModel: {settings.ollama_model}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    msg = update.message
    if not user or not chat or not msg or not msg.text:
        return
    if not _is_allowed(user.id):
        await msg.reply_text("Sorry, this bot is restricted.")
        return

    agent: Agent = context.application.bot_data["agent"]
    subs: Subscriptions = context.application.bot_data["subs"]
    session_id = _session_id_for_chat(chat.id)

    # Record where to reach this user for proactive deliveries (quizzes, digests).
    # Cheap upsert — runs every message but write is small.
    subs.record_telegram(str(user.id), chat.id)

    await context.bot.send_chat_action(chat.id, ChatAction.TYPING)

    # Ollama call is sync + blocking; offload so the bot stays responsive.
    try:
        reply: AgentReply = await asyncio.to_thread(
            agent.handle, msg.text, session_id=session_id, user_id=str(user.id)
        )
    except Exception as e:
        log.exception("agent.handle failed")
        await msg.reply_text(f"Something broke: {e}")
        return

    # Telegram message limit is 4096 chars — chunk if needed.
    text = reply.text or "(empty reply)"
    for chunk in _chunked(text, 4000):
        await msg.reply_text(chunk)


def _chunked(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i:i + size]


# ---------- entry point ----------

def run() -> None:
    settings.ensure_dirs()
    if not settings.telegram_bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. Create a bot via @BotFather on Telegram, "
            "copy the token into .env, then re-run."
        )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, settings.log_level, logging.INFO),
    )

    app: Application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .build()
    )
    app.bot_data["agent"] = _build_agent()
    app.bot_data["subs"] = Subscriptions()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Karna Telegram bot starting… (Ctrl+C to stop)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()

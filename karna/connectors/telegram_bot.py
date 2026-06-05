"""Telegram interface.

    python main.py telegram

Requires TELEGRAM_BOT_TOKEN in .env (get one from @BotFather).
Optionally set TELEGRAM_ALLOWED_USER_IDS to restrict who can talk to it.

One Telegram chat == one Karna session. We key sessions by chat_id so
group chats (later — for org-scoped memory) get their own thread.

Slash commands match the CLI: /price /rsi /macd /bb /atr /sma /ema /chart
are deterministic (no LLM, instant). Plus Telegram-specific niceties:
/start /status /history /new.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from telegram import BotCommand, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from karna.agent.core import Agent, AgentReply, new_session_id
from karna.agent.market import format_indicator, format_quote
from karna.config import settings
from karna.data.router import get_router as get_data_router
from karna.memory.memory import build_memory
from karna.memory.subscriptions import Subscriptions
from karna.trading import indicators


log = logging.getLogger(__name__)


# ---------- infra helpers ----------

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


def _build_agent() -> Agent:
    memory = build_memory()
    graph = _try_graph_store()
    return Agent(memory=memory, graph_store=graph)


def _is_allowed(user_id: int) -> bool:
    allowed = settings.allowed_user_ids
    return not allowed or user_id in allowed


def _session_id_for_chat(chat_id: int) -> str:
    return f"tg-{chat_id}"


def _chunked(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i:i + size]


async def _guard(update: Update) -> bool:
    """Allowlist check + null-safety. Returns False if we should drop the message."""
    if not update.effective_user or not _is_allowed(update.effective_user.id):
        if update.message:
            await update.message.reply_text("Sorry, this bot is restricted.")
        return False
    return True


def _arg_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Join /command args back into a single space-separated string."""
    args = context.args or []
    return " ".join(args).strip()


def _ticker_arg(context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    """Extract the first arg as an uppercase ticker, or None."""
    arg = _arg_text(context)
    return arg.split()[0].upper() if arg else None


# ---------- session lifecycle ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    await update.message.reply_text(
        "Hi — I'm Karna. Ask me anything about trading and I'll teach you, "
        "starting from whatever foundations you need.\n\n"
        "Try: *what is RSI on RELIANCE?* or */price TCS*\n"
        "Hit / in the input box to see all commands.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    agent: Agent = context.application.bot_data["agent"]
    n_turns = agent.memory.count()
    sem_count = agent.memory.semantic.count() if agent.memory.semantic else 0
    await update.message.reply_text(
        f"Stored turns: {n_turns}\nSemantic-indexed: {sem_count}\nModel: {settings.ollama_model}"
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the current chat's session id so context starts fresh."""
    if not await _guard(update):
        return
    chat = update.effective_chat
    if not chat:
        return
    overrides = context.application.bot_data.setdefault("session_overrides", {})
    overrides[chat.id] = new_session_id()
    await update.message.reply_text("New session started — earlier turns won't appear in immediate context.")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat = update.effective_chat
    agent: Agent = context.application.bot_data["agent"]
    overrides = context.application.bot_data.get("session_overrides", {})
    session_id = overrides.get(chat.id) or _session_id_for_chat(chat.id)
    rows = agent.memory.history(session_id, limit=20)
    if not rows:
        await update.message.reply_text("(no history in this session)")
        return
    lines = [f"*{t.role}*: {t.content[:160]}" for t in rows]
    text = "\n\n".join(lines)
    for chunk in _chunked(text, 4000):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)


# ---------- deterministic market commands (no LLM) ----------

async def _send(update: Update, text: str) -> None:
    for chunk in _chunked(text, 4000):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)


async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    ticker = _ticker_arg(context)
    if not ticker:
        await update.message.reply_text("Usage: /price RELIANCE")
        return
    try:
        q = await asyncio.to_thread(get_data_router().get_quote, ticker)
        await _send(update, format_quote(q))
    except Exception as e:
        await update.message.reply_text(f"Couldn't fetch {ticker}: {e}")


async def _indicator_cmd(
    update: Update, context: ContextTypes.DEFAULT_TYPE, short: str,
) -> None:
    if not await _guard(update):
        return
    ticker = _ticker_arg(context)
    if not ticker:
        await update.message.reply_text(f"Usage: /{short} RELIANCE")
        return
    try:
        df = await asyncio.to_thread(get_data_router().get_ohlcv, ticker, days=120)
        result = await asyncio.to_thread(indicators.compute, short, df)
        await _send(update, f"*{ticker}* — {format_indicator(short, result)}")
    except Exception as e:
        await update.message.reply_text(f"Couldn't compute {short.upper()} for {ticker}: {e}")


async def cmd_rsi(update, context):  return await _indicator_cmd(update, context, "rsi")
async def cmd_macd(update, context): return await _indicator_cmd(update, context, "macd")
async def cmd_bb(update, context):   return await _indicator_cmd(update, context, "bb")
async def cmd_atr(update, context):  return await _indicator_cmd(update, context, "atr")
async def cmd_sma(update, context):  return await _indicator_cmd(update, context, "sma")
async def cmd_ema(update, context):  return await _indicator_cmd(update, context, "ema")


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    ticker = _ticker_arg(context)
    if not ticker:
        await update.message.reply_text("Usage: /chart RELIANCE")
        return
    try:
        df = await asyncio.to_thread(get_data_router().get_ohlcv, ticker, days=15)
    except Exception as e:
        await update.message.reply_text(f"Couldn't fetch {ticker}: {e}")
        return
    df = df.tail(10)
    lines = [f"*{ticker}* — last 10 sessions", "```"]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Date'].date()}  O {r['Open']:>9.2f}  H {r['High']:>9.2f}  "
            f"L {r['Low']:>9.2f}  C {r['Close']:>9.2f}"
        )
    lines.append("```")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ---------- natural-language path (LLM) ----------

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
    overrides = context.application.bot_data.get("session_overrides", {})
    session_id = overrides.get(chat.id) or _session_id_for_chat(chat.id)

    subs.record_telegram(str(user.id), chat.id)

    await context.bot.send_chat_action(chat.id, ChatAction.TYPING)

    try:
        reply: AgentReply = await asyncio.to_thread(
            agent.handle, msg.text, session_id=session_id, user_id=str(user.id)
        )
    except Exception as e:
        log.exception("agent.handle failed")
        await msg.reply_text(f"Something broke: {e}")
        return

    text = reply.text or "(empty reply)"
    for chunk in _chunked(text, 4000):
        await msg.reply_text(chunk)


# ---------- entry point ----------

# Commands shown in Telegram's UI menu when the user taps "/" in the input.
BOT_COMMAND_MENU: list[tuple[str, str]] = [
    ("start",   "Welcome / quick intro"),
    ("status",  "Show counters and active model"),
    ("price",   "/price TICKER — live quote"),
    ("rsi",     "/rsi TICKER — RSI with reading"),
    ("macd",    "/macd TICKER — MACD signal"),
    ("bb",      "/bb TICKER — Bollinger Bands"),
    ("atr",     "/atr TICKER — ATR (volatility)"),
    ("sma",     "/sma TICKER — 50-day SMA"),
    ("ema",     "/ema TICKER — 20-day EMA"),
    ("chart",   "/chart TICKER — last 10 OHLC bars"),
    ("history", "Recent turns in this session"),
    ("new",     "Start a fresh session id"),
]


async def _register_menu(app: Application) -> None:
    try:
        await app.bot.set_my_commands([BotCommand(c, d) for c, d in BOT_COMMAND_MENU])
        log.info("Telegram command menu published")
    except Exception as e:
        log.warning("Couldn't publish command menu: %s", e)


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
        .post_init(_register_menu)
        .build()
    )
    app.bot_data["agent"] = _build_agent()
    app.bot_data["subs"] = Subscriptions()
    app.bot_data["session_overrides"] = {}  # chat_id -> session_id

    # Session / meta
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("new",     cmd_new))
    app.add_handler(CommandHandler("history", cmd_history))
    # Deterministic market data
    app.add_handler(CommandHandler("price",  cmd_price))
    app.add_handler(CommandHandler("rsi",    cmd_rsi))
    app.add_handler(CommandHandler("macd",   cmd_macd))
    app.add_handler(CommandHandler("bb",     cmd_bb))
    app.add_handler(CommandHandler("atr",    cmd_atr))
    app.add_handler(CommandHandler("sma",    cmd_sma))
    app.add_handler(CommandHandler("ema",    cmd_ema))
    app.add_handler(CommandHandler("chart",  cmd_chart))
    # Catch-all natural language → agent
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Karna Telegram bot starting… (Ctrl+C to stop)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()

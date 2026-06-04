"""Outbound channel delivery.

Background jobs (scheduler, reflectors, alerts) push messages to users
without going through the live Telegram polling Application. We keep a
process-wide singleton telegram.Bot and call its async API via
asyncio.run() — fine because these calls are infrequent (quizzes, daily
digests) and we don't need the bot's update queue.

If TELEGRAM_BOT_TOKEN isn't set, sends are silently dropped with a log
warning. That keeps the rest of the system runnable without a token.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from karna.config import settings
from karna.memory.subscriptions import Subscriptions


log = logging.getLogger(__name__)


_bot = None  # lazy telegram.Bot singleton
_subs: Optional[Subscriptions] = None


def _get_bot():
    """Lazy import + construct telegram.Bot."""
    global _bot
    if _bot is not None:
        return _bot
    if not settings.telegram_bot_token:
        log.warning("TELEGRAM_BOT_TOKEN not set; outbound sends will be no-ops")
        return None
    from telegram import Bot
    _bot = Bot(token=settings.telegram_bot_token)
    return _bot


def _get_subs() -> Subscriptions:
    global _subs
    if _subs is None:
        _subs = Subscriptions()
    return _subs


def send_telegram(user_id: str, text: str) -> bool:
    """Push `text` to `user_id`'s known Telegram chat. Returns True on success.

    Safe to call from any thread or background scheduler. Drops silently
    (with a warning) when the user has no Telegram subscription or the
    bot token is missing.
    """
    bot = _get_bot()
    if bot is None:
        return False

    sub = _get_subs().get(user_id)
    if sub is None or sub.telegram_chat_id is None:
        log.info("no Telegram subscription for user %s; skipping send", user_id)
        return False

    async def _send():
        await bot.send_message(chat_id=sub.telegram_chat_id, text=text)

    try:
        asyncio.run(_send())
        return True
    except RuntimeError:
        # Already inside an event loop (e.g. caller is async) — schedule on it.
        loop = asyncio.get_event_loop()
        loop.create_task(_send())
        return True
    except Exception as e:
        log.exception("send_telegram to %s failed: %s", user_id, e)
        return False

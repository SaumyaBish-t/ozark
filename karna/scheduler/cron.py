"""APScheduler bootstrap.

Two cron-style jobs at Sprint 2:
    - daily_decay         : Ebbinghaus pass over the session store
    - hourly_review_check : pick a stale concept per subscribed user, push
                            a quiz to Telegram. Hourly is conservative;
                            tune once we see real usage.

Run with: `python main.py scheduler`

Uses BlockingScheduler so it can be the foreground process. If you want
to embed the scheduler inside the Telegram bot process, swap to
BackgroundScheduler — same job registration code.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from karna.config import settings
from karna.scheduler.retention_checker import daily_decay_job, hourly_review_job


log = logging.getLogger(__name__)


def build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone="Asia/Kolkata")

    # Decay sweep at 3 AM IST — quiet hours, no contention with the bot.
    sched.add_job(
        daily_decay_job,
        trigger=CronTrigger(hour=3, minute=0),
        id="daily_decay",
        replace_existing=True,
    )

    # Review/quiz nudge once an hour, on the hour. Each job decides per-user
    # whether anything is actually due (cheap Neo4j query).
    sched.add_job(
        hourly_review_job,
        trigger=CronTrigger(minute=0),
        id="hourly_review",
        replace_existing=True,
    )

    return sched


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, settings.log_level, logging.INFO),
    )
    settings.ensure_dirs()
    sched = build_scheduler()
    log.info("Karna scheduler started. Jobs: %s", [j.id for j in sched.get_jobs()])
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopping")


if __name__ == "__main__":
    run()

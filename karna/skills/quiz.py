"""Quiz skill — generate, ask, judge.

Workflow:
    1. pick_concept_for_review(user_id) — Neo4j query for a concept whose
       comprehension edge hasn't been touched in N days
    2. generate_question(concept) — LLM produces a question via the
       generate_quiz skill's question_prompt
    3. judge_answer(concept, question, answer) — LLM judges, updates the
       graph's UNDERSTANDS.level

The scheduler triggers (1) → posts (2) to Telegram. When the user replies,
the message handler routes to (3).

Sprint 2 ships the functions standalone. Sprint 6 will wrap them with the
skill versioning + evaluator + reflector machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from karna.llm.ollama import OllamaClient, get_client as get_ollama
from karna.skills.manager import load as load_skill
from karna.trading.concepts import Concept, get as get_concept


log = logging.getLogger(__name__)


# Default "stale" cutoff. A concept whose comprehension edge is older than
# this is a candidate for review. Calibrate against real user data later.
REVIEW_STALE_DAYS = 3


@dataclass
class QuizPrompt:
    """Output of generate(): what to send the user."""
    concept_id: str
    concept_name: str
    last_level: str
    question: str


@dataclass
class QuizJudgement:
    """Output of judge(): graded answer + feedback."""
    new_level: str           # mastered | solid | reviewed | initial
    feedback: str
    raw: str


def pick_concept_for_review(
    graph_store,
    user_id: str,
    *,
    stale_days: int = REVIEW_STALE_DAYS,
) -> Optional[tuple[Concept, str]]:
    """Return (concept, last_level) for a concept the user should review.

    Strategy: pick the user's UNDERSTANDS edges whose level is below
    'mastered' and whose `updated_at` is older than `stale_days`. Prefer
    the oldest one — that's what's most at risk of being forgotten.

    Returns None if there's nothing due.
    """
    cutoff_ms = stale_days * 24 * 60 * 60 * 1000
    with graph_store._driver.session() as s:  # noqa: SLF001
        rec = s.run(
            """
            MATCH (:User {id: $uid})-[r:UNDERSTANDS]->(c:Concept)
            WHERE r.level <> 'mastered'
              AND timestamp() - coalesce(r.updated_at, 0) > $cutoff
            RETURN c.id AS cid, r.level AS level, r.updated_at AS ts
            ORDER BY ts ASC
            LIMIT 1
            """,
            uid=user_id, cutoff=cutoff_ms,
        ).single()
    if not rec:
        return None
    try:
        concept = get_concept(rec["cid"])
    except KeyError:
        log.warning("graph references unknown concept id: %s", rec["cid"])
        return None
    return concept, rec["level"]


def generate_question(
    concept: Concept,
    last_level: str,
    *,
    llm: Optional[OllamaClient] = None,
) -> QuizPrompt:
    """Ask the LLM to generate a comprehension question for the concept."""
    llm = llm or get_ollama()
    skill = load_skill("generate_quiz")
    prompt = skill.prompt(
        "question_prompt",
        concept_name=concept.name,
        concept_summary=concept.summary or "(no summary on file)",
        last_level=last_level,
    )
    question = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.6,
    ).strip()
    # Strip surrounding quotes the model sometimes wraps the question in.
    if question and question[0] in '"“' and question[-1] in '"”':
        question = question[1:-1].strip()
    return QuizPrompt(
        concept_id=concept.id,
        concept_name=concept.name,
        last_level=last_level,
        question=question,
    )


def judge_answer(
    concept: Concept,
    question: str,
    answer: str,
    *,
    llm: Optional[OllamaClient] = None,
) -> QuizJudgement:
    """LLM-as-judge: grade the user's reply, return new comprehension level."""
    llm = llm or get_ollama()
    skill = load_skill("generate_quiz")
    prompt = skill.prompt(
        "judge_prompt",
        concept_name=concept.name,
        question=question,
        answer=answer,
    )
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
    ).strip()

    # Parse: first line is the label, rest is feedback.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    label = (lines[0].lower() if lines else "reviewed")
    label = label.split()[0].strip(".,:")  # tolerate trailing punctuation
    if label not in {"mastered", "solid", "reviewed", "initial"}:
        # Model didn't follow format — default to a conservative 'reviewed'.
        log.warning("judge returned unparsable label %r; defaulting to 'reviewed'", lines[0] if lines else "")
        label = "reviewed"
    feedback = " ".join(lines[1:]) if len(lines) > 1 else ""
    return QuizJudgement(new_level=label, feedback=feedback, raw=raw)


def apply_judgement(
    graph_store,
    user_id: str,
    judgement: QuizJudgement,
    concept_id: str,
) -> None:
    """Persist the new comprehension level on the user's graph edge."""
    graph_store.set_comprehension(user_id, concept_id, judgement.new_level)

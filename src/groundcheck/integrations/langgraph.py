"""``make_guard_node``: a LangGraph node factory that runs a Guard check as a graph
step.

Returns a plain async ``state -> state-updates`` function, LangGraph's own node
convention — this module never imports ``langgraph`` itself; the caller's
graph-building code does. See ``docs/SPEC.md`` §5.9.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from groundcheck.core.guard import Guard

DEFAULT_REPORT_KEY = "groundcheck_report"

GuardNode = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def make_guard_node(
    guard: Guard,
    answer_key: str,
    evidence_key: str,
    report_key: str = DEFAULT_REPORT_KEY,
    question_key: str | None = None,
) -> GuardNode:
    """Build an async LangGraph node.

    Reads ``answer_key``/``evidence_key`` from state, runs a Guard check, and
    returns the state updates: ``answer_key`` (replaced with ``safe_answer``) and
    ``report_key`` (the full report).

    Args:
        guard: The ``Guard`` to check graph state with.
        answer_key: State key holding the generated answer.
        evidence_key: State key holding the retrieved evidence.
        report_key: State key to attach the ``GuardReport`` under.
        question_key: State key holding the question, if any. When ``None``, no
            question context is passed to the guard.

    Returns:
        An async ``state -> state-updates`` function, ready to pass to
        ``graph.add_node(...)``.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        answer = state[answer_key]
        evidence = state[evidence_key]
        question = state.get(question_key, "") if question_key else ""

        report = await guard.acheck(answer, evidence, question=question)

        return {answer_key: report.safe_answer, report_key: report}

    return node

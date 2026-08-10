"""``GroundCheckCallback``: a LangChain callback that verifies chain outputs against
their retrieved documents.

Requires the ``[langchain]`` extra — imports ``langchain_core`` at module level,
since nothing in GroundCheck's core ever imports this integration module (only an
explicit ``from groundcheck.integrations.langchain import ...`` does). See
``docs/SPEC.md`` §5.9.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from groundcheck.core.guard import Guard

DEFAULT_ANSWER_KEY = "answer"
DEFAULT_DOCUMENTS_KEY = "documents"
DEFAULT_REPORT_KEY = "groundcheck_report"


class GroundCheckCallback(AsyncCallbackHandler):  # type: ignore[misc]
    # ^ mypy's strict mode forbids subclassing an Any-typed base, which is what
    # AsyncCallbackHandler resolves to when langchain_core isn't installed
    # (see the ignore_missing_imports override in pyproject.toml). Real
    # subclassing works fine at runtime; this only affects unstubbed type-checking.
    """On ``on_chain_end``, verifies ``outputs[answer_key]`` against
    ``outputs[documents_key]`` and mutates ``outputs`` in place: replaces the
    answer with ``report.safe_answer`` and attaches the report under
    ``report_key``. A no-op if either key is missing from ``outputs``.
    """

    def __init__(
        self,
        guard: Guard,
        answer_key: str = DEFAULT_ANSWER_KEY,
        documents_key: str = DEFAULT_DOCUMENTS_KEY,
        report_key: str = DEFAULT_REPORT_KEY,
        question_key: str | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
            guard: The ``Guard`` to check chain outputs with.
            answer_key: Key in ``outputs`` holding the generated answer.
            documents_key: Key in ``outputs`` holding the retrieved documents.
            report_key: Key to attach the ``GuardReport`` under.
            question_key: Key in ``outputs`` holding the question, if any. When
                ``None``, no question context is passed to the guard.
        """
        self._guard = guard
        self._answer_key = answer_key
        self._documents_key = documents_key
        self._report_key = report_key
        self._question_key = question_key

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """LangChain hook: verify and mutate ``outputs`` in place.

        Args:
            outputs: The chain's output dict. Mutated in place — see the class
                docstring.
            **kwargs: Other LangChain callback arguments (``run_id``, etc.),
                unused here.
        """
        if self._answer_key not in outputs or self._documents_key not in outputs:
            return

        answer = outputs[self._answer_key]
        evidence = _coerce_evidence(outputs[self._documents_key])
        question = outputs.get(self._question_key, "") if self._question_key else ""

        report = await self._guard.acheck(answer, evidence, question=question)
        outputs[self._report_key] = report
        outputs[self._answer_key] = report.safe_answer


def _coerce_evidence(documents: Any) -> list[Any]:
    """LangChain retriever docs expose ``.page_content``; plain strings/dicts pass
    through untouched for ``Guard`` to normalize itself."""
    coerced = []
    for doc in documents:
        text = getattr(doc, "page_content", None)
        coerced.append(text if text is not None else doc)
    return coerced

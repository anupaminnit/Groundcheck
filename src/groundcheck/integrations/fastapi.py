"""``@guarded``: an endpoint decorator that runs a Guard check on the response.

Framework-agnostic by design: it only needs the endpoint's return value to expose
``answer_field``/``evidence_field`` as dict keys or object attributes, so this
module never imports ``fastapi`` itself and works with any async endpoint shaped
that way. Uses ``functools.wraps`` so FastAPI's own signature introspection (for
request parsing and OpenAPI docs) still sees the original function. See
``docs/SPEC.md`` §5.9.

Note: if the endpoint returns a Pydantic model, ``report_field`` (and
``answer_field``) must already be declared fields on it — Pydantic v2 rejects
assigning undeclared attributes by default.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from groundcheck.core.guard import Guard

DEFAULT_REPORT_FIELD = "groundcheck"

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def guarded(
    guard: Guard,
    answer_field: str = "answer",
    evidence_field: str = "evidence",
    report_field: str | None = DEFAULT_REPORT_FIELD,
    question_field: str | None = None,
) -> Callable[[F], F]:
    """Decorate an async endpoint: verify its response, replace the answer with
    ``safe_answer``, and attach the report under ``report_field``.

    Args:
        guard: The ``Guard`` to check the endpoint's response with.
        answer_field: Field on the response holding the generated answer.
        evidence_field: Field on the response holding the retrieved evidence.
        report_field: Field to attach the ``GuardReport`` under. Pass ``None`` to
            skip attaching it.
        question_field: Field on the response holding the question, if any. When
            ``None``, no question context is passed to the guard.

    Returns:
        A decorator for an async endpoint function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            response = await func(*args, **kwargs)

            answer = _get_field(response, answer_field)
            evidence = _get_field(response, evidence_field)
            question = _get_field(response, question_field) if question_field else ""

            report = await guard.acheck(answer, evidence, question=question)

            _set_field(response, answer_field, report.safe_answer)
            if report_field is not None:
                _set_field(response, report_field, report)
            return response

        return wrapper  # type: ignore[return-value]

    return decorator


def _get_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj[field]
    return getattr(obj, field)


def _set_field(obj: Any, field: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[field] = value
    else:
        setattr(obj, field, value)

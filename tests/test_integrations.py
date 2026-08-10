"""Tests for integrations/: GroundCheckCallback (LangChain), make_guard_node
(LangGraph), and the guarded decorator (framework-agnostic FastAPI-style).

LangGraph and FastAPI aren't imported here at all — neither integration module
needs them (see their docstrings). ``langchain_core`` *is* a real dependency for
``GroundCheckCallback`` (it subclasses ``AsyncCallbackHandler``), so those tests
import it lazily via the ``callback_cls`` fixture and skip if it's not installed —
``pip install -e ".[dev]"`` alone must still leave the rest of this file runnable.
"""

from __future__ import annotations

import inspect
import json
import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from fakes import FakeProvider
from groundcheck.core.guard import Guard
from groundcheck.core.schemas import GuardReport
from groundcheck.integrations.fastapi import guarded
from groundcheck.integrations.langgraph import make_guard_node

pytestmark = pytest.mark.asyncio


@pytest.fixture
def callback_cls() -> Any:
    pytest.importorskip("langchain_core")
    from groundcheck.integrations.langchain import GroundCheckCallback

    return GroundCheckCallback


def _extractor_response() -> str:
    return json.dumps(
        [
            {
                "text": "Paris is the capital of France.",
                "source_sentence": "Paris is the capital of France.",
                "type": "claim",
            }
        ]
    )


def _judge_response(verdict: str = "SUPPORTED") -> str:
    return json.dumps(
        [
            {
                "claim_id": "claim_0",
                "verdict": verdict,
                "confidence": 0.9,
                "evidence_ids": [],
                "rationale": "ok",
            }
        ]
    )


def _guard() -> Guard:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    return Guard(verifier="llm", policy="annotate", provider=provider)


# --- GroundCheckCallback (LangChain) ----------------------------------------


async def test_callback_no_op_when_keys_missing(callback_cls: Any) -> None:
    callback = callback_cls(_guard())
    outputs = {"something_else": "value"}

    await callback.on_chain_end(outputs, run_id=uuid.uuid4())

    assert outputs == {"something_else": "value"}


async def test_callback_runs_check_and_mutates_outputs(callback_cls: Any) -> None:
    callback = callback_cls(_guard())
    outputs = {
        "answer": "Paris is the capital of France.",
        "documents": ["Paris is the capital of France."],
    }

    await callback.on_chain_end(outputs, run_id=uuid.uuid4())

    assert isinstance(outputs["groundcheck_report"], GuardReport)
    assert outputs["answer"] == outputs["groundcheck_report"].safe_answer


@dataclass
class _FakeDocument:
    page_content: str


async def test_callback_coerces_document_objects(callback_cls: Any) -> None:
    callback = callback_cls(_guard())
    outputs = {
        "answer": "Paris is the capital of France.",
        "documents": [_FakeDocument(page_content="Paris is the capital of France.")],
    }

    await callback.on_chain_end(outputs, run_id=uuid.uuid4())

    assert isinstance(outputs["groundcheck_report"], GuardReport)


async def test_callback_custom_keys(callback_cls: Any) -> None:
    callback = callback_cls(
        _guard(), answer_key="ans", documents_key="docs", report_key="report"
    )
    outputs = {"ans": "Paris is the capital of France.", "docs": ["Paris is the capital."]}

    await callback.on_chain_end(outputs, run_id=uuid.uuid4())

    assert "report" in outputs
    assert "groundcheck_report" not in outputs


async def test_callback_passes_question_through(callback_cls: Any) -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    guard = Guard(verifier="llm", policy="log", provider=provider)
    callback = callback_cls(guard, question_key="question")
    outputs = {
        "answer": "Paris is the capital of France.",
        "documents": ["Paris is the capital of France."],
        "question": "What is the capital of France?",
    }

    await callback.on_chain_end(outputs, run_id=uuid.uuid4())

    judge_call_user_text = provider.calls[1][1]
    assert "What is the capital of France?" in judge_call_user_text


# --- make_guard_node (LangGraph) --------------------------------------------


async def test_guard_node_returns_partial_state_update() -> None:
    node = make_guard_node(_guard(), answer_key="answer", evidence_key="evidence")
    state = {
        "answer": "Paris is the capital of France.",
        "evidence": ["Paris is the capital of France."],
        "unrelated": "untouched",
    }

    result = await node(state)

    assert set(result.keys()) == {"answer", "groundcheck_report"}
    assert isinstance(result["groundcheck_report"], GuardReport)
    assert result["answer"] == result["groundcheck_report"].safe_answer


async def test_guard_node_custom_report_key() -> None:
    node = make_guard_node(
        _guard(), answer_key="answer", evidence_key="evidence", report_key="report"
    )
    state = {"answer": "Paris is the capital of France.", "evidence": ["Paris."]}

    result = await node(state)

    assert set(result.keys()) == {"answer", "report"}


async def test_guard_node_passes_question_through() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    guard = Guard(verifier="llm", policy="log", provider=provider)
    node = make_guard_node(
        guard, answer_key="answer", evidence_key="evidence", question_key="question"
    )
    state = {
        "answer": "Paris is the capital of France.",
        "evidence": ["Paris is the capital of France."],
        "question": "What is the capital of France?",
    }

    await node(state)

    judge_call_user_text = provider.calls[1][1]
    assert "What is the capital of France?" in judge_call_user_text


# --- guarded (framework-agnostic FastAPI-style decorator) -------------------


async def test_guarded_dict_response() -> None:
    @guarded(_guard(), answer_field="answer", evidence_field="evidence")
    async def endpoint() -> dict[str, object]:
        return {"answer": "Paris is the capital of France.", "evidence": ["Paris."]}

    result = await endpoint()

    assert isinstance(result["groundcheck"], GuardReport)
    assert result["answer"] == result["groundcheck"].safe_answer


class _RagResponse(BaseModel):
    answer: str
    evidence: list[str]
    groundcheck: GuardReport | None = None


async def test_guarded_pydantic_model_response() -> None:
    @guarded(_guard(), answer_field="answer", evidence_field="evidence")
    async def endpoint() -> _RagResponse:
        return _RagResponse(answer="Paris is the capital of France.", evidence=["Paris."])

    result = await endpoint()

    assert isinstance(result.groundcheck, GuardReport)
    assert result.answer == result.groundcheck.safe_answer


async def test_guarded_report_field_none_skips_attachment() -> None:
    @guarded(_guard(), answer_field="answer", evidence_field="evidence", report_field=None)
    async def endpoint() -> dict[str, object]:
        return {"answer": "Paris is the capital of France.", "evidence": ["Paris."]}

    result = await endpoint()

    assert "groundcheck" not in result


async def test_guarded_preserves_function_signature() -> None:
    async def endpoint(query: str) -> dict[str, object]:
        return {"answer": query, "evidence": []}

    wrapped = guarded(_guard(), answer_field="answer", evidence_field="evidence")(endpoint)

    assert inspect.signature(wrapped) == inspect.signature(endpoint)


async def test_guarded_passes_question_field() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    @guarded(guard, answer_field="answer", evidence_field="evidence", question_field="question")
    async def endpoint() -> dict[str, object]:
        return {
            "answer": "Paris is the capital of France.",
            "evidence": ["Paris is the capital of France."],
            "question": "What is the capital of France?",
        }

    await endpoint()

    judge_call_user_text = provider.calls[1][1]
    assert "What is the capital of France?" in judge_call_user_text

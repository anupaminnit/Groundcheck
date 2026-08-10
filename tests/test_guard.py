"""Tests for core.guard.Guard: the Phase 1 acceptance criterion (llm verifier + log
policy, against FakeProvider, returns a valid GuardReport), fail-open behavior, the
Phase 2 policy integrations (annotate/redact/block) plus the on_block hook, and the
Phase 3 local/hybrid verifier wiring.
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeProvider
from groundcheck.core.errors import ConfigError, VerifierError
from groundcheck.core.guard import Guard
from groundcheck.core.schemas import Action, Verdict


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
                "evidence_ids": ["chunk_0"],
                "rationale": "ok",
            }
        ]
    )


@pytest.mark.asyncio
async def test_guard_check_returns_valid_report() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    report = await guard.acheck(
        "Paris is the capital of France.",
        evidence=["Paris is the capital of France, per official records."],
        question="What is the capital of France?",
    )

    assert report.action == Action.PASS
    assert report.error is None
    assert report.grounded_score == 1.0
    assert len(report.claims) == 1
    assert report.claims[0].verdict == Verdict.SUPPORTED
    assert report.tokens.input > 0
    assert report.verifier == "llm"


def test_guard_sync_check_wraps_acheck() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    report = guard.check(
        "Paris is the capital of France.", evidence=["Paris is the capital of France."]
    )

    assert report.action == Action.PASS


@pytest.mark.asyncio
async def test_guard_check_raises_if_called_inside_running_loop() -> None:
    guard = Guard(verifier="llm", policy="log", provider=FakeProvider())

    with pytest.raises(RuntimeError):
        guard.check("answer", evidence=[])


@pytest.mark.asyncio
async def test_guard_fails_open_on_verifier_error() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), "not json", "still not json"])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    report = await guard.acheck("Paris is the capital of France.", evidence=["some evidence"])

    assert report.action == Action.ERROR
    assert report.grounded_score == -1.0
    assert report.error is not None
    assert report.safe_answer == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_guard_raises_when_fail_open_is_false() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), "not json", "still not json"])
    guard = Guard(verifier="llm", policy="log", provider=provider, fail_open=False)

    with pytest.raises(VerifierError):
        await guard.acheck("Paris is the capital of France.", evidence=["some evidence"])


@pytest.mark.asyncio
async def test_guard_empty_answer_short_circuits_to_full_score() -> None:
    guard = Guard(verifier="llm", policy="log", provider=FakeProvider())

    report = await guard.acheck("   ", evidence=["irrelevant"])

    assert report.grounded_score == 1.0
    assert report.claims == []
    assert report.action == Action.PASS


def test_guard_rejects_unimplemented_verifier() -> None:
    with pytest.raises(ConfigError):
        Guard(verifier="local", policy="log", provider=FakeProvider())


def test_guard_accepts_dict_evidence() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response()])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    report = guard.check(
        "Paris is the capital of France.",
        evidence=[{"id": "doc_1", "text": "Paris is the capital of France."}],
    )

    assert report.action == Action.PASS


def test_guard_requires_provider_when_none_given() -> None:
    with pytest.raises(ConfigError):
        Guard(verifier="llm", policy="log")


# --- Phase 2: policy integrations + on_block hook ---------------------------


@pytest.mark.asyncio
async def test_guard_annotate_policy_inserts_marker() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("UNSUPPORTED")])
    guard = Guard(verifier="llm", policy="annotate", provider=provider)

    report = await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert report.action == Action.ANNOTATE
    assert "⚠[unverified]" in report.safe_answer


@pytest.mark.asyncio
async def test_guard_redact_policy_removes_flagged_claim() -> None:
    answer = "Paris is the capital of France. It has a population of 5 million people."
    sentence1 = "Paris is the capital of France."
    sentence2 = "It has a population of 5 million people."
    extractor_response = json.dumps(
        [
            {"text": sentence1, "source_sentence": sentence1, "type": "claim"},
            {"text": sentence2, "source_sentence": sentence2, "type": "claim"},
        ]
    )
    judge_response = json.dumps(
        [
            {
                "claim_id": "claim_0",
                "verdict": "CONTRADICTED",
                "confidence": 0.9,
                "evidence_ids": [],
                "rationale": "wrong",
            },
            {
                "claim_id": "claim_1",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence_ids": [],
                "rationale": "ok",
            },
        ]
    )
    provider = FakeProvider(json_responses=[extractor_response, judge_response])
    guard = Guard(verifier="llm", policy="redact", provider=provider)

    report = await guard.acheck(answer, evidence=["irrelevant"])

    assert report.action == Action.REDACT
    assert sentence1 not in report.safe_answer
    assert sentence2 in report.safe_answer


@pytest.mark.asyncio
async def test_guard_redact_policy_escalates_to_block_when_mostly_removed() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("CONTRADICTED")])
    guard = Guard(verifier="llm", policy="redact", provider=provider)

    report = await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert report.action == Action.BLOCK


@pytest.mark.asyncio
async def test_guard_block_policy_replaces_answer_below_threshold() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("UNSUPPORTED")])
    guard = Guard(verifier="llm", policy="block", provider=provider)

    report = await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert report.action == Action.BLOCK
    assert report.safe_answer != "Paris is the capital of France."


@pytest.mark.asyncio
async def test_guard_block_policy_passes_above_threshold() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("SUPPORTED")])
    guard = Guard(verifier="llm", policy="block", provider=provider)

    report = await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert report.action == Action.PASS
    assert report.safe_answer == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_guard_block_policy_invokes_on_block_hook() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("UNSUPPORTED")])
    calls: list[object] = []
    guard = Guard(verifier="llm", policy="block", provider=provider, on_block=calls.append)

    await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_guard_on_block_hook_failure_does_not_break_report() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("UNSUPPORTED")])

    def bad_hook(report: object) -> None:
        raise RuntimeError("boom")

    guard = Guard(verifier="llm", policy="block", provider=provider, on_block=bad_hook)

    report = await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert report.action == Action.BLOCK


@pytest.mark.asyncio
async def test_guard_on_block_hook_supports_async_callbacks() -> None:
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("UNSUPPORTED")])
    calls: list[object] = []

    async def async_hook(report: object) -> None:
        calls.append(report)

    guard = Guard(verifier="llm", policy="block", provider=provider, on_block=async_hook)

    await guard.acheck("Paris is the capital of France.", evidence=["irrelevant"])

    assert len(calls) == 1


# --- Phase 3: local/hybrid verifier wiring ----------------------------------


def test_guard_local_verifier_needs_no_provider_but_needs_local_extra() -> None:
    # local mode must never call build_provider() — if it raised the "no provider
    # configured" ConfigError instead of this one, the wiring would be wrong.
    with pytest.raises(ConfigError, match=r"\[local\]"):
        Guard(verifier="local", policy="log")


def test_guard_hybrid_verifier_also_constructs_nli_verifier() -> None:
    # hybrid needs both a provider (has one here) and NLIVerifier, which fails in
    # this dev environment since torch/transformers aren't installed.
    with pytest.raises(ConfigError, match=r"\[local\]"):
        Guard(verifier="hybrid", policy="log", provider=FakeProvider())

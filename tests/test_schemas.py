"""Tests for core.schemas: model construction, enums, and TokenUsage addition."""

from __future__ import annotations

from groundcheck.core.schemas import (
    Action,
    Claim,
    ClaimVerdict,
    Evidence,
    GuardReport,
    TokenUsage,
    Verdict,
)


def test_evidence_defaults_metadata_to_empty_dict() -> None:
    ev = Evidence(id="chunk_0", text="hello")
    assert ev.metadata == {}


def test_claim_verdict_defaults() -> None:
    claim = Claim(id="claim_0", text="Paris is in France.", span_start=0, span_end=20)
    verdict = ClaimVerdict(claim=claim, verdict=Verdict.SUPPORTED, confidence=0.9)
    assert verdict.evidence_ids == []
    assert verdict.rationale == ""


def test_token_usage_addition() -> None:
    a = TokenUsage(input=10, output=5)
    b = TokenUsage(input=3, output=2)
    total = a + b
    assert total.input == 13
    assert total.output == 7


def test_guard_report_construction() -> None:
    claim = Claim(id="claim_0", text="x", span_start=0, span_end=1)
    verdict = ClaimVerdict(claim=claim, verdict=Verdict.SUPPORTED, confidence=1.0)
    report = GuardReport(
        grounded_score=1.0,
        action=Action.PASS,
        safe_answer="x",
        claims=[verdict],
        verifier="llm",
        latency_ms=5,
        tokens=TokenUsage(),
    )
    assert report.error is None
    assert report.action is Action.PASS


def test_verdict_enum_values() -> None:
    assert {v.value for v in Verdict} == {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "NOT_A_CLAIM",
    }

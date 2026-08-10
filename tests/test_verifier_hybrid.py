"""Tests for core.verifier.hybrid.HybridVerifier: only in-band claims get escalated
to the LLM judge, and its verdicts override the NLI ones for those claims only.
"""

from __future__ import annotations

import pytest

from fakes import FakeVerifier
from groundcheck.core.schemas import Claim, ClaimVerdict, Evidence, TokenUsage, Verdict
from groundcheck.core.verifier.hybrid import HybridVerifier

pytestmark = pytest.mark.asyncio


class _RecordingVerifier:
    """Records every ``pairs`` argument it's called with, then returns preset verdicts."""

    def __init__(self, verdicts: list[ClaimVerdict]) -> None:
        self._verdicts = verdicts
        self.received_pairs: list[list[tuple[Claim, list[Evidence]]]] = []

    async def verify(
        self, pairs: list[tuple[Claim, list[Evidence]]], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        self.received_pairs.append(pairs)
        return list(self._verdicts), TokenUsage()


def _claim(i: int) -> Claim:
    return Claim(id=f"claim_{i}", text=f"claim {i}", span_start=0, span_end=1)


def _verdict(claim: Claim, verdict: Verdict, confidence: float) -> ClaimVerdict:
    return ClaimVerdict(claim=claim, verdict=verdict, confidence=confidence)


async def test_hybrid_escalates_only_in_band_claims() -> None:
    claims = [_claim(i) for i in range(4)]
    pairs = [(c, []) for c in claims]

    nli_verdicts = [
        _verdict(claims[0], Verdict.SUPPORTED, 0.9),  # above band
        _verdict(claims[1], Verdict.UNSUPPORTED, 0.5),  # in band
        _verdict(claims[2], Verdict.UNSUPPORTED, 0.1),  # below band
        _verdict(claims[3], Verdict.CONTRADICTED, 0.6),  # in band
    ]
    nli = FakeVerifier(verdicts=nli_verdicts)

    llm_verdicts = [
        _verdict(claims[1], Verdict.SUPPORTED, 0.95),
        _verdict(claims[3], Verdict.SUPPORTED, 0.95),
    ]
    llm = _RecordingVerifier(llm_verdicts)

    hybrid = HybridVerifier(nli, llm, escalation_band=(0.35, 0.75))

    result, _ = await hybrid.verify(pairs, question="q")

    assert len(llm.received_pairs) == 1
    escalated_claim_ids = [claim.id for claim, _ in llm.received_pairs[0]]
    assert escalated_claim_ids == ["claim_1", "claim_3"]

    assert result[0].verdict == Verdict.SUPPORTED  # untouched — was above band
    assert result[1].verdict == Verdict.SUPPORTED  # escalated, overridden
    assert result[2].verdict == Verdict.UNSUPPORTED  # untouched — was below band
    assert result[3].verdict == Verdict.SUPPORTED  # escalated, overridden


async def test_hybrid_does_not_call_llm_when_nothing_is_in_band() -> None:
    claims = [_claim(0), _claim(1)]
    pairs = [(c, []) for c in claims]
    nli_verdicts = [
        _verdict(claims[0], Verdict.SUPPORTED, 0.95),
        _verdict(claims[1], Verdict.UNSUPPORTED, 0.05),
    ]
    nli = FakeVerifier(verdicts=nli_verdicts)
    llm = _RecordingVerifier([])

    hybrid = HybridVerifier(nli, llm, escalation_band=(0.35, 0.75))

    result, tokens = await hybrid.verify(pairs, question="q")

    assert llm.received_pairs == []
    assert result == nli_verdicts
    assert tokens.input == 0


async def test_hybrid_sums_token_usage_across_both_verifiers() -> None:
    claim = _claim(0)
    pairs = [(claim, [])]
    nli = FakeVerifier(verdicts=[_verdict(claim, Verdict.UNSUPPORTED, 0.5)])

    class _TokenSpendingVerifier(_RecordingVerifier):
        async def verify(
            self, pairs: list[tuple[Claim, list[Evidence]]], question: str
        ) -> tuple[list[ClaimVerdict], TokenUsage]:
            verdicts, _ = await super().verify(pairs, question)
            return verdicts, TokenUsage(input=5, output=5)

    llm = _TokenSpendingVerifier([_verdict(claim, Verdict.SUPPORTED, 0.9)])
    hybrid = HybridVerifier(nli, llm, escalation_band=(0.35, 0.75))

    _, tokens = await hybrid.verify(pairs, question="q")

    assert tokens.input == 5
    assert tokens.output == 5


async def test_hybrid_empty_pairs_returns_empty_without_calling_either_verifier() -> None:
    nli = FakeVerifier(verdicts=[_verdict(_claim(0), Verdict.SUPPORTED, 0.9)])
    llm = _RecordingVerifier([])

    hybrid = HybridVerifier(nli, llm, escalation_band=(0.35, 0.75))

    result, tokens = await hybrid.verify([], question="q")

    assert result == []
    assert tokens.input == 0
    assert llm.received_pairs == []

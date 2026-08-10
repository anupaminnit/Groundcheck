"""Tests for core.verifier.llm_judge.LLMJudgeVerifier, including the malformed-JSON
repair-retry path required by the Phase 1 acceptance criteria in docs/SPEC.md §9.
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeProvider
from groundcheck.core.errors import VerifierError
from groundcheck.core.schemas import Claim, Evidence, Verdict
from groundcheck.core.verifier.llm_judge import LLMJudgeVerifier

pytestmark = pytest.mark.asyncio


def _claim(i: int, text: str = "claim text") -> Claim:
    return Claim(id=f"claim_{i}", text=text, span_start=0, span_end=len(text))


async def test_verify_happy_path() -> None:
    claim = _claim(0)
    evidence = [Evidence(id="chunk_0", text="supporting evidence")]
    response = json.dumps(
        [
            {
                "claim_id": "claim_0",
                "verdict": "SUPPORTED",
                "confidence": 0.95,
                "evidence_ids": ["chunk_0"],
                "rationale": "matches",
            }
        ]
    )
    provider = FakeProvider(json_responses=[response])
    verifier = LLMJudgeVerifier(provider)

    verdicts, tokens = await verifier.verify([(claim, evidence)], question="q")

    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.SUPPORTED
    assert verdicts[0].confidence == 0.95
    assert tokens.input == 10
    assert tokens.output == 10


async def test_verify_recovers_via_repair_retry() -> None:
    claim = _claim(0)
    good = json.dumps(
        [
            {
                "claim_id": "claim_0",
                "verdict": "CONTRADICTED",
                "confidence": 0.8,
                "evidence_ids": [],
                "rationale": "mismatch",
            }
        ]
    )
    provider = FakeProvider(json_responses=["not json at all", good])
    verifier = LLMJudgeVerifier(provider)

    verdicts, tokens = await verifier.verify([(claim, [])], question="q")

    assert verdicts[0].verdict == Verdict.CONTRADICTED
    assert tokens.input == 20  # two calls, 10 tokens each
    assert len(provider.calls) == 2
    assert "valid JSON array" in provider.calls[1][1]


async def test_verify_raises_verifier_error_after_second_failure() -> None:
    claim = _claim(0)
    provider = FakeProvider(json_responses=["still not json", "also not json"])
    verifier = LLMJudgeVerifier(provider)

    with pytest.raises(VerifierError):
        await verifier.verify([(claim, [])], question="q")


async def test_verify_fills_in_missing_claim_ids() -> None:
    claim = _claim(0)
    provider = FakeProvider(json_responses=[json.dumps([])])
    verifier = LLMJudgeVerifier(provider)

    verdicts, _ = await verifier.verify([(claim, [])], question="q")

    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.UNSUPPORTED
    assert verdicts[0].confidence == 0.0


async def test_verify_chunks_large_claim_lists() -> None:
    claims = [_claim(i) for i in range(45)]
    pairs = [(c, []) for c in claims]

    def response_for(chunk_claims: list[Claim]) -> str:
        return json.dumps(
            [
                {
                    "claim_id": c.id,
                    "verdict": "SUPPORTED",
                    "confidence": 1.0,
                    "evidence_ids": [],
                    "rationale": "",
                }
                for c in chunk_claims
            ]
        )

    provider = FakeProvider(
        json_responses=[
            response_for(claims[0:20]),
            response_for(claims[20:40]),
            response_for(claims[40:45]),
        ]
    )
    verifier = LLMJudgeVerifier(provider, chunk_size=20)

    verdicts, _ = await verifier.verify(pairs, question="q")

    assert len(verdicts) == 45
    assert len(provider.calls) == 3


async def test_verify_empty_pairs_returns_empty() -> None:
    provider = FakeProvider()
    verifier = LLMJudgeVerifier(provider)

    verdicts, tokens = await verifier.verify([], question="q")

    assert verdicts == []
    assert tokens.input == 0

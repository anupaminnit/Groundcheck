"""Deterministic test doubles: ``FakeProvider``, ``FakeExtractor``, ``FakeVerifier``.

Unit tests never hit a real LLM or download a model — these doubles stand in for
providers, claim extractors, and verifiers wherever a real backend would go.
"""

from __future__ import annotations

import zlib

from groundcheck.core.schemas import Claim, ClaimVerdict, Evidence, TokenUsage


class FakeProvider:
    """A scripted ``LLMProvider``: ``complete_json`` returns queued responses in
    order; ``embed`` is a deterministic, dependency-free bag-of-words hash."""

    def __init__(self, json_responses: list[str] | None = None) -> None:
        self._responses = list(json_responses or [])
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str, timeout: float) -> tuple[str, TokenUsage]:
        self.calls.append((system, user))
        if not self._responses:
            raise RuntimeError("FakeProvider has no queued responses left.")
        return self._responses.pop(0), TokenUsage(input=10, output=10)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


def _hash_vector(text: str, dims: int = 16) -> list[float]:
    vector = [0.0] * dims
    for token in text.lower().split():
        vector[zlib.crc32(token.encode("utf-8")) % dims] += 1.0
    return vector


class FakeExtractor:
    """A ``ClaimExtractor`` stub returning a preset list of claims regardless of input."""

    def __init__(self, claims: list[Claim] | None = None) -> None:
        self._claims = claims or []

    async def extract(self, answer: str) -> tuple[list[Claim], TokenUsage]:
        return list(self._claims), TokenUsage()


class FakeVerifier:
    """A ``Verifier`` stub returning a preset list of verdicts regardless of input."""

    def __init__(self, verdicts: list[ClaimVerdict] | None = None) -> None:
        self._verdicts = verdicts or []

    async def verify(
        self, pairs: list[tuple[Claim, list[Evidence]]], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        return list(self._verdicts), TokenUsage()

"""Tests for core.evidence: EmbeddingMatcher and its lexical_match fallback."""

from __future__ import annotations

import pytest

from fakes import FakeProvider
from groundcheck.core.errors import ConfigError
from groundcheck.core.evidence import EmbeddingMatcher, lexical_match
from groundcheck.core.schemas import Claim, Evidence


def _claim(text: str, i: int = 0) -> Claim:
    return Claim(id=f"claim_{i}", text=text, span_start=0, span_end=len(text))


def _evidence(texts: list[str]) -> list[Evidence]:
    return [Evidence(id=f"chunk_{i}", text=t) for i, t in enumerate(texts)]


@pytest.mark.asyncio
async def test_embedding_matcher_returns_k_candidates() -> None:
    provider = FakeProvider()
    matcher = EmbeddingMatcher(provider)
    claims = [_claim("Paris is the capital of France")]
    evidence = _evidence(
        [
            "Paris is the capital of France",
            "Water boils at 100 degrees",
            "The Eiffel Tower is in Paris",
        ]
    )

    results = await matcher.match(claims, evidence, k=2)

    assert len(results) == 1
    assert len(results[0]) == 2


@pytest.mark.asyncio
async def test_embedding_matcher_empty_evidence() -> None:
    provider = FakeProvider()
    matcher = EmbeddingMatcher(provider)

    results = await matcher.match([_claim("some claim")], [], k=3)

    assert results == [[]]


@pytest.mark.asyncio
async def test_embedding_matcher_empty_claims() -> None:
    provider = FakeProvider()
    matcher = EmbeddingMatcher(provider)

    results = await matcher.match([], _evidence(["a"]), k=3)

    assert results == []


@pytest.mark.asyncio
async def test_embedding_matcher_falls_back_to_lexical_on_config_error() -> None:
    class NoEmbedProvider(FakeProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise ConfigError("no embeddings")

    matcher = EmbeddingMatcher(NoEmbedProvider())
    claims = [_claim("Paris is the capital of France")]
    evidence = _evidence(["Paris is the capital of France", "unrelated text about weather"])

    results = await matcher.match(claims, evidence, k=1)

    assert results[0][0].text == "Paris is the capital of France"


def test_lexical_match_ranks_by_token_overlap() -> None:
    claims = [_claim("Paris is the capital of France")]
    evidence = _evidence(["Paris is the capital of France", "Water boils at 100 degrees"])

    results = lexical_match(claims, evidence, k=1)

    assert results[0][0].text == "Paris is the capital of France"


def test_lexical_match_respects_k() -> None:
    results = lexical_match([_claim("Paris")], _evidence(["a", "b", "c"]), k=2)
    assert len(results[0]) == 2


def test_lexical_match_empty_inputs() -> None:
    assert lexical_match([], _evidence(["a"]), k=1) == []
    assert lexical_match([_claim("x")], [], k=1) == [[]]

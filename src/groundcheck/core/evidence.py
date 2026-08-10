"""Evidence matching: narrow each claim down to its top-k candidate evidence chunks.

See ``docs/SPEC.md`` §5.2. ``EmbeddingMatcher`` embeds claims and evidence in two
batched calls and ranks by cosine similarity; it falls back to ``lexical_match`` if
the provider has no embedding support (e.g. ``AnthropicProvider.embed`` raises
``ConfigError``). ``lexical_match`` is also used directly in local mode (Phase 3),
which has no provider at all.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import Claim, Evidence
from groundcheck.providers.base import LLMProvider

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
        "on", "at", "to", "for", "and", "or", "it", "this", "that", "with",
        "as", "by", "from", "has", "have", "had", "its",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EvidenceMatcher(Protocol):
    """Ranks candidate evidence per claim."""

    async def match(
        self, claims: list[Claim], evidence: list[Evidence], k: int
    ) -> list[list[Evidence]]:
        """Return the top-``k`` candidate evidence chunks for each claim.

        Args:
            claims: The claims to find candidates for.
            evidence: The full pool of evidence to select candidates from.
            k: How many candidates to return per claim.

        Returns:
            One list of candidates per claim (same order as ``claims``), each of
            length ``min(k, len(evidence))``, ordered by descending score.
        """
        ...


class EmbeddingMatcher:
    """Ranks evidence by cosine similarity between provider embeddings."""

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize the matcher.

        Args:
            provider: The LLM provider used for embedding calls.
        """
        self._provider = provider

    async def match(
        self, claims: list[Claim], evidence: list[Evidence], k: int
    ) -> list[list[Evidence]]:
        """Embed all claims and evidence (two batched calls) and rank by cosine
        similarity. Falls back to ``lexical_match`` if the provider has no
        embedding support.

        Args:
            claims: The claims to find candidates for.
            evidence: The full pool of evidence to select candidates from.
            k: How many candidates to return per claim.

        Returns:
            One list of candidates per claim, ordered by descending similarity.
        """
        if not claims:
            return []
        if not evidence:
            return [[] for _ in claims]

        k = min(k, len(evidence))
        try:
            claim_vectors = await self._provider.embed([c.text for c in claims])
            evidence_vectors = await self._provider.embed([e.text for e in evidence])
        except ConfigError:
            return lexical_match(claims, evidence, k)

        results = []
        for claim_vector in claim_vectors:
            scored = sorted(
                ((_cosine(claim_vector, ev), i) for i, ev in enumerate(evidence_vectors)),
                key=lambda pair: pair[0],
                reverse=True,
            )
            results.append([evidence[i] for _, i in scored[:k]])
        return results


class LexicalMatcher:
    """Ranks evidence by token overlap. Used in local mode, which has no provider."""

    async def match(
        self, claims: list[Claim], evidence: list[Evidence], k: int
    ) -> list[list[Evidence]]:
        """See ``EvidenceMatcher.match``. Delegates to ``lexical_match``."""
        return lexical_match(claims, evidence, k)


def lexical_match(claims: list[Claim], evidence: list[Evidence], k: int) -> list[list[Evidence]]:
    """Rank evidence per claim by token-overlap (Jaccard) score.

    Doesn't need to be great — it only narrows candidates; the verifier decides.

    Args:
        claims: The claims to find candidates for.
        evidence: The full pool of evidence to select candidates from.
        k: How many candidates to return per claim.

    Returns:
        One list of candidates per claim, ordered by descending Jaccard score.
    """
    if not claims:
        return []
    if not evidence:
        return [[] for _ in claims]

    k = min(k, len(evidence))
    evidence_tokens = [_tokenize(e.text) for e in evidence]
    results = []
    for claim in claims:
        claim_tokens = _tokenize(claim.text)
        scored = sorted(
            ((_jaccard(claim_tokens, tokens), i) for i, tokens in enumerate(evidence_tokens)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        results.append([evidence[i] for _, i in scored[:k]])
    return results


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

"""Claim extraction: turn an answer string into a list of atomic, offset-tagged claims.

Two implementations, per ``docs/SPEC.md`` §5.1:

- ``LLMClaimExtractor``: one JSON-mode call, semantically splits and filters claims.
- ``SentenceClaimExtractor``: deterministic sentence splitter, no LLM call. Used in
  local mode, and as ``LLMClaimExtractor``'s fallback when its output can't be
  parsed or yields zero claims.

Contract for both: claims are ordered by ``span_start``, spans never overlap, and any
non-empty answer yields at least one claim.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal, Protocol

from pydantic import BaseModel, TypeAdapter, ValidationError

from groundcheck.core.prompts import EXTRACTOR_V1
from groundcheck.core.schemas import Claim, TokenUsage
from groundcheck.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MIN_CLAIM_LEN = 25
_SENTENCE_END_RE = re.compile(r"[.!?]+")
_ABBREVIATIONS = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
        "vs.", "e.g.", "i.e.", "etc.", "u.s.", "u.k.", "no.", "approx.",
    }
)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ClaimExtractor(Protocol):
    """Turns an answer into a list of claims, plus any LLM tokens spent doing so."""

    async def extract(self, answer: str) -> tuple[list[Claim], TokenUsage]:
        """Extract atomic claims from an answer.

        Args:
            answer: The answer text to extract claims from.

        Returns:
            A tuple of the extracted claims (ordered by ``span_start``, never
            overlapping; at least one for a non-empty answer) and the token usage
            spent extracting them.
        """
        ...


class SentenceClaimExtractor:
    """Deterministic sentence splitter: every sentence becomes one claim."""

    async def extract(self, answer: str) -> tuple[list[Claim], TokenUsage]:
        """Split ``answer`` into sentences and return one claim per sentence.

        Args:
            answer: The answer text to split.

        Returns:
            A tuple of the claims and a zero ``TokenUsage`` (no LLM call is made).
        """
        spans = self._merge_short(self._split(answer))
        claims = [
            Claim(id=f"claim_{i}", text=answer[start:end], span_start=start, span_end=end)
            for i, (start, end) in enumerate(spans)
        ]
        return claims, TokenUsage()

    def _split(self, answer: str) -> list[tuple[int, int]]:
        if not answer.strip():
            return []
        bounds = [0]
        for match in _SENTENCE_END_RE.finditer(answer):
            end = match.end()
            if end >= len(answer):
                continue
            preceding = answer[max(0, match.start() - 6) : end].lower()
            if any(preceding.endswith(abbr) for abbr in _ABBREVIATIONS):
                continue
            rest = answer[end:]
            if rest and rest[0] not in " \n\t":
                continue  # e.g. a decimal like "3.14", not a sentence boundary
            following = rest.lstrip()
            if following and following[0].islower():
                continue
            bounds.append(end)
        bounds.append(len(answer))
        spans = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            trimmed = _trim_whitespace(answer, start, end)
            if trimmed is not None:
                spans.append(trimmed)
        return spans or [(0, len(answer))]

    @staticmethod
    def _merge_short(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(spans) <= 1:
            return spans
        merged = list(spans)
        i = 0
        while i < len(merged) - 1:
            start, end = merged[i]
            if (end - start) < _MIN_CLAIM_LEN:
                _, next_end = merged[i + 1]
                merged[i] = (start, next_end)
                del merged[i + 1]
                continue
            i += 1
        if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < _MIN_CLAIM_LEN:
            prev_start, _ = merged[-2]
            _, last_end = merged[-1]
            merged[-2] = (prev_start, last_end)
            merged.pop()
        return merged


def _trim_whitespace(answer: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


class _ExtractedItem(BaseModel):
    text: str
    source_sentence: str
    type: Literal["claim", "skip"]


class LLMClaimExtractor:
    """Semantic claim extraction via a single JSON-mode LLM call.

    Falls back to ``SentenceClaimExtractor`` (no extra LLM call) if the LLM's
    output can't be parsed, or yields zero claims for a non-empty answer.
    """

    def __init__(self, provider: LLMProvider, timeout: float = 30.0) -> None:
        """Initialize the extractor.

        Args:
            provider: The LLM provider used for the single extraction call.
            timeout: Per-call timeout, in seconds.
        """
        self._provider = provider
        self._timeout = timeout
        self._fallback = SentenceClaimExtractor()

    async def extract(self, answer: str) -> tuple[list[Claim], TokenUsage]:
        """Extract claims from ``answer`` via one JSON-mode LLM call.

        Args:
            answer: The answer text to extract claims from.

        Returns:
            A tuple of the extracted claims and the token usage of the LLM call
            (zero tokens if ``answer`` is empty and no call was made).
        """
        if not answer.strip():
            return [], TokenUsage()

        raw, tokens = await self._provider.complete_json(
            EXTRACTOR_V1, f"ANSWER:\n{answer}", self._timeout
        )
        try:
            items = _parse_extractor_json(raw)
        except (ValueError, ValidationError) as exc:
            logger.warning(
                "LLMClaimExtractor: could not parse extractor output (%s); "
                "falling back to SentenceClaimExtractor",
                exc,
            )
            claims, _ = await self._fallback.extract(answer)
            return claims, tokens

        claims = _claims_from_items(answer, items)
        if not claims:
            logger.warning(
                "LLMClaimExtractor: no claims extracted from a non-empty answer; "
                "falling back to SentenceClaimExtractor"
            )
            claims, _ = await self._fallback.extract(answer)
        return claims, tokens


def _parse_extractor_json(raw: str) -> list[_ExtractedItem]:
    data = json.loads(_strip_code_fences(raw))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of extracted items.")
    return TypeAdapter(list[_ExtractedItem]).validate_python(data)


def _strip_code_fences(raw: str) -> str:
    return _CODE_FENCE_RE.sub("", raw.strip()).strip()


def _claims_from_items(answer: str, items: list[_ExtractedItem]) -> list[Claim]:
    claim_items = [item for item in items if item.type == "claim"]
    spans = _locate_spans(answer, claim_items)
    paired = sorted(zip(spans, claim_items, strict=True), key=lambda pair: pair[0][0])
    return [
        Claim(id=f"claim_{i}", text=item.text, span_start=start, span_end=end)
        for i, ((start, end), item) in enumerate(paired)
    ]


def _locate_spans(answer: str, items: list[_ExtractedItem]) -> list[tuple[int, int]]:
    """Locate each item's source sentence with a cursor that only moves forward.

    This is what makes repeated sentences resolve to their correct, distinct
    occurrences instead of all matching the first one — see ``docs/SPEC.md`` §5.1.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0

    for item in items:
        sentence = item.source_sentence
        idx = answer.find(sentence, cursor)
        if idx != -1:
            span = (idx, idx + len(sentence))
        else:
            fuzzy = _fuzzy_find(answer, sentence, cursor)
            if fuzzy is not None:
                span = fuzzy
            else:
                logger.warning(
                    "LLMClaimExtractor: could not locate source sentence %r; "
                    "using whole-answer span",
                    sentence[:80],
                )
                span = (0, len(answer))

        spans.append(span)
        cursor = span[1]

    return spans


def _fuzzy_find(haystack: str, needle: str, start: int) -> tuple[int, int] | None:
    tokens = needle.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, haystack[start:])
    if match is None:
        return None
    return start + match.start(), start + match.end()

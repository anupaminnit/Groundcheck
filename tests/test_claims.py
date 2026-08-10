"""Tests for core.claims: SentenceClaimExtractor, LLMClaimExtractor, and offset math.

Offset correctness is the highest-risk code in this module (per docs/SPEC.md §10),
so it gets a dedicated parametrized sweep over 10 crafted answers, including unicode
and repeated-sentence cases.
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeProvider
from fixtures.golden import CASES
from groundcheck.core.claims import LLMClaimExtractor, SentenceClaimExtractor

pytestmark = pytest.mark.asyncio


def _extractor_response(items: list[dict[str, str]]) -> str:
    return json.dumps(items)


def _item(text: str, source: str, kind: str = "claim") -> dict[str, str]:
    return {"text": text, "source_sentence": source, "type": kind}


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


# Each case: (answer, extractor items in textual order). All spans are expected to
# resolve, in order, to exactly the `source_sentence` substring of `answer`.
OFFSET_CASES = [
    (
        "Paris is the capital of France.",
        [_item("Paris is the capital of France.", "Paris is the capital of France.")],
    ),
    (
        "Paris is the capital of France. It has a population of 2 million.",
        [
            _item("Paris is the capital of France.", "Paris is the capital of France."),
            _item("Paris has a population of 2 million.", "It has a population of 2 million."),
        ],
    ),
    (
        "Café Müller is in Zürich. The café serves a naïve résumé of Alpine cuisine. "
        "日本語のテキストです。",
        [
            _item("Café Müller is in Zürich.", "Café Müller is in Zürich."),
            _item(
                "The café serves Alpine cuisine.",
                "The café serves a naïve résumé of Alpine cuisine.",
            ),
            _item("The text is in Japanese.", "日本語のテキストです。"),
        ],
    ),
    (
        "The sky is blue. Water is wet. The sky is blue.",
        [
            _item("The sky is blue (first mention).", "The sky is blue."),
            _item("Water is wet.", "Water is wet."),
            _item("The sky is blue (second mention).", "The sky is blue."),
        ],
    ),
    (
        "The Eiffel Tower was completed in 1889. It stands 330 meters tall.",
        [
            _item(
                "The Eiffel Tower was completed in 1889.",
                "The Eiffel Tower was completed in 1889.",
            ),
            _item("The Eiffel Tower stands 330 meters tall.", "It stands 330 meters tall."),
        ],
    ),
    (
        "I think the weather is nice today. The temperature is 22 degrees Celsius.",
        [
            _item("The weather is nice today.", "I think the weather is nice today.", "skip"),
            _item(
                "The temperature is 22 degrees Celsius.",
                "The temperature is 22 degrees Celsius.",
            ),
        ],
    ),
    (
        "\n\n  Mount Everest is the tallest mountain.  \n\nIt is located in the Himalayas.\n",
        [
            _item(
                "Mount Everest is the tallest mountain.", "Mount Everest is the tallest mountain."
            ),
            _item("Mount Everest is located in the Himalayas.", "It is located in the Himalayas."),
        ],
    ),
    (
        "The   report  was    filed on time. It passed review.",
        [
            _item("The report was filed on time.", "The report was filed on time."),
            _item("The report passed review.", "It passed review."),
        ],
    ),
    (
        "Claim one is here. Claim two is here. Claim three is here.",
        [
            _item("Claim one.", "Claim one is here."),
            _item("Claim two.", "Claim two is here."),
            _item("Claim three.", "Claim three is here."),
        ],
    ),
    (
        "北京是中国的首都。 北京是中国的首都。 天气很好。",
        [
            _item("Beijing is the capital of China (first mention).", "北京是中国的首都。"),
            _item("Beijing is the capital of China (second mention).", "北京是中国的首都。"),
            _item("The weather is nice.", "天气很好。"),
        ],
    ),
]


@pytest.mark.parametrize("answer,items", OFFSET_CASES)
async def test_llm_extractor_offsets_correct(answer: str, items: list[dict[str, str]]) -> None:
    provider = FakeProvider(json_responses=[_extractor_response(items)])
    extractor = LLMClaimExtractor(provider)

    claims, _ = await extractor.extract(answer)

    expected_sentences = [item["source_sentence"] for item in items if item["type"] == "claim"]
    assert len(claims) == len(expected_sentences)
    for claim, sentence in zip(claims, expected_sentences, strict=True):
        # Whitespace-normalized: the fuzzy-match fallback can return a span whose
        # literal spacing differs from the LLM's normalized source_sentence.
        found = answer[claim.span_start : claim.span_end]
        assert _normalize_ws(found) == _normalize_ws(sentence)


async def test_llm_extractor_claims_ordered_and_non_overlapping() -> None:
    answer = "Claim one is here. Claim two is here. Claim three is here."
    items = OFFSET_CASES[8][1]
    provider = FakeProvider(json_responses=[_extractor_response(items)])
    extractor = LLMClaimExtractor(provider)

    claims, _ = await extractor.extract(answer)

    starts = [c.span_start for c in claims]
    assert starts == sorted(starts)
    for a, b in zip(claims, claims[1:], strict=False):
        assert a.span_end <= b.span_start


async def test_llm_extractor_skips_are_filtered() -> None:
    answer, items = OFFSET_CASES[5]
    provider = FakeProvider(json_responses=[_extractor_response(items)])
    extractor = LLMClaimExtractor(provider)

    claims, _ = await extractor.extract(answer)

    assert len(claims) == 1
    assert claims[0].text == "The temperature is 22 degrees Celsius."


async def test_llm_extractor_falls_back_on_malformed_json() -> None:
    provider = FakeProvider(json_responses=["not valid json"])
    extractor = LLMClaimExtractor(provider)
    answer = "The sky is blue. Water is wet."

    claims, _ = await extractor.extract(answer)

    assert len(claims) >= 1
    for claim in claims:
        assert claim.text == answer[claim.span_start : claim.span_end]


async def test_llm_extractor_falls_back_when_all_skipped() -> None:
    items = [_item("opinion", "I think this is great.", "skip")]
    provider = FakeProvider(json_responses=[_extractor_response(items)])
    extractor = LLMClaimExtractor(provider)

    claims, _ = await extractor.extract("I think this is great.")

    assert len(claims) == 1  # SentenceClaimExtractor fallback guarantees >=1


async def test_llm_extractor_empty_answer_returns_no_claims() -> None:
    provider = FakeProvider()
    extractor = LLMClaimExtractor(provider)

    claims, tokens = await extractor.extract("   ")

    assert claims == []
    assert tokens.input == 0
    assert tokens.output == 0


@pytest.mark.parametrize(
    "answer",
    [
        "Mount Everest is the tallest mountain. It is over 8,800 meters tall.",
        "café, résumé, naïve. 日本語のテキストです。",
        "Short. Bits. Merge these tiny fragments into a real claim.",
    ],
)
async def test_sentence_extractor_offsets_correct(answer: str) -> None:
    extractor = SentenceClaimExtractor()

    claims, tokens = await extractor.extract(answer)

    assert len(claims) >= 1
    assert tokens.input == 0
    assert tokens.output == 0
    for claim in claims:
        assert answer[claim.span_start : claim.span_end] == claim.text
    starts = [c.span_start for c in claims]
    assert starts == sorted(starts)
    for a, b in zip(claims, claims[1:], strict=False):
        assert a.span_end <= b.span_start


async def test_sentence_extractor_nonempty_answer_yields_at_least_one_claim() -> None:
    extractor = SentenceClaimExtractor()
    claims, _ = await extractor.extract("No terminal punctuation here")
    assert len(claims) == 1


async def test_sentence_extractor_empty_answer_yields_no_claims() -> None:
    extractor = SentenceClaimExtractor()
    claims, _ = await extractor.extract("")
    assert claims == []


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
async def test_sentence_extractor_handles_golden_cases(case) -> None:  # type: ignore[no-untyped-def]
    extractor = SentenceClaimExtractor()

    claims, _ = await extractor.extract(case.answer)

    assert len(claims) >= 1
    for claim in claims:
        assert case.answer[claim.span_start : claim.span_end] == claim.text

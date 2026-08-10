"""Golden fixtures: 5 realistic RAG triples (question, evidence chunks, answer with a
planted hallucination), reused across extractor/judge/policy tests so behavior stays
consistent end-to-end. See ``docs/SPEC.md`` §10.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    name: str
    question: str
    chunks: list[str]
    answer: str


CASES: list[GoldenCase] = [
    GoldenCase(
        name="paris_capital",
        question="What is the capital of France and what is its population?",
        chunks=[
            "Paris is the capital and most populous city of France.",
            "As of the last official estimate, Paris has a population of about "
            "2.1 million people within the city limits.",
        ],
        answer=(
            "Paris is the capital of France. It has a population of about "
            "5 million people within the city limits."
        ),
    ),
    GoldenCase(
        name="boiling_point",
        question="At what temperature does water boil at sea level?",
        chunks=[
            "At standard atmospheric pressure (sea level), water boils at 100 "
            "degrees Celsius, which is 212 degrees Fahrenheit."
        ],
        answer=(
            "Water boils at 100 degrees Celsius at sea level. It also boils faster "
            "if you add salt to it."
        ),
    ),
    GoldenCase(
        name="moon_landing",
        question="When did humans first land on the Moon, and who was aboard?",
        chunks=[
            "Apollo 11 was the spaceflight that first landed humans on the Moon, "
            "on July 20, 1969.",
            "The crew of Apollo 11 consisted of Neil Armstrong, Buzz Aldrin, and "
            "Michael Collins.",
        ],
        answer=(
            "Humans first landed on the Moon on July 20, 1969, during the Apollo "
            "11 mission. The mission commander, Neil Armstrong, was the first "
            "person to walk on the Moon, followed later that year by a second "
            "successful crewed landing."
        ),
    ),
    GoldenCase(
        name="refund_policy",
        question="What is the refund policy for unopened products?",
        chunks=[
            "Unopened products may be returned for a full refund within 30 days "
            "of purchase, provided the original receipt is included."
        ],
        answer=(
            "Unopened products can be refunded in full within 30 days of "
            "purchase. Opened products are also eligible for a full refund within "
            "90 days."
        ),
    ),
    GoldenCase(
        name="ibuprofen_dosage",
        question="What is the typical adult dose of ibuprofen for pain relief?",
        chunks=[
            "For adults, the typical over-the-counter dose of ibuprofen for pain "
            "relief is 200-400 mg every 4 to 6 hours, not exceeding 1200 mg per "
            "day without medical supervision."
        ],
        answer=(
            "The typical adult dose of ibuprofen for pain relief is 200-400 mg "
            "every 4 to 6 hours. It is safe to take up to 3200 mg per day."
        ),
    ),
]

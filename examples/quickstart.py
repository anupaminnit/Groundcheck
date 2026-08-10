"""Minimal end-to-end example: check a RAG answer against its evidence.

Requires provider credentials in the environment, e.g.:
    export GROUNDCHECK_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
"""

from groundcheck import Guard

ANSWER = (
    "Paris is the capital of France. It has a population of about 5 million people."
)
EVIDENCE = [
    "Paris is the capital and most populous city of France.",
    "As of the last official estimate, Paris has a population of about 2.1 million "
    "people within the city limits.",
]
QUESTION = "What is the capital of France, and what is its population?"


def main() -> None:
    guard = Guard(verifier="llm", policy="annotate")
    report = guard.check(ANSWER, EVIDENCE, question=QUESTION)

    print(f"grounded_score: {report.grounded_score:.2f}")
    print(f"action:         {report.action.value}")
    print()
    print(report.safe_answer)


if __name__ == "__main__":
    main()

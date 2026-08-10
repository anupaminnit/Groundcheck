"""Example: verify an answer that was itself generated via LiteLLM, using a
LiteLLM-backed GroundCheck guard.

Run: python examples/litellm_rag.py
Requires: pip install "groundcheck[litellm]", plus whichever provider credentials
litellm needs for the model string below (e.g. OPENAI_API_KEY for "gpt-4o-mini",
or a local Ollama server for "ollama/llama3").
"""

import asyncio

import litellm

from groundcheck import Guard, GuardConfig

EVIDENCE = [
    "Paris is the capital and most populous city of France.",
    "As of the last official estimate, Paris has a population of about 2.1 million "
    "people within the city limits.",
]
QUESTION = "What is the capital of France, and what is its population?"


async def generate_answer(model: str) -> str:
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": "Answer using only the given context."},
            {"role": "user", "content": f"Context:\n{EVIDENCE}\n\nQuestion: {QUESTION}"},
        ],
    )
    return response.choices[0].message.content or ""


async def main() -> None:
    model = "gpt-4o-mini"  # any litellm model string, e.g. "ollama/llama3"
    answer = await generate_answer(model)

    # provider="litellm" is a GuardConfig field, not a Guard(provider=...) kwarg —
    # that kwarg is reserved for an LLMProvider *instance*. Go through config= to
    # select a provider by name.
    guard = Guard(
        config=GuardConfig(verifier="llm", policy="annotate", provider="litellm", model=model)
    )
    report = await guard.acheck(answer, EVIDENCE, question=QUESTION)

    print(f"grounded_score: {report.grounded_score:.2f}")
    print(f"action:         {report.action.value}")
    print()
    print(report.safe_answer)


if __name__ == "__main__":
    asyncio.run(main())

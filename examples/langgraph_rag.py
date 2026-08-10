"""Example: verify a RAG answer inside a LangGraph graph.

Run: python examples/langgraph_rag.py
Requires: pip install langgraph, plus provider credentials in the environment
(e.g. GROUNDCHECK_PROVIDER=openai, OPENAI_API_KEY=sk-...).
"""

import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from groundcheck import Guard
from groundcheck.integrations.langgraph import make_guard_node


class RagState(TypedDict):
    question: str
    documents: list[str]
    answer: str
    groundcheck_report: Any


async def generate(state: RagState) -> dict[str, str]:
    # Stand-in for a real generation step.
    return {"answer": "Paris is the capital of France, with a population of about 5 million."}


def build_graph(guard: Guard) -> Any:
    graph = StateGraph(RagState)
    graph.add_node("generate", generate)
    graph.add_node(
        "verify", make_guard_node(guard, answer_key="answer", evidence_key="documents")
    )
    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


async def main() -> None:
    guard = Guard(verifier="llm", policy="annotate")
    app = build_graph(guard)

    result = await app.ainvoke(
        {
            "question": "What is the capital of France, and what is its population?",
            "documents": [
                "Paris is the capital and most populous city of France.",
                "As of the last official estimate, Paris has a population of about "
                "2.1 million people within the city limits.",
            ],
        }
    )

    print(result["groundcheck_report"])
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())

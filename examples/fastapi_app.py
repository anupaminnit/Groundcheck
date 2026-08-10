"""Example: wrap a FastAPI RAG endpoint with GroundCheck.

Run: uvicorn examples.fastapi_app:app --reload
Requires: pip install "groundcheck[fastapi]", plus provider credentials in the
environment (e.g. GROUNDCHECK_PROVIDER=openai, OPENAI_API_KEY=sk-...).
"""

from fastapi import FastAPI
from pydantic import BaseModel

from groundcheck import Guard, GuardReport
from groundcheck.integrations.fastapi import guarded

app = FastAPI()
guard = Guard(verifier="llm", policy="annotate")


class Query(BaseModel):
    question: str


class RagResponse(BaseModel):
    answer: str
    evidence: list[str]
    groundcheck: GuardReport | None = None


@app.post("/ask", response_model=RagResponse)
@guarded(guard, answer_field="answer", evidence_field="evidence")
async def ask(query: Query) -> RagResponse:
    # Stand-in for a real retrieval + generation step.
    evidence = ["Paris is the capital and most populous city of France."]
    answer = "Paris is the capital of France, with a population of about 5 million."
    return RagResponse(answer=answer, evidence=evidence)

# GroundCheck 🛡️

[![CI](https://github.com/anupaminnit/Groundcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/anupaminnit/Groundcheck/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A hallucination firewall for RAG pipelines.** GroundCheck sits between your RAG pipeline and your users, verifies every claim in a generated answer against the retrieved evidence, and blocks, redacts, or flags anything your documents don't actually support — before it reaches the user.

> Your retriever found the right chunks. Your LLM still made things up. GroundCheck catches it.

```
pip install groundcheck-ai
```

```python
from groundcheck import Guard

guard = Guard(verifier="llm", policy="annotate")

result = guard.check(
    question="What is the refund window for enterprise plans?",
    answer=rag_answer,          # str — what your LLM generated
    evidence=retrieved_chunks,  # list[str] or list[Evidence]
)

result.grounded_score   # 0.87
result.claims            # per-claim verdicts (SUPPORTED / UNSUPPORTED / CONTRADICTED / ...)
result.safe_answer      # answer transformed per your policy
```

## Why

Every production RAG system has the same failure mode: the answer *sounds* right, cites nothing wrong explicitly, and contains one confident sentence that appears nowhere in the retrieved context. Retrieval metrics won't catch it. Users will.

GroundCheck is a verification layer, not another RAG framework:

- **Drop-in** — one function call, or a callback/middleware for your existing stack
- **Framework-agnostic** — plain Python SDK, plus LangChain, LangGraph, and FastAPI integrations
- **Two verification backends** — an LLM judge (batched, one extra call per answer) or a fully local NLI cross-encoder (no data leaves your infra — built for enterprise/regulated environments)
- **Policy-driven** — you decide what happens to unsupported claims: log, annotate, redact, or block
- **Observable** — every check produces a structured report: per-claim verdicts, confidence, evidence spans, latency, token cost

## How it works

```
answer + retrieved chunks
        │
        ▼
┌─────────────────┐   Splits the answer into atomic, checkable claims.
│ Claim extractor │   Opinions, hedges, and meta-text are skipped.
└────────┬────────┘
         ▼
┌─────────────────┐   Narrows each claim to its top-k candidate chunks
│ Evidence matcher│   via embeddings — so the verifier reads less.
└────────┬────────┘
         ▼
┌─────────────────┐   Entailment check per claim:
│    Verifier     │   • llm    — batched judge call (Azure OpenAI / OpenAI / Anthropic)
│                 │   • local  — DeBERTa NLI cross-encoder, CPU-friendly, offline
│                 │   • hybrid — NLI screens, LLM escalates borderline claims
└────────┬────────┘
         ▼
┌─────────────────┐   SUPPORTED claims pass. For the rest, your policy runs:
│  Policy engine  │   log / annotate / redact / block (+ regenerate hook)
└────────┬────────┘
         ▼
safe answer + groundedness report
```

## Policies

| Policy     | What happens to unsupported claims                          | Use when                       |
|------------|-------------------------------------------------------------|--------------------------------|
| `log`      | Nothing changes; report is emitted                          | Shadow mode, measuring baseline |
| `annotate` | Inline markers: `⚠ [unverified]` appended to flagged spans  | Internal tools, analyst UIs     |
| `redact`   | Unsupported sentences removed, answer re-joined cleanly      | Customer-facing chatbots        |
| `block`    | Whole answer withheld below a score threshold; fallback text | High-stakes domains (legal, med)|

All thresholds are configurable. `block` accepts an optional `on_block` callback so you can trigger regeneration with the report injected into the retry prompt.

## Integrations

**LangChain** — one callback on your chain:

```python
from groundcheck.integrations.langchain import GroundCheckCallback
chain.invoke(inputs, config={"callbacks": [GroundCheckCallback(guard)]})
```

**LangGraph** — a verification node between `generate` and `END`:

```python
from groundcheck.integrations.langgraph import make_guard_node
graph.add_node("verify", make_guard_node(guard, answer_key="answer", evidence_key="documents"))
```

**FastAPI** — wrap the endpoint that returns RAG answers:

```python
from groundcheck.integrations.fastapi import guarded

@app.post("/ask")
@guarded(guard, answer_field="answer", evidence_field="sources")
async def ask(query: Query) -> RagResponse: ...
```

## Providers

GroundCheck's judge and extractor run on Azure OpenAI, OpenAI, or Anthropic natively — or on **any OpenAI-compatible endpoint** via `base_url`:

```python
from groundcheck import GuardConfig

# LiteLLM proxy, Ollama, vLLM, LM Studio — anything OpenAI-shaped.
# provider="openai" here is a GuardConfig field (a provider name), not the
# Guard(provider=...) kwarg — that one's reserved for an LLMProvider instance.
guard = Guard(config=GuardConfig(
    verifier="llm", provider="openai", base_url="http://localhost:4000", model="gpt-4o-mini",
))
```

Or use the native LiteLLM adapter for direct access to 100+ providers:

```
pip install "groundcheck-ai[litellm]"
```

```python
guard = Guard(config=GuardConfig(verifier="llm", provider="litellm", model="ollama/llama3"))
```

And to be clear on the other side of the pipe: GroundCheck verifies answers from **any** RAG stack. It only sees the final answer, the question, and the retrieved chunks — how you generated them (LiteLLM, LangChain, raw SDKs) is irrelevant.

## Local / air-gapped mode

```
pip install "groundcheck-ai[local]"
```

```python
guard = Guard(verifier="local")   # DeBERTa-v3 NLI, runs on CPU, no API calls
```

No tokens, no per-check cost, nothing leaves your network. Accuracy is lower than the LLM judge on long, compositional claims — `verifier="hybrid"` gets you most of the accuracy at a fraction of the cost.

## CLI

```bash
groundcheck check --answer answer.txt --evidence chunks/ --policy annotate --format json
```

Useful for CI: fail a pipeline if a golden-set answer drops below a groundedness threshold.

## Report schema

```json
{
  "grounded_score": 0.87,
  "action": "annotate",
  "claims": [
    {
      "text": "Enterprise plans have a 60-day refund window.",
      "verdict": "CONTRADICTED",
      "confidence": 0.94,
      "evidence_ids": ["chunk_03"],
      "rationale": "Evidence states the refund window is 30 days."
    }
  ],
  "latency_ms": 840,
  "verifier": "llm",
  "tokens": {"input": 1420, "output": 210}
}
```

## Roadmap

- [x] Core SDK: claim extraction, LLM judge (Azure OpenAI / OpenAI / Anthropic)
- [x] Policies (annotate / redact / block) + CLI + OpenAI-compatible `base_url`
- [x] Local NLI + hybrid verifier
- [x] LangChain / LangGraph / FastAPI integrations + native LiteLLM provider
- [ ] Eval harness against RAGTruth benchmark
- [ ] Streaming support (verify sentence-by-sentence as tokens arrive)
- [ ] Dashboard (score trends per endpoint/tenant)

## Contributing

Issues and PRs welcome. 

## License

MIT
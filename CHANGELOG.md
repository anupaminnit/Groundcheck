# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — Unreleased

Initial release.

### Added

- **Core SDK**: `Guard`, `GuardConfig`, `GuardReport`, `Verdict`, `Evidence` public
  API. `Guard.acheck()` (async) and `Guard.check()` (sync wrapper).
- **Claim extraction**: `LLMClaimExtractor` (one JSON-mode call, with offset
  recovery for repeated/unicode sentences) and `SentenceClaimExtractor`
  (deterministic, no LLM call; used in local mode and as the LLM extractor's
  fallback).
- **Evidence matching**: `EmbeddingMatcher` (provider embeddings, cosine
  similarity) with an automatic lexical fallback; `LexicalMatcher` for local mode.
- **Verifiers**: `LLMJudgeVerifier` (batched judge calls with JSON-repair retry),
  `NLIVerifier` (local DeBERTa cross-encoder, no LLM calls, lazy `torch`/
  `transformers` import), `HybridVerifier` (NLI first, escalates only
  low-confidence claims to the LLM judge).
- **Policies**: `log`, `annotate` (word-boundary-safe inline markers), `redact`
  (span removal + whitespace/connector cleanup, escalates to `block` above a
  60%-removed threshold), `block` (fallback message below `block_threshold`,
  with an optional `on_block` regeneration hook).
- **Providers**: `OpenAIProvider` (+ `base_url` for any OpenAI-compatible
  endpoint), `AzureOpenAIProvider`, `AnthropicProvider`, `LiteLLMProvider`
  (100+ providers via a verbatim model string, lazy `litellm` import).
- **CLI**: `groundcheck check ANSWER_FILE EVIDENCE_FILE` — JSON/pretty output,
  exits 1 below `--threshold` (CI-usable).
- **Integrations**: `GroundCheckCallback` (LangChain), `make_guard_node`
  (LangGraph), `guarded` (framework-agnostic endpoint decorator).
- **Extras**: `[local]` (NLI), `[langchain]`, `[fastapi]`, `[litellm]`.
- Examples: `quickstart.py`, `langgraph_rag.py`, `fastapi_app.py`, `litellm_rag.py`.

[0.1.0]: https://github.com/anupaminnit/Groundcheck/releases/tag/v0.1.0

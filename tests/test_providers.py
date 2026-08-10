"""Tests for the Phase 1→2 retrofit (OpenAIProvider + GuardConfig +
GROUNDCHECK_BASE_URL) and the Phase 4 LiteLLMProvider (lazy import + mocked
litellm.acompletion/aembedding — the real `litellm` package is never installed
here, only injected into sys.modules, matching how NLIVerifier's torch import is
tested without installing torch).
"""

from __future__ import annotations

import sys
import types

import pytest

from groundcheck.config import GuardConfig
from groundcheck.core.errors import ConfigError, VerifierError
from groundcheck.providers import build_provider
from groundcheck.providers.litellm import LiteLLMProvider
from groundcheck.providers.openai import OpenAIProvider


def test_openai_provider_constructs_client_with_custom_base_url() -> None:
    provider = OpenAIProvider(api_key="test-key", base_url="http://localhost:11434/v1")

    assert str(provider._client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_openai_provider_defaults_to_no_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROUNDCHECK_BASE_URL", raising=False)

    provider = OpenAIProvider(api_key="test-key")

    assert "openai.com" in str(provider._client.base_url)


def test_openai_provider_reads_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDCHECK_BASE_URL", "http://localhost:8000/v1")

    provider = OpenAIProvider(api_key="test-key")

    assert str(provider._client.base_url).rstrip("/") == "http://localhost:8000/v1"


def test_build_provider_passes_base_url_through_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = GuardConfig(provider="openai", base_url="http://localhost:11434/v1")

    provider = build_provider(config)

    assert isinstance(provider, OpenAIProvider)
    assert str(provider._client.base_url).rstrip("/") == "http://localhost:11434/v1"


# --- LiteLLMProvider (Phase 4) -----------------------------------------------


def test_litellm_provider_import_does_not_pull_in_litellm() -> None:
    assert "litellm" not in sys.modules


def test_litellm_provider_raises_config_error_without_extra() -> None:
    # Check actual installability, not sys.modules membership — litellm may be
    # installed but not yet imported by anything in this test run.
    try:
        import litellm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("litellm is installed; can't test the ImportError path here.")
    with pytest.raises(ConfigError, match=r"\[litellm\]"):
        LiteLLMProvider(model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_litellm_provider_complete_json_against_mocked_acompletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_acompletion(**kwargs: object) -> types.SimpleNamespace:
        calls.append(kwargs)
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        message = types.SimpleNamespace(content="fake completion")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.acompletion = fake_acompletion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    provider = LiteLLMProvider(model="ollama/llama3")
    content, tokens = await provider.complete_json("system", "user", timeout=10.0)

    assert content == "fake completion"
    assert tokens.input == 10
    assert tokens.output == 5
    assert calls[0]["model"] == "ollama/llama3"


@pytest.mark.asyncio
async def test_litellm_provider_wraps_errors_as_verifier_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_acompletion(**kwargs: object) -> None:
        raise RuntimeError("upstream boom")

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.acompletion = failing_acompletion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    provider = LiteLLMProvider(model="ollama/llama3")

    with pytest.raises(VerifierError, match="upstream boom"):
        await provider.complete_json("system", "user", timeout=10.0)


@pytest.mark.asyncio
async def test_litellm_provider_embed_requires_embed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = types.ModuleType("litellm")
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    provider = LiteLLMProvider(model="ollama/llama3")

    with pytest.raises(ConfigError, match="embed_model"):
        await provider.embed(["text"])


@pytest.mark.asyncio
async def test_litellm_provider_embed_against_mocked_aembedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aembedding(**kwargs: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.aembedding = fake_aembedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    provider = LiteLLMProvider(model="ollama/llama3", embed_model="ollama/nomic-embed-text")
    vectors = await provider.embed(["hello"])

    assert vectors == [[0.1, 0.2, 0.3]]


def test_build_provider_constructs_litellm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    config = GuardConfig(provider="litellm", model="ollama/llama3")

    provider = build_provider(config)

    assert isinstance(provider, LiteLLMProvider)


def test_build_provider_requires_model_for_litellm() -> None:
    config = GuardConfig(provider="litellm")

    with pytest.raises(ConfigError, match="model"):
        build_provider(config)

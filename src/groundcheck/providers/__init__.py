"""LLM provider backends: a common protocol plus Azure OpenAI, OpenAI, Anthropic,
and LiteLLM implementations."""

from __future__ import annotations

from groundcheck.config import GuardConfig
from groundcheck.core.errors import ConfigError
from groundcheck.providers.anthropic import AnthropicProvider
from groundcheck.providers.azure_openai import AzureOpenAIProvider
from groundcheck.providers.base import LLMProvider
from groundcheck.providers.litellm import LiteLLMProvider
from groundcheck.providers.openai import OpenAIProvider


def build_provider(config: GuardConfig) -> LLMProvider:
    """Construct the ``LLMProvider`` named by ``config.provider``.

    Args:
        config: Supplies ``provider``, ``model``, and ``base_url``.

    Returns:
        The constructed provider.

    Raises:
        ConfigError: No provider is configured, the named provider is missing
            required credentials, or ``provider="litellm"`` was given without
            ``model``.
    """
    if config.provider is None:
        raise ConfigError(
            "No provider configured. Set GROUNDCHECK_PROVIDER=azure|openai|anthropic|litellm, "
            "or pass provider=<LLMProvider instance> to Guard explicitly."
        )
    if config.provider == "azure":
        return AzureOpenAIProvider()
    if config.provider == "openai":
        kwargs: dict[str, str] = {"base_url": config.base_url} if config.base_url else {}
        if config.model:
            kwargs["model"] = config.model
        return OpenAIProvider(**kwargs)
    if config.provider == "anthropic":
        return AnthropicProvider(model=config.model) if config.model else AnthropicProvider()
    if config.provider == "litellm":
        if not config.model:
            raise ConfigError(
                "provider='litellm' requires GuardConfig.model, e.g. model='ollama/llama3'."
            )
        return LiteLLMProvider(model=config.model)
    raise ConfigError(f"Unknown provider: {config.provider!r}")


__all__ = [
    "LLMProvider",
    "AzureOpenAIProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "LiteLLMProvider",
    "build_provider",
]

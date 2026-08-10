"""LiteLLM provider backend: a native adapter over ``litellm.acompletion`` /
``litellm.aembedding``, for direct access to 100+ providers via a single verbatim
model string — e.g. ``"azure/gpt-4o"``, ``"ollama/llama3"``, ``"bedrock/claude-3-5-sonnet"``.

Lazily imports ``litellm`` inside ``__init__``. Requires ``pip install
groundcheck[litellm]``. See ``docs/SPEC.md`` §5.8.
"""

from __future__ import annotations

from typing import Any

from groundcheck.core.errors import ConfigError, VerifierError
from groundcheck.core.schemas import TokenUsage


class LiteLLMProvider:
    """LLMProvider backed by LiteLLM's universal ``acompletion``/``aembedding``."""

    def __init__(self, model: str, embed_model: str | None = None) -> None:
        """Initialize the provider.

        Args:
            model: LiteLLM model string, passed through verbatim, e.g.
                ``"azure/gpt-4o"`` or ``"ollama/llama3"``.
            embed_model: LiteLLM model string for embeddings. Required only if
                ``embed()`` is actually called.

        Raises:
            ConfigError: The ``[litellm]`` extra isn't installed.
        """
        try:
            import litellm
        except ImportError as exc:
            raise ConfigError(
                "LiteLLMProvider requires the [litellm] extra. "
                "Run: pip install groundcheck[litellm]"
            ) from exc

        self._litellm = litellm
        self._model = model
        self._embed_model = embed_model

    async def complete_json(
        self, system: str, user: str, timeout: float
    ) -> tuple[str, TokenUsage]:
        """See ``LLMProvider.complete_json``.

        Raises:
            VerifierError: The underlying ``litellm.acompletion`` call raised.
        """
        try:
            response = await self._litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=timeout,
            )
        except Exception as exc:
            raise VerifierError(str(exc)) from exc

        content = response.choices[0].message.content or ""
        return content, _token_usage(getattr(response, "usage", None))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """See ``LLMProvider.embed``.

        Raises:
            ConfigError: No ``embed_model`` was configured.
            VerifierError: The underlying ``litellm.aembedding`` call raised.
        """
        if not self._embed_model:
            raise ConfigError(
                "LiteLLMProvider.embed() requires embed_model, e.g. "
                "LiteLLMProvider(model=..., embed_model='text-embedding-3-small')."
            )
        try:
            response = await self._litellm.aembedding(model=self._embed_model, input=texts)
        except Exception as exc:
            raise VerifierError(str(exc)) from exc

        return [_extract_embedding(item) for item in response.data]


def _extract_embedding(item: Any) -> list[float]:
    if isinstance(item, dict):
        return item["embedding"]  # type: ignore[no-any-return]
    return item.embedding  # type: ignore[no-any-return]


def _token_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input=getattr(usage, "prompt_tokens", 0) or 0,
        output=getattr(usage, "completion_tokens", 0) or 0,
    )

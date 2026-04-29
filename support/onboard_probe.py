"""Connectivity probes for the onboarding flow.

After the user fills in chat/embedding backend fields we want to fail fast
on broken endpoints or wrong API keys *before* writing config to disk. The
probe is intentionally thin: it constructs the same provider the agent
will use and issues a minimal call. Anything richer belongs in real tests.
"""

from __future__ import annotations

from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.schema import Message
from localmelo.support import config as _config_mod
from localmelo.support.backends import get_backend


class ProbeError(RuntimeError):
    """Raised when a provider probe fails."""


async def probe_chat(cfg: _config_mod.Config) -> None:
    """Send a one-message ping through the configured chat backend."""
    backend = get_backend(cfg.chat_backend)
    provider: BaseLLMProvider = backend.build_chat_provider(cfg)
    try:
        response = await provider.chat(
            [Message(role="user", content="ping")],
            tools=None,
        )
    except Exception as exc:  # noqa: BLE001 — wrap as ProbeError for callers
        raise ProbeError(f"chat probe failed: {exc}") from exc
    finally:
        await provider.close()

    if response is None or not isinstance(response, Message):
        raise ProbeError("chat probe returned no message")


async def probe_embedding(cfg: _config_mod.Config) -> None:
    """Embed a single token through the configured embedding backend.

    Skipped silently when ``embedding_backend == 'none'``.
    """
    if cfg.embedding_backend == "none" or not cfg.embedding_backend:
        return

    backend = get_backend(cfg.embedding_backend)
    provider: BaseEmbeddingProvider | None = backend.build_embedding_provider(cfg)
    if provider is None:
        raise ProbeError(
            f"embedding probe failed: backend '{cfg.embedding_backend}' "
            "did not return a provider"
        )
    try:
        vectors = await provider.embed(["ping"])
    except Exception as exc:  # noqa: BLE001
        raise ProbeError(f"embedding probe failed: {exc}") from exc
    finally:
        await provider.close()

    if not vectors or not vectors[0]:
        raise ProbeError("embedding probe returned an empty vector")


async def probe_all(cfg: _config_mod.Config) -> None:
    """Run chat then embedding probes; first failure aborts."""
    await probe_chat(cfg)
    await probe_embedding(cfg)

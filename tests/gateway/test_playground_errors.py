"""Playground routes report model-server failures as 502.

Requires fastapi (gateway extra). Collection-safe without it.
"""

from __future__ import annotations

from typing import Any

pytest = __import__("pytest")
pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from localmelo.melo.contracts.providers import ProviderError  # noqa: E402
from localmelo.support.config import Config, LocalBackendConfig  # noqa: E402
from localmelo.support.gateway.webapp import mount  # noqa: E402


class _SeedingFailsPlayground:
    """Stands in for SmokePlayground, whose default store is under ~/.cache."""

    async def create_session(self, **kwargs: Any) -> dict[str, Any]:
        # What seeding a scenario raises when the embedding server fails.
        raise ProviderError(
            "POST /embeddings failed with HTTP 503: busy",
            status_code=503,
            retryable=True,
        )

    async def close_all(self) -> None:
        pass


# mount() registers its shutdown hook with FastAPI's deprecated on_event.
@pytest.mark.filterwarnings(r"ignore:\s*on_event is deprecated:DeprecationWarning")
def test_a_provider_error_while_creating_a_session_is_a_502() -> None:
    app = FastAPI()
    app.state.smoke_playground = _SeedingFailsPlayground()
    cfg = Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(chat_url="http://llm.test/v1", chat_model="m"),
    )
    mount(app, cfg, enable_playground=True)
    response = TestClient(app).post("/v1/playground/session", json={})
    assert response.status_code == 502
    assert response.json() == {"error": "POST /embeddings failed with HTTP 503: busy"}

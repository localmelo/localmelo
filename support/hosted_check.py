"""The Test button for a hosted service: its model list, then one short chat.

Two requests, in this order:

1. ``GET <address>/models`` with the key. It tells a wrong key (401), an account
   the service refused (402, 403) and an address nobody answers from a working
   service, and lists the models to choose from. A service with no list (404,
   405) is not a failure.
2. One short chat that offers one tool and asks the model to call it. It tells a
   model the service does not have (404), and whether the model can use tools,
   which melo needs.

The result is a code and the facts, never the key: every detail is redacted, and
the request carries the key in a header only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final

import httpx

from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.contracts.redaction import redact_secrets
from localmelo.melo.schema import Message, ToolDef
from localmelo.support import config, hosted, keychain
from localmelo.support.backends import discovery
from localmelo.support.backends.registry import get_backend

LIST_TIMEOUT_S: Final = 10.0
# A model that thinks first can take a while to answer even a short request.
CHAT_TIMEOUT_S: Final = 60.0
# The most models a result lists.
MAX_MODELS: Final = 200
# The results, in the order the console words them.
WORKS: Final = "works"
NO_KEY: Final = "no_key"
WRONG_KEY: Final = "wrong_key"
REFUSED: Final = "refused"
NO_MODEL: Final = "no_model"
BUSY: Final = "busy"
TIMEOUT: Final = "timeout"
UNREACHABLE: Final = "unreachable"
ERROR: Final = "error"

# The catalog key of the words for each failure, by result code.
WORDS: Final = {
    NO_KEY: "hosted.test.noKey",
    WRONG_KEY: "hosted.test.wrongKey",
    REFUSED: "hosted.test.refused",
    NO_MODEL: "hosted.test.noModel",
    BUSY: "hosted.test.busy",
    TIMEOUT: "hosted.test.timeout",
    UNREACHABLE: "hosted.test.unreachable",
    ERROR: "hosted.test.error",
}

_TOOL = ToolDef(
    "echo",
    "Say a word back.",
    {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
_ASK = "Call the echo tool with the text hello."


@dataclass(frozen=True)
class HostedCheck:
    """What the Test button found.

    *result* is one of the codes above or ``keychain_<kind>`` when the saved key
    could not be read. *status* is the HTTP status of a refusal, *models* the
    chat models the service lists, *tools* whether the model called the test
    tool (``None`` when no chat was sent) and *detail* a short, redacted note.
    """

    ok: bool
    result: str
    models: tuple[str, ...] = ()
    status: int | None = None
    detail: str = ""
    tools: bool | None = None


def _address(backend: str, section: config.CloudVendorConfig) -> str:
    """The address the section's requests go to."""
    if section.base_url:
        return section.base_url.rstrip("/")
    preset = hosted.preset_of(backend, "")
    return preset.base_url.rstrip("/") if preset is not None else ""


def failure(status: int | None, text: str) -> tuple[str, str]:
    """The result code and detail for an HTTP *status* (or none) with body *text*."""
    detail = redact_secrets(" ".join(text.split())[:200])[0]
    if status == 401:
        return WRONG_KEY, ""
    if status in (402, 403):
        return REFUSED, detail
    if status == 404:
        return NO_MODEL, detail
    if status == 429:
        return BUSY, detail
    if status is None:
        return (TIMEOUT if "timeout" in text.lower() else UNREACHABLE), detail
    return ERROR, detail


async def list_models(
    address: str, key: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[list[str], HostedCheck | None]:
    """The chat models at *address* and, when the request failed, why.

    A service with no list answers 404 or 405, which is no failure: the list
    is empty and the chat decides.
    """
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(
            timeout=LIST_TIMEOUT_S, transport=transport, follow_redirects=False
        ) as client:
            response = await client.get(f"{address}/models", headers=headers)
    except httpx.TimeoutException as exc:
        return [], HostedCheck(False, TIMEOUT, detail=type(exc).__name__)
    except httpx.RequestError as exc:
        return [], HostedCheck(False, UNREACHABLE, detail=type(exc).__name__)
    if response.status_code in (404, 405, 501):
        return [], None
    if response.status_code >= 300:  # a redirect is never followed
        result, detail = failure(response.status_code, response.text)
        return [], HostedCheck(
            False, result, status=response.status_code, detail=detail
        )
    try:
        body = response.json()
    except ValueError:
        return [], None
    names = (name.removeprefix("models/") for name in discovery.parse_model_ids(body))
    models = sorted(name for name in names if discovery.is_chat_model(name))
    return models[:MAX_MODELS], None


async def check_hosted(
    cfg: config.Config,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    provider: BaseLLMProvider | None = None,
) -> HostedCheck:
    """Test the hosted service *cfg*'s chat backend names.

    *transport* and *provider* are for tests; a real call builds the provider
    through the registry, as a run does.
    """
    backend = cfg.chat_backend
    section: config.CloudVendorConfig = getattr(cfg, backend)
    try:
        key = hosted.api_key(section)
    except keychain.KeychainError as exc:
        return HostedCheck(False, f"keychain_{exc.kind}")
    if not key:
        return HostedCheck(False, NO_KEY)
    models, failed = await list_models(
        _address(backend, section), key, transport=transport
    )
    if failed is not None:
        return failed
    model = section.chat_model
    if (
        models
        and model not in models
        and not any(m.endswith(f"/{model}") for m in models)
    ):
        return HostedCheck(False, NO_MODEL, models=tuple(models))
    owned = provider is None
    if provider is None:
        provider = get_backend(backend).build_chat_provider(cfg)
    try:
        async with asyncio.timeout(CHAT_TIMEOUT_S):
            reply = await provider.chat([Message("user", _ASK)], tools=[_TOOL])
    except TimeoutError:
        return HostedCheck(False, TIMEOUT, models=tuple(models))
    except ProviderError as exc:
        result, detail = failure(exc.status_code, str(exc))
        return HostedCheck(
            False, result, models=tuple(models), status=exc.status_code, detail=detail
        )
    finally:
        if owned:
            await provider.close()
    return HostedCheck(True, WORKS, models=tuple(models), tools=bool(reply.tool_calls))

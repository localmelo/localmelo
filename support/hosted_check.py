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
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx

from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.contracts.redaction import REDACTED, redact_secrets
from localmelo.melo.schema import Message, ToolDef
from localmelo.support import config, hosted, keychain
from localmelo.support.backends import discovery
from localmelo.support.backends.registry import get_backend

LIST_TIMEOUT_S: Final = 10.0
# A model that thinks first can take a while to answer even a short request.
CHAT_TIMEOUT_S: Final = 60.0
# The most models a result lists.
MAX_MODELS: Final = 200
# The longest model name that is kept.
MAX_NAME: Final = 200
# The most of a listing's answer that is read; a longer one is a failure, not a list.
MAX_BODY: Final = 1024 * 1024
# How long a listing is reused, so a menu opened again does not ask again.
LISTING_TTL_S: Final = 60.0
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


def failure(status: int | None, text: str, key: str = "") -> tuple[str, str]:
    """The result code and detail for an HTTP *status* (or none) with body *text*.

    The *key* the request carried is cut out of *text* first, whatever its shape; the
    patterns :func:`redact_secrets` knows come after.
    """
    if key:
        text = text.replace(key, REDACTED)
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
    # An uncompressed answer: nothing to inflate past the limit below.
    headers = {"Accept-Encoding": "identity"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        # The whole request, not each phase, is bounded; so is the answer that is read.
        async with (
            asyncio.timeout(LIST_TIMEOUT_S),
            httpx.AsyncClient(
                timeout=LIST_TIMEOUT_S, transport=transport, follow_redirects=False
            ) as client,
            client.stream("GET", f"{address}/models", headers=headers) as response,
        ):
            status = response.status_code
            if status in (404, 405, 501):  # no list: the chat decides
                return [], None
            # A compressed answer would be inflated before the size is known: refused unread.
            if response.headers.get("content-encoding", "").strip().lower() not in (
                "",
                "identity",
            ):
                return [], HostedCheck(False, ERROR, status=status, detail="Encoded")
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw += chunk
                if len(raw) > MAX_BODY:
                    return [], HostedCheck(
                        False, ERROR, status=status, detail="TooLarge"
                    )
    except TimeoutError:
        return [], HostedCheck(False, TIMEOUT, detail="TimeoutError")
    except httpx.TimeoutException as exc:
        return [], HostedCheck(False, TIMEOUT, detail=type(exc).__name__)
    except httpx.RequestError as exc:
        return [], HostedCheck(False, UNREACHABLE, detail=type(exc).__name__)
    if status >= 300:  # a redirect is never followed
        result, detail = failure(status, raw.decode("utf-8", "replace"), key)
        return [], HostedCheck(False, result, status=status, detail=detail)
    try:
        body = json.loads(raw)
    except ValueError:
        return [], None
    names = (name.removeprefix("models/") for name in discovery.parse_model_ids(body))
    # A name is the service's text: one printable line of a sane length, shown as text only.
    models = sorted(
        name
        for name in names
        if name.isprintable()
        and len(name) <= MAX_NAME
        and discovery.is_chat_model(name)
    )
    return models[:MAX_MODELS], None


def lacks(models: list[str] | tuple[str, ...], model: str) -> bool:
    """Whether the listing *models* is known and does not hold *model*."""
    return (
        bool(models)
        and model not in models
        and not any(name.endswith(f"/{model}") for name in models)
    )


class Listings:
    """The listings of the services, reused for a short while, one in flight per service.

    A listing is asked at an address with a key; its answer is kept in memory for
    :data:`LISTING_TTL_S` seconds (never on disk). A transient failure (a timeout, no
    answer) is not kept. The key is part of the entry as a digest, never as itself,
    so a new key asks again. Callers that ask while a listing is in flight wait for it.
    """

    def __init__(
        self,
        ttl_s: float = LISTING_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._done: dict[
            tuple[str, str], tuple[float, tuple[list[str], HostedCheck | None]]
        ] = {}
        self._busy: dict[
            tuple[str, str], asyncio.Future[tuple[list[str], HostedCheck | None]]
        ] = {}

    def clear(self) -> None:
        self._done.clear()

    async def get(
        self,
        address: str,
        key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> tuple[list[str], HostedCheck | None]:
        """:func:`list_models` for *address* with *key*, reused or shared as above."""
        entry = (address, hashlib.sha256(key.encode()).hexdigest()[:16])
        hit = self._done.get(entry)
        if hit is not None and self._clock() - hit[0] < self._ttl_s:
            return hit[1]
        waiting = self._busy.get(entry)
        if waiting is None:
            waiting = asyncio.ensure_future(
                list_models(address, key, transport=transport)
            )
            self._busy[entry] = waiting
            waiting.add_done_callback(lambda done: self._finish(entry, done))
        return await asyncio.shield(waiting)

    def _finish(
        self,
        entry: tuple[str, str],
        done: asyncio.Future[tuple[list[str], HostedCheck | None]],
    ) -> None:
        self._busy.pop(entry, None)
        if done.cancelled() or done.exception() is not None:
            return
        answer = done.result()
        if answer[1] is None or answer[1].result not in (TIMEOUT, UNREACHABLE):
            self._done[entry] = (self._clock(), answer)


LISTINGS: Final = Listings()


@dataclass(frozen=True)
class Listing:
    """What a service's model list gave: whether a key exists, the models, and why not.

    *models* is empty for a service with no list (then the chat decides) and when the
    listing failed; *check* then holds the failure (its *result* is a code of this
    module or ``keychain_<kind>``).
    """

    key: bool
    models: tuple[str, ...] = ()
    check: HostedCheck | None = None


async def service_models(
    preset: hosted.Preset,
    address: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    listings: Listings | None = None,
) -> Listing:
    """The models *preset* lists at *address*, once a key for it exists.

    The key is the preset's (:func:`hosted.preset_key`): it goes to *address* only
    when that is the address the key belongs to; any other is refused before a
    request is made. One ``GET <address>/models``, never a chat.
    """
    state = hosted.preset_key_state(preset, address)
    if state == hosted.ADDRESS:
        return Listing(False, check=HostedCheck(False, f"keychain_{hosted.ADDRESS}"))
    if state not in (hosted.ENV, hosted.KEYCHAIN):
        return Listing(False)
    try:
        credential = hosted.preset_key(preset, address)
    except keychain.KeychainError as exc:
        return Listing(True, check=HostedCheck(False, f"keychain_{exc.kind}"))
    if not credential.key:
        return Listing(False)
    models, failed = await (listings or LISTINGS).get(
        credential.address, credential.key, transport=transport
    )
    return Listing(True, tuple(models), failed)


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
        credential = hosted.api_key(backend, section)
    except keychain.KeychainError as exc:
        return HostedCheck(False, f"keychain_{exc.kind}")
    if not credential.key:
        return HostedCheck(False, NO_KEY)
    models, failed = await list_models(
        credential.address, credential.key, transport=transport
    )
    if failed is not None:
        return failed
    model = section.chat_model
    if lacks(models, model):
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
        result, detail = failure(exc.status_code, str(exc), credential.key)
        return HostedCheck(
            False, result, models=tuple(models), status=exc.status_code, detail=detail
        )
    finally:
        if owned:
            await provider.close()
    return HostedCheck(True, WORKS, models=tuple(models), tools=bool(reply.tool_calls))

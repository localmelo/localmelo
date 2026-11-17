"""The one list of chat models that can be used now, and the switch to one of them.

A hosted service is in the list once a key for it exists (its variable or its
keychain item); it lists its own models (:func:`hosted_check.service_models`). The
local server is the one in use, or else the one used last (``[services.local]``). With
no hosted key there is no list: the page, the terminal and the requests stay as they
were for a local server. A service whose listing fails is still a group, with the
reason, so it never hides the others.

A switch changes the config in memory (:func:`apply`); the caller saves it and tells
the chats, as a settings change does, so a turn that runs finishes on the model it
started with and the next message uses the new one. The embedding choice is not read
or written here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import httpx

from localmelo.support import hosted, hosted_check
from localmelo.support.backends import discovery, model_offer
from localmelo.support.backends.openai_family import LOCAL_BACKENDS
from localmelo.support.backends.registry import get_backend, in_use

if TYPE_CHECKING:
    from localmelo.support.config import Config

LOCAL: Final = hosted.LOCAL
MAX_MODEL: Final = 200


class ChoiceError(Exception):
    """A switch that cannot be made: *code* is ``unknown_group``, ``no_key``,
    ``address``, ``bad_model``, ``no_model`` or ``keychain_<kind>``."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Group:
    """A server or service in the list, with the models to pick from.

    *current* is the model in use when this group is the one in use, else "".
    *serves* marks a local server that serves its own model (one row to go back to
    it). *check* is why a hosted listing failed, *problem* why the local server's did.
    """

    id: str
    label: str
    hosted: bool
    models: tuple[str, ...] = ()
    current: str = ""
    serves: bool = False
    check: hosted_check.HostedCheck | None = None
    problem: str = ""


def in_use_group(cfg: Config) -> str:
    """The id of the group the chat is on: a hosted preset's, else ``local``."""
    preset = hosted.hosted_preset(cfg)
    return preset.id if preset is not None else LOCAL


def local_backend(cfg: Config, services: dict[str, dict[str, str]]) -> str:
    """The local server of the list: the one in use, else the one used last; "" when none."""
    if cfg.chat_backend in LOCAL_BACKENDS:
        return cfg.chat_backend
    last = services.get(LOCAL, {}).get("backend", "")
    return last if last in LOCAL_BACKENDS else ""


def address_for(preset: hosted.Preset, services: dict[str, dict[str, str]]) -> str:
    """The address of *preset*: its own, or for "Other" the one saved with it."""
    if preset.free_address:
        return services.get(preset.id, {}).get("base_url", "")
    return preset.base_url


def keyed(cfg: Config) -> list[tuple[hosted.Preset, str]]:
    """The services that have a key now, with their addresses, in the presets' order."""
    services = hosted.last_models(cfg)
    found = []
    for preset in hosted.PRESETS:
        address = address_for(preset, services)
        if address and hosted.preset_key_state(preset, address) in (
            hosted.ENV,
            hosted.KEYCHAIN,
        ):
            found.append((preset, address))
    return found


def valid_model(model: str) -> bool:
    return bool(model) and len(model) <= MAX_MODEL and model.isprintable()


async def _local_group(cfg: Config, backend: str, *, listing: bool) -> Group:
    registry = get_backend(backend)
    label = in_use(cfg, backend)[0]
    now = cfg.chat_backend == backend
    section = getattr(cfg, backend)
    if not registry.model_required:
        return Group(LOCAL, label, False, serves=True, current=label if now else "")
    if not listing:
        return Group(LOCAL, label, False, current=section.chat_model if now else "")
    try:
        listed = await discovery.list_models(
            backend, section.chat_url, timeout_s=discovery.DISCOVERY_TIMEOUT_S
        )
    except discovery.DiscoveryError as exc:
        return Group(LOCAL, label, False, problem=str(exc))
    models = tuple(model_offer.offer(listed).chat or listed)
    return Group(LOCAL, label, False, models, section.chat_model if now else "")


async def _hosted_group(
    cfg: Config,
    preset: hosted.Preset,
    address: str,
    transport: httpx.AsyncBaseTransport | None,
    *,
    listing: bool,
) -> Group:
    active = hosted.hosted_preset(cfg) is preset
    current = getattr(cfg, preset.backend).chat_model if active else ""
    if not listing:
        return Group(preset.id, preset.label, True, current=current)
    found = await hosted_check.service_models(preset, address, transport=transport)
    return Group(
        preset.id, preset.label, True, found.models, current, check=found.check
    )


async def collect(
    cfg: Config,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    listing: bool = True,
) -> list[Group]:
    """The groups of the list; empty while no hosted service has a key.

    With *listing* false no server is asked for its models: the groups are only named,
    which is what the page needs to know there is a list.
    """
    services = hosted.last_models(cfg)
    have = await asyncio.to_thread(keyed, cfg)  # asks the keychain: not on the loop
    if not have:
        return []
    jobs = [
        _hosted_group(cfg, preset, address, transport, listing=listing)
        for preset, address in have
    ]
    backend = local_backend(cfg, services)
    # The server melo_serve serves is listed by its own list; the others are listed here.
    if backend and not (backend == "melo_serve" and cfg.chat_backend == backend):
        jobs.insert(0, _local_group(cfg, backend, listing=listing))
    return list(await asyncio.gather(*jobs))


async def apply(
    cfg: Config,
    group: str,
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Put the chat of *cfg* on *model* of *group*; raises :class:`ChoiceError`.

    A hosted service must have a key and, when it lists its models, hold *model*; no
    chat is sent and no key is typed again. Only the chat backend and its section
    change: the embedding choice and every other service's key and model stay.
    """
    services = hosted.last_models(cfg)
    if group == LOCAL:
        backend = local_backend(cfg, services)
        if not backend:
            raise ChoiceError("unknown_group")
        if get_backend(backend).model_required and not valid_model(model):
            raise ChoiceError("bad_model")
        hosted.use_local(cfg, backend, model)
        return
    preset = hosted.get(group)
    if preset is None:
        raise ChoiceError("unknown_group")
    address = address_for(preset, services)
    state = hosted.preset_key_state(preset, address) if address else hosted.NONE
    if state == hosted.ADDRESS:
        raise ChoiceError(hosted.ADDRESS)
    if state not in (hosted.ENV, hosted.KEYCHAIN):
        raise ChoiceError("no_key")
    if not valid_model(model):
        raise ChoiceError("bad_model")
    found = await hosted_check.service_models(preset, address, transport=transport)
    if found.check is not None and found.check.result.startswith("keychain_"):
        raise ChoiceError(found.check.result)
    if hosted_check.lacks(found.models, model):
        raise ChoiceError("no_model")
    hosted.use_service(cfg, preset, model, address=address)

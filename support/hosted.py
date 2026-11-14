"""Hosted services: the presets Settings and the setup list offer, and their keys.

A preset is a row of product data, not a new backend. Each one names the config
section it fills (``backend``: ``openai``, ``anthropic``, ``gemini`` or
``nvidia``), the address of the service, the models worth suggesting, the
environment variable that held the key before the keychain, the request
fields that turn the model's thinking off, and which models read images. A service that speaks the OpenAI
format (DeepSeek, and the free-form "Other") uses the ``openai`` backend with its
own address, so the section's address tells the presets apart.

The key is looked up in this order: the environment variable the section names
(``api_key_env``), when it is set; then the keychain item the section refers to
(``api_key = "keychain:localmelo.<name>"``, see :mod:`localmelo.support.keychain`).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

from localmelo.support import keychain
from localmelo.support.words import t_in

if TYPE_CHECKING:
    from localmelo.support.config import CloudVendorConfig, Config

THINKING = ("", "on", "off")
OTHER_ID: Final = "other"
ENV_NAME_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

# DeepSeek's switch: a request with this field makes the model answer at once.
_DEEPSEEK_THINKING_OFF: Final[Mapping[str, Any]] = {"thinking": {"type": "disabled"}}


@dataclass(frozen=True)
class Preset:
    """One hosted service, as Settings and the setup list offer it.

    *backend* is the config section it fills and *base_url* its address ("" for
    the free-form "Other", whose address is typed). A preset whose id is its
    backend uses the backend's built-in address, which the config leaves empty.
    *thinking_off* holds the request fields that turn thinking off (empty: the
    service has no such switch), and *off_models* the models that start with
    thinking off. *image_models* and *text_models* are lowercase name prefixes of
    the models known to read images and known not to; a service's model listing
    does not say, so ``[agent] tool_images = "auto"`` follows these.
    """

    id: str
    label: str
    backend: str
    base_url: str
    models: tuple[str, ...]
    key_env: str
    thinking_off: Mapping[str, Any] = field(default_factory=dict)
    off_models: tuple[str, ...] = ()
    image_models: tuple[str, ...] = ()
    text_models: tuple[str, ...] = ()

    @property
    def service(self) -> str:
        """The keychain service of this preset's key."""
        return keychain.service_for(self.id)

    @property
    def stored_url(self) -> str:
        """The address the config stores: empty for a backend's built-in one."""
        return "" if self.id == self.backend else self.base_url

    @property
    def free_address(self) -> bool:
        """Whether the person types the address."""
        return self.id == OTHER_ID


PRESETS: Final[tuple[Preset, ...]] = (
    Preset(
        "openai",
        "OpenAI",
        "openai",
        "https://api.openai.com/v1",
        ("gpt-4o", "gpt-4o-mini"),
        "OPENAI_API_KEY",
        image_models=("gpt-4o", "gpt-4.1", "gpt-5"),
    ),
    Preset(
        "gemini",
        "Google Gemini",
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        ("gemini-2.0-flash",),
        "GEMINI_API_KEY",
        image_models=("gemini-",),
    ),
    Preset(
        "anthropic",
        "Anthropic",
        "anthropic",
        "https://api.anthropic.com/v1",
        ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"),
        "ANTHROPIC_API_KEY",
        image_models=(
            "claude-3",
            "claude-sonnet",
            "claude-opus",
            "claude-haiku",
            "claude-fable",
        ),
    ),
    Preset(
        "nvidia",
        "NVIDIA",
        "nvidia",
        "https://integrate.api.nvidia.com/v1",
        ("nvidia/llama-3.1-nemotron-70b-instruct",),
        "NVIDIA_API_KEY",
        text_models=("nvidia/llama-3.1-nemotron",),
    ),
    Preset(
        "deepseek",
        "DeepSeek",
        "openai",
        "https://api.deepseek.com/v1",
        ("deepseek-flash", "deepseek-v4-pro"),
        "DEEPSEEK_API_KEY",
        thinking_off=_DEEPSEEK_THINKING_OFF,
        off_models=("deepseek-flash",),
        image_models=("deepseek-flash",),
        text_models=("deepseek-v4-pro",),
    ),
    Preset(OTHER_ID, "Other (OpenAI format)", "openai", "", (), ""),
)
BY_ID: Final = {preset.id: preset for preset in PRESETS}


def get(preset_id: str) -> Preset | None:
    """The preset *preset_id* names, if there is one."""
    return BY_ID.get(preset_id)


def _same_address(left: str, right: str) -> bool:
    return left.strip().rstrip("/").lower() == right.strip().rstrip("/").lower()


def preset_of(backend: str, base_url: str) -> Preset | None:
    """The preset a ``[<backend>]`` section with *base_url* is.

    An empty address is the backend's own service. A known address is its preset's.
    Any other address on ``openai`` is "Other"; on a vendor's section it is still
    that vendor (a proxy), whose address the section keeps.
    """
    if not base_url.strip():
        return BY_ID.get(backend)
    for preset in PRESETS:
        if preset.backend == backend and _same_address(preset.base_url, base_url):
            return preset
    return BY_ID[OTHER_ID] if backend == "openai" else BY_ID.get(backend)


def preset_for(backend: str, section: CloudVendorConfig) -> Preset | None:
    """The preset of the configured section *backend*."""
    return preset_of(backend, section.base_url)


def label_for(backend: str, section: CloudVendorConfig) -> str:
    """What a banner calls the service of *section*: ``DeepSeek``, ``OpenAI`` …

    A free-form address is called by its host.
    """
    preset = preset_for(backend, section)
    if preset is None:
        return backend
    if preset.free_address:
        return urlparse(section.base_url).hostname or preset.label
    return preset.label


def thinking_off(preset: Preset | None, section: CloudVendorConfig) -> bool:
    """Whether the section's requests turn the model's thinking off.

    ``thinking = "off"`` or ``"on"`` decides; empty follows the preset: off for
    the models that start off. A service with no switch never is.
    """
    if preset is None or not preset.thinking_off:
        return False
    if section.thinking == "off":
        return True
    if section.thinking == "on":
        return False
    return section.chat_model in preset.off_models


def request_body(backend: str, section: CloudVendorConfig) -> dict[str, Any]:
    """The extra request fields of *section*: its preset's thinking switch, then ``extra_body``.

    A field the section's own ``extra_body`` sets wins over the switch.
    """
    preset = preset_for(backend, section)
    body: dict[str, Any] = {}
    if preset is not None and thinking_off(preset, section):
        body.update(preset.thinking_off)
    body.update(section.extra_body)
    return body


def retry_body(backend: str, section: CloudVendorConfig) -> Mapping[str, Any]:
    """The fields that turn thinking off for one retry; empty when it cannot.

    Empty when the service has no switch, thinking is already off, or the section's
    own ``extra_body`` sets the field.
    """
    preset = preset_for(backend, section)
    if preset is None or not preset.thinking_off or thinking_off(preset, section):
        return {}
    if any(name in section.extra_body for name in preset.thinking_off):
        return {}
    return dict(preset.thinking_off)


def reads_images(backend: str, section: CloudVendorConfig) -> bool | None:
    """Whether the section's model reads images, as its preset declares.

    ``True`` for a model the preset names as reading images, ``False`` for one it
    names as text only, ``None`` for any other (unknown, so ``tool_images = "auto"``
    sends no pixels).
    """
    preset = preset_for(backend, section)
    if preset is None:
        return None
    model = section.chat_model.strip().lower().removeprefix("models/")
    if any(model.startswith(prefix) for prefix in preset.image_models):
        return True
    if any(model.startswith(prefix) for prefix in preset.text_models):
        return False
    return None


def image_args(backend: str, section: CloudVendorConfig) -> dict[str, Any]:
    """The provider arguments that answer ``takes_images`` for the section.

    A service the presets know is not asked: its host has no oMLX or Ollama route,
    so the preset's answer stands. The free-form address may be one of those behind
    a proxy, so it is still asked.
    """
    preset = preset_for(backend, section)
    return {
        "reads_images": reads_images(backend, section),
        "probe_images": preset is None or preset.free_address,
    }


def presets_json() -> list[dict[str, Any]]:
    """The presets as the Settings page reads them."""
    return [
        {
            "id": preset.id,
            "label": preset.label,
            "backend": preset.backend,
            "base_url": preset.base_url,
            "stored_url": preset.stored_url,
            "models": list(preset.models),
            "key_env": preset.key_env,
            # "Other" has a key per address: the key routes say which one.
            "key_ref": "" if preset.free_address else keychain.ref_for(preset.service),
            "free_address": preset.free_address,
            "thinking_switch": bool(preset.thinking_off),
            "thinking_off_models": list(preset.off_models),
        }
        for preset in PRESETS
    ]


# -- The key -----------------------------------------------------------------

ENV = "env"
KEYCHAIN = "keychain"
NONE = "none"
# The key cannot go to this address: it belongs to another service or address.
ADDRESS = keychain.ADDRESS
# What a service remembers beside the key (see :func:`last_models`).
LOCAL: Final = "local"
_MAX_NAME: Final = 200


def address_of(backend: str, section: CloudVendorConfig) -> str:
    """The address *section*'s requests go to: its own, else its backend's built-in one."""
    own = section.base_url.strip().rstrip("/")
    if own:
        return own
    preset = BY_ID.get(backend)
    return preset.base_url.rstrip("/") if preset is not None else ""


def service_for(preset: Preset, address: str) -> str:
    """The keychain item that holds *preset*'s key for the service at *address*.

    A preset with a built-in address has one item. "Other" has one per address: the
    item's name holds a digest of the address, so a key saved for one address is
    never found for another.
    """
    if not preset.free_address:
        return preset.service
    digest = hashlib.sha256(address.strip().rstrip("/").lower().encode()).hexdigest()
    return keychain.service_for(f"{preset.id}-{digest[:12]}")


def address_allowed(preset: Preset, address: str) -> bool:
    """Whether *preset*'s key may be sent to *address*.

    A built-in address is the only one its preset's key goes to. For "Other" any
    address is allowed here, because the key's item is named by it
    (:func:`service_for`): another address finds no key.
    """
    if preset.free_address:
        return bool(address.strip())
    return _same_address(address, preset.base_url)


def _env_refused(section: CloudVendorConfig, address: str) -> bool:
    """Whether the variable *section* names is a service's own (``OPENAI_API_KEY``) and *address* is not that service's.

    A variable of a name of the person's own (``MY_GATEWAY_KEY``) goes where the section sends:
    the person chose that pairing. A service's own variable goes only to that service.
    """
    owner = next(
        (p for p in PRESETS if p.key_env and p.key_env == section.api_key_env), None
    )
    return owner is not None and not address_allowed(owner, address)


def key_state(backend: str, section: CloudVendorConfig) -> str:
    """Where the section's key comes from now.

    ``env`` (the named variable is set), ``keychain`` (the item is saved),
    ``missing`` (the item the config names is not there), ``locked``,
    ``unavailable`` (no keychain on this computer), ``address`` (the key belongs to
    another address than the section's) or ``none`` (no key named).
    """
    preset = preset_for(backend, section)
    address = address_of(backend, section)
    if section.api_key_env and os.environ.get(section.api_key_env):
        return ADDRESS if _env_refused(section, address) else ENV
    try:
        service = keychain.parse_ref(section.api_key)
    except ValueError:
        return NONE
    if service is None:
        return NONE
    if (
        preset is None
        or service != service_for(preset, address)
        or not address_allowed(preset, address)
    ):
        return ADDRESS
    answer = keychain.status(service)
    return KEYCHAIN if answer == keychain.SAVED else answer


def api_key(backend: str, section: CloudVendorConfig) -> str:
    """The section's API key, or "" when none is named.

    The environment variable wins when it is set. A keychain item that is missing
    or cannot be read raises :class:`localmelo.support.keychain.KeychainError`
    with the fix. So does a key that belongs to another address than the one the
    section sends to (kind ``address``): a stored key goes only to its own service,
    and this is the one place a section's key is read.
    """
    preset = preset_for(backend, section)
    address = address_of(backend, section)
    if section.api_key_env:
        value = os.environ.get(section.api_key_env, "")
        if value:
            if _env_refused(section, address):
                raise keychain.KeychainError(ADDRESS)
            return value
    service = keychain.parse_ref(section.api_key)
    if service is None:
        return ""
    if (
        preset is None
        or service != service_for(preset, address)
        or not address_allowed(preset, address)
    ):
        raise keychain.KeychainError(ADDRESS)
    return keychain.read(service)


def preset_key_state(preset: Preset, address: str) -> str:
    """Whether the person holds a key for *preset* at *address*, with no secret read.

    ``env`` (the preset's own variable is set), ``keychain`` (its item is saved),
    ``address`` (the preset's key does not go to *address*), else what the keychain
    says (``missing``, ``locked``, ``unavailable``).
    """
    if not address_allowed(preset, address):
        return ADDRESS
    if preset.key_env and os.environ.get(preset.key_env):
        return ENV
    answer = keychain.status(service_for(preset, address))
    return KEYCHAIN if answer == keychain.SAVED else answer


def preset_key(preset: Preset, address: str) -> str:
    """The key the person holds for *preset*, to send to *address* and nowhere else.

    The preset's own variable, else its keychain item; "" when there is none.
    Raises :class:`localmelo.support.keychain.KeychainError` (kind ``address``)
    before anything is read when *address* is not the one the key belongs to, and
    for a keychain that is locked or fails.
    """
    if not address_allowed(preset, address):
        raise keychain.KeychainError(ADDRESS)
    if preset.key_env:
        value = os.environ.get(preset.key_env, "")
        if value:
            return value
    try:
        return keychain.read(service_for(preset, address))
    except keychain.KeychainError as exc:
        if exc.kind in (keychain.MISSING, keychain.UNAVAILABLE):
            return ""
        raise


# -- What each service remembers ---------------------------------------------


def _text(value: Any, limit: int) -> str:
    """*value* when it is one printable line of at most *limit* characters, else ""."""
    ok = isinstance(value, str) and value.isprintable() and len(value) <= limit
    return value if ok else ""


def clean_services(raw: Any) -> dict[str, dict[str, str]]:
    """The ``[services.<id>]`` tables of a config file, as far as they are well formed.

    An id is a preset's (``chat_model``, and ``base_url`` for "Other") or ``local``
    (``backend``); anything else is dropped.
    """
    out: dict[str, dict[str, str]] = {}
    if not isinstance(raw, Mapping):
        return out
    for ident, entry in raw.items():
        if not isinstance(entry, Mapping):
            continue
        if ident == LOCAL:
            fields = {"backend": _text(entry.get("backend"), _MAX_NAME)}
        elif ident in BY_ID:
            fields = {"chat_model": _text(entry.get("chat_model"), _MAX_NAME)}
            if BY_ID[ident].free_address:
                fields["base_url"] = _text(entry.get("base_url"), 2000)
        else:
            continue
        kept = {name: value for name, value in fields.items() if value}
        if kept:
            out[str(ident)] = kept
    return out


def hosted_preset(cfg: Config) -> Preset | None:
    """The preset the chat backend of *cfg* is on, when it is a hosted one."""
    root = BY_ID.get(cfg.chat_backend)
    if root is None or root.id != root.backend:
        return None
    return preset_for(root.backend, getattr(cfg, root.backend))


def last_models(cfg: Config) -> dict[str, dict[str, str]]:
    """The last model of each service, and the last local server, by id.

    What the config file kept (``cfg.services``), with the service in use read from
    its section: several services share one section, so the section holds only the
    one in use, and a save writes this whole map.
    """
    out = clean_services(cfg.services)
    preset = hosted_preset(cfg)
    if preset is not None:
        section = getattr(cfg, preset.backend)
        if section.chat_model:
            entry = {"chat_model": section.chat_model}
            if preset.free_address:
                entry["base_url"] = section.base_url.strip()
            out[preset.id] = entry
    elif cfg.chat_backend:
        out[LOCAL] = {"backend": cfg.chat_backend}
    return out


def stored_services(cfg: Config) -> dict[str, dict[str, str]]:
    """What a config file keeps of :func:`last_models`: nothing until a hosted service is involved.

    A setup that has only ever used a local server writes no ``[services.*]`` table, so
    its file is the one it was.
    """
    found = last_models(cfg)
    return found if any(ident != LOCAL for ident in found) else {}


def use_service(cfg: Config, preset: Preset, model: str, *, address: str = "") -> None:
    """Put the chat of *cfg* on *model* of *preset*; nothing is written to disk.

    A service that is already in use keeps its key and thinking choice. Another
    starts from its own: the key its preset holds (its variable when set, else its
    keychain item), no thinking choice. A service with a section of its own keeps a
    key it already has. The other services' keys and models are not touched: they
    are in the keychain and in ``services``.
    """
    cfg.services = last_models(cfg)  # what is in use now is kept before it changes
    section = getattr(cfg, preset.backend)
    address = address.strip().rstrip("/") if preset.free_address else ""
    was_same = hosted_preset(cfg) is preset and (
        not preset.free_address or _same_address(section.base_url, address)
    )
    # A section that only this service writes keeps the key it has; a shared one (OpenAI, DeepSeek and
    # "Other" write ``[openai]``) holds the key of the service in use, which is not this one's.
    shared = sum(item.backend == preset.backend for item in PRESETS) > 1
    kept = was_same or (
        not shared and key_state(preset.backend, section) in (ENV, KEYCHAIN)
    )
    section.base_url = address or preset.stored_url
    if not kept:
        service = service_for(preset, address or preset.base_url)
        env = bool(preset.key_env and os.environ.get(preset.key_env))
        saved = not env and keychain.status(service) == keychain.SAVED
        section.api_key_env = preset.key_env if env else ""
        section.api_key = keychain.ref_for(service) if saved else ""
    if not was_same and shared:
        section.thinking = ""
    section.chat_model = model
    cfg.chat_backend = preset.backend
    cfg.chat_alias = ""


def use_local(cfg: Config, backend: str, model: str) -> None:
    """Put the chat of *cfg* on the local server *backend*, with *model* when it names one."""
    from localmelo.support.backends.registry import get_backend

    cfg.services = last_models(cfg)
    if cfg.chat_backend != backend:
        cfg.chat_alias = ""
    cfg.chat_backend = backend
    if get_backend(backend).model_required:
        getattr(cfg, backend).chat_model = model


def key_errors(name: str, section: CloudVendorConfig, example: str) -> list[str]:
    """What is wrong with the key of the ``[name]`` section, in plain words.

    *example* is an environment variable name to suggest, such as ``OPENAI_API_KEY``.
    """
    try:
        keychain.parse_ref(section.api_key)
    except ValueError as exc:
        return [f"[{name}] api_key: {exc}"]
    state = key_state(name, section)
    if state in (ENV, KEYCHAIN):
        return []
    if state == ADDRESS:
        return [f"[{name}] {t_in('en', keychain.words_key(ADDRESS))}"]
    if state == keychain.MISSING:
        service = keychain.parse_ref(section.api_key) or ""
        return [
            f"[{name}] the key {service} is not in the keychain. Open Settings, "
            "choose the model server and paste the key again."
        ]
    if state == keychain.LOCKED:
        return [
            f"[{name}] the keychain is locked. Unlock it (open Keychain Access, or "
            "sign in to this Mac again), then try again."
        ]
    if state == keychain.UNAVAILABLE:
        return [
            f"[{name}] the keychain is not available on this computer. Set "
            "api_key_env to the name of a variable that holds the key "
            f"(e.g. {example})."
        ]
    if section.api_key_env:
        return [
            f"[{name}] environment variable ${section.api_key_env} is not set. "
            "Export it before starting the gateway, or paste the key in Settings."
        ]
    return [
        f"[{name}] api_key_env is empty. Paste the key in Settings (Model server), "
        "or set the environment variable name that holds your API key "
        f"(e.g. {example})."
    ]


def hosted_fields_errors(name: str, section: CloudVendorConfig) -> list[str]:
    """The shape errors of a hosted section's ``api_key`` and ``thinking``."""
    errors: list[str] = []
    if not isinstance(section.api_key, str):
        errors.append(f"[{name}] api_key must be a string, got {section.api_key!r}")
    else:
        try:
            keychain.parse_ref(section.api_key)
        except ValueError as exc:
            errors.append(f"[{name}] api_key: {exc}")
    if section.thinking not in THINKING:
        errors.append(
            f"[{name}] thinking {section.thinking!r} is not recognised. "
            f"Must be one of {THINKING}."
        )
    return errors

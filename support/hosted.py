"""Hosted services: the presets Settings and the setup list offer, and their keys.

A preset is a row of product data, not a new backend. Each one names the config
section it fills (``backend``: ``openai``, ``anthropic``, ``gemini`` or
``nvidia``), the address of the service, the models worth suggesting, the
environment variable that held the key before the keychain, and the request
fields that turn the model's thinking off. A service that speaks the OpenAI
format (DeepSeek, and the free-form "Other") uses the ``openai`` backend with its
own address, so the section's address tells the presets apart.

The key is looked up in this order: the environment variable the section names
(``api_key_env``), when it is set; then the keychain item the section refers to
(``api_key = "keychain:localmelo.<name>"``, see :mod:`localmelo.support.keychain`).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

from localmelo.support import keychain

if TYPE_CHECKING:
    from localmelo.support.config import CloudVendorConfig

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
    thinking off.
    """

    id: str
    label: str
    backend: str
    base_url: str
    models: tuple[str, ...]
    key_env: str
    thinking_off: Mapping[str, Any] = field(default_factory=dict)
    off_models: tuple[str, ...] = ()

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
    ),
    Preset(
        "anthropic",
        "Anthropic",
        "anthropic",
        "https://api.anthropic.com/v1",
        ("claude-sonnet-4-20250514",),
        "ANTHROPIC_API_KEY",
    ),
    Preset(
        "gemini",
        "Google Gemini",
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        ("gemini-2.0-flash",),
        "GEMINI_API_KEY",
    ),
    Preset(
        "nvidia",
        "NVIDIA",
        "nvidia",
        "https://integrate.api.nvidia.com/v1",
        ("nvidia/llama-3.1-nemotron-70b-instruct",),
        "NVIDIA_API_KEY",
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
            "key_ref": keychain.ref_for(preset.service),
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


def key_state(section: CloudVendorConfig) -> str:
    """Where the section's key comes from now.

    ``env`` (the named variable is set), ``keychain`` (the item is saved),
    ``missing`` (the item the config names is not there), ``locked``,
    ``unavailable`` (no keychain on this computer) or ``none`` (no key named).
    """
    if section.api_key_env and os.environ.get(section.api_key_env):
        return ENV
    try:
        service = keychain.parse_ref(section.api_key)
    except ValueError:
        return NONE
    if service is None:
        return NONE
    answer = keychain.status(service)
    return KEYCHAIN if answer == keychain.SAVED else answer


def api_key(section: CloudVendorConfig) -> str:
    """The section's API key, or "" when none is named.

    The environment variable wins when it is set. A keychain item that is missing
    or cannot be read raises :class:`localmelo.support.keychain.KeychainError`
    with the fix.
    """
    if section.api_key_env:
        value = os.environ.get(section.api_key_env, "")
        if value:
            return value
    service = keychain.parse_ref(section.api_key)
    if service is None:
        return ""
    return keychain.read(service)


def key_errors(name: str, section: CloudVendorConfig, example: str) -> list[str]:
    """What is wrong with the key of the ``[name]`` section, in plain words.

    *example* is an environment variable name to suggest, such as ``OPENAI_API_KEY``.
    """
    try:
        keychain.parse_ref(section.api_key)
    except ValueError as exc:
        return [f"[{name}] api_key: {exc}"]
    state = key_state(section)
    if state in (ENV, KEYCHAIN):
        return []
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

"""Persistent configuration stored at ~/.cache/localmelo/config.toml."""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


CONFIG_DIR = os.path.expanduser("~/.cache/localmelo")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")

VALID_CHAT_BACKENDS = (
    "mlc",
    "ollama",
    "vllm",
    "sglang",
    "openai",
    "gemini",
    "anthropic",
    "nvidia",
)
VALID_EMBEDDING_BACKENDS = ("mlc", "ollama", "vllm", "sglang", "none")


@dataclass
class LocalBackendConfig:
    """Config shape shared by all local OpenAI-compatible backends."""

    chat_url: str = ""
    chat_model: str = ""
    embedding_url: str = ""
    embedding_model: str = ""
    # Extra top-level fields merged into every chat request body, e.g.
    # {"chat_template_kwargs": {"enable_thinking": False}}.
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudVendorConfig:
    """Config shape shared by all cloud vendor backends."""

    api_key_env: str = ""  # env var name, e.g. OPENAI_API_KEY
    chat_model: str = ""
    base_url: str = ""  # optional override; each vendor has a built-in default
    # Extra top-level fields merged into every chat request body.
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayConfig:
    port: int = 8401
    host: str = "127.0.0.1"


@dataclass
class SchedulerConfig:
    """The live gate's settings (S2): off, or one of the two enforcing arms.

    ``n_srv`` is the model server's slot count, ``r_f`` the reserve kept for
    callers the gate cannot see (the CLI and direct runs) and ``n_fg`` the
    background cap while a chat is focused. The call log holds one JSONL
    file per gateway run; an empty ``call_log_dir`` means the default.
    """

    policy: str = "off"  # off | fcfs | class
    n_srv: int = 8
    r_f: int = 1
    n_fg: int = 1
    call_log: bool = True
    call_log_dir: str = ""
    # The GPU lock and the battery (S2b). An empty path means no lock
    # reader; the gateway's battery reader is always on. ``record`` only
    # observes; ``wait`` holds every workspace call while a foreign lock is
    # present.
    gpu_lock_file: str = ""
    lock_self: str = ""
    lock_gate: str = "record"  # record | wait


@dataclass
class Config:
    chat_backend: str = (
        ""  # mlc | ollama | vllm | sglang | openai | gemini | anthropic | nvidia
    )
    embedding_backend: str = ""  # mlc | ollama | vllm | sglang | none

    # HTTP timeout of each chat request, in seconds, for every chat backend.
    # None keeps each chat provider's own default (120 s; 300 s for the
    # ollama backend). Embedding requests keep their own timeout.
    request_timeout_s: float | None = None

    # Local backends
    mlc: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    ollama: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    vllm: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    sglang: LocalBackendConfig = field(default_factory=LocalBackendConfig)

    # Cloud vendors
    openai: CloudVendorConfig = field(default_factory=CloudVendorConfig)
    gemini: CloudVendorConfig = field(default_factory=CloudVendorConfig)
    anthropic: CloudVendorConfig = field(default_factory=CloudVendorConfig)
    nvidia: CloudVendorConfig = field(default_factory=CloudVendorConfig)

    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def is_configured(self) -> bool:
        return self.chat_backend != ""

    @property
    def has_embedding(self) -> bool:
        if self.embedding_backend in ("", "none"):
            return False
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.embedding_backend)
            return backend.has_embedding(self)
        except KeyError:
            return False

    def validate(self) -> list[str]:
        """Return a list of validation errors. Empty list means valid."""
        errors: list[str] = []
        errors.extend(self._validate_scheduler())

        # -- chat_backend validation --
        if not self.chat_backend:
            errors.append("chat_backend is not set (run 'melo --reconfigure')")
            return errors

        if self.chat_backend not in VALID_CHAT_BACKENDS:
            errors.append(
                f"chat_backend {self.chat_backend!r} is not recognised. "
                f"Must be one of {VALID_CHAT_BACKENDS}."
            )
            return errors

        # -- embedding_backend validation --
        if not self.embedding_backend:
            errors.append("embedding_backend is not set (run 'melo --reconfigure')")
            return errors

        if self.embedding_backend not in VALID_EMBEDDING_BACKENDS:
            errors.append(
                f"embedding_backend {self.embedding_backend!r} is not recognised. "
                f"Must be one of {VALID_EMBEDDING_BACKENDS}."
            )
            return errors

        # -- Delegate chat backend validation --
        chat_errors = self._validate_chat_backend()
        errors.extend(chat_errors)
        errors.extend(self._validate_extra_body())

        # -- Delegate embedding backend validation (skip if "none") --
        if self.embedding_backend != "none":
            emb_errors = self._validate_embedding_backend()
            errors.extend(emb_errors)

        # -- Chat request timeout --
        errors.extend(self._validate_request_timeout())

        # -- Gateway validation --
        if self.gateway.port <= 0 or self.gateway.port > 65535:
            errors.append(
                f"[gateway] port={self.gateway.port} is out of range (1-65535)"
            )

        return errors

    def _validate_scheduler(self) -> list[str]:
        """``[scheduler]`` values are inside their documented ranges."""
        scheduler = self.scheduler
        errors: list[str] = []
        if scheduler.policy not in ("off", "fcfs", "class"):
            errors.append(
                f"[scheduler] policy={scheduler.policy!r} is not one of "
                "('off', 'fcfs', 'class')"
            )
        for name, minimum in (("n_srv", 1), ("r_f", 0), ("n_fg", 0)):
            value: object = getattr(scheduler, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                errors.append(
                    f"[scheduler] {name} must be an integer >= {minimum}, "
                    f"got {value!r}"
                )
        if not isinstance(scheduler.call_log, bool):
            errors.append(
                f"[scheduler] call_log must be a boolean, got {scheduler.call_log!r}"
            )
        if not isinstance(scheduler.call_log_dir, str):
            errors.append(
                "[scheduler] call_log_dir must be a string, "
                f"got {scheduler.call_log_dir!r}"
            )
        if scheduler.lock_gate not in ("record", "wait"):
            errors.append(
                f"[scheduler] lock_gate={scheduler.lock_gate!r} is not one of "
                "('record', 'wait')"
            )
        for name in ("gpu_lock_file", "lock_self"):
            value = getattr(scheduler, name)
            if not isinstance(value, str):
                errors.append(f"[scheduler] {name} must be a string, got {value!r}")
            elif "\x00" in value:
                errors.append(f"[scheduler] {name} must not contain a NUL byte")
        return errors

    def _validate_request_timeout(self) -> list[str]:
        """``request_timeout_s`` is unset or a finite number of seconds > 0."""
        value: object = self.request_timeout_s
        if value is None or _is_positive_seconds(value):
            return []
        return [
            "request_timeout_s must be a finite number of seconds > 0, "
            f"got {value!r}"
        ]

    def _validate_chat_backend(self) -> list[str]:
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.chat_backend)
            return backend.validate(self)
        except KeyError:
            return [f"chat_backend {self.chat_backend!r}: no registered adapter found."]

    def _validate_extra_body(self) -> list[str]:
        from localmelo.support.providers.llm._protocol import extra_body_errors

        section = getattr(self, self.chat_backend, None)
        if section is None:
            return []
        return [
            f"[{self.chat_backend}] {err}"
            for err in extra_body_errors(getattr(section, "extra_body", {}))
        ]

    def _validate_embedding_backend(self) -> list[str]:
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.embedding_backend)
            return backend.validate_embedding(self)
        except KeyError:
            return [
                f"embedding_backend {self.embedding_backend!r}: "
                f"no registered adapter found."
            ]

    def validate_or_raise(self) -> None:
        """Validate and raise ConfigError if any issues are found."""
        errors = self.validate()
        if errors:
            msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(msg)


def _is_positive_seconds(value: object) -> bool:
    """Whether *value* is a finite number > 0.

    An integer counts (TOML reads ``600`` as one); a bool does not, and
    neither does an integer too large to be a float.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        seconds = float(value)
    except OverflowError:
        return False
    return math.isfinite(seconds) and seconds > 0


# -- Compatibility aliases (temporary, reduces churn in files not yet migrated) --
MlcConfig = LocalBackendConfig
OllamaConfig = LocalBackendConfig
VllmConfig = LocalBackendConfig
SglangConfig = LocalBackendConfig

# -- Section names for iteration --

_LOCAL_SECTIONS = ("mlc", "ollama", "vllm", "sglang")
_CLOUD_SECTIONS = ("openai", "gemini", "anthropic", "nvidia")

# -- Migration helpers --

_CLOUD_API_PROVIDER_MIGRATION: dict[str, str] = {
    "openai": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "nvidia": "nvidia",
}


def load() -> Config:
    if not os.path.isfile(CONFIG_PATH):
        return Config()

    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)

    cfg = Config()

    # -- Migration: old single `backend` field --
    if "backend" in data and "chat_backend" not in data:
        cfg = _migrate_legacy(data)
        return cfg

    # -- Migration: old `mlc-llm` / `cloud_api` keys in chat_backend/embedding_backend --
    raw_chat = data.get("chat_backend", "")
    raw_emb = data.get("embedding_backend", "")
    cfg.chat_backend = _migrate_backend_key(raw_chat, data)
    cfg.embedding_backend = _migrate_embedding_key(raw_emb)

    # -- Load local backend sections --
    for section in _LOCAL_SECTIONS:
        if section in data:
            target = getattr(cfg, section)
            for k, v in data[section].items():
                if hasattr(target, k):
                    setattr(target, k, v)
            # Migration: flat chat_template_kwargs -> extra_body
            _migrate_extra_body(target, data[section])
        # Migration: old [mlc] section with chat_port -> chat_url
        if section == "mlc" and "mlc" in data:
            _migrate_mlc_section(cfg.mlc, data["mlc"])

    # -- Load cloud vendor sections --
    for section in _CLOUD_SECTIONS:
        if section in data:
            target = getattr(cfg, section)
            for k, v in data[section].items():
                if hasattr(target, k):
                    setattr(target, k, v)
            # Migration: flat chat_template_kwargs -> extra_body
            _migrate_extra_body(target, data[section])

    # Migration: old [cloud_api] section -> vendor-specific section
    if "cloud_api" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["cloud_api"])
    elif "online" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["online"])

    if "gateway" in data:
        for k, v in data["gateway"].items():
            if hasattr(cfg.gateway, k):
                setattr(cfg.gateway, k, v)

    # A [scheduler] that is not a table is ignored, like an unknown key; a
    # malformed value inside a real table is left for Config.validate.
    if isinstance(data.get("scheduler"), dict):
        for k, v in data["scheduler"].items():
            if hasattr(cfg.scheduler, k):
                setattr(cfg.scheduler, k, v)

    # A bad value is left for Config.validate to report.
    cfg.request_timeout_s = data.get("request_timeout_s")

    return cfg


def _migrate_backend_key(raw: str, data: dict) -> str:
    """Migrate old chat_backend keys to new ones."""
    if raw == "mlc-llm":
        return "mlc"
    if raw == "cloud_api":
        provider = data.get("cloud_api", {}).get("provider", "")
        return _CLOUD_API_PROVIDER_MIGRATION.get(provider, raw)
    return raw


def _migrate_embedding_key(raw: str) -> str:
    """Migrate old embedding_backend keys to new ones."""
    if raw == "mlc-llm":
        return "mlc"
    return raw


def _migrate_mlc_section(mlc: LocalBackendConfig, section: dict) -> None:
    """Migrate old [mlc] with chat_port/gpu_memory_gb to URL-based config."""
    if "chat_port" in section and not mlc.chat_url:
        port = section["chat_port"]
        mlc.chat_url = f"http://127.0.0.1:{port}/v1"
        if not mlc.embedding_url:
            mlc.embedding_url = mlc.chat_url


def _migrate_extra_body(
    target: LocalBackendConfig | CloudVendorConfig, section: Mapping[str, Any]
) -> None:
    """Fold a flat ``chat_template_kwargs`` table into ``extra_body``.

    ``extra_body`` is the canonical form; a section-level
    ``chat_template_kwargs = {...}`` is accepted as shorthand and is
    written back in canonical form by :func:`save`. An explicit
    ``extra_body.chat_template_kwargs`` wins over the flat key. A bad
    ``extra_body`` value is left for :meth:`Config.validate` to report.
    """
    flat = section.get("chat_template_kwargs")
    if not isinstance(flat, dict) or not isinstance(target.extra_body, dict):
        return
    if "chat_template_kwargs" not in target.extra_body:
        target.extra_body = {**target.extra_body, "chat_template_kwargs": flat}


def _migrate_cloud_api_section(cfg: Config, section: dict) -> None:
    """Migrate old [cloud_api] or [online] section to vendor-specific config."""
    vendor = cfg.chat_backend
    target = getattr(cfg, vendor, None)
    if target is None:
        return
    for k, v in section.items():
        if k in ("provider", "local_embedding"):
            continue
        if hasattr(target, k):
            setattr(target, k, v)
    _migrate_extra_body(target, section)


def _migrate_legacy(data: dict) -> Config:
    """Migrate the oldest format with a single `backend` field."""
    cfg = Config()
    old = data["backend"]

    if old == "mlc-llm":
        cfg.chat_backend = "mlc"
        cfg.embedding_backend = "mlc"
        if "mlc" in data:
            for k, v in data["mlc"].items():
                if hasattr(cfg.mlc, k):
                    setattr(cfg.mlc, k, v)
            _migrate_mlc_section(cfg.mlc, data["mlc"])
            _migrate_extra_body(cfg.mlc, data["mlc"])
    elif old == "online":
        online_sec = data.get("online", {})
        provider = online_sec.get("provider", "")
        cfg.chat_backend = _CLOUD_API_PROVIDER_MIGRATION.get(provider, "")
        if online_sec.get("local_embedding", False):
            cfg.embedding_backend = "mlc"
        else:
            cfg.embedding_backend = "none"
        if cfg.chat_backend in _CLOUD_SECTIONS:
            _migrate_cloud_api_section(cfg, online_sec)
    elif old == "ollama":
        cfg.chat_backend = "ollama"
        ollama_sec = data.get("ollama", {})
        cfg.embedding_backend = (
            "ollama" if ollama_sec.get("embedding_model", "") else "none"
        )
        if "ollama" in data:
            for k, v in data["ollama"].items():
                if hasattr(cfg.ollama, k):
                    setattr(cfg.ollama, k, v)
            _migrate_extra_body(cfg.ollama, data["ollama"])

    if "gateway" in data:
        for k, v in data["gateway"].items():
            if hasattr(cfg.gateway, k):
                setattr(cfg.gateway, k, v)

    cfg.request_timeout_s = data.get("request_timeout_s")

    return cfg


_BARE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_string(text: str) -> str:
    # JSON escapes the quotation mark, the backslash and the control
    # characters below U+0020 the way a TOML basic string does; TOML also
    # requires DEL to be escaped. Everything else is written as is: an
    # ASCII-only JSON literal would escape characters outside the BMP as
    # UTF-16 surrogate pairs, which TOML rejects.
    return json.dumps(text, ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_value(value: Any) -> str:
    """Render *value* as a single-line TOML value (inline tables for dicts)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            k = str(key)
            k = k if _BARE_TOML_KEY.match(k) else _toml_string(k)
            items.append(f"{k} = {_toml_value(item)}")
        return "{" + ", ".join(items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ConfigError(f"Cannot write {type(value).__name__} value {value!r} to TOML")


def save(cfg: Config) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)

    lines: list[str] = []
    lines.append(f"chat_backend = {_toml_string(cfg.chat_backend)}")
    lines.append(f"embedding_backend = {_toml_string(cfg.embedding_backend)}")
    # Top-level keys go before the first table. Only a value that is set
    # is written, so a config without one is saved as it was before.
    if cfg.request_timeout_s is not None:
        lines.append(f"request_timeout_s = {_toml_value(cfg.request_timeout_s)}")
    lines.append("")

    # Local backend sections
    for section in _LOCAL_SECTIONS:
        sec = getattr(cfg, section)
        lines.append(f"[{section}]")
        lines.append(f"chat_url = {_toml_string(sec.chat_url)}")
        lines.append(f"chat_model = {_toml_string(sec.chat_model)}")
        lines.append(f"embedding_url = {_toml_string(sec.embedding_url)}")
        lines.append(f"embedding_model = {_toml_string(sec.embedding_model)}")
        if sec.extra_body:
            lines.append(f"extra_body = {_toml_value(sec.extra_body)}")
        lines.append("")

    # Cloud vendor sections
    for section in _CLOUD_SECTIONS:
        sec = getattr(cfg, section)
        lines.append(f"[{section}]")
        lines.append(f"api_key_env = {_toml_string(sec.api_key_env)}")
        lines.append(f"chat_model = {_toml_string(sec.chat_model)}")
        lines.append(f"base_url = {_toml_string(sec.base_url)}")
        if sec.extra_body:
            lines.append(f"extra_body = {_toml_value(sec.extra_body)}")
        lines.append("")

    lines.append("[gateway]")
    lines.append(f"port = {cfg.gateway.port}")
    lines.append(f"host = {_toml_string(cfg.gateway.host)}")
    lines.append("")

    # Only a settings section that differs from the defaults is written, so
    # an existing config is saved exactly as before and a reconfigure keeps
    # a non-default scheduler (decision D7).
    scheduler = cfg.scheduler
    if scheduler != SchedulerConfig():
        lines.append("[scheduler]")
        lines.append(f"policy = {_toml_string(scheduler.policy)}")
        lines.append(f"n_srv = {scheduler.n_srv}")
        lines.append(f"r_f = {scheduler.r_f}")
        lines.append(f"n_fg = {scheduler.n_fg}")
        lines.append(f"call_log = {'true' if scheduler.call_log else 'false'}")
        lines.append(f"call_log_dir = {_toml_string(scheduler.call_log_dir)}")
        lines.append(f"gpu_lock_file = {_toml_string(scheduler.gpu_lock_file)}")
        lines.append(f"lock_self = {_toml_string(scheduler.lock_self)}")
        lines.append(f"lock_gate = {_toml_string(scheduler.lock_gate)}")
        lines.append("")

    # TOML files are UTF-8. Encode before opening the file, so text that
    # cannot be encoded (a str holding a lone surrogate) fails the save
    # without truncating the existing config.
    text = "\n".join(lines)
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        bad = exc.object[exc.start : exc.end]
        raise ConfigError(f"Cannot write {bad!r} to TOML: {exc.reason}") from exc
    with open(CONFIG_PATH, "wb") as f:
        f.write(data)

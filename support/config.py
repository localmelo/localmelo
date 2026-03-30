"""Persistent configuration stored at ~/.cache/localmelo/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field


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


@dataclass
class CloudVendorConfig:
    """Config shape shared by all cloud vendor backends."""

    api_key_env: str = ""  # env var name, e.g. OPENAI_API_KEY
    chat_model: str = ""
    base_url: str = ""  # optional override; each vendor has a built-in default


@dataclass
class GatewayConfig:
    port: int = 8401
    host: str = "127.0.0.1"


@dataclass
class Config:
    chat_backend: str = (
        ""  # mlc | ollama | vllm | sglang | openai | gemini | anthropic | nvidia
    )
    embedding_backend: str = ""  # mlc | ollama | vllm | sglang | none

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

        # -- Delegate embedding backend validation (skip if "none") --
        if self.embedding_backend != "none":
            emb_errors = self._validate_embedding_backend()
            errors.extend(emb_errors)

        # -- Gateway validation --
        if self.gateway.port <= 0 or self.gateway.port > 65535:
            errors.append(
                f"[gateway] port={self.gateway.port} is out of range (1-65535)"
            )

        return errors

    def _validate_chat_backend(self) -> list[str]:
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.chat_backend)
            return backend.validate(self)
        except KeyError:
            return [f"chat_backend {self.chat_backend!r}: no registered adapter found."]

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

    # Migration: old [cloud_api] section -> vendor-specific section
    if "cloud_api" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["cloud_api"])
    elif "online" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["online"])

    if "gateway" in data:
        for k, v in data["gateway"].items():
            if hasattr(cfg.gateway, k):
                setattr(cfg.gateway, k, v)

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

    if "gateway" in data:
        for k, v in data["gateway"].items():
            if hasattr(cfg.gateway, k):
                setattr(cfg.gateway, k, v)

    return cfg


def save(cfg: Config) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)

    lines: list[str] = []
    lines.append(f'chat_backend = "{cfg.chat_backend}"')
    lines.append(f'embedding_backend = "{cfg.embedding_backend}"')
    lines.append("")

    # Local backend sections
    for section in _LOCAL_SECTIONS:
        sec = getattr(cfg, section)
        lines.append(f"[{section}]")
        lines.append(f'chat_url = "{sec.chat_url}"')
        lines.append(f'chat_model = "{sec.chat_model}"')
        lines.append(f'embedding_url = "{sec.embedding_url}"')
        lines.append(f'embedding_model = "{sec.embedding_model}"')
        lines.append("")

    # Cloud vendor sections
    for section in _CLOUD_SECTIONS:
        sec = getattr(cfg, section)
        lines.append(f"[{section}]")
        lines.append(f'api_key_env = "{sec.api_key_env}"')
        lines.append(f'chat_model = "{sec.chat_model}"')
        lines.append(f'base_url = "{sec.base_url}"')
        lines.append("")

    lines.append("[gateway]")
    lines.append(f"port = {cfg.gateway.port}")
    lines.append(f'host = "{cfg.gateway.host}"')
    lines.append("")

    with open(CONFIG_PATH, "w") as f:
        f.write("\n".join(lines))

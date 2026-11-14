"""The backends that speak the OpenAI API: one row each.

A model server on this computer is a row of :data:`LOCAL_SERVERS` (its facts are
fields of the row); a hosted service is a row of :data:`HOSTED_SERVICES`, built
from :data:`localmelo.support.hosted.PRESETS`, which holds its address, key
variable and model suggestions. :data:`LOCAL_BACKENDS` and :data:`CLOUD_BACKENDS`
are their keys, and every other list of backends in the code derives from them.

Adding a local backend: a row below, and a ``LocalBackendConfig`` field of the
same name on :class:`localmelo.support.config.Config`. Adding a hosted one: a
:class:`~localmelo.support.hosted.Preset` whose id is its backend, and a
``CloudVendorConfig`` field. A server that needs its own behaviour (Ollama's
native chat, oMLX's label) subclasses :class:`LocalBackend`.

This module stays light: the config imports it, so the HTTP providers are
imported where a provider is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final

from localmelo.support import hosted, melo_serve
from localmelo.support.backends.base import BaseBackend, configured_model

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config

# What a label or an error calls a server of the OpenAI-compatible kind.
OPENAI_COMPATIBLE_NAME: Final = "OpenAI-compatible server"


@dataclass(frozen=True)
class LocalBackend(BaseBackend):
    """A model server on this computer that speaks the OpenAI API.

    *url* is where it listens by default, as the setup offers it; *chat_model* and
    *embedding_model* are the models the setup suggests (a chat model also is the
    example in an error). *menu* is its line in the setup lists. *template_kwargs*
    says the server takes ``chat_template_kwargs``, which turns thinking off.
    """

    key: str
    display_name: str
    menu: str
    url: str
    chat_model: str = ""
    embedding_model: str = ""
    template_kwargs: bool = False

    @property
    def detect_url(self) -> str:
        """Where to look for the server: its address with a numeric host, ending in ``/v1``."""
        root = self.url.rstrip("/").removesuffix("/v1")
        return f"{root.replace('//localhost:', '//127.0.0.1:')}/v1"

    def name_in_errors(self, cfg: Config, *, embedding: bool = False) -> str:
        """The server's name in a validation message."""
        return self.label(cfg, embedding=embedding)

    def validate(self, cfg: Config) -> list[str]:
        section, name = getattr(cfg, self.key), self.name_in_errors(cfg)
        errors: list[str] = []
        if not section.chat_url:
            errors.append(
                f"[{self.key}] chat_url is empty. "
                f"Set it to your {name} URL (e.g. {self.url})."
            )
        if self.model_required and not section.chat_model:
            example = f" (e.g. {self.chat_model})" if self.chat_model else ""
            errors.append(
                f"[{self.key}] chat_model is empty. "
                f"Specify the {name} model name{example}."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        section = getattr(cfg, self.key)
        name = self.name_in_errors(cfg, embedding=True)
        errors: list[str] = []
        if not section.embedding_url:
            errors.append(
                f"[{self.key}] embedding_url is empty. "
                f"Set it to your {name} URL for embedding."
            )
        if self.model_required and not section.embedding_model:
            errors.append(
                f"[{self.key}] embedding_model is empty. "
                f"Set the {name} embedding model name."
            )
        return errors

    def has_embedding(self, cfg: Config) -> bool:
        section = getattr(cfg, self.key)
        return bool(
            section.embedding_url
            and (section.embedding_model or not self.model_required)
        )

    def request_model(self, cfg: Config, *, embedding: bool = False) -> str:
        """The model a provider built now asks for; raises when there is none."""
        return self.model_name(cfg, embedding=embedding)

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        from localmelo.support.backends.openai_compat import (
            build_openai_chat,
            normalize_url,
        )

        section = getattr(cfg, self.key)
        return build_openai_chat(
            base_url=normalize_url(section.chat_url),
            model=self.request_model(cfg),
            extra_body=section.extra_body,
            timeout=cfg.request_timeout_s,
            reasoning_field=section.reasoning_field or None,
            requires_reasoning_replay=section.requires_reasoning_replay,
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        from localmelo.support.backends.openai_compat import (
            build_openai_embedding,
            normalize_url,
        )

        section = getattr(cfg, self.key)
        if not self.has_embedding(cfg):
            return None
        return build_openai_embedding(
            base_url=normalize_url(section.embedding_url),
            model=self.request_model(cfg, embedding=True),
        )


class OllamaBackend(LocalBackend):
    """Ollama chats on its native API, so thinking works and ``/v1`` is not used."""

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        section = getattr(cfg, self.key)
        url = section.chat_url.rstrip("/").removesuffix("/v1")
        # Unless request_timeout_s is set, the provider keeps its default
        # (CHAT_TIMEOUT_S); httpx would read timeout=None as no timeout at all.
        options: dict[str, Any] = (
            {} if cfg.request_timeout_s is None else {"timeout": cfg.request_timeout_s}
        )
        return OllamaNativeChat(
            base_url=url,
            model=section.chat_model,
            extra_body=section.extra_body,
            requires_reasoning_replay=section.requires_reasoning_replay,
            **options,
        )


class VllmBackend(LocalBackend):
    """The key ``vllm`` stands for oMLX, vLLM and every other OpenAI-compatible server.

    ``chat_backend = "omlx"`` is the same backend, spelled so.
    """

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        """``oMLX`` when the config says so or a listing of the server did."""
        from localmelo.support.backends import discovery

        section = getattr(cfg, self.key)
        if embedding:
            url = section.embedding_url or section.chat_url
            declared = cfg.embedding_alias
        else:
            url, declared = section.chat_url, cfg.chat_alias
        return discovery.server_name(url, declared=declared == "omlx")


class MeloServeBackend(LocalBackend):
    """melo_serve: one address for the models it serves, switched in one command.

    The config names no model (an empty one is the usual): what melo_serve
    serves now is read from a snapshot of its state when a provider is built or
    a banner is drawn, so a switch made with ``melo-serve use`` is followed from
    the next build. The label and the model come from the same snapshot. This
    row only reads; ``melo serve`` is what asks melo_serve to start.
    """

    model_required: ClassVar[bool] = False

    def in_use(self, cfg: Config, *, embedding: bool = False) -> tuple[str, str]:
        """``Splash via melo_serve`` and the model, or ``melo_serve`` and none."""
        return self._in_use(cfg, melo_serve.snapshot(), embedding)

    def _in_use(
        self, cfg: Config, snap: melo_serve.Snapshot, embedding: bool
    ) -> tuple[str, str]:
        served = snap.roles.get(melo_serve.ROLES[embedding])
        named = configured_model(cfg, self.key, embedding=embedding)
        if served is None:
            return self.display_name, named
        via = served.backend_name
        label = f"{via} via {self.display_name}" if via else self.display_name
        return label, named or served.model

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        return self.in_use(cfg, embedding=embedding)[0]

    def model_name(self, cfg: Config, *, embedding: bool = False) -> str:
        return self.in_use(cfg, embedding=embedding)[1]

    def name_in_errors(self, cfg: Config, *, embedding: bool = False) -> str:
        """Validation reads the config only, never melo_serve's state."""
        return self.display_name

    def welcome_note(self, label: str) -> str:
        """The server that really answers: a model served through melo_serve."""
        return label

    def request_model(self, cfg: Config, *, embedding: bool = False) -> str:
        snap = melo_serve.snapshot()
        model = self._in_use(cfg, snap, embedding)[1]
        if not model:
            raise melo_serve.unavailable(snap, melo_serve.ROLES[embedding])
        return model

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        """None while no embedding model is served: the chat still starts.

        One snapshot answers both whether one is served and which model to ask for.
        """
        from localmelo.support.backends.openai_compat import (
            build_openai_embedding,
            normalize_url,
        )

        section = cfg.melo_serve
        snap = melo_serve.snapshot()
        model = self._in_use(cfg, snap, True)[1]
        if not section.embedding_url or not model:
            return None
        return build_openai_embedding(
            base_url=normalize_url(section.embedding_url), model=model
        )


LOCAL_SERVERS: Final[tuple[LocalBackend, ...]] = (
    LocalBackend(
        "mlc",
        "MLC-LLM",
        "MLC (local endpoint)",
        "http://localhost:8400/v1",
        chat_model="qwen3-1.7b",
        embedding_model="nomic-embed",
    ),
    OllamaBackend(
        "ollama",
        "Ollama",
        "Ollama (local endpoint)",
        "http://localhost:11434",
        chat_model="qwen3:8b",
        embedding_model="nomic-embed-text",
    ),
    # A menu is drawn before any server is chosen, so its line cannot tell oMLX
    # from vLLM; a banner can (see discovery.server_name).
    VllmBackend(
        "vllm",
        OPENAI_COMPATIBLE_NAME,
        "oMLX or vLLM (OpenAI-compatible, local)",
        "http://localhost:8000/v1",
        template_kwargs=True,
    ),
    LocalBackend(
        "sglang",
        "SGLang",
        "SGLang (local endpoint)",
        "http://localhost:30000/v1",
        template_kwargs=True,
    ),
    # It stands in for the stack it replaces (oMLX or Splash), which is reached
    # as vllm today, so it takes the same chat_template_kwargs.
    MeloServeBackend(
        "melo_serve",
        "melo_serve",
        "melo_serve (local models, one address)",
        melo_serve.DEFAULT_URL,
        template_kwargs=True,
    ),
)
LOCAL_BACKENDS: Final = tuple(server.key for server in LOCAL_SERVERS)
# The address to suggest when no server was found: the OpenAI-compatible row's.
USUAL_URL: Final = next(s for s in LOCAL_SERVERS if s.key == "vllm").detect_url


@dataclass(frozen=True)
class HostedBackend(BaseBackend):
    """A hosted service that speaks the OpenAI API; its preset has the rest."""

    key: str
    display_name: str

    @property
    def preset(self) -> hosted.Preset:
        return hosted.BY_ID[self.key]

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        """The service this section's address names: DeepSeek, OpenAI or its host."""
        return hosted.label_for(self.key, getattr(cfg, self.key))

    def validate(self, cfg: Config) -> list[str]:
        section, preset = getattr(cfg, self.key), self.preset
        errors = hosted.key_errors(self.key, section, preset.key_env)
        if not section.chat_model:
            errors.append(
                f"[{self.key}] chat_model is empty. "
                f"Specify the model name (e.g. {preset.models[0]})."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        return [f"{self.key} is a chat-only backend and cannot be used for embedding."]

    def has_embedding(self, cfg: Config) -> bool:
        return False

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        from localmelo.support.backends.openai_compat import build_openai_chat

        section = getattr(cfg, self.key)
        return build_openai_chat(
            base_url=hosted.address_of(self.key, section),
            model=section.chat_model,
            api_key=hosted.api_key(self.key, section),
            extra_body=hosted.request_body(self.key, section),
            thinking_off_body=hosted.retry_body(self.key, section),
            **hosted.image_args(self.key, section),
            timeout=cfg.request_timeout_s,
            reasoning_field=section.reasoning_field or None,
            requires_reasoning_replay=section.requires_reasoning_replay,
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None


HOSTED_SERVICES: Final[tuple[HostedBackend, ...]] = tuple(
    HostedBackend(preset.id, preset.label)
    for preset in hosted.PRESETS
    if preset.id == preset.backend
)
CLOUD_BACKENDS: Final = tuple(service.key for service in HOSTED_SERVICES)

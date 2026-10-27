"""The terminal UI's model backend: which server and model, and its Agents.

:class:`AgentFactory` knows where the session's model comes from (the
config, a URL given with ``--base-url``, a server found on this machine,
or the first run's answers), builds the session's Agents on it, lists the
server's models, and saves a new choice as the default when the session
may (``/model``, ``/backend``).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from typing import Final

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support import config, onboard
from localmelo.support.backends import discovery
from localmelo.support.cli.views import BackendInfo
from localmelo.support.sessions.agents import build_session_agent

URL_LABEL: Final = "OpenAI-compatible"


def config_backend(cfg: config.Config, model: str | None) -> BackendInfo:
    """The configured chat backend, with *model* when one is given."""
    from localmelo.support.backends.registry import get_backend

    key = cfg.chat_backend
    section = getattr(cfg, key, None)
    try:
        label = get_backend(key).display_name
    except KeyError:
        label = key
    return BackendInfo(
        key=key,
        label=label,
        model=model or getattr(section, "chat_model", "") or "default",
        base_url=getattr(section, "chat_url", "") or getattr(section, "base_url", ""),
        embedding="" if not cfg.has_embedding else cfg.embedding_backend,
        key_env=getattr(section, "api_key_env", ""),
    )


def url_backend(
    url: str, model: str | None, *, source: str, listed: int = 0
) -> BackendInfo:
    """A server at *url* that is not the config's."""
    return BackendInfo(
        key=discovery.OPENAI_COMPATIBLE,
        label=URL_LABEL,
        model=model or "default",
        base_url=url,
        source=source,
        listed=listed,
    )


class AgentFactory:
    """Builds the session's Agents on its backend, and changes the backend."""

    def __init__(
        self,
        *,
        backend: BackendInfo,
        cfg: config.Config | None,
        options: AgentOptions,
        workspace: str,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        request_timeout_s: float | None = None,
        saves: bool = True,
    ) -> None:
        self.backend = backend
        self._cfg = cfg
        self._options = options
        self._workspace = workspace
        self._embedding_base_url = embedding_base_url
        self._embedding_model = embedding_model
        self._request_timeout_s = request_timeout_s
        # Whether /model and /backend save their choice as the default: not
        # for a --base-url session whose save question was answered no.
        self.saves = saves
        self._live: list[Agent] = []

    @property
    def memory(self) -> bool:
        return self._options.memory

    async def build(self, model: str | None = None, *, url: str | None = None) -> Agent:
        """A started Agent on the backend, with *model* and on *url* when given."""
        model = model or self.backend.model
        if url is not None or self._cfg is None:
            agent = await build_session_agent(
                cfg=None,
                base_url=url or self.backend.base_url,
                model=model,
                workspace=self._workspace,
                options=self._options,
                embedding_base_url=self._embedding_base_url,
                embedding_model=self._embedding_model,
                request_timeout_s=self._request_timeout_s,
            )
        else:
            agent = await build_session_agent(
                cfg=self._cfg,
                base_url=None,
                model=model,
                workspace=self._workspace,
                options=self._options,
                request_timeout_s=self._request_timeout_s,
            )
        self._live.append(agent)
        return agent

    async def close(self, agent: Agent) -> None:
        """Close *agent*, which this factory built."""
        with contextlib.suppress(ValueError):
            self._live.remove(agent)
        await agent.close()

    def live_agents(self) -> list[Agent]:
        """The Agents built and not yet closed (a switch may hold two)."""
        return list(self._live)

    def listable(self) -> bool:
        """Whether the backend's server can list its models (not a cloud vendor)."""
        return self.backend.local and bool(self.backend.base_url)

    async def list_models(self, url: str | None = None) -> list[str]:
        """The models the server lists; :class:`DiscoveryError` when it cannot."""
        key = discovery.OPENAI_COMPATIBLE if url else self.backend.key
        return await discovery.list_models(key, url or self.backend.base_url)

    def use_model(self, model: str) -> None:
        self.backend = replace(self.backend, model=model, listed=0)

    def use_url(self, url: str, model: str) -> None:
        """Talk to the server at *url* from now on, with *model*."""
        self._cfg = None
        self.backend = url_backend(
            url, model, source="saved" if self.saves else "--base-url"
        )

    def save(self) -> str:
        """Save the backend as the default; ``""``, or why it was not saved.

        A configured backend keeps its key and gets the new model; any other
        server is saved as an OpenAI-compatible one (``chat_backend =
        "vllm"``), with embedding off, as the first run saves it.
        """
        backend = self.backend
        if not self.saves:
            return "not saved (--base-url)"
        captured = io.StringIO()
        try:
            if backend.source == "config" and self._cfg is not None:
                cfg = config.load()
                section = getattr(cfg, cfg.chat_backend, None)
                if cfg.chat_backend != backend.key or section is None:
                    return "not saved: the config file changed"
                section.chat_model = backend.model
                config.save(cfg)
                self._cfg = cfg
                return ""
            with contextlib.redirect_stdout(captured):
                saved = onboard.save_chat_server(backend.base_url, backend.model)
        except (OSError, config.ConfigError) as exc:
            return f"not saved: {exc}"
        if not saved:
            return captured.getvalue().strip() or "not saved"
        self.backend = replace(backend, source="saved", listed=0)
        return ""

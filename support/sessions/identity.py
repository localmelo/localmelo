"""Who melo says it is in a chat, and which model it runs on.

A model that is told nothing answers "who are you?" with the identity it was
trained with. The chat surfaces (the terminal chat, the console and the app)
therefore give their Agent one paragraph for its system prompt
(``AgentOptions.identity``): melo from localmelo, the model it runs on, the
server that serves it and whether that server is on this computer. The
builders here make the paragraph from the backend an Agent is built on, so a
model switch (a new Agent) names the new model from the next turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from localmelo.support.backends import discovery
from localmelo.support.loopback import is_loopback

if TYPE_CHECKING:
    from localmelo.support.config import Config

# The chat backends of servers on this computer or on the user's network, as
# the terminal's banner calls them; every other backend is a cloud provider.
LOCAL_BACKENDS: Final = frozenset({"mlc", "ollama", "vllm", "sglang"})

IDENTITY_TEXT: Final = (
    "You are melo, the assistant built into localmelo: an open-source agent "
    "that runs on the user's own computer, works in their folder with tools, "
    "and asks before it changes anything.\n"
    "{running}\n"
    "If the user asks who or what you are, say that you are melo from "
    "localmelo and name {named}. The model only powers you: never present "
    "yourself as the model's own assistant or as made by its maker.\n"
    "Answer in the user's language. For a greeting or a question you can "
    "answer directly, reply without using tools."
)

# What the server's generic name reads as in a sentence.
_GENERIC_SERVER: Final = "an " + discovery.OPENAI_COMPATIBLE_NAME
_NAME_CHARS: Final = 120
_UNNAMED: Final = ("", "default")


def _clean(text: str) -> str:
    """*text* on one line and no longer than a model name needs to be."""
    return " ".join(text.split())[:_NAME_CHARS]


def host_of(url: str) -> str:
    """The host name of *url*, without a user, a password or a port; ``""`` if none."""
    try:
        return _clean(urlsplit(url).hostname or "")
    except ValueError:
        return ""


def identity_text(model: str | None, server: str, *, local: bool, url: str = "") -> str:
    """The identity paragraph for a chat on *model*, served by *server*.

    *model* is ``None``, empty or ``default`` when the model's name is not
    known; the paragraph then names the server only. *local* says the
    backend is a server of the user's (not a cloud provider): it is "on this
    computer" when *url* points to this computer or nowhere, and "at <host>"
    when it points elsewhere. A cloud provider has no place to name.
    """
    name = _clean(model or "")
    server = _clean(server)
    server = _GENERIC_SERVER if server == discovery.OPENAI_COMPATIBLE_NAME else server
    host = host_of(url) if local else ""
    where = ""
    if local:
        where = f" at {host}" if host and not is_loopback(host) else " on this computer"
    if name in _UNNAMED:
        running = f"You are running on a model served by {server}{where}."
        named = "the server that serves you"
    else:
        running = f"You are running on the model {name}, served by {server}{where}."
        named = "the model you run on"
    return IDENTITY_TEXT.format(running=running, named=named)


def url_identity(url: str, model: str | None) -> str:
    """The identity of a chat on the OpenAI-compatible server at *url*."""
    return identity_text(model, discovery.server_name(url), local=True, url=url)


def config_identity(cfg: Config) -> str:
    """The identity of a chat on *cfg*'s chat backend and model."""
    from localmelo.support.backends.registry import get_backend

    key = cfg.chat_backend
    section = getattr(cfg, key, None)
    try:
        server = get_backend(key).label(cfg)
    except KeyError:
        server = key
    url = getattr(section, "chat_url", "") or getattr(section, "base_url", "")
    return identity_text(
        getattr(section, "chat_model", ""), server, local=key in LOCAL_BACKENDS, url=url
    )

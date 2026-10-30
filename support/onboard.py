"""Interactive backend setup: the first-run questions and the full wizard."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from localmelo.support import config
from localmelo.support.backends import discovery
from localmelo.support.onboard_probe import ProbeError, probe_chat, probe_embedding


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val or default


def _ask_int(prompt: str, default: int, lo: int = 0, hi: int = 999999) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"    Must be between {lo} and {hi}")
        except ValueError:
            print("    Enter a number")


def _ask_choice(prompt: str, options: list[str], default: int = 1) -> int:
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    return _ask_int(prompt, default, lo=1, hi=len(options))


def _confirm(prompt: str = "Save and continue?") -> bool:
    val = _ask(prompt, "Y").lower()
    return val in ("y", "yes", "")


# ── Backend choice lists ──

# oMLX is the "vllm" backend, so its entry names it. A menu is drawn before any
# server is chosen, so it cannot tell oMLX from vLLM; a banner can (see
# discovery.server_name).
VLLM_LABEL = "oMLX or vLLM (OpenAI-compatible, local)"

CHAT_BACKENDS: list[tuple[str, str]] = [
    ("mlc", "MLC (local endpoint)"),
    ("ollama", "Ollama (local endpoint)"),
    ("vllm", VLLM_LABEL),
    ("sglang", "SGLang (local endpoint)"),
    ("openai", "OpenAI"),
    ("gemini", "Google Gemini"),
    ("anthropic", "Anthropic"),
    ("nvidia", "NVIDIA"),
]

EMBEDDING_BACKENDS: list[tuple[str, str]] = [
    ("mlc", "MLC (local endpoint)"),
    ("ollama", "Ollama (local endpoint)"),
    ("vllm", VLLM_LABEL),
    ("sglang", "SGLang (local endpoint)"),
    ("none", "None (no long-term memory)"),
]


# ── Chat backend handlers ──

_LOCAL_DEFAULTS: dict[str, tuple[str, str]] = {
    "mlc": ("http://localhost:8400/v1", "qwen3-1.7b"),
    "ollama": ("http://localhost:11434", "qwen3:8b"),
    "vllm": ("http://localhost:8000/v1", ""),
    "sglang": ("http://localhost:30000/v1", ""),
}

_CLOUD_DEFAULTS: dict[str, tuple[str, str]] = {
    "openai": ("OPENAI_API_KEY", "gpt-4o"),
    "gemini": ("GEMINI_API_KEY", "gemini-2.0-flash"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-4-20250514"),
    "nvidia": ("NVIDIA_API_KEY", "nvidia/llama-3.1-nemotron-70b-instruct"),
}


def _onboard_local(cfg: config.Config, key: str) -> bool:
    """Local (OpenAI-compatible) chat backend setup: URL and model name.

    The defaults are the section's current values, else the backend's.
    """
    default_url, default_model = _LOCAL_DEFAULTS.get(
        key, ("http://localhost:8000/v1", "")
    )
    section = getattr(cfg, key, None)
    if section is not None:
        default_url = section.chat_url or default_url
        default_model = section.chat_model or default_model

    name = "oMLX or vLLM" if key == "vllm" else key
    print(f"\n  -- {name} (local endpoint) --\n")

    url = _ask("Base URL", default_url)
    model = _ask("Chat model name", default_model)

    # Store in the matching config section
    if section is not None:
        section.chat_url = url
        section.chat_model = model

    print()
    return _confirm()


def _onboard_cloud(cfg: config.Config, key: str) -> bool:
    """Cloud vendor chat backend setup: API key env var and model name.

    The defaults are the section's current values, else the vendor's.
    """
    default_env, default_model = _CLOUD_DEFAULTS.get(key, ("", ""))

    print(f"\n  -- {key} (cloud API) --\n")

    section = getattr(cfg, key, None)
    if section is not None:
        section.api_key_env = _ask(
            "API key env var", section.api_key_env or default_env
        )
        section.chat_model = _ask("Chat model", section.chat_model or default_model)

    print()
    return _confirm()


# ── Chat backend setup dispatch ──

_CHAT_SETUP: dict[str, Callable[[config.Config], bool]] = {
    "mlc": lambda cfg: _onboard_local(cfg, "mlc"),
    "ollama": lambda cfg: _onboard_local(cfg, "ollama"),
    "vllm": lambda cfg: _onboard_local(cfg, "vllm"),
    "sglang": lambda cfg: _onboard_local(cfg, "sglang"),
    "openai": lambda cfg: _onboard_cloud(cfg, "openai"),
    "gemini": lambda cfg: _onboard_cloud(cfg, "gemini"),
    "anthropic": lambda cfg: _onboard_cloud(cfg, "anthropic"),
    "nvidia": lambda cfg: _onboard_cloud(cfg, "nvidia"),
}


# ── Embedding backend handlers ──

_EMBEDDING_DEFAULTS: dict[str, tuple[str, str]] = {
    "mlc": ("http://localhost:8400/v1", "nomic-embed"),
    "ollama": ("http://localhost:11434", "nomic-embed-text"),
    "vllm": ("http://localhost:8000/v1", ""),
    "sglang": ("http://localhost:30000/v1", ""),
}


def backend_defaults() -> dict[str, dict[str, str]]:
    """Default field values per backend section, as the setup wizard offers them."""
    defaults: dict[str, dict[str, str]] = {}
    for key, (chat_url, chat_model) in _LOCAL_DEFAULTS.items():
        emb_url, emb_model = _EMBEDDING_DEFAULTS.get(key, ("", ""))
        defaults[key] = {
            "chat_url": chat_url,
            "chat_model": chat_model,
            "embedding_url": emb_url,
            "embedding_model": emb_model,
        }
    for key, (api_key_env, chat_model) in _CLOUD_DEFAULTS.items():
        defaults[key] = {
            "api_key_env": api_key_env,
            "chat_model": chat_model,
        }
    return defaults


def _onboard_embedding_local(cfg: config.Config, key: str) -> None:
    """Set up a local embedding endpoint: URL and model name.

    The defaults are the section's current values, else the backend's.
    """
    default_url, default_model = _EMBEDDING_DEFAULTS.get(
        key, ("http://localhost:8000/v1", "")
    )
    section = getattr(cfg, key, None)
    if section is not None:
        default_url = section.embedding_url or default_url
        default_model = section.embedding_model or default_model

    url = _ask("Embedding endpoint URL", default_url)
    model = _ask("Embedding model name", default_model)

    if section is not None:
        section.embedding_url = url
        section.embedding_model = model


# ── Main setup flow ──


@dataclass(frozen=True)
class EmbeddingChoice:
    """An embedding model the chat server already lists.

    *key* is the chat backend's key (and so the embedding backend's),
    *url* the chat server's URL, *model* the listed id, and *label* the
    wizard's line for it.
    """

    key: str
    url: str
    model: str
    label: str


def _recommended_order(candidates: list[str]) -> list[str]:
    """*candidates* in :func:`discovery.recommend_embedding` order."""
    remaining = list(candidates)
    ordered: list[str] = []
    while remaining:
        candidate = discovery.recommend_embedding(remaining)
        assert candidate is not None  # remaining is not empty
        ordered.append(candidate)
        remaining.remove(candidate)
    return ordered


def _embedding_choices(cfg: config.Config, models: list[str]) -> list[EmbeddingChoice]:
    """The chat server's embedding models, as wizard options; pure.

    Only a local chat backend with a URL can be listed. The order is the
    recommendation order, and the label names the server so a candidate is
    clear next to the generic local-endpoint options.
    """
    key = cfg.chat_backend
    if key not in _EMBEDDING_DEFAULTS:
        return []
    section = getattr(cfg, key, None)
    url = getattr(section, "chat_url", "")
    if not url:
        return []
    return [
        EmbeddingChoice(
            key=key,
            url=url,
            model=model,
            label=f"{model} on this server ({url})",
        )
        for model in _recommended_order(discovery.embedding_models(models))
    ]


class _FullMenuError(Exception):
    """The guided reconfigure found no server to keep: show the full menu.

    *found* is the offered server the user declined, if one was offered.
    """

    def __init__(self, found: discovery.FoundServer | None = None) -> None:
        super().__init__()
        self.found = found


def run_backend_setup(*, guided: bool = False) -> config.Config | None:
    """Run the split backend setup flow.

    With *guided* (default mode's ``--reconfigure``), a model server found on
    this machine is offered first, as the first run does; the full menu below
    follows only when none is found or the user declines it.

    1. Choose and configure the chat backend.
    2. Choose and configure the embedding backend.
    3. Check them, then save and return the Config.

    Every question defaults to the current config's value; the embedding
    backend defaults to none when the config has none. Returns Config if
    successful, None if cancelled.
    """

    declined: discovery.FoundServer | None = None
    if guided:
        try:
            setup = run_first_setup(ask_keep=True)
        except _FullMenuError as menu:
            declined = menu.found
        else:
            return config.load() if setup is not None and setup.saved else None

    print("\n  Welcome to localmelo.\n")
    cfg = config.load()

    # ── 1. Chat backend ──
    chat_keys = [key for key, _ in CHAT_BACKENDS]
    chat_display = [name for _, name in CHAT_BACKENDS]
    # The default is the saved backend; with none saved, the declined server's
    # kind, as the first run would save it (see _with_chat_server).
    default_chat = cfg.chat_backend
    if default_chat not in chat_keys and declined is not None:
        ollama = asyncio.run(discovery.is_ollama(declined.url))
        default_chat = "ollama" if ollama else "vllm"
    current_chat = chat_keys.index(default_chat) + 1 if default_chat in chat_keys else 1
    choice = _ask_choice("Chat backend", chat_display, default=current_chat)
    selected_chat_key = CHAT_BACKENDS[choice - 1][0]

    handler = _CHAT_SETUP.get(selected_chat_key)
    if handler is None:
        print(
            f"  Backend '{selected_chat_key}' has no onboarding handler yet. "
            f"Cannot continue."
        )
        return None

    ok = handler(cfg)

    if not ok:
        print("  Setup cancelled.")
        return None

    # ── 2. Embedding backend ──
    # The chat backend is needed before the embedding question: its own
    # server's listed models become extra options.
    cfg.chat_backend = selected_chat_key
    emb_keys = [key for key, _ in EMBEDDING_BACKENDS]
    choices: list[EmbeddingChoice] = []
    if cfg.has_embedding:
        # A working embedding setup keeps today's menu and defaults, and
        # sends no listing: the saved model may not be on the chat server.
        current_emb = (
            cfg.embedding_backend if cfg.embedding_backend in emb_keys else "none"
        )
        default_emb = emb_keys.index(current_emb) + 1
    else:
        models: list[str] = []
        chat_section = getattr(cfg, selected_chat_key, None)
        chat_url = getattr(chat_section, "chat_url", "")
        if selected_chat_key in _EMBEDDING_DEFAULTS and chat_url:
            try:
                models = asyncio.run(discovery.list_models(selected_chat_key, chat_url))
            except discovery.DiscoveryError:
                models = []  # the end-of-wizard probe still reports the problem
        choices = _embedding_choices(cfg, models)
        default_emb = len(emb_keys) + 1 if choices else emb_keys.index("none") + 1
    emb_display = [name for _, name in EMBEDDING_BACKENDS] + [
        choice.label for choice in choices
    ]
    print("\n  Embedding model for long-term memory:\n")
    emb_choice = _ask_choice("Embedding backend", emb_display, default=default_emb)

    picked_choice: EmbeddingChoice | None = None
    old_embedding = ("", "")
    if emb_choice > len(emb_keys):
        picked = choices[emb_choice - len(emb_keys) - 1]
        picked_choice = picked
        selected_emb_key = picked.key
        cfg.embedding_backend = selected_emb_key
        section = getattr(cfg, selected_emb_key)
        old_embedding = (section.embedding_url, section.embedding_model)
        section.embedding_url = picked.url
        section.embedding_model = picked.model
    else:
        selected_emb_key = EMBEDDING_BACKENDS[emb_choice - 1][0]
        if selected_emb_key != "none":
            _onboard_embedding_local(cfg, selected_emb_key)
        else:
            print("\n  No embedding: agent will not have long-term memory.")

    cfg.embedding_backend = selected_emb_key

    if not _run_probes(cfg, retry=lambda: handler(cfg)):
        return None
    if picked_choice is not None and cfg.embedding_backend == "none":
        # The detected model failed its probe: never leave it in the saved
        # file just because it was listed. Restore what the table had.
        section = getattr(cfg, picked_choice.key)
        section.embedding_url, section.embedding_model = old_embedding

    _spell_omlx(cfg)
    config.save(cfg)
    print(f"\n  Config saved to {_home_path(config.CONFIG_PATH)}")
    return cfg


_SAVING_EMBEDDING_OFF = (
    "Saving with embedding off: no long-term memory " "(melo setup sets it again)."
)
_EMBEDDING_OFF = "Long-term memory is off; melo setup turns it on."
_NO_EMBEDDING_OLLAMA = (
    "No embedding model on this server. ollama pull qwen3-embedding:0.6b, "
    "then melo setup."
)
_NO_EMBEDDING_LOCAL = (
    "No embedding model on this server. Serve Qwen3-Embedding-0.6B on it, "
    "then melo setup."
)


def _print_probe_failed(exc: ProbeError, *, blank_line: bool = True) -> None:
    """The failed-check lines, shared by first run and the wizard."""
    if blank_line:
        print()
    _say(f"Probe failed: {exc}", indent=2)
    _say(_SAVING_EMBEDDING_OFF, indent=2)


def _run_probes(cfg: config.Config, retry: Callable[[], bool] | None = None) -> bool:
    """Check the chat and embedding backends before saving.

    Returns True if config should be saved. When the chat check fails, the
    user may retry, which calls *retry* to ask again (its False cancels),
    save anyway, or cancel, the default. A failed embedding check saves
    with embedding off, so long-term memory never blocks the chat setup.
    """
    while True:
        print("\n  Probing backends ...")
        try:
            asyncio.run(probe_chat(cfg))
        except ProbeError as exc:
            print(f"\n  Probe failed: {exc}")
            if _wants_a_key(cfg, exc):
                print(f"  Note: {discovery.NO_API_KEY_HINT}.")
            if retry is not None:
                choice = _choose("  [r]etry, [s]ave anyway, [N]o [N]: ", "rsn", "n")
            else:
                choice = _choose("  [s]ave anyway, [N]o [N]: ", "sn", "n")
            if choice == "r" and retry is not None:
                if not retry():
                    print("  Setup cancelled — config not saved.")
                    return False
                continue
            if choice != "s":
                print("  Setup cancelled — config not saved.")
                return False
            print("  Saving despite failed probe.")
            break
        else:
            break
    try:
        asyncio.run(probe_embedding(cfg))
    except ProbeError as exc:
        _print_probe_failed(exc)
        cfg.embedding_backend = "none"
        return True
    print("  Backends OK.")
    return True


def _wants_a_key(cfg: config.Config, exc: ProbeError) -> bool:
    """Whether a local chat server refused the check for want of an API key."""
    status = getattr(exc.__cause__, "status_code", None)
    return status in (401, 403) and cfg.chat_backend in _LOCAL_DEFAULTS


# ── First run: a URL and a model ──


@dataclass(frozen=True)
class FirstSetup:
    """The model server the first run chose, and whether it was saved."""

    url: str
    model: str
    saved: bool


_FIRST_RUN_URL = "http://127.0.0.1:8000/v1"


def _question(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def _choose(prompt: str, letters: str, default: str, *, invalid: str = "") -> str:
    """Ask *prompt* until the answer's first letter is one of *letters*.

    Enter chooses *default*; a whole word (``retry``, ``save anyway``)
    counts by its first letter. *invalid* is what an unknown answer is told,
    where the prompt names its keys otherwise than by their letters.
    """
    while True:
        answer = input(prompt).strip().lower() or default
        if answer[0] in letters:
            return answer[0]
        print(invalid or f"  Type {', '.join(letters[:-1])} or {letters[-1]}.")


def _home_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _columns() -> int | None:
    """The terminal's width in cells; None when output is not a terminal."""
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except (OSError, ValueError):
        return None


def _say(text: str, *, indent: int = 0, hang: int = 0) -> None:
    """Print *text* wrapped at word boundaries to the terminal's width.

    Continuation rows start *indent* + *hang* cells in. A word longer than a
    row (a URL) stays whole. Without a terminal the text is printed as it
    is, one line for a pipe or a log.
    """
    width = _columns()
    if width is None:
        print(" " * indent + text)
        return
    print(
        "\n".join(
            textwrap.wrap(
                text,
                max(width - 1, 20),
                initial_indent=" " * indent,
                subsequent_indent=" " * (indent + hang),
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    )


def _port(url: str) -> str:
    tail = url.split("://", 1)[-1].split("/", 1)[0]
    return tail.rsplit(":", 1)[1] if ":" in tail else tail


def _wait_for_a_server(urls: Sequence[str]) -> discovery.FoundServer | str | None:
    """Say how to start a model server, then look again each time Enter is pressed.

    Returns the server a look found, or the address typed in its place (the
    caller checks it), or None when the user quits. EOFError and
    KeyboardInterrupt reach the caller.
    """
    how = discovery.how_to_start()
    _say(f"No model server is running yet. To start one with {how.name}:")
    print()
    for command in how.commands:
        print(f"  {command}")
    print()
    _say(f"{how.note} The guide shows how:")
    print(discovery.GUIDE_URL)
    while True:
        answer = input(
            "Press Enter to check again, type an address, or q to quit: "
        ).strip()
        if answer.lower() in ("q", "quit"):
            return None
        if answer:
            return answer
        found = asyncio.run(discovery.detect_local_server(urls))
        if found is not None:
            return found
        ports = ", ".join(_port(url) for url in urls)
        _say(f"✗ No model server found yet (looked at ports {ports}).", hang=2)


def _servers_named() -> str:
    """Examples of a model server, for the line that says what the first run needs."""
    mac = sys.platform == "darwin"
    return "oMLX, Ollama, LM Studio…" if mac else "Ollama, LM Studio, vLLM…"


class ModelChoiceError(Exception):
    """Several models are served, none is known to be loaded, and none was named."""


def _arrow_keys() -> bool:
    """Whether stdin and stdout are a terminal that an arrow-key list can draw on."""
    try:
        ttys = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False
    return ttys and os.environ.get("TERM") != "dumb"


def _pick_model(
    url: str, models: list[str], loaded: discovery.LoadedModels, default: str | None
) -> str:
    """Choose from *models* with the arrow-key list; a close cancels the setup."""
    from localmelo.support.cli.picker import ModelPicker, ModelRow, run_inline
    from localmelo.support.cli.theme import symbols_for, wants_no_color

    rows = [
        ModelRow(model, model in loaded.ids, discovery.model_kind(model))
        for model in models
    ]
    chosen = run_inline(
        lambda width, invalidate: ModelPicker(
            rows,
            where=url,
            start=default,
            title="Choose a model",
            width=width,
            invalidate=invalidate,
            symbols=symbols_for(sys.stdout.encoding),
        ),
        no_color=wants_no_color(),
    )
    if chosen is None:
        raise KeyboardInterrupt
    return chosen.id


def _choose_model(url: str, models: list[str]) -> str:
    """The model to use: asked from *models*, unless it is the one chat model.

    Chat models are listed first, an embedding or reranking model is tagged
    and is never the default. With several chat models the default is the
    one the server says is loaded; if it does not say, there is no default.
    On a terminal the list is chosen from with the arrow keys (embedding and
    reranking models left out when there is a chat model; with no default
    nothing is highlighted until a key moves). Otherwise the models are
    numbered, Enter alone asks again, and without a terminal an empty answer
    raises :class:`ModelChoiceError`.
    """
    chat = [model for model in models if discovery.model_kind(model) == "chat"]
    if len(models) == 1 and chat:
        return models[0]
    listed = chat + [model for model in models if model not in chat]
    # Only a choice between chat models needs to know which are in memory.
    loaded = (
        asyncio.run(discovery.loaded_models(url))
        if len(chat) > 1
        else discovery.LoadedModels()
    )
    arrows = _arrow_keys()
    if not arrows:
        print(f"Models on {url}:")
        for i, model in enumerate(listed, 1):
            tags = [] if model in chat else [discovery.model_kind(model)]
            tags += ["loaded"] if model in loaded.ids else []
            suffix = f"  ({', '.join(tags)})" if tags else ""
            print(f"  [{i}] {model}{suffix}")
    if not chat:
        _say(
            "These look like embedding or reranking models; chat needs a chat model.",
            indent=2,
        )
    pick = discovery.pick_model(chat or listed, loaded)
    default = str(listed.index(pick) + 1) if pick else ""
    if len(chat) > 1 and not default and loaded.needs_key:
        _say(discovery.NO_STATUS_KEY_NOTE, indent=2)
    if arrows:
        if len(chat) > 1 and pick is None:
            _say(discovery.LOAD_COST_NOTE, indent=2)
        return _pick_model(url, chat or listed, loaded, pick)
    while True:
        raw = _question("Model", default)
        if not raw:
            if not sys.stdin.isatty():
                raise ModelChoiceError(discovery.NO_DEFAULT_MODEL)
            _say(discovery.NO_DEFAULT_HINT, indent=2)
        elif raw.isdigit() and 1 <= int(raw) <= len(listed):
            return listed[int(raw) - 1]
        elif raw in models:
            return raw
        else:
            _say(f"Type a number from 1 to {len(listed)}, or a model id.", indent=2)


def _spell_omlx(cfg: config.Config) -> None:
    """Spell a ``vllm`` backend ``omlx`` where a model listing named oMLX.

    The first run, ``/backend`` and the wizard list the server they save, so
    the file says oMLX and every later banner does, with no request of its own.
    """
    if cfg.chat_backend == "vllm" and discovery.is_omlx(cfg.vllm.chat_url):
        cfg.chat_alias = config.OMLX_ALIAS
    if cfg.embedding_backend == "vllm" and discovery.is_omlx(
        cfg.vllm.embedding_url or cfg.vllm.chat_url
    ):
        cfg.embedding_alias = config.OMLX_ALIAS


def _with_chat_server(
    cfg: config.Config,
    url: str,
    model: str,
    *,
    ollama: bool,
    embedding_model: str | None = None,
) -> config.Config:
    """*cfg* with its chat on *url*, as :func:`save_chat_server` writes it.

    *ollama* makes the chat backend ``ollama``; otherwise it is ``vllm``, spelled
    ``omlx`` when a listing of *url* named oMLX. With *embedding_model*, the same
    table's embedding URL and model are set too, so a first run can probe exactly
    what it will save. A file's ``omlx`` spelling stays while the chat server is
    the same. Nothing is written to disk.
    """
    if ollama or cfg.chat_backend != "vllm" or cfg.vllm.chat_url != url:
        cfg.chat_alias = ""  # the file's "omlx" named another server
    if ollama:
        cfg.chat_backend = "ollama"
        cfg.ollama.chat_url = url
        cfg.ollama.chat_model = model
    else:
        cfg.chat_backend = "vllm"
        cfg.vllm.chat_url = url
        cfg.vllm.chat_model = model
    if embedding_model is not None:
        cfg.embedding_alias = ""
        cfg.embedding_backend = cfg.chat_backend
        embedding_section = getattr(cfg, cfg.chat_backend)
        embedding_section.embedding_url = url
        embedding_section.embedding_model = embedding_model
    _spell_omlx(cfg)
    return cfg


async def save_chat_server(
    url: str,
    model: str,
    *,
    keep_embedding: bool = False,
    ollama: bool | None = None,
    embedding_model: str | None = None,
) -> bool:
    """Save *url* and *model* as the default chat backend; False if it failed.

    An Ollama server (it answers ``GET /api/version``) is saved as the
    ``ollama`` backend in the ``[ollama]`` table; the backend chats on
    Ollama's native API with its 300 s timeout. Any other OpenAI-compatible
    server is saved as ``vllm``, spelled ``omlx`` when a listing named oMLX.
    *ollama* is the answer to that question when the caller asked it
    already. With *embedding_model*, the chat backend's table also gets that
    embedding URL and model (the first-run choice). Otherwise the first run
    turns embedding off; with *keep_embedding* (a server chosen in a
    configured session) the embedding backend stays. The config's other
    values are kept.
    """
    cfg = config.load()
    if ollama is None:
        ollama = await discovery.is_ollama(url)
    _with_chat_server(cfg, url, model, ollama=ollama, embedding_model=embedding_model)
    if embedding_model is None and not keep_embedding:
        cfg.embedding_backend = "none"
    try:
        config.save(cfg)
    except (OSError, config.ConfigError) as exc:
        _say(f"Could not save {_home_path(config.CONFIG_PATH)}: {exc}")
        return False
    _say(f"Saved to {_home_path(config.CONFIG_PATH)} (melo setup changes it).")
    return True


def _offer_embedding(
    url: str, model: str, models: list[str], *, ollama: bool | None, first: bool = False
) -> tuple[str | None, bool | None]:
    """Ask whether to use the server's embedding model; the model and Ollama answer.

    *ollama* is the answer to ``GET /api/version`` when the caller already
    asked it, else None. Returns ``(embedding_model, ollama)``: the model
    to save, or None for embedding off, and the Ollama answer to pass on so
    the first run asks the server only once. With no candidate, a *first*
    run says only that memory is off; ``melo setup`` says how to add a model.
    """
    candidates = [
        candidate
        for candidate in discovery.embedding_models(models)
        if candidate != model
    ]
    if not candidates:
        if first:
            _say(_EMBEDDING_OFF)
            return None, ollama
        if ollama is None:
            ollama = asyncio.run(discovery.is_ollama(url))
        _say(_NO_EMBEDDING_OLLAMA if ollama else _NO_EMBEDDING_LOCAL)
        return None, ollama
    candidate = discovery.recommend_embedding(candidates)
    assert candidate is not None  # candidates is not empty
    answer = _choose(f"Use {candidate} for long-term memory? [Y/n]: ", "yn", "y")
    if answer == "n":
        _say(_EMBEDDING_OFF)
        return None, ollama
    if ollama is None:
        ollama = asyncio.run(discovery.is_ollama(url))
    probe_cfg = _with_chat_server(
        config.load(), url, model, ollama=ollama, embedding_model=candidate
    )
    try:
        asyncio.run(probe_embedding(probe_cfg))
    except ProbeError as exc:
        _print_probe_failed(exc, blank_line=False)
        return None, ollama
    return candidate, ollama


def run_first_setup(*, ask_keep: bool = False) -> FirstSetup | None:
    """Ask for a model server's URL and model, check them, and save them.

    The first line says what the first run needs: a model server. A server
    found on this machine is offered with one question (``n`` asks for a URL,
    prefilled with it); the model is chosen from the server's list. With none found, it prints the commands that start one
    and looks again on each Enter; an address typed instead is checked as a
    URL is. Only ``GET <url>/models`` is sent, and ``GET /api/version`` once
    when saving, to tell Ollama apart. When the check of a URL fails the user
    may retry, start without saving, or quit (None). EOFError and
    KeyboardInterrupt reach the caller.

    With *ask_keep* (``--reconfigure``), a ``n``, or no server found while a
    backend is saved, raises :class:`_FullMenuError` before anything is saved,
    and a model server without an embedding model is told how to get one.
    """
    urls = discovery.detect_urls()
    # A reconfigure knows what melo needs; a first run says it, also when it
    # reaches this question through `melo serve`.
    first = not ask_keep or not config.load().is_configured
    if first:
        _say(f"melo uses a model server on this computer ({_servers_named()}).")
    found = asyncio.run(discovery.detect_local_server(urls)) if urls else None
    typed = ""
    if found is None and urls and first:
        waited = _wait_for_a_server(urls)
        if waited is None:
            return None
        if isinstance(waited, str):
            typed = waited
        else:
            found = waited
    if ask_keep and not typed and (found is None or not found.models):
        raise _FullMenuError()
    keep = bool(typed)
    default_url = typed
    if found is not None and found.models:
        count = len(found.models)
        _say(
            f"Found a model server at {found.url} "
            f"({count} model{'' if count == 1 else 's'})."
        )
        default_url = found.url
        if _choose("Use it? [Y/n]: ", "yn", "y") == "y":
            keep = True
        elif ask_keep:
            raise _FullMenuError(found)  # a first run asks for another URL below
    elif found is not None:
        _say(f"A model server at {found.url} answered, but {found.problem}.")
        default_url = found.url
    elif not typed:
        default_url = _FIRST_RUN_URL  # detection is off: the usual address is a guess

    while True:
        url = default_url if keep else _question("Server URL", default_url)
        keep = False
        try:
            if found is not None and found.models and url == found.url:
                models = found.models
            else:
                models = asyncio.run(
                    discovery.list_models(discovery.OPENAI_COMPATIBLE, url)
                )
            reason = "" if models else "it lists no models"
        except discovery.DiscoveryError as exc:
            reason = str(exc)
        if not reason:
            break
        _say(f"✗ {url}: {reason}.", hang=2)
        choice = _choose(
            "Enter tries again · s starts without saving · q quits: ",
            "rsq",
            "r",
            invalid="  Press Enter, s or q.",
        )
        if choice == "q":
            return None
        if choice == "s":
            model = _question("Model name", "default")
            _say("Not saved: melo asks again next time.")
            return FirstSetup(url=url, model=model, saved=False)
        default_url = url

    model = _choose_model(url, models)
    count = len(models)
    _say(
        f"✓ {url} lists {count} model{'' if count == 1 else 's'}; using {model}.",
        hang=2,
    )
    embedding_model, ollama = _offer_embedding(
        url, model, models, ollama=None, first=first
    )
    saved = asyncio.run(
        save_chat_server(
            url,
            model,
            ollama=ollama,
            embedding_model=embedding_model,
        )
    )
    return FirstSetup(url=url, model=model, saved=saved)


def offer_to_save(url: str, model: str) -> bool:
    """Ask once whether to save *url* and *model* as the default backend."""
    if (
        _choose(f"Save {url} · {model} as the default backend? [Y/n] ", "yn", "y")
        == "n"
    ):
        return False
    return asyncio.run(save_chat_server(url, model))

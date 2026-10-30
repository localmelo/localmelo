"""Interactive backend setup: the first-run questions and the full wizard."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
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

CHAT_BACKENDS: list[tuple[str, str]] = [
    ("mlc", "MLC (local endpoint)"),
    ("ollama", "Ollama (local endpoint)"),
    ("vllm", "vLLM (local endpoint)"),
    ("sglang", "SGLang (local endpoint)"),
    ("openai", "OpenAI"),
    ("gemini", "Google Gemini"),
    ("anthropic", "Anthropic"),
    ("nvidia", "NVIDIA"),
]

EMBEDDING_BACKENDS: list[tuple[str, str]] = [
    ("mlc", "MLC (local endpoint)"),
    ("ollama", "Ollama (local endpoint)"),
    ("vllm", "vLLM (local endpoint)"),
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

    print(f"\n  -- {key} (local endpoint) --\n")

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

    config.save(cfg)
    print(f"\n  Config saved to {_home_path(config.CONFIG_PATH)}")
    return cfg


_SAVING_EMBEDDING_OFF = (
    "Saving with embedding off: no long-term memory " "(melo setup sets it again)."
)
_EMBEDDING_OFF = "Long-term memory off (melo setup sets it again)."
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
    prefix = "\n" if blank_line else ""
    print(f"{prefix}  Probe failed: {exc}")
    print(f"  {_SAVING_EMBEDDING_OFF}")


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


def _choose(prompt: str, letters: str, default: str) -> str:
    """Ask *prompt* until the answer's first letter is one of *letters*.

    Enter chooses *default*; a whole word (``retry``, ``save anyway``)
    counts by its first letter.
    """
    while True:
        answer = input(prompt).strip().lower() or default
        if answer[0] in letters:
            return answer[0]
        print(f"  Type {', '.join(letters[:-1])} or {letters[-1]}.")


def _home_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _port(url: str) -> str:
    tail = url.split("://", 1)[-1].split("/", 1)[0]
    return tail.rsplit(":", 1)[1] if ":" in tail else tail


class ModelChoiceError(Exception):
    """Several models are served, none is known to be loaded, and none was named."""


def _choose_model(url: str, models: list[str]) -> str:
    """The model to use: asked from *models*, unless it is the one chat model.

    With several chat models the default is the one the server says is
    loaded; if it does not say, there is no default and Enter asks again.
    Without a terminal an empty answer then raises :class:`ModelChoiceError`.
    """
    chat = [model for model in models if discovery.is_chat_model(model)]
    if len(models) == 1 and chat:
        return models[0]
    # Only a choice between chat models needs to know which are in memory.
    loaded = (
        asyncio.run(discovery.loaded_models(url))
        if len(chat) > 1
        else discovery.LoadedModels()
    )
    print(f"Models on {url}:")
    for i, model in enumerate(models, 1):
        print(f"  [{i}] {model}{'  (loaded)' if model in loaded.ids else ''}")
    if not chat:
        print(
            "  These look like embedding or reranking models; chat needs a chat model."
        )
    pick = discovery.pick_model(models, loaded)
    default = str(models.index(pick) + 1) if pick else ""
    if len(chat) > 1 and not default and loaded.needs_key:
        print(f"  {discovery.NO_STATUS_KEY_NOTE}")
    while True:
        raw = _question("Model", default)
        if not raw:
            if not sys.stdin.isatty():
                raise ModelChoiceError(discovery.NO_DEFAULT_MODEL)
            print(f"  {discovery.NO_DEFAULT_HINT}")
        elif raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        elif raw in models:
            return raw
        else:
            print(f"  Type a number from 1 to {len(models)}, or a model id.")


def _with_chat_server(
    cfg: config.Config,
    url: str,
    model: str,
    *,
    ollama: bool,
    embedding_model: str | None = None,
) -> config.Config:
    """*cfg* with its chat on *url*, as :func:`save_chat_server` writes it.

    *ollama* makes the chat backend ``ollama``; otherwise it is ``vllm``.
    With *embedding_model*, the same table's embedding URL and model are
    set too, so a first run can probe exactly what it will save. Nothing
    is written to disk.
    """
    if ollama:
        cfg.chat_backend = "ollama"
        cfg.ollama.chat_url = url
        cfg.ollama.chat_model = model
    else:
        cfg.chat_backend = "vllm"
        cfg.vllm.chat_url = url
        cfg.vllm.chat_model = model
    if embedding_model is not None:
        cfg.embedding_backend = cfg.chat_backend
        embedding_section = getattr(cfg, cfg.chat_backend)
        embedding_section.embedding_url = url
        embedding_section.embedding_model = embedding_model
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
    server is saved as ``vllm``. *ollama* is the answer to that question
    when the caller asked it already. With *embedding_model*, the chat
    backend's table also gets that embedding URL and model (the first-run
    choice). Otherwise the first run turns embedding off; with
    *keep_embedding* (a server chosen in a configured session) the
    embedding backend stays. The config's other values are kept.
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
        print(f"Could not save {_home_path(config.CONFIG_PATH)}: {exc}")
        return False
    print(f"Saved to {_home_path(config.CONFIG_PATH)} (melo setup changes it).")
    return True


def _offer_embedding(
    url: str, model: str, models: list[str], *, ollama: bool | None
) -> tuple[str | None, bool | None]:
    """Ask whether to use the server's embedding model; the model and Ollama answer.

    *ollama* is the answer to ``GET /api/version`` when the caller already
    asked it, else None. Returns ``(embedding_model, ollama)``: the model
    to save, or None for embedding off, and the Ollama answer to pass on so
    the first run asks the server only once.
    """
    candidates = [
        candidate
        for candidate in discovery.embedding_models(models)
        if candidate != model
    ]
    if not candidates:
        if ollama is None:
            ollama = asyncio.run(discovery.is_ollama(url))
        print(_NO_EMBEDDING_OLLAMA if ollama else _NO_EMBEDDING_LOCAL)
        return None, ollama
    candidate = discovery.recommend_embedding(candidates)
    assert candidate is not None  # candidates is not empty
    answer = _choose(f"Use {candidate} for long-term memory? [Y/n]: ", "yn", "y")
    if answer == "n":
        print(_EMBEDDING_OFF)
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

    A server found on this machine prefills the URL; the model is chosen
    from the server's list. Only ``GET <url>/models`` is sent, and
    ``GET /api/version`` once when saving, to tell Ollama apart. When the
    check fails the user may retry, start without saving, or quit
    (None). EOFError and KeyboardInterrupt reach the caller.

    With *ask_keep* (``--reconfigure``), the found server is offered with one
    question instead of the URL question; no server found, or a ``n``,
    raises :class:`_FullMenuError` before anything is saved.
    """
    urls = discovery.detect_urls()
    found = asyncio.run(discovery.detect_local_server(urls)) if urls else None
    if ask_keep and (found is None or not found.models):
        raise _FullMenuError()
    keep = False
    if found is not None and found.models:
        count = len(found.models)
        print(
            f"Found a model server at {found.url} "
            f"({count} model{'' if count == 1 else 's'})."
        )
        default_url = found.url
        if ask_keep:
            if _choose("Use it? [Y/n]: ", "yn", "y") == "n":
                raise _FullMenuError(found)
            keep = True
    elif found is not None:
        print(f"A model server at {found.url} answered, but {found.problem}.")
        default_url = found.url
    else:
        if urls:
            ports = ", ".join(_port(url) for url in urls)
            print(
                f"No model server found on this computer (looked at ports {ports}). "
                "Start one (for example ollama serve), or enter its URL."
            )
        # URLs named in the environment are the ones to try first.
        overridden = discovery.DETECT_URLS_ENV in os.environ
        default_url = urls[0] if overridden and urls else _FIRST_RUN_URL

    while True:
        url = default_url if keep else _question("Model server URL", default_url)
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
        print(f"✗ {url}: {reason}.")
        choice = _choose("[r]etry, [s]tart without saving, [q]uit [r]: ", "rsq", "r")
        if choice == "q":
            return None
        if choice == "s":
            model = _question("Model name", "default")
            print("Not saved: melo asks again next time.")
            return FirstSetup(url=url, model=model, saved=False)
        default_url = url

    model = _choose_model(url, models)
    count = len(models)
    print(f"✓ {url} lists {count} model{'' if count == 1 else 's'}; using {model}.")
    embedding_model, ollama = _offer_embedding(url, model, models, ollama=None)
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

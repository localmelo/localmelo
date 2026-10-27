"""Interactive backend setup: the first-run questions and the full wizard."""

from __future__ import annotations

import asyncio
import os
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


def _confirm(prompt: str = "Start?") -> bool:
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


def run_backend_setup() -> config.Config | None:
    """Run the split backend setup flow.

    1. Choose and configure the chat backend.
    2. Choose and configure the embedding backend.
    3. Check them, then save and return the Config.

    Every question defaults to the current config's value; the embedding
    backend defaults to none when the config has none. Returns Config if
    successful, None if cancelled.
    """

    print("\n  Welcome to localmelo.\n")
    cfg = config.load()

    # ── 1. Chat backend ──
    chat_keys = [key for key, _ in CHAT_BACKENDS]
    chat_display = [name for _, name in CHAT_BACKENDS]
    current_chat = (
        chat_keys.index(cfg.chat_backend) + 1 if cfg.chat_backend in chat_keys else 1
    )
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
    emb_keys = [key for key, _ in EMBEDDING_BACKENDS]
    emb_display = [name for _, name in EMBEDDING_BACKENDS]
    current_emb = cfg.embedding_backend if cfg.embedding_backend in emb_keys else "none"
    emb_choice = _ask_choice(
        "Embedding backend", emb_display, default=emb_keys.index(current_emb) + 1
    )
    selected_emb_key = EMBEDDING_BACKENDS[emb_choice - 1][0]

    if selected_emb_key != "none":
        _onboard_embedding_local(cfg, selected_emb_key)
    else:
        print("\n  No embedding: agent will not have long-term memory.")

    cfg.chat_backend = selected_chat_key
    cfg.embedding_backend = selected_emb_key

    if not _run_probes(cfg, retry=lambda: handler(cfg)):
        return None

    config.save(cfg)
    print(f"\n  Config saved to {_home_path(config.CONFIG_PATH)}")
    return cfg


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
        print(f"\n  Probe failed: {exc}")
        print(
            "  Saving with embedding off: no long-term memory "
            "(melo --reconfigure sets it again)."
        )
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


def _choose_model(url: str, models: list[str]) -> str:
    """The model to use: asked from *models*, unless it is the one chat model."""
    chat = [model for model in models if discovery.is_chat_model(model)]
    if len(models) == 1 and chat:
        return models[0]
    print(f"Models on {url}:")
    for i, model in enumerate(models, 1):
        print(f"  [{i}] {model}")
    if not chat:
        print(
            "  These look like embedding or reranking models; chat needs a chat model."
        )
    default = models.index(discovery.pick_model(models) or models[0]) + 1
    while True:
        raw = _question("Model", str(default))
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        if raw in models:
            return raw
        print(f"  Type a number from 1 to {len(models)}, or a model id.")


def save_chat_server(url: str, model: str) -> bool:
    """Save *url* and *model* as the default chat backend; False if it failed.

    Any OpenAI-compatible server is saved as the ``vllm`` backend, with
    embedding off. The config's other values are kept.
    """
    cfg = config.load()
    cfg.chat_backend = "vllm"
    cfg.vllm.chat_url = url
    cfg.vllm.chat_model = model
    cfg.embedding_backend = "none"
    try:
        config.save(cfg)
    except (OSError, config.ConfigError) as exc:
        print(f"Could not save {_home_path(config.CONFIG_PATH)}: {exc}")
        return False
    print(f"Saved to {_home_path(config.CONFIG_PATH)} (melo --reconfigure changes it).")
    return True


def run_first_setup() -> FirstSetup | None:
    """Ask for a model server's URL and model, check them, and save them.

    A server found on this machine prefills the URL; the model is chosen
    from the server's list. Only ``GET <url>/models`` is sent. When the
    check fails the user may retry, start anyway without saving, or quit
    (None). EOFError and KeyboardInterrupt reach the caller.
    """
    urls = discovery.detect_urls()
    found = asyncio.run(discovery.detect_local_server(urls)) if urls else None
    if found is not None and found.models:
        count = len(found.models)
        print(
            f"Found a model server at {found.url} "
            f"({count} model{'' if count == 1 else 's'})."
        )
        default_url = found.url
    elif found is not None:
        print(f"A model server at {found.url} answered, but {found.problem}.")
        default_url = found.url
    else:
        if urls:
            ports = ", ".join(_port(url) for url in urls)
            print(f"No model server answered on this machine (ports {ports}).")
        # URLs named in the environment are the ones to try first.
        overridden = discovery.DETECT_URLS_ENV in os.environ
        default_url = urls[0] if overridden and urls else _FIRST_RUN_URL

    while True:
        url = _question("Model server URL", default_url)
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
        choice = _choose("[r]etry, [s]tart anyway, [q]uit [r]: ", "rsq", "r")
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
    return FirstSetup(url=url, model=model, saved=save_chat_server(url, model))


def offer_to_save(url: str, model: str) -> bool:
    """Ask once whether to save *url* and *model* as the default backend."""
    if (
        _choose(f"Save {url} · {model} as the default backend? [Y/n] ", "yn", "y")
        == "n"
    ):
        return False
    return save_chat_server(url, model)

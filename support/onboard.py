"""Interactive backend setup flow for first-time configuration."""

from __future__ import annotations

from collections.abc import Callable

from localmelo.support import config


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
    """Local (OpenAI-compatible) chat backend setup: URL and model name."""
    default_url, default_model = _LOCAL_DEFAULTS.get(
        key, ("http://localhost:8000/v1", "")
    )

    print(f"\n  -- {key} (local endpoint) --\n")

    url = _ask("Base URL", default_url)
    model = _ask("Chat model name", default_model)

    # Store in the matching config section
    section = getattr(cfg, key, None)
    if section is not None:
        section.chat_url = url
        section.chat_model = model

    print()
    return _confirm()


def _onboard_cloud(cfg: config.Config, key: str) -> bool:
    """Cloud vendor chat backend setup: API key env var and model name."""
    default_env, default_model = _CLOUD_DEFAULTS.get(key, ("", ""))

    print(f"\n  -- {key} (cloud API) --\n")

    section = getattr(cfg, key, None)
    if section is not None:
        section.api_key_env = _ask("API key env var", default_env)
        section.chat_model = _ask("Chat model", default_model)

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
    """Set up a local embedding endpoint: URL and model name."""
    default_url, default_model = _EMBEDDING_DEFAULTS.get(
        key, ("http://localhost:8000/v1", "")
    )

    url = _ask("Embedding endpoint URL", default_url)
    model = _ask("Embedding model name", default_model)

    section = getattr(cfg, key, None)
    if section is not None:
        section.embedding_url = url
        section.embedding_model = model


# ── Main setup flow ──


def run_backend_setup() -> config.Config | None:
    """Run the split backend setup flow.

    1. Choose and configure the chat backend.
    2. Choose and configure the embedding backend.
    3. Save and return the Config.

    Returns Config if successful, None if cancelled.
    """

    print("\n  Welcome to localmelo.\n")

    # ── 1. Chat backend ──
    chat_display = [name for _, name in CHAT_BACKENDS]
    choice = _ask_choice("Chat backend", chat_display, default=1)
    selected_chat_key = CHAT_BACKENDS[choice - 1][0]

    handler = _CHAT_SETUP.get(selected_chat_key)
    if handler is None:
        print(
            f"  Backend '{selected_chat_key}' has no onboarding handler yet. "
            f"Cannot continue."
        )
        return None

    cfg = config.load()
    ok = handler(cfg)

    if not ok:
        print("  Setup cancelled.")
        return None

    # ── 2. Embedding backend ──
    emb_display = [name for _, name in EMBEDDING_BACKENDS]
    emb_choice = _ask_choice("Embedding backend", emb_display, default=1)
    selected_emb_key = EMBEDDING_BACKENDS[emb_choice - 1][0]

    if selected_emb_key != "none":
        _onboard_embedding_local(cfg, selected_emb_key)
    else:
        print("\n  No embedding: agent will not have long-term memory.")

    cfg.chat_backend = selected_chat_key
    cfg.embedding_backend = selected_emb_key

    config.save(cfg)
    print(f"\n  Config saved to {config.CONFIG_PATH}")
    return cfg

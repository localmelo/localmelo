"""Settings, models and health routes for API v2 under ``/api``.

``GET /api/health`` is served while the store is unavailable: the console's
``Chats are unavailable`` banner needs it, and the model server's
reachability comes from the cached probe without waiting for the network.
``GET/PUT /api/settings``, ``POST /api/settings/test``,
``GET /api/settings/key-status`` and ``GET /api/models`` read and update
``config.toml`` without ever returning, accepting or logging an API key value.
``GET /api/setup/detect`` looks for a model server on this machine, as the
terminal's first run does, for the welcome's setup card.

The console's Settings saves one change at a time: each ``PUT /api/settings``
carries the revision it read, so a change made elsewhere meanwhile is a 409,
and its Undo is the same request with the earlier value.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.contracts.events import EVENT_SCHEMA_VERSION
from localmelo.melo.schema import SESSION_TEXT_CHARS
from localmelo.support import config, onboard, onboard_probe
from localmelo.support import settings as setting_keys
from localmelo.support.backends import discovery
from localmelo.support.backends.discovery import DISCOVERY_TIMEOUT_S, DiscoveryError
from localmelo.support.backends.registry import get_backend
from localmelo.support.gateway.console.errors import ApiError
from localmelo.support.gateway.console.probe import redact_userinfo, url_label
from localmelo.support.gateway.console.registry import DEFAULT_WORKSPACE
from localmelo.support.gateway.console.running import running_status
from localmelo.support.gateway.guard import BODY_BYTES_LIMIT
from localmelo.support.gateway.i18n import t
from localmelo.support.onboard_probe import ProbeError
from localmelo.support.option_tiers import (
    ADVANCED,
    EVERYDAY_OLD,
    EXPERIMENTAL_ISSUES,
    EXPERIMENTAL_KEYS,
    OPTION_ROWS,
    OPTION_TIERS,
    SHORT_ROWS,
)
from localmelo.support.sessions import display
from localmelo.support.sessions.runner import DEFAULT_RESUME_MODE

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

# Seconds ``POST /api/settings/test`` waits for ``probe_all`` at most.
SETTINGS_TEST_TIMEOUT_S: Final = 30.0

# Re-exported so tests may patch either ``onboard_probe.probe_all`` /
# ``discovery.list_models`` or the names on this module.
_ORIG_PROBE_ALL = onboard_probe.probe_all
_ORIG_LIST_MODELS = discovery.list_models
probe_all = _ORIG_PROBE_ALL
list_models = _ORIG_LIST_MODELS

_ENV_NAME_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_LOCAL_FIELDS: Final = (
    "chat_url",
    "chat_model",
    "embedding_url",
    "embedding_model",
)
_CLOUD_FIELDS: Final = (
    "api_key_env",
    "chat_model",
    "base_url",
)


class _SettingsBaseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat_backend: StrictStr
    embedding_backend: StrictStr
    sections: dict[StrictStr, Any] = Field(default_factory=dict)


class SettingsTestBody(_SettingsBaseBody):
    """Request body of ``POST /api/settings/test``."""

    workspace: StrictStr = DEFAULT_WORKSPACE


class SettingsPutBody(_SettingsBaseBody):
    """Request body of ``PUT /api/settings``."""

    workspace: StrictStr
    revision: StrictStr
    # ``[ui] mode``; absent leaves the saved mode as it is.
    mode: StrictStr | None = None
    # ``[memory] across_chats``; absent leaves it as it is.
    across_chats: StrictBool | None = None
    # ``[memory] retriever``, how a chat finds its notes; absent leaves it as it is.
    memory_retriever: StrictStr | None = None
    # ``[console] language``: ``auto``, ``en`` or ``zh-Hans``; absent leaves it as it is.
    language: StrictStr | None = None
    # ``[approvals] mode``, how new chats start: ``ask`` or ``auto``; absent leaves it.
    approvals: StrictStr | None = None


def _check_single_line(field: str, value: str) -> None:
    if config.has_control_or_surrogate(value):
        raise ApiError(
            400,
            "invalid_request",
            f"{field}: Must be a single line with no control characters.",
        )


def _check_workspace(workspace: str) -> None:
    problem = config.gateway_workspace_problem(workspace, allow_empty=False)
    if problem is not None:
        raise ApiError(
            400,
            "invalid_request",
            f"workspace: {problem}",
        )


def _file_revision() -> str:
    """First 16 hex characters of the SHA-256 of ``config.toml``, or ``""``."""
    try:
        with open(config.config_path(), "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return hashlib.sha256(data).hexdigest()[:16]


def _redact(text: str) -> str:
    """Return *text* with any secret shapes replaced by ``[redacted]``."""
    return redact_secrets(text)[0]


def model_json(cfg: Any) -> dict[str, Any]:
    """The settings' chat model, with the registry's label for its backend.

    A ``vllm`` backend is labelled ``oMLX`` when the config says so or the
    health probe's last model listing did.
    """
    backend = cfg.chat_backend
    section = getattr(cfg, backend, None) if backend else None
    name = getattr(section, "chat_model", "") if section is not None else ""
    try:
        label = get_backend(backend).label(cfg)
    except KeyError:
        label = backend
    return {
        "backend": backend,
        "backend_label": label,
        "name": name,
        "embedding": cfg.embedding_backend or "none",
    }


async def health_json(console: Console) -> dict[str, Any]:
    """The whole ``GET /api/health`` body."""
    entries = console.sessions.entries()
    return {
        "ok": True,
        "api": 2,
        "events_schema": EVENT_SCHEMA_VERSION,
        "version": display.localmelo_version(),
        "configured": console.cfg.is_configured,
        "config_errors": [_redact(err) for err in console.cfg.validate()],
        "model": model_json(console.cfg),
        "model_server": console.probe.snapshot(),
        "limits": {
            "text_chars": MAX_QUERY_LEN,
            "attachment_budget_chars": SESSION_TEXT_CHARS,
            "body_bytes": BODY_BYTES_LIMIT,
        },
        "sessions": {
            "open": len(entries),
            "running": sum(1 for entry in entries if entry.runner.busy),
            "pending_approvals": sum(
                len(console.broker.pending(entry.session_id)) for entry in entries
            ),
            "store": console.store_state(),
            "store_reason": console.store_reason(),
            "store_error": console.store_error(),
            "resume_mode": DEFAULT_RESUME_MODE,
        },
        "status": await running_status(console),
        "uptime_s": round(console.uptime_s(), 1),
    }


async def setup_detect_json(console: Console) -> dict[str, Any]:
    """The whole ``GET /api/setup/detect`` body: the model server on this Mac.

    ``state`` is ``found``, ``needs_key`` or ``none``. ``backend`` is what the
    terminal's first run would save (``ollama`` or ``vllm``) and ``name`` is
    ``Ollama``, ``oMLX`` or empty. ``models`` are the chat models, ``loaded``
    those the server says are in memory, and ``suggested`` the one to offer:
    null when several are served and none is known loaded, so the user picks.
    ``start`` is how to start a server on this computer, for the card that finds none.
    """
    transport = console.options.detect_transport
    urls = discovery.detect_urls()
    found = (
        await discovery.detect_local_server(urls, transport=transport) if urls else None
    )
    body: dict[str, Any] = dict.fromkeys(("name", "backend", "url", "problem"), "")
    body.update(
        state="none",
        models=[],
        loaded=[],
        suggested=None,
        start=discovery.how_to_start().to_json(),
    )
    if found is None:
        return body
    body["url"] = url_label(found.url)
    body["problem"] = _redact(redact_userinfo(found.problem))
    if discovery.NO_API_KEY_HINT in found.problem:
        return {**body, "state": "needs_key"}
    ollama = await discovery.is_ollama(
        found.url, timeout_s=discovery.DETECT_TIMEOUT_S, transport=transport
    )
    omlx = discovery.is_omlx(found.url)
    body["name"] = "Ollama" if ollama else discovery.OMLX_NAME if omlx else ""
    chat = [model for model in found.models if discovery.is_chat_model(model)]
    if not chat:
        return {**body, "problem": body["problem"] or "it lists only embedding models"}
    loaded = discovery.LoadedModels()
    if len(chat) > 1:
        loaded = await discovery.loaded_models(found.url, transport=transport)
    return {
        **body,
        "state": "found",
        "backend": "ollama" if ollama else "vllm",
        "models": chat,
        "loaded": [model for model in chat if model in loaded.ids],
        "suggested": discovery.pick_model(chat, loaded),
    }


def _ui_mode(cfg: config.Config) -> str:
    """The saved ``[ui] mode``; a value the CLI warns about reads as default."""
    return cfg.ui.mode if cfg.ui.mode in config.UI_MODES else "default"


def _advanced_options() -> list[dict[str, str]]:
    """The advanced and experimental ``melo`` options for the developer view.

    Read-only: the tier label, and for an experimental option the issue that
    decides it. The user tier is what the console already shows.
    """
    return [
        {"option": name, "tier": tier, "issue": EXPERIMENTAL_ISSUES.get(name, "")}
        for name, tier in OPTION_TIERS.items()
        if (tier == ADVANCED or name in EXPERIMENTAL_ISSUES)
        and name not in EVERYDAY_OLD
    ]


def _shown(value: Any) -> str:
    """A setting's value as ``melo -h all`` prints it: text bare, the rest as JSON."""
    text = value if isinstance(value, str) and value else setting_keys.show(value)
    return _redact(text)


def _developer_settings(cfg: config.Config) -> list[dict[str, Any]]:
    """Every ``config.toml`` key ``melo -h all`` lists, by section, with its value.

    A key for one run only (``file`` false) shows its default: the file cannot
    hold it. ``flag`` is the old option that sets it, ``issue`` the issue that
    decides a comparison.
    """
    values = setting_keys.flat(cfg)
    groups: dict[str, list[dict[str, Any]]] = {}
    for key in setting_keys.catalog().values():
        default = _shown(key.default)
        groups.setdefault(key.section, []).append(
            {
                "key": key.name,
                "value": _shown(values[key.name]) if key.file else default,
                "default": default,
                "flag": key.flag,
                "issue": EXPERIMENTAL_ISSUES.get(key.flag)
                or EXPERIMENTAL_KEYS.get(key.name, ""),
                "file": key.file,
            }
        )
    return [{"section": name, "keys": keys} for name, keys in groups.items()]


def _run_options() -> list[dict[str, str]]:
    """The options of ``melo -h all`` past the everyday ones: they set one run."""
    return [
        {"option": f"{names} {meta}".strip(), "text": text}
        for names, meta, text in OPTION_ROWS[SHORT_ROWS:]
    ]


def settings_json() -> dict[str, Any]:
    """The whole ``GET /api/settings`` body."""
    revision = _file_revision()
    cfg = config.load()
    sections: dict[str, dict[str, Any]] = {}
    for name in config._LOCAL_SECTIONS:
        sec = getattr(cfg, name)
        sections[name] = {
            "chat_url": str(sec.chat_url),
            "chat_model": str(sec.chat_model),
            "embedding_url": str(sec.embedding_url),
            "embedding_model": str(sec.embedding_model),
        }
    for name in config._CLOUD_SECTIONS:
        sec = getattr(cfg, name)
        env_name = str(sec.api_key_env)
        sections[name] = {
            "api_key_env": env_name,
            "api_key_set": bool(env_name and os.environ.get(env_name)),
            "chat_model": str(sec.chat_model),
            "base_url": str(sec.base_url),
        }
    ws = (
        cfg.gateway.workspace
        if config.gateway_workspace_problem(cfg.gateway.workspace, allow_empty=False)
        is None
        else ""
    )
    return {
        "chat_backend": cfg.chat_backend,
        "embedding_backend": cfg.embedding_backend,
        "workspace": ws or DEFAULT_WORKSPACE,
        "sections": sections,
        "defaults": onboard.backend_defaults(),
        "errors": [_redact(err) for err in cfg.validate()],
        "choices": {
            "chat_backends": [
                {
                    "key": key,
                    "label": label,
                    "kind": "local" if key in config._LOCAL_SECTIONS else "cloud",
                }
                for key, label in onboard.CHAT_BACKENDS
            ],
            "embedding_backends": [
                {"key": key, "label": label}
                for key, label in onboard.EMBEDDING_BACKENDS
            ],
        },
        "mode": _ui_mode(cfg),
        "across_chats": cfg.memory.across_chats,
        "memory_retriever": cfg.memory.retriever,
        "language": cfg.console.language,
        "approvals": (
            cfg.approvals.mode if cfg.approvals.mode in config.APPROVAL_MODES else "ask"
        ),
        "advanced_options": _advanced_options(),
        "developer_settings": _developer_settings(cfg),
        "run_options": _run_options(),
        "file_label": display.home_label(config.config_path()),
        "revision": revision,
    }


def _validate_sections(sections: dict[str, Any]) -> None:
    for sec_name, sec_fields in sections.items():
        _check_single_line("sections", sec_name)
        if sec_name in config._LOCAL_SECTIONS:
            allowed: tuple[str, ...] = _LOCAL_FIELDS
        elif sec_name in config._CLOUD_SECTIONS:
            allowed = _CLOUD_FIELDS
        else:
            raise ApiError(
                400,
                "invalid_request",
                f"sections.{sec_name}: Unknown backend section.",
            )
        if not isinstance(sec_fields, dict):
            raise ApiError(
                400,
                "invalid_request",
                f"sections.{sec_name}: Input should be a valid dictionary.",
            )
        for field_name, field_val in sec_fields.items():
            if not isinstance(field_name, str) or field_name not in allowed:
                raise ApiError(
                    400,
                    "invalid_request",
                    f"sections.{sec_name}.{field_name}: Extra inputs are not permitted.",
                )
            if not isinstance(field_val, str):
                raise ApiError(
                    400,
                    "invalid_request",
                    f"sections.{sec_name}.{field_name}: Input should be a valid string.",
                )
            _check_single_line(f"sections.{sec_name}.{field_name}", field_val)
            if (
                sec_name in config._CLOUD_SECTIONS
                and field_name == "api_key_env"
                and field_val
                and not _ENV_NAME_RE.fullmatch(field_val)
            ):
                raise ApiError(
                    400,
                    "invalid_request",
                    f"sections.{sec_name}.api_key_env: Invalid environment variable name.",
                )


def _build_candidate(
    body: SettingsTestBody | SettingsPutBody,
    *,
    check_revision: str | None = None,
) -> tuple[config.Config, config.Config]:
    """Validate *body* and apply it onto a fresh ``config.load()`` copy."""
    _check_single_line("chat_backend", body.chat_backend)
    _check_single_line("embedding_backend", body.embedding_backend)
    _check_workspace(body.workspace)
    if check_revision is not None:
        _check_single_line("revision", check_revision)
    _validate_sections(body.sections)
    mode = getattr(body, "mode", None)
    if mode is not None and mode not in config.UI_MODES:
        raise ApiError(
            400, "invalid_request", f"mode: Must be one of {config.UI_MODES}."
        )
    retriever = getattr(body, "memory_retriever", None)
    if retriever is not None and retriever not in config.MEMORY_RETRIEVERS:
        raise ApiError(
            400,
            "invalid_request",
            f"memory_retriever: Must be one of {config.MEMORY_RETRIEVERS}.",
        )
    language = getattr(body, "language", None)
    if language is not None and language not in config.CONSOLE_LANGUAGES:
        raise ApiError(
            400,
            "invalid_request",
            f"language: Must be one of {config.CONSOLE_LANGUAGES}.",
        )

    approvals = getattr(body, "approvals", None)
    if approvals is not None and approvals not in config.APPROVAL_MODES:
        raise ApiError(
            400,
            "invalid_request",
            f"approvals: Must be one of {config.APPROVAL_MODES}.",
        )

    if check_revision is not None and check_revision != _file_revision():
        raise ApiError(
            409,
            "settings_changed",
            "The settings file changed since it was loaded.",
        )

    before = config.load()
    candidate = config.load()
    candidate.chat_backend = body.chat_backend
    candidate.embedding_backend = body.embedding_backend
    candidate.gateway.workspace = body.workspace
    candidate.ui.mode = mode or _ui_mode(candidate)
    across = getattr(body, "across_chats", None)
    if across is not None:
        candidate.memory.across_chats = across
    if retriever is not None:
        candidate.memory.retriever = retriever
    if language is not None:
        candidate.console.language = language
    if approvals is not None:
        candidate.approvals.mode = approvals
    for sec_name, sec_fields in body.sections.items():
        target = getattr(candidate, sec_name)
        for field_name, field_val in sec_fields.items():
            setattr(target, field_name, field_val)

    errors = [_redact(err) for err in candidate.validate()]
    if errors:
        raise ApiError(
            400,
            "invalid_settings",
            errors[0],
            {"errors": errors},
        )
    return before, candidate


def _spell_server(before: config.Config, candidate: config.Config) -> None:
    """Spell a ``vllm`` chat backend ``omlx`` as the terminal's first run does.

    The file's ``omlx`` stays only while the chat server is the one it named,
    and a server whose last model listing named oMLX is written so.
    """
    server = (candidate.chat_backend, candidate.vllm.chat_url)
    if server != (before.chat_backend, before.vllm.chat_url):
        candidate.chat_alias = ""
    onboard._spell_omlx(candidate)


def _check_probe_url(field: str, url: str) -> None:
    if not url_label(url):
        raise ApiError(
            400, "invalid_request", f"{field}: Expected an absolute http(s) URL."
        )


def _check_candidate_probe_urls(candidate: config.Config) -> None:
    backend = candidate.chat_backend
    section = getattr(candidate, backend)
    if backend in config._LOCAL_SECTIONS:
        _check_probe_url(f"{backend}.chat_url", section.chat_url)
    elif backend in config._CLOUD_SECTIONS and section.base_url:
        _check_probe_url(f"{backend}.base_url", section.base_url)
    backend = candidate.embedding_backend
    if backend in config._LOCAL_SECTIONS:
        _check_probe_url(
            f"{backend}.embedding_url", getattr(candidate, backend).embedding_url
        )


def _changed_fields(before: config.Config, after: config.Config) -> list[str]:
    changed: list[str] = []
    if before.chat_backend != after.chat_backend:
        changed.append("chat_backend")
    if before.embedding_backend != after.embedding_backend:
        changed.append("embedding_backend")
    if (before.gateway.workspace or DEFAULT_WORKSPACE) != (
        after.gateway.workspace or DEFAULT_WORKSPACE
    ):
        changed.append("workspace")
    if _ui_mode(before) != _ui_mode(after):
        changed.append("mode")
    if before.memory.across_chats != after.memory.across_chats:
        changed.append("across_chats")
    if before.memory.retriever != after.memory.retriever:
        changed.append("memory_retriever")
    if before.console.language != after.console.language:
        changed.append("language")
    if before.approvals.mode != after.approvals.mode:
        changed.append("approvals")
    for sec_name in config._LOCAL_SECTIONS:
        b_sec = getattr(before, sec_name)
        a_sec = getattr(after, sec_name)
        for field_name in _LOCAL_FIELDS:
            if getattr(b_sec, field_name) != getattr(a_sec, field_name):
                changed.append(f"{sec_name}.{field_name}")
    for sec_name in config._CLOUD_SECTIONS:
        b_sec = getattr(before, sec_name)
        a_sec = getattr(after, sec_name)
        for field_name in _CLOUD_FIELDS:
            if getattr(b_sec, field_name) != getattr(a_sec, field_name):
                changed.append(f"{sec_name}.{field_name}")
    return changed


async def _call_probe_all(cfg: config.Config) -> None:
    fn = probe_all if probe_all is not _ORIG_PROBE_ALL else onboard_probe.probe_all
    await fn(cfg)


async def _call_list_models(backend: str, url: str) -> list[str]:
    fn = list_models if list_models is not _ORIG_LIST_MODELS else discovery.list_models
    return await fn(backend, url, timeout_s=DISCOVERY_TIMEOUT_S)


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the health, settings and models routes on *app*."""

    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return JSONResponse(content=await health_json(console))

    @app.get("/api/settings")
    async def api_get_settings() -> JSONResponse:
        return JSONResponse(content=settings_json())

    @app.get("/api/setup/detect")
    async def api_setup_detect() -> JSONResponse:
        return JSONResponse(content=await setup_detect_json(console))

    @app.get("/api/settings/key-status")
    async def api_key_status(name: str | None = None) -> JSONResponse:
        if not name or not _ENV_NAME_RE.fullmatch(name):
            raise ApiError(
                400,
                "invalid_request",
                "name: Invalid environment variable name.",
            )
        return JSONResponse(content={"set": bool(os.environ.get(name))})

    @app.put("/api/settings")
    async def api_put_settings(body: SettingsPutBody) -> JSONResponse:
        before, candidate = _build_candidate(body, check_revision=body.revision)
        _spell_server(before, candidate)
        changed = _changed_fields(before, candidate)
        config.save(candidate)
        logger.info(
            "settings saved: chat_backend=%s embedding_backend=%s changed=%s",
            candidate.chat_backend,
            candidate.embedding_backend,
            ",".join(changed) or "none",
        )
        running_host = console.cfg.gateway.host
        running_port = console.cfg.gateway.port
        active = copy.deepcopy(candidate)
        active.gateway.host = running_host
        active.gateway.port = running_port
        console.cfg = active
        reset = getattr(console.probe, "reset", None)
        if reset is not None:
            reset(active)
        await console.sessions.settings_changed()
        return JSONResponse(content=settings_json())

    @app.post("/api/settings/test")
    async def api_test_settings(body: SettingsTestBody) -> JSONResponse:
        _, candidate = _build_candidate(body)
        _check_candidate_probe_urls(candidate)
        try:
            async with asyncio.timeout(SETTINGS_TEST_TIMEOUT_S):
                await _call_probe_all(candidate)
        except TimeoutError:
            return JSONResponse(
                content={"ok": False, "error": "The test did not finish within 30 s."}
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err = _redact(redact_userinfo(str(exc) or type(exc).__name__))
            # Which model did not answer: the embedding probe's errors say so.
            part = (
                "embedding"
                if isinstance(exc, ProbeError)
                and str(exc).startswith("embedding probe")
                else "chat"
            )
            return JSONResponse(content={"ok": False, "error": err, "failed": part})
        return JSONResponse(content={"ok": True})

    @app.get("/api/models")
    async def api_models(
        backend: str | None = None, url: str | None = None
    ) -> JSONResponse:
        if backend is not None:
            _check_single_line("backend", backend)
        if url is not None:
            _check_single_line("url", url)

        saved = config.load() if os.path.isfile(config.config_path()) else console.cfg
        resolved_backend = backend if backend else saved.chat_backend
        if resolved_backend in config._CLOUD_SECTIONS:
            raise ApiError(
                400,
                "invalid_request",
                t("errors.listLocalOnly"),
            )
        if resolved_backend not in config._LOCAL_SECTIONS:
            raise ApiError(
                400,
                "invalid_request",
                f"backend: Unknown local backend {resolved_backend!r}.",
            )

        if url:
            resolved_url = url
        else:
            sec = getattr(saved, resolved_backend, None)
            resolved_url = getattr(sec, "chat_url", "") if sec is not None else ""
        if not resolved_url:
            raise ApiError(
                400,
                "invalid_request",
                t("errors.noChatUrl"),
            )

        _check_probe_url("url", resolved_url)
        shown_url = url_label(resolved_url)
        try:
            models = await _call_list_models(resolved_backend, resolved_url)
        except DiscoveryError as exc:
            reason = _redact(redact_userinfo(str(exc)))
            raise ApiError(
                502,
                "backend_unreachable",
                t("errors.listFailed", url=shown_url, reason=reason),
            ) from exc
        if not models:
            raise ApiError(
                502,
                "backend_unreachable",
                t("errors.noModelsAt", url=shown_url),
            )
        embedding = discovery.embedding_models(models)
        return JSONResponse(
            content={
                "models": models,
                "embedding_models": embedding,
                "suggested_embedding": discovery.recommend_embedding(embedding),
            }
        )

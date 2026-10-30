"""Settings, models and health routes for API v2 under ``/api``.

``GET /api/health`` is served while the store is unavailable: the console's
``Chats are unavailable`` banner needs it, and the model server's
reachability comes from the cached probe without waiting for the network.
``GET/PUT /api/settings``, ``POST /api/settings/test``,
``GET /api/settings/key-status`` and ``GET /api/models`` read and update
``config.toml`` without ever returning, accepting or logging an API key value.
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
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.contracts.events import EVENT_SCHEMA_VERSION
from localmelo.melo.schema import SESSION_TEXT_CHARS
from localmelo.support import config, onboard, onboard_probe
from localmelo.support.backends import discovery
from localmelo.support.backends.discovery import DISCOVERY_TIMEOUT_S, DiscoveryError
from localmelo.support.backends.registry import get_backend
from localmelo.support.gateway.console.errors import ApiError
from localmelo.support.gateway.console.probe import redact_userinfo, url_label
from localmelo.support.gateway.console.registry import DEFAULT_WORKSPACE
from localmelo.support.gateway.console.running import running_status
from localmelo.support.gateway.guard import BODY_BYTES_LIMIT
from localmelo.support.option_tiers import ADVANCED, EXPERIMENTAL_ISSUES, OPTION_TIERS
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
        if tier == ADVANCED or name in EXPERIMENTAL_ISSUES
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
        "advanced_options": _advanced_options(),
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
            return JSONResponse(content={"ok": False, "error": err})
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
                "Listing models works for local backends only; type the model name.",
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
                "url: No chat URL is configured.",
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
                f"Could not list models at {shown_url}: {reason}",
            ) from exc
        if not models:
            raise ApiError(
                502,
                "backend_unreachable",
                f"No models found at {shown_url}.",
            )
        return JSONResponse(content={"models": models})

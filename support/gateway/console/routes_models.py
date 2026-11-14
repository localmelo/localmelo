"""The models of the hosted services, and the one switch list: ``/api/hosted/models``, ``/api/chat-models``.

``GET /api/hosted/models?preset=<id>[&address=<url>][&model=<name>]`` is what Settings
lists once a key for the service exists: one ``GET <address>/models``, never a chat
(the Test button sends one). The key is the preset's own and goes only to its address:
a built-in preset's address is built in and an address named for it must be that one
(409 ``key_address`` before any request); "Other" keeps its key for the address it was
saved with, so another address finds no key.

``GET /api/chat-models`` is the switch list, grouped by server and service, and
``POST /api/chat-models/switch`` the switch: it saves the choice like a settings change,
so the next message uses it and a turn that runs finishes on the model it started with.
Neither reads or writes the embedding choice.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from localmelo.melo.contracts.redaction import redact_secrets
from localmelo.support import config, hosted, hosted_check, keychain, model_choices
from localmelo.support.gateway.console.errors import ApiError
from localmelo.support.gateway.console.probe import redact_userinfo
from localmelo.support.gateway.i18n import t

from .routes_settings import (
    _hosted_choices,
    _hosted_message,
    _key_error,
    _kind_words,
    apply_settings,
    key_service,
    model_json,
)

if TYPE_CHECKING:
    from . import Console


class SwitchBody(BaseModel):
    """Request body of ``POST /api/chat-models/switch``: a group and a model of its list."""

    model_config = ConfigDict(extra="forbid")

    group: StrictStr = Field(min_length=1, max_length=64)
    model: StrictStr = Field(default="", max_length=model_choices.MAX_MODEL)


def _saved(console: Console) -> config.Config:
    """The saved config, else the running one (as the models route reads it)."""
    return config.load() if os.path.isfile(config.config_path()) else console.cfg


def _reason(group: model_choices.Group) -> str:
    """Why a group's listing failed, in the request's language; "" when it did not."""
    if group.check is not None:
        return redact_secrets(_hosted_message(group.check, group.current))[0]
    return redact_secrets(redact_userinfo(group.problem))[0] if group.problem else ""


def _group_json(group: model_choices.Group, labels: dict[str, str]) -> dict[str, Any]:
    return {
        "id": group.id,
        "label": labels.get(group.id) or _kind_words(group.label),
        "kind": "hosted" if group.hosted else "local",
        "serves": group.serves,
        "models": list(group.models),
        "current": group.current,
        "problem": _reason(group),
    }


def _refusal(error: model_choices.ChoiceError, model: str) -> ApiError:
    code = error.code
    if code.startswith("keychain_"):
        return _key_error(keychain.KeychainError(code.removeprefix("keychain_")))
    if code == hosted.ADDRESS:
        return ApiError(409, "key_address", t(keychain.words_key(hosted.ADDRESS)))
    if code == "no_key":
        return ApiError(409, "no_key", t("hosted.test.noKey"))
    if code == "no_model":
        return ApiError(400, "no_model", t("hosted.test.noModel", model=model))
    if code == "bad_model":
        return ApiError(400, "invalid_request", "model: Expected a model name.")
    return ApiError(400, "invalid_request", "group: Unknown server or service.")


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the hosted model list and the switch list on *app*."""

    @app.get("/api/hosted/models")
    async def api_hosted_models(
        preset: str, address: str | None = None, model: str = ""
    ) -> JSONResponse:
        found = hosted.get(preset)
        if found is None:
            raise ApiError(400, "invalid_request", "preset: Unknown hosted service.")
        # The address is the preset's own, or "Other"'s: a key is never sent anywhere else.
        key_service(found, address)
        where = (address or "").strip() if found.free_address else found.base_url
        listing = await hosted_check.service_models(found, where)
        failed = listing.check
        shown = redact_secrets(_hosted_message(failed, model))[0] if failed else ""
        gone = (
            failed is None and bool(model) and hosted_check.lacks(listing.models, model)
        )
        return JSONResponse(
            content={
                "preset": found.id,
                "key": listing.key,
                "listed": bool(listing.models),
                "models": list(listing.models),
                "result": failed.result if failed else "",
                "message": shown,
                "gone": gone,
                "line": t("hosted.test.noModel", model=model) if gone else "",
            }
        )

    @app.get("/api/chat-models")
    async def api_chat_models(models: int = 1) -> JSONResponse:
        # ``?models=0`` names the groups and asks no server for its models.
        cfg = _saved(console)
        groups = await model_choices.collect(cfg, listing=models != 0)
        labels = {item["id"]: item["label"] for item in _hosted_choices()}
        return JSONResponse(
            content={
                "hosted": bool(groups),
                "current": {
                    "group": model_choices.in_use_group(cfg),
                    "model": getattr(
                        getattr(cfg, cfg.chat_backend, None), "chat_model", ""
                    ),
                },
                "groups": [_group_json(group, labels) for group in groups],
            }
        )

    @app.post("/api/chat-models/switch")
    async def api_switch_chat_model(body: SwitchBody) -> JSONResponse:
        candidate = copy.deepcopy(_saved(console))
        try:
            await model_choices.apply(candidate, body.group, body.model)
        except model_choices.ChoiceError as exc:
            raise _refusal(exc, body.model) from None
        errors = [redact_secrets(err)[0] for err in candidate.validate()]
        if errors:
            raise ApiError(400, "invalid_settings", errors[0], {"errors": errors})
        await apply_settings(console, candidate, "chat_model")
        return JSONResponse(content={"ok": True, "model": model_json(console.cfg)})

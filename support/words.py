"""The words of the console's catalogs, readable without the gateway.

``support/gateway/static/i18n/<language>.json`` holds one catalog per language
(keys are flat and grouped by screen; a value is a string with ``{name}``
parameters, or, for a count, ``{"one": ..., "other": ...}`` picked by the
parameter ``n``). The gateway (:mod:`localmelo.support.gateway.i18n`), the page
and the macOS app read them; so can a surface that must run without the gateway
extra, such as the terminal, which writes words in English through
:func:`t_in` ``("en", key, ...)``. English is the source language: a key a
catalog lacks reads in English.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Final

LANGUAGES: Final = ("en", "zh-Hans")
CATALOGS: Final = Path(__file__).with_name("gateway") / "static" / "i18n"

_PARAM: Final = re.compile(r"\{(\w+)\}")

Entry = str | Mapping[str, str]


@cache
def catalog(language: str) -> dict[str, Entry]:
    """The catalog of *language*, read once."""
    data: dict[str, Entry] = json.loads(
        (CATALOGS / f"{language}.json").read_text(encoding="utf-8")
    )
    return data


def plural_form(language: str, n: object) -> str:
    """``one`` or ``other``: English has both, Chinese only ``other``."""
    return "one" if language == "en" and n == 1 else "other"


def format_entry(entry: Entry, params: Mapping[str, Any], language: str) -> str:
    """*entry* with its parameters filled in; a count picks its plural form first."""
    if not isinstance(entry, str):
        entry = entry.get(plural_form(language, params.get("n")), entry["other"])
    return _PARAM.sub(
        lambda found: (
            str(params[found.group(1)]) if found.group(1) in params else found.group()
        ),
        entry,
    )


def t_in(language: str, key: str, /, **params: Any) -> str:
    """The words of *key* in *language*, falling back to English, then to the key."""
    own = catalog(language) if language in LANGUAGES else {}
    entry = own.get(key)
    if entry is None:
        language, entry = "en", catalog("en").get(key)
    return key if entry is None else format_entry(entry, params, language)


def said(key: str, /, **params: Any) -> dict[str, Any]:
    """Words to store with a record (a card, an echo) so each reader reads them in their language.

    ``{"key": key, "params": params}``; a parameter may itself be a :func:`said`,
    or a time of day, ``{"clock": minutes after midnight}``, which a reader
    writes as its language writes a clock (here, as ``8:30``), or a step's
    sentence, ``{"gist": {id, params}}`` (see :func:`gist_sentence`).
    """
    return {"key": key, "params": params}


def clock_words(minute: int) -> str:
    """``8:30`` for 510 minutes after midnight."""
    return f"{minute // 60}:{minute % 60:02d}"


def said_in(language: str, words: Mapping[str, Any]) -> str:
    """The words of a :func:`said` in *language*."""
    params = words.get("params")
    filled: dict[str, Any] = {}
    for name, value in params.items() if isinstance(params, Mapping) else ():
        if isinstance(value, Mapping) and "key" in value:
            value = said_in(language, value)
        elif isinstance(value, Mapping) and "clock" in value:
            value = clock_words(int(value["clock"]))
        elif isinstance(value, Mapping) and "gist" in value:
            value = gist_sentence(value["gist"], language)
        filled[name] = value
    return t_in(language, str(words.get("key")), **filled)


def gist_names(names: Sequence[object], language: str = "en") -> str:
    """*names* as a list in *language*: ``Safari``, ``Safari and Notes``, ``Safari, Notes and Mail``."""
    found = [str(name) for name in names]
    if len(found) < 2:
        return "".join(found)
    if len(found) == 2:
        return t_in(language, "gist.list.two", first=found[0], second=found[1])
    head = t_in(language, "gist.list.sep").join(found[:-1])
    return t_in(language, "gist.list.many", head=head, last=found[-1])


def gist_sentence(node: object, language: str = "en") -> str:
    """The sentence of a gist's ``say`` node (``{id, params}``) in *language*.

    A parameter may be another sentence (``then``, ``more``) or a list of names
    (``apps``). In English the sentence after ``then`` starts in lower case.
    """
    if not isinstance(node, Mapping) or not isinstance(node.get("id"), str):
        return ""
    params = node.get("params")
    filled: dict[str, Any] = {}
    for name, value in params.items() if isinstance(params, Mapping) else ():
        if isinstance(value, Mapping):
            value = gist_sentence(value, language)
            if name == "next" and language.startswith("en"):
                value = value[:1].lower() + value[1:]
        elif isinstance(value, list):
            value = gist_names(value, language)
        filled[name] = value
    return t_in(language, f"gist.say.{node['id']}", **filled)


def gist_text(gist: object, language: str = "en") -> tuple[str, str]:
    """The sentence an approval card starts with, and its warning, in *language*.

    *gist* is what ``melo.checker.review.gist.Gist.to_json`` gives; anything
    else reads as nothing, ``("", "")``. The warning is ``""`` unless the
    model's words disagree with what the code does.
    """
    if not isinstance(gist, Mapping):
        return "", ""
    warn = gist.get("warn")
    warning = ""
    if isinstance(warn, Mapping) and isinstance(warn.get("id"), str):
        params = warn.get("params")
        warning = t_in(
            language,
            f"gist.warn.{warn['id']}",
            **(params if isinstance(params, Mapping) else {}),
        )
    return gist_sentence(gist.get("say"), language), warning


def gist_note(gist: object, language: str = "en") -> str:
    """The model's words under a risky step's sentence, labelled as the model's; ``""`` if none."""
    note = gist.get("note") if isinstance(gist, Mapping) else None
    if not isinstance(note, str) or not note:
        return ""
    return t_in(language, "gist.say.model", text=note)

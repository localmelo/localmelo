"""The review of one tool call: routine, high-impact or undecided (#279).

The checker's rules (:mod:`localmelo.melo.checker.review`) classify each
``shell_exec``, ``python_exec`` and ``file_write`` call before an approval
policy decides. A session grant covers routine calls only: a high-impact
call asks every time and is never granted. Under a grant, and in auto
mode, the user's model reviews an undecided call (``by="model"``). In auto
mode (#339) nobody is asked: a call that is not routine is blocked, unless
the user allowed that exact step once (:func:`step_key`) or asked for it in
their own words. This module imports only the standard library.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

Verdict = Literal["routine", "high", "undecided"]
RULES_VERSION: Final = 1

# The card's title per category, strongest first: a call's title and the
# order of its reasons follow this order.
TITLES: Final[dict[str, str]] = {
    "self": "Change localmelo itself?",
    "credentials": "Use your passwords or keys?",
    "system": "Change system settings?",
    "persistence": "Set up something that runs later?",
    "remote_code": "Run code downloaded from the internet?",
    "install": "Install software?",
    "restore": "Install the project's packages?",
    "env": "Create a programming environment?",
    "download": "Download files to this Mac?",
    "outside_write": "Change files outside {folder}?",
    "delete": "Delete files that cannot be restored?",
    "service": "Open a network service?",
    "send_out": "Send data from this Mac?",
    "opaque": "Run code melo cannot check?",
    "beyond_request": "Do something you did not ask for?",
    "unreviewed": "Run a step melo could not review?",
}
# Each category in a few words, for a blocked step's tag ("blocked: installs
# software") and the reason an auto-mode block records.
CATEGORY_WORDS: Final[dict[str, str]] = {
    "self": "changes localmelo itself",
    "credentials": "uses passwords or keys",
    "system": "changes system settings",
    "persistence": "sets up something that runs later",
    "remote_code": "runs code from the internet",
    "install": "installs software",
    "restore": "installs the project's packages",
    "env": "creates a programming environment",
    "download": "downloads files",
    "outside_write": "changes files outside the folder",
    "delete": "deletes files",
    "service": "opens a network service",
    "send_out": "sends data from this Mac",
    "opaque": "runs code melo cannot check",
    "beyond_request": "does something you did not ask for",
    "unreviewed": "could not be reviewed",
}
# Auto mode: the categories the user's own words can allow (they can name
# what the step acts on), and those only an "Allow once" of the exact step
# can. ``self`` is in neither: it is never allowed.
WORDS_CAN_ALLOW: Final = frozenset(
    {
        "install",
        "restore",
        "env",
        "download",
        "outside_write",
        "delete",
        "send_out",
        "service",
        "beyond_request",
    }
)
CLICK_ONLY: Final = frozenset(
    {"credentials", "system", "persistence", "remote_code", "opaque", "unreviewed"}
)
# The reason a call of the ``self`` category is denied without asking.
SELF_DENIAL: Final = "melo never lets a call change its own runtime, settings or keys"
# The tools whose arguments the rules read (melo.checker.review); another
# tool is routine to them, and auto mode makes one that is not read-only
# undecided.
RULED_TOOLS: Final = frozenset(["shell_exec", "python_exec", "file_write", "file_edit"])
# A model's category that can let a call run: none, or one words can allow.
MODEL_CAN_ALLOW: Final = frozenset({"none", ""}) | WORDS_CAN_ALLOW


@dataclass(frozen=True)
class ModelAnswer:
    """What the user's model answered about one call (see ``melo.checker.review.model``).

    ``verdict`` is ``routine`` or ``ask``; ``fits_request`` whether the call
    serves what the user asked (``yes``, ``no`` or ``unclear``);
    ``user_asked`` whether the user's own words ask for it; ``reason`` one
    sentence for the user; ``ms`` how long the review took. ``error`` names
    why there is no usable answer (a timeout, a failed request, a reply that
    does not parse, the turn's limit, a call too long to review), and the
    other fields are then empty.
    """

    verdict: str = ""
    category: str = ""
    fits_request: str = ""
    user_asked: str = ""
    reason: str = ""
    ms: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        if self.error:
            return {"error": self.error, "ms": self.ms}
        return {
            "verdict": self.verdict,
            "category": self.category,
            "fits_request": self.fits_request,
            "user_asked": self.user_asked,
            "reason": self.reason,
            "ms": self.ms,
        }


@dataclass(frozen=True)
class ActionReview:
    """What the rules found in one call, and the model when it was asked.

    ``categories`` are keys of :data:`TITLES`, strongest first; ``reasons``
    are at most three sentences for the person, each starting with a verb
    ("Installs playwright with pip."); ``paths`` are write paths outside
    the workspace that the call names. A review the model settled has
    ``by="model"`` and its :class:`ModelAnswer`; a high-impact call the
    user's words allowed keeps its categories with the verdict ``routine``.
    ``doubts`` are what the rules could not tell about a high-impact call
    (the reasons an undecided call has in ``reasons``): words never allow
    such a call. ``inputs`` name the local files whose content the rules
    read, each as ``path:digest``; :func:`step_key` includes them, so a
    changed file makes another step. ``changed`` names the files changed
    in this chat that a runner (``pytest``…) runs and melo could not read,
    as they are now: not part of the key, it tells one review of the step
    from the next.
    """

    verdict: Verdict
    categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    by: Literal["rules", "model"] = "rules"
    rules_version: int = RULES_VERSION
    model: ModelAnswer | None = None
    doubts: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "categories": list(self.categories),
            "reasons": list(self.reasons),
            "paths": list(self.paths),
            "by": self.by,
            "rules_version": self.rules_version,
        }
        if self.doubts:
            out["doubts"] = list(self.doubts)
        if self.model is not None:
            out["model"] = self.model.to_json()
        return out


def covered_by_grant(review: ActionReview | None) -> bool:
    """Whether a session grant of the call's tool lets it run unasked."""
    return review is None or review.verdict == "routine"


def offers_grant(review: ActionReview | None) -> bool:
    """Whether the card may offer "allow for the session": never for high impact."""
    return review is None or review.verdict != "high"


def review_title(review: ActionReview | None, folder: str) -> str:
    """The card's title for a high-impact call, or ``""``."""
    if review is None or review.verdict != "high" or not review.categories:
        return ""
    return TITLES.get(review.categories[0], "Run this?").format(folder=folder)


def because(review: ActionReview) -> str:
    """The reasons as one clause: "installs x; creates y" (no final stop)."""
    return "; ".join(r[:1].lower() + r[1:].rstrip(".") for r in review.reasons)


def block_words(review: ActionReview | None) -> str:
    """The first category in a few words ("installs software"), or ``""``."""
    if review is None or not review.categories:
        return ""
    return CATEGORY_WORDS.get(review.categories[0], "")


def _normalized(tool_name: str, arguments: Mapping[str, Any]) -> Any:
    """What a call does: Python as its syntax tree, a command as its exact text."""
    if tool_name == "python_exec":
        code = str(arguments.get("code") or "")
        try:
            return ["python", ast.dump(ast.parse(code))]
        except (SyntaxError, ValueError, RecursionError):
            return ["python-text", " ".join(code.split())]
    if tool_name == "shell_exec":
        command = arguments.get("command")
        # The exact text: a "#" inside a word, or a quote, changes what runs.
        if isinstance(command, list):
            return ["shell-argv", [str(part) for part in command]]
        return ["shell", str(command or "")]
    if tool_name in ("file_write", "file_edit"):
        rest = {k: v for k, v in arguments.items() if k != "path"}
        body = json.dumps(rest, sort_keys=True, ensure_ascii=False, default=str)
        path = os.path.realpath(str(arguments.get("path") or ""))
        return ["file", path, hashlib.sha256(body.encode()).hexdigest()]
    return [
        "call",
        json.loads(json.dumps(dict(arguments), sort_keys=True, default=str)),
    ]


def step_key(
    tool_name: str,
    arguments: Mapping[str, Any],
    workspace_root: str | None,
    inputs: Sequence[str] = (),
) -> str:
    """The hash of one step: the same call in the same folder, on the same files.

    Python is compared as its syntax tree, so a comment or a reflowed line
    is the same step; a shell command by its exact text; a file write by
    its resolved path and a hash of what it writes. *inputs* are the
    review's (:attr:`ActionReview.inputs`): the local files it read, with
    their digests. 32 hex characters.
    """
    root = os.path.realpath(workspace_root) if workspace_root else ""
    shape = [tool_name, root, _normalized(tool_name, arguments), sorted(inputs)]
    text = json.dumps(shape, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]

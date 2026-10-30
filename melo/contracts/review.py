"""The review of one tool call: routine, high-impact or undecided (#279).

The checker's rules (:mod:`localmelo.melo.checker.review`) classify each
``shell_exec``, ``python_exec`` and ``file_write`` call before an approval
policy decides. A session grant covers routine calls only: a high-impact
call asks every time and is never granted, and an undecided one asks
under a grant too. This module imports only the standard library.
"""

from __future__ import annotations

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
}
# The reason a call of the ``self`` category is denied without asking.
SELF_DENIAL: Final = "melo never lets a call change its own runtime, settings or keys"


@dataclass(frozen=True)
class ActionReview:
    """What the rules found in one call.

    ``categories`` are keys of :data:`TITLES`, strongest first; ``reasons``
    are at most three sentences for the person, each starting with a verb
    ("Installs playwright with pip."); ``paths`` are write paths outside
    the workspace that the call names.
    """

    verdict: Verdict
    categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    by: Literal["rules", "model"] = "rules"
    rules_version: int = RULES_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "categories": list(self.categories),
            "reasons": list(self.reasons),
            "paths": list(self.paths),
            "by": self.by,
            "rules_version": self.rules_version,
        }


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

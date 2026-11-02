"""No command or script is judged more leniently than it was.

``data/review_baseline.json`` holds what the rules said about a large set of
commands and scripts before the hardening of steps that reach macOS or destroy
work (rules version 2): the ones the review of the approval sentence was
checked against (scripting bridges, ``open``, wrappers, Python process calls),
the cases of this hardening, and the cases that later reviews of it found.
Every case is run again. A verdict may go from ``routine`` to ``undecided`` or
``high``, and a category may be added; it may never go the other way, and a
category the rules found must still be found.

When the rules version is raised again, regenerate the file from a checkout
of the release before the change (never from the working tree), so the
promise moves forward with the release and does not weaken.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import RULES_VERSION

DATA = json.loads((Path(__file__).parent / "data" / "review_baseline.json").read_text())
CASES: list[dict[str, Any]] = DATA["cases"]
STRICTNESS = {"routine": 0, "undecided": 1, "high": 2}


@pytest.fixture(scope="module")
def ws(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("home") / "ws"
    folder.mkdir()
    (folder / "script.sh").write_text("echo hi\n")
    return folder


def test_the_baseline_is_the_one_the_rules_started_from() -> None:
    assert DATA["rules_version"] == RULES_VERSION - 1
    assert len(CASES) >= 600
    assert len({case["id"] for case in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_a_case_is_never_judged_more_leniently(ws: Path, case: dict[str, Any]) -> None:
    found = Checker().review_action(
        ActionReviewPayload(
            case["tool"],
            case["arguments"],
            str(ws),
            home=str(ws.parent),
            executable="/opt/melo/bin/python3",
            prefix="/opt/melo",
        )
    )
    assert STRICTNESS[found.verdict] >= STRICTNESS[case["verdict"]], (
        case["id"],
        case["verdict"],
        found.verdict,
    )
    lost = set(case["categories"]) - set(found.categories)
    assert not lost, (case["id"], "no longer found:", sorted(lost))

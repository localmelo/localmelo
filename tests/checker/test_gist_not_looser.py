"""No warning of the approval card disappears.

``data/gist_baseline.json`` holds calls whose words (a ``why``) the card
warned about before the target comparison and the honest git wording were
added. Every call is run again; the warning may change its id (a step that
now shows a second effect), but it may never be gone: new rules about what
words cover must not hide a warning the card had.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.checker.review.gist import WARN_IDS, step_gist

DATA = json.loads((Path(__file__).parent / "data" / "gist_baseline.json").read_text())
CASES: list[dict[str, Any]] = DATA["cases"]


@pytest.fixture(scope="module")
def ws(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("home") / "ws"
    folder.mkdir()
    (folder / "script.sh").write_text("echo hi\n")
    return folder


def test_the_baseline_has_calls_of_every_kind() -> None:
    assert len(CASES) >= 80
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert {"delete", "send_out"} <= {case["warn"] for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_a_gist_warning_never_disappears(ws: Path, case: dict[str, Any]) -> None:
    review = Checker().review_action(
        ActionReviewPayload(
            case["tool"],
            case["arguments"],
            str(ws),
            home=str(ws.parent),
            executable="/opt/melo/bin/python3",
            prefix="/opt/melo",
        )
    )
    found = step_gist(case["tool"], case["arguments"], review)
    assert found is not None
    assert found.warn in WARN_IDS, (case["id"], case["warn"], "no warning now")

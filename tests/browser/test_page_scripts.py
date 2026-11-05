"""The page scripts, run in a JS engine (jsdom) instead of only checked for syntax.

``tests/browser/js/page_scripts.test.mjs`` holds the cases. They need ``node``
and the one dev dependency in ``tests/browser/js`` (``npm ci --prefix
tests/browser/js``); without them this file is skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from localmelo.support.browser.pagescript import (
    COMMON_JS,
    DISARM_JS,
    PAGE_JS,
    SETTLE_JS,
    act_script,
)

JS = Path(__file__).with_name("js")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (JS / "node_modules" / "jsdom").is_dir(),
    reason="needs node and jsdom (npm ci --prefix tests/browser/js)",
)


def test_the_page_scripts_do_what_they_claim_in_a_js_engine(tmp_path: Path) -> None:
    whole = act_script({})
    scripts = {
        "page": PAGE_JS,
        "common": COMMON_JS,
        # The act script up to where its spec goes: the test fills the spec in,
        # and the node lookups after it (null, or an object the test makes).
        "actPrefix": whole[: whole.rindex("})({") + 3],
        "settle": SETTLE_JS,
        "disarm": DISARM_JS,
    }
    exported = tmp_path / "scripts.json"
    exported.write_text(json.dumps(scripts), encoding="utf-8")
    ran = subprocess.run(
        ["node", "--test", str(JS / "page_scripts.test.mjs")],
        cwd=JS,
        env={**os.environ, "MELO_SCRIPTS": str(exported)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr

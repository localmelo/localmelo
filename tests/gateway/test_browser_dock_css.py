"""The live browser card docked at the top of the chat leaves the chat readable (#576).

The dock floats over the conversation (it takes no room, so docking never moves the
chat). Whatever scrolls under it must not show around it: no line cut off in the gap
above the card, none through its rounded corners, and none sliced at its bottom edge.
The stylesheet gives the card a backdrop the same width and a little taller on both
sides, in the chat's own colour, that fades out under the card.
"""

from __future__ import annotations

import re

from localmelo.support.gateway import webapp


def _rule(selector: str) -> dict[str, str]:
    css = (webapp._STATIC / "workspace.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    body = re.search(rf"(?:^|\}})\s*{re.escape(selector)}\s*\{{([^}}]*)\}}", css)
    assert body is not None, f"no rule for {selector}"
    return dict(
        (name.strip(), value.strip())
        for name, value in re.findall(r"([a-z-]+)\s*:\s*([^;]+);?", body.group(1))
    )


def test_the_dock_has_a_backdrop_that_hides_what_scrolls_under_it() -> None:
    dock = _rule(".bdock")
    backdrop = _rule(".bdock::before")
    # Behind the card, inside the sticky anchor, over the conversation.
    assert backdrop["position"] == "absolute"
    assert backdrop["z-index"] == "-1"
    # The chat's own colour from the top edge down, fading out below the card.
    assert backdrop["background"].startswith("linear-gradient(to bottom, var(--canvas)")
    assert backdrop["background"].endswith("transparent)")
    # Above the card the gap shows no half line; below it a line fades out instead of being sliced.
    assert backdrop["top"].startswith("calc(-1 *")
    assert backdrop["bottom"].startswith("calc(-1 *")
    # The card itself stays an overlay that takes no room.
    assert dock["position"] == "absolute"
    assert _rule(".bdock-anchor")["height"] == "0"

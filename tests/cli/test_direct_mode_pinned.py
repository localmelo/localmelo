"""Direct mode's functions are pinned: the interactive UI changed none of them.

Each function's source is hashed as main had it before the UI (``d232b9e``,
after #50 and P3). A change to one of them changes what the benchmark runs,
so it must come with its own review and a new hash here. R2a added the three
reasoning flags and their refusals, so ``_direct_options`` and
``_build_direct_mode_agent`` have new hashes; R2a's review fixes also moved
``_build_direct_mode_agent``'s memory-directory creation after the refusals.
R4 added ``--note-writer`` and the compress modes, which change the same two
functions again; R5's seven retrieval flags change them a third time, and its
review fixes moved the effective-recall check in ``_build_direct_mode_agent``
after a replay-requiring backend converts ``compress`` to ``compress+replay``,
so that combination is a usage error, not a traceback. R6-lite's two memory
flags and their config fallback change them a fourth. P5 moved ``--workspace``
into direct mode, so ``_direct_options`` resolves and checks the folder and
sets ``workspace_root``; it has a new hash. R2b's names-only ``[agent] tool_env``
config inheritance changes only ``_build_direct_mode_agent``; URL sessions and
default options stay unchanged. P13b's loop notice (``shell_exec`` is off;
``--builtins all`` turns it on) moved ``_run`` again, and ``_parse_builtins``
is pinned for the first time: it returns every built-in explicitly for
``--builtins all``. T65b's review fixes name ``melo "" --builtins all`` in
the notice, so ``_run`` has a new hash again. Worker history adds opt-in
``--save-session`` and import dispatch to ``_run_direct`` and ``_run``.
Context accounting resolves its opt-in budget before Agent construction, so
``_build_direct_mode_agent`` has a new hash. The opt-in worker brief routes
through ``_run_direct`` and ``_run``; review fixes handle broken worker pipes
in ``_run_direct``. Inbox routing changes both hashes again without changing
the ordinary direct path; ``--inbox-fd`` changes them once more. Fixed prefix
adds an opt-in flag and config fallback to ``_direct_options`` and
``_build_direct_mode_agent``; the default output and provider wire remain
covered by the legacy golden. Captured history also adds an opt-in flag and
config fallback to those two functions. Digest recall adds config-backed gate
settings only for explicit reasoning modes, and T95bR's review fixes make a
refused config value a worker ``startup_error`` after a valid brief, so
``_build_direct_mode_agent`` has a new hash; legacy behavior remains covered by
the golden and without ``--brief`` it keeps the old usage error. ``_check_args`` is not pinned.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parents[2] / "__main__.py"

PINNED = {
    "_run_direct": "97e9281d6ac524b259bdde8ec6b2aa2cff6ea3874e665c6d0fc46c48c3a3bcff",
    "_run": "03c29e4466d8bfe09f0edd6a47cd435d9eda3ceb05fe8c99783c96fb9781f519",
    "_run_one": "0c323f5d879739c205efe3ac0c71d651be2496721b68437ac30c93c6b6f70f31",
    "_report": "5ce2d43919e8bea8c5cfdb9689acd104e4bf65d2a8c377d9913f25b2cd909ecf",
    "_status_line": "51cf6b34fa09e121214b0a06e29e74d3d7d38d68b8e19ba5c68b0809abc1821b",
    "_exit_code": "bacf133c2381454b95644b216daebd09f279e75ff0496a319ef3c98c554432a5",
    "_one_line": "4dc03ea8c7f5f5896d9fd58887f5290dfbe1faf4e6286d272ef354567e8f7ef7",
    "_localmelo_version": "7c4a81e650ed6d8cc5ad1f6227b4ce75fe56028ebc7fa27ce5377fda6445db5d",
    "_build_direct_mode_agent": "8f29945472bcba2277cfcbb5bc4a45f329088fac484183065aa9af907a46ce89",
    "_direct_embedding": "67275e465432460acbb15d5b30649502641de4b37f1758723f066428662c6a7d",
    "_ExitSignals": "7859cea6c228917546ac80014d4745b789ca9c6d732f38aa949de9156f3ff34d",
    "_exit_by_signal": "d889130f2eb811f91d0a9134073847d7226adaa0b7b319499058f3c7d71f7f3b",
    "_direct_tool_providers": "be7d50021417d3f0c8ed549aa93ffbf400cd7aea2a3aa33c62987f5a008a3b8a",
    "_direct_options": "3a228a27542384b8bc78f57fb33b3c7b0dc109e66850c40247218c9373ace26c",
    "_parse_builtins": "fe4a7f24ab1f970fc094ebb737d29a9bec9de742165ae68743a312496dd98877",
    "_progress_line": "0c09c6422345a998c72b4c76c76ed8f69913ef7e7537427a7b687a4d98256d59",
    "_routine_name": "2e8cc3333e1b71fc20a3b3247fbe1c8af5b2101fa0448200ce50f46421bb6ea9",
    "_enable_line_editing": "dbabe09864704ee4e435380ce32e19234ce0a3cafb072a7cc39cf053a9904b07",
    "_prompt": "b0dff37b82143a746c012103f3ab29d477ba69b52bc918534e7ea9c959cddebc",
}


def _sources() -> dict[str, str]:
    text = MAIN.read_text(encoding="utf-8")
    return {
        node.name: ast.get_source_segment(text, node) or ""
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


@pytest.mark.parametrize("name", sorted(PINNED))
def test_a_direct_mode_function_is_unchanged(name: str) -> None:
    source = _sources()[name]
    assert hashlib.sha256(source.encode()).hexdigest() == PINNED[name]

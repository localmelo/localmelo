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
``--builtins all``. Review fixes name ``melo "" --builtins all`` in
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
settings only for explicit reasoning modes, and review fixes make a
refused config value a worker ``startup_error`` after a valid brief, so
``_build_direct_mode_agent`` has a new hash; legacy behavior remains covered by
the golden and without ``--brief`` it keeps the old usage error. ``--window-eviction``
adds an opt-in flag and config fallback to the same two functions (default ``slide``
changes nothing). ``--reflect-tool-guard`` adds an opt-in flag (no config
fallback) to ``_direct_options`` (default off changes nothing). The no-backend
message now points at the first-run flow, in both the plain and the ``--brief``
branch of ``_build_direct_mode_agent``, so it has a new hash. Recording every
direct run as a worker session by default adds a ``record`` argument to
``_run_direct`` and ``_run``, which turn it into the ``--save-session`` path; the
model's requests, stdout, JSON and exit codes are the same, as the legacy golden
and ``tests/cli/test_record_direct.py`` check. ``fixed_prefix``, ``window_eviction``
and ``reflect_tool_guard`` now default to on, ``chunk`` and on, so
``_build_direct_mode_agent`` takes all three from the config file when no flag names
them (an absent key is the new default, ``fixed_prefix = false`` stays off) and has a
new hash; ``--fixed-prefix off --window-eviction slide --reflect-tool-guard off``
send the earlier requests, which the legacy goldens check with those values.
``--cache-layout`` and ``--window-messages`` add two opt-in flags to ``_direct_options``
and a check of their config values to ``_build_direct_mode_agent`` (the defaults ``v2``
and 20 send today's requests), so both have new hashes. ``--brief-items`` adds one more
opt-in flag to ``_direct_options`` and a check of its config value to
``_build_direct_mode_agent`` (default off changes nothing), so both have new hashes
again. A ``--base-url`` run with no ``--chat-model`` asks the server for its models
before its first request (``_direct_model``, the interactive UI's choice), so
``_build_direct_mode_agent`` takes the model from it instead of ``default`` and has a
new hash; with ``--chat-model`` that call is not made and the requests are the same,
as ``tests/cli/test_direct_model_choice.py`` checks. The cache layout now defaults to
``v3`` and the window to 64 messages on a local server (20 on a cloud vendor), so
``_build_direct_mode_agent`` names the server the run talks to when it resolves the
cache settings (a URL counts as local) and accepts an unset window in the config, and
has a new hash; ``--cache-layout v2 --window-messages 20`` send the earlier requests,
which the legacy goldens check with those values. The experimental ``[agent] lean_output``
(no flag; ``-X agent.lean_output=true``) adds a check of its config value and its
application to a task, not to the ``melo ""`` loop, to ``_build_direct_mode_agent``,
which has a new hash; off, the default, sends the requests it sent, as
``tests/agent/test_lean_output.py`` pins. The developer setting ``[agent] confine_direct_runs``
(no flag; ``-X agent.confine_direct_runs=true``) adds a check of its config value and its
application to a plain direct run (a task or the ``melo ""`` loop, never ``--brief``) to
``_build_direct_mode_agent``, which has a new hash again; off, the default, leaves the
options and so the requests, stdout and exit codes as they were, as
``tests/cli/test_confine_direct_runs.py`` checks. ``[agent] tool_images`` and its two
counts (no flag; ``-X agent.tool_images=on``) add their config values to the options
in ``_build_direct_mode_agent``, which has a new hash; a run whose tools return no
image sends the same requests in every mode, and ``auto``, the default, sends no
pixels unless the server says the model reads images, as
``tests/agent/test_tool_images.py`` and ``tests/cli/test_tool_images_e2e.py`` check.
The developer setting ``[agent] answer_on_stop`` (no flag; ``-X agent.answer_on_stop=on``)
adds a check of its config value and its application to the options in
``_build_direct_mode_agent``, which has a new hash; off, the then default, leaves the options
and so the requests, stdout and exit codes as they were, as
``tests/agent/test_gate_patient.py`` checks. The setting now defaults to on, as the
reflection gate now defaults to ``patient`` (``AgentOptions``): ``_build_direct_mode_agent``
takes the config value whichever way it points (a new hash again, and the options object
is the same one while the value agrees), ``_direct_options`` is unchanged, and
``--reflection-gate flags -X agent.answer_on_stop=off`` send the earlier requests, as
``tests/agent/test_legacy_freeze.py`` and ``tests/cli/test_answer_on_stop_config.py`` check. The developer setting ``[agent] max_calls``
(no flag; ``-X agent.max_calls=30``) adds a check of its config value and its application
to the options in ``_build_direct_mode_agent``, which has a new hash again; unset, the
default, leaves the options and so the requests, stdout and exit codes as they were, as
``tests/agent/test_call_budget.py`` checks. A task's own built-ins (``git_status``,
``file_edit``) are left out of ``--builtins all``, so ``_parse_builtins`` has a new hash
and ``all`` still names the five built-ins it named, with the same requests.
``_check_args`` is not pinned.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parents[2] / "__main__.py"

PINNED = {
    "_run_direct": "1140042ce37895f4fa58eeaa48025283a1e96b26c48803e124cec59a58ca3aa3",
    "_run": "3c226c89202da0e724a9d77c419b78ae30ea548c2000abd023897e38e365a1a5",
    "_run_one": "0c323f5d879739c205efe3ac0c71d651be2496721b68437ac30c93c6b6f70f31",
    "_report": "5ce2d43919e8bea8c5cfdb9689acd104e4bf65d2a8c377d9913f25b2cd909ecf",
    "_status_line": "51cf6b34fa09e121214b0a06e29e74d3d7d38d68b8e19ba5c68b0809abc1821b",
    "_exit_code": "bacf133c2381454b95644b216daebd09f279e75ff0496a319ef3c98c554432a5",
    "_one_line": "4dc03ea8c7f5f5896d9fd58887f5290dfbe1faf4e6286d272ef354567e8f7ef7",
    "_localmelo_version": "7c4a81e650ed6d8cc5ad1f6227b4ce75fe56028ebc7fa27ce5377fda6445db5d",
    "_build_direct_mode_agent": "e63a11d7f6528c1e3c0f95900a525ab8f85627d27baff707b13579d58b2a2f67",
    "_direct_embedding": "67275e465432460acbb15d5b30649502641de4b37f1758723f066428662c6a7d",
    "_ExitSignals": "7859cea6c228917546ac80014d4745b789ca9c6d732f38aa949de9156f3ff34d",
    "_exit_by_signal": "d889130f2eb811f91d0a9134073847d7226adaa0b7b319499058f3c7d71f7f3b",
    "_direct_tool_providers": "be7d50021417d3f0c8ed549aa93ffbf400cd7aea2a3aa33c62987f5a008a3b8a",
    "_direct_options": "4ca334b646b0410e488f768afbe1ae8864706641c81a81ff67df7c019f6379bf",
    "_parse_builtins": "93c1b46e6db728f4114df712300ae06fccffd830e5d6e5df49da47abaa3fc460",
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

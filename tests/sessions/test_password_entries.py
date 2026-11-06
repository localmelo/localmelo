"""A password typed into a session's runner or the terminal's prompt is hidden once (#547).

The runner is where every surface's text comes in (the terminal calls
``run_turn(text, title=text)``): the title, the stored text, the next turn's
earlier turns and what a review reads after a reload hold the handle, never the
value. The prompt's history file keeps the handle too.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from localmelo.melo.contracts.sign_in import VAULT
from localmelo.support.cli.app import open_history
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.sessions.test_runner_earlier import (  # noqa: F401
    Chat,
    _reply,
    store,
)

PASSWORD = "hunter2-Secret"
SAY = f"sign in to forum.example, my password is {PASSWORD}"


def dump(path: Path) -> str:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return "\n".join(conn.iterdump())


async def test_the_title_the_stored_text_and_the_next_turns_context_hold_the_handle(
    store: SessionStore, tmp_path: Path  # noqa: F811
) -> None:
    chat = Chat(store, None, tmp_path / "w")
    try:
        await chat.say(SAY, "Signed in.")
        await chat.say("and now?", "Done.")
        everything = dump(tmp_path / "sessions.db") + "".join(
            str(call["messages"]) for call in chat.model.calls
        )
        assert PASSWORD not in everything + repr(chat.kwargs)
        assert "my password is [password-1]" in everything
        assert VAULT.value("[password-1]") == PASSWORD
        # The earlier turns the next turn's run is given hold the handle too.
        assert chat.kwargs[-1]["earlier_turns"][0][0].endswith("[password-1]")
    finally:
        await chat.close()


async def test_a_turn_the_person_did_not_send_is_run_as_not_their_message(
    store: SessionStore, tmp_path: Path  # noqa: F811
) -> None:
    """A task another agent made, started with ``from_person=False``, gives the run no words of theirs (#547)."""
    chat = Chat(store, None, tmp_path / "w")
    try:
        chat.model.plans.append(_reply("Done."))
        await chat.runner.start_turn(SAY, from_person=False)
        await chat.runner.wait()
        assert chat.kwargs[-1]["from_person"] is False
        # Not the person's message: the store keeps no typed text for it.
        assert chat.runner.session_id is not None
        [turn] = await store.list_turns(chat.runner.session_id)
        assert PASSWORD not in dump(tmp_path / "sessions.db")
        assert not turn.typed_text
    finally:
        await chat.close()


def test_the_prompts_history_file_keeps_the_handle_not_the_password(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history"
    history = open_history(str(path))
    history.append_string(SAY)
    history.append_string("what is in this folder")
    kept = path.read_text()
    assert PASSWORD not in kept and "my password is [password-1]" in kept
    assert "what is in this folder" in kept

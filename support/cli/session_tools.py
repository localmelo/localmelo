"""Session editing, inspection and explicit clipboard commands for the terminal."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from localmelo.melo.schema import INTERRUPT_LABEL, NO_ANSWER_LABEL, split_thinking
from localmelo.support.cli import memory_import, memory_notes, terminal
from localmelo.support.cli.session_views import memory_view, usage_view
from localmelo.support.cli.text import cut_cells
from localmelo.support.sessions.display import clean_output
from localmelo.support.sessions.store import TurnInfo, UnknownSessionError, check_title
from localmelo.support.sessions.sync import SessionMark

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession


class SessionTools:
    def __init__(self, session: CliSession) -> None:
        self.session = session
        self.pending_title: str | None = None

    def reset(self) -> None:
        self.pending_title = None

    async def turns(
        self, *, limit: int | None = None, before: int | None = None
    ) -> list[TurnInfo]:
        runner = self.session.runner
        if runner is None or runner.session_id is None:
            return []
        return await self.session.store.list_turns(
            runner.session_id, limit=limit, before=before
        )

    async def run(self, name: str, arguments: str) -> None:
        s = self.session
        if name == "memory" and arguments not in ("", "window", "import", "undo"):
            raise ValueError(
                "Use /memory, /memory import, /memory undo, or /memory window "
                "for the model's window."
            )
        if arguments and name not in ("title", "copy", "memory"):
            raise ValueError(f"Use /{name}.")
        if name == "title":
            await self.title(arguments)
        elif name == "undo":
            await self.undo()
        elif name == "usage":
            s.out.print(usage_view(await self.turns(), width=s.out.width))
        elif name == "memory" and not arguments:
            if s.state != "idle":
                raise ValueError(
                    "/memory opens when the turn ends; /memory window works now."
                )
            await memory_notes.manage(s)
        elif name == "memory" and arguments == "import":
            if s.state != "idle":
                raise ValueError("/memory import opens when the turn ends.")
            await memory_import.bring_in(s)
        elif name == "memory" and arguments == "undo":
            if s.remote is None:
                raise ValueError(
                    "/memory undo works in the main chat, where melo says what it tidied."
                )
            await s.remote.undo_sleep()
        elif name == "memory":
            loaded = s.runner.last_resume if s.runner is not None else None
            s.out.print(
                memory_view(
                    s.agent.conversation(),
                    s.factory.backend,
                    window_max=s.agent.options.window_messages,
                    memory=s.factory.memory,
                    loaded=loaded,
                    width=s.out.width,
                    symbols=s.renderer.symbols,
                )
            )
        elif name == "copy":
            await self.copy(arguments)

    async def title(self, text: str) -> None:
        s = self.session
        runner = s.runner
        if text:
            if not 1 <= len(text.strip()) <= 200:
                raise ValueError("A title has 1 to 200 characters.")
            text = check_title(text)
            self.pending_title = text
            s.title = text
            await self.apply_title()
        elif runner is not None and runner.session_id is not None:
            info = await s.store.get_session(runner.session_id)
            if s.runner is runner:
                if info is None:
                    raise UnknownSessionError("The chat was deleted.")
                s.title = info.title
        s.out.print(s._text(f"Title: {s.title or 'New chat'}"))

    async def apply_title(self) -> None:
        s, text = self.session, self.pending_title
        runner = s.runner
        if text is None or runner is None or runner.session_id is None:
            return
        if not await s.store.rename_session(runner.session_id, text):
            raise UnknownSessionError("The chat was deleted.")
        if s.runner is runner and self.pending_title == text:
            self.pending_title = None
            s.title = text
        await s.navigation.refresh_refs()

    def _remember_undo(self, turn: TurnInfo) -> None:
        s = self.session
        mark = s.navigation.mark
        s.navigation.mark = SessionMark(mark.foreign_newest, mark.undone + 1)
        if s.app is not None:
            buffer = s.app.buffer
            buffer.history.append_string(turn.typed_text or turn.user_text)
            buffer.reset(document=buffer.document)

    async def undo(self) -> None:
        s = self.session
        if s.runner is None:
            raise ValueError("Nothing to undo.")
        await s.navigation.synchronize()
        work = asyncio.ensure_future(self._undo_and_report())
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            while True:
                try:
                    await asyncio.shield(work)
                    break
                except asyncio.CancelledError:
                    if work.cancelled():
                        raise
            raise

    async def _undo_and_report(self) -> None:
        s = self.session
        assert s.runner is not None
        turn = await s.runner.undo()
        if turn is None:
            raise ValueError("Nothing to undo.")
        self._remember_undo(turn)
        line = ((turn.typed_text or turn.user_text).splitlines() or [""])[0]
        first = cut_cells(clean_output(line).strip(), 60, s.renderer.symbols.ellipsis)
        ending = "" if first.endswith((".", "!", "?", "。", "！", "？", "…")) else "."
        s.out.print(
            s._text(
                f"Undid turn {turn.turn}: {first}{ending} Files the tools changed stay changed."
            )
        )
        await s.navigation.refresh_refs()

    async def copy(self, argument: str) -> None:
        if argument and (
            len(argument) > 19
            or not argument.isascii()
            or not argument.isdecimal()
            or int(argument) < 1
        ):
            raise ValueError("Use /copy [turn].")
        selected = None
        if argument:
            number = int(argument)
            before = number + 1 if number < 2**63 - 1 else None
            turns = await self.turns(limit=1, before=before)
            if turns and turns[0].turn == number:
                selected = turns[0]
        else:
            before = None
            while turns := await self.turns(limit=50, before=before):
                selected = next(
                    (
                        t
                        for t in reversed(turns)
                        if not t.undone and t.status != "running"
                    ),
                    None,
                )
                if selected is not None:
                    break
                before = turns[0].turn
        if selected is None:
            raise ValueError(
                f"Turn {argument} has no answer."
                if argument
                else "No answer to copy yet."
            )
        answer = split_thinking(selected.answer, strip_answer=False)[1]
        if (
            selected.status == "running"
            or not answer.strip()
            or answer.strip()
            in (
                INTERRUPT_LABEL,
                NO_ANSWER_LABEL.format(status=selected.status),
            )
        ):
            raise ValueError(f"Turn {selected.turn} has no answer.")
        mode = await terminal.copy_text(answer, self.session.out.write_control)
        if mode == "pbcopy":
            text = (
                f"✓ Copied the answer of turn {selected.turn} ({len(answer):,} chars)."
            )
        else:
            text = f"Sent the answer of turn {selected.turn} ({len(answer):,} chars) to the terminal clipboard (OSC 52; support varies)."
        self.session.out.print(self.session._text(text))

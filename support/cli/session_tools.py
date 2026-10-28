"""Session editing, inspection and explicit clipboard commands for the terminal."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from localmelo.melo.schema import INTERRUPT_LABEL, NO_ANSWER_LABEL, split_thinking
from localmelo.support.cli import terminal
from localmelo.support.cli.session_views import memory_view, usage_view
from localmelo.support.sessions.store import TurnInfo, UnknownSessionError
from localmelo.support.sessions.sync import SessionMark

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession


class SessionTools:
    def __init__(self, session: CliSession) -> None:
        self.session = session
        self.pending_title: str | None = None

    def reset(self) -> None:
        self.pending_title = None

    async def turns(self) -> list[TurnInfo]:
        runner = self.session.runner
        if runner is None or runner.session_id is None:
            return []
        return await self.session.store.list_turns(runner.session_id)

    async def run(self, name: str, arguments: str) -> None:
        s = self.session
        if arguments and name not in ("title", "copy"):
            raise ValueError(f"Use /{name}.")
        if name == "title":
            await self.title(arguments)
        elif name == "undo":
            await self.undo()
        elif name == "usage":
            s.out.print(usage_view(await self.turns()))
        elif name == "memory":
            loaded = s.runner.last_resume if s.runner is not None else None
            s.out.print(
                memory_view(
                    s.agent.conversation(),
                    s.factory.backend,
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
            if not 1 <= len(text) <= 200:
                raise ValueError("A title has 1 to 200 characters.")
            self.pending_title = text
            s.title = text
            await self.apply_title()
        elif runner is not None and runner.session_id is not None:
            info = await s.store.get_session(runner.session_id)
            if s.runner is runner:
                if info is None:
                    raise UnknownSessionError("The session was deleted.")
                s.title = info.title
        s.out.print(s._text(f"Title: {s.title or 'New session'}"))

    async def apply_title(self) -> None:
        s, text = self.session, self.pending_title
        runner = s.runner
        if text is None or runner is None or runner.session_id is None:
            return
        if not await s.store.rename_session(runner.session_id, text):
            raise UnknownSessionError("The session was deleted.")
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
        work = asyncio.ensure_future(s.runner.undo())
        try:
            turn = await asyncio.shield(work)
        except asyncio.CancelledError:
            turn = await work
            if turn is not None:
                self._remember_undo(turn)
            raise
        if turn is None:
            raise ValueError("Nothing to undo.")
        self._remember_undo(turn)
        first = " ".join((turn.typed_text or turn.user_text).split())[:60]
        s.out.print(
            s._text(
                f"Undid turn {turn.turn}: {first}. Files the tools changed stay changed."
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
        turns = await self.turns()
        selected = (
            next((t for t in reversed(turns) if t.turn == int(argument)), None)
            if argument
            else next(
                (t for t in reversed(turns) if not t.undone and t.status != "running"),
                None,
            )
        )
        if selected is None:
            raise ValueError(
                f"Turn {argument} has no answer."
                if argument
                else "No answer to copy yet."
            )
        answer = split_thinking(selected.answer)[1]
        if (
            selected.status == "running"
            or not answer
            or answer
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

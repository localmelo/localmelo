"""Navigation and synchronization of the terminal's saved chat sessions."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from localmelo.support.cli.render import ToolOutput
from localmelo.support.cli.views import resume_recap
from localmelo.support.sessions.display import home_label
from localmelo.support.sessions.runner import ResumeLoader, ResumeMode, SessionRunner
from localmelo.support.sessions.store import (
    SessionInfo,
    UnknownSessionError,
    process_holder,
)
from localmelo.support.sessions.sync import SessionMark, session_mark

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

DELETED = "This chat was deleted in another window. /new starts a new one; your draft is kept."
BUSY = "This chat is running a turn in another melo window. Wait for it to finish, or start a new chat with /new."


class SessionNavigation:
    """One resolver for startup, /resume and /sessions; never opens the web file."""

    def __init__(self, session: CliSession, mode: ResumeMode) -> None:
        self.session, self.mode = session, mode
        self.mark = SessionMark(0, 0)
        self.refs: tuple[str, ...] = ()

    async def startup(
        self,
        continue_session: bool,
        ref: str | None,
        on_choice: Callable[[SessionInfo | None], None] | None = None,
    ) -> None:
        """Choose first, announce the chosen session, then restore its conversation."""
        s = self.session
        announced = False
        try:
            info: SessionInfo | None = None
            note = ""
            if continue_session:
                rows = await s.store.list_sessions(
                    workspace=s.agent.workspace_root,
                    limit=1,
                    min_turns=1,
                )
                info = rows[0] if rows else None
                if info is None:
                    note = f"No earlier chat in {home_label(s.agent.workspace_root or '')}; starting a new one."
            elif ref:
                if not await self.resolve(ref):
                    print(
                        f"melo: no chat matches '{ref}' (melo -r lists them)",
                        file=sys.stderr,
                    )
                    s.exit_code = 2
                    assert s.app is not None
                    s.app.exit()
                    return
                info = await self.choose(ref=ref)
            elif await s.store.list_sessions(
                workspace=s.agent.workspace_root, limit=1, min_turns=1
            ):
                info = await self.choose()
            else:
                note = f"No chats yet in {home_label(s.agent.workspace_root or '')}."
            if on_choice is not None:
                on_choice(info)
            announced = True
            if note:
                s.out.print(s._text(note))
            if info is not None:
                await self.resume(info)
        except asyncio.CancelledError:
            if not announced and not s._closed and on_choice is not None:
                on_choice(None)
            raise
        except (ValueError, UnknownSessionError) as exc:
            if not announced and on_choice is not None:
                on_choice(None)
            message = DELETED if isinstance(exc, UnknownSessionError) else str(exc)
            s.out.print(s._text(f"✗ {message}", style="error"))
        finally:
            s._state, s._work = "idle", None

    async def refresh_refs(self) -> None:
        rows = await self.session.store.list_sessions(limit=20, min_turns=1)
        self.refs = tuple(
            value for row in rows for value in (row.id, row.title) if value
        )

    async def resolve(self, ref: str, *, exact_only: bool = False) -> list[SessionInfo]:
        store = self.session.store
        rows = [
            s for s in await store.find_sessions(ref) if s.kind == "chat" and s.turns
        ]
        if not rows and not exact_only:
            rows = await store.list_sessions(query=ref, limit=50, min_turns=1)
        return rows

    async def choose(
        self,
        *,
        ref: str | None = None,
        query: str = "",
        choices: list[SessionInfo] | None = None,
    ) -> SessionInfo | None:
        """Resolve or pick an identity without changing the active conversation."""
        s = self.session
        if ref:
            choices = await self.resolve(ref, exact_only=True)
            if len(choices) == 1:
                return choices[0]
            choices = choices or await self.resolve(ref)
            if not choices:
                raise ValueError(f"No chat matches '{ref}'.")
        assert s.app is not None
        return await s.app.choose_session(
            s.store,
            s.agent.workspace_root or "",
            query=query,
            choices=choices,
        )

    async def select(
        self,
        *,
        ref: str | None = None,
        query: str = "",
        choices: list[SessionInfo] | None = None,
    ) -> bool:
        info = await self.choose(ref=ref, query=query, choices=choices)
        if info is None:
            return False
        await self.resume(info)
        return True

    async def resume(self, info: SessionInfo) -> None:
        s = self.session
        candidate = SessionRunner(
            s.agent,
            s.store,
            info.id,
            on_event=s.on_event,
            progress=True,
            resume_mode=self.mode,
            before_turn=s._prepare_turn,
        )
        previous = s.agent.conversation()
        try:
            mark = await session_mark(s.store, info.id, holder=process_holder("cli"))
            turns = await s.store.list_turns(info.id)
            current = await s.store.get_session(info.id)
            if current is None:
                raise UnknownSessionError(f"Unknown session: {info.id}")
            if current.kind != "chat" or not current.turns:
                raise ValueError("No finished turns in this chat.")
            await candidate.resume()
            await self.metadata(current)
        except BaseException:
            if s.agent.conversation() != previous:
                s.agent.load_conversation(previous)
            await candidate.close()
            raise
        old, s.runner = s.runner, candidate
        self.mark = mark
        s.approvals.reset()
        s.session_tools.reset()
        s.title = current.title
        s.renderer.clear_session()
        s.renderer.context_tokens = 0
        assert candidate.last_resume is not None
        s.out.print(
            resume_recap(
                current,
                turns,
                candidate.last_resume,
                width=s.out.width,
                symbols=s.renderer.symbols,
                developer=s.developer,
            )
        )
        if old is not None:
            await old.close()
        await self.refresh_refs()

    async def metadata(self, info: SessionInfo) -> None:
        s = self.session
        workspace = s.agent.workspace_root
        if info.workspace != workspace or info.model != s.factory.backend.model:
            await s.store.update_session(
                info.id,
                workspace=workspace,
                model=s.factory.backend.model,
            )
            if info.workspace != workspace:
                s.out.print(
                    s._text(
                        f"This chat last worked in {home_label(info.workspace or '')}; "
                        f"it now works in {home_label(workspace or '')}.",
                    )
                )

    async def synchronize(self, reload: ResumeLoader | None = None) -> None:
        s = self.session
        if s.runner is None or s.runner.session_id is None:
            return
        sid = s.runner.session_id
        info = await s.store.get_session(sid)
        if info is None:
            raise UnknownSessionError(f"Unknown session: {sid}")
        s.title = info.title
        mark = await session_mark(s.store, sid, holder=process_holder("cli"))
        if mark != self.mark:
            await (reload or s.runner.resume)()
            s.out.print(
                s._text(
                    "◆ This chat changed in another window; the latest turns "
                    "were reloaded into the model's memory.",
                )
            )
        self.mark = mark
        await self.metadata(info)

    async def tool_outputs(self, argument: str) -> list[ToolOutput]:
        """Read missing call output in durable event order, including pages past 1,000."""
        s = self.session
        calls = list(s.renderer.tool_outputs)
        if s.runner is None or s.runner.session_id is None:
            return calls
        sid = s.runner.session_id
        turns = [t for t in await s.store.list_turns(sid) if not t.undone]
        prefix, dot, number = argument.partition(".")
        if dot and prefix.isdigit() and number.isdigit():
            wanted = [int(prefix)]
        elif argument and not argument.isdigit():
            return calls
        else:
            wanted = [t.turn for t in reversed(turns)]
        for turn in wanted:
            if any(c.ref.startswith(f"{turn}.") for c in calls):
                break
            after, count = 0, 0
            while True:
                events = await s.store.list_events(
                    sid, turns=(turn, turn), after_id=after
                )
                for stored in events:
                    event = stored.event
                    if event.get("type") != "tool.finished":
                        continue
                    data = event.get("data", {})
                    count += 1
                    status = str(data.get("status") or "")
                    error = str(data.get("error") or "")
                    text = (
                        error
                        if error and status != "success"
                        else str(data.get("output") or "")
                    )
                    calls.append(
                        ToolOutput(
                            f"{turn}.{count}",
                            str(data.get("name") or "?"),
                            "",
                            status,
                            text,
                        )
                    )
                if len(events) < 1000:
                    break
                last_id = events[-1].id
                assert last_id is not None  # list_events returns durable events
                after = last_id
            if count or dot:
                break
        return sorted(
            calls, key=lambda call: tuple(int(n) for n in call.ref.split("."))
        )

    async def saved_hint(self) -> str:
        """The exit line for a saved chat: how to come back to it.

        ``melo -c`` when it is the newest chat of this folder and melo ran in
        that folder, so ``-c`` finds it; its id for ``melo -r`` otherwise.
        """
        s = self.session
        if s.runner is not None and s.runner.session_id is not None:
            info = await s.store.get_session(s.runner.session_id)
            if info is not None and info.turns:
                workspace = s.agent.workspace_root
                newest = await s.store.list_sessions(
                    workspace=workspace, limit=1, min_turns=1
                )
                if newest and newest[0].id == info.id and _runs_in(workspace):
                    return f'◆ Saved "{info.title}". Continue it with: melo -c'
                return f'◆ Saved "{info.title}". Resume it with: melo -r {info.id}'
        return ""


def _runs_in(workspace: str | None) -> bool:
    """Whether the process runs in *workspace*, the folder ``melo -c`` looks in."""
    try:
        return workspace is not None and os.path.realpath(
            workspace
        ) == os.path.realpath(os.getcwd())
    except OSError:  # the folder it started in is gone
        return False

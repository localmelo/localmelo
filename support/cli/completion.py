"""Completion of slash commands, with each command's line in the menu."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.containers import Window, WindowRenderInfo
from prompt_toolkit.layout.controls import UIContent
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.layout.menus import CompletionsMenu, CompletionsMenuControl

from localmelo.support.cli.commands import COMMANDS
from localmelo.support.cli.text import cells, cut_cells
from localmelo.support.cli.theme import Symbols

# Cells before a name: the mark and a space, so the names stand under the
# typed text (the prompt is one cell and a space too).
LEAD = 2


class AttachmentCompletion(Completion):
    """A file explicitly chosen from the completion menu."""

    def __init__(self, path: str, start_position: int) -> None:
        super().__init__(
            f"@{path}", start_position=start_position, display_meta="attach"
        )
        self.path = path


class SlashCompleter(Completer):
    """Completes ``/`` and a command name as the draft's start, then ``/model``'s names.

    *models* gives the ids of the latest model listing, offered after
    ``/model``.
    """

    def __init__(
        self,
        models: Callable[[], Sequence[str]] = lambda: (),
        sessions: Callable[[], Sequence[str]] = lambda: (),
    ) -> None:
        self._models = models
        self._sessions = sessions
        self.workspace: Callable[[], str | None] = lambda: None

    def _paths(self, text: str) -> Iterable[Completion]:
        match = re.search(r"(?<!\S)@([^\s@]*)$", text)
        root = self.workspace()
        if match is None or not root:
            return
        prefix = match.group(1)
        if os.path.isabs(prefix) or any(
            part in (".", "..") for part in prefix.split("/")
        ):
            return
        directory, partial = os.path.split(prefix)
        root = os.path.realpath(root)
        folder = os.path.join(root, directory)
        # The loader refuses all symlinks; don't reveal their contents in the menu.
        current = root
        for part in directory.split("/") if directory else ():
            current = os.path.join(current, part)
            if os.path.islink(current):
                return
        try:
            with os.scandir(folder) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    if not entry.name.startswith(partial) or entry.is_symlink():
                        continue
                    path = os.path.join(directory, entry.name)
                    if entry.is_file(follow_symlinks=False):
                        yield AttachmentCompletion(path, -len(prefix) - 1)
                    elif entry.is_dir(follow_symlinks=False):
                        yield Completion(f"@{path}/", start_position=-len(prefix) - 1)
        except OSError:
            return

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            yield from self._paths(text)
            return
        if "\n" in text:
            return
        name, space, argument = text[1:].partition(" ")
        if not space:
            for command in COMMANDS:
                # The bare "/" lists each command once; an alias row appears
                # only for a typed prefix of it (the description names it).
                spellings = (
                    (command.name, *command.aliases) if name else (command.name,)
                )
                for alias in spellings:
                    if alias.startswith(name):
                        usage = f"/{alias}"
                        yield Completion(
                            usage,
                            start_position=-len(text),
                            display=usage,
                            display_meta=command.line(alias),
                        )
            return
        if name in ("verbose", "thinking"):
            argument = argument.lstrip()
            for value in ("on", "off"):
                if value.startswith(argument) and argument != value:
                    yield Completion(value, start_position=-len(argument))
            return
        if name in ("model", "resume"):
            argument = argument.lstrip()
            values = self._models() if name == "model" else self._sessions()
            for model in values:
                if model.startswith(argument):
                    yield Completion(model, start_position=-len(argument))


class _Scrollbar(Margin):
    """A slim bar of *glyph*, drawn only when the rows overflow the window."""

    def __init__(self, glyph: str) -> None:
        self._glyph = glyph

    def get_width(self, get_ui_content: Callable[[], UIContent]) -> int:
        return 1

    def create_margin(
        self, window_render_info: WindowRenderInfo, width: int, height: int
    ) -> StyleAndTextTuples:
        total, rows = (
            window_render_info.content_height,
            window_render_info.window_height,
        )
        if total <= rows:
            return []
        size = max(1, rows * rows // total)
        top = round((rows - size) * window_render_info.vertical_scroll / (total - rows))
        fragments: StyleAndTextTuples = []
        for row in range(rows):
            if row:
                fragments.append(("", "\n"))
            on = top <= row < top + size
            fragments.append(
                ("class:scrollbar.button", self._glyph) if on else ("", " ")
            )
        return fragments


class _ListControl(CompletionsMenuControl):
    """The rows: ``❯ /model   Pick a model``, the chosen one marked like a picker's."""

    def __init__(self, mark: str, ellipsis: str) -> None:
        self._mark, self._ellipsis = mark, ellipsis

    def _get_menu_width(self, max_width: int, complete_state: CompletionState) -> int:
        widest = max(cells(item.display_text) for item in complete_state.completions)
        return min(max_width, max(self.MIN_WIDTH, LEAD + widest + 1))

    def create_content(self, width: int, height: int) -> UIContent:
        state = get_app().current_buffer.complete_state
        if not state:
            return UIContent()
        names = self._get_menu_width(width, state)
        notes = self._get_menu_meta_width(width - names, state)

        def get_line(row: int) -> StyleAndTextTuples:
            item = state.completions[row]
            kind = ".current" if row == state.complete_index else ""
            name = cut_cells(item.display_text, names - LEAD - 1, self._ellipsis)
            lead = f"{self._mark} " if kind else " " * LEAD
            line: StyleAndTextTuples = [
                (
                    f"class:completion-menu.completion{kind}",
                    lead + name + " " * (names - LEAD - cells(name)),
                )
            ]
            if notes:
                note = cut_cells(item.display_meta_text, notes - 2, self._ellipsis)
                line.append(
                    (
                        f"class:completion-menu.meta.completion{kind}",
                        f" {note}" + " " * (notes - 1 - cells(note)),
                    )
                )
            return line

        return UIContent(
            get_line=get_line,
            cursor_position=Point(x=0, y=state.complete_index or 0),
            line_count=len(state.completions),
        )


class CompletionList(CompletionsMenu):
    """The completion list below the input: no block, the theme's colours.

    prompt_toolkit's menu with the picker's mark on the chosen row and a slim
    scrollbar that shows only when there are more rows than fit.
    """

    def __init__(
        self, symbols: Symbols, *, max_rows: int, scroll_offset: int = 1
    ) -> None:
        super().__init__(max_height=max_rows, scroll_offset=scroll_offset)
        window = cast(Window, self.content)
        window.content = _ListControl(symbols.selection, symbols.ellipsis)
        window.right_margins = [_Scrollbar(symbols.gutter)]

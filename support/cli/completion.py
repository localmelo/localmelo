"""Completion of slash commands, with each command's line in the menu."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from localmelo.support.cli.commands import COMMANDS


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

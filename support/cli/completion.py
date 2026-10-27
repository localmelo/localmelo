"""Completion of slash commands, with each command's line in the menu."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from localmelo.support.cli.commands import COMMANDS


class SlashCompleter(Completer):
    """Completes ``/`` and a command name as the draft's start, then ``/model``'s names.

    *models* gives the ids of the latest model listing, offered after
    ``/model``.
    """

    def __init__(self, models: Callable[[], Sequence[str]] = lambda: ()) -> None:
        self._models = models

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        name, space, argument = text[1:].partition(" ")
        if not space:
            for command in COMMANDS:
                for alias in (command.name, *command.aliases):
                    if alias.startswith(name):
                        usage = f"/{alias}"
                        yield Completion(
                            usage,
                            start_position=-len(text),
                            display=f"{usage} {command.args}".rstrip(),
                            display_meta=command.summary,
                        )
            return
        if name == "model":
            argument = argument.lstrip()
            for model in self._models():
                if model.startswith(argument):
                    yield Completion(model, start_position=-len(argument))

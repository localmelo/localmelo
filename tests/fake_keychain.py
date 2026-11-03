"""A stand-in for the macOS ``security`` tool, for tests of the keychain route.

It understands the four commands melo sends and keeps the items in memory. It
records every call so a test can check that a key was never on a command line.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

LOCKED_MESSAGE = "security: SecKeychainSearchCopyNext: User interaction is not allowed."
NOT_FOUND_MESSAGE = (
    "security: SecKeychainSearchCopyNext: The specified item could not be found "
    "in the keychain."
)


@dataclass
class FakeSecurity:
    """The ``keychain._security`` hook backed by a dict of ``(service, account)``."""

    items: dict[tuple[str, str], str] = field(default_factory=dict)
    calls: list[tuple[list[str], str | None]] = field(default_factory=list)
    locked: bool = False

    def __call__(
        self, args: list[str], stdin: str | None = None
    ) -> tuple[int, str, str]:
        self.calls.append((list(args), stdin))
        if self.locked:
            return 36, "", LOCKED_MESSAGE
        if args[:1] == ["-i"]:
            return self._interactive(stdin or "")
        return self._run(args)

    def _interactive(self, text: str) -> tuple[int, str, str]:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                words = shlex.split(line)
            except ValueError as exc:
                return 1, "", f"security: {exc}"
            code, out, err = self._run(words)
            if code != 0:
                return code, out, err
        return 0, "", ""

    def _run(self, words: list[str]) -> tuple[int, str, str]:
        command, rest = words[0], words[1:]
        options = {
            rest[i]: rest[i + 1]
            for i in range(len(rest) - 1)
            if rest[i].startswith("-")
        }
        item = (options.get("-s", ""), options.get("-a", ""))
        if command == "add-generic-password":
            if item in self.items and "-U" not in rest:
                return (
                    45,
                    "",
                    "security: The specified item already exists in the keychain.",
                )
            self.items[item] = options["-w"]
            return 0, "", ""
        if command == "find-generic-password":
            if item not in self.items:
                return 44, "", NOT_FOUND_MESSAGE
            return 0, self.items[item] + "\n" if "-w" in rest else "attributes\n", ""
        if command == "delete-generic-password":
            if item not in self.items:
                return 44, "", NOT_FOUND_MESSAGE
            del self.items[item]
            return 0, "", ""
        return 1, "", f'security: unknown command "{command}"'

    def argv_text(self) -> str:
        """Every argument of every call, as one string."""
        return "\n".join(" ".join(args) for args, _ in self.calls)

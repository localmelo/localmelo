"""A built-in tool as a function of text, for tests of the tools that never return pictures.

A built-in's type says it may return :class:`ToolOutput` (``file_read`` of a PDF
does, with the pictures of its pages); ``shell_exec`` and ``python_exec`` never
do, and a test of their text reads it through :func:`text_tool`.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from localmelo.melo.executor.builtins import ToolFunction

TextTool = Callable[..., Coroutine[Any, Any, str]]


def text_tool(tool: ToolFunction) -> TextTool:
    """*tool* as a function whose result is text; a result that is not text fails the test."""

    async def call(*args: Any, **kwargs: Any) -> str:
        result = await tool(*args, **kwargs)
        assert isinstance(result, str), result
        return result

    return call

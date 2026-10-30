"""Memory across chats: records of what the user said, an index of every chat, recall.

See :class:`~localmelo.support.memory.service.ChatMemory`. The files live in
the memory folder (``~/.cache/localmelo/memory`` by default), mode 0700:
``records.db`` and ``index.db``.
"""

from __future__ import annotations

from localmelo.support.memory.service import ChatMemory, MemoryTools, TurnMemory

__all__ = ["ChatMemory", "MemoryTools", "TurnMemory"]

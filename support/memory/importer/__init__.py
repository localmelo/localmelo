"""Bring in what other coding assistants on this computer remember.

:func:`build_plan` scans files under HOME that other assistants keep (their
instruction files, memory notes and memory databases, :mod:`.sources`), drops
what is not the user's to keep (:mod:`.rules`) and returns an
:class:`ImportPlan`; :func:`select` and :func:`write` turn the chosen sources
into notes. :class:`~localmelo.support.memory.service.ChatMemory` wraps them.
"""

from __future__ import annotations

from localmelo.support.memory.importer.plan import (
    ImportPlan,
    Note,
    build_plan,
    select,
    write,
)

__all__ = ["ImportPlan", "Note", "build_plan", "select", "write"]

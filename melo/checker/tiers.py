"""Permission tiers for tools: the fixed table and the MCP derivation.

A tier is a label, not a denial: :func:`tier_requires_approval` says
whether a run with an approval policy asks before a call of that tier.
``read`` is the only tier that never asks by itself. The tier table wins
over a tool's own claims, and an MCP server's annotations are only hints.

This module imports :mod:`localmelo.melo.schema` only, so the checker,
the executor and the agent can all read a tier without an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from localmelo.melo.schema import ToolDef, ToolTier

# The tier of every built-in, by name. A test keeps this set equal to
# builtin_names(), so a new built-in cannot be added without a tier.
BUILTIN_TIERS: Mapping[str, ToolTier] = {
    "file_read": "read",
    "list_files": "read",
    "file_write": "write",
    "file_edit": "write",
    "git_status": "read",
    "shell_exec": "exec",
    "python_exec": "exec",
}


def tier_of(tool_def: ToolDef) -> ToolTier:
    """The permission tier of *tool_def*.

    The fixed table wins over the tool's name, so a provided or custom tool
    that reuses a built-in's name inherits that built-in's tier. Every other
    tool is derived from its MCP annotations; with none, it is ``write``.
    """
    builtin = BUILTIN_TIERS.get(tool_def.name)
    if builtin is not None:
        return builtin
    return tier_from_annotations(tool_def.annotations)


def tier_from_annotations(annotations: Mapping[str, Any]) -> ToolTier:
    """The tier an MCP tool's annotations claim, cautiously read.

    Only the three boolean hints matter here: ``openWorldHint`` outranks
    the local categories, then ``readOnlyHint`` (unless the tool also
    claims ``destructiveHint``) gives ``read``, and anything else is
    ``write``. A value that is not a boolean is ignored, as is a missing
    hint. ``idempotentHint`` says nothing about side effects and is unused.
    """
    read_only = annotations.get("readOnlyHint")
    destructive = annotations.get("destructiveHint")
    open_world = annotations.get("openWorldHint")
    read_only = read_only if isinstance(read_only, bool) else None
    destructive = destructive if isinstance(destructive, bool) else None
    open_world = open_world if isinstance(open_world, bool) else None
    if open_world is True:
        # Reaches outside this machine, read-only or not.
        return "network"
    if read_only is True and destructive is not True:
        # readOnly, unless it also claims to be destructive.
        return "read"
    # Anything else, including no usable annotation at all: a tool is not
    # read-only unless it says so, which is the MCP default too.
    return "write"


def tier_requires_approval(tier: ToolTier) -> bool:
    """Whether a call of a tool of *tier* asks an approval policy."""
    return tier != "read"


__all__ = [
    "BUILTIN_TIERS",
    "tier_from_annotations",
    "tier_of",
    "tier_requires_approval",
]

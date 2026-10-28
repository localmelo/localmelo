"""The permission tiers: the built-in table and the MCP derivation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from localmelo.melo.checker import BUILTIN_TIERS, Checker, tier_of
from localmelo.melo.checker.tiers import tier_from_annotations, tier_requires_approval
from localmelo.melo.executor import builtin_names
from localmelo.melo.schema import ToolDef


def _tool(name: str = "mcp__web__fetch", **annotations: Any) -> ToolDef:
    return ToolDef(
        name=name,
        description="",
        parameters={},
        annotations=annotations,
    )


class TestTheBuiltInTable:
    def test_every_builtin_has_its_tier(self) -> None:
        assert BUILTIN_TIERS == {
            "file_read": "read",
            "file_write": "write",
            "shell_exec": "exec",
            "python_exec": "exec",
        }

    def test_the_table_names_exactly_the_builtins(self) -> None:
        # A new built-in cannot be added without a tier.
        assert set(BUILTIN_TIERS) == builtin_names()

    def test_the_table_wins_over_a_tools_annotations(self) -> None:
        assert tier_of(_tool("file_read", openWorldHint=True)) == "read"
        assert tier_of(_tool("file_write", readOnlyHint=True)) == "write"
        assert tier_of(_tool("python_exec", readOnlyHint=True)) == "exec"

    def test_the_checker_exposes_the_derivation(self) -> None:
        checker = Checker()
        assert checker.tier(_tool("file_read")) == "read"
        assert checker.tier(_tool("mcp__web__fetch")) == "write"


class TestTheMcpDerivation:
    @pytest.mark.parametrize(
        ("annotations", "tier"),
        [
            # Not read-only, local: write.
            ({}, "write"),
            ({"readOnlyHint": False}, "write"),
            ({"destructiveHint": False}, "write"),
            ({"destructiveHint": True}, "write"),
            # Explicitly read-only, local: read.
            ({"readOnlyHint": True}, "read"),
            ({"readOnlyHint": True, "destructiveHint": False}, "read"),
            # Contradictory: the cautious reading wins.
            ({"readOnlyHint": True, "destructiveHint": True}, "write"),
            # Open world outranks the local categories.
            ({"openWorldHint": True}, "network"),
            ({"openWorldHint": True, "readOnlyHint": True}, "network"),
            ({"openWorldHint": True, "destructiveHint": True}, "network"),
            ({"openWorldHint": False, "readOnlyHint": True}, "read"),
            # A hint that is not a boolean is ignored, never fatal.
            ({"readOnlyHint": "true"}, "write"),
            ({"readOnlyHint": 1}, "write"),
            ({"readOnlyHint": None}, "write"),
            ({"openWorldHint": "yes"}, "write"),
            ({"openWorldHint": 1}, "write"),
            ({"destructiveHint": "yes", "readOnlyHint": True}, "read"),
            # idempotentHint says nothing about side effects.
            ({"idempotentHint": True}, "write"),
            ({"idempotentHint": True, "readOnlyHint": True}, "read"),
            # Unknown keys are kept for the future and do not decide.
            ({"title": "Fetch", "extra": {"nested": True}}, "write"),
        ],
        ids=[
            "absent",
            "read-only-false",
            "destructive-false",
            "destructive-true",
            "read-only-true",
            "read-only-not-destructive",
            "read-only-and-destructive",
            "open-world",
            "open-world-read-only",
            "open-world-destructive",
            "open-world-false-read-only",
            "read-only-string",
            "read-only-int",
            "read-only-none",
            "open-world-string",
            "open-world-int",
            "destructive-string-read-only",
            "idempotent",
            "idempotent-read-only",
            "unknown-keys",
        ],
    )
    def test_the_truth_table(self, annotations: dict[str, Any], tier: str) -> None:
        assert tier_from_annotations(annotations) == tier
        assert tier_of(_tool(**annotations)) == tier

    def test_a_non_bool_hint_does_not_hide_the_others(self) -> None:
        assert (
            tier_from_annotations(
                {"openWorldHint": "yes", "readOnlyHint": True, "destructiveHint": False}
            )
            == "read"
        )


class TestApproval:
    @pytest.mark.parametrize(
        ("tier", "asks"),
        [("read", False), ("write", True), ("exec", True), ("network", True)],
    )
    def test_read_is_the_only_tier_that_does_not_ask(
        self, tier: Any, asks: bool
    ) -> None:
        assert tier_requires_approval(tier) is asks

    def test_a_custom_tool_is_write_and_asks(self) -> None:
        tool = _tool("deploy")
        assert tier_of(tool) == "write"
        assert tier_requires_approval(tier_of(tool))


class TestTheCopy:
    def test_the_mapping_is_copied_on_construction(self) -> None:
        annotations: dict[str, Any] = {"readOnlyHint": True}
        tool = ToolDef(
            name="mcp__web__fetch",
            description="",
            parameters={},
            annotations=annotations,
        )
        annotations["readOnlyHint"] = False
        assert tool.annotations == {"readOnlyHint": True}

    def test_the_copy_really_copies(self) -> None:
        # The mapping is read-only by contract, but the definition must stay
        # copyable: a MappingProxyType would make this raise.
        tool = _tool(readOnlyHint=True)
        assert copy.deepcopy(tool).annotations == {"readOnlyHint": True}

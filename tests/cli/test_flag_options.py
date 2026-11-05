"""The table of flags that set an AgentOptions field."""

from __future__ import annotations

import dataclasses

import pytest

from localmelo import __main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support.flag_options import FLAG_OPTIONS, flag_options


def test_every_row_names_a_flag_and_an_option_field() -> None:
    dests = {action.dest for action in cli._build_parser()._actions}
    fields = {field.name for field in dataclasses.fields(AgentOptions)}
    assert {row.dest for row in FLAG_OPTIONS} <= dests
    assert {row.field for row in FLAG_OPTIONS} <= fields
    assert len({row.dest for row in FLAG_OPTIONS}) == len(FLAG_OPTIONS)


def test_the_direct_mode_flags_are_the_rows_and_the_flags_without_one() -> None:
    assert set(cli._DIRECT_FLAGS) == {
        *(row.dest for row in FLAG_OPTIONS),
        *cli._OTHER_DIRECT_FLAGS,
    }
    assert cli._DIRECT_FLAGS["recall_min_score"] == "--recall-min-score"


def test_a_given_flag_sets_its_field_and_the_rest_set_nothing() -> None:
    args = cli._build_parser().parse_args(["--timeout", "9", "--memory", "on", "task"])
    assert flag_options(args) == {"timeout_s": 9, "memory": True}
    assert flag_options(cli._build_parser().parse_args(["task"])) == {}


@pytest.mark.parametrize(
    ("argv", "field", "value"),
    [
        (["--recall-min-score", "0"], "recall_min_score", 0.0),  # 0 is a value
        (["--keep-decision-text"], "keep_decision_text", True),
    ],
)
def test_a_value_that_is_falsy_still_counts_as_given(
    argv: list[str], field: str, value: object
) -> None:
    args = cli._build_parser().parse_args([*argv, "task"])
    assert flag_options(args) == {field: value}

"""Whether a name is a person's: ``people.person``, shared by the importer and the
chat's owner check."""

from __future__ import annotations

import pytest

from localmelo.support.memory import people
from localmelo.support.memory.importer import rules


@pytest.mark.parametrize(
    "name",
    [
        "Marco",
        "Tom Briggs",
        "Anne-Marie",
        "McKay",
        "Dr. Tanaka",
        "马可",
        "小宇",
        "妈妈",
        "同事",
    ],
)
def test_a_name_is_a_person(name: str) -> None:
    assert people.person(name)


@pytest.mark.parametrize(
    "name",
    [
        "Python",
        "Git",
        "GitHub",
        "Makefile",
        "README",
        "English",
        "开发",
        "项目需要",
        "默认",
    ],
)
def test_a_language_a_tool_a_file_or_a_modifier_is_no_person(name: str) -> None:
    assert not people.person(name)


def test_the_importer_keeps_using_the_same_gate() -> None:
    assert rules.person is people.person

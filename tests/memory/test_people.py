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
        "小明",
        "老王",
        "阿强",
        "王小明",
        "老师",
        "老板",
        "阿姨",
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
        # Longer than a 小/老/阿 nickname, or an everyday word that starts like one.
        "小程序",
        "老版本",
        "小朋友",
        "小时",
        "小说",
        "小组",
        "小区",
        "小心",
        "小型",
        "小数",
        "小学",
        "老家",
        "老式",
        "阿里",
    ],
)
def test_a_language_a_tool_a_file_or_a_modifier_is_no_person(name: str) -> None:
    assert not people.person(name)


def test_the_importer_keeps_using_the_same_gate() -> None:
    assert rules.person is people.person

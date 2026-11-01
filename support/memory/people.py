"""Whether a name is a person's: the gate on a possessive or a subject.

"Marco's studio" is a fact about another person; "Python's GIL", "Git's index" and
"开发的代码" are not. :func:`person` says which of the two a name before a
possessive is. The importer files a note under People with it
(:mod:`localmelo.support.memory.importer.rules`) and the chat's owner check reads
a note's owner with it (:func:`localmelo.support.memory.records.wrong_owner`).
This module imports nothing of :mod:`localmelo.support.memory`, so both can use it.
"""

from __future__ import annotations

import re

# What a possessive is not a person's name: a language, a tool or command, a file.
_NOT_PEOPLE = re.compile(
    r"english|chinese|mandarin|cantonese|japanese|korean|french|german|spanish|italian"
    r"|portuguese|russian|arabic|hindi|dutch|swedish|latin|python|rust|swift|java"
    r"|javascript|typescript|kotlin|ruby|perl|lua|haskell|scala|julia|golang|matlab"
    r"|fortran|bash|zsh|shell|git|github|gitlab|docker|make|cmake|conda|pip|pipx|uv"
    r"|npm|node|nvm|brew|homebrew|xcode|vim|emacs|neovim|tmux|ssh|curl|pytest|ruff"
    r"|black|mypy|linux|macos|ubuntu|windows|android|safari|chrome|firefox|claude"
    r"|codex|gemini|hermes|openai|anthropic|slurm|kubernetes|terraform|ansible"
    r"|direnv|poetry|cargo|gradle|maven|webpack|react|django|flask|fastapi|pandas"
    r"|numpy|pytorch|tensorflow|jupyter|makefile|dockerfile|readme|license|changelog"
    r"|gemfile|procfile|macbook|apple|google|microsoft|amazon|meta|slack|notion"
    r"|obsidian|figma|jira|hugging",
    re.IGNORECASE,
)
_ZH_RELATIVES = re.compile(
    "妈妈|爸爸|哥哥|姐姐|弟弟|妹妹|妻子|丈夫|女儿|儿子|父母|家人|女朋友|男朋友"
    "|导师|同事|同学|朋友|室友|医生|教授"
)
# Common surnames, less the ones that start everyday words (高效, 方法, 文件, 金融).
_ZH_NAME = re.compile(
    "(?:[小老阿]|[王李张刘陈杨黄赵周吴徐孙胡朱林郭马罗梁宋郑谢韩唐冯董萧程曹袁邓傅沈彭吕苏卢"
    "蒋蔡贾丁魏薛叶阎潘杜戴汪姜谭廖邹熊陆郝崔邱秦邵孟侯赖龚顾])[\u4e00-\u9fff]{1,2}"
)


def _word_of_name(word: str) -> bool:
    """Whether *word* is written like a name: "Marco", "Anne-Marie", "McKay"; not
    "README", "GitHub" or "iPhone"."""
    for part in word.split("-"):
        part = re.sub(r"^(?:Mc|Mac)(?=[A-Z])", "", part)
        if not (part[:1].isupper() and part[1:].isalpha() and part[1:].islower()):
            return False
    return True


def person(owner: str) -> bool:
    """Whether *owner*, the name before a possessive, can be a person.

    A capitalised name that is no language, tool, command, file or code
    identifier ("Python's", "Git's", "Makefile's", "GitHub's" are not people); a
    Chinese name with a surname or 小/老/阿, or a relative or colleague ("妈妈", "同事").
    """
    if re.search("[\u4e00-\u9fff]", owner):
        return bool(_ZH_RELATIVES.fullmatch(owner) or _ZH_NAME.fullmatch(owner))
    words = re.sub(r"^(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.\s+", "", owner).split()
    return bool(words) and all(
        _word_of_name(w) and not _NOT_PEOPLE.fullmatch(w) for w in words
    )

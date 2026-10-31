"""What is not kept, and what topic a kept line gets (no model, rules only).

A line is dropped when it holds a secret, private data of someone else (an
email address that is not the user's own, an account or Apple ID, a phone
number, a named person with an ID) or is about the other tool itself (its tool
calls, slash commands and settings), which says nothing about the user.
"""

from __future__ import annotations

import re
from pathlib import Path

from localmelo.support.memory import records
from localmelo.support.memory.importer.sources import (
    MAX_FILE_BYTES,
    Statement,
    readable,
)

# A "feedback" topic does not exist yet; the lines that would be one are preferences.
FEEDBACK_TOPIC = "feedback" if "feedback" in records.TOPICS else "preferences"
_MECHANICS = re.compile(
    r"\b(?:Edit|Write|Read|Bash|Grep|Glob|MultiEdit|NotebookEdit|WebFetch|WebSearch)"
    r" tools?\b"
    r"|\b(?:TaskStop|TaskCreate|SendMessage|ToolSearch|ExitPlanMode|EnterWorktree)\b"
    r"|(?<![\w/~.])/(?:compact|clear|memory|init|resume|config|permissions|hooks|mcp"
    r"|agents|model|login|logout|status|help|cost|review|doctor)\b(?!/)"
    r"|\bMCP\b|\bmcpServers\b|settings(?:\.local)?\.json|\b(?:Pre|Post)ToolUse\b"
    r"|\.claude/(?:hooks|commands|agents|skills)|\bpermissions?\.(?:allow|deny)\b"
    r"|\ballowedTools\b|\bbypassPermissions\b|\bhooks? (?:config|settings)\b"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_ACCOUNT = re.compile(
    r"\bApple ?ID\b|苹果\s?ID|\b(?:Apple|iCloud|Google|GitHub|Microsoft) account\b"
    r"|\baccount (?:id|number|name)\b|账号|账户",
    re.IGNORECASE,
)
_PHONE = re.compile(
    r"\+\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,}"
    r"|(?<!\d)1[3-9]\d{9}(?!\d)"
    r"|(?<!\d)\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"
)
_ID = re.compile(
    r"(?:\b(?:ID|UID|SSN|passport|student id|employee id)\b|学号|工号|身份证|护照)"
    r"[\s:：=#]*[A-Za-z0-9-]{4,}",
    re.IGNORECASE,
)
_PERSON = re.compile(
    r"\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b(?:my|her|his) (?:advisor|sister|brother|mother|father"
    r"|friend|wife|husband|colleague|student)\b|导师|同事|朋友|妻子|老婆|老公|室友|同学",
)
_FEEDBACK = re.compile(
    r"^(?:please\s+)?(?:always|never|don'?t|do not|use|prefer|keep|put|run|write|avoid"
    r"|ask|only|no|stop|reply|respond|check|make|read)\b"
    r"|\b(?:must|should|never|always|avoid)\b"
    r"|不要|不得|必须|应该|优先|一律|禁止|务必|尽量|别(?!人)|请|只|先|用(?!户)|放|写|跑|保持",
    re.IGNORECASE,
)
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
_ABOUT = re.compile(
    r"^(?:the user\b|user\b|用户|我(?!们)|i(?:'m|’m| am| have| live| work)\b|my\b)",
    re.IGNORECASE,
)


def own_email(home: Path) -> str:
    """The user's ``user.email`` from the git config under *home*; ``""`` without."""
    for rel in (".gitconfig", ".config/git/config"):
        path = readable(home / rel, home, limit=MAX_FILE_BYTES)
        if path is None:
            continue
        lines = path.read_text("utf-8", errors="ignore").splitlines()
        user = False
        for line in (item.strip() for item in lines):
            if line.startswith("["):
                user = line.lower().startswith("[user]")
            elif user and (
                found := re.match(r"email\s*=\s*(\S+)", line, re.IGNORECASE)
            ):
                return found[1].lower()
    return ""


def reject(text: str, email: str) -> str:
    """Why *text* is not kept: ``secret``, ``third_party`` or ``mechanics``; ``""`` if it is."""
    if records.sensitive(text):
        return "secret"
    emails = {m.lower() for m in _EMAIL.findall(text)}
    if (
        emails - {email}
        or _ACCOUNT.search(text)
        or _PHONE.search(text)
        or (_ID.search(text) and _PERSON.search(text))
    ):
        return "third_party"
    return "mechanics" if _MECHANICS.search(text) else ""


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


def topic_for(st: Statement) -> str:
    """The topic of a kept line, from its words and its note file's ``type``."""
    text = st.text
    if st.kind in ("project", "reference"):
        return "projects"
    if records.topic_of(text, request=False) == "decisions":
        return "decisions"
    owner = records._owner(text)  # "Marco's studio …": a fact about another person
    if owner and owner != "user" and person(owner):
        return "people"
    instruction = _FEEDBACK.search(text) is not None or st.kind == "feedback"
    about = _ABOUT.match(text) or (st.kind == "user" and not instruction)
    if about and not records._PREFERENCE.search(text):
        return "about"
    return FEEDBACK_TOPIC if instruction else "preferences"

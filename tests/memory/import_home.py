"""A synthetic HOME for the import tests: files of several assistants, invented content.

Nothing here is read from the real home. :func:`build` returns the HOME folder
(under *tmp*) and fills it with every kind of file the import knows, plus the
ones it must refuse: a link out of HOME, a file over the size cap, credentials.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

OWN = "me@example.test"
SECRET = "sk-ant-api03-" + "Ab1" * 12
LEAK = "LEAK-LINE this sentence lives outside the home folder and must never be read."
CLAUDE_MD = f"""# Machine rules
@~/.claude/rules/python.md
@{{outside}}/leak.md
@rules/linked.md
@rules/big.md
@~/.codex/auth.json

## Environment
- Use uv for new Python projects.
- Never add a Co-Authored-By line to commits.
- Use the Edit tool to change files.
- Run /compact when the context is long.
- Configure MCP servers in settings.json.
- The deploy key is {SECRET}.
- The wifi password is hunter2.
- Ask Dr. Priya Rao (student ID 4821937) before changing grades.
- Contact backup@other.test about outages.
- Write to {OWN} for questions about the lab.
- Call +1 412 555 0187 if the build breaks.
- The developer account uses a friend's Apple ID.
- I am a PhD student in robotics.

## Notes
Models live under ~/Developer/ml_source.
[A link alone](docs/x.md)
Why: because of an old incident.
```
Edit tool inside a code block is not a statement.
```
"""
CHAIN = [
    "Prefer tabs in Makefiles.",
    "Name branches after issue numbers.",
    "Store screenshots beside the page they show.",
    "Freeze every dependency version.",
    "Archive finished experiments monthly.",
]
AGENTS_MD = """# Working rules
## Git
- Use uv for new Python projects.
- Keep commits small and focused on one change.
- Reply to the user with the SendMessage tool.
"""
GEMINI_MD = """# 语言
- 和我对话可以用中文。
- 代码注释一律用英文。
- 新项目优先使用 uv 管理环境。
"""
USER_MD = "Xu prefers concise replies.§I am a PhD student in robotics.§Reach Maya Chen at maya.chen@other.test"
MEMORY_MD = (
    "The lab cluster uses SLURM with the gpu partition.§ \n§Use /memory to edit notes."
)


def note(kind: str, body: str, name: str = "n") -> str:
    return f"---\nname: {name}\ndescription: d\ntype: {kind}\n---\n{body}\n"


def lines(tag: int, count: int) -> str:
    return "\n".join(
        f"- Remember zeta{tag}x{n} handles qq{tag}y{n} near ww{tag}z{n}."
        for n in range(count)
    )


def put(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build(tmp: Path) -> Path:
    home, outside = tmp / "fake-home", tmp / "outside"
    home.mkdir()
    put(outside / "leak.md", LEAK + "\n")
    put(
        home / ".gitconfig",
        f"[core]\n\tbare = false\n[user]\n\tname = Test User\n\temail = {OWN}\n",
    )
    claude = home / ".claude"
    put(claude / "CLAUDE.md", CLAUDE_MD.replace("{outside}", str(outside)))
    for n, text in enumerate(CHAIN, 1):
        nxt = f"\n@deep{n + 1}.md" if n < len(CHAIN) else ""
        name = "python.md" if n == 1 else f"deep{n}.md"
        put(claude / "rules" / name, f"- {text}{nxt}\n")
    (claude / "rules" / "linked.md").symlink_to(outside / "leak.md")
    put(
        claude / "rules" / "big.md",
        ("- Padding line that is long enough to count. " * 8 + "\n") * 800,
    )
    app = home / "work" / "app"
    app.mkdir(parents=True)
    key = "".join(c if c.isalnum() else "-" for c in str(app))
    memory = claude / "projects" / key / "memory"
    put(memory / "MEMORY.md", "- [Role](user_role.md) — what the user does\n")
    put(
        memory / "user_role.md",
        note("user", "The user is a robotics researcher.\n**Why:** told directly."),
    )
    put(
        memory / "feedback_style.md",
        note(
            "feedback", "Keep answers short and plain.\nUse uv for new Python projects."
        ),
    )
    put(
        memory / "project_app.md",
        note("project", "The app keeps its fixtures under tests/data."),
    )
    put(
        memory / "ref_board.md",
        note("reference", "Open bugs are tracked on the team board."),
    )
    gone = claude / "projects" / (key.replace("app", "vanished")) / "memory"
    put(
        gone / "project_old.md",
        note("project", "The vanished app used a legacy build."),
    )
    profile = home / ".cc-switch" / "claude-accounts" / "a1" / "config"
    put(
        profile / "CLAUDE.md",
        "@~/.claude/CLAUDE.md\n- Prefer plain words over jargon.\n",
    )
    put(
        home / ".cc-switch" / "claude-accounts" / "a2" / "config" / "CLAUDE.md",
        CLAUDE_MD.replace("{outside}", str(outside)),
    )
    put(home / ".codex" / "AGENTS.md", AGENTS_MD)
    put(home / ".codex" / "auth.json", '{"token": "CODEX-TOKEN-NEVER-READ"}')
    put(home / ".codex" / "history.jsonl", "CODEX-HISTORY-NEVER-READ\n")
    db = sqlite3.connect(home / ".codex" / "memories_1.sqlite")
    db.execute(
        "CREATE TABLE stage1_outputs (thread_id TEXT PRIMARY KEY, raw_memory TEXT)"
    )
    db.execute(
        "INSERT INTO stage1_outputs VALUES ('t1', '## Testing\n- Prefers pytest over unittest.\n- Run the Bash tool for builds.')"
    )
    db.execute("INSERT INTO stage1_outputs VALUES ('t2', NULL)")
    db.commit()
    db.close()
    put(home / ".gemini" / "GEMINI.md", GEMINI_MD)
    put(home / ".hermes" / "memories" / "USER.md", USER_MD)
    put(home / ".hermes" / "memories" / "MEMORY.md", MEMORY_MD)
    return home

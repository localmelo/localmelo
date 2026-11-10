"""The parts of the agent pages that the code owns, written from the code.

`docs/for-agents.md` holds regions `<!-- agent-docs:ID -->...<!-- /agent-docs -->`
that `sync` fills in: `help` is `melo -h all` (every option, command word and
setting with its default) and `routes` is every HTTP route the gateway serves,
read from its route decorators, with the reference section that documents it.
`docs/llms.txt` is written whole from `docs/manifest.json`: one line per page
that has a `summary`.

    uv run python scripts/agent_docs.py sync [--check]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

try:
    from localmelo.scripts.build_docs import _slug
except ImportError:  # run as `python scripts/agent_docs.py`
    from build_docs import _slug  # type: ignore[no-redef,import-not-found]

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://localmelo.github.io/localmelo/"
METHODS = {"get", "post", "put", "patch", "delete", "websocket"}

_REGION = re.compile(
    r"<!-- agent-docs:(?P<id>[\w-]+) -->(?P<body>.*?)<!-- /agent-docs -->", re.S
)
_PLACEHOLDER = re.compile(r"\\\{[^}]*\\\}")


def help_block() -> str:
    from localmelo.support import option_tiers

    return "```text\n" + option_tiers.help_all() + "```"


def routes(root: Path = ROOT) -> list[tuple[str, str]]:
    """Every (METHOD, path) the gateway serves, in source order, by file name."""
    found: list[tuple[str, str]] = []
    for path in sorted((root / "support" / "gateway").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Attribute)
                    and deco.func.attr in METHODS
                    and deco.args
                    and isinstance(deco.args[0], ast.Constant)
                    and isinstance(deco.args[0].value, str)
                ):
                    method = deco.func.attr.upper()
                    found.append(
                        ("WS" if method == "WEBSOCKET" else method, deco.args[0].value)
                    )
    return found


def _headings(text: str) -> Iterator[tuple[str, str]]:
    """(heading text, anchor) of each Markdown heading outside code fences."""
    used: dict[str, int] = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if fenced or not line.startswith("#"):
            continue
        title = line.lstrip("#").strip().replace("`", "")
        base = _slug(title)
        count = used.get(base, 0)
        used[base] = count + 1
        yield title, f"{base}-{count}" if count else base


def _where(method: str, path: str, pages: dict[str, str]) -> str:
    """A link to the reference section that documents the route, or ``—``."""
    route = _PLACEHOLDER.sub(r"\\{[^}]*\\}", re.escape(path)) + r"(?![\w{/])"
    with_method = re.compile(rf"\b{method} `?{route}")
    for name, text in pages.items():
        for title, anchor in _headings(text):
            if with_method.search(title):
                return f"[{name}](reference/{name}.md#{anchor})"
    for pattern in (with_method, re.compile(route)):
        for name, text in pages.items():
            if pattern.search(text):
                return f"[{name}](reference/{name}.md)"
    return "—"


def routes_block(root: Path = ROOT) -> str:
    pages = {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted((root / "docs" / "reference").glob("*.md"))
    }
    rows = ["| Route | Documented in |", "|---|---|"]
    rows += [
        f"| `{m} {p}` | {_where(m, p, pages)} |"
        for m, p in sorted(routes(root), key=lambda r: (r[1], r[0]))
    ]
    return "\n".join(rows)


def _title(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        match = re.search(r"<title>(.*?)</title>", text, re.S)
        title = match.group(1) if match else path.stem
        return title.split("·")[0].split("|")[0].strip()
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def llms_text(root: Path = ROOT) -> str:
    manifest = json.loads((root / "docs" / "manifest.json").read_text())
    lines = [
        "# localmelo",
        "",
        "> localmelo is a local personal agent: it runs tasks with tools and memory "
        "on a model server you run or on a hosted model.",
        "> The `melo` command runs one task with fixed exit codes and a JSON line, "
        "or opens a chat; a gateway on 127.0.0.1 serves a web console and an HTTP API.",
        "> It is pre-alpha: interfaces may change before v0.1.0. "
        f"Agents start at [localmelo for agents]({SITE}for-agents.md).",
    ]
    pages = [p for p in manifest["pages"] if p.get("summary") and p["en_source"]]
    for section in sorted(manifest["sections"], key=lambda s: s["order"]):
        listed = sorted(
            (p for p in pages if p["section"] == section["id"]),
            key=lambda p: p["order"],
        )
        if not listed:
            continue
        lines += ["", f"## {section['title']}", ""]
        for page in listed:
            source = page["en_source"]
            title = _title(root / source)
            url = SITE + source.removeprefix("docs/")
            lines.append(f"- [{title}]({url}): {page['summary']}")
    return "\n".join(lines) + "\n"


def wanted(root: Path = ROOT) -> dict[Path, str]:
    """Each generated file and the text it must hold."""
    makers = {"help": help_block, "routes": lambda: routes_block(root)}
    page = root / "docs" / "for-agents.md"

    def fill(match: re.Match[str]) -> str:
        name = match.group("id")
        return f"<!-- agent-docs:{name} -->\n{makers[name]()}\n<!-- /agent-docs -->"

    return {
        page: _REGION.sub(fill, page.read_text(encoding="utf-8")),
        root / "docs" / "llms.txt": llms_text(root),
    }


def sync(check: bool = False, root: Path = ROOT) -> list[str]:
    """Rewrite the generated files; with *check*, only report what is stale."""
    stale = []
    for path, text in wanted(root).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(str(path.relative_to(root)))
            if not check:
                path.write_text(text, encoding="utf-8", newline="\n")
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("command", choices=("sync",))
    parser.add_argument("--check", action="store_true", help="only report stale files")
    args = parser.parse_args(argv)
    stale = sync(check=args.check)
    for name in stale:
        print(("stale: " if args.check else "wrote: ") + name)
    if args.check and stale:
        print("Run: uv run python scripts/agent_docs.py sync", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

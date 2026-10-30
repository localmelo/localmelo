"""Confined writes (#261): the profile, the run's temp folder, refusals, fail-closed.

The policy tests need no sandbox. The tests marked ``real`` start children
under ``sandbox-exec`` and skip where a profile cannot be applied (off
macOS, or inside a sandbox that refuses another). None writes outside
``tmp_path``: the "home folder" of a refused write is a folder there, the
child's ``HOME``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localmelo.melo.executor import sandbox
from localmelo.melo.executor.builtins import ToolFunction, rooted_builtins
from localmelo.melo.executor.env import child_env
from localmelo.melo.executor.process_gate import PROCESS_RECORD
from localmelo.melo.executor.sandbox import (
    TMP_DIR,
    ConfinementError,
    WriteConfinement,
    allowed_paths,
    cache_dirs,
    confine,
    git_dirs,
    profile,
    quote,
)

HOME = "/Users/someone"


def _confinement(root: str = "/ws", *paths: str) -> WriteConfinement:
    return WriteConfinement(
        root=root, tmp=f"{root}/{TMP_DIR}", paths=(root, *paths), profile=""
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ]
        + list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A repository with one commit, and a linked worktree of it."""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / "a.txt").write_text("a\n")
    _git(main, "add", "a.txt")
    _git(main, "commit", "-q", "-m", "a")
    tree = tmp_path / "tree"
    _git(main, "worktree", "add", "-q", "-b", "work", str(tree))
    return main.resolve(), tree.resolve()


class TestProfile:
    def test_paths_are_quoted(self) -> None:
        assert quote('/a "b"\\c') == '"/a \\"b\\"\\\\c"'
        text = profile(['/w") (allow file-write* (subpath "/'])
        assert '(subpath "/w\\") (allow file-write* (subpath \\"/")' in text

    @pytest.mark.parametrize("char", ["\n", "\t", "\x00", "\x7f"])
    def test_a_control_character_refuses(self, char: str) -> None:
        with pytest.raises(ConfinementError, match="control character"):
            profile([f"/ws{char}x"])

    def test_the_profile_denies_writes_but_the_allowlist(self) -> None:
        lines = profile(["/ws", "/c"]).splitlines()
        assert lines[:4] == [
            "(version 1)",
            "(allow default)",
            "(deny file-write*)",
            "(allow file-write*",
        ]
        assert [line.strip() for line in lines[4:]] == [
            *(f'(literal "{device}")' for device in sandbox.DEVICES),
            '(subpath "/dev/fd")',
            '(regex #"^/dev/ttys[0-9]+$")',
            '(subpath "/ws")',
            '(subpath "/c"))',
        ]


class TestAllowlist:
    def test_the_caches_follow_home_and_their_variables(self) -> None:
        assert cache_dirs({"HOME": HOME}) == [
            f"{HOME}/.cache/uv",
            f"{HOME}/Library/Caches/black",
            f"{HOME}/.npm",
        ]
        assert cache_dirs(
            {
                "HOME": HOME,
                "XDG_CACHE_HOME": "/x",
                "BLACK_CACHE_DIR": "/b",
                "npm_config_cache": "/n",
            }
        ) == ["/x/uv", "/b", "/n"]
        assert cache_dirs({"UV_CACHE_DIR": "/u", "NPM_CONFIG_CACHE": "rel"}) == ["/u"]

    def test_the_allowlist_is_exact(self, tmp_path: Path) -> None:
        root = (tmp_path / "ws").resolve()
        root.mkdir()
        env = {"HOME": str(tmp_path), "UV_CACHE_DIR": str(root / "uv")}
        assert allowed_paths(str(root), env) == [
            str(root),  # the uv cache inside it is not listed again
            str(tmp_path.resolve() / "Library" / "Caches" / "black"),
            str(tmp_path.resolve() / ".npm"),
        ]

    def test_a_linked_worktree_adds_its_git_dir_and_the_objects(
        self, tmp_path: Path
    ) -> None:
        main, tree = _worktree(tmp_path)
        assert git_dirs(str(tree)) == [
            str(main / ".git" / "worktrees" / "tree"),
            str(main / ".git" / "objects"),
        ]
        assert git_dirs(str(main)) == []  # a .git folder is inside the root

    def test_a_git_file_that_is_not_named_back_adds_nothing(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / ".git").write_text(f"gitdir: {tmp_path}\n")
        (tmp_path / "commondir").write_text("..\n")
        (tmp_path / "gitdir").write_text("/elsewhere/.git\n")
        assert git_dirs(str(root)) == []


class TestNote:
    def test_a_refused_write_gets_one_line_with_its_path(self) -> None:
        c = _confinement()
        text = "touch: /Users/x/out.txt: Operation not permitted"
        assert c.note(text) == (
            "melo: writing outside the workspace is not allowed in this run "
            "(refused: /Users/x/out.txt); write in /ws, and put scratch files "
            "in $TMPDIR.\n"
        )
        assert c.refused == 1

    @pytest.mark.parametrize(
        ("text", "path"),
        [
            ("PermissionError: [Errno 1] Operation not permitted: '/a b/c'", "/a b/c"),
            ("fatal: Unable to create '/r/.git/index.lock': Operation not permitted", "/r/.git/index.lock"),
            ("/bin/sh: line 1: /etc/x: Operation not permitted", "/etc/x"),
            ("npm error EPERM: operation not permitted, open '/n/_logs/x'", "/n/_logs/x"),
            ("failed to open file `/u/.git`: Operation not permitted (os error 1)", "/u/.git"),
            ("touch: out.txt: Operation not permitted", None),
        ],
    )  # fmt: skip
    def test_the_path_is_found_in_each_tool_s_error(
        self, text: str, path: str | None
    ) -> None:
        note = _confinement().note(text)
        assert note.startswith("melo: writing outside the workspace")
        assert (f"(refused: {path})" in note) if path else ("refused" not in note)

    def test_writes_are_counted_and_paths_capped(self) -> None:
        c = _confinement()
        text = "\n".join(f"rm: /o/{n}: Operation not permitted" for n in range(5))
        assert "(refused: /o/0, /o/1, /o/2 and 2 more)" in c.note(text)
        assert c.refused == 5

    def test_other_output_and_allowed_paths_add_nothing(self) -> None:
        c = _confinement("/ws", "/cache")
        assert c.note("all fine\nPermission denied") == ""
        assert c.note("chmod: /ws/f: Operation not permitted") == ""
        assert c.note("x: /cache/y: Operation not permitted") == ""
        assert c.refused == 0

    def test_the_environment_points_at_the_run_s_folder(self) -> None:
        assert _confinement().env() == dict.fromkeys(
            ("TMPDIR", "TMP", "TEMP"), f"/ws/{TMP_DIR}"
        )


class TestConfine:
    def test_off_macos_nothing_is_confined(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="linux"))
        assert confine(str(tmp_path), {}) is None
        assert not (tmp_path / TMP_DIR).exists()

    @pytest.mark.parametrize(
        ("program", "why"),
        [
            ("/nonexistent/sandbox-exec", "/nonexistent/sandbox-exec is missing"),
            ("/usr/bin/false", "sandbox-exec refused the profile: exit 1"),
        ],
    )
    def test_it_fails_closed_with_why_and_how_to_turn_it_off(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        program: str,
        why: str,
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="darwin"))
        monkeypatch.setattr(sandbox, "SANDBOX_EXEC", program)
        with pytest.raises(ConfinementError) as exc:
            confine(str(tmp_path), {})
        assert str(exc.value) == (
            f"writes cannot be confined to the workspace: {why}; set "
            "agent.confine_writes to false to run without it "
            "(-X agent.confine_writes=false for one run)"
        )
        assert "\n" not in str(exc.value)

    def test_a_folder_that_cannot_be_made_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="darwin"))
        (tmp_path / TMP_DIR).write_text("a file in the way")
        with pytest.raises(ConfinementError, match=f"cannot make .*{TMP_DIR}"):
            confine(str(tmp_path), {})


# ── Real children under sandbox-exec ──


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    return root


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = (tmp_path / "home").resolve()
    path.mkdir()
    return path


def _real(root: Path, env: dict[str, str]) -> WriteConfinement:
    if sys.platform != "darwin":
        pytest.skip("writes are confined only on macOS")
    try:
        found = confine(str(root), child_env(env))
    except ConfinementError as exc:
        pytest.skip(str(exc))
    assert found is not None
    return found


def _tools(
    root: Path, env: dict[str, str], **kwargs: float
) -> tuple[WriteConfinement, ToolFunction, ToolFunction]:
    confinement = _real(root, env)
    tools = rooted_builtins(str(root), tool_env=env, confinement=confinement, **kwargs)
    return confinement, tools["shell_exec"][0], tools["python_exec"][0]


class TestReal:
    async def test_a_write_in_the_workspace_and_its_temp_folder_works(
        self, ws: Path, home: Path
    ) -> None:
        c, shell, python = _tools(ws, {"HOME": str(home)})
        made = await shell('echo hi > in.txt && mktemp "$TMPDIR/x.XXXXXX"')
        assert made.startswith(f"{ws}/{TMP_DIR}/") and Path(made).is_file()
        assert (ws / "in.txt").read_text() == "hi\n"
        code = "import tempfile; print(tempfile.gettempdir())"
        assert await python(code) == f"{ws}/{TMP_DIR}"
        assert (ws / TMP_DIR / ".gitignore").read_text() == "*\n"
        assert c.refused == 0

    async def test_a_write_in_the_home_folder_is_refused(
        self, ws: Path, home: Path
    ) -> None:
        c, shell, python = _tools(ws, {"HOME": str(home)})
        out = await shell("touch ~/outside.txt")
        assert out.splitlines()[:2] == [
            "[exit code 1]",
            "melo: writing outside the workspace is not allowed in this run "
            f"(refused: {home}/outside.txt); write in {ws}, and put scratch "
            "files in $TMPDIR.",
        ]
        assert out.endswith(f"{home}/outside.txt: Operation not permitted")
        with pytest.raises(RuntimeError) as exc:
            await python("import os; open(os.path.expanduser('~/p.txt'), 'w')")
        assert str(exc.value).splitlines()[1].startswith("melo: writing outside")
        assert "PermissionError: [Errno 1] Operation not permitted" in str(exc.value)
        assert not (home / "outside.txt").exists() and not (home / "p.txt").exists()
        assert c.refused == 2

    async def test_a_quoted_path_cannot_add_a_rule(self, tmp_path: Path) -> None:
        root = tmp_path / 'w") (allow file-write* (subpath "/'
        root.mkdir()
        c, shell, _ = _tools(root, {"HOME": str(tmp_path)})
        assert await shell("touch in && echo ok") == "ok"
        assert "Operation not permitted" in await shell(f"touch {tmp_path}/out")
        assert not (tmp_path / "out").exists()

    async def test_the_gated_launch_of_a_saved_task_is_confined_too(
        self, ws: Path, home: Path
    ) -> None:
        recorded: list[int] = []

        async def record(pid: int) -> None:
            recorded.append(pid)

        c, shell, python = _tools(ws, {"HOME": str(home)})
        token = PROCESS_RECORD.set(record)
        try:
            out = await shell("echo $$; touch in ~/out")
            assert int(out.splitlines()[2]) == recorded[0]  # the leader execs
            assert out.splitlines()[1].startswith("melo: writing outside")
            assert await python("open('py.txt', 'w').write('x')") == ""
        finally:
            PROCESS_RECORD.reset(token)
        assert len(recorded) == 2 and c.refused == 1
        assert (ws / "in").exists() and (ws / "py.txt").exists()
        assert not (home / "out").exists()

    async def test_cd_out_and_write_is_refused(self, ws: Path, home: Path) -> None:
        c, shell, _ = _tools(ws, {"HOME": str(home)})
        out = await shell("cd .. && echo x > escaped.txt; git init -q ../repo")
        assert "Operation not permitted" in out and c.refused == 2
        assert not (ws.parent / "escaped.txt").exists()
        assert not (ws.parent / "repo").exists()

    async def test_uv_run_and_pytest_work_in_a_small_project(
        self, ws: Path, home: Path, tmp_path: Path
    ) -> None:
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv is not installed")
        (ws / "pyproject.toml").write_text(
            '[project]\nname = "p"\nversion = "0"\nrequires-python = ">=3.11"\n'
            "dependencies = []\n"
        )
        (ws / "test_p.py").write_text(
            "def test_tmp(tmp_path):\n    (tmp_path / 'x').write_text('x')\n"
        )
        env = {
            "HOME": str(home),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "UV_PYTHON": sys.executable,
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_OFFLINE": "1",
        }
        c, shell, _ = _tools(ws, env, tool_timeout_s=120)
        out = await shell(f"{uv} run --quiet python -c 'print(6 * 7)'")
        assert out.splitlines()[-1] == "42" and (ws / ".venv").is_dir()
        out = await shell(f"{sys.executable} -m pytest -q test_p.py")
        assert "1 passed" in out and (ws / ".pytest_cache").is_dir()
        assert c.refused == 0

    async def test_git_status_and_add_work_in_a_worktree(
        self, tmp_path: Path, home: Path
    ) -> None:
        main, tree = _worktree(tmp_path)
        c, shell, _ = _tools(tree, {"HOME": str(home)})
        (tree / "b.txt").write_text("b\n")
        assert await shell("git add b.txt && git status --short") == "A  b.txt"
        out = await shell("git -c user.name=t -c user.email=t@t commit -qm b")
        assert out.startswith("[exit code 128]\nmelo: writing outside")
        out = await shell(f"touch {main}/x; git -C {main} checkout -q -b evil")
        assert "Operation not permitted" in out and c.refused == 3
        assert not (main / "x").exists()
        status = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=main,
            capture_output=True,
            text=True,
            check=True,
        )
        assert status.stdout == "## main\n"

    async def test_a_timeout_still_kills_the_process_group(
        self, ws: Path, home: Path
    ) -> None:
        _, shell, _ = _tools(ws, {"HOME": str(home)}, tool_timeout_s=1)
        pid_file = ws / "pids"
        try:
            with pytest.raises(TimeoutError, match="timed out after 1 s"):
                await shell(f'sleep 60 & echo $$ $! > "{pid_file}"; wait')
            for pid in map(int, pid_file.read_text().split()):
                for _ in range(250):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.02)
                else:
                    raise AssertionError(f"{pid} survived the timeout")
        finally:
            if pid_file.exists():
                for left in pid_file.read_text().split():
                    with contextlib.suppress(OSError):
                        os.kill(int(left), signal.SIGKILL)

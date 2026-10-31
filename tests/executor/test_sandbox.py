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
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localmelo.melo.contracts import own_gateway
from localmelo.melo.contracts.own_gateway import OwnGateway
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
    user_temp_dir,
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


def _no_getconf(*args: object, **kwargs: object) -> object:
    raise OSError("no getconf")


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

    def test_the_allowlist_is_exact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = (tmp_path / "ws").resolve()
        root.mkdir()
        temp = "/private/var/folders/xy/abc/T"
        monkeypatch.setattr(sandbox, "user_temp_dir", lambda: temp)
        env = {"HOME": str(tmp_path), "UV_CACHE_DIR": str(root / "uv")}
        assert allowed_paths(str(root), env) == [
            str(root),  # the uv cache inside it is not listed again
            temp,
            str(tmp_path.resolve() / "Library" / "Caches" / "black"),
            str(tmp_path.resolve() / ".npm"),
        ]
        monkeypatch.setattr(sandbox, "user_temp_dir", lambda: None)
        assert temp not in allowed_paths(str(root), env)

    def test_the_user_temp_folder_is_the_darwin_one_or_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        found = user_temp_dir()
        if sys.platform == "darwin":
            assert found is not None and found.startswith("/private/var/folders/")
            assert found == os.path.realpath(os.confstr(65537) or "")
        for given in ("/tmp/", "/private/tmp", HOME, "rel/T", ""):
            monkeypatch.setattr(sandbox.os, "confstr", lambda _name, v=given: v)
            monkeypatch.setattr(sandbox.subprocess, "run", _no_getconf)
            assert user_temp_dir() is None, given

    def test_without_confstr_getconf_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unknown(_name: int) -> str:
            raise ValueError("unrecognized configuration name")

        def getconf(argv: list[str], **kwargs: object) -> object:
            assert argv == ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"]
            return SimpleNamespace(stdout="/var/folders/xy/abc/T/\n")

        monkeypatch.setattr(sandbox.os, "confstr", unknown)
        monkeypatch.setattr(sandbox.subprocess, "run", getconf)
        monkeypatch.setattr(sandbox.os.path, "realpath", lambda p: "/private" + p)
        assert user_temp_dir() == "/private/var/folders/xy/abc/T/"
        monkeypatch.setattr(sandbox.subprocess, "run", _no_getconf)
        assert user_temp_dir() is None

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


@pytest.fixture
def no_user_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the user's temp folder out: pytest's tmp_path is inside it."""
    monkeypatch.setattr(sandbox, "user_temp_dir", lambda: None)


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


class TestRealUserTemp:
    async def test_mktemp_works_in_the_user_temp_folder(
        self, ws: Path, home: Path
    ) -> None:
        c, shell, _ = _tools(ws, {"HOME": str(home)})
        temp = user_temp_dir()
        assert temp is not None and temp in c.paths
        out = await shell('d=$(mktemp -d) && f=$(mktemp) && echo "$d" "$f"')
        made, made_file = (Path(part) for part in out.split())
        try:
            assert made.is_dir() and made_file.is_file() and c.refused == 0
            assert os.path.realpath(made).startswith(temp + "/")
            assert os.path.realpath(made_file).startswith(temp + "/")
        finally:
            made_file.unlink(missing_ok=True)
            made.rmdir()
        outside = Path(f"/private/tmp/melo-confine-test-{os.getpid()}-{id(c)}")
        try:
            assert "Operation not permitted" in await shell(f"touch {outside}")
            assert c.refused == 1 and not outside.exists()
        finally:
            outside.unlink(missing_ok=True)  # only if the profile let it through


@pytest.mark.usefixtures("no_user_temp")
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


# ── Auto mode closes melo's own gateway (#339) ──


def test_the_auto_profile_closes_the_gateway_s_files_and_port() -> None:
    guard = OwnGateway(port=8401, files=("/h/.cache/localmelo/gateway/auth",))
    text = profile(["/ws"], guard)
    # The denials come after the folders allowed, so they win.
    assert text.index("(allow file-write*") < text.index("(deny file-read* file-write*")
    assert '(subpath "/h/.cache/localmelo/gateway/auth")' in text
    assert text.endswith('(deny network-outbound (remote ip "localhost:8401"))\n')
    assert "network" not in profile(["/ws"], OwnGateway(port=None, files=()))
    assert "file-read" not in profile(["/ws"])


def test_the_default_guard_covers_the_sign_in_and_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    found = own_gateway.current()
    base = os.path.realpath(tmp_path / ".cache/localmelo/gateway")
    assert {f"{base}/auth", f"{base}/sessions.db", f"{base}/sessions.db-wal"} <= set(
        found.files
    )
    own_gateway.register(8471, [str(tmp_path / "other.db")])
    try:
        found = own_gateway.current()
        assert found.port == 8471
        assert os.path.realpath(tmp_path / "other.db-shm") in found.files
    finally:
        own_gateway.unregister(8471)
    assert own_gateway.current().port is None
    # No gateway here (the terminal): the settings' port is closed instead.
    own_gateway.expect(8472)
    try:
        assert 'localhost:8472"' in profile(["/ws"], own_gateway.current())
    finally:
        own_gateway.expect(None)


def _listening() -> tuple[socket.socket, int]:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    port = server.getsockname()[1]
    if 8400 <= port <= 8799 or port in (8955, 11434, 11500):
        server.close()
        pytest.skip("the system picked a port the benchmark keeps")
    return server, port


@pytest.mark.usefixtures("no_user_temp")
class TestRealGuard:
    async def test_an_auto_call_cannot_find_the_token_nor_reach_the_gateway(
        self, ws: Path, home: Path
    ) -> None:
        auth = home / ".cache/localmelo/gateway/auth"
        auth.mkdir(parents=True)
        (auth / "token").write_text("secret")
        (home / ".cache/other.txt").write_text("fine")
        gateway, gateway_port = _listening()
        other, other_port = _listening()
        if sys.platform != "darwin":
            pytest.skip("writes are confined only on macOS")
        try:
            guard = OwnGateway(port=gateway_port, files=(str(auth.resolve()),))
            try:
                c = confine(str(ws), child_env({"HOME": str(home)}), guard=guard)
            except ConfinementError as exc:
                pytest.skip(str(exc))
            assert c is not None
            tools = rooted_builtins(
                str(ws), tool_env={"HOME": str(home)}, confinement=c
            )
            shell, python = tools["shell_exec"][0], tools["python_exec"][0]
            found = await shell(
                "find ~/.cache -name token -exec cat {} \\; ; cat ~/.cache/other.txt"
            )
            assert "secret" not in found and "fine" in found
            assert "may not read or change melo's own sign-in" in found
            code = (
                "import glob, os, socket\n"
                "print([n for r, d, f in os.walk(os.path.expanduser('~/.cache')) for n in f])\n"
                "print(glob.glob(os.path.expanduser('~/.cache/*/gateway/auth/*')))\n"
                f"socket.create_connection(('127.0.0.1', {other_port}), 2).close()\n"
                "print('other port ok')\n"
                f"socket.create_connection(('127.0.0.1', {gateway_port}), 2)\n"
            )
            with pytest.raises(RuntimeError) as failed:
                await python(code)
            said = str(failed.value)
            assert "['other.txt']" in said and "\n[]\n" in said
            assert "other port ok" in said and "PermissionError" in said
            sent = await shell(
                f"curl -s -m 2 -X POST http://127.0.0.1:{gateway_port}/api/x; echo rc=$?"
            )
            assert "rc=7" in sent  # curl: could not connect
        finally:
            gateway.close()
            other.close()

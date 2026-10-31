"""Task client parser, safety and gateway contract."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from localmelo.support.task_cli import ClientError, address, execute, parser, read_token


def test_address_accepts_only_loopback() -> None:
    assert address("http://localhost:8901") == "http://127.0.0.1:8901"
    assert address("http://[::1]:8901") == "http://[::1]:8901"
    for url in (
        "http://example.com:8901",
        "http://127.0.0.1:8901/x",
        "http://user@localhost:8901",
        "http://localhost:8901?q=x",
    ):
        with pytest.raises(ValueError):
            address(url)


def test_read_token_requires_existing_private_regular_file(tmp_path: Path) -> None:
    auth = tmp_path / "auth"
    auth.mkdir(mode=0o700)
    path = auth / "token"
    with pytest.raises((OSError, ValueError)):
        read_token(auth)
    assert not path.exists()
    path.write_text("a" * 64, encoding="ascii")
    path.chmod(0o600)
    assert read_token(auth) == "a" * 64
    path.chmod(0o644)
    with pytest.raises((OSError, ValueError)):
        read_token(auth)
    path.unlink()
    target = tmp_path / "other"
    target.write_text("a" * 64)
    path.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        read_token(auth)


def test_token_path_matches_gateway_symlink_boundary(tmp_path: Path) -> None:
    home = tmp_path / "home"
    auth = home / "gateway" / "auth"
    auth.mkdir(parents=True, mode=0o700)
    token = auth / "token"
    token.write_text("a" * 64, encoding="ascii")
    token.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    assert read_token(alias / "gateway" / "auth") == "a" * 64
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "gateway").symlink_to(home / "gateway", target_is_directory=True)
    with pytest.raises(OSError):
        read_token(unsafe / "gateway" / "auth")


def test_new_brief_sends_content_and_request_key(tmp_path: Path) -> None:
    brief = tmp_path / "brief"
    brief.write_text("ordinary work\n", encoding="utf-8")
    requests = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(202, json={"task": {"task_id": "abc", "state": "queued"}})

    args = parser().parse_args(["--json", "new", "--brief", str(brief)])
    out = io.StringIO()
    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(reply)
    ) as client:
        assert execute(args, client, out=out, stdin=io.StringIO()) == 0
    assert requests[0]["objective"] == "ordinary work\n"
    assert requests[0]["request_key"]
    assert "level" not in requests[0]
    assert json.loads(out.getvalue())["task"]["task_id"] == "abc"


def test_control_and_list_use_expected_api() -> None:
    calls = []

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"tasks": []})
        return httpx.Response(200, json={"task": {"task_id": "abc", "state": "queued"}})

    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(reply)
    ) as client:
        assert (
            execute(
                parser().parse_args(["list", "--level", "P2"]),
                client,
                out=io.StringIO(),
                stdin=io.StringIO(),
            )
            == 0
        )
        assert (
            execute(
                parser().parse_args(["pause", "abc"]),
                client,
                out=io.StringIO(),
                stdin=io.StringIO(),
            )
            == 0
        )
        assert (
            execute(
                parser().parse_args(["level", "abc", "auto"]),
                client,
                out=io.StringIO(),
                stdin=io.StringIO(),
            )
            == 0
        )
    assert calls[0].url.params["level"] == "2"
    assert json.loads(calls[1].content) == {"state": "pause"}
    assert json.loads(calls[2].content) == {"level": "auto"}


@pytest.mark.parametrize(
    "args,method,body",
    [
        (["show", "abc"], "GET", None),
        (
            ["resume", "abc", "--level", "P2"],
            "PATCH",
            {"state": "resume", "level": "P2"},
        ),
        (["stop", "abc"], "PATCH", {"state": "stop"}),
    ],
)
def test_show_resume_stop_requests(args: list[str], method: str, body: object) -> None:
    calls: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"task": {"task_id": "abc", "state": "queued"}})

    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(reply)
    ) as client:
        assert (
            execute(
                parser().parse_args(args),
                client,
                out=io.StringIO(),
                stdin=io.StringIO(),
            )
            == 0
        )
    assert calls[0].method == method and calls[0].url.path == "/api/tasks/abc"
    assert (json.loads(calls[0].content) if calls[0].content else None) == body


def test_invalid_mode_and_non_utf8_brief(tmp_path: Path) -> None:
    brief = tmp_path / "brief"
    brief.write_bytes(b"\xff")
    args = parser().parse_args(["new", "--brief", str(brief)])
    with (
        httpx.Client(
            base_url="http://127.0.0.1:8901",
            transport=httpx.MockTransport(lambda request: pytest.fail("must not call")),
        ) as client,
        pytest.raises(ValueError),
    ):
        execute(args, client, out=io.StringIO(), stdin=io.StringIO())
    with pytest.raises(SystemExit):
        parser().parse_args(["level", "abc", "first"])


def test_ambiguous_create_reports_reusable_key() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fixture", request=request)

    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(fail)
    ) as client:
        args = parser().parse_args(["new", "task", "--request-key", "retry-key"])
        with pytest.raises(ClientError, match="retry-key"):
            execute(args, client, out=io.StringIO(), stdin=io.StringIO())


def test_http_errors_do_not_expose_server_body() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409, json={"error": {"code": "not_paused", "message": "private fixture"}}
        )

    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(fail)
    ) as client:
        args = parser().parse_args(["resume", "abc"])
        with pytest.raises(ClientError) as exc:
            execute(args, client, out=io.StringIO(), stdin=io.StringIO())
    assert "not_paused" in str(exc.value)
    assert "private fixture" not in str(exc.value)


def test_human_output_strips_terminal_control_sequences() -> None:
    with httpx.Client(
        base_url="http://127.0.0.1:8901",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "tasks": [
                        {
                            "task_id": "abc",
                            "state": "queued",
                            "spec": {"title": chr(27) + "[2JTask"},
                        }
                    ]
                },
            )
        ),
    ) as client:
        output = io.StringIO()
        assert (
            execute(
                parser().parse_args(["list"]), client, out=output, stdin=io.StringIO()
            )
            == 0
        )
    # Columns are padded to the widest cell, so the header lines up with the row.
    assert output.getvalue() == (
        "ID   LEVEL         STATE   TITLE\n" "abc  not recorded  queued  ?[2JTask\n"
    )
    assert chr(27) not in output.getvalue()


def test_list_human_header_is_printed_for_an_empty_list() -> None:
    with httpx.Client(
        base_url="http://127.0.0.1:8901",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"tasks": []})
        ),
    ) as client:
        output = io.StringIO()
        assert (
            execute(
                parser().parse_args(["list"]), client, out=output, stdin=io.StringIO()
            )
            == 0
        )
    assert output.getvalue() == "ID  LEVEL  STATE  TITLE\n"


@pytest.mark.parametrize(
    "host,expected",
    [
        ("::1", "::1"),
        ("[::1]", "::1"),
        ("127.0.0.2", "127.0.0.2"),
        ("127.1", "127.0.0.1"),
        ("2130706433", "127.0.0.1"),
        ("localhost.", "127.0.0.1"),
        ("0:0:0:0:0:0:0:1", "::1"),
    ],
)
def test_main_configured_loopback_uses_guard_compatible_host(
    host: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import localmelo.support.config as config
    import localmelo.support.task_cli as cli

    monkeypatch.setattr(
        config,
        "load",
        lambda: SimpleNamespace(gateway=SimpleNamespace(host=host, port=8901)),
    )
    monkeypatch.setattr(cli, "read_token", lambda path: "a" * 64)
    factory = httpx.Client
    requests: list[httpx.Request] = []

    def client(**kwargs: Any) -> httpx.Client:
        def reply(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"tasks": []})

        return factory(**kwargs, transport=httpx.MockTransport(reply))

    monkeypatch.setattr(cli.httpx, "Client", client)
    assert cli.main(["list"]) == 0
    assert requests[0].url.host == expected
    host_header = "[::1]:8901" if expected == "::1" else "127.0.0.1:8901"
    assert requests[0].headers["Host"] == host_header


def test_explicit_gateway_ignores_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import localmelo.support.task_cli as cli

    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / ".cache/localmelo/config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("not valid = [", encoding="utf-8")
    monkeypatch.setattr(cli, "read_token", lambda path: "a" * 64)
    factory = httpx.Client
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kw: factory(
            **kw,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"tasks": []})
            ),
        ),
    )
    assert cli.main(["--gateway", "http://127.0.0.1:8901", "list"]) == 0
    assert cli.main(["--gateway", "", "list"]) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--gateway", "http://127.0.0.1:8901", "list"],
        ["--gateway", "", "list"],
        ["follow", "abc", "--after", "-1"],
    ],
    ids=["list", "bad-gateway", "bad-cursor"],
)
def test_melo_task_is_this_client(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``melo task ...`` makes the same requests and exits the same way, under its own name."""
    import localmelo.support.task_cli as cli
    from localmelo.support.command_words import expand

    monkeypatch.setattr(cli, "read_token", lambda path: "a" * 64)
    requests: list[httpx.Request] = []
    factory = httpx.Client

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"tasks": []})

    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kw: factory(**kw, transport=httpx.MockTransport(reply)),
    )
    direct = cli.main(argv)
    plain = (len(requests), capsys.readouterr().err)
    requests.clear()
    with pytest.raises(SystemExit) as exc:
        expand(["task", *argv])
    assert exc.value.code == direct
    assert (len(requests), capsys.readouterr().err) == (
        plain[0],
        plain[1].replace("melo-task:", "melo task:"),
    )


@pytest.mark.parametrize("cursor", ["-1", "9223372036854775808"])
def test_bad_cursor_is_usage_error(
    cursor: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import localmelo.support.task_cli as cli

    assert cli.main(["follow", "abc", "--after", cursor]) == 2
    assert "cursor" in capsys.readouterr().err


@pytest.mark.parametrize("status", [401, 404, 409])
def test_main_follow_error_is_stderr_only(
    status: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import localmelo.support.task_cli as cli

    monkeypatch.setattr(cli, "read_token", lambda path: "a" * 64)
    factory = httpx.Client
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kw: factory(
            **kw, transport=httpx.MockTransport(lambda request: httpx.Response(status))
        ),
    )
    assert (
        cli.main(["--gateway", "http://127.0.0.1:8901", "--json", "follow", "abc"]) == 1
    )
    captured = capsys.readouterr()
    assert captured.out == "" and str(status) in captured.err


def test_gateway_down_names_address_and_interrupt_is_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import localmelo.support.task_cli as cli

    monkeypatch.setattr(cli, "read_token", lambda path: "a" * 64)
    factory = httpx.Client

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda **kw: factory(**kw, transport=httpx.MockTransport(fail)),
    )
    assert cli.main(["--gateway", "http://127.0.0.1:8901", "list"]) == 1
    assert "http://127.0.0.1:8901" in capsys.readouterr().err
    monkeypatch.setattr(
        cli, "execute", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    assert cli.main(["--gateway", "http://127.0.0.1:8901", "follow", "abc"]) == 130
    assert capsys.readouterr().err == ""


def test_main_sends_bearer_without_proxy_or_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import localmelo.support.task_cli as cli

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "read_token", lambda path: "a" * 64)
    factory = httpx.Client
    options = {}
    requests = []

    def client(**kwargs: Any) -> httpx.Client:
        options.update(kwargs)

        def reply(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"tasks": []})

        return factory(**kwargs, transport=httpx.MockTransport(reply))

    monkeypatch.setattr(cli.httpx, "Client", client)
    assert cli.main(["--gateway", "http://127.0.0.1:8901", "--json", "list"]) == 0
    assert options["follow_redirects"] is False and options["trust_env"] is False
    assert requests[0].headers["Authorization"] == "Bearer " + "a" * 64
    assert "a" * 64 not in capsys.readouterr().out


def _run(args: list[str], payload: dict[str, Any]) -> str:
    with httpx.Client(
        base_url="http://127.0.0.1:8901",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    ) as client:
        output = io.StringIO()
        code = execute(
            parser().parse_args(args), client, out=output, stdin=io.StringIO()
        )
    assert code == 0
    return output.getvalue()


_HELD = {
    "task_id": "abc",
    "state": "held",
    "title": "Write it",
    "level": 2,
    "wait_reason": "approval",
    "pending_approvals": [
        {
            "tool_name": "file_write",
            "arguments": {"path": "/w/out.txt"},
            "target": {"path_label": "out.txt", "exists": False, "size_bytes": 0},
        }
    ],
}
_LINE = (
    "Waiting for your approval: file_write out.txt "
    "(answer it in the web console, Tasks)"
)


def test_show_names_the_approval_a_held_task_waits_for() -> None:
    text = _run(["show", "abc"], {"task": _HELD})
    assert f"  {_LINE}\n" in text
    assert "Waiting: " not in text
    assert "  Reason: approval\n" in text
    assert "not recorded" not in text


def test_show_gives_a_shell_command_and_keeps_a_recorded_reason() -> None:
    task = {
        **_HELD,
        "reason": "scheduler",
        "pending_approvals": [
            {"tool_name": "shell_exec", "arguments": {"command": "make  test\n"}}
        ],
    }
    text = _run(["show", "abc"], {"task": task})
    assert "Waiting for your approval: shell_exec make test (answer it" in text
    assert "  Reason: scheduler\n" in text


def test_show_without_a_pending_approval_gives_the_wait_and_no_reason() -> None:
    task = {**_HELD, "pending_approvals": []}
    text = _run(["show", "abc"], {"task": task})
    assert "  Waiting: approval\n" in text
    assert "Reason:" not in text and "not recorded" not in text


def test_show_prints_only_the_reason_and_wait_a_task_has() -> None:
    finished = {"task_id": "abc", "state": "done", "title": "Write it", "level": 2}
    text = _run(["show", "abc"], {"task": finished})
    assert text.splitlines()[1:] == ["  Title: Write it", "  Level: P2"]
    text = _run(["show", "abc"], {"task": {**finished, "reason": "budget"}})
    assert "  Reason: budget\n" in text and "Waiting:" not in text


def test_list_adds_the_approval_line_under_a_waiting_task() -> None:
    quiet = {"task_id": "def", "state": "queued", "title": "Other"}
    text = _run(["list"], {"tasks": [_HELD, quiet]})
    assert text.splitlines()[1:] == [
        "abc  P2            held    Write it",
        f"  {_LINE}",
        "def  not recorded  queued  Other",
    ]


@pytest.mark.parametrize(
    "state,says_requested", [("held", True), ("running", True), ("cancelled", False)]
)
def test_stop_says_when_the_task_has_not_stopped_yet(
    state: str, says_requested: bool
) -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"task": {"task_id": "abc", "state": state}})

    out = io.StringIO()
    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(reply)
    ) as client:
        code = execute(
            parser().parse_args(["stop", "abc"]), client, out=out, stdin=io.StringIO()
        )
    assert code == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == f"abc  {state}"
    if says_requested:
        assert lines[1].startswith(
            "  stop requested; the task stops when its worker ends"
        )
        assert lines[1].endswith("melo-task show abc")
    else:
        assert len(lines) == 1


def test_default_mode_says_levels_in_words_and_takes_them() -> None:
    finished = {"task_id": "abc", "state": "done", "title": "Write it", "level": 1}
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = (
            {"tasks": [finished]}
            if request.method == "GET" and request.url.path == "/api/tasks"
            else {"task": finished}
        )
        return httpx.Response(200, json=body)

    with httpx.Client(
        base_url="http://127.0.0.1:8901", transport=httpx.MockTransport(answer)
    ) as client:
        out = io.StringIO()
        for argv in (
            ["show", "abc"],
            ["list"],
            ["level", "abc", "If idle"],
            ["new", "x", "--level", "today"],
        ):
            execute(
                parser().parse_args(argv),
                client,
                out=out,
                stdin=io.StringIO(),
                words=True,
            )
    text = out.getvalue()
    assert "  Level: Today\n" in text
    assert "abc  Today  done   Write it" in text
    assert json.loads(seen[2].content) == {"level": "P4"}
    assert json.loads(seen[3].content)["level"] == "P1"
    with pytest.raises(SystemExit):
        parser().parse_args(["level", "abc", "soon"])


def test_main_reads_the_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from localmelo.support import task_cli

    monkeypatch.setenv("HOME", str(tmp_path))
    assert task_cli._developer() is False

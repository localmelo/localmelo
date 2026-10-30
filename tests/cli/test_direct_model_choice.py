"""A direct run with ``--base-url`` and no ``--chat-model`` picks its model like the UI.

The server is a fake behind ``httpx.MockTransport``: the model list, the status
routes and the chat requests all go to it and are counted, so nothing here
reaches a model server. Its host is ``models.test``, which never resolves.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.support.backends import discovery
from localmelo.support.providers.llm import openai_compat

# The tests in this module run the real choice (see the conftest fixture).
LISTS_MODELS = True

# Taken at import: the CLI conftest replaces the status lookup for every test.
LIST_MODELS = discovery.list_models
LOADED_MODELS = discovery.loaded_models

URL = "http://models.test/v1"
BIG, SMALL = "big-27B", "small-8B"
HINT = "Pass --chat-model <id>."
KEY_NOTE = "The server did not say which models are loaded (it needs a key for that)."
OMLX_ALL = {"models": [{"id": BIG, "loaded": False}, {"id": SMALL, "loaded": False}]}


class FakeServer:
    """An OpenAI-compatible server that counts every request it gets.

    ``status`` answers ``GET /v1/models/status``: None for no such route, an
    int for that HTTP status, a dict for a JSON body. ``listing`` is the HTTP
    status of ``GET /v1/models``.
    """

    def __init__(
        self,
        models: list[str],
        *,
        status: int | dict[str, Any] | None = None,
        listing: int = 200,
    ) -> None:
        self.models = models
        self.status = status
        self.listing = listing
        self.seen: list[tuple[str, str]] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path))
        path = request.url.path
        if request.method == "POST":
            self.bodies.append(json.loads(request.content))
            message = {"role": "assistant", "content": "done"}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        if path == "/v1/models":
            if self.listing != 200:
                return httpx.Response(self.listing)
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if path == "/v1/models/status" and self.status is not None:
            if isinstance(self.status, int):
                return httpx.Response(self.status)
            return httpx.Response(200, json=self.status)
        return httpx.Response(404)

    @property
    def gets(self) -> list[str]:
        return [path for method, path in self.seen if method == "GET"]

    @property
    def models_sent(self) -> set[str]:
        return {str(body["model"]) for body in self.bodies}


class Run(NamedTuple):
    code: int
    out: str
    err: str


class Melo:
    """Runs melo's ``main()`` in this process against a fake server."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.monkeypatch = monkeypatch
        self.capsys = capsys

    def __call__(self, server: FakeServer, *argv: str) -> Run:
        transport = httpx.MockTransport(server)
        patch = self.monkeypatch.setattr
        patch(
            discovery,
            "list_models",
            functools.partial(LIST_MODELS, transport=transport),
        )
        patch(
            discovery,
            "loaded_models",
            functools.partial(LOADED_MODELS, transport=transport),
        )

        class Chat(openai_compat.OpenAICompatLLM):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, transport=transport, **kwargs)

        patch(openai_compat, "OpenAICompatLLM", Chat)
        patch(sys, "argv", ["melo", "--no-save-session", *argv])
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = 0 if exc.code is None else int(exc.code)
        out, err = self.capsys.readouterr()
        return Run(code, out, err)


@pytest.fixture
def melo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Melo:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return Melo(monkeypatch, capsys)


def _brief(tmp_path: Path) -> str:
    path = tmp_path / "brief.md"
    path.write_text("Say done.", encoding="utf-8")
    return str(path)


def _refusal(*names: str) -> str:
    listing = "".join(f"\n  {name}" for name in names)
    return (
        f"{URL} serves several models and it is not known which one is loaded:"
        f"{listing}\n{HINT}"
    )


# ── Choosing ──


def test_the_only_chat_model_is_requested(melo: Melo) -> None:
    server = FakeServer(["bge-embed", "Qwen3-8B"])
    run = melo(server, "--base-url", URL, "--", "task")
    assert (run.code, run.err) == (0, "")
    assert server.models_sent == {"Qwen3-8B"}
    assert server.gets == ["/v1/models"]  # one chat model: no status lookup


def test_the_loaded_model_of_several_is_requested(melo: Melo) -> None:
    status = {"models": [{"id": BIG, "loaded": False}, {"id": SMALL, "loaded": True}]}
    server = FakeServer([BIG, SMALL], status=status)
    run = melo(server, "--base-url", URL, "--", "task")
    assert (run.code, run.err) == (0, "")
    assert server.models_sent == {SMALL}  # never the one that is not in memory


@pytest.mark.parametrize(
    ("status", "says_key"),
    [
        (401, True),  # a server with a key answers its status route like the listing
        (403, True),
        (None, False),  # no status route
        (OMLX_ALL, False),  # none in memory
        ({"models": [{"id": BIG}, {"id": SMALL}]}, False),  # more than one in memory
    ],
    ids=["401", "403", "no-route", "none-loaded", "several-loaded"],
)
def test_several_models_and_none_known_loaded_exit_2_with_the_list(
    melo: Melo, status: int | dict[str, Any] | None, says_key: bool
) -> None:
    server = FakeServer([BIG, SMALL], status=status)
    run = melo(server, "--base-url", URL, "--", "task")
    assert run.code == 2 and run.out == ""
    assert run.err.startswith("melo: error: " + _refusal(BIG, SMALL).split("\n  ")[0])
    assert "\n  " + BIG in run.err and "\n  " + SMALL in run.err
    assert run.err.endswith(HINT + "\n")
    # A key-protected status route is said, in one line right before the hint.
    assert (KEY_NOTE in run.err) is says_key
    assert (f"\n{KEY_NOTE}\n{HINT}\n" in run.err) is says_key
    assert server.bodies == []  # no request that could load a model


def test_the_loop_is_refused_the_same_way(melo: Melo) -> None:
    server = FakeServer([BIG, SMALL])
    run = melo(server, "--base-url", URL, "")
    assert run.code == 2 and run.err.endswith(HINT + "\n")
    assert server.bodies == []


def test_a_usage_error_comes_before_any_request(melo: Melo) -> None:
    server = FakeServer([BIG, SMALL])
    # Only the builder refuses a retriever that has no embedding provider.
    argv = ["--reasoning", "compress", "--retriever", "embedding", "--", "task"]
    run = melo(server, "--base-url", URL, *argv)
    assert run.code == 2 and "embedding needs an embedding provider" in run.err
    assert server.seen == []


# ── A server that cannot be listed ──


@pytest.mark.parametrize("output", [[], ["--output-format", "text"]], ids=["", "text"])
@pytest.mark.parametrize(
    ("server", "reason"),
    [
        (FakeServer([], listing=503), "HTTP 503"),
        (FakeServer([]), "it lists no models"),
    ],
    ids=["503", "empty"],
)
def test_an_unlisted_server_gets_default_and_one_warning_line(
    melo: Melo, server: FakeServer, reason: str, output: list[str]
) -> None:
    run = melo(server, "--base-url", URL, *output, "--", "task")
    assert run.code == 0
    assert run.err == (
        f"melo: warning: Could not list the models at {URL} ({reason}); "
        "asking for the model 'default'.\n"
    )
    assert server.models_sent == {"default"}


def test_json_output_has_no_warning_and_still_asks_for_default(melo: Melo) -> None:
    server = FakeServer([], listing=503)
    run = melo(server, "--base-url", URL, "--output-format", "json", "--", "task")
    assert (run.code, run.err) == (0, "")
    assert json.loads(run.out)["status"] == "completed" and run.out.count("\n") == 1
    assert server.models_sent == {"default"}


# ── --chat-model: nothing changes ──


@pytest.mark.parametrize("mode", ["text", "json", "brief"])
def test_a_named_model_sends_no_listing_request(
    melo: Melo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    def refused(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model choice ran with --chat-model")

    monkeypatch.setattr(cli, "_direct_model", refused)
    server = FakeServer([BIG, SMALL], status=OMLX_ALL)
    task = {
        "text": ["--", "task"],
        "json": ["--output-format", "json", "--", "task"],
        "brief": ["--brief", _brief(tmp_path)],
    }[mode]
    run = melo(server, "--base-url", URL, "--chat-model", "M", *task)
    assert run.code == 0, run.err
    assert server.gets == []  # nothing but chat requests
    assert server.bodies and server.models_sent == {"M"}
    assert {path for _, path in server.seen} == {"/v1/chat/completions"}


# ── The worker contract: --brief ──


def test_a_brief_run_requests_the_chosen_model(melo: Melo, tmp_path: Path) -> None:
    server = FakeServer(["Qwen3-8B"])
    run = melo(server, "--base-url", URL, "--brief", _brief(tmp_path))
    assert run.code == 0, run.err
    assert json.loads(run.out)["status"] == "completed"
    assert server.models_sent == {"Qwen3-8B"}


def test_a_brief_run_with_an_unlisted_server_asks_for_default_quietly(
    melo: Melo, tmp_path: Path
) -> None:
    server = FakeServer([], listing=503)
    run = melo(server, "--base-url", URL, "--brief", _brief(tmp_path))
    assert (run.code, run.err) == (0, "")
    assert server.models_sent == {"default"}


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_a_brief_refusal_is_a_startup_error_record_with_exit_2(
    melo: Melo, tmp_path: Path, fmt: str
) -> None:
    server = FakeServer([BIG, SMALL])
    run = melo(
        server,
        "--base-url",
        URL,
        "--brief",
        _brief(tmp_path),
        "--output-format",
        fmt,
    )
    assert run.code == 2 and run.out.count("\n") == 1
    record = json.loads(run.out)
    if fmt == "stream-json":
        assert record["type"] == "result"
        record = record["result"]
    assert record["status"] == "startup_error" and record["exit_code"] == 2
    assert record["run"] is None and record["run_id"] is None
    assert record["error"] == " ".join(_refusal(BIG, SMALL).split())
    assert f"melo: worker startup failed: {record['error']}" in run.err
    assert server.bodies == []


def test_a_brief_refusal_keeps_the_key_note_before_the_hint(
    melo: Melo, tmp_path: Path
) -> None:
    server = FakeServer([BIG, SMALL], status=401)
    run = melo(server, "--base-url", URL, "--brief", _brief(tmp_path))
    error = json.loads(run.out)["error"]
    assert run.code == 2 and error.endswith(f"{KEY_NOTE} {HINT}")
    assert BIG in error and SMALL in error


def test_a_long_list_does_not_cut_the_hint_out_of_the_record(
    melo: Melo, tmp_path: Path
) -> None:
    names = [f"organisation/model-family-{n:02d}-instruct-q4" for n in range(30)]
    run = melo(FakeServer(names), "--base-url", URL, "--brief", _brief(tmp_path))
    error = json.loads(run.out)["error"]
    assert run.code == 2 and len(error) > 500
    assert error.endswith(HINT) and names[0] in error and names[-1] in error


def test_a_refusal_record_on_a_broken_pipe_exits_1(
    melo: Melo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def write(self, text: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise BrokenPipeError

        def fileno(self) -> int:
            raise OSError("not a file")

    monkeypatch.setattr(sys, "stdout", Broken())
    server = FakeServer([BIG, SMALL])
    run = melo(server, "--base-url", URL, "--brief", _brief(tmp_path))
    assert run.code == 1  # as for any startup record that cannot be written
    assert "worker output failed: broken pipe" in run.err


def test_plain_json_output_stays_resultless_on_a_refusal(melo: Melo) -> None:
    run = melo(
        FakeServer([BIG, SMALL]), "--base-url", URL, "--output-format", "json", "t"
    )
    assert run.code == 2 and run.out == ""
    assert run.err.endswith(HINT + "\n")

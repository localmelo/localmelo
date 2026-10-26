"""melo-retrieve: refusals, input errors, evaluation lines, providers and export.

Every test runs ``main([...])`` in process, so each is a plain function:
``main`` runs its own event loop. Model requests go to a local
ThreadingHTTPServer on 127.0.0.1, port 0, that answers from a script and
records every request; nothing reaches a model server. Expected values were
derived by hand from the R3 spec (Requirements 14 and 15).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.retrieval import (
    RawCall,
    doc_to_dict,
    docs_from_calls,
    load_raw_calls,
)
from localmelo.melo.memory.retrieval.llm import RETRIEVE_SYSTEM_PROMPT
from localmelo.melo.schema import Message, ToolCall, ToolDef, ToolResult
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.retrieve import main
from localmelo.tests.memory.history_fixture import write_history_sync

FIXTURES = Path(__file__).parents[1] / "memory" / "fixtures" / "retrieval"
CORPUS = str(FIXTURES / "corpus.jsonl")
QUERIES = str(FIXTURES / "queries.jsonl")
RAW_CALLS = str(FIXTURES / "raw_calls.jsonl")
INTERLEAVE = str(FIXTURES / "interleave_request.json")
FILES = ["--corpus", CORPUS, "--queries", QUERIES]

# Never contacted: the runs that name it end before any request.
UNUSED_URL = "http://127.0.0.1:9/v1"

USAGE = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}

QUERY_KEYS = [
    "type",
    "qid",
    "source",
    "candidates",
    "relevant",
    "hits",
    "recall@1",
    "recall@3",
    "rr",
    "ms_cold",
    "ms_warm",
    "ms_interleaved",
    "fallbacks",
    "usage",
]
SUMMARY_KEYS = [
    "type",
    "retriever",
    "model",
    "dim",
    "n",
    "calls",
    "recall@1",
    "recall@3",
    "mrr",
    "ms_cold_p50",
    "ms_warm_p50",
    "ms_warm_p95",
    "ms_interleaved_p50",
    "ms_interleaved_p95",
    "prompt_tokens_mean",
    "cached_tokens_mean",
    "completion_tokens_mean",
    "fallbacks",
]
NO_FALLBACKS = {"timeout": 0, "parse": 0, "provider_error": 0}

# Each fixture query: its candidates (excluded ones not counted) and its
# relevant ids, worked out from corpus.jsonl and queries.jsonl.
FIXTURE_QUERIES = [
    ("en-append", 8, ["t-en:d2", "t-en:n2"]),
    ("zh-reverse", 6, ["t-zh:d1", "t-zh:n1"]),
    ("en-foldr", 14, ["t-en:t3.1", "t-en:n3"]),
    ("en-append-excluded", 7, ["t-en:n2"]),
    ("en-tests-before-3", 4, ["t-en:n1"]),
    ("zh-empty", 6, ["t-zh:n2", "t-zh:t2.1"]),
]


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests to 127.0.0.1 go straight to it, whatever proxy is set."""
    for name in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


# ── A scripted model server ─────────────────────────────────────────────────


@dataclass
class Request:
    path: str
    headers: dict[str, str]  # names in lower case
    body: Any


def _reply(content: str) -> tuple[int, Any]:
    message = {"role": "assistant", "content": content}
    return 200, {"choices": [{"message": message}], "usage": USAGE}


def _is_retrieval(body: Any) -> bool:
    messages = body.get("messages") or [{}]
    return bool(messages[0].get("content") == RETRIEVE_SYSTEM_PROMPT)


def _embedding(text: str) -> list[float]:
    """The stub's: each lowercase token adds 1 at sha256(token) % 16."""
    vector = [0.0] * 16
    for token in text.lower().split():
        vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 16] += 1.0
    return vector


class ModelServer:
    """``POST …/chat/completions`` and ``…/embeddings`` on 127.0.0.1, port 0.

    :attr:`chat` answers each chat body with ``(status, body)``: an object
    is sent as JSON, a string as text. Embeddings are the stub's 16
    numbers per input, or :attr:`embeddings_status` with a text body when
    it is not 200. :attr:`requests` keeps every request, in order.
    """

    def __init__(self) -> None:
        self.chat: Callable[[Any], tuple[int, Any]] = lambda body: (
            _reply('{"ids": ["n1"]}') if _is_retrieval(body) else _reply("Done.")
        )
        self.embeddings_status = 200
        self.requests: list[Request] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                headers = {name.lower(): value for name, value in self.headers.items()}
                server.requests.append(Request(self.path, headers, body))
                if self.path.endswith("/embeddings"):
                    status, answer = server._embeddings(body)
                elif self.path.endswith("/chat/completions"):
                    status, answer = server.chat(body)
                else:
                    status, answer = 404, "not found"
                data = (
                    answer.encode()
                    if isinstance(answer, str)
                    else json.dumps(answer).encode()
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # no request log in the test output

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def chat_requests(self) -> list[Request]:
        return [r for r in self.requests if r.path.endswith("/chat/completions")]

    def _embeddings(self, body: Any) -> tuple[int, Any]:
        if self.embeddings_status != 200:
            return self.embeddings_status, "embedding backend down"
        inputs = body.get("input", [])
        texts = [inputs] if isinstance(inputs, str) else inputs
        data = [
            {"object": "embedding", "index": i, "embedding": _embedding(text)}
            for i, text in enumerate(texts)
        ]
        return 200, {"object": "list", "model": "stub-embedding", "data": data}

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Iterator[ModelServer]:
    model_server = ModelServer()
    try:
        yield model_server
    finally:
        model_server.close()


def _lines(out: str) -> list[dict[str, Any]]:
    assert out.endswith("\n")
    return [json.loads(line) for line in out.splitlines()]


# ── Usage errors ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--export-run", "run.jsonl", "--retriever", "bm25", "--k", "3"],
            "--export-run cannot be combined with --retriever, --k",
        ),
        (
            [],
            "an evaluation needs --retriever, --corpus and --queries (missing: "
            "--retriever, --corpus, --queries); or give --export-run",
        ),
        (
            ["--retriever", "bm25", "--corpus", CORPUS],
            "an evaluation needs --retriever, --corpus and --queries (missing: "
            "--queries); or give --export-run",
        ),
        (["--retriever", "llm", *FILES], "--retriever llm needs --base-url"),
        (
            ["--retriever", "embedding", *FILES],
            "--retriever embedding needs --embedding-base-url",
        ),
        (
            ["--retriever", "bm25", *FILES, "--retrieval-extra-body", "{}"],
            "--retrieval-extra-body applies only to --retriever llm",
        ),
        (
            [
                "--retriever",
                "embedding",
                *FILES,
                "--embedding-base-url",
                UNUSED_URL,
                "--retrieval-timeout",
                "5",
            ],
            "--retrieval-timeout applies only to --retriever llm",
        ),
        (
            ["--retriever", "bm25", *FILES, "--embedding-base-url", UNUSED_URL],
            "--embedding-base-url applies only to --retriever embedding",
        ),
        (
            [
                "--retriever",
                "llm",
                *FILES,
                "--base-url",
                UNUSED_URL,
                "--embedding-model",
                "m",
            ],
            "--embedding-model applies only to --retriever embedding",
        ),
        (
            ["--retriever", "bm25", *FILES, "--base-url", UNUSED_URL],
            "--base-url applies only to --retriever llm or --interleave",
        ),
        (
            [
                "--retriever",
                "embedding",
                *FILES,
                "--embedding-base-url",
                UNUSED_URL,
                "--chat-model",
                "m",
            ],
            "--chat-model applies only to --retriever llm or --interleave",
        ),
        (
            ["--retriever", "bm25", *FILES, "--interleave", INTERLEAVE],
            "--interleave needs --base-url",
        ),
        (
            ["--retriever", "llm", *FILES, "--base-url", "http://127.0.0.1:80o0/v1"],
            "--base-url is not a valid URL: Invalid port: '80o0'",
        ),
        (
            [
                "--retriever",
                "embedding",
                *FILES,
                "--embedding-base-url",
                "http://127.0.0.1:80o0/v1",
            ],
            "--embedding-base-url is not a valid URL: Invalid port: '80o0'",
        ),
        (
            [
                "--retriever",
                "bm25",
                *FILES,
                "--interleave",
                INTERLEAVE,
                "--base-url",
                "http://[::1",
            ],
            "--base-url is not a valid URL: Invalid port: ':1'",
        ),
        (
            ["--retriever", "bm25", *FILES, "--k", "0"],
            "argument --k: must be at least 1, got 0",
        ),
        (
            ["--retriever", "bm25", *FILES, "--k", "1,x"],
            "argument --k: not an integer: 'x'",
        ),
        (
            ["--retriever", "bm25", *FILES, "--repeat", "-1"],
            "argument --repeat: must be at least 0, got -1",
        ),
        (
            [
                "--retriever",
                "llm",
                *FILES,
                "--base-url",
                UNUSED_URL,
                "--retrieval-timeout",
                "0",
            ],
            "argument --retrieval-timeout: must be a finite number of seconds > 0, "
            "got 0",
        ),
    ],
)
def test_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("usage: melo-retrieve")
    assert err.splitlines()[-1] == f"melo-retrieve: error: {message}"


def test_an_unknown_retriever_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--retriever", "tfidf", *FILES]) == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: melo-retrieve")
    assert err.splitlines()[-1].startswith(
        "melo-retrieve: error: argument --retriever: invalid choice: 'tfidf'"
    )


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: melo-retrieve")
    assert "--export-run DIR|FILE" in out


# ── Input errors ────────────────────────────────────────────────────────────


def _input_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Run *argv*, check exit 2 with one line and no usage; return the line."""
    assert main(argv) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1 and err.endswith("\n")
    assert err.startswith("melo-retrieve: error: ")
    return err.rstrip("\n").removeprefix("melo-retrieve: error: ")


def test_a_missing_or_unreadable_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "none.jsonl")
    argv = ["--retriever", "bm25", "--corpus", missing, "--queries", QUERIES]
    assert _input_error(argv, capsys) == f"{missing}: No such file or directory"
    argv = ["--retriever", "bm25", "--corpus", str(tmp_path), "--queries", QUERIES]
    assert _input_error(argv, capsys) == f"{tmp_path}: Is a directory"


def test_text_that_is_not_utf8_names_its_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(b'{"doc_id": "a", "kind": "note", "text": "t"}\n\xff\xfe\n')
    argv = ["--retriever", "bm25", "--corpus", str(corpus), "--queries", QUERIES]
    assert _input_error(argv, capsys) == f"{corpus}:2: not UTF-8 text"


def test_an_unpaired_surrogate_names_its_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Valid JSON in valid UTF-8, but a string no output line could hold.
    problem = "a string holds the unpaired surrogate \\ud83d, which UTF-8 cannot encode"
    calls = tmp_path / "calls.jsonl"
    calls.write_text(
        '{"task_id": "t", "seq": 0, "thinking": "Let me read list_ops.py."}\n'
        '{"task_id": "t", "seq": 1, "thinking": "The test fails \\ud83d here."}\n',
        encoding="utf-8",
    )
    assert _input_error(["--export-run", str(calls)], capsys) == f"{calls}:2: {problem}"
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"doc_id": "a\\ud83d", "kind": "note", "text": "t"}\n', encoding="utf-8"
    )
    argv = ["--retriever", "bm25", "--corpus", str(corpus), "--queries", QUERIES]
    assert _input_error(argv, capsys) == f"{corpus}:1: {problem}"


def test_a_corpus_error_names_the_file_and_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"doc_id": "a", "kind": "note", "text": "t"}\n\n'
        '{"doc_id": "b", "kind": "memo", "text": "t"}\n',
        encoding="utf-8",
    )
    argv = ["--retriever", "bm25", "--corpus", str(corpus), "--queries", QUERIES]
    assert _input_error(argv, capsys) == (
        f"{corpus}:3: kind must be one of note, draft, thought, digest, reflection, "
        "long, got 'memo'"
    )


def test_a_relevant_id_outside_its_query_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "qid": "q",
                "query": "reverse",
                "scope": {"task_id": "t-en"},
                "relevant": ["t-zh:n1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    argv = ["--retriever", "bm25", "--corpus", CORPUS, "--queries", str(queries)]
    assert _input_error(argv, capsys) == (
        f"{queries}:1: relevant id 't-zh:n1' is not a candidate of the query: its "
        "task is 't-zh', not 't-en'"
    )


def test_a_queries_file_without_queries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text("\n\n", encoding="utf-8")
    argv = ["--retriever", "bm25", "--corpus", CORPUS, "--queries", str(queries)]
    assert _input_error(argv, capsys) == f"{queries}: no queries"


@pytest.mark.parametrize(
    ("extra_body", "message"),
    [
        (
            "{",
            "--retrieval-extra-body: not JSON: Expecting property name enclosed in "
            "double quotes at column 2",
        ),
        ("[1]", "--retrieval-extra-body: not a JSON object"),
        (
            '{"model": "other"}',
            "--retrieval-extra-body: extra_body must not set 'model'; localmelo "
            "manages that field",
        ),
        (
            '{"x": "cut \\ud83d"}',
            "--retrieval-extra-body: a string holds an unpaired surrogate, which "
            "UTF-8 cannot encode",
        ),
    ],
)
def test_a_bad_retrieval_extra_body(
    extra_body: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--retriever",
        "llm",
        *FILES,
        "--base-url",
        UNUSED_URL,
        "--retrieval-extra-body",
        extra_body,
    ]
    assert _input_error(argv, capsys) == message


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "{",
            "1: not JSON: Expecting property name enclosed in double quotes at "
            "column 2",
        ),
        ("[1]", " not a JSON object with a non-empty messages list"),
        ('{"messages": []}', " not a JSON object with a non-empty messages list"),
        ('{"messages": "hi"}', " not a JSON object with a non-empty messages list"),
        (
            '{"messages": [{"role": "user", "content": "hi \\ud83d"}]}',
            " not valid as a request body: a string holds an unpaired surrogate, "
            "which UTF-8 cannot encode",
        ),
        (
            '{"messages": [{"role": "user", "content": "hi"}], "temperature": NaN}',
            " not valid as a request body: a number is NaN or infinite, which JSON "
            "cannot hold",
        ),
    ],
)
def test_a_bad_interleave_file(
    text: str, problem: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = tmp_path / "request.json"
    request.write_text(text, encoding="utf-8")
    argv = [
        "--retriever",
        "bm25",
        *FILES,
        "--interleave",
        str(request),
        "--base-url",
        UNUSED_URL,
    ]
    assert _input_error(argv, capsys) == f"{request}:{problem}"


# ── Evaluation ──────────────────────────────────────────────────────────────


def _check_query_line(line: dict[str, Any], keys: list[str]) -> None:
    assert list(line) == keys
    assert line["type"] == "query"
    assert isinstance(line["qid"], str) and isinstance(line["source"], str)
    assert isinstance(line["candidates"], int)
    assert all(isinstance(doc_id, str) for doc_id in line["relevant"])
    for rank, hit in enumerate(line["hits"], start=1):
        assert list(hit) == ["doc_id", "score", "rank"]
        assert isinstance(hit["doc_id"], str) and hit["rank"] == rank
        assert isinstance(hit["score"], int | float)
    for key in keys:
        if key.startswith("recall@") or key == "rr":
            assert isinstance(line[key], int | float) and 0 <= line[key] <= 1
    assert isinstance(line["ms_cold"], int)
    assert list(line["fallbacks"]) == ["timeout", "parse", "provider_error"]
    assert isinstance(line["usage"], dict)


def test_bm25_on_the_fixture_files(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--retriever", "bm25", *FILES, "--k", "3,1"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    *queries, summary = _lines(out)
    assert len(queries) == 6
    for line, (qid, candidates, relevant) in zip(queries, FIXTURE_QUERIES, strict=True):
        _check_query_line(line, QUERY_KEYS)
        assert (line["qid"], line["candidates"], line["relevant"]) == (
            qid,
            candidates,
            relevant,
        )
        assert len(line["ms_warm"]) == 3
        assert all(isinstance(ms, int) for ms in line["ms_warm"])
        assert line["ms_interleaved"] is None
        assert (line["fallbacks"], line["usage"]) == (NO_FALLBACKS, {})
    # The excluded draft is never a hit.
    assert "t-en:d2" not in [hit["doc_id"] for hit in queries[3]["hits"]]

    assert list(summary) == SUMMARY_KEYS
    assert (summary["type"], summary["retriever"], summary["model"]) == (
        "summary",
        "bm25",
        None,
    )
    # 6 queries, each with 1 cold and 3 warm calls.
    assert (summary["dim"], summary["n"], summary["calls"]) == (None, 6, 24)
    for key in ("recall@1", "recall@3", "mrr"):
        assert isinstance(summary[key], int | float) and 0 <= summary[key] <= 1
    for key in ("ms_cold_p50", "ms_warm_p50", "ms_warm_p95"):
        assert isinstance(summary[key], int)
    # No interleave, and no tokens for bm25.
    for key in (
        "ms_interleaved_p50",
        "ms_interleaved_p95",
        "prompt_tokens_mean",
        "cached_tokens_mean",
        "completion_tokens_mean",
    ):
        assert summary[key] is None
    assert summary["fallbacks"] == NO_FALLBACKS


def test_repeat_0_leaves_the_warm_figures_null(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--retriever", "bm25", *FILES, "--repeat", "0", "--k", "1"]) == 0
    *queries, summary = _lines(capsys.readouterr().out)
    assert [list(line)[6] for line in queries] == ["recall@1"] * 6
    assert {line["ms_warm"] is None for line in queries} == {True}
    assert (summary["calls"], summary["ms_warm_p50"], summary["ms_warm_p95"]) == (
        6,
        None,
        None,
    )


def test_the_extra_body_can_set_the_temperature(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--retriever", "llm", *FILES, "--base-url", server.base_url]
    argv += ["--retrieval-extra-body", '{"temperature": 0.7}', "--repeat", "0"]
    assert main(argv) == 0
    capsys.readouterr()
    temperatures = {r.body["temperature"] for r in server.chat_requests()}
    assert temperatures == {0.7}


def test_llm_against_the_test_server(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--retriever",
        "llm",
        *FILES,
        "--base-url",
        server.base_url,
        "--chat-model",
        "stub-model",
        "--retrieval-extra-body",
        '{"chat_template_kwargs": {"enable_thinking": false}}',
        "--retrieval-timeout",
        "5",
        "--interleave",
        INTERLEAVE,
        "--repeat",
        "1",
        "--k",
        "1,3",
    ]
    assert main(argv) == 0
    out, err = capsys.readouterr()
    assert err == ""

    requests = server.chat_requests()
    # Per query: cold, warm, then the interleave request before the
    # interleaved call.
    assert ["R" if _is_retrieval(r.body) else "I" for r in requests] == (
        ["R", "R", "I", "R"] * 6
    )
    expected_interleave = json.loads(Path(INTERLEAVE).read_text(encoding="utf-8"))
    expected_interleave.update(model="stub-model", stream=False)
    for request in requests:
        assert request.headers["x-localmelo-attempt"] == "1"
        if _is_retrieval(request.body):
            assert "tools" not in request.body and "tool_choice" not in request.body
            assert len(request.body["messages"]) == 2
            assert request.body["chat_template_kwargs"] == {"enable_thinking": False}
            assert request.body["temperature"] == 0
            assert request.body["model"] == "stub-model"
        else:
            # As the file has it, model replaced, stream false, no extra body.
            assert request.body == expected_interleave

    *queries, summary = _lines(out)
    for line, (qid, _candidates, _relevant) in zip(
        queries, FIXTURE_QUERIES, strict=True
    ):
        _check_query_line(line, QUERY_KEYS)
        assert line["qid"] == qid
        assert len(line["ms_warm"]) == 1 and len(line["ms_interleaved"]) == 1
        # Three calls, each with the server's usage.
        assert line["usage"] == {
            "prompt_tokens": 300,
            "completion_tokens": 15,
            "total_tokens": 315,
        }
    # n1 is the first note among each query's candidates.
    assert queries[0]["hits"] == [{"doc_id": "t-en:n0", "score": 1.0, "rank": 1}]
    assert (summary["retriever"], summary["model"], summary["dim"]) == (
        "llm",
        "stub-model",
        None,
    )
    assert isinstance(summary["ms_interleaved_p50"], int)
    assert isinstance(summary["ms_interleaved_p95"], int)
    assert (
        summary["prompt_tokens_mean"],
        summary["cached_tokens_mean"],
        summary["completion_tokens_mean"],
    ) == (100.0, None, 5.0)
    assert summary["fallbacks"] == NO_FALLBACKS


def test_embedding_against_the_test_server(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--retriever",
        "embedding",
        *FILES,
        "--embedding-base-url",
        server.base_url,
        "--embedding-model",
        "stub-embedding",
        "--repeat",
        "0",
    ]
    assert main(argv) == 0
    out, err = capsys.readouterr()
    assert err == ""
    *queries, summary = _lines(out)
    assert len(queries) == 6
    assert (summary["retriever"], summary["model"], summary["dim"]) == (
        "embedding",
        "stub-embedding",
        16,
    )
    assert summary["fallbacks"] == NO_FALLBACKS
    assert server.chat_requests() == []
    for request in server.requests:
        assert request.path == "/v1/embeddings"
        assert list(request.body) == ["model", "input"]
        assert request.body["model"] == "stub-embedding"
        assert 1 <= len(request.body["input"]) <= 32


def test_a_warning_when_every_llm_call_falls_back(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    server.chat = lambda body: _reply("I would pick the first note.")
    argv = ["--retriever", "llm", *FILES, "--base-url", server.base_url]
    assert main([*argv, "--repeat", "0"]) == 0
    out, err = capsys.readouterr()
    assert _lines(out)[-1]["fallbacks"] == {
        "timeout": 0,
        "parse": 6,
        "provider_error": 0,
    }
    assert err == (
        "melo-retrieve: warning: every llm call fell back to bm25 (parse 6); "
        "check --base-url\n"
    )


def test_a_warning_when_every_embedding_call_falls_back(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    server.embeddings_status = 500
    argv = ["--retriever", "embedding", *FILES, "--embedding-base-url", server.base_url]
    assert main([*argv, "--repeat", "1"]) == 0
    out, err = capsys.readouterr()
    assert _lines(out)[-1]["fallbacks"] == {
        "timeout": 0,
        "parse": 0,
        "provider_error": 12,
    }
    assert err == (
        "melo-retrieve: warning: every embedding call fell back to bm25 "
        "(provider_error 12); check --embedding-base-url\n"
    )


def test_no_warning_when_some_calls_answer(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["no ids here", '{"ids": ["n1"]}'] * 6)
    server.chat = lambda body: _reply(next(answers))
    argv = ["--retriever", "llm", *FILES, "--base-url", server.base_url]
    assert main([*argv, "--repeat", "1"]) == 0
    out, err = capsys.readouterr()
    assert _lines(out)[-1]["fallbacks"]["parse"] == 6
    assert err == ""


def test_a_failed_interleave_request_ends_the_run_with_exit_1(
    server: ModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    server.chat = lambda body: (
        _reply('{"ids": ["n1"]}') if _is_retrieval(body) else (500, "upstream\ndown")
    )
    argv = [
        "--retriever",
        "llm",
        *FILES,
        "--base-url",
        server.base_url,
        "--interleave",
        INTERLEAVE,
    ]
    assert main(argv) == 1
    out, err = capsys.readouterr()
    # The first query had not finished: no line.
    assert out == ""
    assert err == (
        "melo-retrieve: error: interleave request failed: POST /chat/completions "
        "failed with HTTP 500: upstream down\n"
    )


def _interrupt_on_call(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """Make the n-th chat request raise KeyboardInterrupt, as a Ctrl-C would."""
    original = OpenAICompatLLM.chat
    calls: list[int] = []

    async def chat(
        self: OpenAICompatLLM,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        calls.append(1)
        if len(calls) == n:
            raise KeyboardInterrupt
        return await original(self, messages, tools)

    monkeypatch.setattr(OpenAICompatLLM, "chat", chat)


def test_ctrl_c_exits_130_with_whole_lines_and_no_traceback(
    server: ModelServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _interrupt_on_call(monkeypatch, 3)
    argv = ["--retriever", "llm", *FILES, "--base-url", server.base_url]
    assert main([*argv, "--repeat", "0"]) == 130
    out, err = capsys.readouterr()
    # Two queries finished; each line is whole JSON.
    assert [line["qid"] for line in _lines(out)] == ["en-append", "zh-reverse"]
    assert "Traceback" not in err
    assert err == ""


# ── Closing the providers ───────────────────────────────────────────────────


def _record_closes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    closed: list[str] = []
    providers: list[tuple[str, Any]] = [
        ("llm", OpenAICompatLLM),
        ("embedding", OpenAICompatEmbedding),
    ]
    for name, provider in providers:

        async def close(
            self: Any, _name: str = name, _close: Any = provider.close
        ) -> None:
            closed.append(_name)
            await _close(self)

        monkeypatch.setattr(provider, "close", close)
    return closed


def test_the_providers_are_closed_after_a_finished_run(
    server: ModelServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = _record_closes(monkeypatch)
    assert main(["--retriever", "llm", *FILES, "--base-url", server.base_url]) == 0
    assert closed == ["llm"]
    argv = ["--retriever", "embedding", *FILES, "--embedding-base-url", server.base_url]
    assert main(argv) == 0
    assert closed == ["llm", "embedding"]


@pytest.mark.parametrize("retriever", ["llm", "embedding"])
def test_the_provider_is_closed_after_a_failed_interleave(
    retriever: str, server: ModelServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = _record_closes(monkeypatch)
    server.chat = lambda body: (
        _reply('{"ids": ["n1"]}') if _is_retrieval(body) else (500, "down")
    )
    argv = ["--retriever", retriever, *FILES, "--base-url", server.base_url]
    if retriever == "embedding":
        argv += ["--embedding-base-url", server.base_url]
    assert main([*argv, "--interleave", INTERLEAVE]) == 1
    assert closed == [retriever]


def test_the_provider_is_closed_after_ctrl_c(
    server: ModelServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = _record_closes(monkeypatch)
    _interrupt_on_call(monkeypatch, 1)
    assert main(["--retriever", "llm", *FILES, "--base-url", server.base_url]) == 130
    assert closed == ["llm"]


def test_the_embedding_provider_is_closed_after_ctrl_c(
    server: ModelServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed = _record_closes(monkeypatch)
    calls: list[list[str]] = []

    async def embed(self: OpenAICompatEmbedding, texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        raise KeyboardInterrupt  # as a Ctrl-C during the first request would

    monkeypatch.setattr(OpenAICompatEmbedding, "embed", embed)
    argv = ["--retriever", "embedding", *FILES, "--embedding-base-url", server.base_url]
    assert main(argv) == 130
    assert (len(calls), closed) == (1, ["embedding"])
    out, err = capsys.readouterr()
    assert (out, err) == ("", "")


# ── Export ──────────────────────────────────────────────────────────────────

EXPORT_LINE = (
    "melo-retrieve: exported 19 documents (7 notes, 1 reflections, 3 drafts, "
    "8 thought chunks) from 2 tasks\n"
)


def _raw_calls() -> list[RawCall]:
    return load_raw_calls(Path(RAW_CALLS).read_text(encoding="utf-8").split("\n"))


def _expected_export() -> str:
    docs = docs_from_calls(_raw_calls())
    return "".join(
        json.dumps(doc_to_dict(doc), ensure_ascii=False) + "\n" for doc in docs
    )


def test_export_a_raw_call_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--export-run", RAW_CALLS]) == 0
    out, err = capsys.readouterr()
    assert out == _expected_export()
    assert len(out.splitlines()) == 19
    assert err == EXPORT_LINE


def test_export_a_history_directory_and_its_database_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls())
    assert main(["--export-run", str(tmp_path)]) == 0
    assert capsys.readouterr() == (_expected_export(), EXPORT_LINE)
    # The file itself, found by its header whatever its name.
    renamed = tmp_path / "run.data"
    shutil.copyfile(database, renamed)
    assert main(["--export-run", str(renamed)]) == 0
    assert capsys.readouterr() == (_expected_export(), EXPORT_LINE)


@pytest.mark.skipif(not Path("/dev/fd").is_dir(), reason="needs /dev/fd")
def test_export_raw_calls_through_a_pipe(capsys: pytest.CaptureFixture[str]) -> None:
    # As a process substitution, <(cat calls.jsonl), passes them: a pipe's
    # /dev/fd path, whose bytes go to their first reader only.
    data = Path(RAW_CALLS).read_bytes()
    read_end, write_end = os.pipe()
    writer = threading.Thread(target=_write_and_close, args=(write_end, data))
    writer.start()
    try:
        assert main(["--export-run", f"/dev/fd/{read_end}"]) == 0
    finally:
        writer.join(timeout=5)
        os.close(read_end)
    assert capsys.readouterr() == (_expected_export(), EXPORT_LINE)


def _write_and_close(fd: int, data: bytes) -> None:
    with os.fdopen(fd, "wb") as file:
        file.write(data)


def test_export_a_history_without_llm_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_history_sync(tmp_path / "history.db", _raw_calls(), llm_calls=False)
    assert _input_error(["--export-run", str(tmp_path)], capsys) == (
        f"{tmp_path / 'history.db'} has no llm_calls table: it was written before "
        "localmelo recorded model calls. Export the calls as raw-call JSONL instead "
        "(see docs/retrieval.md)."
    )


def test_export_a_history_with_a_missing_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls(), llm_calls=False)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE llm_calls (task_id TEXT, seq INTEGER, kind TEXT, step "
            "INTEGER, attempt INTEGER, outcome TEXT, content TEXT, timestamp REAL)"
        )
        connection.commit()
    finally:
        connection.close()
    assert _input_error(["--export-run", str(database)], capsys) == (
        f"{database}: llm_calls has no column thinking, tool_calls_json, step_ids_json"
    )


def test_export_a_history_row_that_breaks_the_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "history.db"
    write_history_sync(database, _raw_calls())
    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE llm_calls SET outcome = 'aux' WHERE seq = 0")
        connection.commit()
    finally:
        connection.close()
    assert _input_error(["--export-run", str(database)], capsys) == (
        f"{database}: llm_calls row t-en/0: a plan call's outcome is tools, answer "
        "or empty, not aux"
    )


def test_export_warns_about_step_ids_without_a_steps_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "history.db"
    call = RawCall(
        "t",
        0,
        thinking="Let me read both files.",
        tool_calls=(
            ToolCall("file_read", {"path": "app.py"}, id="c1"),
            ToolCall("file_read", {"path": "lib.py"}, id="c2"),
        ),
        tool_results=(
            ToolResult("file_read", output="x = 1\n"),
            ToolResult("file_read", output="y = 2\n"),
        ),
        outcome="tools",
    )
    ids = write_history_sync(database, [call])[("t", 0)]
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE llm_calls SET step_ids_json = ?",
            (json.dumps([ids[0], "gone-1", "gone-2"]),),
        )
        connection.commit()
    finally:
        connection.close()
    assert main(["--export-run", str(tmp_path)]) == 0
    out, err = capsys.readouterr()
    assert err.splitlines() == [
        "melo-retrieve: warning: 2 step ids in llm_calls.step_ids_json had no "
        "matching steps row; their results are left out",
        "melo-retrieve: exported 2 documents (1 notes, 0 reflections, 0 drafts, "
        "1 thought chunks) from 1 tasks",
    ]
    # The second read has no result: one observed line, for app.py.
    note = _lines(out)[0]
    assert note["text"].count("observed: read app.py: 6 chars, sha ") == 1
    assert "lib.py:" not in note["text"]


def test_export_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "none.jsonl")
    assert _input_error(["--export-run", missing], capsys) == (
        f"{missing}: No such file or directory"
    )
    assert _input_error(["--export-run", str(tmp_path)], capsys) == (
        f"{tmp_path / 'history.db'}: No such file or directory"
    )
    calls = tmp_path / "calls.jsonl"
    calls.write_text('{"task_id": "t", "seq": 0}\n{"task_id": "t"}\n', encoding="utf-8")
    assert _input_error(["--export-run", str(calls)], capsys) == (
        f"{calls}:2: seq is missing"
    )


def test_the_exported_corpus_reads_back_with_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--export-run", RAW_CALLS]) == 0
    corpus = tmp_path / "exported.jsonl"
    corpus.write_text(capsys.readouterr().out, encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "qid": "q",
                "query": "append concatenation list1 list2",
                "scope": {"task_id": "t-en"},
                "relevant": ["d2", "n2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    argv = ["--retriever", "bm25", "--corpus", str(corpus), "--queries", str(queries)]
    assert main(argv) == 0
    query, summary = _lines(capsys.readouterr().out)
    # The 10 docs of t-en.
    assert (query["candidates"], summary["n"]) == (10, 1)


def test_pyproject_names_the_console_script() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["melo-retrieve"] == "localmelo.support.retrieve:main"

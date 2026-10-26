"""``melo-retrieve``: compare memory retrievers offline, or export a run as a corpus.

Evaluation::

    melo-retrieve --retriever bm25|llm|embedding --corpus FILE --queries FILE
                  [--k 1,3,5] [--repeat N] [--retrieval-timeout SECONDS]
                  [--interleave REQUEST.json] [--base-url URL] [--chat-model NAME]
                  [--retrieval-extra-body JSON] [--embedding-base-url URL]
                  [--embedding-model NAME]

runs every labelled query against the corpus with one retriever and prints
one JSON line per query as it finishes, then a summary line. Export::

    melo-retrieve --export-run DIR|FILE

prints the corpus lines of a run's model calls: from its ``history.db`` (in
DIR, or the file itself), or from raw-call JSONL. ``docs/retrieval.md``
documents the formats, the output lines and the exit codes.

The command reads no config file and writes nothing but stdout and stderr.
Exit codes: 0 when the run finished (fallbacks included), 2 for usage and
input errors, 1 for a failed interleave request, 130 on Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx

from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.memory.retrieval import (
    RETRIEVER_NAMES,
    CorpusError,
    Doc,
    EvalReport,
    LabelledQuery,
    RawCall,
    Retriever,
    build_retriever,
    doc_to_dict,
    docs_from_calls,
    evaluate,
    load_corpus,
    load_queries,
    load_raw_calls,
    send_from_provider,
)
from localmelo.melo.memory.retrieval.corpus import (
    SQLITE_HEADER,
    NoCallsTableError,
    load_history_calls,
)
from localmelo.melo.memory.retrieval.llm import RETRIEVE_TIMEOUT_S
from localmelo.support.providers._http import attempt_headers, post_json
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm._protocol import prepare_extra_body
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

PROG = "melo-retrieve"

EXIT_OK = 0
EXIT_INTERLEAVE_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DEFAULT_KS = (1, 3, 5)
DEFAULT_REPEAT = 3
DEFAULT_MODEL = "default"  # as melo's --chat-model and --embedding-model
# The providers' HTTP timeout: their own default, or more, so that it never
# cuts a retrieval request before --retrieval-timeout does.
MIN_HTTP_TIMEOUT_S = 120.0
# The temperature of an llm retrieval request unless --retrieval-extra-body sets
# one. Picking ids is a choice, not writing, and a fixed choice makes runs
# repeatable; a server's own default may be 1.0.
RETRIEVAL_TEMPERATURE = 0

HISTORY_FILE = "history.db"

# The flags of an evaluation, with their argparse destinations.
_EVALUATION_FLAGS = {
    "--retriever": "retriever",
    "--corpus": "corpus",
    "--queries": "queries",
    "--k": "k",
    "--repeat": "repeat",
    "--retrieval-timeout": "retrieval_timeout",
    "--interleave": "interleave",
    "--base-url": "base_url",
    "--chat-model": "chat_model",
    "--retrieval-extra-body": "retrieval_extra_body",
    "--embedding-base-url": "embedding_base_url",
    "--embedding-model": "embedding_model",
}

_T = TypeVar("_T")


class _InputError(Exception):
    """An input the run cannot use, reported in one line with exit code 2."""


class _InterleaveError(Exception):
    """The interleave request failed; the run ends with exit code 1."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``melo-retrieve`` with *argv* (default: ``sys.argv[1:]``).

    Returns the exit code. Usage errors print argparse's usage line and
    one error line; input errors print one error line.
    """
    try:
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
            _check_flags(parser, args)
        except SystemExit as exc:  # --help, or a usage error argparse printed
            return _exit_code(exc)
        try:
            if args.export_run is not None:
                return _export_run(args.export_run)
            return _evaluate(args)
        except _InputError as exc:
            _error(str(exc))
            return EXIT_USAGE
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


# ── Flags ──


def _positive_int(text: str) -> int:
    """An integer >= 1; a copy of melo's own check (``__main__.py``)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _ks(text: str) -> tuple[int, ...]:
    """``--k``: integers >= 1 separated by commas, deduplicated and sorted."""
    return tuple(sorted({_positive_int(part) for part in text.split(",")}))


def _non_negative_int(text: str) -> int:
    """An integer >= 0, checked as ``_positive_int`` checks its own."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def _seconds(text: str) -> float:
    """A finite number of seconds > 0; a copy of melo's own check."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number of seconds > 0, got {text}"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Compare memory retrievers offline on labelled queries, or export "
            "a run's model calls as a retrieval corpus."
        ),
        epilog="Formats, output lines and exit codes: docs/retrieval.md",
    )
    # Every flag defaults to None, so that a given flag can be told from
    # an absent one; the documented defaults apply after the checks.
    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--retriever", choices=RETRIEVER_NAMES, help="the retriever to evaluate"
    )
    evaluation.add_argument(
        "--corpus", metavar="FILE", help="corpus JSONL, one document per line"
    )
    evaluation.add_argument(
        "--queries",
        metavar="FILE",
        help="labelled-query JSONL, one query per line",
    )
    evaluation.add_argument(
        "--k",
        type=_ks,
        metavar="K[,K...]",
        help="the k of recall@k, separated by commas (default: 1,3,5)",
    )
    evaluation.add_argument(
        "--repeat",
        type=_non_negative_int,
        metavar="N",
        help=(
            "warm calls per query, and interleaved calls with --interleave "
            "(default: 3)"
        ),
    )
    evaluation.add_argument(
        "--retrieval-timeout",
        type=_seconds,
        metavar="SECONDS",
        help=(
            "llm only: seconds before a retrieval request falls back to bm25 "
            "(default: 30)"
        ),
    )
    evaluation.add_argument(
        "--interleave",
        metavar="REQUEST.json",
        help=(
            "a chat request body to send to --base-url before each "
            "interleaved call; its time is not counted"
        ),
    )
    evaluation.add_argument(
        "--base-url",
        metavar="URL",
        help=(
            "OpenAI-compatible chat API base URL, including /v1 "
            "(--retriever llm, --interleave)"
        ),
    )
    evaluation.add_argument(
        "--chat-model", metavar="NAME", help="chat model name (default: default)"
    )
    evaluation.add_argument(
        "--retrieval-extra-body",
        metavar="JSON",
        help=(
            "llm only: a JSON object merged into every retrieval request body, "
            'e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\''
        ),
    )
    evaluation.add_argument(
        "--embedding-base-url",
        metavar="URL",
        help=(
            "OpenAI-compatible embedding API base URL, including /v1 "
            "(--retriever embedding)"
        ),
    )
    evaluation.add_argument(
        "--embedding-model",
        metavar="NAME",
        help="embedding model name (default: default)",
    )
    export = parser.add_argument_group("export")
    export.add_argument(
        "--export-run",
        metavar="DIR|FILE",
        help=(
            "print a run's model calls as corpus lines: a directory holding "
            "history.db, a history.db file, or raw-call JSONL"
        ),
    )
    return parser


def _check_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse flags that do not go together (``parser.error``: exit 2)."""
    given = [
        flag
        for flag, dest in _EVALUATION_FLAGS.items()
        if getattr(args, dest) is not None
    ]
    if args.export_run is not None:
        if given:
            parser.error(f"--export-run cannot be combined with {', '.join(given)}")
        return
    missing = [
        flag for flag in ("--retriever", "--corpus", "--queries") if flag not in given
    ]
    if missing:
        parser.error(
            "an evaluation needs --retriever, --corpus and --queries "
            f"(missing: {', '.join(missing)}); or give --export-run"
        )
    llm = args.retriever == "llm"
    embedding = args.retriever == "embedding"
    if llm and args.base_url is None:
        parser.error("--retriever llm needs --base-url")
    if embedding and args.embedding_base_url is None:
        parser.error("--retriever embedding needs --embedding-base-url")
    for flag in ("--retrieval-extra-body", "--retrieval-timeout"):
        if flag in given and not llm:
            parser.error(f"{flag} applies only to --retriever llm")
    for flag in ("--embedding-base-url", "--embedding-model"):
        if flag in given and not embedding:
            parser.error(f"{flag} applies only to --retriever embedding")
    for flag in ("--base-url", "--chat-model"):
        if flag in given and not (llm or args.interleave is not None):
            parser.error(f"{flag} applies only to --retriever llm or --interleave")
    if args.interleave is not None and args.base_url is None:
        parser.error("--interleave needs --base-url")
    for flag, url in (
        ("--base-url", args.base_url),
        ("--embedding-base-url", args.embedding_base_url),
    ):
        if url is None:
            continue
        try:
            # The URL as the providers and the interleave client parse it.
            httpx.URL(url.rstrip("/"))
        except httpx.InvalidURL as exc:
            parser.error(f"{flag} is not a valid URL: {exc}")


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return EXIT_OK
    return exc.code if isinstance(exc.code, int) else EXIT_USAGE


# ── Evaluation ──


@dataclass(frozen=True)
class _Evaluation:
    """What an evaluation runs, once every input is read and checked."""

    retriever: str
    corpus: list[Doc]
    queries: list[LabelledQuery]
    ks: tuple[int, ...]
    repeats: int
    timeout_s: float
    base_url: str | None
    chat_model: str
    extra_body: dict[str, Any]
    embedding_base_url: str | None
    embedding_model: str
    interleave_body: dict[str, Any] | None

    @property
    def model(self) -> str | None:
        """The model the summary line names; ``None`` for bm25."""
        if self.retriever == "llm":
            return self.chat_model
        if self.retriever == "embedding":
            return self.embedding_model
        return None


def _evaluate(args: argparse.Namespace) -> int:
    chat_model = args.chat_model or DEFAULT_MODEL
    extra_body = {
        "temperature": RETRIEVAL_TEMPERATURE,
        **_extra_body(args.retrieval_extra_body),
    }
    corpus = _load(args.corpus, load_corpus)
    queries = _load(args.queries, lambda lines: load_queries(lines, corpus))
    if not queries:
        raise _InputError(f"{args.queries}: no queries")
    interleave_body = (
        None if args.interleave is None else _interleave_body(args.interleave)
    )
    if interleave_body is not None:
        interleave_body["model"] = chat_model
        interleave_body["stream"] = False
    evaluation = _Evaluation(
        retriever=args.retriever,
        corpus=corpus,
        queries=queries,
        ks=args.k or DEFAULT_KS,
        repeats=DEFAULT_REPEAT if args.repeat is None else args.repeat,
        timeout_s=(
            RETRIEVE_TIMEOUT_S
            if args.retrieval_timeout is None
            else args.retrieval_timeout
        ),
        base_url=args.base_url,
        chat_model=chat_model,
        extra_body=extra_body,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model or DEFAULT_MODEL,
        interleave_body=interleave_body,
    )
    return asyncio.run(_run_evaluation(evaluation))


async def _run_evaluation(evaluation: _Evaluation) -> int:
    """Evaluate, printing each line as it is ready; close every client after."""
    http_timeout = max(MIN_HTTP_TIMEOUT_S, evaluation.timeout_s)
    async with contextlib.AsyncExitStack() as clients:
        llm: OpenAICompatLLM | None = None
        if evaluation.retriever == "llm":
            assert evaluation.base_url is not None  # checked with the flags
            llm = OpenAICompatLLM(
                evaluation.base_url,
                evaluation.chat_model,
                timeout=http_timeout,
                extra_body=evaluation.extra_body,
            )
            clients.push_async_callback(llm.close)
        embedder: OpenAICompatEmbedding | None = None
        if evaluation.retriever == "embedding":
            assert evaluation.embedding_base_url is not None  # checked with the flags
            embedder = OpenAICompatEmbedding(
                evaluation.embedding_base_url, evaluation.embedding_model
            )
            clients.push_async_callback(embedder.close)
        interleave: Callable[[], Awaitable[None]] | None = None
        if evaluation.interleave_body is not None:
            assert evaluation.base_url is not None  # checked with the flags
            client = httpx.AsyncClient(
                base_url=evaluation.base_url.rstrip("/"), timeout=http_timeout
            )
            clients.push_async_callback(client.aclose)
            interleave = _interleave(client, evaluation.interleave_body)

        send = None if llm is None else send_from_provider(llm)
        options: dict[str, Any] = (
            {"timeout_s": evaluation.timeout_s} if evaluation.retriever == "llm" else {}
        )

        def fresh() -> Retriever:
            # A new retriever, with empty caches, for each query's cold call;
            # one provider serves them all.
            return build_retriever(
                evaluation.retriever, send=send, embedding=embedder, **options
            )

        try:
            report = await evaluate(
                fresh,
                corpus=evaluation.corpus,
                queries=evaluation.queries,
                ks=evaluation.ks,
                repeats=evaluation.repeats,
                interleave=interleave,
                model=evaluation.model,
                on_query=lambda result: _print_line(result.to_line(evaluation.ks)),
            )
        except _InterleaveError as exc:
            _error(f"interleave request failed: {exc}")
            return EXIT_INTERLEAVE_FAILED
        _print_line(report.summary_line())
        _warn_if_every_call_fell_back(report, evaluation.retriever)
        return EXIT_OK


def _interleave(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> Callable[[], Awaitable[None]]:
    """A coroutine function that posts *body* to ``/chat/completions`` once."""

    async def interleave() -> None:
        try:
            await post_json(
                client, "/chat/completions", body, headers=attempt_headers()
            )
        except ProviderError as exc:
            raise _InterleaveError(" ".join(str(exc).split())) from exc

    return interleave


def _warn_if_every_call_fell_back(report: EvalReport, retriever: str) -> None:
    if retriever == "bm25" or report.calls == 0:
        return
    if sum(report.fallbacks.values()) < report.calls:
        return
    reasons = ", ".join(
        f"{reason} {count}" for reason, count in report.fallbacks.items() if count
    )
    flag = "--base-url" if retriever == "llm" else "--embedding-base-url"
    _warning(f"every {retriever} call fell back to bm25 ({reasons}); check {flag}")


# ── Inputs ──


def _read_text(path: str) -> str:
    """The UTF-8 text of the file at *path*."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise _InputError(f"{path}: {exc.strerror or exc}") from None
    return _decode(path, data)


def _decode(path: str, data: bytes) -> str:
    """*data*, read from the file at *path*, as UTF-8 text."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data.count(b"\n", 0, exc.start) + 1
        raise _InputError(f"{path}:{line}: not UTF-8 text") from None


def _load(path: str, loader: Callable[[list[str]], _T], text: str | None = None) -> _T:
    """*loader* on the lines of the file at *path*, its errors as input errors.

    *text* is the file's text when the caller has read it already.
    """
    # Split at "\n" only: a JSON string may hold other line separators.
    lines = (_read_text(path) if text is None else text).split("\n")
    try:
        return loader(lines)
    except CorpusError as exc:
        where = path if exc.line is None else f"{path}:{exc.line}"
        raise _InputError(f"{where}: {exc}") from None


def _body_problem(body: object) -> str:
    """Why *body* cannot go out as a JSON request body, or ``""``.

    Python's json reads NaN, Infinity and an escaped unpaired surrogate,
    but a request body cannot hold them: httpx would fail while sending.
    """
    try:
        json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeEncodeError:
        return "a string holds an unpaired surrogate, which UTF-8 cannot encode"
    except ValueError:
        return "a number is NaN or infinite, which JSON cannot hold"
    return ""


def _extra_body(text: str | None) -> dict[str, Any]:
    """The ``--retrieval-extra-body`` object, checked as the provider checks it."""
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _InputError(
            f"--retrieval-extra-body: not JSON: {exc.msg} at column {exc.colno}"
        ) from None
    except (ValueError, RecursionError):
        raise _InputError("--retrieval-extra-body: not JSON") from None
    if not isinstance(value, dict):
        raise _InputError("--retrieval-extra-body: not a JSON object")
    try:
        body = prepare_extra_body(value)
    except ValueError as exc:
        raise _InputError(f"--retrieval-extra-body: {exc}") from None
    problem = _body_problem(body)
    if problem:
        raise _InputError(f"--retrieval-extra-body: {problem}")
    return body


def _interleave_body(path: str) -> dict[str, Any]:
    """The request body in the ``--interleave`` file."""
    text = _read_text(path)
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _InputError(
            f"{path}:{exc.lineno}: not JSON: {exc.msg} at column {exc.colno}"
        ) from None
    except (ValueError, RecursionError):
        raise _InputError(f"{path}: not JSON") from None
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("messages"), list)
        or not body["messages"]
    ):
        raise _InputError(f"{path}: not a JSON object with a non-empty messages list")
    problem = _body_problem(body)
    if problem:
        raise _InputError(f"{path}: not valid as a request body: {problem}")
    return body


# ── Export ──


def _export_run(target: str) -> int:
    """Print the corpus lines of the run at *target*, then a count on stderr."""
    if Path(target).is_dir():
        calls, skipped = _history(str(Path(target) / HISTORY_FILE))
    else:
        data = _unless_sqlite(target)
        if data is None:
            calls, skipped = _history(target)
        else:
            text = _decode(target, data)
            calls, skipped = _load(target, load_raw_calls, text), 0
    docs = docs_from_calls(calls)
    for doc in docs:
        _print_line(doc_to_dict(doc))
    if skipped:
        _warning(
            f"{skipped} step ids in llm_calls.step_ids_json had no matching "
            "steps row; their results are left out"
        )
    kinds = [doc.kind for doc in docs]
    tasks = len({call.task_id for call in calls})
    print(
        f"{PROG}: exported {len(docs)} documents ({kinds.count('note')} notes, "
        f"{kinds.count('reflection')} reflections, {kinds.count('draft')} drafts, "
        f"{kinds.count('thought')} thought chunks) from {tasks} tasks",
        file=sys.stderr,
    )
    return EXIT_OK


def _unless_sqlite(path: str) -> bytes | None:
    """The bytes of the file at *path*; ``None`` when it has the SQLite header.

    The file is opened once, and its bytes after the header are read from
    the same handle: a pipe or FIFO (``/dev/stdin``, a process
    substitution) gives its bytes to its first reader only.
    """
    try:
        with open(path, "rb") as file:
            head = file.read(len(SQLITE_HEADER))
            return None if head == SQLITE_HEADER else head + file.read()
    except OSError as exc:
        raise _InputError(f"{path}: {exc.strerror or exc}") from None


def _history(path: str) -> tuple[list[RawCall], int]:
    """The calls of the ``history.db`` at *path*, its errors as input errors."""
    if not Path(path).is_file():
        raise _InputError(f"{path}: No such file or directory")
    try:
        return load_history_calls(path)
    except NoCallsTableError:
        raise _InputError(
            f"{path} has no llm_calls table: it was written before localmelo "
            "recorded model calls. Export the calls as raw-call JSONL instead "
            "(see docs/retrieval.md)."
        ) from None
    except CorpusError as exc:
        raise _InputError(f"{path}: {exc}") from None


# ── Output ──


def _print_line(obj: dict[str, Any]) -> None:
    """*obj* as one JSON line on stdout, written whole and flushed."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error(message: str) -> None:
    print(f"{PROG}: error: {message}", file=sys.stderr)


def _warning(message: str) -> None:
    print(f"{PROG}: warning: {message}", file=sys.stderr)

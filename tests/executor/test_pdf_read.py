"""file_read of a PDF: text by page, a picture for a page with no text, one-line refusals.

The PDFs are built in memory (:mod:`localmelo.tests.pdf_fixtures`) and read by
the real child process; no model and no network. One test runs the agent with a
scripted model that cannot see pictures.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import signal
import struct
import subprocess
import sys
import textwrap
import time
import tomllib
import zlib
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._tool_images import MAX_IMAGE_WIDTH, WITHHELD_NOTE
from localmelo.melo.checker import Checker
from localmelo.melo.checker.validators import MAX_OUTPUT_LEN
from localmelo.melo.executor import (
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    builtin_names,
    pdf,
    register_builtins,
)
from localmelo.melo.executor import builtins as builtins_module
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import Message, ToolCall, decode_base64
from localmelo.tests.agent.test_tool_images import ScriptedLLM
from localmelo.tests.pdf_fixtures import LOCKED, make_pdf

CV = (
    "Jane Doe\nEducation: MSc in Robotics (2024)\n"
    "Experience: Acme Corp, engineer, 2019 - 2023, Pittsburgh"
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return Path(os.path.realpath(path))


def _executor(root: Path) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="interactive"), workspace_root=str(root))
    register_builtins(
        executor,
        hippo,
        names=builtin_names(),
        mode="interactive",
        workspace_root=str(root),
    )
    return executor


async def _read(root: Path, name: str, **arguments: Any) -> ExecutionOutcome:
    return await _executor(root).execute_structured(
        ExecutionRequest("file_read", {"path": name, **arguments})
    )


def _width(image: Any) -> int:
    raw = decode_base64(image.data)
    assert raw is not None and raw.startswith(b"\x89PNG\r\n\x1a\n")
    return int(struct.unpack(">I", raw[16:20])[0])


# ── Text ──


async def test_an_english_pdf_reads_as_text_page_by_page(root: Path) -> None:
    (root / "cv.pdf").write_bytes(make_pdf([CV, "Globex Corp\nSkills: Python"]))
    outcome = await _read(root, "cv.pdf")
    assert outcome.status == ExecutionStatus.SUCCESS and outcome.images == []
    text = outcome.output
    order = ["--- page 1 of 2 ---", "Jane Doe", "--- page 2 of 2 ---", "Globex Corp"]
    assert [text.index(part) for part in order] == sorted(text.index(p) for p in order)
    # A row stays whole: a place and a date stay on the line of their employer.
    assert "Experience: Acme Corp, engineer, 2019 - 2023, Pittsburgh\n" in text
    # start_line and end_line count the lines of that text.
    part = (await _read(root, "cv.pdf", start_line=2, end_line=3)).output
    assert part.startswith("Jane Doe\nEducation: MSc in Robotics (2024)\n")
    assert "Experience" not in part and "showing 2-3" in part
    assert "start_line=4" in part


async def test_a_chinese_pdf_keeps_its_words_and_rows(root: Path) -> None:
    rows = "李明\n教育经历：示例大学 计算机学院 2020.09 – 2024.06 中国北京\nMixed 中英 line"
    (root / "简历.pdf").write_bytes(make_pdf([rows]))
    outcome = await _read(root, "简历.pdf")
    assert outcome.output == f"--- page 1 of 1 ---\n{rows}\n"


async def test_a_long_pdf_is_cut_by_lines_and_its_last_page_picture_comes_when_read(
    root: Path,
) -> None:
    pages = [
        "\n".join(f"Row {page} {line} " + "word " * 10 for line in range(50))
        for page in range(25)
    ]
    (root / "long.pdf").write_bytes(make_pdf([*pages, None]))
    first = await _read(root, "long.pdf")
    assert len(first.output) <= MAX_OUTPUT_LEN and "truncated" in first.output
    assert first.images == [] and "no text" not in first.output
    total = int(
        first.output.split("the file has ")[1].split(" lines")[0].replace(",", "")
    )
    # The page line and the line about the page with no text are the last two.
    last = await _read(root, "long.pdf", start_line=total - 1)
    assert "[page 26: no text, picture attached]" in last.output
    assert len(last.images) == 1


# ── Pages with no text ──


async def test_a_page_with_no_text_comes_as_a_picture_and_never_as_an_empty_document(
    root: Path,
) -> None:
    (root / "scan.pdf").write_bytes(
        make_pdf(["Cover page with some words in it", None, None, None, None, None])
    )
    outcome = await _read(root, "scan.pdf")
    assert outcome.status == ExecutionStatus.SUCCESS
    text = outcome.output
    # At most three pictures come with one result, in page order, no wider than a screenshot.
    assert len(outcome.images) == pdf.MAX_PICTURES == 3
    assert all(_width(image) <= MAX_IMAGE_WIDTH for image in outcome.images)
    for page in (2, 3, 4):
        assert f"[page {page}: no text, picture attached]" in text
    for page in (5, 6):
        assert f"[page {page}: no text, picture not sent]" in text
    assert "Pages 5-6 have no text and no picture came with this result" in text
    # The note says where to read on, and that read gets the pictures left out.
    line = int(text.split("start_line=")[1].split(" ")[0])
    more = await _read(root, "scan.pdf", start_line=line)
    assert len(more.images) == 2 and "picture attached" in more.output
    assert "no picture came" not in more.output
    assert pdf.PICTURE_WIDTH == MAX_IMAGE_WIDTH


async def test_a_model_that_cannot_see_pictures_is_told_which_pages_it_did_not_read(
    root: Path,
) -> None:
    (root / "scan.pdf").write_bytes(
        make_pdf(["Cover page with some words in it", None])
    )
    call = ToolCall(tool_name="file_read", arguments={"path": "scan.pdf"}, id="c0")
    llm = ScriptedLLM(
        [Message(role="assistant", tool_calls=[call]), Message("assistant", "done")],
        takes=False,
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            attempt_policy="none",
            builtins=frozenset({"file_read"}),
            workspace_root=str(root),
        ),
    )
    assert await agent.run("read scan.pdf") == "done"
    [content] = [m.content for m in llm.requests[-1] if m.role == "tool"]
    assert "[page 2: no text, picture attached]" in content
    assert "say that this page could not be read" in content
    assert WITHHELD_NOTE in content


# ── Files melo cannot read ──


@pytest.mark.parametrize(
    ("data", "words"),
    [
        (b"%PDF-1.4\nthis is not a pdf at all\n", "damaged or is not a PDF"),
        (LOCKED, "password-protected"),
        (make_pdf(["x"]) + b"\n%" + b"p" * 2_500_000, "larger than 2 MB"),
    ],
    ids=["broken", "password", "too-large"],
)
async def test_a_file_that_cannot_be_read_gives_one_plain_line(
    root: Path, monkeypatch: pytest.MonkeyPatch, data: bytes, words: str
) -> None:
    monkeypatch.setattr(pdf, "MAX_BYTES", 2 * 1024 * 1024)
    (root / "bad.pdf").write_bytes(data)
    outcome = await _read(root, "bad.pdf")
    assert outcome.status == ExecutionStatus.ERROR and words in outcome.error
    assert "\n" not in outcome.error and "Traceback" not in outcome.error


async def test_a_pdf_is_known_by_its_first_bytes_not_its_name(root: Path) -> None:
    (root / "resume.txt").write_bytes(make_pdf([CV]))
    (root / "notes.pdf").write_text("plain notes, not a PDF\n")
    assert "--- page 1 of 1 ---" in (await _read(root, "resume.txt")).output
    assert (await _read(root, "notes.pdf")).output == "plain notes, not a PDF\n"


def _child(tmp_path: Path, code: str) -> Path:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(code))
    return script


async def test_a_child_that_hangs_is_killed_and_one_that_crashes_is_named(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "cv.pdf").write_bytes(make_pdf([CV]))
    pid_file = tmp_path / "pid"
    hang = _child(
        tmp_path,
        f"""
        import os, time
        open({str(pid_file)!r}, "w").write(str(os.getpid()))
        time.sleep(60)
        """,
    )
    monkeypatch.setattr(pdf, "_CHILD", hang)
    monkeypatch.setattr(pdf, "TIMEOUT_S", 1.0)
    outcome = await _read(root, "cv.pdf")
    assert outcome.status == ExecutionStatus.ERROR
    assert "took longer than 1 seconds" in outcome.error
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)

    # Any child that dies of a signal is read the same way; SIGSEGV would also
    # write a system crash report on every run.
    crash = _child(
        tmp_path, f"import os, signal\nos.kill(os.getpid(), {signal.SIGKILL})"
    )
    monkeypatch.setattr(pdf, "_CHILD", crash)
    outcome = await _read(root, "cv.pdf")
    assert "the PDF reader stopped on it" in outcome.error


# ── What the child may cost, and what the parent believes of it ──


async def test_a_child_that_grows_past_the_memory_cap_is_killed(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "cv.pdf").write_bytes(make_pdf([CV]))
    pid_file = tmp_path / "pid"
    grow = _child(
        tmp_path,
        f"""
        import os, sys, time
        open({str(pid_file)!r}, "w").write(str(os.getpid()))
        blocks = []
        for _ in range(60):  # 600 MB, touched, a block every 50 ms
            blocks.append(bytearray(b"x" * (10 * 1024 * 1024)))
            time.sleep(0.05)
        time.sleep(3)
        sys.stdout.write('{{"status": "broken"}}')
        """,
    )
    monkeypatch.setattr(pdf, "_CHILD", grow)
    monkeypatch.setattr(pdf, "MAX_RSS_BYTES", 150 * 1024 * 1024)
    monkeypatch.setattr(pdf, "WATCH_S", 0.05)
    outcome = await _read(root, "cv.pdf")
    assert outcome.status == ExecutionStatus.ERROR
    assert "more than 150 MB of memory" in outcome.error and "\n" not in outcome.error
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_the_child_ends_itself_by_its_own_clock_when_nobody_answers() -> None:
    # No bytes and no end of input: the parent is gone or stuck. The child's
    # alarm (a wall clock, whatever the child does) ends it, so its life is bounded.
    limits = {"pages": 1, "chars": 1, "cpu": 30, "wall": 1}
    proc = subprocess.Popen(
        [sys.executable, "-I", str(pdf._CHILD), "text", json.dumps(limits)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc.wait(timeout=8) == -signal.SIGALRM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _answering(body: str) -> str:
    return f"import sys\nsys.stdout.write({body})"


@pytest.mark.parametrize(
    "script",
    [
        _answering("'[' * 200_000"),  # a decoder that runs out of depth
        _answering(
            '\'{"status": "ok", "pages": 1, "texts": ["\' + \'a\' * 5_000_000 + \'"]}\''
        ),
    ],
    ids=["too-deep", "too-much-text"],
)
async def test_an_answer_the_parent_cannot_use_gives_one_plain_line(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    (root / "cv.pdf").write_bytes(make_pdf([CV]))
    monkeypatch.setattr(pdf, "_CHILD", _child(tmp_path, script))
    outcome = await _read(root, "cv.pdf")
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error.startswith("melo could not read this PDF")
    assert "\n" not in outcome.error and "Traceback" not in outcome.error


@pytest.mark.parametrize(
    "answer",
    [
        {"pages": True, "texts": ["abc"]},
        {"pages": "1", "texts": ["abc"]},
        {"pages": 3, "texts": ["a", "b", "c"]},  # more pages than MAX_PAGES (2 here)
        {"pages": 1, "texts": ["x" * 11]},  # more text than MAX_CHARS (10 here)
        {"pages": 1, "texts": ["a", "b"]},  # more texts than pages
        {"pages": 1, "texts": []},
    ],
)
async def test_the_parent_holds_the_child_to_the_text_it_may_answer(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    async def fake(
        mode: str, limits: dict[str, Any], data: bytes, limit: int
    ) -> dict[str, Any]:
        return {"status": "ok", **answer}

    monkeypatch.setattr(pdf, "_ask", fake)
    monkeypatch.setattr(pdf, "MAX_PAGES", 2)
    monkeypatch.setattr(pdf, "MAX_CHARS", 10)
    with pytest.raises(pdf.PdfError, match="damaged or is not a PDF"):
        await pdf.read_text(b"%PDF-")


async def test_the_parent_cleans_the_text_the_child_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(
        mode: str, limits: dict[str, Any], data: bytes, limit: int
    ) -> dict[str, Any]:
        return {"status": "ok", "pages": 1, "texts": ["a\x1b[31mb\x00 c  \r\nd\n\n"]}

    monkeypatch.setattr(pdf, "_ask", fake)
    assert (await pdf.read_text(b"%PDF-")).texts == ("a[31mb c\nd",)


def _png_bytes(
    width: int, height: int, order: tuple[str, ...] = ("IHDR", "IDAT", "IEND")
) -> bytes:
    """A whole PNG (all black), its chunks in *order*."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    bodies = {
        "IHDR": struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        "IDAT": zlib.compress(b"\x00" * (height * (1 + 3 * width))),
        "IEND": b"",
    }
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        chunk(kind.encode(), bodies[kind]) for kind in order
    )


def _png_of(width: int, height: int) -> str:
    return base64.b64encode(_png_bytes(width, height)).decode()


async def test_the_parent_admits_only_the_pictures_it_asked_for_within_the_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _png_of(600, 800)
    monkeypatch.setattr(pdf, "PICTURE_BYTES", len(good))
    answers: list[Any] = [
        5,  # not a list
        [1, "x", None],  # not mappings
        [{"page": True, "png": good}],  # a flag is not a page
        [{"page": 9, "png": good}],  # a page it did not ask for
        [{"page": 2, "png": base64.b64encode(b"not a picture").decode()}],
        [{"page": 2, "png": _png_of(5000, 10)}],  # wider than a screenshot
        [{"page": 2, "png": _png_of(10, 9000)}],  # taller
        [{"page": 2, "png": good + "AAAA"}],  # over the bytes one picture may take
    ]

    async def fake(
        mode: str, limits: dict[str, Any], data: bytes, limit: int
    ) -> dict[str, Any]:
        return {"status": "ok", "pictures": answers.pop(0)}

    monkeypatch.setattr(pdf, "_ask", fake)
    while answers:
        assert await pdf.render_pages(b"%PDF-", [2, 3]) == {}
    answers.append([{"page": 2, "png": good}, {"page": 2, "png": good}])
    assert list(await pdf.render_pages(b"%PDF-", [2, 3])) == [2]


async def test_a_picture_is_made_smaller_to_fit_the_bytes_it_may_take(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "scan.pdf").write_bytes(
        make_pdf(["Cover page with some words in it", None])
    )
    # Drawn at full size this page is 16,400 characters of base64.
    monkeypatch.setattr(pdf, "PICTURE_BYTES", 8_000)
    outcome = await _read(root, "scan.pdf")
    [image] = outcome.images
    assert len(image.data) <= 8_000 and _width(image) < 1_200
    assert "[page 2: no text, picture attached]" in outcome.output
    # A page that cannot be made small enough is a failed page, not an error of the read.
    monkeypatch.setattr(pdf, "PICTURE_BYTES", 100)
    outcome = await _read(root, "scan.pdf")
    assert outcome.status == ExecutionStatus.SUCCESS and outcome.images == []
    assert "[page 2: no text, picture failed]" in outcome.output


def test_a_file_given_by_its_path_is_opened_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Any] = []
    real_open = open

    def counting(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins_module, "open", counting, raising=False)
    (tmp_path / "a.txt").write_text("h\u00e9llo\r\nworld\n", encoding="utf-8")
    (tmp_path / "cv.pdf").write_bytes(make_pdf([CV]))
    assert (
        builtins_module._read_as_given(str(tmp_path / "a.txt")) == "h\u00e9llo\nworld\n"
    )
    assert isinstance(builtins_module._read_as_given(str(tmp_path / "cv.pdf")), bytes)
    assert len(opened) == 2


def test_pypdfium2_is_never_built_from_source_by_uv() -> None:
    project = tomllib.loads(
        (Path(pdf.__file__).parents[2] / "pyproject.toml").read_text()
    )
    assert "pypdfium2" in project["tool"]["uv"]["no-build-package"]


def _answering_pictures(
    monkeypatch: pytest.MonkeyPatch, *pictures: dict[str, Any]
) -> None:
    async def fake(
        mode: str, limits: dict[str, Any], data: bytes, limit: int
    ) -> dict[str, Any]:
        return {"status": "ok", "pictures": list(pictures)}

    monkeypatch.setattr(pdf, "_ask", fake)


_GOOD_PNG = _png_bytes(600, 800)


@pytest.mark.parametrize(
    "broken",
    [
        _GOOD_PNG[:-5],  # cut short
        _GOOD_PNG[:-12],  # no end chunk
        _GOOD_PNG + b"x",  # bytes after the end chunk
        _GOOD_PNG[:41] + bytes([_GOOD_PNG[41] ^ 1]) + _GOOD_PNG[42:],  # a bad CRC
        _png_bytes(600, 800, ("IDAT", "IHDR", "IEND")),  # the header is not first
        _png_bytes(600, 800, ("IHDR", "IEND")),  # no image data
    ],
    ids=["truncated", "no-end", "trailing", "bad-crc", "header-not-first", "no-data"],
)
async def test_a_picture_that_is_not_a_whole_png_leaves_its_page_failed(
    monkeypatch: pytest.MonkeyPatch, broken: bytes
) -> None:
    _answering_pictures(
        monkeypatch, {"page": 2, "png": base64.b64encode(broken).decode()}
    )
    doc = pdf.PdfText(2, ("Cover page with some words in it", ""))
    shown, note, images = await pdf.with_pictures(
        b"%PDF-", doc, doc.text, lambda page: "start_line=1"
    )
    assert images == [] and "[page 2: no text, picture failed]" in shown
    assert "attached" not in note and "no picture could be made" in note


async def test_a_whole_png_within_the_limits_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answering_pictures(monkeypatch, {"page": 2, "png": _png_of(600, 800)})
    doc = pdf.PdfText(2, ("Cover page with some words in it", ""))
    shown, _, images = await pdf.with_pictures(
        b"%PDF-", doc, doc.text, lambda page: "start_line=1"
    )
    assert len(images) == 1 and "[page 2: no text, picture attached]" in shown


# ── Watching the child's memory ──


def _fake_ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``ps`` that records its pid and then sleeps, put first on the PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "ps.pid"
    script = bin_dir / "ps"
    script.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 30\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return pid_file


async def _gone(pid_file: Path) -> bool:
    """Whether the process whose pid is in *pid_file* is gone (not a zombie), within 2 seconds."""
    for _ in range(40):
        try:
            os.kill(int(pid_file.read_text()), 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


async def test_a_look_at_the_childs_memory_has_a_deadline_and_a_cancelled_one_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = _fake_ps(tmp_path, monkeypatch)
    # A look that does not answer in time reads as "could not tell", and ends.
    monkeypatch.setattr(pdf, "PS_TIMEOUT_S", 0.3, raising=False)
    assert await asyncio.wait_for(pdf._resident(os.getpid()), 5) is None
    assert await _gone(pid_file)
    # A look that is cancelled while it waits leaves no process behind.
    pid_file.unlink()
    monkeypatch.setattr(pdf, "PS_TIMEOUT_S", 30.0, raising=False)
    look = asyncio.create_task(pdf._resident(os.getpid()))
    while not pid_file.exists():
        await asyncio.sleep(0.02)
    look.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await look
    assert await _gone(pid_file)


async def test_a_child_whose_memory_cannot_be_read_is_stopped_with_a_plain_line(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "cv.pdf").write_bytes(make_pdf([CV]))
    pid_file = tmp_path / "pid"
    sleeper = _child(
        tmp_path,
        f"""
        import os, time
        open({str(pid_file)!r}, "w").write(str(os.getpid()))
        time.sleep(4)
        """,
    )

    async def blind(pid: int) -> int | None:
        return None

    monkeypatch.setattr(pdf, "_CHILD", sleeper)
    monkeypatch.setattr(pdf, "_resident", blind)
    monkeypatch.setattr(pdf, "WATCH_S", 0.05)
    began = time.monotonic()
    outcome = await _read(root, "cv.pdf")
    assert outcome.status == ExecutionStatus.ERROR
    assert "could not check how much memory" in outcome.error
    assert "\n" not in outcome.error and time.monotonic() - began < 3
    assert await _gone(pid_file)


async def test_pdf_reads_in_one_process_run_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "log"
    child = _child(
        tmp_path,
        f"""
        import sys, time
        log = open({str(log)!r}, "a")
        log.write("start\\n")
        log.flush()
        time.sleep(0.4)
        log.write("end\\n")
        log.close()
        sys.stdout.write('{{"status": "ok", "pages": 1, "texts": ["some words on the only page"]}}')
        """,
    )
    monkeypatch.setattr(pdf, "_CHILD", child)
    await asyncio.gather(pdf.read_text(b"%PDF-"), pdf.read_text(b"%PDF-"))
    assert log.read_text().split() == ["start", "end", "start", "end"]

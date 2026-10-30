"""Pictures that tools return reach a model that can see (``tool_images``, CU-0).

No network and no model: a scripted provider records each request. The store
keeps the newest ``keep`` images and retires ``retire`` at a time; requests
change only when a tool returns an image and the switch is on.
"""

from __future__ import annotations

import base64
import shutil
import struct
import zlib
from collections import deque
from typing import Any

import pytest

import localmelo.melo.checker.checker as checker_mod
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._request_accounting import _image_dimensions
from localmelo.melo.agent._tool_images import (
    MAX_IMAGE_WIDTH,
    UNCHANGED_NOTE,
    ToolImageStore,
    shrink,
)
from localmelo.melo.checker.checker import IMAGE_PROMPT_CHARS, Checker
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor import ToolOutput
from localmelo.melo.schema import (
    ImagePart,
    Message,
    ToolCall,
    ToolDef,
    ToolResult,
    decode_base64,
)


def frame(n: int) -> ImagePart:
    """A distinct fake picture (no header: never scaled)."""
    return ImagePart.from_bytes(f"\x89PNG fake frame {n}".encode(), "image/png")


def png(width: int, height: int) -> bytes:
    """A real PNG of *width* x *height* (a gradient), standard library only."""
    rows = b"".join(
        b"\x00" + bytes((x + y) % 256 for x in range(width * 3)) for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + data))
        return struct.pack(">I", len(data)) + kind + data + crc

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class ScriptedLLM(BaseLLMProvider):
    """Replies in order, records every request, and answers ``takes_images``."""

    model = "vision-model"

    def __init__(self, replies: list[Message], takes: Any = None) -> None:
        self.replies = deque(replies)
        self.requests: list[list[Message]] = []
        self.takes = takes
        self.asked = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        if not self.replies:
            return Message(role="assistant", content="done")
        return self.replies.popleft()

    async def takes_images(self) -> bool | None:
        self.asked += 1
        if isinstance(self.takes, Exception):
            raise self.takes
        return self.takes  # type: ignore[no-any-return]

    async def close(self) -> None:
        pass


def shots(n: int, *, same: bool = False) -> list[Message]:
    """*n* steps that each call ``shot`` once, then an answer."""
    return [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(tool_name="shot", arguments={"n": i, "same": same}, id=f"c{i}")
            ],
        )
        for i in range(n)
    ] + [Message(role="assistant", content="done")]


def agent_for(llm: ScriptedLLM, *, text_only: bool = False, **options: Any) -> Agent:
    agent = Agent(
        llm=llm,
        options=AgentOptions(attempt_policy="none", tool_exposure="all", **options),
    )
    agent.hippo.register_tool(
        ToolDef(name="shot", description="Capture the screen.", parameters={})
    )

    async def shot(n: int = 0, same: bool = False) -> ToolOutput:
        if text_only:
            return ToolOutput(text=f"frame {n} as text")
        return ToolOutput(text=f"frame {n}", images=[frame(0 if same else n)])

    agent.executor.register("shot", shot)
    return agent


def sent_images(request: list[Message]) -> list[str]:
    """The frames whose pixels a request carries, oldest first."""
    return [
        base64.b64decode(image.data).decode().removeprefix("\x89PNG fake ")
        for message in request
        if message.role == "tool"
        for image in message.images
        if image.data
    ]


async def run(llm: ScriptedLLM, steps: int = 6, **options: Any) -> Agent:
    agent = agent_for(llm, **options)
    assert await agent.run("take screenshots") == "done"
    assert len(llm.requests) == steps + 1
    return agent


class TestModes:
    async def test_off_sends_no_pixels_and_never_asks(self) -> None:
        llm = ScriptedLLM(shots(2), takes=True)
        await run(llm, 2, tool_images="off")
        assert all(sent_images(r) == [] for r in llm.requests)
        assert llm.asked == 0

    async def test_on_sends_the_newest_pictures_without_asking(self) -> None:
        llm = ScriptedLLM(shots(2))
        await run(llm, 2, tool_images="on")
        assert [sent_images(r) for r in llm.requests] == [
            [],
            ["frame 0"],
            ["frame 0", "frame 1"],
        ]
        assert llm.asked == 0
        # The tool message keeps its text and the line that names the image;
        # history and working memory keep references only.
        last = [m for m in llm.requests[-1] if m.role == "tool"][-1]
        assert last.content == f"frame 1\n[1 image: {frame(1).ref}]"

    @pytest.mark.parametrize(
        ("answer", "sent"), [(True, True), (False, False), (None, False)]
    )
    async def test_auto_asks_once_and_follows_the_answer(
        self, answer: bool | None, sent: bool
    ) -> None:
        llm = ScriptedLLM(shots(3), takes=answer)
        await run(llm, 3)  # "auto" is the default
        assert llm.asked == 1
        assert bool(sent_images(llm.requests[-1])) is sent

    async def test_auto_reads_a_failed_question_as_no(self) -> None:
        llm = ScriptedLLM(shots(2), takes=RuntimeError("no route"))
        await run(llm, 2)
        assert llm.asked == 1
        assert all(sent_images(r) == [] for r in llm.requests)

    @pytest.mark.parametrize("mode", ["auto", "on", "off"])
    async def test_a_text_only_run_sends_the_same_requests_in_every_mode(
        self, mode: str
    ) -> None:
        def requests(tool_images: str) -> list[list[tuple[Any, ...]]]:
            return [
                [(m.role, m.content, m.tool_call_id, m.images) for m in r]
                for r in llms[tool_images].requests
            ]

        llms = {name: ScriptedLLM(shots(3), takes=True) for name in ("off", mode)}
        for name, llm in llms.items():
            agent = agent_for(llm, text_only=True, tool_images=name)
            assert await agent.run("take screenshots") == "done"
            assert llm.asked == 0  # no image: nothing to ask about
        assert requests(mode) == requests("off")


class TestBudget:
    async def test_keep_three_and_retire_two_at_a_time(self) -> None:
        llm = ScriptedLLM(shots(7))
        await run(llm, 7, tool_images="on")
        counts = [len(sent_images(r)) for r in llm.requests]
        # K=3, B=2: 3 to 4 pictures once warm; the 5th and 7th retire two.
        assert counts == [0, 1, 2, 3, 4, 3, 4, 3]
        assert sent_images(llm.requests[-1]) == ["frame 4", "frame 5", "frame 6"]

    async def test_the_prefix_keeps_its_pixels_between_retirements(self) -> None:
        llm = ScriptedLLM(shots(4))
        await run(llm, 4, tool_images="on")
        third, fourth = llm.requests[3], llm.requests[4]
        # Request 4 extends request 3: every message of 3 is sent again as it was.
        assert [(m.content, m.images) for m in fourth[: len(third)]] == [
            (m.content, m.images) for m in third
        ]

    async def test_keep_and_retire_are_settings(self) -> None:
        llm = ScriptedLLM(shots(4))
        await run(llm, 4, tool_images="on", tool_images_keep=1, tool_images_retire=1)
        assert [len(sent_images(r)) for r in llm.requests] == [0, 1, 1, 1, 1]

    async def test_a_repeated_picture_is_sent_once_and_named_as_unchanged(
        self,
    ) -> None:
        llm = ScriptedLLM(shots(3, same=True))
        await run(llm, 3, tool_images="on")
        assert sent_images(llm.requests[-1]) == ["frame 0"]
        tools = [m for m in llm.requests[-1] if m.role == "tool"]
        assert [m.content.endswith(UNCHANGED_NOTE) for m in tools] == [
            False,
            True,
            True,
        ]

    async def test_the_prompt_cap_counts_each_picture_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        original = Checker.pre_plan

        async def spy(self: Checker, messages: list[Message], **kwargs: Any) -> Any:
            seen.append(sum(1 for m in messages for i in m.images if i.data))
            return await original(self, messages, **kwargs)

        monkeypatch.setattr(Checker, "pre_plan", spy)
        await run(ScriptedLLM(shots(2)), 2, tool_images="on")
        assert max(seen) == 2

    @pytest.mark.parametrize("on", [True, False])
    async def test_pictures_push_old_turns_out_of_the_prompt_cap(
        self, on: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 30,000 characters of text fit beside the references; beside four
        # pictures at 20,000 characters each they do not.
        monkeypatch.setattr(checker_mod, "IMAGE_PROMPT_CHARS", 20_000)
        llm = ScriptedLLM(shots(4))
        agent = agent_for(llm, tool_images="on" if on else "off")
        agent.hippo.working.append(Message(role="user", content="x" * 30_000))
        assert await agent.run("take screenshots") == "done"
        last = llm.requests[-1]
        assert any("x" * 30_000 in m.content for m in last) is not on
        assert sent_images(last)[-1:] == (["frame 3"] if on else [])


class TestChecker:
    async def test_a_picture_counts_and_a_reference_does_not(self) -> None:
        checker = Checker()
        image = frame(1)
        text = Message(role="tool", content="x" * 90_000, images=[image])
        assert (await checker.pre_plan([text])).allowed
        two = Message(role="user", content="", images=[image, image])
        result = await checker.pre_plan([text, two])
        assert not result.allowed
        assert f"{90_000 + 3 * IMAGE_PROMPT_CHARS} chars" in result.reason
        refs = Message(role="user", content="", images=[image.as_reference()] * 9)
        assert (await checker.pre_plan([text, refs])).allowed


def tool_message(image: ImagePart, call: str) -> Message:
    return Message(
        role="tool", content=call, tool_call_id=call, images=[image.as_reference()]
    )


class TestStore:
    async def test_hydrate_is_the_same_list_when_nothing_is_live(self) -> None:
        store = ToolImageStore()
        messages = [tool_message(frame(1), "a")]
        assert store.hydrate(messages) is messages
        store.keep(messages)  # nothing prepared: nothing goes live
        assert store.hydrate(messages) is messages

    async def test_payloads_live_only_as_long_as_their_picture(self) -> None:
        store = ToolImageStore(keep=1, retire=1)
        for n in range(3):
            image = frame(n)
            await store.prepare([ToolResult(tool_name="shot", images=[image])])
            store.keep([tool_message(image, f"c{n}")])
        assert [i.sha256 for i in store.live] == [frame(2).sha256]
        assert list(store._payloads) == [frame(2).sha256]

    async def test_a_withheld_result_keeps_no_payload(self) -> None:
        store = ToolImageStore()
        await store.prepare([ToolResult(tool_name="shot", images=[frame(1)])])
        store.keep([Message(role="tool", content="[withheld]", tool_call_id="c")])
        assert store.live == [] and store._payloads == {}


requires_sips = pytest.mark.skipif(
    shutil.which("sips") is None, reason="scaling uses macOS sips"
)


class TestShrink:
    def test_a_picture_within_the_width_is_kept_as_it_is(self) -> None:
        image = ImagePart.from_bytes(png(640, 4), "image/png")
        assert shrink(image) is image
        assert shrink(frame(1)) == frame(1)  # no header it reads: kept

    @requires_sips
    def test_a_wider_picture_becomes_a_jpeg_1280_wide(self) -> None:
        image = ImagePart.from_bytes(png(2560, 1600), "image/png")
        scaled = shrink(image)
        assert scaled.mime_type == "image/jpeg"
        assert _image_dimensions(decode_base64(scaled.data) or b"") == (
            MAX_IMAGE_WIDTH,
            800,
        )

    @requires_sips
    async def test_the_store_scales_once_when_it_stores(self) -> None:
        wide = ImagePart.from_bytes(png(2000, 10), "image/png")
        store = ToolImageStore()
        await store.prepare([ToolResult(tool_name="shot", images=[wide])])
        store.keep([tool_message(wide, "c")])
        [sent] = store.hydrate([tool_message(wide, "c")])  # a new reference
        assert sent.images[0].data == ""  # only the reference that went live
        [message] = [tool_message(wide, "c")]
        store.keep([message])  # the same digest: a repeat, not a new payload
        [sent] = store.hydrate([message])
        assert sent.content.endswith(UNCHANGED_NOTE)
        payload = store._payloads[wide.sha256]
        assert payload.mime_type == "image/jpeg"
        assert base64.b64decode(payload.data)[:2] == b"\xff\xd8"

"""Pictures that come with words the person adds while a run works.

No network and no model: a scripted provider records each request. An inbox that
can name the pictures of a note (``pictures(offset)``) makes the run's next
request carry them as image parts of the person's message; an inbox that cannot
behaves as it always did.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._tool_images import USER_KEEP
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolDef


def picture(n: int) -> ImagePart:
    return ImagePart.from_bytes(f"\x89PNG note picture {n}".encode(), "image/png")


ECHO = ToolDef(
    name="echo",
    description="Echo text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)


def step(call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "x"}, id=call_id)],
    )


def done() -> Message:
    return Message(role="assistant", content="done")


class ScriptedLLM(BaseLLMProvider):
    """Plays its replies in order and records every request."""

    model = "vision-model"

    def __init__(self, *replies: Message) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        return self.replies.popleft()

    async def takes_images(self) -> bool | None:
        return None

    async def close(self) -> None:
        pass


class Inbox:
    """Notes that appear once the run has made *after* requests; some have pictures."""

    def __init__(
        self,
        llm: ScriptedLLM,
        notes: dict[int, str],
        pictures: dict[int, list[ImagePart]] | None = None,
        *,
        after: int = 1,
    ) -> None:
        self.llm = llm
        self.notes = notes
        self.held = pictures
        self.after = after
        self.accepted: list[int] = []

    def validate(self, workspace: str | None = None) -> None:
        return None

    def poll(self) -> list[tuple[int, str]]:
        if len(self.llm.requests) < self.after:
            return []
        return [(n, text) for n, text in self.notes.items() if n not in self.accepted]

    def accept(self, offset: int) -> None:
        self.accepted.append(offset)

    def pictures(self, offset: int) -> Sequence[ImagePart]:
        return (self.held or {}).get(offset, [])


class PlainInbox(Inbox):
    """An inbox written before notes could carry pictures: no ``pictures`` method."""

    pictures = None  # type: ignore[assignment]


def agent_for(llm: ScriptedLLM) -> Agent:
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            attempt_policy="none",
            builtins=frozenset(),
            tool_exposure="all",
            record_answers=True,
        ),
    )

    async def echo(text: str = "") -> str:
        return text

    agent.hippo.register_tool(ECHO)
    agent.executor.register("echo", echo)
    return agent


def carried(request: list[Message]) -> list[ImagePart]:
    return [i for m in request if m.role == "user" for i in m.images if i.data]


class Seen:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)


async def test_the_next_request_carries_the_pictures_of_a_note_beside_its_words() -> (
    None
):
    llm = ScriptedLLM(step("c1"), done())
    sent = [picture(1), picture(2)]
    inbox = Inbox(llm, {5: "use these two"}, {5: sent})
    result = await agent_for(llm).run_task("start", inbox=inbox)
    assert result.status == "completed"
    # Not the first request: the note arrived after it.
    assert carried(llm.requests[0]) == []
    second = llm.requests[1]
    assert [i.data for i in carried(second)] == [i.data for i in sent]
    last = [m for m in second if m.role == "user"][-1]
    assert last.content.endswith("use these two")
    assert [i.data for i in last.images] == [i.data for i in sent]
    assert inbox.accepted == [5]


async def test_the_accepted_event_names_the_note_and_holds_no_pixels() -> None:
    llm = ScriptedLLM(step("c1"), done())
    inbox = Inbox(llm, {5: "look"}, {5: [picture(3)]})
    seen = Seen()
    await agent_for(llm).run_task("start", inbox=inbox, on_event=seen)
    # The picture travels in the request, so only the event could leak its pixels.
    assert len(carried(llm.requests[1])) == 1
    accepted = [e for e in seen.events if e.type == "inbox.accepted"]
    assert [e.data["offset"] for e in accepted] == [5]
    assert "iVBOR" not in str(accepted[0].data)
    assert all(not e.images for e in accepted)


async def test_working_memory_keeps_references_only_and_a_later_run_still_sends_them() -> (
    None
):
    llm = ScriptedLLM(step("c1"), done(), done())
    sent = picture(4)
    inbox = Inbox(llm, {5: "look"}, {5: [sent]})
    agent = agent_for(llm)
    await agent.run_task("start", inbox=inbox)
    kept = [
        i
        for m in agent.hippo.working.get_window()
        if m.role == "user"
        for i in m.images
    ]
    assert [i.data for i in kept] == [""]
    assert kept[0].sha256 == sent.sha256
    await agent.run_task("what colour was it?")
    assert [i.data for i in carried(llm.requests[-1])] == [sent.data]


async def test_an_inbox_without_pictures_sends_the_words_as_before() -> None:
    llm = ScriptedLLM(step("c1"), done())
    inbox = PlainInbox(llm, {5: "just words"})
    result = await agent_for(llm).run_task("start", inbox=inbox)
    assert result.status == "completed"
    assert all(carried(request) == [] for request in llm.requests)
    last = [m for m in llm.requests[1] if m.role == "user"][-1]
    assert last.content.endswith("just words") and last.images == []
    assert inbox.accepted == [5]


async def test_a_note_with_no_pictures_adds_none_beside_one_that_has() -> None:
    llm = ScriptedLLM(step("c1"), done())
    sent = picture(5)
    inbox = Inbox(llm, {5: "first", 6: "second"}, {6: [sent]})
    await agent_for(llm).run_task("start", inbox=inbox)
    assert [i.data for i in carried(llm.requests[1])] == [sent.data]
    assert inbox.accepted == [5, 6]


async def test_notes_whose_pictures_pass_what_one_step_keeps_are_read_on_the_next_step() -> (
    None
):
    """A step reads whole notes while their pictures fit; the rest wait, unread, for the next one."""
    llm = ScriptedLLM(step("c1"), step("c2"), done())
    per_note = USER_KEEP // 2
    pictures = {n: [picture(n * 10 + k) for k in range(per_note)] for n in (1, 2, 3)}
    inbox = Inbox(llm, {1: "a", 2: "b", 3: "c"}, pictures)
    result = await agent_for(llm).run_task("start", inbox=inbox)
    assert result.status == "completed"
    # Two notes fill the step; the third is read on the step after, not lost.
    assert len(carried(llm.requests[1])) == 2 * per_note
    assert inbox.accepted == [1, 2, 3]
    third = [m for m in llm.requests[2] if m.role == "user"][-1]
    assert third.content.endswith("c")
    assert [i.data for i in third.images] == [i.data for i in pictures[3]]


async def test_a_picture_the_checker_would_refuse_is_left_out_not_sent() -> None:
    llm = ScriptedLLM(step("c1"), done())
    odd = ImagePart.from_bytes(b"gif", "image/tiff")
    good = picture(6)
    inbox = Inbox(llm, {5: "look"}, {5: [odd, good]})
    result = await agent_for(llm).run_task("start", inbox=inbox)
    assert result.status == "completed"
    assert [i.data for i in carried(llm.requests[1])] == [good.data]

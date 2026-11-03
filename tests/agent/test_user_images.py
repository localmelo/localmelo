"""Pictures the person adds to a message reach a model that can see.

No network and no model: a scripted provider records each request. The query
message keeps reference copies, the store keeps the payloads of the newest few,
and a picture whose payload is gone is named in the message instead.
"""

from __future__ import annotations

from collections import deque

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._tool_images import USER_KEEP, ToolImageStore
from localmelo.melo.checker.validators import MAX_IMAGES_PER_RESULT
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.schema import PICTURES_REFUSED_ERROR, ImagePart, Message, ToolDef


def picture(n: int) -> ImagePart:
    return ImagePart.from_bytes(f"\x89PNG fake picture {n}".encode(), "image/png")


class ScriptedLLM(BaseLLMProvider):
    """Answers "done" and records every request."""

    model = "vision-model"

    def __init__(self) -> None:
        self.replies: deque[Message] = deque()
        self.requests: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        return Message(role="assistant", content="done")

    async def takes_images(self) -> bool | None:
        return None

    async def close(self) -> None:
        pass


def agent_for(llm: ScriptedLLM) -> Agent:
    return Agent(
        llm=llm, options=AgentOptions(attempt_policy="none", record_answers=True)
    )


def users(request: list[Message]) -> list[Message]:
    return [m for m in request if m.role == "user" and m.images]


async def test_the_planning_request_carries_the_pictures_of_the_message() -> None:
    llm = ScriptedLLM()
    agent = agent_for(llm)
    sent = [picture(1), picture(2)]
    result = await agent.run_task("what is in these?", images=sent)
    assert result.status == "completed"
    carried = users(llm.requests[0])
    assert len(carried) == 1
    assert carried[0].content.startswith("what is in these?")
    assert [i.data for i in carried[0].images] == [i.data for i in sent]


async def test_working_memory_keeps_references_only() -> None:
    llm = ScriptedLLM()
    agent = agent_for(llm)
    sent = picture(3)
    await agent.run_task("look", images=[sent])
    kept = [m for m in agent.hippo.working.get_window() if m.role == "user"]
    assert [i.data for m in kept for i in m.images] == [""]
    assert kept[0].images[0].sha256 == sent.sha256


async def test_a_later_turn_still_sends_an_earlier_picture() -> None:
    llm = ScriptedLLM()
    agent = agent_for(llm)
    first = picture(4)
    await agent.run_task("first", images=[first])
    await agent.run_task("and again, what colour was it?")
    request = llm.requests[-1]
    assert [i.data for m in users(request) for i in m.images] == [first.data]


async def test_a_picture_past_the_newest_few_is_named_not_sent() -> None:
    llm = ScriptedLLM()
    agent = agent_for(llm)
    old = picture(0)
    await agent.run_task("old", images=[old])
    for n in range(1, USER_KEEP + 1):
        await agent.run_task(f"more {n}", images=[picture(n)])
    request = llm.requests[-1]
    first = next(m for m in request if m.role == "user" and "old" in m.content)
    assert all(not i.data for i in first.images)
    assert "no longer attached" in first.content and old.ref in first.content
    newest = [i.data for m in users(request) for i in m.images if i.data]
    assert len(newest) == USER_KEEP


async def test_a_picture_the_checker_would_refuse_stops_the_run_before_it_starts() -> (
    None
):
    llm = ScriptedLLM()
    agent = agent_for(llm)
    with pytest.raises(ValueError, match="picture refused"):
        await agent.run_task("x", images=[ImagePart.from_bytes(b"gif", "image/tiff")])
    with pytest.raises(ValueError, match="at most"):
        await agent.run_task(
            "x", images=[picture(n) for n in range(MAX_IMAGES_PER_RESULT + 1)]
        )
    assert llm.requests == []


def test_keep_user_returns_references_and_keeps_the_newest() -> None:
    store = ToolImageStore()
    refs = store.keep_user([picture(n) for n in range(USER_KEEP + 2)])
    assert all(not r.data and r.sha256 for r in refs)
    message = Message(role="user", content="q", images=refs)
    hydrated = store.hydrate([message])[0]
    # The two oldest lost their pixels; the rest have them.
    assert [bool(i.data) for i in hydrated.images] == [False, False] + [
        True
    ] * USER_KEEP


# ── A server that refuses the pictures ──


class RefusingLLM(ScriptedLLM):
    """Fails every request with *error*; the model says whether it reads pictures."""

    def __init__(self, error: ProviderError, reads: bool | None) -> None:
        super().__init__()
        self.error = error
        self.reads = reads

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        raise self.error

    async def takes_images(self) -> bool | None:
        return self.reads


BARE_400 = "POST /chat/completions failed with HTTP 400"


@pytest.mark.parametrize(
    ("message", "status", "reads", "sent", "refused"),
    [
        # A bare 400 to a request with pictures, from a model not known to read them.
        (BARE_400, 400, None, True, True),
        (BARE_400, 415, False, True, True),
        # The error names the pictures: whatever the model says it reads.
        (BARE_400 + ": image input is not supported", 400, True, True, True),
        ("model has no vision support", None, True, True, True),
        # Not the pictures: the model says it reads them, or the status is not a refusal.
        (BARE_400, 400, True, True, False),
        ("POST /chat/completions failed with HTTP 500", 500, None, True, False),
        # A request with no pictures never counts, whatever the error says.
        (BARE_400, 400, None, False, False),
        (BARE_400 + ": image input is not supported", 400, None, False, False),
    ],
)
async def test_a_request_refused_for_its_pictures_ends_with_one_plain_line(
    message: str, status: int | None, reads: bool | None, sent: bool, refused: bool
) -> None:
    llm = RefusingLLM(ProviderError(message, status_code=status), reads)
    agent = agent_for(llm)
    result = await agent.run_task("what is this?", images=[picture(7)] if sent else [])
    assert result.status == "provider_error"
    assert result.error == (PICTURES_REFUSED_ERROR if refused else message)


async def test_a_refusal_for_context_size_is_not_a_refusal_of_pictures() -> None:
    error = ProviderError(BARE_400, status_code=400, context_overflow=True)
    result = await agent_for(RefusingLLM(error, None)).run_task(
        "what is this?", images=[picture(8)]
    )
    assert result.status == "provider_error"
    assert result.error != PICTURES_REFUSED_ERROR

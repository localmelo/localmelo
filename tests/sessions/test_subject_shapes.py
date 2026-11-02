"""The person's message shapes keep the chat's subject (#452), on the subject stub.

The main chat talks about a friend (Alex); a new side chat then asks, in the
mixed Chinese and English the person types, to "read his case studies and
résumé first", corrects it, lists roles for "她 … 他" (the pronouns sound alike
and are typed interchangeably), and says "去查一下" (go look it up), whose tool
steps fill the window and end on the user's own profile (Sam). The stub answers
about whoever the context names: the person named first in the last user
message, else the person named last in the request. Every turn's answer is
about Alex. :func:`shapes` is also the stub half of the measurement in the
change's report: before the change, the first message and "去查一下" answered
about Sam.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.schema import Message
from localmelo.support.memory import ChatMemory
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.agent.test_earlier_turns import SubjectModel, _agent, _call

OWNER = "The user is Sam Rivera, a robotics student."
MAIN = [
    (
        "看一下這個 example.com 的人的背景是什麼",
        "這個人是 Alex Kim，一名 UX 研究員，做支付產品的用戶研究。",
    ),
    (
        "你覺得Alex還有什麼職業出路 besides ux research",
        "Alex 的組合比純 UXR 寬：產品經理、服務設計、研究運營都適合。",
    ),
]
# The side chat's messages, as the person typed them (generic names).
SHAPES = (
    "你應該分析完他的case study and resume再給我答案 還有未來5年適合他的新型行業",
    "是分析Alex的",
    "把你剛剛講的適配她的新型行業的role 都list 出來 所以他可以開始去找and 投工作",
    "去查一下",
)


async def shapes(tmp_path: Path, store: SessionStore, memory: ChatMemory) -> list[str]:
    """The side chat's answers to :data:`SHAPES`, after the main chat's turns."""
    model = SubjectModel()
    main_agent = _agent(
        model, workspace_root=str(_folder(tmp_path / "main")), identity=OWNER
    )
    main = SessionRunner(
        main_agent, store, None, new_session=NewSession(kind="front"), memory=memory
    )
    try:
        for words, answer in MAIN:
            model.plans.append(Message(role="assistant", content=answer))
            await main.run_turn(words, title=words)
        await memory.drain()
    finally:
        await main.close()
        await main_agent.close()
    model = SubjectModel()
    side_agent = _agent(
        model, workspace_root=str(_folder(tmp_path / "side")), identity=OWNER
    )
    side = SessionRunner(
        side_agent, store, None, new_session=NewSession(kind="chat"), memory=memory
    )
    answers: list[str] = []
    try:
        for words in SHAPES:
            if words == "去查一下":
                model.plans.extend(
                    [
                        _call("look_up", part="boards"),
                        _call("look_up", part="more boards"),
                        _call("read_profile", part="education"),
                        _call("read_profile", part="work"),
                        _call("read_profile", part="projects"),
                    ]
                )
            result = await side.run_turn(words, title=words)
            answers.append(result.result)
    finally:
        await side.close()
        await side_agent.close()
    return answers


def _folder(path: Path) -> Path:
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def test_every_shape_stays_on_the_friend(
    tmp_path: Path, store: SessionStore, memory: ChatMemory
) -> None:
    answers = await shapes(tmp_path, store, memory)
    assert answers == ["Roles that fit Alex."] * len(SHAPES)


def test_the_shapes_name_nobody_but_the_correction() -> None:
    # Only the correction names the friend; the rest lean on the chat.
    named: list[Any] = [shape for shape in SHAPES if "Alex" in shape]
    assert named == ["是分析Alex的"]

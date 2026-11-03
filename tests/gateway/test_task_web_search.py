"""A task's worker gets web_search when ``[web] search`` is on, and not otherwise."""

from __future__ import annotations

from localmelo.tests.gateway import test_task_folders as folders
from localmelo.tests.gateway.test_workers import eventually

Setup = folders.Setup
# The workers fixture of the task-folder tests, under its own name here.
setup = folders.setup


async def _tools(setup: Setup, name: str) -> list[str]:
    workers, _, _, built, home = setup
    project = home / name
    project.mkdir()
    await workers.create({"objective": "look it up", "workspace": str(project)})
    await eventually(lambda: len(built) == 1)
    names: list[str] = []
    for provider in built[0]["tools"]:
        names += [tool.definition.name for tool in await provider.start()]
    return names


async def test_a_worker_has_web_search_when_the_setting_is_on(setup: Setup) -> None:
    setup[1].web.search = True
    names = await _tools(setup, "on")
    assert "web_search" in names and "ask_owner" in names


async def test_a_worker_has_no_web_search_by_default(setup: Setup) -> None:
    assert setup[1].web.search is False
    assert "web_search" not in await _tools(setup, "off")


async def test_the_workers_search_is_the_task_kind_that_asks_its_chat(
    setup: Setup,
) -> None:
    setup[1].web.search = True
    workers, _, _, built, home = setup
    project = home / "kind"
    project.mkdir()
    await workers.create({"objective": "look it up", "workspace": str(project)})
    await eventually(lambda: len(built) == 1)
    [search] = [p for p in built[0]["tools"] if type(p).__name__ == "WebSearchProvider"]
    assert search.task is True

"""A task's worker gets web_search when ``[web] search`` is on, and not otherwise."""

from __future__ import annotations

from typing import Any

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.web.search_tool import WebSearchProvider
from localmelo.tests.gateway import test_task_folders as folders
from localmelo.tests.gateway.test_workers import eventually

Setup = folders.Setup
# The workers fixture of the task-folder tests, under its own name here.
setup = folders.setup


async def _built(setup: Setup, name: str) -> dict[str, Any]:
    workers, _, _, built, home = setup
    project = home / name
    project.mkdir()
    await workers.create({"objective": "look it up", "workspace": str(project)})
    await eventually(lambda: len(built) == 1)
    return built[0]


async def test_a_worker_has_web_search_when_the_setting_is_on(setup: Setup) -> None:
    setup[1].web.search = True
    assert (await _built(setup, "on"))["web_search"] is True


async def test_a_worker_has_no_web_search_by_default(setup: Setup) -> None:
    assert setup[1].web.search is False
    assert (await _built(setup, "off"))["web_search"] is False


async def test_a_worker_has_one_search_provider_that_cannot_send_unasked_words(
    setup: Setup,
) -> None:
    setup[1].web.search = True
    search = WebSearchProvider(cfg=(await _built(setup, "kind"))["cfg"])
    [tool] = await search.start()
    # A worker's call runs with no approval policy unless its chat is asked: a
    # search that needs asking is refused where nobody can be asked.
    result = await tool.function(
        query="why does sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789 fail"
    )
    assert isinstance(result, ToolOutput)
    assert result.is_error and "nobody to ask" in result.text

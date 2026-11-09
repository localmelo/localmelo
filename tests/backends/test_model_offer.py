"""What a server offers for chat, and whether the configured model is on it."""

from __future__ import annotations

import pytest

from localmelo.support.backends.model_offer import check_chat_model, offer

LISTING = ["qwen3-8b", "Qwen3-Embedding-0.6B", "bge-m3", "bge-reranker-v2", "gemma-3"]


def test_the_chat_models_are_listed_apart_from_embedding_and_reranking() -> None:
    found = offer(LISTING)
    assert found.chat == ("qwen3-8b", "gemma-3")
    assert found.embedding == ("Qwen3-Embedding-0.6B", "bge-m3")


def test_a_model_on_the_server_needs_nothing() -> None:
    check = check_chat_model("gemma-3", LISTING)
    assert (check.state, check.use, check.line()) == ("ok", "", "")


@pytest.mark.parametrize(
    ("configured", "models"),
    [("", ["a", "b"]), ("old", []), ("old", ["only-embedding-embed"])],
    ids=["no model named", "nothing listed", "no chat model listed"],
)
def test_nothing_is_claimed_when_there_is_nothing_to_compare(
    configured: str, models: list[str]
) -> None:
    assert check_chat_model(configured, models).state == "ok"


def test_the_only_chat_model_replaces_one_that_is_gone() -> None:
    check = check_chat_model("old", ["new", "embed-small", "bge-m3"])
    assert (check.state, check.use) == ("switch", "new")
    assert check.line() == (
        "“old” is not on the server, so melo uses “new”, "
        "the only chat model it offers."
    )


def test_several_chat_models_ask_instead_of_guessing() -> None:
    check = check_chat_model("old", ["a", "b", "embed"])
    assert (check.state, check.use, check.chat) == ("missing", "", ("a", "b"))
    assert check.line() == (
        "“old” is not on the server. Pick one of the 2 chat models it offers "
        "from the list."
    )


def test_the_line_reads_in_the_language_asked_for() -> None:
    line = check_chat_model("old", ["new"]).line("zh-Hans")
    assert "old" in line
    assert "new" in line
    assert line != check_chat_model("old", ["new"]).line("en")

"""``[memory]`` recall settings: developer settings with their shipped defaults.

``recall_form``, ``recall_list_max``, ``recall_open_chars``, ``recall_source``,
``compress_parallel`` and ``compress_cache`` are checked against their bounds and values, set for one run
with ``-X``, and listed by the developer help only, each with the issue that decides
it.
"""

from __future__ import annotations

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.config import Config, MemoryConfig

CHANGED = {
    "recall_form": "list",
    "recall_list_max": 35,
    "recall_open_chars": 9000,
    "recall_source": "items",
    "compress_parallel": 4,
    "compress_cache": "cache/compress.jsonl",
}
# The issue that decides each setting that is still a comparison, as the developer
# help names it; ``compress_parallel`` and ``compress_cache`` are ordinary advanced
# settings and carry none.
ISSUE = {name: "#428" for name in CHANGED if not name.startswith("compress_")}


def errors(**memory: object) -> list[str]:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return [error for error in cfg.validate() if "[memory]" in error]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("recall_form", "excerpt"),
        ("recall_form", "list"),
        ("recall_list_max", 1),
        ("recall_list_max", 50),
        ("recall_list_max", 20),
        ("recall_open_chars", 500),
        ("recall_open_chars", 16000),
        ("recall_open_chars", 4000),
        ("recall_source", "turns"),
        ("recall_source", "items"),
        ("recall_source", "auto"),
        ("compress_parallel", 1),
        ("compress_parallel", 8),
        ("compress_parallel", 4),
        ("compress_cache", ""),
        ("compress_cache", "~/runs/compress_cache.jsonl"),
    ],
)
def test_the_bounds_and_values_are_accepted(name: str, value: object) -> None:
    assert errors(**{name: value}) == []


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "recall_form",
            "table",
            "[memory] recall_form 'table' is not recognised. "
            "Must be one of ('excerpt', 'list').",
        ),
        (
            "recall_list_max",
            0,
            "[memory] recall_list_max must be from 1 to 50, got 0",
        ),
        (
            "recall_list_max",
            51,
            "[memory] recall_list_max must be from 1 to 50, got 51",
        ),
        (
            "recall_list_max",
            2.5,
            "[memory] recall_list_max must be a whole number, got 2.5",
        ),
        (
            "recall_open_chars",
            499,
            "[memory] recall_open_chars must be from 500 to 16000, got 499",
        ),
        (
            "recall_open_chars",
            16001,
            "[memory] recall_open_chars must be from 500 to 16000, got 16001",
        ),
        (
            "recall_source",
            "both",
            "[memory] recall_source 'both' is not recognised. "
            "Must be one of ('turns', 'items', 'auto').",
        ),
        (
            "compress_parallel",
            0,
            "[memory] compress_parallel must be from 1 to 8, got 0",
        ),
        (
            "compress_parallel",
            9,
            "[memory] compress_parallel must be from 1 to 8, got 9",
        ),
        (
            "compress_cache",
            "a\x00b",
            "[memory] compress_cache must be a path string without NUL",
        ),
    ],
)
def test_a_value_out_of_range_is_a_config_error_that_says_why(
    name: str, value: object, message: str
) -> None:
    assert errors(**{name: value}) == [message]


def test_x_sets_each_for_one_run_and_only_the_developer_help_lists_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv: list[str] = []
    for name, value in CHANGED.items():
        argv += ["-X", f"memory.{name}={value}"]
    cli._parse([*argv, "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert config.overrides() == {f"memory.{k}": v for k, v in CHANGED.items()}
    assert config.load().memory == MemoryConfig(**CHANGED)  # type: ignore[arg-type]
    assert config_cli.main(["keys", "recall_form", "-X", argv[1]]) == 0
    assert "memory.recall_form" in capsys.readouterr().out
    developer = cli._build_parser(developer=True).format_help().splitlines()
    plain = cli._build_parser(developer=False).format_help()
    for name in CHANGED:
        key = f"memory.{name}"
        assert any(
            line.startswith(f"  {key} ")
            and (line.endswith(f"({ISSUE[name]})") if name in ISSUE else True)
            for line in developer
        ), key
        if name not in ISSUE:
            (line,) = (line for line in developer if line.startswith(f"  {key} "))
            assert not line.endswith(")"), key
        assert key not in plain


@pytest.mark.parametrize(
    "setting",
    [
        "memory.recall_form=table",
        "memory.recall_list_max=0",
        "memory.recall_list_max=51",
        "memory.recall_list_max=many",
        "memory.recall_open_chars=499",
        "memory.recall_open_chars=16001",
        "memory.recall_open_chars=long",
        "memory.recall_source=both",
        "memory.compress_parallel=0",
        "memory.compress_parallel=9",
        "memory.compress_parallel=many",
    ],
)
def test_x_refuses_a_bad_value_with_a_clear_message(
    setting: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli._parse(["-X", setting, "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert exit_.value.code == 2
    assert setting.partition("=")[0] in capsys.readouterr().err

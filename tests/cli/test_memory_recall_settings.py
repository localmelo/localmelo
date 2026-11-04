"""``[memory]`` recall settings: developer settings with their shipped defaults.

``recall_records``, ``recall_docs``, ``recall_chars``, ``recall_notes``,
``search_when_empty``, ``recall_min_cosine``, ``recall_min_coverage``,
``recall_item_chars``, ``recall_excerpt``, ``recall_form``, ``recall_list_max``,
``recall_open_chars``, ``excerpt_open``, ``recall_source``, ``open_sources``,
``compress_at_chars``, ``compress_parallel``, ``compress_cache``, ``note_gate``, ``note_min_cosine``,
``followup_query``, ``gather`` (``guide`` by default) and ``gather_min_cosine`` are saved only when they differ from the default, checked
against their bounds and values, set for one run with ``-X``, and listed by the
developer help only, each with the issue that decides it.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli
from localmelo.support.config import RECALL_KEYS, Config, MemoryConfig, save

CHANGED = {
    "recall_records": 8,
    "recall_docs": 0,
    "recall_chars": 2500,
    "recall_notes": "off",
    "search_when_empty": "hint",
    "recall_min_cosine": 0.45,
    "recall_min_coverage": 0.4,
    "recall_item_chars": 800,
    "recall_excerpt": "match",
    "recall_form": "list",
    "recall_list_max": 35,
    "recall_open_chars": 9000,
    "excerpt_open": "off",  # "on" is the default
    "recall_source": "items",
    "open_sources": "on",
    "compress_at_chars": 20000,
    "compress_parallel": 4,
    "compress_cache": "cache/compress.jsonl",
    "note_gate": "model",
    "note_min_cosine": 0.65,
    "followup_query": "message",
    "gather": "off",  # "guide" is the default
    "gather_min_cosine": 0.3,
}
# The issue that decides each setting, as the developer help names it.
ISSUE = {name: "#349" for name in CHANGED} | {
    "recall_form": "#428",
    "recall_list_max": "#428",
    "recall_open_chars": "#428",
    "excerpt_open": "#428",
    "recall_source": "#428",
    "open_sources": "#428",
    "compress_at_chars": "#428",
    "compress_parallel": "#428",
    "compress_cache": "#428",
    "note_gate": "#442",
    "note_min_cosine": "#442",
    "followup_query": "#452",
    "gather": "#428",
    "gather_min_cosine": "#428",
}


def errors(**memory: object) -> list[str]:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return [error for error in cfg.validate() if "[memory]" in error]


def test_they_are_saved_only_when_changed_and_read_back(tmp_path: Path) -> None:
    assert set(RECALL_KEYS) == set(CHANGED)
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        assert (
            "recall_" not in path.read_text() and "search_when" not in path.read_text()
        )
        assert "followup_query" not in path.read_text()
        assert "gather" not in path.read_text()
        cfg.memory = MemoryConfig(  # the defaults
            recall_records=3,
            recall_notes="on",
            recall_min_cosine=0.55,
            followup_query="recent",
            excerpt_open="on",
            gather="guide",
            gather_min_cosine=0.40,
        )
        save(cfg)
        assert "[memory]" not in path.read_text()
        assert "excerpt_open" not in path.read_text()
        cfg.memory = MemoryConfig(**CHANGED)  # type: ignore[arg-type]
        save(cfg)
        text = path.read_text()
        assert (
            "[memory]\nrecall_records = 8\nrecall_docs = 0\nrecall_chars = 2500\n"
            'recall_notes = "off"\nsearch_when_empty = "hint"\n'
            "recall_min_cosine = 0.45\nrecall_min_coverage = 0.4\n"
            'recall_item_chars = 800\nrecall_excerpt = "match"\n'
            'recall_form = "list"\nrecall_list_max = 35\nrecall_open_chars = 9000\n'
            'excerpt_open = "off"\nrecall_source = "items"\nopen_sources = "on"\n'
            'compress_at_chars = 20000\ncompress_parallel = 4\ncompress_cache = "cache/compress.jsonl"\n'
            'note_gate = "model"\nnote_min_cosine = 0.65\n'
            'followup_query = "message"\n'
            'gather = "off"\ngather_min_cosine = 0.3\n'
        ) in text
        assert config.load().memory == MemoryConfig(**CHANGED)  # type: ignore[arg-type]
        for name, value in CHANGED.items():  # each alone
            cfg.memory = MemoryConfig(**{name: value})  # type: ignore[arg-type]
            save(cfg)
            keys = [line.split(" ")[0] for line in path.read_text().splitlines()]
            assert [key for key in keys if key in CHANGED] == [name]
            assert config.load().memory == cfg.memory


def test_excerpt_open_is_on_by_default_and_saved_only_when_off(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    assert MemoryConfig().excerpt_open == "on" and cfg.memory.excerpt_open == "on"
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        # The default is not written, and a config without the key reads as on.
        save(cfg)
        assert "excerpt_open" not in path.read_text()
        assert config.load().memory.excerpt_open == "on"
        cfg.memory = MemoryConfig(excerpt_open="on")
        save(cfg)
        assert "excerpt_open" not in path.read_text() and "[memory]" not in (
            path.read_text()
        )
        # Off is written, read back, and written again the same.
        cfg.memory = MemoryConfig(excerpt_open="off")
        save(cfg)
        text = path.read_text()
        assert '[memory]\nexcerpt_open = "off"\n' in text
        loaded = config.load()
        assert loaded.memory.excerpt_open == "off"
        save(loaded)
        assert path.read_text() == text
        # A file that still says "on" is the default, and saves without the key.
        path.write_text('chat_backend = "vllm"\n[memory]\nexcerpt_open = "on"\n')
        loaded = config.load()
        assert loaded.memory.excerpt_open == "on"
        save(loaded)
        assert "excerpt_open" not in path.read_text()


def test_gather_is_guide_by_default_and_saved_only_when_it_is_not(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    assert MemoryConfig().gather == "guide" and cfg.memory.gather == "guide"
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        # The default is not written, and a config without the key reads as guide.
        save(cfg)
        assert "gather" not in path.read_text()
        assert config.load().memory.gather == "guide"
        cfg.memory = MemoryConfig(gather="guide")
        save(cfg)
        assert "gather" not in path.read_text() and "[memory]" not in path.read_text()
        # The other two values are written, read back, and written again the same.
        for value in ("off", "wide"):
            cfg.memory = MemoryConfig(gather=value)
            save(cfg)
            text = path.read_text()
            assert f'[memory]\ngather = "{value}"\n' in text
            loaded = config.load()
            assert loaded.memory.gather == value
            save(loaded)
            assert path.read_text() == text
        # A file that still says "guide" is the default, and saves without the key.
        path.write_text('chat_backend = "vllm"\n[memory]\ngather = "guide"\n')
        loaded = config.load()
        assert loaded.memory.gather == "guide"
        save(loaded)
        assert "gather" not in path.read_text()
        # A file written while the default was off keeps its off.
        path.write_text('chat_backend = "vllm"\n[memory]\ngather = "off"\n')
        assert config.load().memory.gather == "off"


def test_a_saved_value_is_kept_and_a_missing_one_is_the_larger_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        # A config with no value for either key gets the shipped defaults (#349).
        path.write_text('chat_backend = "vllm"\n[memory]\nrecall_docs = 5\n')
        memory = config.load().memory
        assert (memory.recall_chars, memory.recall_item_chars) == (4000, 900)
        assert memory.recall_docs == 5
        # A config that saved the earlier defaults keeps them: its value is read as
        # written, and saving it again writes both keys, since neither is the default.
        path.write_text("[memory]\nrecall_chars = 900\nrecall_item_chars = 234\n")
        cfg = config.load()
        assert (cfg.memory.recall_chars, cfg.memory.recall_item_chars) == (900, 234)
        save(cfg)
        assert "recall_chars = 900\nrecall_item_chars = 234\n" in path.read_text()
        memory = config.load().memory
        assert (memory.recall_chars, memory.recall_item_chars) == (900, 234)
        # The shipped default itself is never written.
        cfg.memory = MemoryConfig(recall_chars=4000, recall_item_chars=900)
        save(cfg)
        assert "recall_" not in path.read_text()
        assert (MemoryConfig().recall_chars, MemoryConfig().recall_item_chars) == (
            4000,
            900,
        )


def test_a_file_without_followup_query_gets_the_new_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        path.write_text("[memory]\nrecall_records = 5\n")
        assert config.load().memory.followup_query == "recent"
        path.write_text('[memory]\nfollowup_query = "message"\n')
        assert config.load().memory.followup_query == "message"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("recall_records", 0),
        ("recall_records", 20),
        ("recall_docs", 0),
        ("recall_docs", 20),
        ("recall_chars", 200),
        ("recall_chars", 8000),
        ("recall_notes", "on"),
        ("recall_notes", "off"),
        ("search_when_empty", "off"),
        ("search_when_empty", "hint"),
        ("recall_min_cosine", 0.2),
        ("recall_min_cosine", 0.95),
        ("recall_min_cosine", 0.55),
        ("recall_min_coverage", 0.1),
        ("recall_min_coverage", 1.0),
        ("recall_min_coverage", 1),  # a whole number is a number
        ("recall_item_chars", 120),
        ("recall_item_chars", 2000),
        ("recall_excerpt", "head"),
        ("recall_excerpt", "match"),
        ("recall_form", "excerpt"),
        ("recall_form", "list"),
        ("recall_list_max", 1),
        ("recall_list_max", 50),
        ("recall_list_max", 20),
        ("recall_open_chars", 500),
        ("recall_open_chars", 16000),
        ("recall_open_chars", 4000),
        ("excerpt_open", "off"),
        ("excerpt_open", "on"),
        ("recall_source", "turns"),
        ("recall_source", "items"),
        ("open_sources", "off"),
        ("open_sources", "on"),
        ("compress_at_chars", 4000),
        ("compress_at_chars", 60000),
        ("compress_at_chars", 12000),
        ("compress_parallel", 1),
        ("compress_parallel", 8),
        ("compress_parallel", 4),
        ("compress_cache", ""),
        ("compress_cache", "~/runs/compress_cache.jsonl"),
        ("note_gate", "typed"),
        ("note_gate", "pointed"),
        ("note_gate", "meaning"),
        ("note_gate", "model"),
        ("note_min_cosine", 0.2),
        ("note_min_cosine", 0.95),
        ("note_min_cosine", 0.55),
        ("followup_query", "recent"),
        ("followup_query", "message"),
        ("gather", "off"),
        ("gather", "wide"),
        ("gather", "guide"),
        ("gather_min_cosine", 0.2),
        ("gather_min_cosine", 0.55),
        ("gather_min_cosine", 0.4),
    ],
)
def test_the_bounds_and_values_are_accepted(name: str, value: object) -> None:
    assert errors(**{name: value}) == []


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("recall_records", -1, "[memory] recall_records must be from 0 to 20, got -1"),
        ("recall_records", 21, "[memory] recall_records must be from 0 to 20, got 21"),
        ("recall_docs", 21, "[memory] recall_docs must be from 0 to 20, got 21"),
        (
            "recall_chars",
            199,
            "[memory] recall_chars must be from 200 to 8000, got 199",
        ),
        (
            "recall_chars",
            8001,
            "[memory] recall_chars must be from 200 to 8000, got 8001",
        ),
        ("recall_docs", "3", "[memory] recall_docs must be a whole number, got '3'"),
        ("recall_docs", 2.5, "[memory] recall_docs must be a whole number, got 2.5"),
        (
            "recall_records",
            True,
            "[memory] recall_records must be a whole number, got True",
        ),
        (
            "recall_notes",
            "maybe",
            "[memory] recall_notes 'maybe' is not recognised. "
            "Must be one of ('on', 'off').",
        ),
        (
            "recall_notes",
            False,
            "[memory] recall_notes False is not recognised. "
            "Must be one of ('on', 'off').",
        ),
        (
            "search_when_empty",
            "always",
            "[memory] search_when_empty 'always' is not recognised. "
            "Must be one of ('off', 'hint').",
        ),
        (
            "recall_min_cosine",
            0.19,
            "[memory] recall_min_cosine must be from 0.2 to 0.95, got 0.19",
        ),
        (
            "recall_min_cosine",
            0.96,
            "[memory] recall_min_cosine must be from 0.2 to 0.95, got 0.96",
        ),
        (
            "recall_min_coverage",
            0.09,
            "[memory] recall_min_coverage must be from 0.1 to 1.0, got 0.09",
        ),
        (
            "recall_min_coverage",
            1.01,
            "[memory] recall_min_coverage must be from 0.1 to 1.0, got 1.01",
        ),
        (
            "recall_min_coverage",
            2,
            "[memory] recall_min_coverage must be from 0.1 to 1.0, got 2",
        ),
        (
            "recall_item_chars",
            119,
            "[memory] recall_item_chars must be from 120 to 2000, got 119",
        ),
        (
            "recall_item_chars",
            2001,
            "[memory] recall_item_chars must be from 120 to 2000, got 2001",
        ),
        (
            "recall_item_chars",
            250.5,
            "[memory] recall_item_chars must be a whole number, got 250.5",
        ),
        (
            "recall_item_chars",
            True,
            "[memory] recall_item_chars must be a whole number, got True",
        ),
        (
            "recall_excerpt",
            "tail",
            "[memory] recall_excerpt 'tail' is not recognised. "
            "Must be one of ('head', 'match').",
        ),
        (
            "recall_excerpt",
            True,
            "[memory] recall_excerpt True is not recognised. "
            "Must be one of ('head', 'match').",
        ),
        (
            "recall_form",
            "table",
            "[memory] recall_form 'table' is not recognised. "
            "Must be one of ('excerpt', 'list').",
        ),
        (
            "recall_form",
            True,
            "[memory] recall_form True is not recognised. "
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
            "recall_list_max",
            True,
            "[memory] recall_list_max must be a whole number, got True",
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
            "recall_open_chars",
            "4000",
            "[memory] recall_open_chars must be a whole number, got '4000'",
        ),
        (
            "excerpt_open",
            "maybe",
            "[memory] excerpt_open 'maybe' is not recognised. "
            "Must be one of ('off', 'on').",
        ),
        (
            "excerpt_open",
            True,
            "[memory] excerpt_open True is not recognised. "
            "Must be one of ('off', 'on').",
        ),
        (
            "recall_source",
            "both",
            "[memory] recall_source 'both' is not recognised. "
            "Must be one of ('turns', 'items').",
        ),
        (
            "open_sources",
            True,
            "[memory] open_sources True is not recognised. "
            "Must be one of ('off', 'on').",
        ),
        (
            "compress_at_chars",
            3999,
            "[memory] compress_at_chars must be from 4000 to 60000, got 3999",
        ),
        (
            "compress_at_chars",
            60001,
            "[memory] compress_at_chars must be from 4000 to 60000, got 60001",
        ),
        (
            "compress_at_chars",
            "12000",
            "[memory] compress_at_chars must be a whole number, got '12000'",
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
            "compress_parallel",
            "4",
            "[memory] compress_parallel must be a whole number, got '4'",
        ),
        (
            "compress_cache",
            "a\x00b",
            "[memory] compress_cache must be a path string without NUL",
        ),
        (
            "compress_cache",
            7,
            "[memory] compress_cache must be a path string without NUL",
        ),
        (
            "note_gate",
            "typing",
            "[memory] note_gate 'typing' is not recognised. "
            "Must be one of ('typed', 'pointed', 'meaning', 'model').",
        ),
        (
            "note_gate",
            True,
            "[memory] note_gate True is not recognised. "
            "Must be one of ('typed', 'pointed', 'meaning', 'model').",
        ),
        (
            "note_min_cosine",
            0.19,
            "[memory] note_min_cosine must be from 0.2 to 0.95, got 0.19",
        ),
        (
            "note_min_cosine",
            0.96,
            "[memory] note_min_cosine must be from 0.2 to 0.95, got 0.96",
        ),
        (
            "note_min_cosine",
            "0.5",
            "[memory] note_min_cosine must be a number, got '0.5'",
        ),
        (
            "note_min_cosine",
            True,
            "[memory] note_min_cosine must be a number, got True",
        ),
        (
            "followup_query",
            "history",
            "[memory] followup_query 'history' is not recognised. "
            "Must be one of ('recent', 'message').",
        ),
        (
            "followup_query",
            False,
            "[memory] followup_query False is not recognised. "
            "Must be one of ('recent', 'message').",
        ),
        (
            "gather",
            "narrow",
            "[memory] gather 'narrow' is not recognised. "
            "Must be one of ('off', 'wide', 'guide').",
        ),
        (
            "gather",
            True,
            "[memory] gather True is not recognised. "
            "Must be one of ('off', 'wide', 'guide').",
        ),
        (
            "gather_min_cosine",
            0.19,
            "[memory] gather_min_cosine must be from 0.2 to 0.55, got 0.19",
        ),
        (
            "gather_min_cosine",
            0.56,
            "[memory] gather_min_cosine must be from 0.2 to 0.55, got 0.56",
        ),
        (
            "gather_min_cosine",
            "0.4",
            "[memory] gather_min_cosine must be a number, got '0.4'",
        ),
        (
            "gather_min_cosine",
            True,
            "[memory] gather_min_cosine must be a number, got True",
        ),
        (
            "gather_min_cosine",
            float("nan"),
            "[memory] gather_min_cosine must be from 0.2 to 0.55, got nan",
        ),
        (
            "recall_min_cosine",
            True,
            "[memory] recall_min_cosine must be a number, got True",
        ),
        (
            "recall_min_coverage",
            False,
            "[memory] recall_min_coverage must be a number, got False",
        ),
        (
            "recall_min_cosine",
            "0.5",
            "[memory] recall_min_cosine must be a number, got '0.5'",
        ),
        (
            "recall_min_coverage",
            float("nan"),
            "[memory] recall_min_coverage must be from 0.1 to 1.0, got nan",
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
    assert config_cli.main(["keys", "recall_records", "-X", argv[1]]) == 0
    assert "memory.recall_records" in capsys.readouterr().out
    developer = cli._build_parser(developer=True).format_help().splitlines()
    plain = cli._build_parser(developer=False).format_help()
    for name in CHANGED:
        key = f"memory.{name}"
        assert any(
            line.startswith(f"  {key} ") and line.endswith(f"({ISSUE[name]})")
            for line in developer
        ), key
        assert key not in plain


@pytest.mark.parametrize(
    "setting",
    [
        "memory.recall_records=21",
        "memory.recall_chars=100",
        "memory.recall_docs=two",
        "memory.recall_notes=maybe",
        "memory.search_when_empty=always",
        "memory.recall_min_cosine=0.1",
        "memory.recall_min_cosine=high",
        "memory.recall_min_cosine=true",
        "memory.recall_min_coverage=1.5",
        "memory.recall_item_chars=100",
        "memory.recall_item_chars=wide",
        "memory.recall_excerpt=tail",
        "memory.recall_form=table",
        "memory.recall_list_max=0",
        "memory.recall_list_max=51",
        "memory.recall_list_max=many",
        "memory.recall_open_chars=499",
        "memory.recall_open_chars=16001",
        "memory.recall_open_chars=long",
        "memory.excerpt_open=maybe",
        "memory.recall_source=both",
        "memory.open_sources=maybe",
        "memory.compress_at_chars=3999",
        "memory.compress_at_chars=60001",
        "memory.compress_at_chars=big",
        "memory.compress_parallel=0",
        "memory.compress_parallel=9",
        "memory.compress_parallel=many",
        "memory.note_gate=typing",
        "memory.note_min_cosine=0.1",
        "memory.note_min_cosine=high",
        "memory.followup_query=history",
        "memory.followup_query=on",
        "memory.gather=narrow",
        "memory.gather=true",
        "memory.gather_min_cosine=0.1",
        "memory.gather_min_cosine=0.6",
        "memory.gather_min_cosine=low",
    ],
)
def test_x_refuses_a_bad_value_with_a_clear_message(
    setting: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli._parse(["-X", setting, "--base-url", "http://127.0.0.1:1/v1", "q"])
    assert exit_.value.code == 2
    assert setting.partition("=")[0] in capsys.readouterr().err


def test_a_whole_number_bar_is_saved_and_read_back_and_x_reads_it_as_a_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        path.write_text("[memory]\nrecall_min_coverage = 1\n")
        assert config.load().memory.recall_min_coverage == 1
        assert not errors(recall_min_coverage=config.load().memory.recall_min_coverage)
        cfg.memory = MemoryConfig(recall_min_coverage=1)
        save(cfg)
        assert "recall_min_coverage = 1\n" in path.read_text()
        cli._parse(
            [
                "-X",
                "memory.recall_min_coverage=1",
                "--base-url",
                "http://127.0.0.1:1/v1",
                "q",
            ]
        )
        assert config.overrides() == {"memory.recall_min_coverage": 1.0}
        assert isinstance(config.overrides()["memory.recall_min_coverage"], float)

"""``-X KEY=VALUE``: one setting for one run, and ``melo config keys|get|set``.

A key that has an old option becomes that option before parsing, so it gives the
namespace, the checks and the refusals the option gives. Any other key overrides the
config this process loads; the config file never sees it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import localmelo.__main__ as cli
from localmelo.support import config, config_cli, settings
from localmelo.support.option_tiers import SETTINGS


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _namespace(argv: list[str]) -> dict[str, Any]:
    return vars(cli._parse(argv)[1])


def _exit(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    with pytest.raises(SystemExit) as exc:
        cli._parse(argv)
    assert isinstance(exc.value.code, int)
    return exc.value.code, capsys.readouterr().err


BASE = ["--base-url", "http://x/v1"]
EMBED = [*BASE, "--embedding-base-url", "http://e/v1"]
# (the key, the value -X gets, the old option and its value, options both need)
# fmt: off
ROWS: list[tuple[str, str, list[str], list[str]]] = [
    ("agent.attempt_policy", "none", ["--attempt-policy", "none"], []),
    ("agent.reflection_gate", "utility", ["--reflection-gate", "utility"], []),
    ("agent.fixed_prefix", "false", ["--fixed-prefix", "off"], []),
    ("agent.window_eviction", "slide", ["--window-eviction", "slide"], []),
    ("agent.reflect_tool_guard", "off", ["--reflect-tool-guard", "off"], []),
    ("agent.cache_layout", '"v3"', ["--cache-layout", "v3"], []),
    ("agent.window_messages", "64", ["--window-messages", "64"], []),
    ("agent.brief_items", "true", ["--brief-items", "on"], []),
    ("agent.reasoning", "compress", ["--reasoning", "compress"], []),
    ("agent.replay_turns", "2", ["--replay-turns", "2"], []),
    ("agent.note_writer", "heuristic", ["--note-writer", "heuristic"], []),
    ("agent.captured_history", "on", ["--captured-history", "on"], []),
    ("agent.context_window", "8192", ["--context-window", "8192"], []),
    ("agent.output_reserve", "512", ["--output-reserve", "512"], []),
    ("agent.safety_margin", "256", ["--safety-margin", "256"], []),
    ("agent.tool_exposure", "all", ["--tool-exposure", "all"], []),
    ("retrieval.retriever", "bm25", ["--retriever", "bm25"], []),
    ("retrieval.recall_at", '["start", "step"]', ["--recall-at", "start,step"], []),
    ("retrieval.k", "2", ["--recall-k", "2"], []),
    ("retrieval.kinds", "note,draft", ["--recall-kinds", "note,draft"], []),
    ("retrieval.placement", "message", ["--recall-placement", "message"], []),
    ("retrieval.min_score", "0.5", ["--recall-min-score", "0.5"], []),
    ("retrieval.extra_body", '{"top_k": 3}', ["--retrieval-extra-body", '{"top_k": 3}'], []),
    ("memory.admit_on_outcome", "true", ["--admit-on-outcome", "on"], []),
    ("memory.keep_decision_text", "true", ["--keep-decision-text"], []),
    ("memory.dir", "/tmp/mem", ["--memory-dir", "/tmp/mem"], []),
    ("request_timeout_s", "45.5", ["--request-timeout", "45.5"], []),
    ("sessions.record_direct", "false", ["--no-save-session"], []),
    ("endpoint.embedding_url", "http://e/v1", ["--embedding-base-url", "http://e/v1"], BASE),
    ("endpoint.embedding_model", "bge", ["--embedding-model", "bge"], EMBED),
    ("endpoint.reasoning_field", "reasoning", ["--reasoning-field", "reasoning"], []),
]
# fmt: on


def test_the_rows_cover_all_31_settings() -> None:
    assert {row[0] for row in ROWS} == {item.key for item in SETTINGS}
    assert [row[2][0] for row in ROWS] == [item.flag for item in SETTINGS]


@pytest.mark.parametrize(
    ("key", "value", "old", "extra"), ROWS, ids=[r[0] for r in ROWS]
)
def test_a_setting_gives_the_namespace_of_its_old_option(
    key: str, value: str, old: list[str], extra: list[str]
) -> None:
    new = _namespace(["-X", f"{key}={value}", *extra, "--", "q"])
    assert config.overrides() == {}
    assert new == _namespace([*old, *extra, "--", "q"])


@pytest.mark.parametrize(
    "form",
    [
        ["-X", "retrieval.k=2"],
        ["-Xretrieval.k=2"],
        ["--set", "retrieval.k=2"],
        ["--set=retrieval.k=2"],
        ["-X", "retrieval.k=9", "-X", "retrieval.k=2"],  # the last one wins
    ],
)
def test_each_form_of_the_setter_is_the_option(form: list[str]) -> None:
    assert _namespace([*form, "q"]) == _namespace(["--recall-k", "2", "q"])


def test_booleans_read_true_on_false_and_off() -> None:
    for word, flag in (("true", "on"), ("on", "on"), ("false", "off"), ("off", "off")):
        assert _namespace(["-X", f"agent.fixed_prefix={word}", "q"]) == _namespace(
            ["--fixed-prefix", flag, "q"]
        )


@pytest.mark.parametrize(
    ("setter", "old"),
    [
        (["-X", "retrieval.k=99"], ["--recall-k", "99"]),
        (["-X", "agent.cache_layout=v9"], ["--cache-layout", "v9"]),
        (["-X", "retrieval.retriever=nope"], ["--retriever", "nope"]),
        (
            ["-X", "retrieval.retriever=off", "-X", "retrieval.k=2"],
            ["--retriever", "off", "--recall-k", "2"],
        ),
    ],
)
def test_a_refusal_is_the_options_refusal(
    setter: list[str], old: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    via_setter = _exit([*setter, "q"], capsys)
    assert via_setter[0] == 2
    assert via_setter == _exit([*old, "q"], capsys)


def test_a_value_the_option_refuses_names_what_it_allows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, err = _exit(["-X", "agent.cache_layout=v9", "q"], capsys)
    assert code == 2 and "choose from 'v2', 'v3'" in err


def test_a_setting_reaches_the_run_like_its_option() -> None:
    from localmelo.tests.cli.test_direct_mode import _main

    via_x = _main("-X", "retrieval.k=2", "-X", "agent.window_messages=64", "q")
    via_option = _main("--recall-k", "2", "--window-messages", "64", "q")
    assert via_x.code == via_option.code == 0
    assert via_x.options == via_option.options and via_x.options.recall_k == 2
    assert via_x.agent.run_task.await_args == via_option.agent.run_task.await_args


# -- what is and is not a setting, and where it stops ------------------------------


def test_a_setting_after_the_separator_is_task_text() -> None:
    assert _namespace(["--", "-X", "retrieval.k=2"])["query"] == ["-X", "retrieval.k=2"]
    assert _namespace(["-Xmx is a java flag"])["query"] == ["-Xmx is a java flag"]
    assert config.overrides() == {}


def test_a_setting_first_makes_the_next_word_a_task_not_a_command() -> None:
    assert _namespace(["-X", "cli.stream=true", "--", "serve"])["query"] == ["serve"]
    assert config.overrides() == {"cli.stream": True}
    # A lone word without `--` is refused (test_command_words.py); a longer task is not.
    assert _namespace(["-X", "cli.stream=true", "serve", "the", "docs"])["query"] == [
        "serve",
        "the",
        "docs",
    ]


def test_a_setting_works_with_a_command_word(tmp_path: Path) -> None:
    args = _namespace(["serve", "-X", "scheduler.policy=fcfs", "-p", "9000"])
    assert (args["serve"], args["port"]) == (True, 9000)
    assert config.overrides() == {"scheduler.policy": "fcfs"}
    brief = tmp_path / "brief.md"
    brief.write_text("Write hello.txt\n", encoding="utf-8")
    via_word = _namespace(["worker", str(brief), "-X", "agent.cache_layout=v3"])
    assert via_word == _namespace(["--brief", str(brief), "--cache-layout", "v3"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-X"], "-X needs KEY=VALUE"),
        (["-X", "retrieval.k"], "-X needs KEY=VALUE, got 'retrieval.k'"),
        (["-X", "retrieval.kk=1", "q"], "nearest: retrieval.k, retrieval.kinds"),
        # `mcp_config` is a key, so the nearest key beats the option of that name.
        (["-X", "agent.mcp_config=f", "q"], "nearest: mcp_config"),
        (["--set", "nope=1", "q"], "unknown setting 'nope'"),
        (["-X", "scheduler.policy=nope", "q"], "('off', 'fcfs', 'class')"),
        (["-X", "cli.status_line=loud", "q"], "is not one of ('auto', 'off')"),
        (["-X", "scheduler.n_srv=many", "q"], "expected a whole number"),
        (["-X", "cli.stream=maybe", "q"], "expected true or false"),
    ],
)
def test_a_wrong_setting_is_a_usage_error_that_says_what_to_do(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, err = _exit(argv, capsys)
    assert code == 2 and message in err
    if "unknown" in message or "nearest" in message:
        assert "melo config keys" in err


@pytest.mark.parametrize(
    ("key", "use"),
    [
        ("agent.max_steps", "-s N (--steps)"),
        ("max_steps", "-s N (--steps)"),
        ("agent.steps", "-s N (--steps)"),
        ("agent.timeout", "-t SEC (--timeout)"),
        ("agent.workspace", "-w DIR (--workspace)"),
        ("agent.chat_model", "-m NAME (--model)"),
        ("agent.model", "-m NAME (--model)"),
        ("endpoint.base_url", "-u URL (--url)"),
        ("agent.output_format", "-o FORMAT (--output)"),
        ("agent.yolo", "-y (--yolo)"),
        ("agent.resume", "-r [ID] (--resume)"),
        ("agent.builtins", "--builtins SET"),
        ("memory", "--memory on|off"),
        ("agent.mcp-only", "--mcp-only"),
        ("agent.save_session", "--title T"),
    ],
)
def test_a_key_named_like_an_option_says_the_option(
    key: str, use: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, err = _exit(["-X", f"{key}=5", "q"], capsys)
    assert code == 2
    assert err == f"melo: error: {key} is not a setting; use {use} for this run.\n"


def test_melo_config_says_the_option_too(capsys: pytest.CaptureFixture[str]) -> None:
    expected = (
        "melo: error: agent.max_steps is not a setting; "
        "use -s N (--steps) for this run.\n"
    )
    assert _config(["get", "agent.max_steps"], capsys)[1:] == ("", expected)
    assert _config(["set", "agent.max_steps", "5"], capsys)[1:] == ("", expected)


def test_a_name_that_is_neither_a_key_nor_an_option_keeps_the_nearest_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, err = _exit(["-X", "retrieval.kk=1", "q"], capsys)
    assert code == 2
    assert err == (
        "melo: error: unknown setting 'retrieval.kk'; nearest: retrieval.k, "
        "retrieval.kinds, retrieval.retriever. Every key: melo config keys\n"
    )
    # A real key whose name an option shares is still a key.
    assert settings.lookup("gateway.workspace").name == "gateway.workspace"


def test_a_key_without_an_option_overrides_the_config_this_process_loads() -> None:
    _namespace(
        [
            "-X",
            "cli.stream=off",
            "-X",
            "scheduler.n_srv=4",
            "-X",
            "ui.mode=developer",
            "q",
        ]
    )
    assert config.overrides() == {
        "cli.stream": False,
        "scheduler.n_srv": 4,
        "ui.mode": "developer",
    }
    cfg = config.load()
    assert (cfg.cli.stream, cfg.scheduler.n_srv, cfg.ui.mode) == (False, 4, "developer")
    config.set_overrides({})
    assert config.load().cli.stream is True


def test_record_direct_true_beats_a_file_that_says_false() -> None:
    config.save(config.Config(sessions=config.SessionsConfig(record_direct=False)))
    assert config.load().sessions.record_direct is False
    assert _namespace(["-X", "sessions.record_direct=false", "q"])["no_save_session"]
    _namespace(["-X", "sessions.record_direct=true", "q"])
    assert config.load().sessions.record_direct is True


# -- melo config keys | get | set --------------------------------------------------


def _config(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    try:
        code = config_cli.main(argv)
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        code = exc.code
    out, err = capsys.readouterr()
    return code, out, err


def test_keys_lists_value_default_and_source(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = config.Config()
    cfg.retrieval.k = 5
    config.save(cfg)
    code, out, _ = _config(["keys", "recall", "-X", "retrieval.min_score=0.5"], capsys)
    rows = [line.split() for line in out.splitlines()]
    assert code == 0 and rows[0] == ["key", "value", "default", "source"]
    assert {row[0]: row[1:] for row in rows[1:]} == {
        "retrieval.recall_at": ["[]", "[]", "default"],
        "retrieval.k": ["5", "3", "file"],
        "retrieval.kinds": ["[]", "[]", "default"],
        "retrieval.placement": ['"tool"', '"tool"', "file"],  # save() writes the table
        "retrieval.min_score": ["0.5", "0.35", "-X"],
        # The memory recall settings (#349) have "recall" in their names too.
        "memory.recall_records": ["3", "3", "default"],
        "memory.recall_docs": ["2", "2", "default"],
        "memory.recall_chars": ["900", "900", "default"],
        "memory.recall_notes": ['"on"', '"on"', "default"],
        "memory.recall_min_cosine": ["0.55", "0.55", "default"],
        "memory.recall_min_coverage": ["0.6", "0.6", "default"],
        "memory.recall_item_chars": ["200", "200", "default"],
        "memory.recall_excerpt": ['"head"', '"head"', "default"],
        "memory.recall_form": ['"excerpt"', '"excerpt"', "default"],
        "memory.recall_list_max": ["20", "20", "default"],
        "memory.recall_open_chars": ["4000", "4000", "default"],
    }


def test_keys_lists_every_key_once_and_marks_one_run_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _ = _config(["keys", "-X", "agent.tool_exposure=all"], capsys)
    names = [line.split()[0] for line in out.splitlines()[1:]]
    assert (
        code == 0
        and names == list(settings.catalog())
        and len(set(names)) == len(names)
    )
    row = next(
        r.split() for r in out.splitlines() if r.startswith("agent.tool_exposure")
    )
    assert row[1:] == ['"all"', '"retrieve"', "-X", "(one", "run", "only)"]
    assert "chat_backend" not in names and "cli.reasoning_tags" not in names


@pytest.mark.parametrize(
    ("backend", "window", "said"),
    [
        ("vllm", None, "64 (local server)"),
        ("openai", None, "20 (cloud)"),
        ("vllm", 32, "32"),
    ],
)
def test_keys_and_get_show_the_window_with_its_reason(
    capsys: pytest.CaptureFixture[str], backend: str, window: int | None, said: str
) -> None:
    cfg = config.Config(chat_backend=backend)
    cfg.agent.window_messages = window
    config.save(cfg)
    default = "20 (cloud)" if backend == "openai" else "64 (local server)"
    row = _config(["keys", "window_messages"], capsys)[1].splitlines()[1]
    source = "file" if window else "default"
    assert " ".join(row.split()) == f"agent.window_messages {said} {default} {source}"
    assert _config(["get", "agent.window_messages"], capsys)[:2] == (0, said + "\n")


def test_keys_with_no_match_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _config(["keys", "zzz"], capsys)
    assert (code, out) == (1, "") and "no setting matches 'zzz'" in err


def test_get_prints_one_value(capsys: pytest.CaptureFixture[str]) -> None:
    assert _config(["get", "scheduler.policy"], capsys)[:2] == (0, '"off"\n')
    assert _config(["get", "agent.replay_turns"], capsys)[:2] == (0, "unset\n")
    assert _config(["get", "agent.attempt_policy"], capsys)[:2] == (0, '"reflect"\n')
    assert _config(["get", "retrieval.k", "-X", "retrieval.k=8"], capsys)[:2] == (
        0,
        "8\n",
    )


def test_set_checks_a_key_and_saves_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / ".cache/localmelo/config.toml"
    code, out, _ = _config(["set", "retrieval.k", "4"], capsys)
    assert code == 0 and out.startswith(
        "retrieval.k = 4  (~/.cache/localmelo/config.toml)"
    )
    assert config.load().retrieval.k == 4
    assert _config(["set", "cli.stream", "off"], capsys)[0] == 0
    assert config.load().cli.stream is False
    assert _config(["set", "agent.replay_turns", "none"], capsys)[0] == 0
    refused = _config(["set", "retrieval.k", "3", "-X", "cli.stream=on"], capsys)
    assert refused[0] == 2 and "-X does not apply to melo config set" in refused[2]
    before = path.read_text(encoding="utf-8")
    for key, value, why in (
        ("scheduler.policy", "nope", "('off', 'fcfs', 'class')"),
        ("retrieval.k", "many", "expected a whole number"),
        ("agent.attempt_policy", "none", "is for one run only; use melo -X"),
        ("retrieval.kk", "1", "nearest: retrieval.k"),
    ):
        code, _, err = _config(["set", key, value], capsys)
        assert code == 2 and why in err, key
    assert path.read_text(encoding="utf-8") == before


def test_config_hints_at_a_task_and_rejects_a_stray_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _, err = _config(["the", "nginx", "server"], capsys)
    assert code == 2 and err.endswith('melo "config the nginx server"\n')
    assert _config(["--bogus"], capsys)[0] == 2
    assert _config(["set", "retrieval.k"], capsys)[2].startswith(
        "melo: error: usage: melo config"
    )


# -- the old options of the settings are listed as aliases of their keys -------------


def test_the_old_option_of_each_setting_is_listed_with_its_key() -> None:
    from localmelo.support.option_tiers import OLD_SPELLINGS

    for item in SETTINGS:
        assert OLD_SPELLINGS[item.flag].startswith(f"-X {item.key}=")
    assert OLD_SPELLINGS["--no-save-session"] == "-X sessions.record_direct=false"
    assert OLD_SPELLINGS["--keep-decision-text"] == "-X memory.keep_decision_text=true"

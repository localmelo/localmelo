"""``--brief-items`` and ``[agent] brief_items``: experimental, off by default."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import config
from localmelo.support.option_tiers import EXPERIMENTAL, OPTION_TIERS

from .test_reasoning_flags import _parse, _write_config


def _built(argv: list[str]) -> bool:
    parser, args = _parse(argv)
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"].brief_items
    assert isinstance(built, bool)
    return built


def test_the_flag_is_experimental_and_off_by_default() -> None:
    assert OPTION_TIERS["--brief-items"] == EXPERIMENTAL
    parser, args = _parse(["q"])
    assert cli._direct_options(parser, args).brief_items is False
    parser, args = _parse(["--brief-items", "on", "q"])
    assert cli._direct_options(parser, args).brief_items is True


def test_a_bad_value_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        cli._build_parser().parse_args(["--brief-items", "maybe", "q"])
    assert exc.value.code == 2


def test_the_config_key_applies_only_when_the_flag_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent="\n[agent]\nbrief_items = true\n")
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _built(["q"]) is True
    assert _built(["--brief-items", "off", "q"]) is False

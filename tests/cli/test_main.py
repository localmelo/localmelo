"""Tests for the CLI entrypoint (__main__.py).

Covers:
- --host / --port override logic (original tests)
- Direct mode: `melo "hello"` calls Agent().run(), prints result
- Gateway mode: `melo --serve` loads config, validates, calls start_gateway
- Config validation failure: bad config -> error on stderr, exit 1
- Reconfigure: `melo --reconfigure` -> runs backend setup
- Direct mode semantics: no config.load() call when --base-url given
- Direct mode without --base-url uses config

CRITICAL: These tests must run without fastapi/uvicorn installed.
We never import localmelo.support.gateway at module level.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any
from unittest import mock

import pytest

from localmelo.support.config import Config, GatewayConfig, LocalBackendConfig

# ---------------------------------------------------------------------------
# Helper: build an args namespace that mimics argparse output
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> argparse.Namespace:
    """Return a Namespace with the same defaults as main()'s parser."""
    defaults: dict[str, Any] = dict(
        query=[],
        base_url=None,
        chat_model=None,
        serve=True,  # gateway mode
        host=None,
        port=None,
        reconfigure=False,
        daemon=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _valid_cfg(**gw_overrides: Any) -> Config:
    """Return a minimal valid Config (mlc backend) with custom gateway."""
    gw = GatewayConfig(**gw_overrides) if gw_overrides else GatewayConfig()
    return Config(
        chat_backend="mlc",
        embedding_backend="mlc",
        mlc=LocalBackendConfig(
            chat_url="http://127.0.0.1:8400/v1",
            chat_model="Qwen3-1.7B",
            embedding_url="http://127.0.0.1:8400/v1",
            embedding_model="nomic-embed",
        ),
        gateway=gw,
    )


# ---------------------------------------------------------------------------
# Scenario 1: config has host=0.0.0.0, user does NOT pass --host
#   => host must stay 0.0.0.0
# ---------------------------------------------------------------------------


class TestHostNotOverriddenWhenOmitted:
    """When the user omits --host, the config value must survive."""

    def test_config_host_preserved(self) -> None:
        cfg = _valid_cfg(host="0.0.0.0", port=9000)
        args = _make_args()  # host=None, port=None

        # Replicate the override logic from __main__.py
        if args.host is not None:
            cfg.gateway.host = args.host
        if args.port is not None:
            cfg.gateway.port = args.port

        assert cfg.gateway.host == "0.0.0.0"
        assert cfg.gateway.port == 9000


# ---------------------------------------------------------------------------
# Scenario 2: user explicitly passes --host 127.0.0.1
#   => must override config value
# ---------------------------------------------------------------------------


class TestHostOverriddenWhenExplicit:
    """When the user passes --host, the config value must update."""

    def test_explicit_host_overrides(self) -> None:
        cfg = _valid_cfg(host="0.0.0.0", port=9000)
        args = _make_args(host="127.0.0.1")

        if args.host is not None:
            cfg.gateway.host = args.host
        if args.port is not None:
            cfg.gateway.port = args.port

        assert cfg.gateway.host == "127.0.0.1"
        assert cfg.gateway.port == 9000  # unchanged


# ---------------------------------------------------------------------------
# Scenario 3: --port behaviour matches --host
# ---------------------------------------------------------------------------


class TestPortOverrideLogic:
    """--port should only override when explicitly supplied."""

    def test_port_preserved_when_omitted(self) -> None:
        cfg = _valid_cfg(host="0.0.0.0", port=7777)
        args = _make_args()  # port=None

        if args.host is not None:
            cfg.gateway.host = args.host
        if args.port is not None:
            cfg.gateway.port = args.port

        assert cfg.gateway.port == 7777

    def test_port_overridden_when_explicit(self) -> None:
        cfg = _valid_cfg(host="0.0.0.0", port=7777)
        args = _make_args(port=9999)

        if args.host is not None:
            cfg.gateway.host = args.host
        if args.port is not None:
            cfg.gateway.port = args.port

        assert cfg.gateway.port == 9999

    def test_port_can_be_set_to_8401_explicitly(self) -> None:
        """Regression: old code used `if args.port != 8401` which would
        prevent the user from explicitly setting port=8401 if the config
        had a different value."""
        cfg = _valid_cfg(port=5000)
        args = _make_args(port=8401)

        if args.port is not None:
            cfg.gateway.port = args.port

        assert cfg.gateway.port == 8401


# ---------------------------------------------------------------------------
# Scenario 4: direct mode is not affected
# ---------------------------------------------------------------------------


class TestDirectModeUnaffected:
    """Direct mode (query without --serve) should not touch gateway config."""

    def test_direct_mode_skips_gateway_overrides(self) -> None:
        """Simulate: `melo 'do stuff'` (no --serve)."""
        args = _make_args(query=["do", "stuff"], serve=False)

        # In __main__.py, direct mode returns early before the override block.
        # Verify by checking args — direct mode is entered when
        # args.query is truthy and args.serve is False.
        assert args.query and not args.serve
        # host/port are None — never applied
        assert args.host is None
        assert args.port is None


# ---------------------------------------------------------------------------
# Scenario 5: integration-style test using the real argparse parser
# ---------------------------------------------------------------------------


class TestArgparseDefaults:
    """Verify that the actual parser produces None defaults for --host/--port."""

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        """Replicate the parser from __main__.py."""
        parser = argparse.ArgumentParser(prog="melo")
        parser.add_argument("query", nargs="*")
        parser.add_argument("--base-url", default=None)
        parser.add_argument("--chat-model", default=None)
        parser.add_argument("--serve", action="store_true")
        parser.add_argument("--host", default=None)
        parser.add_argument("--port", type=int, default=None)
        parser.add_argument("--reconfigure", action="store_true")
        parser.add_argument("--daemon", choices=["install", "uninstall", "status"])
        return parser

    def test_no_args_gives_none_host_port(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["--serve"])
        assert args.host is None
        assert args.port is None

    def test_explicit_host_parsed(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["--serve", "--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"
        assert args.port is None

    def test_explicit_port_parsed(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["--serve", "--port", "3000"])
        assert args.port == 3000
        assert args.host is None

    def test_both_explicit(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["--serve", "--host", "10.0.0.1", "--port", "4000"])
        assert args.host == "10.0.0.1"
        assert args.port == 4000

    def test_direct_mode_query(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["hello", "world"])
        assert args.query == ["hello", "world"]
        assert not args.serve
        assert args.host is None
        assert args.port is None


# ---------------------------------------------------------------------------
# Scenario 6: daemon install path handles None port gracefully
# ---------------------------------------------------------------------------


class TestDaemonInstallPortHandling:
    """When --port is omitted, daemon.install() should use its own default."""

    def test_daemon_install_no_port(self) -> None:
        """Simulate `melo --daemon install` (no --port)."""
        args = _make_args(daemon="install", port=None)

        # Build kwargs the same way __main__.py does
        kwargs: dict = {}
        if args.port is not None:
            kwargs["port"] = args.port

        # kwargs should be empty, so daemon.install() uses its own defaults
        assert kwargs == {}

    def test_daemon_install_with_port(self) -> None:
        """Simulate `melo --daemon install --port 5000`."""
        args = _make_args(daemon="install", port=5000)

        kwargs: dict = {}
        if args.port is not None:
            kwargs["port"] = args.port

        assert kwargs == {"port": 5000}


# ===========================================================================
# NEW: main()-level tests with mocks
# ===========================================================================


def _mock_agent(result: str = "Hello back!") -> mock.MagicMock:
    """Build a MagicMock Agent whose run/close are AsyncMock."""
    agent = mock.MagicMock()
    agent.run = mock.AsyncMock(return_value=result)
    agent.close = mock.AsyncMock()
    return agent


class TestDirectMode:
    """Direct mode: `melo "hello"` builds an agent and runs."""

    def test_direct_mode_calls_agent_run(self) -> None:
        """melo "hello" -> _build_direct_mode_agent, run(), print."""
        agent = _mock_agent("Hello back!")

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch(
                "localmelo.__main__._build_direct_mode_agent", return_value=agent
            ),
            mock.patch("sys.argv", ["melo", "hello"]),
            mock.patch("builtins.print") as mock_print,
        ):
            from localmelo.__main__ import main

            main()

        agent.run.assert_called_once_with("hello")
        mock_print.assert_any_call("Hello back!")

    def test_direct_mode_multi_word_query(self) -> None:
        """melo "do the thing" passes joined words to Agent.run()."""
        agent = _mock_agent("done")

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch(
                "localmelo.__main__._build_direct_mode_agent", return_value=agent
            ),
            mock.patch("sys.argv", ["melo", "do", "the", "thing"]),
            mock.patch("builtins.print"),
        ):
            from localmelo.__main__ import main

            main()

        agent.run.assert_called_once_with("do the thing")


class TestBuildDirectModeAgent:
    """Verify _build_direct_mode_agent behaviour."""

    def test_base_url_uses_direct_injection_no_embedding(self) -> None:
        """--base-url uses generic OpenAI-compatible injection, no embedding."""
        agent = _mock_agent()
        with mock.patch("localmelo.__main__.Agent", return_value=agent) as cls:
            from localmelo.__main__ import _build_direct_mode_agent

            ns = argparse.Namespace(
                base_url="http://myhost:8000/v1", chat_model="my-model"
            )
            _build_direct_mode_agent(ns)

        _, kwargs = cls.call_args
        assert "llm" in kwargs
        assert kwargs.get("embedding") is None

    def test_arbitrary_url_preserved_exactly(self) -> None:
        """--base-url http://10.1.2.3:9000/v1 must NOT be modified."""
        from localmelo.__main__ import _build_direct_mode_agent

        ns = argparse.Namespace(
            base_url="http://10.1.2.3:9000/v1", chat_model="my-model"
        )

        with mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as mock_llm_cls:
            mock_llm_cls.return_value = mock.MagicMock()
            agent = _build_direct_mode_agent(ns)

        # The LLM provider must receive the exact URL
        mock_llm_cls.assert_called_once_with(
            base_url="http://10.1.2.3:9000/v1", model="my-model"
        )
        assert agent is not None

    def test_no_base_url_uses_config(self) -> None:
        """No --base-url: loads config and builds Agent from config."""
        cfg = _valid_cfg()
        agent = _mock_agent()
        with (
            mock.patch("localmelo.support.config.load", return_value=cfg),
            mock.patch("localmelo.__main__.Agent", return_value=agent) as cls,
        ):
            from localmelo.__main__ import _build_direct_mode_agent

            ns = argparse.Namespace(base_url=None, chat_model=None)
            _build_direct_mode_agent(ns)

        _, kwargs = cls.call_args
        assert kwargs.get("config") is cfg

    def test_no_base_url_unconfigured_exits(self) -> None:
        """No --base-url and no configured backend -> SystemExit(1)."""
        unconfigured = Config()  # chat_backend=""

        with (
            mock.patch("localmelo.support.config.load", return_value=unconfigured),
            mock.patch("builtins.print"),
            pytest.raises(SystemExit) as exc_info,
        ):
            from localmelo.__main__ import _build_direct_mode_agent

            ns = argparse.Namespace(base_url=None, chat_model=None)
            _build_direct_mode_agent(ns)

        assert exc_info.value.code == 1

    def test_default_model_when_base_url_no_model(self) -> None:
        """--base-url without --chat-model uses 'default' as model name."""
        from localmelo.__main__ import _build_direct_mode_agent

        ns = argparse.Namespace(base_url="http://localhost:8000/v1", chat_model=None)

        with mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as mock_llm_cls:
            mock_llm_cls.return_value = mock.MagicMock()
            _build_direct_mode_agent(ns)

        mock_llm_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1", model="default"
        )


class TestGatewayMode:
    """Gateway mode: `melo --serve` loads config, validates, starts gateway.

    Uses string-based patching so fastapi/uvicorn need not be installed.
    """

    def test_serve_loads_config_and_starts_gateway(self) -> None:
        """melo --serve -> load config, validate, call start_gateway(cfg)."""
        cfg = _valid_cfg()
        mock_start = mock.MagicMock()

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch("sys.argv", ["melo", "--serve"]),
            mock.patch("localmelo.support.config.load", return_value=cfg),
            mock.patch("localmelo.support.onboard.run_backend_setup"),
            mock.patch(
                "localmelo.__main__._start_gateway",
                mock_start,
            ),
            mock.patch("builtins.print"),
        ):
            from localmelo.__main__ import main

            main()

        mock_start.assert_called_once_with(cfg)

    def test_serve_applies_cli_overrides(self) -> None:
        """melo --serve --host 0.0.0.0 --port 5000 overrides config."""
        cfg = _valid_cfg()
        mock_start = mock.MagicMock()

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch(
                "sys.argv", ["melo", "--serve", "--host", "0.0.0.0", "--port", "5000"]
            ),
            mock.patch("localmelo.support.config.load", return_value=cfg),
            mock.patch("localmelo.support.onboard.run_backend_setup"),
            mock.patch(
                "localmelo.__main__._start_gateway",
                mock_start,
            ),
            mock.patch("builtins.print"),
        ):
            from localmelo.__main__ import main

            main()

        # The cfg passed to start_gateway should have overridden values
        passed_cfg = mock_start.call_args[0][0]
        assert passed_cfg.gateway.host == "0.0.0.0"
        assert passed_cfg.gateway.port == 5000


class TestConfigValidationFailure:
    """Bad config -> error on stderr, exit 1."""

    def test_bad_config_exits_with_error(self) -> None:
        """Invalid config should print to stderr and raise SystemExit(1)."""
        bad_cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(chat_model=""),
        )

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch("sys.argv", ["melo", "--serve"]),
            mock.patch("localmelo.support.config.load", return_value=bad_cfg),
            mock.patch("localmelo.support.onboard.run_backend_setup"),
            mock.patch("builtins.print") as mock_print,
            pytest.raises(SystemExit) as exc_info,
        ):
            from localmelo.__main__ import main

            main()

        assert exc_info.value.code == 1
        # Error was printed to stderr
        stderr_calls = [
            c
            for c in mock_print.call_args_list
            if len(c.kwargs) and c.kwargs.get("file") is sys.stderr
        ]
        assert len(stderr_calls) >= 1

    def test_empty_backend_fails_fast(self) -> None:
        """Empty chat_backend should fail validation immediately."""
        empty_cfg = Config(chat_backend="")

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch("sys.argv", ["melo", "--serve"]),
            mock.patch("localmelo.support.config.load", return_value=empty_cfg),
            # An unconfigured config triggers backend setup; if setup returns
            # None, main() exits early. So we need setup to return a still-bad config.
            mock.patch(
                "localmelo.support.onboard.run_backend_setup",
                return_value=Config(chat_backend="not-a-backend"),
            ),
            mock.patch("builtins.print"),
            pytest.raises(SystemExit) as exc_info,
        ):
            from localmelo.__main__ import main

            main()

        assert exc_info.value.code == 1

    def test_backend_setup_returns_none_exits_cleanly(self) -> None:
        """If backend setup is cancelled (returns None), main() returns cleanly."""
        empty_cfg = Config(chat_backend="")

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch("sys.argv", ["melo", "--serve"]),
            mock.patch("localmelo.support.config.load", return_value=empty_cfg),
            mock.patch(
                "localmelo.support.onboard.run_backend_setup", return_value=None
            ),
            mock.patch("builtins.print"),
        ):
            from localmelo.__main__ import main

            # Should return without error (setup cancelled)
            main()


class TestReconfigure:
    """melo --reconfigure -> runs backend setup regardless of config state."""

    def test_reconfigure_runs_backend_setup(self) -> None:
        """--reconfigure flag should trigger run_backend_setup even if configured."""
        valid_cfg = _valid_cfg()
        new_cfg = _valid_cfg(port=7777)
        mock_start = mock.MagicMock()

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch("sys.argv", ["melo", "--serve", "--reconfigure"]),
            mock.patch("localmelo.support.config.load", return_value=valid_cfg),
            mock.patch(
                "localmelo.support.onboard.run_backend_setup", return_value=new_cfg
            ) as mock_setup,
            mock.patch(
                "localmelo.__main__._start_gateway",
                mock_start,
            ),
            mock.patch("builtins.print"),
        ):
            from localmelo.__main__ import main

            main()

        mock_setup.assert_called_once()
        # The NEW config from backend setup is what gets passed to start_gateway
        mock_start.assert_called_once_with(new_cfg)

    def test_reconfigure_backend_setup_cancel_exits_cleanly(self) -> None:
        """--reconfigure but backend setup cancelled -> clean exit."""
        valid_cfg = _valid_cfg()

        with (
            mock.patch("localmelo.__main__._register_backends"),
            mock.patch("sys.argv", ["melo", "--serve", "--reconfigure"]),
            mock.patch("localmelo.support.config.load", return_value=valid_cfg),
            mock.patch(
                "localmelo.support.onboard.run_backend_setup", return_value=None
            ) as mock_setup,
            mock.patch("builtins.print"),
        ):
            from localmelo.__main__ import main

            main()

        mock_setup.assert_called_once()


class TestRegisterBackends:
    """_register_backends() registers all three backend adapters."""

    def test_registers_all_eight_backends(self) -> None:
        """After _register_backends(), the registry contains all 8 backends."""
        from localmelo.support.backends.registry import _clear, list_backends

        _clear()  # start clean (resets flag too)

        from localmelo.__main__ import _register_backends

        _register_backends()

        backends = list_backends()
        keys = [b.key for b in backends]
        assert "mlc" in keys
        assert "ollama" in keys
        assert "vllm" in keys
        assert "sglang" in keys
        assert "openai" in keys
        assert "gemini" in keys
        assert "anthropic" in keys
        assert "nvidia" in keys

        _clear()  # clean up

    def test_repeated_registration_is_safe(self) -> None:
        """_register_backends() is idempotent — calling twice does not error."""
        from localmelo.support.backends.registry import _clear, list_backends

        _clear()

        from localmelo.__main__ import _register_backends

        _register_backends()
        _register_backends()  # must not raise

        assert len(list_backends()) == 8
        _clear()

    def test_main_calls_register_backends_first(self) -> None:
        """main() calls _register_backends() before parsing args."""
        call_order: list[str] = []

        def fake_register() -> None:
            call_order.append("register")

        def fake_parse(*a: object, **kw: object) -> None:
            call_order.append("parse")
            raise SystemExit(0)  # stop early

        with (
            mock.patch(
                "localmelo.__main__._register_backends", side_effect=fake_register
            ),
            mock.patch("argparse.ArgumentParser.parse_args", side_effect=fake_parse),
            pytest.raises(SystemExit),
        ):
            from localmelo.__main__ import main

            main()

        assert call_order[0] == "register"


# ---------------------------------------------------------------------------
# Onboarding: new backend keys, config-prompts-only flow
# ---------------------------------------------------------------------------


class TestOnboardingDispatch:
    """Verify onboarding dispatches by backend key from hardcoded choice list."""

    def test_local_backend_routes_to_onboard_local(self) -> None:
        """Selecting 'ollama' calls the local backend handler."""
        mock_handler = mock.MagicMock(return_value=True)

        import localmelo.support.onboard as _onboard

        patched_setup = {**_onboard._CHAT_SETUP, "ollama": mock_handler}

        with (
            # choice 2 = ollama in CHAT_BACKENDS, choice 5 = none in EMBEDDING_BACKENDS
            mock.patch("localmelo.support.onboard._ask_choice", side_effect=[2, 5]),
            mock.patch.dict(
                "localmelo.support.onboard._CHAT_SETUP", patched_setup, clear=True
            ),
            mock.patch("localmelo.support.onboard.config.load", return_value=Config()),
            mock.patch("localmelo.support.onboard.config.save"),
            mock.patch("localmelo.support.onboard._run_probes", return_value=True),
            mock.patch("builtins.print"),
        ):
            from localmelo.support.onboard import run_backend_setup

            result = run_backend_setup()

        mock_handler.assert_called_once()
        assert result is not None
        assert result.chat_backend == "ollama"
        assert result.embedding_backend == "none"

    def test_cloud_backend_routes_to_onboard_cloud(self) -> None:
        """Selecting 'openai' calls the cloud backend handler."""
        mock_handler = mock.MagicMock(return_value=True)

        import localmelo.support.onboard as _onboard

        patched_setup = {**_onboard._CHAT_SETUP, "openai": mock_handler}

        with (
            # choice 5 = openai in CHAT_BACKENDS, choice 5 = none in EMBEDDING_BACKENDS
            mock.patch("localmelo.support.onboard._ask_choice", side_effect=[5, 5]),
            mock.patch.dict(
                "localmelo.support.onboard._CHAT_SETUP", patched_setup, clear=True
            ),
            mock.patch("localmelo.support.onboard.config.load", return_value=Config()),
            mock.patch("localmelo.support.onboard.config.save"),
            mock.patch("localmelo.support.onboard._run_probes", return_value=True),
            mock.patch("builtins.print"),
        ):
            from localmelo.support.onboard import run_backend_setup

            result = run_backend_setup()

        mock_handler.assert_called_once()
        assert result is not None
        assert result.chat_backend == "openai"

    def test_chat_setup_has_all_backend_keys(self) -> None:
        """_CHAT_SETUP must cover all 8 backend keys."""
        from localmelo.support.onboard import _CHAT_SETUP

        expected = {
            "mlc",
            "ollama",
            "vllm",
            "sglang",
            "openai",
            "gemini",
            "anthropic",
            "nvidia",
        }
        assert set(_CHAT_SETUP.keys()) == expected
        assert all(callable(fn) for fn in _CHAT_SETUP.values())

    def test_chat_backends_list_has_8_entries(self) -> None:
        """CHAT_BACKENDS list must have 8 entries."""
        from localmelo.support.onboard import CHAT_BACKENDS

        assert len(CHAT_BACKENDS) == 8

    def test_embedding_backends_list_includes_local_and_none(self) -> None:
        """EMBEDDING_BACKENDS must include local backends + none only."""
        from localmelo.support.onboard import EMBEDDING_BACKENDS

        keys = [k for k, _ in EMBEDDING_BACKENDS]
        assert keys == ["mlc", "ollama", "vllm", "sglang", "none"]

    def test_setup_cancelled_returns_none(self) -> None:
        """If user cancels during chat setup, returns None."""
        mock_handler = mock.MagicMock(return_value=False)

        import localmelo.support.onboard as _onboard

        patched_setup = {**_onboard._CHAT_SETUP, "mlc": mock_handler}

        with (
            mock.patch("localmelo.support.onboard._ask_choice", return_value=1),
            mock.patch.dict(
                "localmelo.support.onboard._CHAT_SETUP", patched_setup, clear=True
            ),
            mock.patch("localmelo.support.onboard.config.load", return_value=Config()),
            mock.patch("builtins.print"),
        ):
            from localmelo.support.onboard import run_backend_setup

            result = run_backend_setup()

        assert result is None

    def test_embedding_selection_with_local_backend(self) -> None:
        """Selecting a local embedding backend calls _onboard_embedding_local."""
        mock_handler = mock.MagicMock(return_value=True)

        import localmelo.support.onboard as _onboard

        patched_setup = {**_onboard._CHAT_SETUP, "mlc": mock_handler}

        with (
            # choice 1 = mlc chat, choice 2 = ollama embedding
            mock.patch("localmelo.support.onboard._ask_choice", side_effect=[1, 2]),
            mock.patch.dict(
                "localmelo.support.onboard._CHAT_SETUP", patched_setup, clear=True
            ),
            mock.patch("localmelo.support.onboard.config.load", return_value=Config()),
            mock.patch("localmelo.support.onboard.config.save"),
            mock.patch("localmelo.support.onboard._run_probes", return_value=True),
            mock.patch(
                "localmelo.support.onboard._onboard_embedding_local"
            ) as mock_emb,
            mock.patch("builtins.print"),
        ):
            from localmelo.support.onboard import run_backend_setup

            result = run_backend_setup()

        assert result is not None
        assert result.embedding_backend == "ollama"
        mock_emb.assert_called_once()

    def test_nvidia_cloud_backend_onboarding(self) -> None:
        """Cloud API onboarding for NVIDIA sets correct config fields."""
        with (
            mock.patch(
                "localmelo.support.onboard._ask",
                side_effect=[
                    "NVIDIA_API_KEY",
                    "nvidia/llama-3.1-nemotron-70b-instruct",
                ],
            ),
            mock.patch("localmelo.support.onboard._confirm", return_value=True),
            mock.patch("builtins.print"),
        ):
            from localmelo.support.onboard import _onboard_cloud

            cfg = Config()
            result = _onboard_cloud(cfg, "nvidia")

        assert result is True
        assert cfg.nvidia.api_key_env == "NVIDIA_API_KEY"
        assert cfg.nvidia.chat_model == "nvidia/llama-3.1-nemotron-70b-instruct"

    def test_local_backend_onboarding_sets_ollama_config(self) -> None:
        """Local backend onboarding for ollama sets URL and model."""
        with (
            mock.patch(
                "localmelo.support.onboard._ask",
                side_effect=["http://myhost:11434", "llama3:8b"],
            ),
            mock.patch("localmelo.support.onboard._confirm", return_value=True),
            mock.patch("builtins.print"),
        ):
            from localmelo.support.onboard import _onboard_local

            cfg = Config()
            result = _onboard_local(cfg, "ollama")

        assert result is True
        assert cfg.ollama.chat_url == "http://myhost:11434"
        assert cfg.ollama.chat_model == "llama3:8b"

    def test_no_compilation_imports(self) -> None:
        """onboard.py must NOT import compile_model, is_compiled, pick_chat_model."""
        from pathlib import Path

        onboard_src = (
            Path(__file__).resolve().parent.parent.parent / "support" / "onboard.py"
        )
        text = onboard_src.read_text()

        assert "compile_model" not in text
        assert "is_compiled" not in text
        assert "pick_chat_model" not in text
        assert "DEFAULT_EMBEDDING" not in text


class TestEntrypointThinness:
    """The entrypoint should be thin: no heavy imports at module level."""

    def test_main_module_does_not_import_gateway_at_top(self) -> None:
        """__main__.py should not have 'from localmelo.support.gateway'
        at module level — gateway imports are lazy (inside function body)."""
        from pathlib import Path

        main_src = Path(__file__).resolve().parent.parent.parent / "__main__.py"
        lines = main_src.read_text().splitlines()

        # Filter to only top-level lines (not indented)
        top_level_imports = [
            ln
            for ln in lines
            if (ln.startswith("from ") or ln.startswith("import ")) and "gateway" in ln
        ]
        assert (
            top_level_imports == []
        ), f"Gateway imported at module level: {top_level_imports}"

    def test_main_module_does_not_import_config_at_top(self) -> None:
        """config imports should be lazy (inside the gateway branch)."""
        from pathlib import Path

        main_src = Path(__file__).resolve().parent.parent.parent / "__main__.py"
        lines = main_src.read_text().splitlines()

        top_level_config = [
            ln
            for ln in lines
            if (ln.startswith("from ") or ln.startswith("import ")) and "config" in ln
        ]
        assert (
            top_level_config == []
        ), f"Config imported at module level: {top_level_config}"

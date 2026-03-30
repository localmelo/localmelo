"""Tests for Config -> backend registry delegation with split chat/embedding."""

from __future__ import annotations

import pytest

from localmelo.support.config import Config, GatewayConfig, LocalBackendConfig


class _StubBackend:
    """Minimal stub that satisfies the delegation protocol in Config.

    This is intentionally NOT a BaseBackend subclass -- it only implements
    the methods that Config delegates to (validate / validate_embedding /
    has_embedding).  This keeps the test independent of the full BaseBackend
    ABC.
    """

    def __init__(
        self,
        key: str,
        *,
        validate_errors: list[str] | None = None,
        validate_embedding_errors: list[str] | None = None,
        has_embedding_value: bool = True,
    ) -> None:
        self.key = key
        self._validate_errors = validate_errors or []
        self._validate_embedding_errors = validate_embedding_errors or []
        self._has_embedding_value = has_embedding_value
        self.validate_called = False
        self.validate_embedding_called = False

    def validate(self, cfg: Config) -> list[str]:
        self.validate_called = True
        return list(self._validate_errors)

    def validate_embedding(self, cfg: Config) -> list[str]:
        self.validate_embedding_called = True
        return list(self._validate_embedding_errors)

    def has_embedding(self, cfg: Config) -> bool:
        return self._has_embedding_value


class TestRegistryDelegation:
    """Config should delegate chat and embedding validation independently
    to registered backend adapters."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):  # type: ignore[no-untyped-def]
        import localmelo.support.backends.registry as _reg
        from localmelo.support.backends.registry import _clear

        _clear()
        _reg._DEFAULT_BACKENDS_REGISTERED = True
        yield
        # Restore: allow auto-registration for subsequent test modules.
        _clear()

    # -- has_embedding delegates to backend --

    def test_has_embedding_mlc(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("mlc", has_embedding_value=True)
        _BACKENDS["mlc"] = stub  # type: ignore[assignment]

        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        assert cfg.has_embedding is True

    def test_has_embedding_ollama_with_model(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("ollama", has_embedding_value=True)
        _BACKENDS["ollama"] = stub  # type: ignore[assignment]

        cfg = Config(chat_backend="ollama", embedding_backend="ollama")
        assert cfg.has_embedding is True

    def test_has_embedding_ollama_without_model(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("ollama", has_embedding_value=False)
        _BACKENDS["ollama"] = stub  # type: ignore[assignment]

        cfg = Config(chat_backend="ollama", embedding_backend="ollama")
        assert cfg.has_embedding is False

    def test_has_embedding_none(self) -> None:
        cfg = Config(chat_backend="openai", embedding_backend="none")
        assert cfg.has_embedding is False

    def test_has_embedding_empty(self) -> None:
        cfg = Config(chat_backend="", embedding_backend="")
        assert cfg.has_embedding is False

    def test_has_embedding_unknown_backend(self) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="unknown")
        assert cfg.has_embedding is False

    # -- validate: chat backend delegation --

    def test_validate_delegates_chat_to_registered_backend(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("mlc", validate_errors=[])
        _BACKENDS["mlc"] = stub  # type: ignore[assignment]

        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(chat_url="", chat_model=""),
        )
        errors = cfg.validate()
        assert errors == []
        assert stub.validate_called

    def test_validate_delegates_chat_errors_from_backend(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("ollama", validate_errors=["[ollama] something is wrong"])
        _BACKENDS["ollama"] = stub  # type: ignore[assignment]

        cfg = Config(chat_backend="ollama", embedding_backend="none")
        errors = cfg.validate()
        assert "[ollama] something is wrong" in errors
        assert stub.validate_called

    def test_validate_reports_not_recognised_when_chat_not_registered(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(chat_url="", chat_model=""),
        )
        errors = cfg.validate()
        assert any("no registered adapter" in e for e in errors)

    # -- validate: embedding backend delegation --

    def test_validate_delegates_embedding_to_registered_backend(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        chat_stub = _StubBackend("mlc", validate_errors=[])
        emb_stub = _StubBackend("ollama", validate_embedding_errors=[])
        _BACKENDS["mlc"] = chat_stub  # type: ignore[assignment]
        _BACKENDS["ollama"] = emb_stub  # type: ignore[assignment]

        cfg = Config(
            chat_backend="mlc",
            embedding_backend="ollama",
        )
        errors = cfg.validate()
        assert errors == []
        assert chat_stub.validate_called
        assert emb_stub.validate_embedding_called

    def test_validate_delegates_embedding_errors(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        chat_stub = _StubBackend("mlc", validate_errors=[])
        emb_stub = _StubBackend(
            "ollama",
            validate_embedding_errors=["[ollama] embedding_model is empty"],
        )
        _BACKENDS["mlc"] = chat_stub  # type: ignore[assignment]
        _BACKENDS["ollama"] = emb_stub  # type: ignore[assignment]

        cfg = Config(chat_backend="mlc", embedding_backend="ollama")
        errors = cfg.validate()
        assert "[ollama] embedding_model is empty" in errors

    def test_validate_skips_embedding_when_none(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        chat_stub = _StubBackend("mlc", validate_errors=[])
        _BACKENDS["mlc"] = chat_stub  # type: ignore[assignment]

        cfg = Config(chat_backend="mlc", embedding_backend="none")
        errors = cfg.validate()
        assert errors == []
        assert chat_stub.validate_called

    # -- validate: independent chat + embedding errors --

    def test_chat_and_embedding_errors_collected_independently(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        chat_stub = _StubBackend(
            "ollama", validate_errors=["[ollama] chat_model is empty"]
        )
        emb_stub = _StubBackend(
            "mlc",
            validate_embedding_errors=["[mlc] embedding_model is empty"],
        )
        _BACKENDS["ollama"] = chat_stub  # type: ignore[assignment]
        _BACKENDS["mlc"] = emb_stub  # type: ignore[assignment]

        cfg = Config(chat_backend="ollama", embedding_backend="mlc")
        errors = cfg.validate()
        assert any("chat_model" in e for e in errors)
        assert any("embedding_model" in e for e in errors)

    # -- validate: general --

    def test_validate_empty_chat_backend_skips_delegation(self) -> None:
        cfg = Config(chat_backend="", embedding_backend="none")
        errors = cfg.validate()
        assert len(errors) == 1
        assert "chat_backend is not set" in errors[0]

    def test_validate_unknown_chat_backend_returns_early(self) -> None:
        cfg = Config(chat_backend="not-a-backend", embedding_backend="none")
        errors = cfg.validate()
        assert len(errors) == 1
        assert "not recognised" in errors[0]

    def test_validate_gateway_still_checked_after_delegation(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("mlc", validate_errors=[])
        _BACKENDS["mlc"] = stub  # type: ignore[assignment]

        cfg = Config(
            chat_backend="mlc",
            embedding_backend="none",
            gateway=GatewayConfig(port=99999),
        )
        errors = cfg.validate()
        assert any("gateway" in e and "port" in e for e in errors)

    def test_validate_gateway_checked_with_backend_errors(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("ollama", validate_errors=["[ollama] chat_model is empty"])
        _BACKENDS["ollama"] = stub  # type: ignore[assignment]

        cfg = Config(
            chat_backend="ollama",
            embedding_backend="none",
            gateway=GatewayConfig(port=0),
        )
        errors = cfg.validate()
        assert any("chat_model" in e for e in errors)
        assert any("gateway" in e for e in errors)

    # -- cloud vendors delegate directly using their key --

    def test_openai_delegates_to_openai_backend(self) -> None:
        from localmelo.support.backends.registry import _BACKENDS

        stub = _StubBackend("openai", validate_errors=[])
        _BACKENDS["openai"] = stub  # type: ignore[assignment]

        cfg = Config(chat_backend="openai", embedding_backend="none")
        errors = cfg.validate()
        assert errors == []
        assert stub.validate_called

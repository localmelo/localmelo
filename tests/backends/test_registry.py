"""Tests for the backend adapter contract and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from localmelo.support.backends.base import BaseBackend
from localmelo.support.backends.registry import (
    _clear,
    ensure_defaults_registered,
    get_backend,
    list_backends,
    register,
)

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config


# -- Fake backend for testing --


class FakeBackend(BaseBackend):
    """Concrete implementation of BaseBackend for testing."""

    def __init__(
        self,
        key: str = "fake",
        display_name: str = "Fake Backend",
    ) -> None:
        self._key = key
        self._display_name = display_name

    @property
    def key(self) -> str:
        return self._key

    @property
    def display_name(self) -> str:
        return self._display_name

    def validate(self, cfg: Config) -> list[str]:
        return []

    def validate_embedding(self, cfg: Config) -> list[str]:
        return []

    def has_embedding(self, cfg: Config) -> bool:
        return True

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        return object()  # type: ignore[return-value]

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None


# -- Fixtures --


@pytest.fixture(autouse=True)
def _clean_registry():  # type: ignore[no-untyped-def]
    """Ensure each test starts with a clean registry.

    Sets ``_DEFAULT_BACKENDS_REGISTERED = True`` so that
    ``ensure_defaults_registered()`` is a no-op -- this lets tests
    exercise the registry in isolation with only FakeBackends.
    Tests in ``TestEnsureDefaultsRegistered`` override this.
    """
    import localmelo.support.backends.registry as _reg

    _clear()
    _reg._DEFAULT_BACKENDS_REGISTERED = True
    yield
    # Restore: allow auto-registration for subsequent test modules.
    _clear()


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def fake_cfg() -> Config:
    from localmelo.support.config import Config

    return Config(chat_backend="fake")


# -- Registration and lookup --


class TestRegisterAndGet:
    def test_register_and_retrieve(self, fake_backend: FakeBackend) -> None:
        register(fake_backend)
        assert get_backend("fake") is fake_backend

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown backend 'nope'"):
            get_backend("nope")

    def test_get_unknown_shows_available(self, fake_backend: FakeBackend) -> None:
        register(fake_backend)
        with pytest.raises(KeyError, match="Available: fake"):
            get_backend("nope")

    def test_get_unknown_empty_registry(self) -> None:
        with pytest.raises(KeyError, match=r"Available: \(none\)"):
            get_backend("nope")

    def test_duplicate_registration_is_idempotent(
        self, fake_backend: FakeBackend
    ) -> None:
        register(fake_backend)
        register(fake_backend)  # should not raise
        assert len(list_backends()) == 1
        assert get_backend("fake") is fake_backend

    def test_register_multiple_backends(self) -> None:
        a = FakeBackend(key="alpha", display_name="Alpha")
        b = FakeBackend(key="beta", display_name="Beta")
        register(a)
        register(b)
        assert get_backend("alpha") is a
        assert get_backend("beta") is b


# -- list_backends --


class TestListBackends:
    def test_empty_registry(self) -> None:
        assert list_backends() == []

    def test_returns_registered_backends(self) -> None:
        a = FakeBackend(key="a", display_name="A")
        b = FakeBackend(key="b", display_name="B")
        register(a)
        register(b)
        result = list_backends()
        assert result == [a, b]

    def test_insertion_order_preserved(self) -> None:
        c = FakeBackend(key="c", display_name="C")
        a = FakeBackend(key="a", display_name="A")
        register(c)
        register(a)
        assert [b.key for b in list_backends()] == ["c", "a"]


# -- _clear --


class TestClear:
    def test_clear_resets_state(self, fake_backend: FakeBackend) -> None:
        import localmelo.support.backends.registry as _reg

        register(fake_backend)
        assert len(list_backends()) == 1
        _clear()
        # Suppress auto-registration to verify the clear actually emptied.
        _reg._DEFAULT_BACKENDS_REGISTERED = True
        assert list_backends() == []

    def test_clear_allows_re_registration(self, fake_backend: FakeBackend) -> None:
        register(fake_backend)
        _clear()
        register(fake_backend)
        assert get_backend("fake") is fake_backend


# -- ensure_defaults_registered --

_EXPECTED_KEYS = {
    "mlc",
    "ollama",
    "vllm",
    "sglang",
    "openai",
    "gemini",
    "anthropic",
    "nvidia",
}


class TestEnsureDefaultsRegistered:
    """Tests for lazy auto-registration of built-in backends."""

    @pytest.fixture(autouse=True)
    def _allow_auto_registration(self) -> None:  # noqa: PT004
        import localmelo.support.backends.registry as _reg

        _clear()
        _reg._DEFAULT_BACKENDS_REGISTERED = False

    def test_registers_builtin_backends(self) -> None:
        ensure_defaults_registered()
        backends = list_backends()
        keys = {b.key for b in backends}
        assert keys == _EXPECTED_KEYS
        assert len(backends) == 8

    def test_idempotent(self) -> None:
        ensure_defaults_registered()
        ensure_defaults_registered()  # must not raise
        backends = list_backends()
        keys = {b.key for b in backends}
        assert keys == _EXPECTED_KEYS
        assert len(backends) == 8

    def test_get_backend_auto_registers(self) -> None:
        """get_backend() works without explicit registration."""
        backend = get_backend("mlc")
        assert backend.key == "mlc"

    def test_old_keys_removed(self) -> None:
        """Old keys 'mlc-llm', 'cloud_api', 'online' no longer exist."""
        ensure_defaults_registered()
        for old_key in ("mlc-llm", "cloud_api", "online"):
            with pytest.raises(KeyError, match=f"Unknown backend '{old_key}'"):
                get_backend(old_key)

    def test_list_backends_auto_registers(self) -> None:
        """list_backends() works without explicit registration."""
        backends = list_backends()
        keys = {b.key for b in backends}
        assert keys == _EXPECTED_KEYS
        assert len(backends) == 8

    def test_clear_then_list_re_registers(self) -> None:
        """_clear() resets the flag; next list_backends re-registers."""
        ensure_defaults_registered()
        _clear()
        backends = list_backends()
        keys = {b.key for b in backends}
        assert keys == _EXPECTED_KEYS
        assert len(backends) == 8


# -- Contract satisfaction --


class TestBackendContract:
    """Verify FakeBackend satisfies all abstract methods with correct types."""

    def test_key_returns_str(self, fake_backend: FakeBackend) -> None:
        assert isinstance(fake_backend.key, str)
        assert fake_backend.key == "fake"

    def test_display_name_returns_str(self, fake_backend: FakeBackend) -> None:
        assert isinstance(fake_backend.display_name, str)
        assert fake_backend.display_name == "Fake Backend"

    def test_validate_returns_list_of_strings(
        self, fake_backend: FakeBackend, fake_cfg: Config
    ) -> None:
        errors = fake_backend.validate(fake_cfg)
        assert isinstance(errors, list)
        assert all(isinstance(e, str) for e in errors)

    def test_has_embedding_returns_bool(
        self, fake_backend: FakeBackend, fake_cfg: Config
    ) -> None:
        result = fake_backend.has_embedding(fake_cfg)
        assert isinstance(result, bool)

    def test_build_chat_provider_callable(
        self, fake_backend: FakeBackend, fake_cfg: Config
    ) -> None:
        provider = fake_backend.build_chat_provider(fake_cfg)
        assert provider is not None

    def test_build_embedding_provider_callable(
        self, fake_backend: FakeBackend, fake_cfg: Config
    ) -> None:
        result = fake_backend.build_embedding_provider(fake_cfg)
        assert result is None  # FakeBackend returns None


# -- ABC enforcement --


class TestABCEnforcement:
    def test_cannot_instantiate_base_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseBackend()  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self) -> None:
        class Incomplete(BaseBackend):
            @property
            def key(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]


# -- Package re-exports --


class TestPackageExports:
    def test_base_backend_importable_from_package(self) -> None:
        from localmelo.support.backends import BaseBackend as B

        assert B is BaseBackend

    def test_register_importable_from_package(self) -> None:
        from localmelo.support.backends import register as r

        assert r is register

    def test_get_backend_importable_from_package(self) -> None:
        from localmelo.support.backends import get_backend as g

        assert g is get_backend

    def test_list_backends_importable_from_package(self) -> None:
        from localmelo.support.backends import list_backends as lb

        assert lb is list_backends

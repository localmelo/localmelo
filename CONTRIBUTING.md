# Contributing to localmelo

Thank you for helping. This guide covers the development setup, the checks to
run, the project layout and how to send a change.

## Ways to contribute

- Report a bug or suggest a change in the
  [issues](https://github.com/localmelo/localmelo/issues).
- Fix or improve the documentation.
- Send a small, focused pull request. For a larger change, open an issue
  first so that we can agree on the approach. The roadmap is
  [#2](https://github.com/localmelo/localmelo/issues/2).

## Set up a development environment

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra dev --extra gateway
```

- Keep the directory name `localmelo`: the package is built from it.
- uv creates `.venv/` from the committed `uv.lock`, with Python 3.13 as
  pinned in `.python-version`. localmelo itself supports Python 3.11 or
  later.
- Optionally, run `uv run pre-commit install`. The hooks then run
  `ruff --fix`, `black`, `mypy` and the whole `pytest` suite on every commit.
  That suite includes `tests/smoke/test_backend_smoke.py`, which sends a real
  chat request when a model server answers on `127.0.0.1:11434` or
  `127.0.0.1:8400`.

## Run the checks

Run these before you send a pull request:

```bash
uv run pytest -q --ignore=tests/smoke/test_backend_smoke.py
uv run ruff check .
uv run black --check .
uv run mypy .
```

The first command runs the offline test suite, which needs no model server.
While you work, run the part you change:

```bash
uv run pytest tests/agent -q
```

| Directory | What it tests | Needs |
|---|---|---|
| `tests/agent` | The run loop, run options, results, reflection, tool exposure | `dev` |
| `tests/backends` | Backends, model providers, request timeouts, `extra_body` | `dev` |
| `tests/checker` | The checks | `dev` |
| `tests/cli` | The `melo` command | `dev` |
| `tests/executor` | The executor and the built-in tools | `dev` |
| `tests/memory` | Memory, SQLite persistence, BM25 | `dev` |
| `tests/mcp` | The MCP client and tool provider | `dev` |
| `tests/gateway` | Sessions and the HTTP API | `dev` and `gateway` |
| `tests/integration` | Several parts together | `dev` and `gateway` |
| `tests/smoke` | Offline tests of the smoke tools (parsing, metrics, logging), plus `test_backend_smoke.py`, which talks to a real model server | `dev` (and a model server for that file) |

### Real-backend smoke tests

Run these only with your own model server running:

```bash
uv run pytest -m smoke_backend tests/smoke/test_backend_smoke.py -q
```

They send a chat request to Ollama on `127.0.0.1:11434` and to MLC LLM on
`127.0.0.1:8400`, and skip the server that does not answer.
`tests/smoke/core_loop_test.py` runs longer memory scenarios against real
servers; see [tests/smoke/README.md](tests/smoke/README.md). That README's
default `pytest` command also collects `test_backend_smoke.py`; for an
offline run, use the `--ignore` command above. Results worth keeping go to
[tests_result/](tests_result/README.md).

## Project layout

```text
localmelo/
  __main__.py     the melo command
  melo/           the agent: run loop, checks, executor, memory, contracts
  support/        backends, model providers, gateway, MCP, config, setup wizard
  tests/          tests, mirroring melo/ and support/
  docs/           documentation and the website
  tests_result/   archived results of real-backend runs
```

- `melo/` holds the agent's behavior and the contracts that providers
  implement. It does not import `support/` at module level. The one bridge is
  `Agent(config=...)`, which builds the providers through `support/backends`
  when it is called.
- The directory that holds `pyproject.toml` is the `localmelo` package
  itself, so `melo/agent/agent.py` is imported as
  `localmelo.melo.agent.agent`. Do not create a `localmelo/localmelo/`
  directory.
- `melo/sleep/`, `melo/memory/personalized/` and `support/melo_serving/` hold
  code that the agent does not use yet.
- `PersonalizedSample` has a frozen v1 shape, guarded by
  `tests/memory/test_personalized_schema.py`. Any change to its fields must
  bump `PERSONALIZED_SAMPLE_SCHEMA_VERSION` in the same commit.

The [architecture page](https://localmelo.github.io/localmelo/en/architecture.html)
shows how the parts fit together.

## Add a backend

1. Write a class that implements `BaseBackend` (`support/backends/base.py`),
   in `support/backends/local/` or `support/backends/cloud/`.
2. Register it in `ensure_defaults_registered()` in
   `support/backends/registry.py`.
3. Add its key to `VALID_CHAT_BACKENDS` in `support/config.py`, and to
   `VALID_EMBEDDING_BACKENDS` if it serves embeddings.
4. Give it a config table: a field on `Config`, and its name in
   `_LOCAL_SECTIONS` or `_CLOUD_SECTIONS`, so that `load()` and `save()`
   handle it.
5. Add it to the setup wizard in `support/onboard.py`: `CHAT_BACKENDS`,
   `EMBEDDING_BACKENDS`, the defaults (`_LOCAL_DEFAULTS` or
   `_CLOUD_DEFAULTS`, and `_EMBEDDING_DEFAULTS`) and `_CHAT_SETUP`.
6. Add tests in `tests/backends/`.
7. Document it: the backend table and examples in
   [docs/configuration.md](docs/configuration.md#backends), and the README.

## Commits and pull requests

- Write the commit subject as `[Area] Imperative summary`, for example
  `[Agent] Sort the retrieved extra tools by name`. Areas include Agent,
  Executor, Memory, Docs, Tests and Maintenance.
- Pull request titles also name the type, as in `[Feature][Agent] …` or
  `[Bug][Agent] …`.
- In the body, say what changed and why.
- Keep one topic per pull request. Add tests for changes in behavior.
- Update the documentation in the same pull request, including the Chinese
  translations (`README.zh-CN.md` and `docs/zh-CN/`).
- Write code, comments and documentation in English.
- Never commit secrets, absolute paths from your machine, or personal data.

## Documentation

[docs/README.md](docs/README.md#maintaining-the-docs) explains where the pages
live and how to change them. To preview the website:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory docs
```

Then open `http://127.0.0.1:8080/en/`.

## Report a bug

Open an [issue](https://github.com/localmelo/localmelo/issues) and include:

- the JSON line from `--output-format json`, which names the localmelo
  version;
- the exit code;
- the backend and model;
- your operating system and Python version;
- the steps to reproduce.

Remove API keys and personal paths first. Report security problems as
described in [SECURITY.md](SECURITY.md), not in a public issue.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).

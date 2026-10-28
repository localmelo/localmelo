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
  The suite includes `tests/smoke/test_backend_smoke.py`, whose real-backend
  tests are skipped unless `LOCALMELO_SMOKE=1` is set, so a plain run opens
  no socket and sends no chat request.

## Run the checks

Run these before you send a pull request:

```bash
uv run pytest -n auto -q
uv run ruff check .
uv run black --check .
uv run mypy .
```

The first command runs the offline test suite once, in parallel with
`pytest-xdist`; use it before a pull request. It needs no model server and
sends no chat request. While you work, run only the tests a change affects
with `pytest-testmon`:

```bash
uv run pytest --testmon -q
```

Its first run tests everything and records what each test covers; later runs
test only the tests that cover the changed files. To run one directory
directly, use `uv run pytest tests/agent -q`.

| Directory | What it tests | Needs |
|---|---|---|
| `tests/agent` | The run loop, run options, results, reflection, tool exposure, events, cancellation and approvals | `dev` |
| `tests/backends` | Backends, model providers, request timeouts, `extra_body` | `dev` |
| `tests/checker` | The checks, secret redaction and the sanitizer | `dev` |
| `tests/cli` | The `melo` command, the `melo ""` loop and `melo-retrieve` | `dev` |
| `tests/contracts` | The event and approval contracts | `dev` |
| `tests/executor` | The executor, the built-in tools, approvals and the workspace root | `dev` |
| `tests/memory` | Memory, SQLite persistence, BM25, memory notes and the retrievers | `dev` |
| `tests/mcp` | The MCP client and tool provider | `dev` |
| `tests/scheduler` | The scheduling policy, the simulator and the trace importer | `dev` |
| `tests/gateway` | Sessions, the HTTP API, the `Host` and `Origin` checks and the web workspace; the `.mjs` files test the workspace's JavaScript | `dev` and `gateway`; Node for the `.mjs` files |
| `tests/integration` | Several parts together | `dev` and `gateway` |
| `tests/smoke` | Offline tests of the smoke tools (parsing, metrics, logging), plus the opt-in `test_backend_smoke.py`, which talks to a real model server | `dev` (and a model server for that file) |

The web workspace's JavaScript has its own tests. They need Node (the version
in `.nvmrc`), and pytest does not run them:

```bash
node --test tests/gateway/test_markdown.mjs tests/gateway/test_workspace.mjs
```

### Real-backend smoke tests

Run these only with your own model server running and the opt-in switch:

```bash
LOCALMELO_SMOKE=1 uv run pytest -m smoke_backend tests/smoke/test_backend_smoke.py -q
```

They send a chat request to Ollama on `127.0.0.1:11434` and to MLC LLM on
`127.0.0.1:8400`, and skip the server that does not answer.
`tests/smoke/core_loop_test.py` runs longer memory scenarios against real
servers; see [tests/smoke/README.md](tests/smoke/README.md).
`--ignore=tests/smoke/test_backend_smoke.py` still works for a bench harness,
but a plain run no longer needs it. Results worth keeping go to
[tests_result/](tests_result/README.md).

## Project layout

```text
localmelo/
  __main__.py     the melo command
  melo/           the agent: run loop, checks, executor, memory, contracts
  support/        backends, model providers, gateway, MCP, config, setup
                  wizard, the melo-retrieve command and the scheduler
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
  code that the agent does not use yet. `melo/memory/notes.py` and
  `melo/memory/retrieval/` are used only by `melo-retrieve`
  (`support/retrieve.py`; see [Offline retrieval](docs/retrieval.md)), and
  `support/scheduler/` only by its tests (see [Scheduler](docs/scheduler.md)).
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

[Maintaining the docs](docs/maintaining.md) explains where the pages live,
how to change and translate them, and the translation glossary. To preview
the website:

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

# Contributing to localmelo

Thank you for helping. This page has the short version: the setup, the checks
to run and the pull request checklist. The
[developer guide](https://localmelo.github.io/localmelo/en/developers.html)
is the full one: the repository layout, the test directories, and how to add a
backend, a built-in tool or a check.

## Ways to contribute

- Report a bug or suggest a change in the
  [issues](https://github.com/localmelo/localmelo/issues).
- Fix or improve the documentation.
- Send a small, focused pull request. For a larger change, open an issue
  first so that we can agree on the approach. The roadmap is
  [#2](https://github.com/localmelo/localmelo/issues/2).

## Set up

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo                 # keep this name: the package is built from it
uv sync --extra dev --extra gateway
```

Optionally, `uv run pre-commit install` makes the hooks run the checks below on
every commit.

## Run the checks

```bash
uv run pytest -n auto -q
uv run ruff check .
uv run black --check .
uv run mypy .
```

The first command runs the whole offline suite in parallel; it needs no model
server and sends no request to one (a few CLI tests serve a fake API on
loopback). While you work, run only the tests a change affects with
`uv run pytest --testmon -q`.

Every test runs with `HOME` in a temporary directory. A test that resolves the
real home, or that writes under the real `~/.cache/localmelo`, fails instead of
touching your config, memory, sessions or call logs. Writing the real cache is
never allowed; a test that genuinely needs the real home is marked
`@pytest.mark.allow_real_cache`.

## Pull request checklist

- [ ] One focused change, with tests for a behaviour change.
- [ ] The docs updated: the English page and its Chinese twin
      (`README.zh-CN.md`, `docs/zh-CN/`), plus `docs/for-agents.md` and
      `docs/llms.txt` when agent-facing behaviour changes.
- [ ] A `CHANGELOG.md` entry under Unreleased, with the pull request number.
- [ ] Screenshots for a change to the terminal UI or the website.
- [ ] Code, comments, tests and documentation in English.

The [developer guide](https://localmelo.github.io/localmelo/en/developers.html)
explains each item, and how to add a backend, a built-in tool or a check.

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

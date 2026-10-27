# localmelo documentation

The documentation for localmelo, a local personal agent that runs tasks with
tools and memory on your own model server or a cloud API. The website is at
<https://localmelo.github.io/localmelo/>.

## Start here

- [Quickstart](quickstart.md): install localmelo, run a first task, save a
  configuration and start the gateway.
- [localmelo for agents](for-agents.md): one compact page for AI coding
  agents and LLM tools, indexed by [llms.txt](llms.txt). People can skip it.

## Guides

- [Configuration](configuration.md): the setup wizard, the config file,
  backends, API keys, timeouts and environment variables, and how to update
  or uninstall localmelo.
- [Tools and MCP](tools.md): the built-in tools, MCP servers, which tools a
  step offers, and what the checks do.
- [Gateway and HTTP API](gateway.md): the HTTP API, sessions, the web
  workspace, and running the gateway at login on macOS.
- [Memory](memory.md): what localmelo remembers, and how to keep, inspect and
  reset it.
- [Troubleshooting](troubleshooting.md): error messages and how to fix them.

## Reference

- [Command-line reference](cli.md): modes, options, output, exit codes and
  signals.
- [Run results](run-results.md): statuses, the run record, and retries.
- [Python API](python-api.md): `Agent`, `AgentOptions` and `RunResult`,
  stopping a run, approval policies and the workspace root.
- [Events](events.md): the events a run reports to Python code, their order,
  fields and rules.
- [Sessions](sessions.md): conversations kept in SQLite, their turns, message
  ids, restarts and resume.
- [Offline retrieval](retrieval.md): `melo-retrieve`, which compares memory
  retrievers and exports a run as a corpus: its options, formats, output and
  exit codes.
- [Procedural memory](procedures.md): `melo-procedures`, which learns the
  steps of recurring tasks from recorded runs and replays the progress rules
  offline. Runs do not use it yet.

## Background

- [How localmelo works](how-it-works.md): the run loop, reflection, memory
  and timeouts.
- [Evaluations](evaluations.md): the controlled comparisons behind the
  defaults, such as the reflection gate, with their conditions and limits.
- [Scheduler](scheduler.md): the scheduling policy core and its simulator,
  for maintainers and the benchmark. Runs do not use it yet.
- [Architecture](https://localmelo.github.io/localmelo/en/architecture.html)
  (website): how the parts fit together.
- [Memory design](https://localmelo.github.io/localmelo/en/memory-design.html)
  (website): the maintainer's memory design, parts of which are not built
  yet.
- [Roadmap](https://localmelo.github.io/localmelo/en/updates.html#roadmap)
  (website): plans and dated updates.

## Project

- [Contributing](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md)
- [Maintaining the docs](maintaining.md): the layout, publishing,
  translation rules and glossary
- [Security policy](https://github.com/localmelo/localmelo/blob/main/SECURITY.md)
- [Changelog](https://github.com/localmelo/localmelo/blob/main/CHANGELOG.md)
- [License](https://github.com/localmelo/localmelo/blob/main/LICENSE)

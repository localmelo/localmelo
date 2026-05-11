# tests/smoke — running real-backend smoke

This directory holds two kinds of smoke checks:

1. **Pytest-discoverable smoke** — `test_*.py` files (chat round-trip, thinking
   split, logging dispatch, normalized token metrics). Auto-skipped when the
   local backend is unreachable.
2. **CLI smoke / benchmark** — `core_loop_test.py` is a data-driven script that
   seeds memories, runs scenario queries against a real chat + embedding
   backend, persists into SQLite, and emits per-backend evidence JSON plus a
   markdown comparison report.

The CLI smoke is the real-backend long-term retrieval evidence path for issue
\#4 v1 local backends. It is **not** pytest-collected (filename doesn't match
`test_*.py`) and is invoked manually.

> Memory architecture, retention semantics, and reset workflow are documented
> separately under `docs/` (see issue #4 items 3–4). This README is scoped to
> running the smoke.

## Default pytest run (offline, dev-safe)

```bash
python -m pytest tests/smoke -q
```

Collects:

- `test_backend_smoke.py` — chat round-trip for ollama and mlc. Skipped when
  the backend's `health_url` does not respond on a 0.5 s TCP probe.
- `test_thinking_split.py` — pure parsing tests, no network.
- `test_smoke_normalized.py` — aggregation tests, no network.
- `test_logging_llm_dispatch.py` — provider dispatch tests, no network.

None of these touch online services. `core_loop_test.py` is not collected.

## CLI smoke — `core_loop_test.py`

Three backends are pre-configured in `data/backends.json`:

| id       | name                       | embeddings via | requires |
|----------|----------------------------|----------------|----------|
| `ollama` | Ollama                     | local /v1      | local ollama on :11434 |
| `mlc`    | MLC-LLM                    | local /v1      | local mlc serve on :8400 |
| `online` | Online (OpenAI-compatible) | remote /v1     | `SMOKE_ONLINE_API_KEY` env var |

A backend whose `requires_api_key_env` is set but the env var is missing is
recorded as `skipped` **before** any network probe runs, so the default
suite stays offline.

### Per-backend commands

```bash
# Run from localmelo/

# Ollama only
python tests/smoke/core_loop_test.py --backends ollama

# MLC only
python tests/smoke/core_loop_test.py --backends mlc

# Online only (skipped unless SMOKE_ONLINE_API_KEY is set)
python tests/smoke/core_loop_test.py --backends online
SMOKE_ONLINE_API_KEY=sk-... python tests/smoke/core_loop_test.py --backends online

# All three (online auto-skipped if no key)
python tests/smoke/core_loop_test.py --backends all

# Single scenario across whichever backends you select
python tests/smoke/core_loop_test.py --backends online --scenarios personal_preference

# Re-render compare_test.md from existing JSON without re-running
python tests/smoke/core_loop_test.py --report-only
```

### Environment variables

| Var | Effect |
|---|---|
| `SMOKE_BACKENDS` | Default value for `--backends` (e.g. `ollama,mlc`). |
| `SMOKE_CHAT_URL` | Override the selected backend's `chat_url`. |
| `SMOKE_CHAT_MODEL` | Override the selected backend's `chat_model`. |
| `SMOKE_EMBED_URL` | Override the selected backend's `embed_url`. |
| `SMOKE_EMBED_MODEL` | Override the selected backend's `embed_model`. |
| `SMOKE_ONLINE_API_KEY` | API key used for the `online` backend's chat + embedding + health probe. Unset → online is skipped, never probed. |

### Online safety

- Never commit `SMOKE_ONLINE_API_KEY` — export it in your shell for ad-hoc
  runs. `data/backends.json` stores only the env-var **name**, never the
  value.
- Each scenario calls both chat completion and embedding endpoints multiple
  times. Pick the cheapest models you can (defaults: `gpt-4o-mini` +
  `text-embedding-3-small`) and consider `--scenarios <id>` to limit cost.

## Evidence captured per scenario (in `output/{backend}_test.json`)

- `status` — `completed` / `skipped` / `failed`
- `status_reason` — present on `skipped` / `failed`
- `backend_id`, `backend` (display name)
- `chat_url`, `chat_model`, `embed_url`, `embed_model`
- `mem_dir` — temp directory used for that run
- `seed_memories` — the exact list seeded
- `queries[]` — per-query `text`, `answer`, `thinking`, `expected_keywords`,
  `keywords_found`, `recall_score`, `query_elapsed_ms`
- `metrics` — embedding / chat / normalized / combined latency + token totals
- `sqlite.history_db`, `sqlite.long_term_db` — on-disk persistence paths
- `sqlite.tasks_rows`, `sqlite.steps_rows`, `sqlite.long_term_rows` —
  post-run row counts read back through the public async API

This is sufficient evidence for issue #4 item 1: it identifies which backend
was used, where memory persisted, how many seeds went in, and what recall the
queries achieved.

## Output layout

```
tests/smoke/output/
  ollama_test.json        # per-backend evidence
  mlc_test.json
  online_test.json
  compare_test.md         # cross-backend markdown report
```

Output under `tests/smoke/output/` is transient and ignored by git. When a run
becomes a meaningful baseline, promote it into `tests_result/` using the
archive layout documented in `tests_result/README.md`.

For example, the 2026-05-10 Track 2 v1 local-backend capture is stored at:

```
tests_result/backend-smoke/2026-05-10-qwen3-embedding-0.6b/
  summary.md
  manifest.json
  raw/
```

# Test result archive

This directory stores curated, reviewable test evidence that should survive
past an individual local run.

The archive is intentionally separate from runner output directories such as
`tests/smoke/output/`. Runner output is scratch space; `tests_result/` is the
place for results that are referenced by issues, releases, docs, or later
regression comparisons.

## Layout

```text
tests_result/
  <test-family>/
    README.md
    <yyyy-mm-dd>-<short-scope>/
      summary.md       # human-readable result and interpretation
      manifest.json    # machine-readable run metadata and artifact hashes
      raw/             # original runner outputs for audit/debug
```

Use a new dated run directory for each meaningful benchmark capture. Do not
overwrite old evidence unless the old files were committed by mistake.

## Storage policy

- Keep `summary.md` short and stable enough to link from docs and issues.
- Keep `manifest.json` structured so future tooling can index results.
- Put large or verbose runner output under `raw/`.
- Do not place transient reruns here unless they are intended to be cited.
- If a future run becomes too large for git, keep the summary and manifest here
  and put the heavy artifact location in the manifest.

## Add a run

1. Produce the results. For memory scenarios against real model servers, run
   `tests/smoke/core_loop_test.py` as described in
   [tests/smoke/README.md](../tests/smoke/README.md). It writes to
   `tests/smoke/output/`, which git ignores.
2. Create `<test-family>/<yyyy-mm-dd>-<short-scope>/` and copy the runner's
   output into its `raw/` folder.
3. Write `summary.md`: what was run, the results, and how to read them.
4. Write `manifest.json` with the fields below. On macOS,
   `shasum -a 256 raw/*` prints each file's SHA-256 digest and `wc -c raw/*`
   its size in bytes.
5. Add a row for the run to the family's `README.md`.
6. Scrub the files, as described below, before you commit.

## Manifest fields

These fields come from
[the first archived run](backend-smoke/2026-05-10-qwen3-embedding-0.6b/manifest.json).
Keep them in every manifest.

| Field | Meaning |
|---|---|
| `schema_version` | The manifest format: `tests-result.v1`. |
| `run_id` | The run's directory name. |
| `kind` | The test family, such as `backend-smoke`. |
| `captured_at` | The date of the run, as `YYYY-MM-DD`. |
| `environment` | An object that describes where the run happened, such as `requires_real_local_servers` and `sandbox_safe`. |
| `backends[]` | One entry per backend: `id`, `name`, `chat_model`, `embedding_model`, `runner` (the command that produced the results, as text) and `scenarios[]`. |
| `backends[].scenarios[]` | One entry per scenario: `id`, `score_percent`, `wall_time_seconds` and `status`. |
| `artifacts[]` | One entry per file in `raw/`: `path`, `sha256` and `bytes`. |
| `notes[]` | Notes for the reader, as text. |

A family may add fields of its own. The `backend-smoke` run also records
`issue`, `embedding_model_family`, `timing_scope` and
`online_backend_in_scope`.

## What to record

Record in `summary.md`, and in `environment` where it fits:

- the operating system and its build (`sw_vers` on macOS);
- the hardware: chip and memory;
- the Python version;
- the model server and its version;
- each model and its quantization;
- the localmelo commit, from `git describe --tags --always --dirty`.

## Scrub before committing

The repository is public. Before you commit a run, remove or replace:

- absolute paths that contain your home directory or user name: write `~` or
  `<scratch>` instead;
- user names, host names and e-mail addresses;
- API keys and tokens;
- personal details in seed data, prompts or answers.

# Test Result Archive

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

## Storage Policy

- Keep `summary.md` short and stable enough to link from docs and issues.
- Keep `manifest.json` structured so future tooling can index results.
- Put large or verbose runner output under `raw/`.
- Do not place transient reruns here unless they are intended to be cited.
- If a future run becomes too large for git, keep the summary and manifest here
  and put the heavy artifact location in the manifest.

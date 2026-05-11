# Backend Smoke: Qwen3-Embedding-0.6B

Run date: 2026-05-10

Issue: [#4 Track 2: Memory System](https://github.com/localmelo/localmelo/issues/4)

Purpose: verify persistent long-term retrieval against real local backends.

All tested backends used the Qwen3 0.6B embedding family. The `online` backend
was intentionally out of scope for this capture.

## Results

| Backend | Chat model | Embedding model | Personal | Cross-session | Project dev | GitHub tracking | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| OMLX | `Qwen3-4B` | `Qwen3-Embedding-0.6B` | 92% / 193s | 100% / 164s | 65% / 298s | 73% / 385s | 4/4 completed |
| Ollama | `qwen3:4b` | `qwen3-embedding:0.6b` | 96% / 180s | 100% / 120s | 94% / 208s | 97% / 298s | 4/4 completed |
| MLC-LLM | `qwen3-4b` | `qwen3-embedding` served from `Qwen3-Embedding-0.6B-q0f16-MLC` | 100% / 94s | 100% / 76s | 94% / 138s | 88% / 156s | 4/4 completed |

## Interpretation

- All three local backends completed all four long-term retrieval scenarios.
- MLC-LLM had the fastest wall-clock time in this capture.
- OMLX completed the run with the same embedding model but lower project/GitHub
  scenario scores, so it should be kept as supported evidence rather than the
  current performance baseline.
- The MLC run used a real compiled artifact for
  `Qwen3-Embedding-0.6B-q0f16-MLC`; no symlink or temporary model alias was used.

## Artifacts

Raw outputs are preserved under [`raw/`](raw/):

- `omlx_test.json`
- `ollama_test.json`
- `mlc_test.json`
- `compare.md`

The machine-readable run manifest is [`manifest.json`](manifest.json).

## Reproduction Notes

The runs were executed in the `mlsys` conda environment and required real local
serving processes outside the sandbox for Metal-backed backends.

Ollama and MLC-LLM used `tests/smoke/core_loop_test.py`. OMLX was run through
the local OpenAI-compatible experiment wrapper until OMLX is wired into
`tests/smoke/data/backends.json`.

```bash
# Ollama
SMOKE_EMBED_MODEL=qwen3-embedding:0.6b \
  python tests/smoke/core_loop_test.py --backends ollama

# MLC-LLM
SMOKE_EMBED_URL=http://127.0.0.1:8401/v1 \
SMOKE_EMBED_MODEL=qwen3-embedding \
  python tests/smoke/core_loop_test.py --backends mlc
```

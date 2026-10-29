# Guide: Offline retrieval

This guide shows how to evaluate memory retrievers offline: run a first evaluation, compare retrievers on a model server, and export runs and history for analysis. Read it before you compare retrieval settings; the options and output fields are in the [offline retrieval reference](../reference/retrieval.md).

## Run a first evaluation

`bm25` needs no server. Write a corpus of three documents and two labelled
queries:

```bash
mkdir retrieval-demo
cd retrieval-demo
cat > corpus.jsonl <<'EOF'
{"doc_id": "demo:n1", "kind": "note", "text": "plan: Let me read list_ops.py to find the stubs.\nfound: append(list1, list2) must return a new list; foldr is still a stub", "meta": {"task_id": "demo", "step": 1}}
{"doc_id": "demo:d2", "kind": "draft", "text": "def append(list1, list2):\n    return list1 + list2", "meta": {"task_id": "demo", "step": 2}}
{"doc_id": "demo:t2.1", "kind": "thought", "text": "foldr folds from the right, so the accumulator starts with the last item.", "meta": {"task_id": "demo", "step": 2}}
EOF
cat > queries.jsonl <<'EOF'
{"qid": "q1", "query": "how append joins list1 and list2", "scope": {"task_id": "demo"}, "relevant": ["d2"]}
{"qid": "q2", "query": "foldr accumulator order", "relevant": ["demo:t2.1"]}
EOF
```

The corpus holds a note, a code draft and a thought chunk of one task,
`demo`. Each query lists the documents that answer it in `relevant`. `q1`
can write `d2` for `demo:d2` because its `scope` names the task.

Evaluate `bm25` on them:

```bash
melo-retrieve --retriever bm25 --corpus corpus.jsonl --queries queries.jsonl
```

Example output (values vary):

```json
{"type": "query", "qid": "q1", "source": "", "candidates": 3, "relevant": ["demo:d2"], "hits": [{"doc_id": "demo:d2", "score": 2.1301, "rank": 1}, {"doc_id": "demo:n1", "score": 1.0976, "rank": 2}], "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "rr": 1.0, "ms_cold": 0, "ms_warm": [0, 0, 0], "ms_interleaved": null, "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0}, "usage": {}}
{"type": "query", "qid": "q2", "source": "", "candidates": 3, "relevant": ["demo:t2.1"], "hits": [{"doc_id": "demo:t2.1", "score": 1.5471, "rank": 1}], "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "rr": 1.0, "ms_cold": 0, "ms_warm": [0, 0, 0], "ms_interleaved": null, "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0}, "usage": {}}
{"type": "summary", "retriever": "bm25", "model": null, "dim": null, "n": 2, "calls": 8, "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": 1.0, "ms_cold_p50": 0, "ms_warm_p50": 0, "ms_warm_p95": 0, "ms_interleaved_p50": null, "ms_interleaved_p95": null, "prompt_tokens_mean": null, "cached_tokens_mean": null, "completion_tokens_mean": null, "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0}}
```

- stdout gets one line per query, then a summary line.
- `q1` found its relevant document, `demo:d2`, at rank 1, so its `recall@1`
  and `rr` (reciprocal rank) are 1.0. The note `demo:n1` matched too, at
  rank 2.
- Each query ran one cold call and three warm calls, 8 `calls` in all. Their
  times, in milliseconds, are in the `ms_` keys.
- The summary's `recall@<k>` and `mrr` (mean reciprocal rank) are the means
  over the queries.

[Output](../reference/retrieval.md#output) describes every key.

## Compare retrievers on a model server

`llm` and `embedding` need an OpenAI-compatible server. Run them on the same
files and compare their summary lines with the one from `bm25`. These
examples use vLLM for chat and Ollama for embeddings; use your own server's
address and models.

```bash
melo-retrieve --retriever llm --corpus corpus.jsonl --queries queries.jsonl \
  --base-url http://127.0.0.1:8000/v1 --chat-model Qwen/Qwen3-8B \
  --retrieval-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
melo-retrieve --retriever embedding --corpus corpus.jsonl --queries queries.jsonl \
  --embedding-base-url http://127.0.0.1:11434/v1 --embedding-model nomic-embed-text
```

- `--retrieval-extra-body` adds fields to every retrieval request. On a
  server that accepts `chat_template_kwargs`, such as vLLM, this one turns
  thinking off for models like Qwen3, so the model answers without
  reasoning first.
- With Ollama, pull the embedding model first: `ollama pull nomic-embed-text`.
- Read accuracy in `recall@<k>` and `mrr`, and speed in the `ms_` figures.
  `fallbacks` counts the calls that `bm25` answered instead; see
  [Warnings](../reference/retrieval.md#warnings).

### Measure latency between other requests

A server that caches prompts answers a repeated request faster, so calls
back to back can look faster than calls between other work. Each query runs
one cold call, then `--repeat` warm calls back to back (default: `3`). With
`--interleave`, each query then runs `--repeat` interleaved calls, and sends
the request in a file to `--base-url` before each one, as an agent sends its
own requests in between:

```bash
cat > request.json <<'EOF'
{"messages": [{"role": "system", "content": "You are a coding agent. Use the tools to edit files."}, {"role": "user", "content": "Implement foldr in list_ops.py."}], "max_tokens": 1}
EOF
melo-retrieve --retriever llm --corpus corpus.jsonl --queries queries.jsonl \
  --base-url http://127.0.0.1:8000/v1 --chat-model Qwen/Qwen3-8B \
  --retrieval-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --interleave request.json
```

- The time of the interleave request is not counted. The interleaved calls'
  times are in `ms_interleaved`, `ms_interleaved_p50` and
  `ms_interleaved_p95`.
- For figures close to an agent's, make the request look like its planning
  requests: the same system prompt and tools, a similar length, and no
  `max_tokens` (see [Interleave request](../reference/retrieval.md#interleave-request)).
- A cold call runs on a new retriever, with nothing cached on the client: a
  cold `embedding` call embeds every candidate document. The server may
  still hold what earlier queries sent.

## Export a run

`--export-run` turns the model calls of a run into corpus lines on stdout,
ready for `--corpus`. With the raw calls of a short task:

```bash
cat > calls.jsonl <<'EOF'
{"task_id": "demo", "seq": 0, "step": 1, "thinking": "Let me read list_ops.py to find the stubs.", "tool_calls": [{"tool_name": "file_read", "arguments": {"path": "list_ops.py"}, "id": "call_1"}], "tool_results": [{"output": "def append(list1, list2):\n    pass\n"}], "timestamp": 1761200000.0}
{"task_id": "demo", "seq": 1, "step": 2, "thinking": "append must return a new list, so concatenation is enough. Now I'll write it to list_ops.py:\n```python\ndef append(list1, list2):\n    return list1 + list2\n```", "tool_calls": [{"tool_name": "file_write", "arguments": {"path": "list_ops.py", "content": "def append(list1, list2):\n    return list1 + list2\n"}, "id": "call_2"}], "tool_results": [{"output": "Wrote 51 characters to list_ops.py"}], "timestamp": 1761200010.0}
{"task_id": "demo", "seq": 2, "kind": "reflect", "step": 2, "content": "{\"recommended_action\": \"done\", \"summary\": \"append returns a new list, so the task is done.\"}", "timestamp": 1761200020.0}
EOF
melo-retrieve --export-run calls.jsonl > run-corpus.jsonl
```

Output on stderr:

```text
melo-retrieve: exported 6 documents (2 notes, 1 reflections, 1 drafts, 2 thought chunks) from 1 tasks
```

`run-corpus.jsonl` holds `demo:n0`, `demo:t0.1`, `demo:n1`, `demo:d1`,
`demo:t1.1` and `demo:r2`, in this order. The note of the second call:

```json
{"doc_id": "demo:n1", "kind": "note", "title": "Now I'll write it to list_ops.py: → write list_ops.py", "text": "plan: Now I'll write it to list_ops.py: → write list_ops.py\nfound: append must return a new list, so concatenation is enough.\nobserved: wrote list_ops.py: 51 chars\noutcome: ok", "meta": {"task_id": "demo", "step": 2, "attempt": 1, "seq": 1, "created_at": 1761200010.0, "source": "heuristic", "note_kind": "step", "remember": true, "remember_by": "heuristic"}}
```

### Documents of a run

Tasks come in the order of their first call, and the calls of a task in
`seq` order. Each call gives these documents:

| Call `kind` | Documents |
|---|---|
| `plan` | A note, `<task_id>:n<seq>`; a draft, `<task_id>:d<seq>`, when the reasoning or the reply holds code blocks; thought chunks, `<task_id>:t<seq>.<i>` with i from 1. |
| `reflect` | A reflection, `<task_id>:r<seq>`, when the reply holds one (a JSON object whose `recommended_action` is `continue`, `stop`, `decompose` or `done`, as the agent reads it), and thought chunks. A reply without a reflection gives only its thought chunks. |
| `retrieve`, `estimate` | None. |

- **Notes** say what a reply planned, found, decided, suspected and left
  open, in the lines `plan:`, `found:`, `decided:`, `hypotheses:` and
  `open:`, and what its tool calls returned, in `observed:` and `outcome:`.
  When the reply's text holds a `<note>` block, the note takes those lines
  from it (`meta.source` is `model`). Otherwise it picks sentences of the
  reasoning and the reply by marker words, in English and Chinese
  (`heuristic`).
- **Reflections** are built from the reflection's own fields, as the agent
  reads them, not from the reply's text: `plan:` is the recommended action
  and the summary, `found:` the useful evidence, `decided:` the next
  promising directions (`next: …`), and `hypotheses:` the failed hypotheses
  (`refuted: …`) and what to avoid (`avoid: …`). `meta.source` is `model`.
- **Drafts** hold the code blocks of the reasoning and of the reply.
  `meta.target` is the last file named before the first block, or else the
  file of the task's latest read or write. `meta.status` is `written` once a
  write to that file holds at least 60% of the draft's lines, `superseded`
  when a later draft has the same target, and `pending` otherwise.
- **Thought chunks** hold the reasoning without its code blocks, in pieces
  of up to 1,500 characters, split at paragraphs and sentences.
- Every text and title is made safe to show to a model as plain text: role
  tokens such as `<|im_start|>` and tags such as `<tool_call>` are escaped,
  and `<think>` tags are removed. A document left without text is skipped.
- `meta` holds `task_id`, `step`, `attempt`, `seq` and `created_at` (the
  call's `timestamp`). The other keys depend on the kind:
  - notes and reflections: `source` (`model` or `heuristic`; always `model`
    for a reflection), `note_kind` (`step`, `answer` or `empty` from the
    call's outcome, or `reflection`), and `remember` with `remember_by`.
    `remember` says whether a note looks worth keeping: the block's
    `remember:` line when it has one (`model`), or else `true` when the note
    has a `found:` or `decided:` line (`heuristic`); a reflection has
    `false` and `""`;
  - drafts: `target`, `status`, `lines` (non-blank lines), `blocks`, and
    `cut`, which is `true` when code was left out to keep the draft within
    8,000 characters;
  - thought chunks: `chunk`, the chunk's number.

The ids follow `seq`, so you can write labelled queries from your own record
of the calls before you export them.

### Export a history database

`--export-run` also reads the `history.db` of a run: give the memory
directory that holds it (the `--memory-dir` or `LOCALMELO_MEMORY_DIR` of the
run), or the file itself. `melo` keeps it in `~/.cache/localmelo/memory`
when you [keep memory between runs](memory.md#keep-memory-between-runs). A
file is read as a database when it starts with the SQLite header, and as raw
calls otherwise.

The export reads the model calls from the database's `llm_calls` table,
which `melo` writes whenever it keeps memory (see
[Model calls](../reference/memory.md#model-calls)):

```bash
LOCALMELO_PERSIST_MEMORY=1 melo "List the files"
melo-retrieve --export-run ~/.cache/localmelo/memory > run-corpus.jsonl
```

```text
melo-retrieve: exported 4 documents (2 notes, 0 reflections, 0 drafts, 2 thought chunks) from 1 tasks
```

The numbers depend on the run. Here the model listed the directory and then
answered, with reasoning in both replies, so each of the two calls gave a
note and a thought chunk.

- A `history.db` that no version with model calls has opened has no such
  table, and its export stops with exit code 2:

  ```text
  melo-retrieve: error: <path> has no llm_calls table: it was written before localmelo recorded model calls. Export the calls as raw-call JSONL instead (see docs/retrieval.md).
  ```

  `<path>` is the path of the database. To export such a run, write its
  model calls as [raw calls](../reference/retrieval.md#raw-calls) from a record made outside
  localmelo, such as a proxy's log of the requests and replies.
- A `history.db` from an earlier version that this version then opened gets
  an empty table: its earlier tasks export nothing, and the closing line
  counts only the tasks with model calls.
- Rows older than `LOCALMELO_THINKING_RETENTION_DAYS` (30 days by default)
  have no `thinking`, so they give no thought chunks.
- `content` and `thinking` are stored redacted, the tool calls' arguments
  are not. A draft whose code held a key may then no longer match its
  `file_write`, and a reflection whose JSON holds a bare 4-to-8-digit
  number near a code keyword may no longer parse.

A database that has the table is read as follows:

- The columns read are `task_id`, `seq`, `kind`, `step`, `attempt`,
  `outcome`, `content`, `thinking`, `tool_calls_json`, `step_ids_json` and
  `timestamp`, under the rules of raw calls. A `NULL` `step` or `attempt`
  reads as 0 or 1. A missing column stops the export:
  `melo-retrieve: error: <path>: llm_calls has no column <names>`.
- When `thinking` is not empty and `content` starts with `<think>`, a
  newline, the thinking, a newline, `</think>` and a blank line (the copy
  Ollama's provider adds), the content is read without that copy, so the
  reasoning is read once.
- Tool results come from the `steps` rows that `step_ids_json` lists,
  matched to the calls by their `id`. When a listed row is missing or
  matches no call, one warning goes to stderr:
  `melo-retrieve: warning: <n> step ids in llm_calls.step_ids_json had no matching steps row; their results are left out`.
- Tasks come in the order of their earliest `timestamp`.
- The export never writes to the database and creates no file next to it.
  If a `history.db-wal` file exists, as it does while a run is writing, the
  export reads through it, and SQLite may create `history.db-shm`. Export a
  run that has ended.

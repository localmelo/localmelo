# Offline retrieval

`melo-retrieve` compares memory retrievers offline: it runs labelled queries
against a corpus of documents and reports how often each retriever finds the
right ones, and how fast. It also turns a run's model calls into such a
corpus.

localmelo has three retrievers. Each one ranks documents, such as notes on
the model's reasoning, code drafts and chunks of its thinking, for a query:

| Retriever | How it ranks |
|---|---|
| `bm25` | Okapi BM25 text ranking, with no model. |
| `llm` | The chat model picks the documents from a catalog of them. |
| `embedding` | The cosine similarity of embedding vectors. |

`melo-retrieve` evaluates these retrievers offline; the compress modes use the same `bm25`, `llm` and `embedding` retrievers in the loop. `melo-retrieve` is
installed with localmelo, next to `melo`, and runs the same way: for example
`uv run melo-retrieve --help` inside the clone.

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

[Output](#output) describes every key.

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
  [Warnings](#warnings).

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
  `max_tokens` (see [Interleave request](#interleave-request)).
- A cold call runs on a new retriever, with nothing cached on the client: a
  cold `embedding` call embeds every candidate document. The server may
  still hold what earlier queries sent.

## Options

```text
melo-retrieve --retriever bm25|llm|embedding --corpus FILE --queries FILE [options]
melo-retrieve --export-run DIR|FILE
```

| Option | Default | Description |
|---|---|---|
| `-h`, `--help` | | Show the help and exit. |
| `--retriever bm25\|llm\|embedding` | not set | The retriever to evaluate. |
| `--corpus FILE` | not set | The documents, one per line; see [Corpus](#corpus). |
| `--queries FILE` | not set | The labelled queries, one per line; see [Labelled queries](#labelled-queries). |
| `--k K[,K...]` | `1,3,5` | The values of k for `recall@k`. Each search asks for the largest. |
| `--repeat N` | `3` | Warm calls per query, and interleaved calls with `--interleave`. |
| `--retrieval-timeout SECONDS` | `30` | How long an `llm` retrieval request may take before `bm25` answers instead. |
| `--interleave REQUEST.json` | not set | Send this chat request before each interleaved call; see [Interleave request](#interleave-request). |
| `--base-url URL` | not set | The OpenAI-compatible chat server, for `llm` and `--interleave`. |
| `--chat-model NAME` | `default` | The model to request from `--base-url`. |
| `--retrieval-extra-body JSON` | not set | A JSON object of fields to add to every `llm` retrieval request. Its `temperature`, if any, replaces the default 0. |
| `--embedding-base-url URL` | not set | The OpenAI-compatible embedding server, for `embedding`. |
| `--embedding-model NAME` | `default` | The model to request from `--embedding-base-url`. |
| `--export-run DIR\|FILE` | not set | Print a run's model calls as corpus lines; see [Export a run](#export-a-run). |
| `--export-store DIR\|FILE` | not set | Print the active entries of a typed `long_term.db`; see [The typed long-term store](#the-typed-long-term-store). |
| `--export-decisions DIR\|FILE` | not set | Print its decision log, in `seq` order. |
| `--route CANDIDATES.jsonl` | not set | Score synthetic candidates through the boundary and policy; needs `--memory-dir`. |
| `--memory-dir DIR` | not set | The scratch directory `--route` writes; never the user's memory directory. |

- An evaluation needs `--retriever`, `--corpus` and `--queries`.
  `--export-run` takes no other option.
- `--retriever llm` needs `--base-url`, `--retriever embedding` needs
  `--embedding-base-url`, and `--interleave` needs `--base-url`.
- `--retrieval-timeout` and `--retrieval-extra-body` work only with
  `--retriever llm`; `--embedding-base-url` and `--embedding-model` only
  with `--retriever embedding`; `--base-url` and `--chat-model` only with
  `--retriever llm` or `--interleave`.
- `--k`: positive integers separated by commas, such as `1,3,10`. Repeated
  values count once, and the `recall@<k>` keys come in increasing order of k.
- `--repeat 0` runs only the cold call of each query.
- `--retrieval-timeout`: a number of seconds above 0, and only with
  `--retriever llm`. A retrieval request that has not answered by then
  falls back to `bm25`. The run's HTTP timeout is 120 seconds or
  `--retrieval-timeout`, whichever is longer, and applies to the retrieval
  and interleave requests; an embedding request waits up to 120 seconds.
- `--retrieval-extra-body` must be a JSON object, and it cannot set `model`,
  `messages`, `tools`, `tool_choice` or `stream`. It does not apply to the
  interleave request.
- URLs are used as given, so include `/v1`, as in
  `http://127.0.0.1:8000/v1`.
- `melo-retrieve` sends no API key and reads no config file. To evaluate
  against a cloud API, use [Python](#from-python) with a provider that has
  your key.

A value that is not valid, a missing option, or options that do not go
together are usage errors: `melo-retrieve` prints its usage summary and one
error line on stderr, and exits with code 2.

```text
melo-retrieve: error: --retriever llm needs --base-url
```

## Input formats

The corpus, the labelled queries and the raw calls are JSON Lines files: one
JSON object per line, in UTF-8. Blank lines are skipped and unknown keys are
ignored. Any other problem stops the run with exit code 2 and one line that
names the file and the line:

```text
melo-retrieve: error: queries.jsonl:1: relevant id 'demo:d9' is not in the corpus
```

### Corpus

One document per line, the input of `--corpus` and the output of
`--export-run`:

```json
{"doc_id": "demo:n1", "kind": "note", "title": "Find the stubs", "text": "plan: Let me read list_ops.py to find the stubs.", "meta": {"task_id": "demo", "step": 1}}
```

| Key | Required | Value |
|---|---|---|
| `doc_id` | yes | The document's id: unique in the file, without whitespace. |
| `kind` | yes | `note`, `draft`, `thought`, `reflection`, `digest` or `long`. |
| `text` | yes | The text to search, not empty. |
| `title` | no | A one-line title. The `llm` catalog shows it, or else the first line of `text`. |
| `meta` | no | An object. Its `task_id` (a string) and `step` (an integer, 0 or more) decide which queries see the document. Other keys are kept as they are. |

The export writes notes, drafts, thought chunks and reflections; see
[Export a run](#export-a-run). `digest` (a summary of several notes) and
`long` (a long-term memory entry) are there for corpora you build yourself.

### Labelled queries

One query per line, the input of `--queries`:

```json
{"qid": "q3", "query": "how append joins list1 and list2", "scope": {"task_id": "demo", "before_step": 3, "exclude_ids": ["n1"]}, "relevant": ["d2"], "grades": {"d2": 2}, "source": "gold"}
```

| Key | Required | Value |
|---|---|---|
| `qid` | yes | The query's id, unique in the file. |
| `query` | yes | The text to search for, not blank. |
| `relevant` | yes | The ids of the documents that answer the query: a non-empty list. |
| `scope` | no | Which documents the query searches; see below. |
| `grades` | no | Graded relevance, an integer of 1 or more for ids in `relevant`. It is checked but not scored. |
| `source` | no | A label, such as `gold`, that the query's output line repeats. |

`scope` takes three keys:

- `task_id`: search only the documents whose `meta.task_id` is this.
  Without it, the query searches every document.
- `before_step`: search only the documents whose `meta.step` (0 when it is
  absent) is lower, which existed before that step. An integer of 1 or more;
  it needs `task_id`.
- `exclude_ids`: documents that the retriever must not return. They do not
  count as candidates, but `llm` keeps them in its catalog.

In `relevant`, `grades` and `exclude_ids`, an id without `:` is short for
`<task_id>:<id>`, with the `task_id` of the scope; without one, it is an
error. Every relevant id must be in the corpus and in the query's scope, and
none may be in `exclude_ids`.

### Raw calls

One model call of a run per line, an input of `--export-run`:

```json
{"task_id": "demo", "seq": 0, "kind": "plan", "step": 1, "attempt": 1, "content": "", "thinking": "Let me read list_ops.py to find the stubs.", "tool_calls": [{"tool_name": "file_read", "arguments": {"path": "list_ops.py"}, "id": "call_1"}], "tool_results": [{"tool_name": "file_read", "output": "def append(list1, list2):\n    pass\n", "error": "", "duration_ms": 3.1}], "outcome": "tools", "timestamp": 1761200000.0}
```

| Key | Default | Value |
|---|---|---|
| `task_id` | required | The task's id: not empty, without whitespace. |
| `seq` | required | The call's number in its task: an integer, 0 or more, unique per task. |
| `kind` | `plan` | `plan` (a planning call), `reflect` (a reflection call), or `retrieve` and `estimate` (other calls, which give no documents). |
| `step` | `0` | The run's step. |
| `attempt` | `1` | The run's attempt. |
| `content` | `""` | The text of the reply. |
| `thinking` | `""` | The reasoning of the reply. |
| `tool_calls` | `[]` | The reply's tool calls, each with `tool_name`, `arguments` and `id`. |
| `tool_results` | `[]` | Their results in the same order, each with `tool_name`, `output`, `error` and `duration_ms`. |
| `outcome` | derived | `tools`, `answer` or `empty` for a planning call, `aux` for the other kinds. |
| `timestamp` | `0.0` | When the call was made, in seconds since the epoch. |

- `step` and `attempt` may be `null`, which reads as the default.
- `arguments` is an object, or a string that holds one, as OpenAI request
  bodies carry it. Any other string is kept as
  `{"_raw_arguments": "<the string>"}`.
- A result's `tool_name` defaults to its call's. There are never more
  results than calls: a call without a result did not run.
- Without `outcome`, a planning call's is `tools` when it has tool calls,
  else `answer` when `content` is not blank, else `empty`.

### Interleave request

The file of `--interleave` holds one JSON object: the body of a chat
completions request, with a non-empty `messages` list.

```json
{"messages": [{"role": "system", "content": "You are a coding agent. Use the tools to edit files."}, {"role": "user", "content": "Implement foldr in list_ops.py."}], "max_tokens": 1}
```

- `melo-retrieve` sets `model` to `--chat-model` and `stream` to `false`,
  and posts the rest as it is to `<base-url>/chat/completions`. The body may
  hold `tools` and any other field your server accepts.
- `--retrieval-extra-body` does not apply to it.
- `max_tokens` keeps the reply short, but not with `tools`: a cap that cuts
  a tool call short can make the server fail the request (oMLX answers HTTP
  500 for an incomplete tool call), and the run then stops. With `tools`,
  leave `max_tokens` out; the request's time is not counted anyway.
- If the request fails, the run stops with exit code 1:

  ```text
  melo-retrieve: error: interleave request failed: POST /chat/completions failed: ConnectError: All connection attempts failed
  ```

## Output

stdout gets one JSON line per query, printed when the query is done, and
then one summary line. Warnings and errors go to stderr.

### Query lines

| Key | Value |
|---|---|
| `type` | `"query"` |
| `qid` | The query's `qid`. |
| `source` | The query's `source`, or `""`. |
| `candidates` | How many documents the query searched: those in its scope, without `exclude_ids`. |
| `relevant` | The relevant ids, in full. |
| `hits` | The hits of the cold call, best first, each with `doc_id`, `score` (to 4 decimal places) and `rank` (from 1). |
| `recall@<k>` | One key for each `--k` value: the share of the relevant ids among the first k hits. |
| `rr` | The reciprocal rank: 1 / the rank of the first relevant hit, or 0 without one. |
| `ms_cold` | The time of the cold call, in milliseconds. |
| `ms_warm` | The times of the warm calls, or `null` with `--repeat 0`. |
| `ms_interleaved` | The times of the interleaved calls, or `null` without `--interleave` or with `--repeat 0`. |
| `fallbacks` | How many of the query's calls `bm25` answered instead, by reason: `timeout`, `parse` and `provider_error`. |
| `usage` | The token counts that the server reported for the query's calls, summed by key; `{}` for `bm25` and `embedding`. |

- `recall@<k>` and `rr` are means over all the calls of the query: cold,
  warm and interleaved. They even out the answers of `llm`, which can vary
  from call to call. `hits` shows only the cold call, so recall worked out
  from `hits` alone can differ.
- `score` is the BM25 score for `bm25`, 1 / rank for `llm`, and the cosine
  similarity for `embedding`. After a fallback, it is the BM25 score.

### Summary line

| Key | Value |
|---|---|
| `type` | `"summary"` |
| `retriever` | The retriever evaluated, also when its calls fell back. |
| `model` | `--chat-model` for `llm`, `--embedding-model` for `embedding`, `null` for `bm25`. |
| `dim` | The length of the embedding vectors for `embedding`, else `null`. |
| `n` | The number of queries. |
| `calls` | The number of searches, over all queries. |
| `recall@<k>` | The mean of the queries' `recall@<k>`. |
| `mrr` | The mean of the queries' `rr`: the mean reciprocal rank. |
| `ms_cold_p50` | The median time of the cold calls. |
| `ms_warm_p50`, `ms_warm_p95` | The 50th and 95th percentiles of the warm calls' times. |
| `ms_interleaved_p50`, `ms_interleaved_p95` | The same for the interleaved calls. |
| `prompt_tokens_mean`, `cached_tokens_mean`, `completion_tokens_mean` | The mean token count per call, over the calls that reported it, to 1 decimal place. |
| `fallbacks` | The fallbacks of all calls, by reason. |

- Per query, `calls` counts 1 cold call, `--repeat` warm calls, and
  `--repeat` interleaved calls with `--interleave`.
- Each query weighs the same in the means. `recall@<k>`, `rr` and `mrr` are
  rounded to 4 decimal places.
- Times are whole milliseconds, measured around each search, fallback
  included. Percentiles are nearest-rank: the p-th percentile of n times is
  the ⌈p × n / 100⌉-th smallest.
- `null` marks a figure that does not apply: interleaved figures without
  `--interleave`, warm and interleaved figures with `--repeat 0`, and the
  token means for `bm25` and `embedding`.
- `cached_tokens_mean` is filled for `llm` when the server reports
  `prompt_tokens_details.cached_tokens`, and `null` otherwise.

### Warnings

When every call of an `llm` or `embedding` run fell back to `bm25`, one
warning follows the summary line on stderr, with the reasons and their
counts:

```text
melo-retrieve: warning: every llm call fell back to bm25 (provider_error 8); check --base-url
```

The figures are then those of `bm25`, and the exit code is still 0.
`provider_error` usually means that the server could not be reached,
`timeout` that it took longer than `--retrieval-timeout`, and `parse` that
the model did not answer with a list of ids.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The evaluation or the export finished, fallbacks included. |
| 1 | The interleave request failed. |
| 2 | A usage error, or an input error: a file that cannot be read, a line that breaks its format, or a database that cannot be exported. |
| 130 | Ctrl+C stopped the run. |

A usage error prints the usage summary and one error line; any other error
prints one line. There is no traceback. Lines already printed on stdout stay
whole, so a stopped run leaves complete query lines.

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
[Model calls](memory.md#model-calls)):

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
  model calls as [raw calls](#raw-calls) from a record made outside
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

## The typed long-term store

Three modes read or write the typed `long_term.db` of R6-lite. Each is
mutually exclusive with the evaluation flags other than the model ones and
with `--export-run`, and none contacts a model: `--base-url`, `--chat-model`,
`--embedding-base-url`, `--embedding-model`, `--retrieval-timeout` and
`--retrieval-extra-body` are accepted and ignored, so a bench script can pass
the same flags everywhere. The same store is what [recall in the loop](memory.md#recall-in-the-loop)
reads at `start`, and what `--export-store` shows the bench. `DIR` may be a memory directory (the file is
`long_term.db`) or the file itself.

```bash
melo-retrieve --export-store DIR|FILE
melo-retrieve --export-decisions DIR|FILE
melo-retrieve --route CANDIDATES.jsonl --memory-dir DIR
```

### `--export-store`

One JSON line per **active** entry, in `(created_at, id)` order:

```json
{"id": 1, "text": "I live in Berlin near the river", "kind": "profile",
 "source": "user", "status": "active", "admitted_by": "user_statement",
 "provenance": {"task_id": "t1", "source_ids": ["r1"]},
 "created_at": 0.0, "last_used_at": 0.0, "uses": 0, "supersedes": null,
 "links": [], "embed_model": "", "embed_dim": 0}
```

The bench's T4 reader reads `text` and `provenance.source_ids`. A count goes
to stderr. The file is opened read-only: a v0 file is read with its legacy
columns and the legacy derivation (a reflection by `metadata.type`, otherwise
episodic/step_summary) and is **not** migrated; a file from a newer version is
one line on stderr and exit 2; a missing file is the same.

### `--export-decisions`

One JSON line per decision row, in `seq` order:

```json
{"seq": 1, "candidate_id": "n1", "action": "NEW", "kind": "profile",
 "admitted_by": "user_statement", "entry_id": 1, "target_id": null,
 "reason": "new", "source_ids": ["r1"], "text_sha256": "…", "text": null,
 "timestamp": 0.0, "task_id": "t1"}
```

`source_ids` is the parsed array, and `text` is `null` unless the run had
`--keep-decision-text`. A v0 file has no decisions table and prints none.

### `--route`

Each input line is one candidate:

```json
{"candidate_id": "n1", "source": "user", "text": "…", "source_ids": ["r1"],
 "signal": "user_statement", "task_id": "t1", "session_id": "",
 "task_text": "plan the move"}
```

`source` is one of `reasoning_note`, `digest`, `reflection`, `user` or
`sleep`; `signal` is one of `rehearsal`, `user_request`, `user_statement`,
`outcome` or `sleep`; an unknown value is exit 2. The candidate goes through
the same boundary and policy as a run, into a text-only store under
`--memory-dir`, and one line comes out per input line, in input order:

```json
{"candidate_id": "n1", "action": "NEW", "kind": "profile",
 "admitted_by": "user_statement", "entry_id": 1, "target_id": null,
 "reason": "new", "source_ids": ["r1"]}
```

A boundary refusal prints `"action": "REFUSED"` and the reason. `--memory-dir`
is required and the command refuses the default `~/.cache/localmelo/memory`
and the `LOCALMELO_MEMORY_DIR` value (resolved paths); the directory is
created `0700`. `--route` appends to an existing scratch store rather than
clearing it, so the bench uses a fresh directory per run. `--export-store` on
that directory shows what the route wrote.

## How each retriever ranks

**`bm25`** scores the query's candidates with Okapi BM25, so how rare a word
is depends on those documents alone. Then:

- **Floor.** A hit must contain at least two distinct units of the query.
  The units are its words of two or more characters, and its pairs of
  adjacent Chinese, Japanese or Korean characters; an identifier such as
  `list_ops` counts once. File names and paths do not count, nor do common
  function words and the words of note lines, such as `plan`, `found`,
  `read` and `run`. A query with fewer than two units returns no hits.
- **Kinds.** Thought chunks keep at most one hit per task step, the best,
  and every other kind ranks above every thought chunk.
- **Relative cut.** Among the thought chunks, and among the other kinds,
  hits that score under 0.2 times the best one are dropped.

**`llm`** sends one chat request without tools: a fixed system prompt, then
a catalog of the documents, the aliases not to return, the query, and how
many ids to return. `melo-retrieve` sends it with `temperature` 0, so the
same model and files give the same picks; `--retrieval-extra-body` can set
another.

- Each catalog line reads `[<alias>] <label> · step <n> · <title>`, as in
  `[n1] note · step 1 · Find the stubs`. The alias is a letter for the kind
  (`n` note, `d` draft, `t` thought, `r` reflection, `g` digest, `m` long)
  and the document's position among the documents of that kind.
- Excluded documents keep their place in the catalog, so the catalog
  changes only when the documents do, and a server can reuse its prompt
  cache from one search to the next. Over 12,000 characters, the earliest
  lines are dropped, 3,000 characters at a time.
- The model answers with `{"ids": ["n2", "d1"]}`, best first. Text or a code
  fence around the JSON is accepted, and so is a bare list. Hits score
  1 / rank.
- On a timeout, an error from the server, or a reply without a list of ids,
  `bm25` answers instead, and the result names the reason: `timeout`,
  `provider_error` or `parse`. Nothing is retried.

**`embedding`** embeds each document once, 32 texts per request, and keeps
the vector; it embeds the query on every search. Documents rank by the
cosine similarity of their vector with the query's, and a negative
similarity is left out. An error from the server, or a reply with the wrong
number of vectors, a value that is not a finite number or vectors of another
length, makes `bm25` answer instead, with the reason `provider_error`.

## From Python

> [!NOTE]
> The Python interface of the retrievers is experimental and may change
> before v0.1.0.

`localmelo.melo.memory.retrieval` holds the retrievers, the evaluation and
the file formats. This script searches and evaluates the files of
[the first evaluation](#run-a-first-evaluation):

```python
import asyncio

from localmelo.melo.memory.retrieval import (
    build_retriever,
    evaluate,
    load_corpus,
    load_queries,
)


async def main() -> None:
    with open("corpus.jsonl", encoding="utf-8") as file:
        corpus = load_corpus(file)
    with open("queries.jsonl", encoding="utf-8") as file:
        queries = load_queries(file, corpus)

    # One search.
    retriever = build_retriever("bm25")
    result = await retriever.search("foldr accumulator order", corpus, k=3)
    for hit in result.hits:
        print(hit.rank, hit.doc_id, round(hit.score, 4))

    # The evaluation that melo-retrieve runs, with a new retriever per query.
    report = await evaluate(
        lambda: build_retriever("bm25"),
        corpus=corpus,
        queries=queries,
        ks=(1, 3),
    )
    print(report.recall, report.mrr)


asyncio.run(main())
```

Output:

```text
1 demo:t2.1 1.5471
{1: 1.0, 3: 1.0} 1.0
```

- `build_retriever(name, *, send=None, embedding=None, **options)` returns
  a new retriever with empty caches. `llm` needs `send`, and `embedding`
  needs `embedding`, an embedding provider. An unknown name or option, a
  missing `send` or provider, or a value out of range raises `ValueError`.
- `await retriever.search(query, docs, *, k, exclude=frozenset())` returns
  a `RetrievalResult`: `hits` (each a `Hit` with `doc_id`, `score` and
  `rank`), `retriever` (the one that answered), `ms`, `fallback` and
  `usage`. `docs` is a sequence of `Doc(doc_id, kind, text, title="",
  meta={})`.
- `await evaluate(retriever, *, corpus, queries, ks=(1, 3, 5), repeats=3,
  interleave=None, model=None, on_query=None)` returns an `EvalReport`.
  Pass a function that builds a retriever, as above, to start each query
  cold, or a retriever to share it across queries. `interleave` is an async
  function to await before each interleaved call, and `model` the name that
  the summary line reports. `on_query` gets each query's `QueryResult` as
  soon as it is done.
- `report.queries` holds each query's `QueryResult`.
  `report.summary_line()` and `QueryResult.to_line(ks)` return the
  [output lines](#output) as dicts.
- `load_corpus`, `load_queries`, `load_raw_calls`, `docs_from_calls` and
  `doc_to_dict` read and write the [formats](#input-formats) above. A bad
  line raises `CorpusError`, whose `line` is its 1-based line number.

`build_retriever` takes these options:

| Option | Retrievers | Default | Description |
|---|---|---|---|
| `floor_terms` | all | `2` | How many distinct query units a `bm25` hit must contain; `0` turns the floor off. |
| `relative_cut` | all | `0.2` | Drop `bm25` hits that score under this share of the best one; from 0 up to, but not including, 1. |
| `timeout_s` | `llm` | `30.0` | Seconds before `bm25` answers instead. |
| `min_score` | `embedding` | `0.0` | The lowest cosine similarity kept. |
| `batch_size` | `embedding` | `32` | Texts per embedding request. |

For `llm` and `embedding`, `floor_terms` and `relative_cut` apply to the
`bm25` that answers when they fall back.

With a model server, `send_from_provider` turns a chat provider into the
`send` that `llm` needs:

```python
import asyncio

from localmelo.melo.memory.retrieval import (
    build_retriever,
    load_corpus,
    send_from_provider,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    with open("corpus.jsonl", encoding="utf-8") as file:
        corpus = load_corpus(file)

    llm = OpenAICompatLLM(
        "http://127.0.0.1:8000/v1",
        "Qwen/Qwen3-8B",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        retriever = build_retriever("llm", send=send_from_provider(llm), timeout_s=30.0)
        result = await retriever.search("foldr accumulator order", corpus, k=3)
        hits = [hit.doc_id for hit in result.hits]
        print(result.retriever, result.fallback or "-", hits)
    finally:
        await llm.close()


asyncio.run(main())
```

- `send_from_provider(llm)` sends each retrieval request as one `chat()`
  call without tools, and does not retry it.
- For a cloud API, pass the key as `api_key=` to `OpenAICompatLLM`.
- For `embedding`, pass a provider such as
  `OpenAICompatEmbedding("http://127.0.0.1:11434/v1", "nomic-embed-text")`
  from `localmelo.support.providers.embedding.openai_compat`.
- Close each provider with `await provider.close()` when you are done.

## Recall in the loop

The same three retrievers rank a run's own documents. `--reasoning compress`
(and `compress+replay`) recalls at `start`, `attempt` and `step` by default;
`--recall-at` replaces those triggers, and `tool` adds the `recall` tool.
`--retriever auto` takes `embedding` when one is configured and otherwise
`llm` at `start`/`attempt` and `bm25` at `step`/`tool`; `off` turns recall
off, and a named retriever runs every trigger the user names. A retriever
failure falls back to `bm25`, and the record names the reason.

The offline command below is unchanged: it measures a retriever on a corpus
of recorded calls. A run's blocks are recorded in
`llm_calls.recall_json` (trigger, retriever, query, hits, `ms`, fallback),
and `memory_stats` breaks out the loop's `recall_ms`,
`aux_retrieve_calls` and `retrieve_*` token counts. See
[Recall in the loop](memory.md#recall-in-the-loop).

## See also

- [Memory](memory.md): what localmelo remembers, and where it keeps it.
- [Extra request fields](configuration.md#extra-request-fields-extra_body):
  `chat_template_kwargs` and other server options.
- [Command-line reference](cli.md): the `melo` command.

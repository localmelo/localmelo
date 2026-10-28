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
| `--export-run DIR\|FILE` | not set | Print a run's model calls as corpus lines; see [Export a run](../guides/offline-retrieval.md#export-a-run). |
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
[Export a run](../guides/offline-retrieval.md#export-a-run). `digest` (a summary of several notes) and
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

## The typed long-term store

Three modes read or write the typed `long_term.db` of R6-lite. Each is
mutually exclusive with the evaluation flags other than the model ones and
with `--export-run`, and none contacts a model: `--base-url`, `--chat-model`,
`--embedding-base-url`, `--embedding-model`, `--retrieval-timeout` and
`--retrieval-extra-body` are accepted and ignored, so a bench script can pass
the same flags everywhere. The same store is what [recall in the loop](../concepts/memory.md#recall-in-the-loop)
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
one line on stderr and exit 2. A directory with no `long_term.db` yet — a
bench directory whose run wrote nothing — is an empty store: no line on
stdout, `exported 0 active entries from DIR/long_term.db` on stderr, and
exit 0, so a bench script need not test for the file first. A path that is
not a directory keeps the missing-file error: one line and exit 2.

### `--export-decisions`

One JSON line per decision row, in `seq` order:

```json
{"seq": 1, "candidate_id": "n1", "action": "NEW", "kind": "profile",
 "admitted_by": "user_statement", "entry_id": 1, "target_id": null,
 "reason": "new", "source_ids": ["r1"], "text_sha256": "…", "text": null,
 "timestamp": 0.0, "task_id": "t1"}
```

`source_ids` is the parsed array, and `text` is `null` unless the run had
`--keep-decision-text`. A v0 file has no decisions table and prints none; a
directory with no `long_term.db` prints none and exits 0, with
`exported 0 decisions from DIR/long_term.db` on stderr.

### `--route`

Each input line is one candidate:

```json
{"candidate_id": "n1", "source": "digest", "text": "task: fix it\nfiles: a.py",
 "source_ids": ["r1"], "signal": "rehearsal", "task_id": "t1",
 "session_id": "", "task_text": "plan the move", "task_key": "…"}
```

`source` is one of `reasoning_note`, `digest`, `reflection`, `user` or
`sleep`; `signal` is one of `rehearsal`, `user_request`, `user_statement`,
`outcome` or `sleep`; an unknown value is exit 2. The optional `task_key`
string is stored in the candidate's provenance, so a routed digest can carry
the same key a run's recall looks for; a routed digest without one is
cross-task. The candidate goes through the same boundary and policy as a run,
into a text-only store under `--memory-dir`, and one line comes out per input
line, in input order:

```json
{"candidate_id": "n1", "action": "NEW", "kind": "profile",
 "admitted_by": "user_statement", "entry_id": 1, "target_id": null,
 "reason": "new", "source_ids": ["r1"]}
```

A boundary or admission refusal prints `"action": "REFUSED"` and the reason;
a digest without a nonempty `files:` or `verified:` line is refused at
admission. `--memory-dir`
is required and the command refuses the default `~/.cache/localmelo/memory`
and the `LOCALMELO_MEMORY_DIR` value (resolved paths); the directory is
created `0700`. `--route` appends to an existing scratch store rather than
clearing it, so the bench uses a fresh directory per run. `--export-store` on
that directory shows what the route wrote.

## From Python

> [!NOTE]
> The Python interface of the retrievers is experimental and may change
> before v0.1.0.

`localmelo.melo.memory.retrieval` holds the retrievers, the evaluation and
the file formats. This script searches and evaluates the files of
[the first evaluation](../guides/offline-retrieval.md#run-a-first-evaluation):

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

## See also

- [Memory](memory.md): what localmelo remembers, and where it keeps it.
- [Extra request fields](configuration.md#extra-request-fields-extra_body):
  `chat_template_kwargs` and other server options.
- [Command-line reference](cli.md): the `melo` command.

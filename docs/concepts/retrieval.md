# Concepts: Retrieval

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

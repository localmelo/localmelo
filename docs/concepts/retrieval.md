# Concepts: Retrieval

This page explains how each retriever ranks memory candidates and how a run recalls them. Read it when you need to know why a memory was recalled; the options and the offline comparison are in the [retrieval reference](../reference/retrieval.md).

## How each retriever ranks

| Retriever | Ranks by | Falls back to `bm25` when |
|---|---|---|
| `bm25` | Okapi BM25 over the query's candidates, so how rare a word is depends on those documents alone. | Never: it is the fallback. |
| `llm` | One chat request without tools: the model picks ids from a catalog of the documents. | The request times out, the server errors, or the reply holds no list of ids. Nothing is retried. |
| `embedding` | Cosine similarity of each document's vector with the query's; a negative similarity is left out. | The server errors, or the reply has the wrong number of vectors, a value that is not a finite number or vectors of another length (`provider_error`). |

**`bm25`** then applies these rules:

- **Floor.** A hit must contain at least two distinct units of the query. The units are its words of two or more characters, and its pairs of adjacent Chinese, Japanese or Korean characters; an identifier such as `list_ops` counts once.
- **What does not count.** File names and paths, common function words and the words of note lines, such as `plan`, `found`, `read` and `run`. A query with fewer than two units returns no hits.
- **Kinds.** Thought chunks keep at most one hit per task step, the best, and every other kind ranks above every thought chunk.
- **Relative cut.** Among the thought chunks, and among the other kinds, hits that score under 0.2 times the best one are dropped.

**`llm`** sends a fixed system prompt, then a catalog of the documents, the aliases not to return, the query, and how many ids to return. `melo-retrieve` sends it with `temperature` 0, so the same model and files give the same picks; `--retrieval-extra-body` can set another.

- Each catalog line reads `[<alias>] <label> · step <n> · <title>`, as in `[n1] note · step 1 · Find the stubs`. The alias is a letter for the kind (`n` note, `d` draft, `t` thought, `r` reflection, `g` digest, `m` long) and the document's position among the documents of that kind.
- Excluded documents keep their place in the catalog, so the catalog changes only when the documents do, and a server can reuse its prompt cache from one search to the next. Over 12,000 characters, the earliest lines are dropped, 3,000 characters at a time.
- The model answers with `{"ids": ["n2", "d1"]}`, best first. Text or a code fence around the JSON is accepted, and so is a bare list. Hits score 1 / rank.
- On a failure, `bm25` answers instead, and the result names the reason: `timeout`, `provider_error` or `parse`.

**`embedding`** embeds each document once, 32 texts per request, and keeps the vector; it embeds the query on every search.

## Recall in the loop

The same three retrievers rank a run's own documents.

| Setting | What it does |
|---|---|
| `--reasoning compress` (and `compress+replay`) | Recalls at `start`, `attempt` and `step` by default. |
| `--recall-at` | Replaces those triggers; `tool` adds the `recall` tool. |
| `--retriever auto` | Takes `embedding` when one is configured; otherwise `llm` at `start`/`attempt` and `bm25` at `step`/`tool`. |
| `--retriever off` | Turns recall off; a named retriever runs every trigger the user names. |

- A retriever failure falls back to `bm25`, and the record names the reason.
- The [offline command](../guides/offline-retrieval.md) is unchanged: it measures a retriever on a corpus of recorded calls.
- A run's blocks are recorded in `llm_calls.recall_json` (trigger, retriever, query, hits, `ms`, fallback), and `memory_stats` breaks out the loop's `recall_ms`, `aux_retrieve_calls` and `retrieve_*` token counts.
- See [Recall in the loop](memory.md#recall-in-the-loop).

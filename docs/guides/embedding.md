# Set up an embedding model

An embedding model is optional. localmelo never downloads or starts one: you
run a model server, and localmelo calls a model that server already serves.

## What it buys

Without an embedding backend, working memory and history still work. With
memory on, explicit reasoning modes can admit text-only long-term entries and
find them with BM25; persistence keeps those entries between processes. For
explicit-mode recall, the trigger must be enabled and include the `long` kind.
In `legacy`, long-term writes and recall require an embedding backend. See
[Memory](../concepts/memory.md#what-localmelo-remembers).

An embedding backend adds vector-similarity search for entries with vectors.
Text-only entries are still found with BM25. In `compress` and
`compress+replay`, the `auto` retriever also uses the embedding provider to
rank task-local recall; without one, `auto` uses the LLM at start and attempt
and BM25 at step. The [recall reference](../concepts/memory.md#recall-in-the-loop)
explains the triggers and other modes.

An offline comparison measured the three retrievers at capped grade-2
recall@3 over 97 gold queries: `bm25` 0.284, `llm` 0.576 and `embedding`
0.636. The difference between `embedding` and `llm` is +0.06 with a 95%
confidence interval of [−0.055, +0.179], so at that size the two are not
distinguishable; only the gain over `bm25` is. Choosing an embedding model
first is therefore a product choice — the best point estimate, no extra chat
call, and a model that can be small and local — not a proven win.

localmelo itself never downloads, starts or serves a model. The commands
below are for your own server.

## Check the server

Ask the server what it serves, then ask one model for an embedding:

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": "ping"}'
```

The second reply must hold a non-empty `data[0].embedding` array. A model
listing alone is only a candidate: OpenAI-compatible servers, including oMLX
and Ollama, do not say which ids are embedding models, so the embeddings
request is the check. localmelo uses the same request as its probe and never
saves an embedding model that did not answer it.

## Ollama

The recommended model is `qwen3-embedding:0.6b`; `nomic-embed-text` is a
smaller English-only alternative.

```bash
ollama pull qwen3-embedding:0.6b
curl -sS http://127.0.0.1:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-embedding:0.6b", "input": "ping"}'
melo --reconfigure    # Embedding backend: 2 (Ollama); URL http://localhost:11434; model qwen3-embedding:0.6b
```

In the wizard, choose `2` for `Ollama (local endpoint)`, accept
`http://localhost:11434` and type `qwen3-embedding:0.6b` as the model. The
wizard writes:

```toml
chat_backend = "ollama"
embedding_backend = "ollama"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
embedding_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"
```

Ollama's `embedding_url` has no `/v1`, as its `chat_url` has none; the
backend adds `/v1` for embeddings (`POST /v1/embeddings`). Chat goes to
Ollama's own `/api/chat` API. For the smaller model, pull `nomic-embed-text`
and use that id instead.

## oMLX

oMLX serves embeddings and chats from the same endpoint. Put the model in its
model directory, keep the chat server as it is, and check the id oMLX lists:

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": "ping"}'
```

`GET /v1/models` gives ids but no type, so it cannot tell you which ids are
embedding models: the embeddings request above is the check. The model id is
oMLX's own, usually the folder name. Then set the `vllm` backend to oMLX and
add its embedding fields:

```toml
chat_backend = "vllm"
embedding_backend = "vllm"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"
embedding_url = "http://127.0.0.1:8000/v1"
embedding_model = "Qwen3-Embedding-0.6B"
```

See [oMLX](model-servers.md#omlx) for the rest of that setup.

## vLLM

vLLM serves an embedding model as its own endpoint. This is not run on this
Mac (no vLLM here); check the flag in `vllm serve --help` for your version.

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001
curl -sS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "ping"}'
```

Then set `embedding_backend = "vllm"` with that URL and the served id:

```toml
embedding_backend = "vllm"

[vllm]
embedding_url = "http://127.0.0.1:8001/v1"
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
```

## SGLang and MLC

The same pattern works for SGLang, not run here:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-0.6B --is-embedding --port 30001
curl -sS http://127.0.0.1:30001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "ping"}'
melo --reconfigure    # Embedding backend: 4 (SGLang); URL http://localhost:30001/v1
```

MLC needs a compiled embedding artifact; a chat artifact does not serve
embeddings. With `<artifact>` as the compiled Qwen3-Embedding directory:

```bash
mlc_llm serve <artifact> --port 8080
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "<the served id>", "input": "ping"}'
melo --reconfigure    # Embedding backend: 1 (MLC); URL http://localhost:8080/v1; model <the served id>
```

Neither command is run on this Mac. The archived backend smoke used a
compiled `Qwen3-Embedding-0.6B-q0f16-MLC` artifact on MLC and
`qwen3-embedding` as the served id.

## A second server

Chat and embedding backends are independent. A cloud chat model can use a
local embedding server, and a local chat server can use another local server
for embeddings. In the wizard, the generic `1`–`4` options ask for the
embedding endpoint's URL and model, so pick the family of the embedding
server, not of the chat server:

```toml
chat_backend = "openai"
embedding_backend = "ollama"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"

[ollama]
embedding_url = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
```

See [Mix backends](model-servers.md#mix-backends).

## What changes

- The interactive banner's memory row names the backend and model, for
  example `long-term memory · vLLM · Qwen3-Embedding-0.6B`; without one it
  says `no long-term memory · melo --reconfigure adds one` (or
  `--embedding-base-url adds one` in a `--base-url` session). This row
  describes the embedding configuration, not whether explicit modes can use
  text-only long-term memory.
- A new long-term store starts empty. Eligible entries may be admitted as runs
  end and steps complete; existing text-only entries are not re-embedded.
- Changing the model changes the vector space. A stored vector whose
  dimension differs from the new model's is skipped, so old entries stop
  ranking instead of breaking recall. To reclaim the space, stop localmelo
  and remove `~/.cache/localmelo/memory/long_term.db*`; see
  [Reset memory](memory.md#reset-memory).
- The web workspace also has connection settings. See
  [the web workspace](web-console.md#web-workspace).

## Check it works

- Start `melo` in a terminal and look at the banner's memory row.
- Once `/status` is available, its `memory` row comes from the same text.
- Run a task that uses a tool, end it, then run a related task. Persistence
  can make admitted entries available to later runs; recall also depends on
  the mode, enabled triggers and relevant matches. See
  [Keep memory between runs](memory.md#keep-memory-between-runs).
- To measure a model on your own corpus, use
  `melo-retrieve --retriever embedding …`; see
  [Offline retrieval](../reference/retrieval.md).

## Troubleshooting

- **HTTP 404** on the embeddings request: the server does not serve that
  model id. List the models and use an id it lists.
- **HTTP 400** on the embeddings request: the id is a chat model, not an
  embedding model. Pick an embedding model, not a chat one.
- **Connection refused**: the server is not running at that URL, or the URL
  is wrong.
- **The model changed and recall is empty**: the old vectors have another
  dimension and are skipped. Restore the old model, or remove
  `long_term.db*` and let it fill again.

More errors: [Troubleshooting](../reference/troubleshooting.md#an-embedding-model-cannot-be-used).

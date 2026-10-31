# Guide: Model servers

Run a model server yourself and point localmelo at it: oMLX, Ollama, MLC LLM, vLLM, SGLang or a cloud API. The setting names and defaults are in the [configuration reference](../reference/configuration.md).

- On a Mac with Apple silicon, oMLX is the recommended local server.
- Ollama is the alternative on other systems.
- A cloud API needs only a key in an environment variable.

## Setup preview

- With no backend saved and no server found, terminal first run and `melo setup`
  show a short, read-only preview. Supported Apple-silicon Macs get oMLX with
  Qwen3 4B Instruct (4-bit): the download upper bound, cached amount, free space
  and planned lifetime, with one warning when needed. Other systems show manual
  guidance without cache or lifetime estimates for an unsupported installer.
  With a backend already saved and no server found, `melo setup` opens the full
  backend menu instead.
- The summary is printed once. The list then offers `Check again`,
  `Show how to set one up`, `Enter an address` and `Quit`. `Show how to set one
  up` prints at most six plain numbered steps: download oMLX, optionally check
  its checksum, open it and move it to Applications, download the model, start
  the server, then choose `Enter an address` in melo. Automatic setup is coming
  soon. A `Check again` that finds no server prints the summary again only if
  the cached amount or free space changed.
- Developer mode prints the full details once and keeps a `Review setup plan`
  menu with a lifetime option, manual steps and `Details`. Exact versions,
  checksums, cache paths, per-volume disk readings, plan digest and the proposed
  port are in `Details` or developer mode, not the default summary.
- RAM uses GiB (shown as GB), with the exact byte count in `Details`; less than
  16 GB is a speed warning, not a manual-only restriction. The pinned model is
  about 2.28 GB; oMLX 0.7.3's DMG is about 0.84 GB on macOS 15 or 0.88 GB on
  macOS 26/27. The remaining download includes both. Expansion/staging overhead
  is unmeasured; disk payload is only a lower bound. Nothing installs,
  downloads, starts or saves settings.
- The steps come from the same pinned recipe. They keep the model in the
  standard Hugging Face cache, use a separate server settings directory, and
  disable oMLX's additional cache scan so this instance serves only the chosen
  snapshot. Install the official app yourself and copy the shown commands only
  when ready. The suggested high port is an example, not a reserved port;
  select a free one before starting and enter its address in melo. These commands
  run independently; stop the manual server yourself with Ctrl+C. The planned
  lifetime does not manage it. Cache precedence is `HF_HUB_CACHE`, then legacy
  `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then the XDG/standard default.
  The proposed server state is under `~/.cache/localmelo/server-setup/`.
  Pipes and `TERM=dumb` show a manual-guide link instead of picker choices;
  slow or unavailable setup checks do not block the manual path.
- A direct snapshot's served model ID can be its revision hash, not the
  repository name. Use the ID the server lists. Rechecking an address only
  lists models; it does not load one. The web/app setup card and already
  configured or explicitly addressed runs keep their existing behavior.

## Serve a model with oMLX

- [oMLX](https://github.com/jundot/omlx) serves MLX models through an OpenAI-compatible API on a Mac with Apple silicon and macOS 15 or later.
- First-run setup finds it on port 8000 and asks which models are loaded.
- In the config file it is the `vllm` backend, also spelled `omlx` ([why](model-servers.md#omlx)).

1. Install oMLX with Homebrew:

   ```bash
   brew tap jundot/omlx https://github.com/jundot/omlx
   brew install jundot/omlx/omlx
   ```

   The oMLX macOS app installs the command too, as `~/.omlx/bin/omlx`, and ships the optional native kernels that a few model families (Qwen3.5, for example) need for full speed; the [README](https://github.com/jundot/omlx#install) has the other ways to install it.

2. Put each model in a folder of its own inside one models folder: oMLX serves every subfolder that holds a model (a `config.json` and its weights), and the subfolder's name is the model id. With the Hugging Face command line (`hf`, from `uv tool install huggingface_hub`):

   ```bash
   hf download mlx-community/Qwen3-8B-4bit --local-dir ~/models/Qwen3-8B-4bit
   ```

3. Start the server, and leave it running in its own terminal:

   ```bash
   omlx serve --model-dir ~/models --port 8000
   ```

   - It listens on `http://127.0.0.1:8000`; oMLX saves `--model-dir` and `--port` in its `settings.json` (under `~/.omlx`), so a port saved earlier beats the default 8000: give `--port` to avoid that.
   - The oMLX app and `omlx start` run the server in the background with the folder and port saved in oMLX (the Homebrew service uses `~/.omlx/models` and port 8000): then skip this command.
   - Keep the default host and do not set `--api-key`: localmelo sends no API key to a local server.
   - HTTP 401 (`API key required`) means a key was set, in oMLX's admin page or with `OMLX_API_KEY`: turn on **Allow inference without API key** in the admin page's settings, or unset `OMLX_API_KEY`.
   - With that setting on, listing models and chat work, but oMLX still wants the key on the route that says which models are loaded, so setup cannot mark them.

4. Check that the server lists the model:

   ```bash
   curl -sS http://127.0.0.1:8000/v1/models
   ```

   The reply's `data` list holds an entry with `"id": "Qwen3-8B-4bit"` (it can list MLX models from your Hugging Face cache too). `Failed to connect` means nothing listens on that port: check the port in oMLX's startup log. `API key required` means a key is set: see step 3.

5. Run a task with that id:

   ```bash
   melo --url http://127.0.0.1:8000/v1 --model Qwen3-8B-4bit "What is 6*7?"
   ```

   - oMLX loads the model on its first request, which can take a minute for a large model; if that request times out, add `--request-timeout 600`.
   - Without `--model`, melo asks oMLX which models it has and uses the only chat model or the one that is loaded, for a task and for `melo --url http://127.0.0.1:8000/v1` without a task (the interactive UI).
   - With several chat models and not exactly one loaded, it exits 2, lists them and asks for `--model <id>`.

To make oMLX your default, run `melo` in a terminal before any configuration is saved: it finds oMLX on port 8000, lists its models and saves the one you pick ([First run](configure.md#first-run) shows the exchange). oMLX loads a model when the first request for it arrives, so a listed model is not always in memory. With several chat models, setup asks oMLX which are loaded (`GET /v1/models/status`, one second at most) and chooses like this:

| The server lists | Setup does |
|---|---|
| One chat model | It is the default. |
| Several, and oMLX says which are loaded | Marks them `(loaded)`; the first of them is the default. |
| Several, and none is loaded or oMLX does not say | No default: nothing is highlighted, and Enter does nothing until you move to a model with ↑ ↓ (answers piped in: type the model's number), because choosing for you could start a long load. |

- While a key is set, oMLX does not say which are loaded, even with **Allow inference without API key** on, and setup prints `The server did not say which models are loaded (it needs a key for that).`
- A model with an alias in oMLX is listed under the alias, and its status reply names the alias beside the folder name (`model_alias`), so it is marked and can be the default; a server whose reply has no alias cannot mark such a model.

With a configuration saved already, run step 5 first, so the model is loaded, then `melo setup`:

- In default mode it offers the server it finds, with the first run's questions.
- In the full menu (after `Use another server`, or in developer mode), choose oMLX or vLLM (`3`), enter `http://127.0.0.1:8000/v1` and the model id, and choose `5` (None) for embeddings unless oMLX also serves an embedding model.
- [oMLX configuration](model-servers.md#omlx) shows the file it saves and the table that turns off a Qwen3 model's thinking; the web workspace then uses that model.

## Serve a model with Ollama

[Ollama](https://ollama.com) is the alternative on other systems. Install it from its site, then start the server and pull a model:

```bash
ollama serve
ollama pull qwen3:8b
```

- `ollama serve` keeps running, so run `ollama pull` in another terminal.
- If Ollama already runs as an app or a service, skip the first command.
- First-run setup finds Ollama on port 11434 and lists its models.

## Backends

| Key | Type | Server or API | Default URL | Embeddings |
|---|---|---|---|---|
| `ollama` | local | [Ollama](https://ollama.com) | `http://localhost:11434` (wizard) | yes |
| `mlc` | local | [MLC LLM](https://llm.mlc.ai) | `http://localhost:8400/v1` (wizard) | yes |
| `vllm` | local | [oMLX](https://github.com/jundot/omlx), [vLLM](https://docs.vllm.ai) or another OpenAI-compatible server | `http://localhost:8000/v1` (wizard) | yes |
| `sglang` | local | [SGLang](https://docs.sglang.ai) | `http://localhost:30000/v1` (wizard) | yes |
| `openai` | cloud | OpenAI API | `https://api.openai.com/v1` | no |
| `anthropic` | cloud | Anthropic API | `https://api.anthropic.com/v1` | no |
| `gemini` | cloud | Google Gemini API | `https://generativelanguage.googleapis.com/v1beta/openai` | no |
| `nvidia` | cloud | NVIDIA API | `https://integrate.api.nvidia.com/v1` | no |

- The wizard suggests the local URLs; use your server's address and port.
- localmelo only connects to a local server: you start and stop it yourself.
- `mlc`, `vllm` and `sglang` use the plain OpenAI-compatible API, so another OpenAI-compatible server works with any of them.
- In the config file `omlx` is another spelling of `vllm` ([details](model-servers.md#omlx)).

### oMLX

[oMLX](https://github.com/jundot/omlx) is a supported local server; [Serve a model with oMLX](model-servers.md#serve-a-model-with-omlx) shows how to start it.

- Its API is OpenAI-compatible, and the `vllm` key stands for any OpenAI-compatible server, so oMLX is the `vllm` backend, which the config file spells `omlx`.
- First-run setup saves `chat_backend = "omlx"` for a server whose model listing names oMLX as the owner of its models, and `embedding_backend = "omlx"` when it uses that server's embedding model; any other server is saved as `vllm`.
- Both spellings load as the same backend, with its settings in the `[vllm]` table, and are saved back as written.
- `/status`, developer mode's banner and the web console call the server oMLX when the file says `omlx` or a model listing of the server did, and otherwise an OpenAI-compatible server.

```toml
chat_backend = "omlx"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- `chat_model` is the model's id in oMLX's `GET /v1/models`: its folder name, or the alias set for it in oMLX; `--model ID` replaces it for one run.
- [First run](configure.md#first-run) writes the `chat_backend`, `chat_url` and `chat_model` lines (and, with an embedding model, `embedding_backend = "omlx"` with that table's `embedding_url` and `embedding_model`); add the other lines by hand.
- `request_timeout_s = 600` gives oMLX time to load a model that is not loaded yet; see [Chat request timeout](../reference/configuration.md#chat-request-timeout).
- The `extra_body` table turns off a Qwen3 model's thinking, which makes each step faster; leave it out to keep thinking on. oMLX applies it unless the model's settings in oMLX force `enable_thinking`.
- With thinking off, melo takes a reply's text as the answer even when the server sends it in the reasoning field; a reply that still carries counted reasoning tokens and no text is not.
- For long-term memory, oMLX can serve an embedding model from the same directory: set `embedding_backend = "omlx"` and add `embedding_url = "http://127.0.0.1:8000/v1"` and `embedding_model = "<its id>"` to `[vllm]`; the id and one `POST /v1/embeddings` are the check, since `GET /v1/models` gives no type. See [Set up an embedding model](embedding.md#omlx).

### Ollama with embeddings

```toml
chat_backend = "ollama"
embedding_backend = "ollama"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
embedding_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"
```

Pull both models first: `ollama pull qwen3:8b` and `ollama pull qwen3-embedding:0.6b`. `nomic-embed-text` is a smaller English-only alternative.

### MLC LLM, vLLM or SGLang

```toml
chat_backend = "vllm"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- For MLC LLM or SGLang, use the table `[mlc]` or `[sglang]` and the matching `chat_backend`.
- Set `chat_model` to a model that your server serves; its `GET /v1/models` lists them.
- For embeddings from the same kind of server, set `embedding_backend = "vllm"` (or `mlc`, `sglang`) and add `embedding_url` and `embedding_model` to the table.

### A cloud API

```toml
chat_backend = "openai"
embedding_backend = "none"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"
```

Export the key in the shell that runs `melo`:

```bash
export OPENAI_API_KEY="<your key>"
melo "What is 6*7?"
```

For another vendor, use the table `[anthropic]`, `[gemini]` or `[nvidia]`; the wizard suggests the variables `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and `NVIDIA_API_KEY`.

### Mix backends

Chat and embedding backends are chosen separately: any chat backend works with an embedding backend of `mlc`, `ollama`, `vllm`, `sglang` or `none`. For example, a cloud model with local Ollama embeddings:

```toml
chat_backend = "openai"
embedding_backend = "ollama"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"

[ollama]
embedding_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"
```

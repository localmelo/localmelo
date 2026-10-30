# Guide: Model servers

This guide shows how to run a model server yourself and point localmelo at it: oMLX, Ollama, MLC LLM, vLLM, SGLang or a cloud API. Read it when you choose a backend; the setting names and defaults are in the [configuration reference](../reference/configuration.md). On a Mac with Apple silicon, oMLX is the recommended local server; Ollama is the alternative on other systems.

## Serve a model with oMLX

[oMLX](https://github.com/jundot/omlx) serves MLX models through an
OpenAI-compatible API on a Mac with Apple silicon and macOS 15 or later.
localmelo supports it as a local server: first-run setup finds it on port
8000 and asks it which models are loaded. In the config file it is the
`vllm` backend, also spelled `omlx`; [oMLX](model-servers.md#omlx)
explains why. To use it instead of Ollama:

1. Install oMLX with Homebrew:

   ```bash
   brew tap jundot/omlx https://github.com/jundot/omlx
   brew install jundot/omlx/omlx
   ```

   The oMLX macOS app also installs the command, as `~/.omlx/bin/omlx`. Its
   [README](https://github.com/jundot/omlx#install) has the other ways to
   install it. The Homebrew command above installs oMLX without its
   optional native kernels, which a few model families (Qwen3.5, for
   example) need to run at full speed; the app ships them, and the README
   explains the Homebrew option.

2. Put each model in a folder of its own inside one models folder. oMLX
   serves every subfolder that holds a model (a `config.json` and its
   weights), and a subfolder's name is its model id. For example, with the
   Hugging Face command line (`hf`, from `uv tool install huggingface_hub`):

   ```bash
   hf download mlx-community/Qwen3-8B-4bit --local-dir ~/models/Qwen3-8B-4bit
   ```

3. Start the server, and leave it running in its own terminal:

   ```bash
   omlx serve --model-dir ~/models --port 8000
   ```

   It listens on `http://127.0.0.1:8000`. oMLX saves `--model-dir` and
   `--port` in its `settings.json` (under `~/.omlx` by default) and uses
   them when you leave the flags out, so a port saved earlier wins over the
   default 8000; giving `--port` avoids that. The oMLX app and `omlx start`
   run the server in the background instead, with the folder and port saved
   in oMLX (the Homebrew service starts with `~/.omlx/models` and port
   8000): then skip the command above.

   Keep the default host and do not set `--api-key`: localmelo sends
   no API key to a local server. If requests fail with HTTP 401
   (`API key required`), a key was set earlier, in oMLX's admin page or
   with `OMLX_API_KEY`: turn on **Allow inference without API key** in the
   admin page's settings, or unset `OMLX_API_KEY`. With that setting on,
   listing models and chat work, but oMLX still asks for the key on the
   route that says which models are loaded, so setup cannot mark them (see
   below).

4. Check that the server lists the model:

   ```bash
   curl -sS http://127.0.0.1:8000/v1/models
   ```

   The reply's `data` list holds an entry with `"id": "Qwen3-8B-4bit"`. It
   can also list MLX models from your Hugging Face cache.
   `Failed to connect` means that nothing listens on that port: check the
   port in oMLX's startup log. `API key required` means that a key is set:
   see step 3.

5. Run a task with that id:

   ```bash
   melo --url http://127.0.0.1:8000/v1 --model Qwen3-8B-4bit "What is 6*7?"
   ```

   oMLX loads the model on its first request, which can take a minute for a
   large model. If that request times out, add `--request-timeout 600`.
   Without `--model`, melo asks oMLX which models it has and uses the
   only chat model or the one that is loaded, for a task and for
   `melo --url http://127.0.0.1:8000/v1` without a task (the interactive
   UI); with several chat models and not exactly one loaded, it exits 2, lists
   them and asks for `--model <id>`. Name the model with `--model`
   to skip that lookup.

To make oMLX your default, run `melo` in a terminal before any configuration
is saved: it finds oMLX on port 8000, lists its models and saves the one you
pick; [First run](configure.md#first-run) shows the exchange. oMLX loads a
model when the first request for it arrives, so a listed model is not always
in memory, and loading a large one can take minutes. Setup therefore chooses
like this, and with several chat models it asks oMLX which are loaded
(`GET /v1/models/status`, one second at most):

- One chat model: it is the default.
- Several, and oMLX says which are loaded: setup marks them `(loaded)` and
  makes the first of them the default.
- Several, and none is loaded or oMLX does not say: there is no default.
  Type the number of the model to use; Enter alone asks again, because
  choosing for you could start a long load of a large model. While a key is
  set, oMLX does not say, even with **Allow inference without API key** on,
  and setup prints `The server did not say which models are loaded (it needs
  a key for that).` A model that has an alias in oMLX is listed under the
  alias, and the status reply names the alias beside the folder name
  (`model_alias`, as oMLX 0.7.1rc1 sends it), so the model is marked and can
  be the default under the alias. A server whose reply has no alias cannot
  mark such a model.

With a configuration saved already, run step 5 first, so that the model is
loaded, then `melo setup`: in default mode it offers the server it
finds with the first run's questions. In the full menu (after `n`, or with
`--developer`), choose oMLX or vLLM (`3`) as the chat backend, enter
`http://127.0.0.1:8000/v1` and the model id, and choose `5` (None) as the
embedding backend unless oMLX also serves an embedding model.
[oMLX configuration](model-servers.md#omlx) shows the file they save and the
table to add to turn off a Qwen3 model's thinking. The web workspace then
uses that model; it reads the configuration file.

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

The wizard suggests the local URLs; use your server's address and port.
localmelo only connects to a local server: you start and stop it yourself.
`mlc`, `vllm` and `sglang` use the plain OpenAI-compatible API, so another
OpenAI-compatible server works with any of them. In the config file `omlx` is
another spelling of `vllm` ([details](model-servers.md#omlx)).

### oMLX

[oMLX](https://github.com/jundot/omlx) is a supported local server. Its API
is OpenAI-compatible, and the `vllm` key stands for any OpenAI-compatible
server, vLLM's own included. So oMLX has no backend of its own: it is the
`vllm` backend, and the config file spells it `omlx`. First-run setup saves
`chat_backend = "omlx"` for a server whose model listing names oMLX as the
owner of its models (oMLX does, on every model it lists), and
`embedding_backend = "omlx"` when it uses that server's embedding model; any
other server is saved as `vllm`. Both spellings load as the same backend, with
its settings in the `[vllm]` table, and are saved back as written. The banner,
`/status` and the web console call the server oMLX when the file says `omlx`
or a model listing of the server did, and otherwise an OpenAI-compatible
server. A file that says `vllm` is named after the first listing, which the
check after the banner makes; the banner itself shows what the file says,
unless the run already listed the server.
[Serve a model with oMLX](model-servers.md#serve-a-model-with-omlx) shows how
to start it.

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

- `chat_model` is the model's id in oMLX's `GET /v1/models`: its folder
  name, or the alias set for it in oMLX. `--model ID` replaces it for
  one run.
- [First run](configure.md#first-run) writes the `chat_backend`, `chat_url` and
  `chat_model` lines: run `melo` in a terminal with no config file, while
  oMLX runs. When it offers an embedding model it also writes
  `embedding_backend = "omlx"` with that table's `embedding_url` and
  `embedding_model`. Add the other lines by hand.
- `request_timeout_s = 600` gives oMLX time to load the model when a
  request finds it unloaded; see [Chat request timeout](../reference/configuration.md#chat-request-timeout).
- The `extra_body` table turns off a Qwen3 model's thinking, which makes
  each step faster. Leave it out to keep thinking on. oMLX applies it unless
  the model's settings in oMLX force `enable_thinking`.
- For long-term memory, oMLX can also serve an embedding model from the
  same directory: set `embedding_backend = "omlx"` and add
  `embedding_url = "http://127.0.0.1:8000/v1"` and
  `embedding_model = "<its id>"` to `[vllm]`. oMLX's `GET /v1/models`
  gives ids but no type, so it cannot tell an embedding model from a chat
  model: the id and one `POST /v1/embeddings` are the check. See
  [Set up an embedding model](embedding.md#omlx).

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

Pull both models first: `ollama pull qwen3:8b` and
`ollama pull qwen3-embedding:0.6b`. `nomic-embed-text` is a smaller
English-only alternative.

### MLC LLM, vLLM or SGLang

```toml
chat_backend = "vllm"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- For MLC LLM or SGLang, use the table `[mlc]` or `[sglang]` and the matching
  `chat_backend`.
- Set `chat_model` to a model that your server serves. The server's
  `GET /v1/models` lists them.
- To get embeddings from the same kind of server, set
  `embedding_backend = "vllm"` (or `mlc`, `sglang`) and add `embedding_url`
  and `embedding_model` to the table.

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

For another vendor, use the table `[anthropic]`, `[gemini]` or `[nvidia]`.
The wizard suggests the variables `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and
`NVIDIA_API_KEY`.

### Mix backends

Chat and embedding backends are chosen separately. Any chat backend works
with an embedding backend of `mlc`, `ollama`, `vllm`, `sglang` or `none`.
For example, a cloud model with local Ollama embeddings:

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

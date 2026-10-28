# Guide: Model servers

## Serve a model with oMLX

[oMLX](https://github.com/jundot/omlx) serves MLX models through an
OpenAI-compatible API on a Mac with Apple silicon and macOS 15 or later. To
use it instead of Ollama:

1. Install oMLX with Homebrew:

   ```bash
   brew tap jundot/omlx https://github.com/jundot/omlx
   brew install jundot/omlx/omlx
   ```

   The oMLX macOS app also installs the command, as `~/.omlx/bin/omlx`. Its
   [README](https://github.com/jundot/omlx#install) has the other ways to
   install it.

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
   default 8000; giving `--port` avoids that. Keep the default host and do not set `--api-key`: localmelo sends
   no API key to a local server. If requests fail with HTTP 401
   (`API key required`), a key was set earlier, in oMLX's admin page or
   with `OMLX_API_KEY`: turn on **Allow inference without API key** in the
   admin page's settings, or unset `OMLX_API_KEY`.

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
   melo --base-url http://127.0.0.1:8000/v1 --chat-model Qwen3-8B-4bit "What is 6*7?"
   ```

   oMLX loads the model on its first request, which can take a minute for a
   large model. If that request times out, add `--request-timeout 600`.
   `melo --base-url http://127.0.0.1:8000/v1` without a task opens the
   interactive UI on the first model oMLX lists.

To make oMLX your default, run `melo` in a terminal before any configuration
is saved: it finds oMLX on port 8000, lists its models and saves the one you
pick. With a configuration saved already, run step 5 first, so that the
model is loaded, then the wizard (`melo --reconfigure`): choose vLLM (`3`)
as the chat backend, enter `http://127.0.0.1:8000/v1` and the model id, and
choose `5` (None) as the embedding backend unless oMLX also serves an
embedding model. [Configuration](model-servers.md#omlx) shows the file they
save and the table to add to turn off a Qwen3 model's thinking. The web
workspace then uses that model; it reads the configuration file.

## Backends

| Key | Type | Server or API | Default URL | Embeddings |
|---|---|---|---|---|
| `ollama` | local | [Ollama](https://ollama.com) | `http://localhost:11434` (wizard) | yes |
| `mlc` | local | [MLC LLM](https://llm.mlc.ai) | `http://localhost:8400/v1` (wizard) | yes |
| `vllm` | local | [vLLM](https://docs.vllm.ai) | `http://localhost:8000/v1` (wizard) | yes |
| `sglang` | local | [SGLang](https://docs.sglang.ai) | `http://localhost:30000/v1` (wizard) | yes |
| `openai` | cloud | OpenAI API | `https://api.openai.com/v1` | no |
| `anthropic` | cloud | Anthropic API | `https://api.anthropic.com/v1` | no |
| `gemini` | cloud | Google Gemini API | `https://generativelanguage.googleapis.com/v1beta/openai` | no |
| `nvidia` | cloud | NVIDIA API | `https://integrate.api.nvidia.com/v1` | no |

The wizard suggests the local URLs; use your server's address and port.
localmelo only connects to a local server: you start and stop it yourself.
`mlc`, `vllm` and `sglang` use the plain OpenAI-compatible API, so another
OpenAI-compatible server works with any of them.

### oMLX

[oMLX](https://github.com/jundot/omlx) has an OpenAI-compatible API, so it
uses the `vllm` backend. [Serve a model with oMLX](model-servers.md#serve-a-model-with-omlx)
shows how to start it.

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- `chat_model` is the model's id in oMLX's `GET /v1/models`: its folder
  name, or the alias set for it in oMLX.
- [First run](configure.md#first-run) writes the `chat_backend`, `chat_url` and
  `chat_model` lines: run `melo` in a terminal with no config file, while
  oMLX runs. When it offers an embedding model it also writes
  `embedding_backend = "vllm"` with that table's `embedding_url` and
  `embedding_model`. Add the other lines by hand.
- `request_timeout_s = 600` gives oMLX time to load the model when a
  request finds it unloaded; see [Chat request timeout](../reference/configuration.md#chat-request-timeout).
- The `extra_body` table turns off a Qwen3 model's thinking, which makes
  each step faster. Leave it out to keep thinking on. oMLX applies it unless
  the model's settings in oMLX force `enable_thinking`.
- For long-term memory, oMLX can also serve an embedding model from the
  same directory: set `embedding_backend = "vllm"` and add
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

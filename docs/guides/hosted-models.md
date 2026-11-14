# Guide: Hosted models

Use a model from a hosted service (OpenAI, Google Gemini, Anthropic, NVIDIA, DeepSeek or any service that speaks the OpenAI format) when the model on your Mac is not enough. You set it up in Settings, with no shell work. For a model server you run yourself, see [Model servers](model-servers.md).

- Choose a service, paste your key once, pick a model from the list it offers, save.
- Keep keys for several services; each remembers the model you used last, and one list switches between them.
- The key is kept in your Mac's keychain, never in melo's settings, and goes only to its own service.
- If a thinking model leaves its answer only in its thinking, melo recovers it.

## Set up a service

![Settings, Model server: DeepSeek chosen under Hosted services, with the key entry, the chat model and the thinking switch](../assets/img/hosted-service.webp)

- Open **Settings**, then **Model server**, then **Change…**.
- In the **Model server** list, under **Hosted services**, choose the service. Its address and suggested models fill in.
- Paste your key and choose **Save key**. The service's models appear in the list by themselves.
- Pick a model. The first one the service suggests is offered first; if it lists none, type one.
- Choose **Test connection** to check the key and the model with one short chat, then **Save connection**. Your next message uses the service.

## The services

| Service | Address | Suggested models |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| Anthropic | `https://api.anthropic.com/v1` | `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5` |
| NVIDIA | `https://integrate.api.nvidia.com/v1` | `nvidia/llama-3.1-nemotron-70b-instruct` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-flash`, `deepseek-v4-pro` |
| Other (OpenAI format) | the address you type, up to `/v1` | the model you type |

- The list comes from the service itself (`GET <address>/models`, never a chat) once a key exists: when you open Settings, when you choose the service and after you save a key. It is asked once a minute at most.
- A service that does not list its models, or a failure, says so in plain words; you can still type a model.
- OpenAI, DeepSeek and Other share the `[openai]` section of the config file; Gemini, Anthropic and NVIDIA have their own.
- Use **Other** for a service not in the list, such as a company gateway.

## Switch between models

- The model chip in the console (and in the app's window), the model button under the message box and `/model` in the terminal show one list: the model server on this Mac and every hosted service that has a key, grouped, the one in use ticked.
- Pick a model: your next message uses it. No key is typed again, no test runs and nothing restarts. In the terminal, `/model NAME` or `/model service:NAME` switches with no list.
- A turn that is running finishes on the model it started with.
- A service that cannot be listed is one row that says why; it hides nothing else. A service with no key is not in the list.
- Switching never changes your embedding choice, and melo never changes a saved hosted model by itself.

- Without a hosted key nothing changes: the list is the model server's, as before.
- If the service no longer offers the model you saved, Settings says so at once (`The service has no model called ...`), with no test. Choose another from the list.

## Your key

- **Save key** writes the key to your Mac's keychain, under the name `localmelo.<service>` (for example `localmelo.deepseek`).
- A key goes only to its own service: the built-in address of a service in the list, or for **Other** the address saved with it. melo refuses anything else before it sends a request.
- **Other** keeps one key per address (`localmelo.other-<12 characters>`), so a key saved for one gateway is never found for another. A key an earlier version saved as `localmelo.other` is not used: paste it again.
- The config file keeps only a pointer to it, `api_key = "keychain:localmelo.deepseek"`. melo does not show the key again, and it never reaches the config, the logs or the page.
- **Replace** saves a new key; **Remove** deletes it from the keychain.
- An environment variable still works: if `api_key_env` is set and the variable has a value, it is used before the keychain.
- The macOS app does not see variables from your shell. Paste the key in Settings instead of exporting it.

If melo cannot read the key, it says what to do:

| What you see | What to do |
|---|---|
| The saved key is not in the keychain | Paste the key again in Settings, or run `melo setup`. |
| The keychain is locked | Unlock it (open Keychain Access, or sign in to your Mac again), then try again. |
| No keychain on this computer | Set the key in an environment variable and name it in `api_key_env`. |
| That does not look like an API key | Paste only the key: no spaces, quotes or line breaks. |
| The key was saved for another service or address | Paste it again for this service. melo never sends a key anywhere else. |

## Test the connection

![Settings, Model server: the key saved in the keychain and a test result that says the model answered and used a tool](../assets/img/hosted-test.webp)

- **Test connection** asks the service for its model list, then sends one short chat that offers one tool.
- It tells you in plain words what is wrong, and what to do.
- The key is sent to the service only, in a header.

| Result | What it means |
|---|---|
| Works | The model answered and used the tool. |
| The service did not accept the key (401) | Paste the key again. |
| The service refused this account (402, 403) | Check the plan, balance and access on the service's website. |
| The service has no model called ... | Choose one from the list. |
| The service is busy (429) | Wait a moment, then test again. |
| The service did not answer in time | Try again, or choose a faster model. |
| Cannot reach the service | Check your internet connection and the address. |
| Connected, but the model did not use the test tool | melo needs a model that can use tools. |

## From the terminal

```shell
melo setup
```

- `melo setup` lists the servers on this computer and the same hosted services as Settings.
- With a key, it offers the models the service lists first; with none listed it asks for a name.
- The key is typed hidden and saved to the keychain; Enter keeps a key melo already has.
- **Use an environment variable instead** sets `api_key_env`.
- With no model server found, the first run offers **Use a hosted service**.
- `/model` lists every model you can use now once a service has a key; see [Switch between models](#switch-between-models).

## Thinking models

- Some models think before they answer. Some services have a switch for it; Settings shows the checkbox **Let the model think before it answers** for them.
- A thinking model can end a turn with its answer only in its thinking. melo then asks once more with thinking off.
- If the reply is still empty, melo shows the last lines of the thinking as the answer, marked `(from melo's thinking)`, and the run finishes.
- A reply with no thinking at all still fails, as before.
- A run's result says when either happened (`empty_reply` in the JSON record, and a notice in the chat). To count them, or to run without them, set `[agent] empty_reply_fallback = "off"` (`-X agent.empty_reply_fallback=off` for one run); a turn that ends with only thinking then fails as it did before.
- DeepSeek: `deepseek-flash` starts with thinking off; `deepseek-v4-pro` keeps its own default. Change either with the checkbox or with `thinking` in the config.

## Pictures

- Some tools return pictures, such as a browser screenshot. With `[agent] tool_images = "auto"`, melo sends them to the model when it is known to read images.
- A service's model list does not say, so each service names the models: GPT-4o, GPT-4.1 and GPT-5 (OpenAI), every Gemini model, the Claude models and `deepseek-flash` read images; `deepseek-v4-pro` and NVIDIA's `nemotron` do not.
- Any other model counts as unknown and gets no pictures; its tool messages say it cannot see them. If yours reads images, set `tool_images = "on"`.

## The config file

Settings writes these lines; you can also write them yourself.

```toml
chat_backend = "openai"
embedding_backend = "none"

[openai]
base_url = "https://api.deepseek.com/v1"
api_key = "keychain:localmelo.deepseek"
chat_model = "deepseek-flash"
thinking = "off"             # "on", "off", or leave out to follow the service
```

- `base_url` tells the presets apart: OpenAI has none (its address is built in), DeepSeek has the address above, and any other address is Other.
- Each service you used keeps its last model in a table of its own, written beside the lines above. An older melo ignores these tables and drops them the next time it saves; nothing else changes. Keys are never in them.

```toml
[services.deepseek]
chat_model = "deepseek-flash"

[services.openai]
chat_model = "gpt-4o"

[services.other]
chat_model = "my-model"
base_url = "https://llm.example.com/v1"   # the address the Other key belongs to

[services.local]
backend = "vllm"                          # the model server the list returns to
```
- `api_key_env` still works. See the [configuration reference](../reference/configuration.md#cloud-api-keys) for every key.
- An `extra_body` field of your own wins over the thinking switch.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The test says wrong key (401) after you pasted one | The key may have a space or a line break in it, or belong to another service. Paste it again. |
| The app cannot find a key you exported | The app starts without your shell's variables. Paste the key in Settings. |
| A service has a key but is not in the list | The list names only the services with a key; paste the key in Settings. |
| The list says a service did not answer | The row says why. The other services still work; try again in a minute. |
| A run stops with "melo did not finish an answer" | The model replied with nothing and had no thinking to use. Try another model. |
| The answer ends with `(from melo's thinking)` | The model answered only in its thinking. Turn thinking off for that model, or choose another. |

Next: [Configure melo](configure.md), [The web console](web-console.md).

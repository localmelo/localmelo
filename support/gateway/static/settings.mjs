/** Wire the connection dialog. The caller owns entry buttons and health refresh. */
export function setupSettings({ api, onSaved }) {
  const $ = (id) => document.getElementById(id);
  const dialog = $("settings-dialog");
  const chat = $("s-adapter");
  const embedding = $("s-embed-backend");
  const save = $("create-btn");
  const discover = $("discover-btn");
  const error = $("create-error");
  const reload = $("settings-reload");
  const status = $("settings-status");
  const modelStatus = $("discover-status");
  const keyStatus = $("s-key-status");
  const inputIds = ["s-adapter", "s-chat-url", "s-key-env", "s-base-url", "s-chat-model",
    "s-embed-backend", "s-embed-url", "s-embed-model", "s-workspace"];
  let settings = null;
  let sections = {};
  let generation = 0;
  let discoveryVersion = 0;
  let keyVersion = 0;
  let busy = false;

  function showError(message, conflict = false) {
    error.textContent = message;
    error.hidden = !message;
    reload.hidden = !conflict;
  }

  function enable(ready) {
    for (const id of inputIds) $(id).disabled = !ready;
    save.disabled = !ready;
    discover.disabled = !ready || isCloud();
  }

  function isCloud() {
    return settings?.choices.chat_backends.find((choice) => choice.key === chat.value)?.kind === "cloud";
  }

  function options(node, choices, selected) {
    node.replaceChildren();
    for (const choice of choices) {
      const option = document.createElement("option");
      option.value = choice.key;
      option.textContent = choice.label;
      node.append(option);
    }
    node.value = choices.some((choice) => choice.key === selected) ? selected : choices[0]?.key || "";
  }

  function section(key) {
    return sections[key] || (sections[key] = {});
  }

  function renderChat() {
    discoveryVersion += 1;
    const cloud = isCloud();
    const data = section(chat.value);
    $("s-chat-url-field").hidden = cloud;
    $("s-cloud-fields").hidden = !cloud;
    $("s-chat-url").value = data.chat_url || "";
    $("s-chat-model").value = data.chat_model || "";
    $("s-key-env").value = data.api_key_env || "";
    $("s-base-url").value = data.base_url || "";
    $("s-model-options").replaceChildren();
    modelStatus.textContent = cloud ? "Type the model name." : "Discovery lists models without starting a chat.";
    discover.disabled = busy || cloud;
    if (cloud) checkKey();
    else { keyVersion += 1; keyStatus.textContent = ""; }
  }

  function renderEmbedding() {
    const enabled = embedding.value !== "none" && !!embedding.value;
    $("s-embed-fields").hidden = !enabled;
    const data = enabled ? section(embedding.value) : {};
    $("s-embed-url").value = data.embedding_url || "";
    $("s-embed-model").value = data.embedding_model || "";
  }

  async function checkKey() {
    const version = ++keyVersion;
    const currentGeneration = generation;
    const name = $("s-key-env").value;
    if (!/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(name)) {
      keyStatus.textContent = name ? "Enter a valid environment variable name." : "Enter an environment variable name.";
      return;
    }
    keyStatus.textContent = "Checking key status…";
    try {
      const result = await api(`/settings/key-status?${new URLSearchParams({ name })}`);
      if (version === keyVersion && currentGeneration === generation && dialog.open) {
        keyStatus.textContent = result.set ? "Key is set." : "Key is not set in the gateway environment.";
      }
    } catch {
      if (version === keyVersion && currentGeneration === generation && dialog.open) {
        keyStatus.textContent = "Could not check key status.";
      }
    }
  }

  async function open() {
    const currentGeneration = ++generation;
    discoveryVersion += 1;
    keyVersion += 1;
    settings = null;
    busy = true;
    enable(false);
    showError("");
    status.textContent = "Loading settings…";
    modelStatus.textContent = "";
    keyStatus.textContent = "";
    if (!dialog.open) dialog.showModal();
    try {
      const loaded = await api("/settings");
      if (currentGeneration !== generation || !dialog.open) return;
      settings = loaded;
      sections = {};
      // Only writable fields go back to the server; never copy api_key_set.
      for (const choice of loaded.choices.chat_backends) {
        const fields = choice.kind === "cloud"
          ? ["api_key_env", "chat_model", "base_url"]
          : ["chat_url", "chat_model", "embedding_url", "embedding_model"];
        const saved = loaded.sections[choice.key] || {};
        const defaults = loaded.defaults[choice.key] || {};
        sections[choice.key] = Object.fromEntries(fields.map((field) => [field, saved[field] || defaults[field] || ""]));
      }
      options(chat, loaded.choices.chat_backends, loaded.chat_backend);
      options(embedding, loaded.choices.embedding_backends, loaded.embedding_backend || "none");
      $("s-workspace").value = loaded.workspace;
      busy = false;
      renderChat();
      renderEmbedding();
      enable(true);
      status.textContent = loaded.errors?.length ? "Review your settings before saving." : "";
    } catch (err) {
      if (currentGeneration !== generation || !dialog.open) return;
      busy = false;
      status.textContent = "";
      showError(err.message || "Could not load settings.", true);
    }
  }

  dialog.addEventListener("close", () => {
    generation += 1;
    discoveryVersion += 1;
    keyVersion += 1;
  });
  reload.addEventListener("click", open);
  chat.addEventListener("change", renderChat);
  embedding.addEventListener("change", renderEmbedding);
  for (const [id, field] of [["s-chat-url", "chat_url"], ["s-chat-model", "chat_model"],
    ["s-key-env", "api_key_env"], ["s-base-url", "base_url"]]) {
    $(id).addEventListener("input", () => {
      section(chat.value)[field] = $(id).value;
      if (id === "s-key-env") checkKey();
      if (id === "s-chat-url") {
        discoveryVersion += 1;
        $("s-model-options").replaceChildren();
        modelStatus.textContent = "Discovery lists models without starting a chat.";
        discover.disabled = busy || isCloud();
      }
    });
  }
  for (const [id, field] of [["s-embed-url", "embedding_url"], ["s-embed-model", "embedding_model"]]) {
    $(id).addEventListener("input", () => { section(embedding.value)[field] = $(id).value; });
  }

  discover.addEventListener("click", async () => {
    if (busy || !settings || isCloud()) return;
    const version = ++discoveryVersion;
    const currentGeneration = generation;
    const params = new URLSearchParams({ backend: chat.value, url: $("s-chat-url").value });
    if (!$("s-chat-url").value) {
      modelStatus.textContent = "Enter a chat API URL before discovering models.";
      return;
    }
    discover.disabled = true;
    modelStatus.textContent = "Discovering models…";
    try {
      const result = await api(`/models?${params}`);
      if (version !== discoveryVersion || currentGeneration !== generation || !dialog.open) return;
      $("s-model-options").replaceChildren();
      for (const model of result.models) {
        const option = document.createElement("option");
        option.value = model;
        $("s-model-options").append(option);
      }
      modelStatus.textContent = result.models.length ? "Choose a discovered model or type a name." : "No models found. Type a model name.";
    } catch (err) {
      if (version !== discoveryVersion || currentGeneration !== generation || !dialog.open) return;
      modelStatus.textContent = err.message || "Could not discover models. Type a model name.";
    } finally {
      if (version === discoveryVersion && currentGeneration === generation && dialog.open) discover.disabled = busy || isCloud();
    }
  });

  save.addEventListener("click", async () => {
    if (busy || !settings) return;
    const currentGeneration = generation;
    const chatData = section(chat.value);
    const selectedSections = { [chat.value]: isCloud()
      ? { api_key_env: chatData.api_key_env, base_url: chatData.base_url, chat_model: chatData.chat_model }
      : { chat_url: chatData.chat_url, chat_model: chatData.chat_model } };
    if (embedding.value !== "none") {
      const data = section(embedding.value);
      selectedSections[embedding.value] = {
        ...selectedSections[embedding.value],
        embedding_url: data.embedding_url,
        embedding_model: data.embedding_model,
      };
    }
    const body = { chat_backend: chat.value, embedding_backend: embedding.value,
      workspace: $("s-workspace").value, sections: selectedSections, revision: settings.revision };
    busy = true;
    enable(false);
    showError("");
    status.textContent = "Saving settings…";
    try {
      const saved = await api("/settings", { method: "PUT", body });
      // Saving succeeded even if the dialog was closed while the request ran.
      try { await onSaved?.(saved); } catch { /* The caller owns refresh failures. */ }
      if (currentGeneration !== generation || !dialog.open) return;
      settings = saved;
      status.textContent = "Settings saved.";
      dialog.close();
    } catch (err) {
      if (currentGeneration !== generation || !dialog.open) return;
      status.textContent = "";
      const conflict = err.status === 409 && err.code === "settings_changed";
      showError(conflict ? "The settings file changed since it was loaded. Reload and retry." : (err.message || "Could not save settings."), conflict);
    } finally {
      if (currentGeneration === generation && dialog.open) {
        busy = false;
        enable(true);
      }
    }
  });

  return open;
}

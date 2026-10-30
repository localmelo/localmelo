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
  const testButton = $("settings-test");
  const testStatus = $("settings-test-status");
  const themeIds = ["system", "light", "dark"];
  let theme = "system";
  try {
    const stored = localStorage.getItem("localmelo-console.theme");
    if (themeIds.includes(stored)) theme = stored;
  } catch { /* Storage may be unavailable. */ }
  function applyTheme(value) {
    theme = value;
    document.documentElement.dataset.theme = value;
    for (const name of themeIds) $(`settings-theme-${name}`).checked = name === value;
  }
  applyTheme(theme);
  for (const name of themeIds) {
    $(`settings-theme-${name}`).addEventListener("change", (event) => {
      if (!event.target.checked) return;
      applyTheme(name);
      try { localStorage.setItem("localmelo-console.theme", name); } catch { /* Keep the current theme. */ }
    });
  }
  const tabNames = ["model", "test", "theme", "developer"];
  for (const name of tabNames) {
    $(`settings-tab-${name}`).addEventListener("click", () => {
      for (const item of tabNames) {
        $(`settings-panel-${item}`).hidden = item !== name;
        $(`settings-tab-${item}`).setAttribute("aria-pressed", String(item === name));
      }
    });
  }
  const inputIds = ["s-adapter", "s-chat-url", "s-key-env", "s-base-url", "s-chat-model",
    "s-embed-backend", "s-embed-url", "s-embed-model", "s-workspace", "s-developer"];
  let settings = null;
  let sections = {};
  let generation = 0;
  let discoveryVersion = 0;
  let keyVersion = 0;
  let testVersion = 0;
  let busy = false;

  function invalidateTest() {
    testVersion += 1;
    testStatus.textContent = "";
    testStatus.dataset.state = "";
    testButton.disabled = busy || !settings || !chat.value;
  }

  function showError(message, conflict = false) {
    error.textContent = message;
    error.hidden = !message;
    reload.hidden = !conflict;
  }

  function enable(ready) {
    const chosen = !!chat.value; // Nothing is saved until the user chooses a server.
    for (const id of inputIds) $(id).disabled = !ready;
    for (const id of ["s-chat-url", "s-chat-model"]) $(id).disabled = !ready || !chosen;
    save.disabled = !ready || !chosen;
    discover.disabled = !ready || !chosen || isCloud();
    testButton.disabled = !ready || !chosen;
  }

  function isCloud() {
    return settings?.choices.chat_backends.find((choice) => choice.key === chat.value)?.kind === "cloud";
  }

  function renderDeveloper() {
    $("s-developer-options").hidden = !$("s-developer").checked;
  }

  function options(node, choices, selected, blank = "") {
    const all = blank ? [{ key: "", label: blank }, ...choices] : choices;
    node.replaceChildren();
    for (const choice of all) {
      const option = document.createElement("option");
      option.value = choice.key;
      option.textContent = choice.label;
      node.append(option);
    }
    node.value = all.some((choice) => choice.key === selected) ? selected : all[0]?.key || "";
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
    discover.disabled = busy || cloud || !chat.value;
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
    invalidateTest();
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
      $("settings-file").textContent = `File: ${loaded.file_label || "~/.cache/localmelo/config.toml"}`;
      $("settings-revision").textContent = `Revision: ${loaded.revision || "not saved"}`;
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
      options(chat, loaded.choices.chat_backends, loaded.chat_backend, loaded.chat_backend ? "" : "Choose a server…");
      options(embedding, loaded.choices.embedding_backends, loaded.embedding_backend || "none");
      $("s-workspace").value = loaded.workspace;
      $("s-developer").checked = loaded.mode === "developer";
      const list = $("s-developer-list");
      list.replaceChildren();
      for (const item of loaded.advanced_options || []) {
        const row = document.createElement("li");
        row.textContent = `${item.option}: ${item.tier}${item.issue ? `, decided in ${item.issue}` : ""}`;
        list.append(row);
      }
      renderDeveloper();
      busy = false;
      renderChat();
      renderEmbedding();
      enable(true);
      // "not set (run 'melo --reconfigure')" is advice for the terminal: here the empty select says it.
      const errors = (loaded.errors || []).filter((item) => typeof item === "string" && item && !item.includes("melo --reconfigure"));
      status.textContent = !loaded.chat_backend ? "Choose a server to continue." : errors.length ? "Review your settings before saving." : "";
      showError(errors.join("\n"));
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
    invalidateTest();
  });
  reload.addEventListener("click", open);
  chat.addEventListener("change", () => { invalidateTest(); renderChat(); enable(!busy); });
  embedding.addEventListener("change", () => { invalidateTest(); renderEmbedding(); });
  $("s-workspace").addEventListener("input", invalidateTest);
  $("s-developer").addEventListener("change", renderDeveloper);
  for (const [id, field] of [["s-chat-url", "chat_url"], ["s-chat-model", "chat_model"],
    ["s-key-env", "api_key_env"], ["s-base-url", "base_url"]]) {
    $(id).addEventListener("input", () => {
      section(chat.value)[field] = $(id).value;
      invalidateTest();
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
    $(id).addEventListener("input", () => { section(embedding.value)[field] = $(id).value; invalidateTest(); });
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

  function candidate() {
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
    return { chat_backend: chat.value, embedding_backend: embedding.value,
      workspace: $("s-workspace").value, sections: selectedSections };
  }

  function saveBody() {
    return { ...candidate(), revision: settings.revision, mode: $("s-developer").checked ? "developer" : "default" };
  }

  testButton.addEventListener("click", async () => {
    if (busy || !settings || testButton.disabled) return;
    const version = ++testVersion;
    const currentGeneration = generation;
    testButton.disabled = true;
    testStatus.textContent = "Testing…";
    testStatus.dataset.state = "testing";
    try {
      const result = await api("/settings/test", { method: "POST", body: candidate() });
      if (version !== testVersion || currentGeneration !== generation || !dialog.open) return;
      if (result.ok === true) {
        testStatus.textContent = "Backend reachable";
        testStatus.dataset.state = "success";
      } else {
        const message = typeof result.error === "string" ? result.error.split(/\r?\n/)[0] : "Unknown error";
        testStatus.textContent = `Could not connect: ${message || "Unknown error"}`;
        testStatus.dataset.state = "error";
      }
    } catch (err) {
      if (version !== testVersion || currentGeneration !== generation || !dialog.open) return;
      const message = typeof err.message === "string" ? err.message.split(/\r?\n/)[0] : "Unknown error";
      testStatus.textContent = `Could not connect: ${message || "Unknown error"}`;
      testStatus.dataset.state = "error";
    } finally {
      if (version === testVersion && currentGeneration === generation && dialog.open) testButton.disabled = busy || !settings;
    }
  });

  save.addEventListener("click", async () => {
    if (busy || !settings || !chat.value) return;
    const currentGeneration = generation;
    const body = saveBody();
    busy = true;
    invalidateTest();
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
      const errors = err.code === "invalid_settings" && Array.isArray(err.details?.errors)
        ? err.details.errors.filter((item) => typeof item === "string" && item).join("\n") : "";
      showError(conflict ? "The settings file changed since it was loaded. Reload and retry." : (errors || err.message || "Could not save settings."), conflict);
    } finally {
      if (currentGeneration === generation && dialog.open) {
        busy = false;
        enable(true);
      }
    }
  });

  return open;
}

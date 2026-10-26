import { renderMarkdown } from "./markdown.mjs";

const $ = (id) => document.getElementById(id);
const KEY = "localmelo-pg";
const S = {
  sessions: [],
  activeId: null,
  discovery: null,
  working: false,
  pending: new Set(),
  cancelling: new Set(),
  drafts: {},
  scenarios: [],
  polling: false,
  online: true,
  detailVersions: new Map(),
};
const dialog = $("settings-dialog");
const input = $("input");
const fields = [
  "s-adapter",
  "s-chat-url",
  "s-embed-url",
  "s-chat-model",
  "s-embed-model",
  "s-memory",
];
let previousFocus = null;
let noticeTimer;
let sessionListSnapshot = "";
let messageSnapshot = "";

function stored(key, fallback = {}, storageName = "localStorage") {
  try {
    const value = JSON.parse(globalThis[storageName].getItem(key));
    return value && typeof value === "object" && !Array.isArray(value) ? value : fallback;
  } catch {
    return fallback;
  }
}
function persist(key, value, storageName = "localStorage") {
  try {
    globalThis[storageName].setItem(key, JSON.stringify(value));
  } catch {
    /* Browser policy can deny access to the storage object itself. */
  }
}
function active() {
  return S.sessions.find((s) => s.session_id === S.activeId);
}
function title(s) {
  return s.title || "New chat";
}
function esc(text) {
  return String(text ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}
function notify(message, sticky = false) {
  clearTimeout(noticeTimer);
  $("notice").textContent = message;
  $("notice").hidden = !message;
  if (!sticky && message)
    noticeTimer = setTimeout(() => {
      $("notice").hidden = true;
    }, 7000);
}
async function api(path, body, method = body ? "POST" : "GET") {
  const response = await fetch("/v1/playground" + path, {
    method,
    ...(body
      ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}
function setOptions(id, models, selected, empty) {
  const select = $(id);
  select.replaceChildren();
  if (empty || !models.length) select.add(new Option(empty || "No models available", ""));
  models.forEach((model) => select.add(new Option(model, model)));
  if (models.includes(selected)) select.value = selected;
}
function saveConfig() {
  persist(KEY, {
    adapter: $("s-adapter").value,
    chat_url: $("s-chat-url").value.trim(),
    embed_url: $("s-embed-url").value.trim(),
    chat_model: $("s-chat-model").value,
    embed_model: $("s-embed-model").value,
    memory: $("s-memory").value,
  });
}
function saveDraft() {
  S.drafts[S.activeId || "new"] = input.value;
  persist(KEY + "-drafts", S.drafts, "sessionStorage");
}
function resizeInput() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 190) + "px";
}
function sidebar(open) {
  $("app").classList.toggle("sidebar-open", open);
  $("sidebar-backdrop").hidden = !open;
  $("sidebar-toggle").setAttribute("aria-expanded", String(open));
  const mobile = matchMedia("(max-width: 700px)").matches;
  $("sidebar").inert = mobile && !open;
  document.querySelector(".chat").inert = mobile && open;
  $("sidebar-backdrop").tabIndex = open ? 0 : -1;
  if (mobile && open) $("new-btn").focus();
  if (mobile && !open && document.activeElement?.closest(".sidebar")) $("sidebar-toggle").focus();
}
function syncUI() {
  const s = active();
  const busy = !!s && (s.is_busy || S.pending.has(s.session_id));
  $("header-title").textContent = s ? title(s) : "New chat";
  $("header-model").textContent = s ? s.chat_model : "Connect your model to get started";
  $("connection-label").textContent = !S.online
    ? "Gateway offline"
    : s
      ? busy
        ? "Working"
        : "Ready"
      : "Set up model";
  $("connection-dot").classList.toggle("offline", !s || !S.online);
  $("model-btn").textContent = s ? `${s.chat_model}  ⌄` : "Choose a model  ⌄";
  $("send-btn").hidden = busy;
  $("stop-btn").hidden = !busy;
  $("stop-btn").disabled = !s || S.cancelling.has(s.session_id);
  $("stop-btn").querySelector("span").textContent =
    s && S.cancelling.has(s.session_id) ? "Stopping…" : "Stop";
  $("send-btn").disabled = S.working || busy || !S.online || !input.value.trim();
  $("run-status").textContent = busy ? "Running on your backend" : "";
  fields.forEach((id) => {
    $(id).disabled = S.working;
  });
  $("discover-btn").disabled = S.working;
  $("create-btn").disabled = S.working || !S.discovery || !$("s-chat-model").value;
  $("new-btn").disabled = S.working;
}
function renderSessions() {
  const query = $("search").value.toLowerCase();
  const snapshot = JSON.stringify([
    query,
    S.activeId,
    S.sessions.map((s) => [
      s.session_id,
      title(s),
      s.chat_model,
      s.is_busy,
      s.last_active,
      S.pending.has(s.session_id),
    ]),
  ]);
  if (snapshot === sessionListSnapshot) return;
  sessionListSnapshot = snapshot;
  const focused = document.activeElement;
  const focusedId = focused?.closest(".session-item")?.dataset.sessionId;
  const focusedClose = focused?.classList.contains("session-close");
  $("session-count").textContent = S.sessions.length;
  $("session-list").replaceChildren();
  const shown = [...S.sessions]
    .sort((a, b) => b.last_active - a.last_active)
    .filter((s) => `${title(s)} ${s.chat_model}`.toLowerCase().includes(query));
  if (!shown.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = query ? "No matching chats." : "Your conversations will appear here.";
    $("session-list").append(empty);
  }
  for (const s of shown) {
    const row = document.createElement("div");
    row.dataset.sessionId = s.session_id;
    row.className = "session-item" + (s.session_id === S.activeId ? " active" : "");
    const button = document.createElement("button");
    button.className = "session-info";
    if (s.session_id === S.activeId) button.setAttribute("aria-current", "page");
    button.innerHTML = `<div class="session-title">${esc(title(s))}</div><div class="session-meta">${s.is_busy ? "● Working · " : ""}${esc(s.chat_model)}</div>`;
    button.onclick = () => selectSession(s.session_id);
    const close = document.createElement("button");
    close.className = "session-close";
    close.textContent = "×";
    close.setAttribute("aria-label", `Close chat: ${title(s)}`);
    close.title = "Close chat";
    close.disabled = s.is_busy || S.pending.has(s.session_id);
    close.onclick = () => deleteSession(s.session_id);
    row.append(button, close);
    $("session-list").append(row);
    if (s.session_id === focusedId)
      (focusedClose && !close.disabled ? close : button).focus({ preventScroll: true });
  }
}
function renderMessages(forceScroll = false) {
  const scroller = $("scroll");
  const oldTop = scroller.scrollTop;
  const nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 100;
  const s = active();
  const messages = s?.messages || [];
  const snapshot = JSON.stringify([
    S.activeId,
    s?.chat_model,
    messages,
    s?.is_busy,
    s && S.pending.has(s.session_id),
  ]);
  if (snapshot === messageSnapshot) {
    if (forceScroll) scroller.scrollTop = messages.length ? scroller.scrollHeight : 0;
    return;
  }
  messageSnapshot = snapshot;
  $("welcome").hidden = messages.length > 0 || !!s?.is_busy;
  $("setup-btn").hidden = !!s;
  $("welcome-description").textContent = s
    ? `${s.chat_model} is ready. Start a conversation below.`
    : "A quiet space to think, build, and explore with your own model.";
  $("messages").replaceChildren();
  for (const message of messages) {
    const row = document.createElement("article");
    row.className = "msg " + (["user", "assistant"].includes(message.role) ? message.role : "note");
    if (message.role === "user") {
      const bubble = document.createElement("div");
      bubble.className = "user-bubble";
      bubble.textContent = message.content;
      row.append(bubble);
    } else if (message.role === "assistant") {
      row.innerHTML =
        '<div class="assistant-label"><img class="brand-icon" src="/assets/icon_small.png" width="28" height="28" alt=""> localmelo</div>';
      if (message.thinking) {
        const thinking = document.createElement("details");
        thinking.className = "thinking";
        thinking.innerHTML = '<summary>Reasoning</summary><div class="thinking-body"></div>';
        thinking.lastElementChild.textContent = message.thinking;
        row.append(thinking);
      }
      const answer = document.createElement("div");
      answer.className = "answer";
      answer.innerHTML = renderMarkdown(message.answer || message.content || "");
      row.append(answer);
      const actions = document.createElement("div");
      actions.className = "message-actions";
      const copy = document.createElement("button");
      copy.className = "copy-button";
      copy.textContent = "Copy response";
      copy.onclick = async () => {
        try {
          await navigator.clipboard.writeText(message.answer || message.content || "");
          copy.textContent = "Copied";
          setTimeout(() => {
            copy.textContent = "Copy response";
          }, 2000);
        } catch {
          notify("Clipboard unavailable. Select the response text to copy it.");
        }
      };
      actions.append(copy);
      row.append(actions);
    } else {
      row.textContent = message.content;
    }
    $("messages").append(row);
  }
  if (s?.is_busy || (s && S.pending.has(s.session_id))) {
    const working = document.createElement("div");
    working.className = "msg working";
    working.setAttribute("role", "status");
    working.innerHTML = '<span class="status-dot"></span><span>Working on your request…</span>';
    $("messages").append(working);
  }
  if (forceScroll || nearBottom) scroller.scrollTop = messages.length ? scroller.scrollHeight : 0;
  else scroller.scrollTop = oldTop;
}
async function refreshSession(id, forceScroll = false) {
  try {
    const detail = await api("/session/" + encodeURIComponent(id));
    const index = S.sessions.findIndex((s) => s.session_id === id);
    if (index < 0) return;
    S.sessions[index] = detail;
    S.detailVersions.set(id, detail.last_active);
    if (S.activeId === id) {
      renderMessages(forceScroll);
      syncUI();
    }
    renderSessions();
  } catch (error) {
    if (error.status === 404) {
      S.sessions = S.sessions.filter((s) => s.session_id !== id);
      S.detailVersions.delete(id);
      if (S.activeId === id) {
        const draft = input.value;
        saveDraft();
        S.activeId = null;
        if (draft) S.drafts.new = draft;
        input.value = S.drafts.new || "";
        saveDraft();
        resizeInput();
        renderMessages();
      }
      renderSessions();
      syncUI();
      notify("This chat expired or the gateway restarted. Create a new chat to continue.", true);
    } else {
      throw error;
    }
  }
}
async function selectSession(id, { saveCurrent = true } = {}) {
  if (saveCurrent) saveDraft();
  S.activeId = id;
  persist(KEY + "-view", { id });
  input.value = S.drafts[id] || "";
  resizeInput();
  sidebar(false);
  renderSessions();
  renderMessages(true);
  syncUI();
  try {
    await refreshSession(id, true);
  } catch (error) {
    notify("Could not load chat: " + error.message, true);
  }
}
async function deleteSession(id) {
  const s = S.sessions.find((s) => s.session_id === id);
  if (!s || s.is_busy || S.pending.has(id)) return;
  try {
    await api("/session/" + encodeURIComponent(id), null, "DELETE");
    S.sessions = S.sessions.filter((session) => session.session_id !== id);
    delete S.drafts[id];
    persist(KEY + "-drafts", S.drafts, "sessionStorage");
    if (S.activeId === id) {
      S.activeId = null;
      if (S.sessions.length) await selectSession(S.sessions[0].session_id, { saveCurrent: false });
      else {
        input.value = S.drafts.new || "";
        resizeInput();
        renderMessages();
      }
    }
    renderSessions();
    syncUI();
  } catch (error) {
    notify("Could not close chat: " + error.message, true);
  }
}
function openSettings() {
  previousFocus = document.activeElement;
  $("create-error").hidden = true;
  sidebar(false);
  if (!dialog.open) dialog.showModal();
}
function closeSettings() {
  dialog.close();
}
function connectionChanged() {
  S.discovery = null;
  setOptions("s-chat-model", [], "", "Discover models first");
  setOptions("s-embed-model", [], "", "Disabled");
  $("discover-status").textContent = "Connection changed. Discover models to refresh the list.";
  saveConfig();
  syncUI();
}
async function discover() {
  if (S.working) return;
  if (!$("s-chat-url").value.trim()) {
    $("discover-status").textContent = "Enter your Chat API URL first.";
    $("s-chat-url").focus();
    return;
  }
  const previous = stored(KEY);
  S.working = true;
  S.discovery = null;
  syncUI();
  $("discover-btn").textContent = "Discovering…";
  $("discover-status").textContent = "Contacting your backend…";
  $("create-error").hidden = true;
  try {
    const data = await api("/discover", {
      adapter: $("s-adapter").value,
      chat_url: $("s-chat-url").value.trim(),
      embedding_url: $("s-embed-url").value.trim(),
    });
    S.discovery = data;
    const chatModels = data.chat_models || [];
    const preferred = chatModels.includes(previous.chat_model)
      ? previous.chat_model
      : chatModels.find((model) => !/embed|rerank/i.test(model));
    setOptions("s-chat-model", chatModels, preferred);
    setOptions("s-embed-model", data.embedding_models || [], previous.embed_model, "Disabled");
    $("discover-status").textContent =
      `${chatModels.length} models available. Choose a chat model below.`;
    saveConfig();
  } catch (error) {
    setOptions("s-chat-model", [], "", "Discovery failed");
    setOptions("s-embed-model", [], "", "Disabled");
    $("discover-status").textContent = "Could not connect: " + error.message;
  } finally {
    S.working = false;
    $("discover-btn").textContent = "Discover models";
    syncUI();
  }
}
function memoryPreview() {
  const id = $("s-memory").value;
  const count =
    id === "__all__"
      ? S.scenarios.reduce((sum, s) => sum + s.seed_count, 0)
      : S.scenarios.find((s) => s.id === id)?.seed_count;
  $("memory-count").textContent = count
    ? `${count} seed memories will be loaded.`
    : "No seed memories will be loaded.";
}
async function createSession() {
  if (S.working || !S.discovery || !$("s-chat-model").value) return;
  S.working = true;
  syncUI();
  $("create-btn").textContent = "Creating…";
  $("create-error").hidden = true;
  try {
    saveDraft();
    const data = await api("/session", {
      adapter: $("s-adapter").value,
      chat_url: $("s-chat-url").value.trim(),
      embedding_url: $("s-embed-url").value.trim(),
      chat_model: $("s-chat-model").value,
      embedding_model: $("s-embed-model").value,
      scenario_id: $("s-memory").value,
    });
    S.sessions.push({
      ...data,
      title: "New chat",
      last_active: Date.now() / 1000,
      is_busy: false,
      messages: [],
    });
    if (!S.activeId && input.value) {
      S.drafts[data.session_id] = input.value;
      delete S.drafts.new;
    }
    persist(KEY + "-drafts", S.drafts, "sessionStorage");
    saveConfig();
    closeSettings();
    await selectSession(data.session_id, { saveCurrent: false });
    input.focus();
    notify("Chat ready. Your message is sent only when you press Send.");
  } catch (error) {
    $("create-error").textContent = error.message;
    $("create-error").hidden = false;
  } finally {
    S.working = false;
    $("create-btn").innerHTML = 'Create chat <span aria-hidden="true">→</span>';
    syncUI();
  }
}
async function send() {
  const query = input.value.trim();
  const s = active();
  if (!query || S.working) return;
  if (!s) {
    saveDraft();
    openSettings();
    return;
  }
  const id = s.session_id;
  if (s.is_busy || S.pending.has(id)) return;
  S.pending.add(id);
  s.is_busy = true;
  s.messages ||= [];
  s.messages.push({ role: "user", content: query });
  s.title = s.title === "New chat" ? query.slice(0, 80) : s.title;
  input.value = "";
  saveDraft();
  resizeInput();
  renderMessages(true);
  renderSessions();
  syncUI();
  try {
    await api("/session/" + encodeURIComponent(id) + "/run", { query });
  } catch (error) {
    if (!S.drafts[id]) S.drafts[id] = query;
    if (S.activeId === id && !input.value) {
      input.value = S.drafts[id];
      resizeInput();
    }
    persist(KEY + "-drafts", S.drafts, "sessionStorage");
    notify("Request failed: " + error.message + " Your message is kept as a draft.", true);
  } finally {
    S.pending.delete(id);
    try {
      await refreshSession(id, S.activeId === id);
    } catch (error) {
      // Keep the last known busy state: losing the HTTP connection is not cancellation.
      notify("Connection interrupted. Reconnecting to your chat…", true);
    }
    syncUI();
  }
}
async function stop() {
  const s = active();
  if (!s || S.cancelling.has(s.session_id)) return;
  const id = s.session_id;
  S.cancelling.add(id);
  syncUI();
  try {
    await api("/session/" + encodeURIComponent(id) + "/cancel", {});
    await refreshSession(id);
  } catch (error) {
    notify("Could not stop the response: " + error.message, true);
  } finally {
    S.cancelling.delete(id);
    syncUI();
  }
}
async function poll() {
  if (document.hidden || S.polling) return;
  S.polling = true;
  try {
    const data = await api("/sessions");
    S.online = true;
    const old = new Map(S.sessions.map((s) => [s.session_id, s]));
    S.sessions = data.sessions.map((s) => ({ ...old.get(s.session_id), ...s }));
    renderSessions();
    if (S.activeId && !S.pending.has(S.activeId)) {
      const summary = active();
      const previous = old.get(S.activeId);
      if (
        !summary ||
        summary.is_busy ||
        previous?.is_busy ||
        summary.last_active !== S.detailVersions.get(S.activeId) ||
        !summary.messages
      )
        await refreshSession(S.activeId);
    }
    syncUI();
  } catch {
    S.online = false;
    syncUI();
  } finally {
    S.polling = false;
  }
}
async function init() {
  const config = stored(KEY);
  S.drafts = stored(KEY + "-drafts", {}, "sessionStorage");
  for (const [key, id] of [
    ["adapter", "s-adapter"],
    ["chat_url", "s-chat-url"],
    ["embed_url", "s-embed-url"],
  ])
    if (typeof config[key] === "string") $(id).value = config[key];
  if (!$("s-adapter").value) $("s-adapter").value = "auto";
  input.value = S.drafts.new || "";
  const results = await Promise.allSettled([api("/scenarios"), api("/sessions")]);
  if (results[0].status === "fulfilled") {
    S.scenarios = results[0].value.scenarios;
    S.scenarios.forEach((s) => $("s-memory").add(new Option(s.name, s.id)));
    if (config.memory) $("s-memory").value = config.memory;
    memoryPreview();
  }
  if (results[1].status === "fulfilled") {
    S.sessions = results[1].value.sessions;
    const saved = stored(KEY + "-view").id;
    const id = S.sessions.some((s) => s.session_id === saved) ? saved : S.sessions[0]?.session_id;
    if (id) await selectSession(id);
  } else {
    S.online = false;
    notify("Could not load chats. Check that the gateway is running.", true);
  }
  renderSessions();
  renderMessages();
  syncUI();
  resizeInput();
  sidebar(false);
  setInterval(poll, 2500);
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
input.addEventListener("input", () => {
  saveDraft();
  resizeInput();
  syncUI();
});
for (const id of ["new-btn", "settings-btn", "connection-btn", "model-btn", "setup-btn"])
  $(id).addEventListener("click", openSettings);
$("close-settings").addEventListener("click", closeSettings);
dialog.addEventListener("close", () => {
  if (previousFocus?.isConnected && !previousFocus.closest("[inert]")) previousFocus.focus();
  else $("sidebar-toggle").focus();
});
$("discover-btn").addEventListener("click", discover);
$("create-btn").addEventListener("click", createSession);
$("stop-btn").addEventListener("click", stop);
$("search").addEventListener("input", renderSessions);
$("sidebar-toggle").addEventListener("click", () =>
  sidebar(!$("app").classList.contains("sidebar-open")),
);
$("sidebar-backdrop").addEventListener("click", () => sidebar(false));
matchMedia("(max-width: 700px)").addEventListener("change", () => sidebar(false));
for (const id of ["s-chat-url", "s-embed-url"]) $(id).addEventListener("input", connectionChanged);
$("s-adapter").addEventListener("change", connectionChanged);
for (const id of ["s-chat-model", "s-embed-model"])
  $(id).addEventListener("change", () => {
    saveConfig();
    syncUI();
  });
$("s-memory").addEventListener("change", () => {
  memoryPreview();
  saveConfig();
});
document.querySelectorAll("[data-prompt]").forEach((button) =>
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    saveDraft();
    resizeInput();
    syncUI();
    input.focus();
  }),
);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    openSettings();
  }
  if (event.key === "Escape" && !dialog.open) sidebar(false);
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) poll();
});
init();

/**
 * The workspace page: API v2 in the browser.
 *
 * The page talks only to `/api`: the session list and the machine's S2
 * snapshot (`/api/sessions`), one session's detail and turns, its event
 * stream, its submissions (a send is safe to repeat), the level and
 * presence writes, stop and delete. The turns themselves are folded by
 * `turns.mjs`, which mirrors the terminal's wording; this file only builds
 * the DOM, keeps the drafts and the outbox, and keeps the header in sync.
 *
 * No framework and no build step: this module and `turns.mjs` are plain ES
 * modules, served as they are.
 */

import { renderMarkdown } from "./markdown.mjs";
import { setupSettings } from "./settings.mjs";
import { setupDrawer } from "./drawer.mjs";
import { workerHistory } from "./worker-history.mjs";
import { setupTasks } from "./tasks.mjs";
import { setupStatusRow } from "./status-row.mjs";
import { setupRunning } from "./running.mjs";
import { setupModelCard } from "./setup-card.mjs";
import {
  activity,
  applyEvent,
  approvalTag,
  canContinue,
  canRetry,
  cleanOutput,
  formatBytes,
  formatDuration,
  formatPast,
  formatTokens,
  liveFacts,
  newTurn,
  newView,
  printable,
  reflectionLine,
  statusLine,
  statusText,
  toolState,
  turnDuration,
  workSummary,
} from "./turns.mjs";

const $ = (id) => document.getElementById(id);
const KEY = "localmelo-console";
const STREAM_DELAYS_S = [1, 2, 4, 8, 16, 30];
const PRESENCE_MS = 2500;
const LIST_POLL_MS = 4000;

const S = {
  health: null,
  authenticated: false,
  ready: false, // The first load has chosen what to show, so the welcome is not a guess.
  machine: null,
  sessions: [],
  activeId: null,
  newWorkspace: null,
  drafts: {},
  outbox: {},
  cancelling: new Set(),
  stream: { source: null, sessionId: null, status: "idle", attempt: 0, retryAt: 0, timer: null },
  view: null,
  online: true,
  levelsOpen: false,
  lastPresenceAt: 0,
  noticeKind: "",
};

const input = $("input");
const COMPOSER_PLACEHOLDER = input.placeholder;
let noticeTimer = null;
let sessionListSnapshot = "";
let messageSnapshot = "";
const liveNodes = new Map();
// The tool cards still running, so `tick` can move their clocks.
const runningCards = new Map();
// Request state survives ordinary SSE-driven rerenders, including the guard.
const approvalStates = new Map();
let earlierLoading = false;
let retrying = false;

// ── Small helpers ──

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

function esc(text) {
  return String(text ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character],
  );
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(className, text, onClick) {
  const node = el("button", className, text);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function setTextIfChanged(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

function activeSession() {
  return S.sessions.find((session) => session.id === S.activeId) ?? S.view?.session ?? null;
}

function title(session) {
  return session?.title || "New chat";
}

function levelName(level) {
  return `P${level || 2}`;
}

function notify(message, { sticky = false, kind = "" } = {}) {
  S.noticeKind = message ? kind || "message" : "";
  clearTimeout(noticeTimer);
  $("notice").textContent = message;
  $("notice").hidden = !message;
  if (!sticky && message) {
    noticeTimer = setTimeout(() => {
      $("notice").hidden = true;
      S.noticeKind = "";
    }, 7000);
  }
}

function lock(message = "") {
  if (S.authenticated) saveDraft();
  S.authenticated = false;
  closeStream();
  $("settings-dialog")?.close?.();
  $("app").inert = true;
  $("app").hidden = true;
  $("lock-screen").hidden = false;
  $("lock-status").textContent = message;
  $("lock-title").focus();
}

/** One `/api` call; an API v2 error keeps its code, message and details. */
async function api(path, { body, method, allowLocked = false } = {}) {
  if (!S.authenticated && !allowLocked) {
    const error = new Error("Workspace is locked");
    error.status = 401;
    throw error;
  }
  const response = await fetch("/api" + path, {
    method: method ?? (body ? "POST" : "GET"),
    ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
  });
  if (response.status === 401) {
    lock();
    const error = new Error("Workspace is locked");
    error.status = 401;
    throw error;
  }
  const payload = response.status === 204 ? {} : await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(
      payload?.error?.message || `Request failed (${response.status})`,
    );
    error.status = response.status;
    error.code = payload?.error?.code || "";
    error.details = payload?.error?.details;
    throw error;
  }
  return payload;
}

function newClientRequestId() {
  try {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  } catch {
    /* A browser without crypto falls back to the hex below. */
  }
  let text = "";
  for (let index = 0; index < 32; index += 1) {
    text += Math.floor(Math.random() * 16).toString(16);
  }
  return text;
}

function nowSeconds() {
  return Date.now() / 1000;
}

// ── Drafts ──

function saveDraft() {
  S.drafts[S.activeId || "new"] = input.value;
  persist(KEY + "-drafts", S.drafts, "sessionStorage");
}

function restoreDraft() {
  input.value = S.drafts[S.activeId || "new"] || "";
  resizeInput();
}

function moveDraft(from, to) {
  if (S.drafts[from] === undefined) return;
  S.drafts[to] = S.drafts[from];
  delete S.drafts[from];
  persist(KEY + "-drafts", S.drafts, "sessionStorage");
}

/**
 * Keep a message the store may not have as *slot*'s draft.
 *
 * A refused or unknown send has no turn to reload: the outbox lives in
 * memory, so without the draft a reload lost the text. The oldest unsent
 * text wins, as the old page kept the first draft, and the composer shows it
 * again when it is empty.
 */
function keepDraft(slot, text) {
  if (!S.drafts[slot]) S.drafts[slot] = text;
  persist(KEY + "-drafts", S.drafts, "sessionStorage");
  if ((S.activeId ?? "new") === slot && !input.value) {
    input.value = S.drafts[slot];
    resizeInput();
  }
}

/** Drop the draft a failed send restored, once the store accepted *text*. */
function forgetDraft(slot, text) {
  if (S.drafts[slot] === text) {
    delete S.drafts[slot];
    persist(KEY + "-drafts", S.drafts, "sessionStorage");
  }
  if ((S.activeId ?? "new") === slot && input.value === text) {
    input.value = "";
    resizeInput();
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 190) + "px";
}

// ── The machine line (S2b) ──

function machineLine(machine) {
  if (!machine) return { text: "", label: "" };
  const parts = [];
  let label = "";
  if (machine.readable === false) {
    parts.push("GPU lock unreadable");
    label = "GPU lock unreadable";
  } else if (machine.lock === "foreign") {
    const holder = machine.holder || "another session";
    const purpose = machine.purpose || "";
    const mode = machine.gate === "wait" ? "(chats wait)" : "(recorded)";
    const stale = machine.stale ? " · ETA passed" : "";
    const prefix = `GPU lock: ${holder}`;
    const suffix = ` ${mode}${stale}`;
    // Keep the mode (and holder) visible in the header: spend the line's
    // character budget on the purpose only after the suffix has room.
    const room = Math.max(0, 78 - prefix.length - suffix.length);
    const short =
      purpose && room > 5
        ? purpose.length > room - 3
          ? purpose.slice(0, room - 4) + "…"
          : purpose
        : "";
    parts.push(`${prefix}${short ? ` · ${short}` : ""}${suffix}`);
    label = `${prefix}${purpose ? ` · ${purpose}` : ""}${suffix}`;
  } else if (machine.lock === "self") {
    const stale = machine.stale ? " · ETA passed" : "";
    parts.push(`GPU lock: yours${stale}`);
    label = parts[parts.length - 1];
  }
  if (machine.on_ac === false && typeof machine.battery_pct === "number") {
    const state = machine.battery_state ? ` · ${machine.battery_state}` : "";
    const battery = `Battery ${Math.round(machine.battery_pct)}%${state}`;
    parts.push(battery);
    label = label ? `${label} · ${battery}` : battery;
  }
  return { text: parts.join(" · "), label: label || parts.join(" · ") };
}

let machineSnapshot = "";

// ── The header and the composer ──

function runningTurn() {
  const turns = S.view?.turns ?? [];
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    if (!turns[index].finishedAt) return turns[index];
  }
  return null;
}

function pendingOutbox() {
  return S.outbox[S.activeId ?? "new"] ?? null;
}

/** Whether no model can answer yet: none is configured, or the configured server is known to be down. */
function needsModel(health) {
  if (!health) return false;
  return health.configured === false || health.model_server?.reachable === false;
}

/** Whether the welcome shows the setup card: a model is needed (or being connected), and no chat is open. */
function setupShown() {
  return S.ready && (needsModel(S.health) || setupCard.holding()) && !S.activeId && !pendingOutbox() && !workerMode;
}

function syncUI() {
  const view = S.activeId ? S.view : null;
  const session = activeSession();
  const turn = runningTurn();
  const held = turn && session?.held ? session.held : "";
  const pending = view ? view.pending.size : 0;
  const machine = machineLine(S.machine);
  const chip = $("machine-state");
  const snapshot = `${machine.text}\n${machine.label}`;
  if (snapshot !== machineSnapshot) {
    machineSnapshot = snapshot;
    // Unhide before the text write, so a screen reader announces the first
    // lock instead of a mutation inside a hidden live region.
    chip.hidden = !machine.text;
    chip.textContent = machine.text;
    chip.setAttribute("aria-label", machine.label || "Machine status");
    chip.title = machine.label;
  }
  const model = S.health?.model ?? null;
  $("header-title").textContent = session ? title(session) : "New chat";
  document.title = `localmelo · ${session ? title(session) : "New chat"}${pending ? ` · ${pending}` : ""}${turn ? " ●" : ""}`;
  const storeBanner = $("store-banner");
  const storeError = S.health?.sessions && S.health.sessions.store !== "ok";
  setTextIfChanged(storeBanner, storeError ? `Chats are unavailable: ${S.health.sessions.store_error || "Store unavailable."}` : (!S.online ? "Could not reach the gateway. Check that the gateway is running." : ""));
  storeBanner.hidden = !storeBanner.textContent;
  const setup = setupShown();
  setupCard.show(setup);
  $("welcome").classList.toggle("needs-model", setup);
  input.disabled = setup;
  input.placeholder = setup ? "Connect a model to start" : COMPOSER_PLACEHOLDER;
  const modelBanner = $("model-banner");
  setTextIfChanged(modelBanner, S.health?.configured === false && !setup ? "No model is configured yet. Open connection settings to connect one." : "");
  modelBanner.hidden = !modelBanner.textContent;
  refreshStreamNotice();
  $("header-model").textContent = session
    ? session.model || "Your configured model"
    : model?.name
      ? `${model.name} · ${model.backend_label}`
      : "Connect your model to get started";
  $("connection-label").textContent = !S.online
    ? "Gateway offline"
    : turn
      ? held
        ? `Waiting: ${held}`
        : pending > 0
          ? "Approval needed"
          : "Working"
      : session
        ? "Ready"
        : "New chat";
  $("connection-dot").classList.toggle("offline", !S.online);
  $("model-btn").textContent = model?.name ? `${model.name}  ⌄` : "Choose a model  ⌄";
  $("send-btn").hidden = !!turn;
  $("stop-btn").hidden = !turn;
  $("stop-btn").disabled = !turn || S.cancelling.has(turn.id);
  $("stop-btn").textContent = turn && S.cancelling.has(turn.id) ? "■ Stopping…" : "■ Stop";
  const sending = pendingOutbox();
  $("send-btn").disabled =
    setup ||
    !S.online ||
    !!turn ||
    (sending !== null && sending.state !== "failed") ||
    !input.value.trim();
  $("run-status").textContent = turn ? (held ? `Waiting: ${held}` : "Running on your backend") : "";
  $("level-btn").textContent = levelName(session?.level);
  $("level-btn").disabled = !session;
  $("details-trigger").disabled = !session;
  if (!session && S.levelsOpen) showLevelMenu(false);
  $("new-btn").disabled = false;
  if (taskMode) {
    $("header-title").textContent = "Tasks";
    $("header-model").textContent = "Scheduler tasks";
    document.title = "localmelo · Tasks";
  } else if (workerMode) {
    const heading = workers.state.mode === "detail" ? workers.state.session?.title || "Worker run" : "Workers";
    $("header-title").textContent = heading;
    $("header-model").textContent = "Read-only history";
    document.title = `localmelo · ${heading}`;
  }
  const welcome = $("welcome-description");
  if (welcome) {
    welcome.textContent = session
      ? `${session.model || "Your model"} is ready. Start a conversation below.`
      : "A quiet space to think, build, and explore with your own model.";
  }
}

// ── The session list ──

function renderSessions() {
  const query = $("search").value.toLowerCase();
  const active = S.activeId;
  const snapshot = JSON.stringify([
    query,
    active,
    workerMode,
    S.sessions.map((session) => [
      session.id,
      session.title,
      session.model,
      session.busy,
      session.held || "",
      session.pending_approvals || 0,
      session.updated_at,
    ]),
  ]);
  if (snapshot === sessionListSnapshot) return;
  sessionListSnapshot = snapshot;
  const focused = document.activeElement;
  const focusedId = focused?.closest?.(".session-item")?.dataset?.sessionId;
  const focusedClose = focused?.classList?.contains?.("session-close") === true;
  $("session-count").textContent = S.sessions.length;
  $("session-list").replaceChildren();
  const shown = [...S.sessions]
    .sort((left, right) => (right.updated_at || 0) - (left.updated_at || 0))
    .filter((session) => `${title(session)} ${session.model || ""}`.toLowerCase().includes(query));
  if (shown.length === 0) {
    const empty = el("p", "list-empty", query ? "No matching chats." : "Your conversations will appear here.");
    $("session-list").append(empty);
  }
  for (const session of shown) {
    const row = el("div", "session-item");
    row.dataset.sessionId = session.id;
    if (session.id === active) row.classList.add("active");
    const info = el("button", "session-info");
    info.type = "button";
    if (session.id === active) info.setAttribute("aria-current", "page");
    const marker =
      session.pending_approvals > 0
        ? "● Approval needed · "
        : session.busy
          ? session.held
            ? "● Waiting · "
            : "● Working · "
          : "";
    info.innerHTML = `<div class="session-title">${esc(title(session))}</div><div class="session-meta">${marker}${esc(session.model || "")}</div>`;
    info.addEventListener("click", () => {
      openSession(session.id);
    });
    const close = el("button", "session-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", `Close chat: ${title(session)}`);
    close.title = "Close chat";
    close.disabled = workerMode || !!session.busy || (session.pending_approvals || 0) > 0;
    close.addEventListener("click", () => {
      deleteSession(session.id);
    });
    row.append(info, close);
    $("session-list").append(row);
    if (session.id === focusedId) {
      (focusedClose && !close.disabled ? close : info).focus({ preventScroll: true });
    }
  }
}

async function refreshList() {
  try {
    const body = await api("/sessions");
    S.online = true;
    S.machine = body.machine ?? null;
    S.sessions = body.sessions ?? [];
    const heldChanged = adoptSummary();
    renderSessions();
    syncUI();
    // The work group's summary carries the hold (`Waiting: <held>`); the
    // header and the live row move every second, so redraw the messages
    // when the 4 s poll is the first to learn a hold began or ended.
    if (heldChanged) renderMessages(false);
  } catch {
    S.online = false;
    syncUI();
  }
}

/** Copy the list's live facts onto the open session's copy; held changed. */
function adoptSummary() {
  const session = S.view?.session;
  if (!session) return false;
  const summary = S.sessions.find((item) => item.id === session.id);
  if (!summary) return false;
  const held = session.held;
  for (const key of ["title", "model", "workspace", "workspace_label", "busy", "held", "pending_approvals", "level", "updated_at"]) {
    if (summary[key] !== undefined) session[key] = summary[key];
  }
  return session.held !== held;
}

// ── Loading one session ──

async function loadSession(id) {
  const [detail, turns] = await Promise.all([
    api(`/sessions/${encodeURIComponent(id)}`),
    api(`/sessions/${encodeURIComponent(id)}/turns?limit=10`),
  ]);
  if (S.activeId !== id) return null;
  const view = S.view ?? newView(id);
  view.id = id;
  view.session = detail.session ?? view.session;
  view.tokens = detail.tokens ?? 0;
  view.grants = detail.grants ?? [];
  view.grantedAt = detail.granted_at ?? {};
  view.pending = new Map();
  for (const approval of detail.pending_approvals ?? []) {
    view.pending.set(String(approval.request_id), approval);
  }
  view.hasEarlier = turns.has_earlier === true;
  view.replayFrom = turns.replay_from ?? 0;
  view.lastEventId = view.replayFrom;
  view.turns = (turns.turns ?? []).map((row) => newTurn(row));
  S.view = view;
  renderSessions();
  renderMessages(true);
  syncUI();
  return view;
}

/**
 * Reload a chat that is already open. The view is rebuilt at `replay_from`,
 * so the stream reopens from that cursor and replays what the old one had
 * already carried.
 */
async function reloadSession(id) {
  const view = await loadSession(id);
  if (view !== null && S.activeId === id) startStream(id);
}

async function openSession(id, { saveCurrent = true, scroll = true } = {}) {
  if (!id) {
    newChat();
    return;
  }
  if (saveCurrent) saveDraft();
  if (S.activeId !== id) approvalStates.clear();
  closeStream();
  S.activeId = id;
  drawer.sessionChanged();
  S.view = newView(id);
  restoreDraft();
  persist(KEY + "-view", { id });
  setRoute(id);
  sidebar(false);
  renderSessions();
  renderMessages(true);
  syncUI();
  presence(true);
  try {
    await loadSession(id);
  } catch (error) {
    if (error.status === 404) {
      forgetSession(id);
      notify("This chat expired or the gateway restarted. Create a new chat to continue.", { sticky: true });
      return;
    }
    notify("Could not load the chat: " + error.message, { sticky: true });
  }
  startStream(id);
}

function forgetSession(id) {
  S.sessions = S.sessions.filter((session) => session.id !== id);
  if (S.activeId === id) {
    closeStream();
    S.activeId = null;
    drawer.sessionChanged();
    S.view = null;
    const draft = S.drafts[id];
    delete S.drafts[id];
    if (draft && !S.drafts.new) S.drafts.new = draft;
    persist(KEY + "-drafts", S.drafts, "sessionStorage");
    restoreDraft();
    setRoute(null);
  }
  renderSessions();
  renderMessages(false);
  syncUI();
}

function newChat({ saveCurrent = true } = {}) {
  // Closing the last chat has already deleted its draft: saving the composer
  // then would write that chat's text over a waiting new-chat draft.
  if (saveCurrent) saveDraft();
  if (S.activeId !== null) approvalStates.clear();
  closeStream();
  S.activeId = null;
  drawer.sessionChanged();
  S.view = null;
  setRoute(null);
  restoreDraft();
  sidebar(false);
  renderSessions();
  renderMessages(true);
  syncUI();
  input.focus();
}

// ── The event stream ──

function eventsUrl(id) {
  return `/api/sessions/${encodeURIComponent(id)}/events?after=${S.view?.lastEventId ?? 0}`;
}

function startStream(id) {
  // A chat opened and then left before its load finished keeps no stream:
  // one bound to this id would ignore every frame of the chat now open.
  if (!S.authenticated || S.activeId !== id || workerMode) return;
  closeStream();
  chatStatus.follow(id);
  if (typeof EventSource !== "function") return;
  S.stream.sessionId = id;
  const source = new EventSource(eventsUrl(id));
  S.stream.source = source;
  S.stream.status = "live";
  source.onmessage = (message) => {
    if (S.activeId !== id || !S.view) return;
    let frame;
    try {
      frame = JSON.parse(message.data);
    } catch {
      return;
    }
    applyOutcome(applyEvent(S.view, frame));
    if (frame.type === "approval.requested") refreshApproval(id, String(frame.data?.request_id ?? ""));
    if (frame.type === "approval.resolved") approvalStates.delete(String(frame.data?.request_id ?? ""));
    noteFrame(frame);
    chatStatus.bump();
    renderMessages(false);
    syncUI();
  };
  source.onopen = () => {
    S.stream.status = "live";
    S.stream.attempt = 0;
    S.stream.retryAt = 0;
    if (S.noticeKind === "stream") notify("");
    syncUI();
  };
  source.onerror = () => {
    if (!S.authenticated || S.stream.source !== source || S.activeId !== id || S.view === null) return;
    // EventSource cannot expose a 401. Stop its automatic retry and check
    // authentication before opening another stream.
    source.close();
    S.stream.status = "paused";
    scheduleReconnect();
    refreshStreamNotice();
  };
}

async function refreshApproval(id, requestId) {
  try {
    const detail = await api(`/sessions/${encodeURIComponent(id)}`);
    if (S.activeId !== id || !S.view?.pending.has(requestId)) return;
    const request = (detail.pending_approvals ?? []).find((item) => String(item.request_id) === requestId);
    if (request) S.view.pending.set(requestId, {...S.view.pending.get(requestId), ...request});
    renderMessages(false);
  } catch (error) {
    if (S.activeId === id && S.view?.pending.has(requestId)) {
      notify("Could not load approval details: " + error.message, {sticky: true});
    }
  }
}

function closeStream() {
  chatStatus.clear();
  if (S.stream.source) {
    try {
      S.stream.source.close();
    } catch {
      /* A closed source is fine. */
    }
  }
  S.stream.source = null;
  S.stream.sessionId = null;
  S.stream.status = "idle";
  S.stream.retryAt = 0;
  if (S.stream.timer !== null) {
    clearTimeout(S.stream.timer);
    S.stream.timer = null;
  }
  if (S.noticeKind === "stream") notify("");
}

function scheduleReconnect() {
  const id = S.activeId;
  if (!S.authenticated || !id || S.stream.timer !== null) return;
  const delay = STREAM_DELAYS_S[Math.min(S.stream.attempt, STREAM_DELAYS_S.length - 1)];
  S.stream.attempt += 1;
  S.stream.retryAt = nowSeconds() + delay;
  S.stream.timer = setTimeout(async () => {
    S.stream.timer = null;
    if (S.activeId !== id) return;
    // The chat may be gone (closed in another tab, or the gateway
    // restarted): a stream reopened on it would only 404 for ever.
    try {
      await api(`/sessions/${encodeURIComponent(id)}`);
    } catch (error) {
      if (error.status === 404) {
        forgetSession(id);
        notify(
          "This chat expired or the gateway restarted. Create a new chat to continue.",
          { sticky: true },
        );
        return;
      }
    }
    if (!S.authenticated || S.activeId !== id) return;
    startStream(id);
  }, delay * 1000);
}

/** Mirror what a frame changed outside the view: title, model, the outbox. */
function applyOutcome(outcome) {
  if (outcome.clientRequestId) {
    for (const [slot, entry] of Object.entries(S.outbox)) {
      if (entry.clientRequestId !== outcome.clientRequestId) continue;
      // An accepted message is no longer a draft, including during a retry.
      forgetDraft(slot, entry.text);
      delete S.outbox[slot];
    }
  }
  if (outcome.title !== undefined) {
    if (S.view?.session) S.view.session.title = outcome.title;
    const session = S.sessions.find((item) => item.id === S.activeId);
    if (session) session.title = outcome.title;
    sessionListSnapshot = "";
    // The sidebar's row carries the title; the next 4 s poll is too late.
    renderSessions();
  }
  if (outcome.model !== undefined || outcome.workspace !== undefined || outcome.workspaceLabel !== undefined) {
    const session = S.sessions.find((item) => item.id === S.activeId);
    if (session) {
      if (outcome.model !== undefined) session.model = outcome.model;
      if (outcome.workspace !== undefined) session.workspace = outcome.workspace;
      if (outcome.workspaceLabel !== undefined) session.workspace_label = outcome.workspaceLabel;
    }
    sessionListSnapshot = "";
    renderSessions();
  }
}

/**
 * Keep the open chat's sidebar row in step with the frames it is sent.
 *
 * The row's facts come from the list's 4 s poll; a turn this page watches
 * starting or ending, and an approval arriving or being answered, are known
 * here first, so the marker ("Working", "Waiting", "Approval needed") does
 * not wait for the poll.
 */
function noteFrame(frame) {
  const row = S.sessions.find((item) => item.id === S.activeId);
  if (!S.view || !row) return;
  const type = String(frame?.type ?? "");
  const busy = type === "turn.started" ? true : type === "turn.finished" ? false : row.busy;
  const pending = S.view.pending.size;
  if (busy === row.busy && pending === row.pending_approvals) return;
  row.busy = busy;
  row.pending_approvals = pending;
  renderSessions();
}

/** Keep the stream's banner honest while it is down. */
function refreshStreamNotice() {
  const banner = $("stream-banner");
  banner.hidden = S.stream.status !== "paused";
  if (banner.hidden) return;
  const seconds = Math.max(0, Math.ceil(S.stream.retryAt - nowSeconds()));
  setTextIfChanged($("stream-banner-text"), `Live updates paused. Reconnecting in ${seconds}s…`);
}

// ── Rendering the turns ──

function viewSnapshot() {
  const view = S.view;
  const turns = (view?.turns ?? []).map((turn) => [
    turn.id,
    turn.turn,
    turn.status,
    turn.cause,
    turn.answer,
    turn.result,
    turn.undone,
    turn.runningHere,
    turn.items,
    turn.progress,
    turn.retryUntil,
    turn.outputTokens,
    turn.error,
    turn.finishedAt,
    turn.userToggledGroup,
    turn.userText,
  ]);
  return JSON.stringify([
    S.activeId,
    view?.session?.title,
    view?.session?.held,
    [...(view?.pending?.values() ?? [])],
    view?.hasEarlier ?? false,
    earlierLoading,
    turns,
    S.outbox[S.activeId ?? "new"] ?? null,
  ]);
}

function renderMessages(forceScroll = false, preserveScroll = false) {
  persist(KEY + "-outbox", S.outbox, "sessionStorage");
  const scroller = $("scroll");
  const nearBottom =
    scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 100;
  const oldTop = scroller.scrollTop;
  const snapshot = viewSnapshot();
  if (snapshot === messageSnapshot) {
    if (forceScroll) scroller.scrollTop = scroller.scrollHeight;
    return;
  }
  const view = S.view;
  const turns = view?.turns ?? [];
  const outbox = pendingOutbox();
  const container = $("messages");
  const focus = document.activeElement;
  let focusCard = focus;
  while (focusCard && focusCard !== container && !focusCard.dataset?.requestId) focusCard = focusCard.parentElement;
  const focusRequest = focusCard !== container ? focusCard?.dataset?.requestId : null;
  const focusKey = focusRequest ? focus?.dataset?.approvalFocus : null;
  const focusState = focusRequest ? approvalStates.get(focusRequest) : null;
  if (focusState?.composing && view?.pending.has(focusRequest)) return;
  const selection = focusKey === "reason" ? [focus.selectionStart, focus.selectionEnd, focus.selectionDirection] : null;
  messageSnapshot = snapshot;
  liveNodes.clear();
  runningCards.clear();
  const nodes = turns.map((turn) => turnNode(turn));
  if (view?.hasEarlier) {
    const earlier = button("earlier-turns link-button", earlierLoading ? "Loading earlier turns…" : "Show earlier turns", loadEarlier);
    earlier.disabled = earlierLoading;
    nodes.unshift(earlier);
  }
  if (outbox && !turns.some((turn) => turn.clientRequestId === outbox.clientRequestId)) {
    nodes.push(outboxNode(outbox, S.activeId ?? "new"));
  }
  container.replaceChildren(...nodes);
  if (focusKey && focusRequest) {
    const card = document.getElementById(`approval-card-${focusRequest}`);
    const replacement = card?.querySelector(`[data-approval-focus="${focusKey}"]`);
    if (replacement && !replacement.disabled) {
      replacement.focus({preventScroll: true});
      if (selection && selection[0] !== null) replacement.setSelectionRange?.(...selection);
    }
  }
  $("welcome").hidden = nodes.length > 0 || !!S.activeId;
  if (forceScroll || (nearBottom && !preserveScroll)) scroller.scrollTop = scroller.scrollHeight;
  else scroller.scrollTop = oldTop;
  updateApprovalBar();
}

async function loadEarlier() {
  const view = S.view;
  const id = S.activeId;
  const first = view?.turns[0];
  if (!view || !id || !first || earlierLoading) return;
  const scroller = $("scroll");
  const restoreFocus = document.activeElement?.classList?.contains("earlier-turns") === true;
  earlierLoading = true;
  renderMessages(false, true);
  if (restoreFocus) {
    scroller.tabIndex = -1;
    scroller.focus({preventScroll: true});
  }
  try {
    const page = await api(`/sessions/${encodeURIComponent(id)}/turns?before=${encodeURIComponent(first.turn)}&limit=10&events=true`);
    if (S.view !== view || S.activeId !== id) return;
    // History must not change the active cursor, grants, pending approvals or outbox.
    const history = newView(id);
    history.turns = (page.turns ?? []).map((row) => newTurn(row));
    for (const row of page.turns ?? []) {
      for (const frame of row.events ?? []) applyEvent(history, frame, {ignoreCursor: true});
    }
    const existing = new Set(view.turns.map((turn) => turn.turn));
    view.turns = [...history.turns.filter((turn) => !existing.has(turn.turn)), ...view.turns];
    view.hasEarlier = page.has_earlier === true;
    const top = scroller.scrollTop;
    const height = scroller.scrollHeight;
    earlierLoading = false;
    renderMessages(false, true);
    scroller.scrollTop = top + scroller.scrollHeight - height;
  } catch (error) {
    if (S.view === view) notify("Could not load earlier turns: " + error.message, {sticky: true});
  } finally {
    earlierLoading = false;
    renderMessages(false, true);
    if (restoreFocus && S.view === view && S.activeId === id && document.activeElement === scroller) {
      const children = Array.from($("messages").children);
      const target = children.find((node) => node.classList.contains("earlier-turns"))
        || children.find((node) => node.classList.contains("turn"));
      if (target) {
        if (!target.classList.contains("earlier-turns")) target.tabIndex = -1;
        target.focus({preventScroll: true});
      }
    }
  }
}

function turnNode(turn) {
  const section = el("section", "turn");
  if (turn.undone) section.classList.add("undone");
  section.dataset.turnId = turn.id || String(turn.turn);
  section.append(userNode(turn), assistantNode(turn));
  return section;
}

function userNode(turn) {
  const article = el("article", "msg user");
  article.append(el("div", "user-bubble", turn.userText));
  if (turn.undone) {
    article.append(
      el(
        "p",
        "turn-caption",
        "Retried below · not in melo's memory · its file changes stay",
      ),
    );
  }
  return article;
}

function assistantNode(turn) {
  const article = el("article", "msg assistant");
  const label = el("div", "assistant-label");
  const icon = document.createElement("img");
  icon.className = "brand-icon";
  icon.src = "/assets/icon_small.png";
  icon.width = 28;
  icon.height = 28;
  icon.alt = "";
  label.append(icon, el("span", "", "localmelo"));
  article.append(label);
  if (!turn.finishedAt || turn.items.length > 0) article.append(workGroup(turn));
  if (turn.finishedAt && turn.status !== "completed") {
    if (turn.answer) {
      article.append(el("p", "turn-caption", "Best effort"));
      article.append(answerNode(turn.answer));
    }
    article.append(statusNode(turn));
  } else if (turn.answer) {
    article.append(answerNode(turn.answer), actionsNode(turn));
  }
  return article;
}

function answerNode(answer) {
  const node = el("div", "answer");
  node.innerHTML = renderMarkdown(answer);
  return node;
}

function actionsNode(turn) {
  const actions = el("div", "message-actions");
  const steps = typeof turn.result?.steps === "number" ? turn.result.steps : turn.step || 0;
  const parts = [
    formatDuration(turnDuration(turn)),
    steps === 1 ? "1 step" : `${steps} steps`,
    turn.outputTokens > 0 ? `${formatTokens(turn.outputTokens)} tokens` : "",
  ].filter(Boolean);
  actions.append(el("span", "", parts.join(" · ")));
  actions.append(
    button("copy-button", "Copy response", async () => {
      try {
        await navigator.clipboard.writeText(turn.answer);
      } catch {
        notify("Clipboard unavailable. Select the response text to copy it.");
      }
    }),
  );
  return actions;
}

function statusNode(turn) {
  const info = statusText(turn.status, turn.cause);
  const stopped = turn.status === "cancelled" || (turn.status === "crashed" && turn.cause === "crash");
  const node = el("div", stopped ? "turn-status stopped-line" : "failure-card");
  node.dataset.severity = info.severity;
  node.append(el("p", "failure-title", `${stopped ? "[-]" : info.severity === "error" ? "[x]" : "[!]"} ${info.title}.`));
  const error = String(turn.result?.error || turn.error || "");
  if (!stopped && error) node.append(el("p", "failure-error", cleanOutput(error).split("\n")[0]));
  if (!stopped) {
    let hint = "";
    if (turn.status === "budget_exhausted") hint = "Continue gives it a new step budget; Try again starts over.";
    if (turn.status === "stuck") hint = "melo repeated itself. Rephrase the request, or try again.";
    if (turn.status === "setup_error") hint = "An MCP server failed to start.";
    if (turn.status === "crashed" && turn.cause === "run") hint = "This is a bug in localmelo. The log is in the terminal running melo --serve, or in ~/.localmelo/logs/gateway.stderr.log when it runs as a service.";
    if (turn.status === "provider_error") {
      const backend = S.health?.model?.backend;
      hint = ["mlc", "ollama", "vllm", "sglang"].includes(backend)
        ? `Is it running at ${S.health?.model_server?.url_label || "the configured server"}?`
        : "Check the API key and the model name.";
    }
    if (hint) node.append(el("p", "failure-hint", hint));
    if (turn.status === "provider_error") node.append(button("link-button", "Open settings", () => openSettings()));
  }
  const actions = el("div", "failure-actions");
  const newest = S.view?.turns[S.view.turns.length - 1] === turn;
  if (canContinue(turn, newest)) actions.append(button("link-button", "Continue", () => queueMessage("Continue the task from where you stopped.")));
  if (canRetry(turn, newest)) {
    const retry = button("link-button", "Try again", () => retryTurn(turn));
    retry.disabled = retrying;
    retry.title = "Run this message again from the start. Files it changed stay changed.";
    actions.append(retry);
  }
  if (!stopped) {
    actions.append(button("link-button", "Copy error", async () => {
      try { await navigator.clipboard.writeText(`${statusLine(turn.status, error)}\nrun ${turn.runId}`); }
      catch { notify("Clipboard unavailable. Select the error text to copy it."); }
    }));
  }
  if (turn.status === "setup_error") actions.append(button("link-button", "Show tools", () => drawer.open("tools")));
  node.append(actions);
  if (!stopped) {
    const details = el("details", "failure-details");
    details.append(el("summary", "", "Details"));
    const result = turn.result ?? {};
    const fields = [
      `Status: ${turn.status}`, `Cause: ${turn.cause || "run"}`,
      `Error: ${cleanOutput(error) || "no error message"}`,
      `Steps: ${result.steps ?? turn.step ?? 0}`, `Attempts: ${result.attempts ?? turn.attempt ?? 0}`,
      `Tool calls: ${result.n_tool ?? 0}`, `Tokens: ${turn.outputTokens}`,
      `Duration: ${formatDuration(turnDuration(turn))}`, `Run: ${turn.runId}`,
    ];
    details.append(el("pre", "", fields.join("\n")));
    node.append(details);
  }
  return node;
}

async function retryTurn(turn) {
  if (workerMode) return;
  const id = S.activeId;
  if (!id || retrying || runningTurn() !== null || pendingOutbox() || !canRetry(turn, S.view?.turns[S.view.turns.length - 1] === turn)) return;
  retrying = true;
  try {
    await api(`/sessions/${encodeURIComponent(id)}/retry`, {body: {turn_id: turn.id}});
  } catch (error) {
    notify("Could not retry the turn: " + error.message, {sticky: true});
  } finally {
    retrying = false;
  }
}

function workGroup(turn) {
  const group = el("details", "work-group");
  const open =
    !turn.finishedAt ||
    (S.view?.pending.size ?? 0) > 0 ||
    turn.userToggledGroup ||
    turn.status !== "completed";
  group.open = open;
  group.append(el("summary", "work-summary", workSummary(turn, { held: heldOf(turn) })));
  group.addEventListener("toggle", () => {
    if (group.open !== open) turn.userToggledGroup = group.open;
  });
  for (const item of turn.items) {
    group.append(itemNode(item, turn));
    const request = item.pendingRequestId && S.view?.pending.get(item.pendingRequestId);
    if (request) group.append(approvalCard(request));
  }
  if (!turn.finishedAt) group.append(liveRow(turn));
  return group;
}

function heldOf(turn) {
  if (turn.finishedAt) return "";
  return S.view?.session?.held || "";
}

function itemNode(item, turn, context = {}) {
  switch (item.kind) {
    case "tool":
      return toolCard(item, turn, context);
    case "thinking": {
      const card = el("details", "thinking");
      card.open = item.expanded === true;
      const duration =
        item.durationS === null ? "" : ` for ${formatDuration(item.durationS)}`;
      card.append(el("summary", "", `Thought${duration}`));
      const body = el("div", "thinking-body", item.text);
      card.append(body);
      card.addEventListener("toggle", () => {
        item.expanded = card.open;
      });
      return card;
    }
    case "assistant_text": {
      const node = el("div", "assistant-text");
      node.innerHTML = renderMarkdown(item.text);
      return node;
    }
    case "notice": {
      const node = el("div", "notice-item", `[!] ${item.message}`);
      node.dataset.kind = item.noticeKind || "";
      return node;
    }
    case "fact":
      return el("div", "fact-item", item.message);
    case "reflection": {
      const card = el("details", "reflection");
      card.open = item.expanded === true;
      card.append(el("summary", "", reflectionLine(item)));
      const body = el("div", "reflection-body");
      if (item.summary) body.append(el("p", "", item.summary));
      if (item.rationale && item.rationale !== item.summary) {
        body.append(el("p", "muted", item.rationale));
      }
      card.append(body);
      card.addEventListener("toggle", () => {
        item.expanded = card.open;
      });
      return card;
    }
    default:
      return el("div", "");
  }
}

function approvalNow() {
  return globalThis.performance?.now?.() ?? Date.now();
}

// How the card talks to its view: the chat view redraws its messages and
// answers a session's approval; the Tasks view passes its own (tasks.mjs).
const chatApproval = {
  rerender: () => { messageSnapshot = ""; renderMessages(false); },
  answer: (request, state, choice) => answerApproval(request, state, choice),
  reload: (id) => refreshApproval(S.activeId, id),
};

function approvalCard(request, view = chatApproval) {
  const id = String(request.request_id);
  let state = approvalStates.get(id);
  if (!state) {
    state = {mountedAt: approvalNow(), deny: false, reason: "", showAll: false, submitting: false};
    approvalStates.set(id, state);
  }
  const card = el("section", "approval-card");
  card.id = `approval-card-${id}`;
  card.dataset.requestId = id;
  card.setAttribute("role", "region");
  card.setAttribute("aria-label", "Approval needed");
  const tool = String(request.tool_name || "tool");
  const args = request.arguments || {};
  const workspace = request.workspace_label || view.workspace || S.view?.session?.workspace_label || "the workspace";
  let title = `${tool} wants to run with these arguments:`;
  let content = JSON.stringify(args, null, 2);
  if (tool === "shell_exec") {
    title = `shell_exec wants to run a command in ${workspace}:`;
    content = typeof args.command === "string" ? args.command : JSON.stringify(args.command ?? "");
  } else if (tool === "python_exec") {
    title = `python_exec wants to run Python in ${workspace}:`;
    content = String(args.code ?? "");
  } else if (tool === "file_write") {
    const target = request.target;
    const path = target?.path_label || String(args.path ?? "the file");
    content = String(args.content ?? "");
    title = target
      ? target.exists
        ? `file_write wants to overwrite ${path} (${formatBytes(target.size_bytes)}) with ${content.length} chars:`
        : `file_write wants to create ${path} (${content.length} chars):`
      : `file_write wants to write ${path} (checking target):`;
  }
  card.append(el("p", "approval-title", title));
  const lines = content.split("\n");
  const canFold = tool === "python_exec" || tool === "file_write";
  card.append(el("pre", "approval-content", !canFold || state.showAll ? content : lines.slice(0, 20).join("\n")));
  if (canFold && !state.showAll && lines.length > 20) {
    card.append(button("link-button", `Show all ${lines.length} lines`, () => {
      state.showAll = true;
      view.rerender();
    }));
  }
  const actions = el("div", "approval-actions");
  const denyView = el("div", "deny-view");
  const once = button("secondary-button", "Allow once", () => view.answer(request, state, "once"));
  once.dataset.approvalOnce = "";
  once.dataset.approvalFocus = "once";
  const session = button("secondary-button", `Allow every ${tool} call in this session`, () => view.answer(request, state, "session"));
  session.dataset.approvalFocus = "session";
  const deny = button("secondary-button", "Deny…", () => {
    state.deny = true;
    actions.hidden = true;
    denyView.hidden = false;
    reason.focus();
  });
  deny.dataset.approvalDeny = "";
  deny.dataset.approvalFocus = "deny";
  actions.append(once, session, deny);
  card.append(actions);
  if (tool === "shell_exec" || tool === "python_exec") {
    const kind = tool === "shell_exec" ? "shell command" : "Python code";
    card.append(el("p", "approval-warning", `“Allow every” lets any ${kind} run without asking for the rest of this session. Commands can reach files outside ${workspace}.`));
  }
  const label = el("label", "", "Tell melo why (optional)");
  const reason = el("input", "deny-reason");
  reason.value = state.reason;
  reason.dataset.approvalFocus = "reason";
  reason.addEventListener("compositionstart", () => { state.composing = true; });
  reason.addEventListener("compositionend", () => {
    state.composing = false;
    state.reason = reason.value.slice(0, 500);
    view.rerender();
  });
  reason.maxLength = 500;
  reason.setAttribute("aria-label", "Tell melo why (optional)");
  const counter = el("span", "deny-counter", `${state.reason.length} / 500`);
  reason.addEventListener("input", () => {
    state.reason = reason.value.slice(0, 500);
    counter.textContent = `${state.reason.length} / 500`;
  });
  const confirm = button("secondary-button", "Deny", () => view.answer(request, state, "deny"));
  const back = button("link-button", "Back", () => {
    state.deny = false;
    denyView.hidden = true;
    actions.hidden = false;
    deny.focus();
  });
  reason.addEventListener("keydown", (event) => {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      view.answer(request, state, "deny");
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      back.click();
    }
  });
  label.append(reason);
  denyView.append(label, counter, confirm, back);
  denyView.hidden = !state.deny;
  actions.hidden = state.deny;
  card.append(denyView);
  if (request.arguments_complete !== true) {
    card.append(el("p", "approval-warning", request.arguments_complete === false
      ? "Approval arguments were truncated or changed; they cannot safely be allowed here. You can deny or stop the run."
      : "Complete approval details are not available; allow is disabled."));
  }
  if (request.arguments_complete !== true || (tool === "file_write" && !request.target)) {
    card.append(button("link-button", "Reload approval details", () => view.reload(id)));
  }
  const controls = [once, session, deny, confirm];
  const enable = () => {
    const waiting = approvalNow() - state.mountedAt < 500;
    const unavailable = request.arguments_complete !== true || (tool === "file_write" && !request.target);
    for (const control of controls) control.disabled = waiting || state.submitting || (unavailable && (control === once || control === session));
  };
  enable();
  setTimeout(enable, Math.max(0, 500 - (approvalNow() - state.mountedAt)));
  return card;
}

async function answerApproval(request, state, choice) {
  if (workerMode) return;
  const id = String(request.request_id);
  if (state.submitting || approvalNow() - state.mountedAt < 500 || !S.view?.pending.has(id)) return;
  if (choice !== "deny" && (request.arguments_complete !== true || (request.tool_name === "file_write" && !request.target))) return;
  state.submitting = true;
  messageSnapshot = "";
  renderMessages(false);
  try {
    await api(`/approvals/${encodeURIComponent(id)}`, {
      body: {choice, ...(choice === "deny" ? {reason: state.reason} : {})},
    });
  } catch (error) {
    state.submitting = false;
    notify("Could not answer approval: " + error.message, {sticky: true});
    messageSnapshot = "";
    renderMessages(false);
  }
}

function updateApprovalBar() {
  const bar = $("approval-bar");
  const request = S.view?.pending.values().next().value;
  if (!request) { bar.hidden = true; return; }
  const card = document.getElementById(`approval-card-${request.request_id}`);
  const scroller = $("scroll");
  if (!card || !card.getBoundingClientRect || !scroller.getBoundingClientRect) {
    bar.hidden = false;
    return;
  }
  const bounds = card.getBoundingClientRect();
  const area = scroller.getBoundingClientRect();
  bar.hidden = bounds.bottom > area.top && bounds.top < area.bottom;
}

function goToApproval() {
  const request = S.view?.pending.values().next().value;
  if (!request) return;
  const card = document.getElementById(`approval-card-${request.request_id}`);
  if (!card) return;
  const group = card.closest(".work-group");
  if (group) group.open = true;
  card.scrollIntoView?.({block: "center"});
  const once = card.querySelector("[data-approval-once]");
  if (once && !once.disabled) once.focus();
  else if (once) {
    const state = approvalStates.get(String(request.request_id));
    setTimeout(() => {
      if (document.activeElement === $("approval-go") && !once.disabled) once.focus();
    }, Math.max(0, 500 - (approvalNow() - state.mountedAt)));
  }
}

function toolCard(item, turn, context = {}) {
  // A call of a held chat says why the model server is not being asked yet.
  const state = toolState(item, { held: context.readOnly ? "" : heldOf(turn), now: nowSeconds() });
  const card = el("details", "tool-card");
  card.dataset.state = state.state;
  card.open = item.expanded === true;
  const summary = el("summary", "tool-summary");
  summary.append(el("span", "tool-mark", state.mark));
  summary.append(el("span", "tool-name", item.name));
  if (item.preview) summary.append(el("span", "tool-preview", printable(item.preview)));
  const meta = state.text ? el("span", "tool-meta", state.text) : null;
  if (meta) summary.append(meta);
  if (meta && state.state === "running" && !context.readOnly) {
    runningCards.set(item, {item, turn, node: meta, card});
  }
  card.append(summary);
  card.append(toolBody(item, context));
  card.addEventListener("toggle", () => {
    item.expanded = card.open;
  });
  return card;
}

function toolBody(item, context = {}) {
  const body = el("div", "tool-body");
  const entries = Object.entries(item.arguments ?? {});
  if (entries.length > 0) {
    const section = el("div", "tool-section");
    section.append(el("h4", "", "Arguments"));
    const pre = el("pre", "tool-arguments");
    pre.textContent = entries
      .map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
      .join("\n");
    section.append(pre);
    body.append(section);
  }
  const tag = approvalTag(item.status, item.errorCategory, item.approval);
  if (tag) body.append(el("p", "tool-approval", tag.long));
  if (item.error) body.append(el("p", "tool-error", cleanOutput(item.error)));
  if (item.output) body.append(outputBlock(item, context));
  for (const image of item.images ?? []) {
    const node = document.createElement("img");
    node.className = "tool-image";
    node.loading = "lazy";
    node.alt = `Image from ${item.name}`;
    node.src = `/api/sessions/${encodeURIComponent(context.sessionId ?? S.activeId ?? "")}/blobs/${encodeURIComponent(image.sha256)}`;
    node.title = `${image.mime_type} · ${formatBytes(image.size_bytes)}`;
    body.append(node);
  }
  return body;
}

function outputBlock(item, context = {}) {
  const section = el("div", "tool-section");
  section.append(el("h4", "", "Output"));
  const text = cleanOutput(item.output);
  const lines = text.split("\n");
  const shown =
    item.showAll || (text.length <= 2000 && lines.length <= 40)
      ? text
      : lines.slice(0, 40).join("\n").slice(0, 2000);
  section.append(el("pre", "tool-output", shown));
  // The count is the design's raw character count, not the house's 2.0k.
  const hidden = item.outputChars || text.length;
  if (shown !== text) {
    section.append(
      button("link-button", `Show all ${hidden} characters`, () => {
        item.showAll = true;
        if (context.readOnly) context.render();
        else renderMessages(false);
      }),
    );
  }
  section.append(
    button("link-button", "Copy output", async () => {
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        notify("Clipboard unavailable. Select the output to copy it.");
      }
    }),
  );
  return section;
}

function liveRow(turn) {
  const row = el("div", "live-row");
  row.setAttribute("role", "status");
  // Off: the 1 s ticks would otherwise spam a screen reader.
  row.setAttribute("aria-live", "off");
  const line = el("div", "live-line");
  line.append(el("span", "status-dot"));
  const text = el("span", "live-text", "");
  line.append(text);
  row.append(line);
  const past = pastNode(turn);
  if (past) row.append(past);
  liveNodes.set(turn.id || String(turn.turn), { text, turn, row });
  const now = nowSeconds();
  text.textContent = liveText(turn, now);
  row.dataset.activity = activity(turn, activityOptions(turn, now));
  return row;
}

function activityOptions(turn, now) {
  return {
    held: heldOf(turn),
    pending: S.view?.pending.size ?? 0,
    cancelling: S.cancelling.has(turn.id),
    now,
  };
}

/** The live row's first line: `● Running bash · 12s · step 2 · 412 tokens`. */
function liveText(turn, now) {
  const parts = [activity(turn, activityOptions(turn, now))];
  parts.push(formatDuration(Math.max(0, now - turn.startedAt)));
  parts.push(...liveFacts(turn));
  if (turn.outputTokens > 0) parts.push(`${formatTokens(turn.outputTokens)} tokens`);
  parts.push("Esc to stop");
  return parts.join(" · ");
}

/** The first turn's past facts, as the terminal's level-1 line shows them. */
function pastNode(turn) {
  if (turn.turn !== 1 || !turn.progress) return null;
  const basis = turn.progress.basis;
  if (basis?.match === "new") return null;
  const text = formatPast(turn.progress);
  if (!text) return null;
  return el("div", "live-past", text);
}

function outboxNode(entry, slot) {
  const article = el("article", "msg user");
  const bubble = el("div", "user-bubble", entry.text);
  article.append(bubble);
  const status = el("p", "outbox-state");
  if (entry.state === "sending") status.textContent = "Sending…";
  else if (entry.state === "checking") status.textContent = "Checking…";
  else {
    const label = entry.unknown ? "Outcome unknown" : "Not sent";
    status.append(el("span", "", entry.error ? `${label}: ${entry.error} · ` : `${label} · `));
    status.append(
      button("link-button", "Retry", () => {
        entry.state = "sending";
        renderMessages(false);
        syncUI();
        submit(slot, entry.clientRequestId, entry.text);
      }),
    );
    status.append(
      button("link-button", "Cancel and keep draft", () => {
        cancelSend(slot, entry);
      }),
    );
  }
  article.append(status);
  return article;
}

// ── Sending, stopping, deleting ──

async function send() {
  if (workerMode) return;
  return queueMessage(input.value.trim(), {clearInput: true});
}

async function queueMessage(text, {clearInput = false} = {}) {
  if (workerMode) return;
  const slot = S.activeId ?? "new";
  const previous = S.outbox[slot];
  if (!text || runningTurn() !== null) return;
  if (previous && (previous.state !== "failed" || !clearInput)) return;
  // The same text, failed after a check of unknown outcome, keeps its key:
  // the store answers `repeated` instead of running it twice.
  const again = previous && previous.state === "failed" && previous.text === text;
  const key = again ? previous.clientRequestId : newClientRequestId();
  S.outbox[slot] = { clientRequestId: key, text, state: "sending", error: "" };
  if (clearInput) {
    input.value = "";
    saveDraft();
    resizeInput();
  }
  presence(true);
  renderMessages(true);
  syncUI();
  await submit(slot, key, text);
}

async function submit(slot, key, text) {
  // The slot is where the message was sent from: "new" or its chat.
  const sessionId = slot === "new" ? null : slot;
  try {
    if (!sessionId) {
      const body = await api("/sessions", {
        body: {
          turn: { text, client_request_id: key },
          ...(S.newWorkspace ? { workspace: S.newWorkspace } : {}),
        },
      });
      if (S.outbox[slot]?.clientRequestId !== key) return;
      const created = body.session;
      forgetDraft("new", text);
      delete S.outbox.new;
      delete S.outbox[created.id];
      if (S.activeId === null) {
        S.sessions.unshift(created);
        S.activeId = created.id;
        drawer.sessionChanged();
        S.view = newView(created.id);
        S.view.session = created;
        moveDraft("new", created.id);
        persist(KEY + "-view", { id: created.id });
        setRoute(created.id);
        renderSessions();
        await loadSession(created.id);
        startStream(created.id);
      }
      renderMessages(true);
      syncUI();
      input.focus();
      return;
    }
    const body = await api(`/sessions/${encodeURIComponent(sessionId)}/turns`, {
      body: { text, client_request_id: key },
    });
    // A repeat is accepted too, so either way the text is not a draft now.
    if (S.outbox[slot]?.clientRequestId !== key) return;
    forgetDraft(sessionId, text);
    if (body.repeated === true) {
      // The key was accepted already: this is the turn it started, not a
      // second one, and no new event will say so.
      const entry = S.outbox[slot];
      if (entry && entry.clientRequestId === key) delete S.outbox[slot];
      await reloadSession(sessionId);
      renderMessages(true);
      syncUI();
    }
  } catch (error) {
    const entry = S.outbox[slot];
    if (!entry || entry.clientRequestId !== key) return;
    const refused =
      typeof error.status === "number" &&
      error.status >= 400 &&
      error.status < 500 &&
      error.status !== 408 &&
      error.status !== 429;
    if (refused) {
      entry.state = "failed";
      entry.error = error.message;
      // Keep the text in the draft as well as the tab's persisted outbox.
      keepDraft(slot, text);
      renderMessages(false);
      syncUI();
      return;
    }
    // The answer never arrived: ask the store whether the message was
    // accepted before offering to send it again.
    await checkSubmission(slot, key);
  }
}

async function checkSubmission(slot, key) {
  if (workerMode) return;
  const entry = S.outbox[slot];
  if (!entry) return;
  entry.state = "checking";
  renderMessages(false);
  syncUI();
  const sessionId = slot === "new" ? null : slot;
  try {
    const body = sessionId
      ? await api(`/sessions/${encodeURIComponent(sessionId)}/submissions/${encodeURIComponent(key)}`)
      : await api(`/submissions/${encodeURIComponent(key)}`);
    if (workerMode || S.outbox[slot] !== entry || entry.clientRequestId !== key) return;
    const submission = body.submission ?? {};
    if (submission.state === "accepted") {
      forgetDraft(slot, entry.text);
      delete S.outbox[slot];
      if (sessionId === null && submission.session_id) {
        await refreshList();
        if (S.activeId === null) {
          S.activeId = submission.session_id;
          drawer.sessionChanged();
          S.view = newView(submission.session_id);
          persist(KEY + "-view", { id: submission.session_id });
          setRoute(submission.session_id);
          await reloadSession(submission.session_id);
        }
      } else if (sessionId !== null && S.activeId === sessionId) {
        await reloadSession(sessionId);
      }
      renderMessages(true);
      syncUI();
      return;
    }
    entry.state = "failed";
    entry.error = "";
    entry.unknown = true;
    // Keep the text editable; the persisted key prevents a duplicate retry.
    keepDraft(slot, entry.text);
  } catch (error) {
    if (workerMode || S.outbox[slot] !== entry || entry.clientRequestId !== key) return;
    entry.state = "failed";
    entry.error = error.message;
    entry.unknown = true;
    keepDraft(slot, entry.text);
  }
  renderMessages(false);
  syncUI();
}

async function cancelSend(slot, entry) {
  if (workerMode) return;
  const sessionId = slot === "new" ? null : slot;
  const path = sessionId
    ? `/sessions/${encodeURIComponent(sessionId)}/submissions/${encodeURIComponent(entry.clientRequestId)}/cancel`
    : `/submissions/${encodeURIComponent(entry.clientRequestId)}/cancel`;
  try {
    await api(path, { body: { text: entry.text } });
    if (S.outbox[slot] !== entry) return;
    delete S.outbox[slot];
    keepDraft(slot, entry.text);
    renderMessages(false);
    syncUI();
  } catch (error) {
    if (S.outbox[slot] !== entry) return;
    if (error.code === "submission_withdrawn") {
      // The store never took it, so the text comes back to the composer.
      delete S.outbox[slot];
      keepDraft(slot, entry.text);
      renderMessages(false);
      syncUI();
      return;
    }
    entry.state = "failed";
    entry.error = error.message;
    renderMessages(false);
    syncUI();
  }
}

async function stop() {
  if (workerMode) return;
  const turn = runningTurn();
  if (!turn || !turn.id || S.cancelling.has(turn.id)) return;
  S.cancelling.add(turn.id);
  syncUI();
  try {
    await api(`/turns/${encodeURIComponent(turn.id)}/cancel`, { body: {} });
  } catch (error) {
    if (error.status !== 409 && error.status !== 404) {
      notify("Could not stop the response: " + error.message, { sticky: true });
    }
  } finally {
    S.cancelling.delete(turn.id);
    syncUI();
  }
}

async function deleteSession(id) {
  if (workerMode) return;
  const session = S.sessions.find((item) => item.id === id);
  if (!session || session.busy || (session.pending_approvals || 0) > 0) return;
  try {
    await api(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    delete S.drafts[id];
    delete S.outbox[id];
    persist(KEY + "-drafts", S.drafts, "sessionStorage");
    S.sessions = S.sessions.filter((item) => item.id !== id);
    if (S.activeId === id) {
      closeStream();
      S.activeId = null;
      S.view = null;
      if (S.sessions.length > 0) await openSession(S.sessions[0].id, { saveCurrent: false });
      else newChat({ saveCurrent: false });
    }
    renderSessions();
    syncUI();
  } catch (error) {
    notify("Could not close the chat: " + error.message, { sticky: true });
  }
}

// ── The header's S2 controls ──

function renderLevelMenu() {
  const session = activeSession();
  const level = session?.level || 2;
  const menu = $("level-menu");
  menu.replaceChildren();
  for (const number of [1, 2, 3, 4]) {
    const item = el("button", "level-item", levelName(number) + (number === 1 ? " · urgent" : number === 4 ? " · background" : ""));
    item.type = "button";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(number === level));
    item.addEventListener("click", () => setLevel(number));
    menu.append(item);
  }
}

function showLevelMenu(open) {
  S.levelsOpen = open;
  $("level-menu").hidden = !open;
  $("level-btn").setAttribute("aria-expanded", String(open));
  if (open) renderLevelMenu();
}

async function setLevel(level) {
  if (workerMode) return;
  const id = S.activeId;
  showLevelMenu(false);
  if (!id) return;
  const previous = activeSession()?.level;
  const session = activeSession();
  if (session) session.level = level;
  syncUI();
  try {
    const body = await api(`/sessions/${encodeURIComponent(id)}/level`, { body: { level } });
    // The answer can land after a switch: it belongs to that chat, not to
    // whichever one is open now.
    const listed = S.sessions.find((item) => item.id === id);
    if (listed) listed.level = body.level;
    if (S.view?.session?.id === id) S.view.session.level = body.level;
  } catch (error) {
    const listed = S.sessions.find((item) => item.id === id);
    if (listed) listed.level = previous;
    if (S.view?.session?.id === id) S.view.session.level = previous;
    notify("Could not change the chat level: " + error.message, { sticky: true });
  }
  sessionListSnapshot = "";
  renderSessions();
  syncUI();
}

function presence(force = false) {
  if (workerMode) return;
  const id = S.activeId;
  if (!S.authenticated || !id) return;
  const now = Date.now();
  if (!force && now - S.lastPresenceAt < PRESENCE_MS) return;
  S.lastPresenceAt = now;
  try {
    const request = fetch("/api/presence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: id }),
      keepalive: true,
    });
    request?.then?.((response) => { if (response.status === 401) lock(); }).catch(() => {});
  } catch {
    /* A browser may refuse fetch, or a test may not allow it. */
  }
}

// ── Routing and the sidebar ──

// `workerMode` is true in every read-only side view; `taskMode` marks the task view.
function statusRowNode() {
  const node = el("div", "status-row");
  node.hidden = true;
  return node;
}
const chatStatus = setupStatusRow({ api, root: $("status-row") });
const running = setupRunning({ root: $("running-root"), button: $("running-btn"), menu: $("running-menu") });

function setHealth(body) {
  S.health = body;
  running.update(body?.status);
}

let workerMode = false;
let taskMode = false;
const workerStatus = setupStatusRow({ api, root: statusRowNode() });
const workers = workerHistory(api, () => { if (workerMode && !taskMode) renderWorkers(); }, (fn, ms) => {
  const timer = setInterval(fn, ms);
  return () => globalThis.clearInterval?.(timer);
}, {
  statusRow: workerStatus,
  doc: document,
  open: (url) => (typeof EventSource === "function" ? new EventSource(url) : null),
  schedule: (fn, ms) => setTimeout(fn, ms),
  cancel: (id) => clearTimeout(id),
});
const tasks = setupTasks({
  api,
  statusRow: setupStatusRow({ api, root: statusRowNode() }),
  live: {
    nodes: taskLiveNodes,
    open: (url) => (typeof EventSource === "function" ? new EventSource(url) : null),
    schedule: (fn, ms) => setTimeout(fn, ms),
    cancel: (id) => clearTimeout(id),
  },
  approvalCard,
  root: $("task-panel"),
  newId: newClientRequestId,
  navigate: (hash) => { globalThis.location.hash = hash; },
  every: (fn, ms) => {
    const timer = setInterval(fn, ms);
    return () => globalThis.clearInterval?.(timer);
  },
});

function taskRoute() {
  const match = /^#\/tasks(?:\/(.+))?$/.exec(globalThis.location?.hash || "");
  if (!match) return null;
  try { return match[1] ? decodeURIComponent(match[1]) : ""; }
  catch { return ""; }
}

function showTasks(id) {
  if (!workerMode) {
    saveDraft();
    closeStream();
    drawer.close();
    showLevelMenu(false);
  }
  workers.leave();
  workerMode = true;
  taskMode = true;
  $("worker-panel").hidden = true;
  $("scroll").hidden = true;
  document.querySelector(".composer-area").hidden = true;
  $("details-trigger").hidden = true;
  $("level-btn").hidden = true;
  $("connection-btn").hidden = true;
  $("chats-link").href = S.activeId ? `#/s/${S.activeId}` : "#/new";
  renderSessions();
  $("workers-link").removeAttribute("aria-current");
  $("tasks-link").setAttribute("aria-current", "page");
  $("chats-link").removeAttribute("aria-current");
  sidebar(false);
  syncUI();
  tasks.show(id);
  $("task-panel").querySelector("h1")?.focus();
}

function workerRoute() {
  const match = /^#\/workers(?:\/(.+))?$/.exec(globalThis.location?.hash || "");
  if (!match) return null;
  try { return match[1] ? decodeURIComponent(match[1]) : ""; }
  catch { return ""; }
}

function showWorkers(id) {
  const previousWorker = workers.state.selectedId;
  if (taskMode) {
    tasks.leave();
    taskMode = false;
    $("tasks-link").removeAttribute("aria-current");
  }
  if (!workerMode) {
    saveDraft();
    closeStream();
    drawer.close();
    showLevelMenu(false);
  }
  workerMode = true;
  $("worker-panel").hidden = false;
  $("scroll").hidden = true;
  document.querySelector(".composer-area").hidden = true;
  $("details-trigger").hidden = true;
  $("level-btn").hidden = true;
  $("connection-btn").hidden = true;
  $("chats-link").href = S.activeId ? `#/s/${S.activeId}` : "#/new";
  renderSessions();
  $("workers-link").setAttribute("aria-current", "page");
  $("chats-link").removeAttribute("aria-current");
  sidebar(false);
  if (id) workers.detail(id);
  else if (workers.state.loaded && !workers.state.error) { workers.state.epoch++; workers.state.mode = "list"; workers.state.session = null; workers.state.view = null; workers.state.loading = false; renderWorkers(); }
  else workers.list();
  const previousRow = !id && previousWorker && Array.from($("worker-panel").children).find((node) => node.dataset?.workerId === previousWorker);
  (previousRow || $("worker-panel").querySelector("h1"))?.focus();
}

function leaveWorkers() {
  if (!workerMode) return;
  workers.leave();
  tasks.leave();
  workerMode = false;
  taskMode = false;
  $("worker-panel").hidden = true;
  $("scroll").hidden = false;
  document.querySelector(".composer-area").hidden = false;
  $("details-trigger").hidden = false;
  $("level-btn").hidden = false;
  $("connection-btn").hidden = false;
  renderSessions();
  $("workers-link").removeAttribute("aria-current");
  $("tasks-link").removeAttribute("aria-current");
  $("chats-link").setAttribute("aria-current", "page");
  for (const [slot, entry] of Object.entries(S.outbox)) {
    if (entry.state === "checking") checkSubmission(slot, entry.clientRequestId);
  }
}

// A turn's work, answer and end state as the read-only views show them. `empty`
// is the line for a turn without events, or "" to draw its work group.
function turnWork(turn, sessionId, render, empty) {
  const nodes = [];
  if (empty) nodes.push(el("p", "muted", empty));
  else {
    const group = el("details", "work-group"); group.open = true;
    group.append(el("summary", "work-summary", workSummary(turn, {})));
    for (const item of turn.items) group.append(itemNode(item, turn, {readOnly: true, sessionId, render}));
    nodes.push(group);
  }
  if (turn.answer) nodes.push(answerNode(turn.answer));
  if (turn.status) {
    nodes.push(el("p", "turn-status", statusText(turn.status, turn.cause).title));
    const error = turn.status !== "completed" ? cleanOutput(String(turn.result?.error || turn.error || "")).split("\n")[0] : "";
    if (error) nodes.push(el("p", "failure-error", error));
  }
  return nodes;
}

// The Live section of a task card: its session's turns that hold work or an answer.
function taskLiveNodes(view, render) {
  return view.turns.flatMap((turn) => (turn.items.length || turn.answer ? turnWork(turn, view.id, render, "") : []));
}

function renderWorkers() {
  const state = workers.state;
  const root = $("worker-panel");
  const restoreHeading = document.activeElement?.parentElement === root && document.activeElement?.tagName === "H1";
  const terminal = state.mode === "detail" ? state.session?.kind === "terminal" : state.kind === "terminal";
  const title = el("h1", "", state.mode === "list" ? "Workers" : state.session?.title || (terminal ? "Terminal chat" : "Worker run"));
  title.tabIndex = -1;
  const nodes = [title, el("p", "muted", state.mode === "list" ? "Read-only history" : terminal ? "Read-only terminal chat, secrets redacted" : "Read-only worker run")];
  if (state.mode === "detail") {
    const back = document.createElement("a"); back.href = "#/workers"; back.textContent = "Back to workers"; nodes.push(back);
  }
  if (state.mode === "list") {
    const kinds = el("div", "worker-kinds");
    for (const [value, text] of [["worker", "Worker runs"], ["terminal", "Terminal chats"]]) {
      const chip = button("secondary-button", text, () => {
        if (state.kind === value) return;
        state.kind = value; state.filters = {status: "", source: "", parent: ""}; workers.list();
      });
      chip.setAttribute("aria-pressed", String(state.kind === value));
      kinds.append(chip);
    }
    nodes.push(kinds);
    const filters = el("form", "worker-filters");
    const select = (label, name, choices, selected) => {
      const wrapper = el("label", "", label);
      const control = document.createElement("select"); control.name = name;
      for (const [value, text] of choices) {
        const option = document.createElement("option"); option.value = value; option.textContent = text;
        control.append(option);
      }
      control.value = selected; wrapper.append(control); filters.append(wrapper);
      return control;
    };
    const statuses = [...new Set(["running", "completed", "crashed", "budget_exhausted", "stuck", "cancelled", ...state.rows.map((row) => row.status).filter(Boolean), state.filters.status].filter(Boolean))];
    const status = select("Status", "status", [["", "All statuses"], ...statuses.map((s) => [s, s])], state.filters.status);
    // Source and parent run describe a run; a terminal chat has neither.
    const source = terminal ? null : select("Source", "source", [["", "All sources"], ["localmelo", "localmelo"], ["dsh", "dsh"]], state.filters.source);
    const parent = document.createElement("input"); parent.value = state.filters.parent;
    if (!terminal) { const label = el("label", "", "Parent run"); label.append(parent); filters.append(label); }
    const apply = document.createElement("button"); apply.type = "submit"; apply.className = "primary-button"; apply.textContent = "Apply"; filters.append(apply);
    filters.append(button("secondary-button", "Clear filters", () => { state.filters = {status: "", source: "", parent: ""}; workers.list(); }));
    filters.addEventListener("submit", (event) => { event.preventDefault(); state.filters = {status: status.value, source: source?.value ?? "", parent: parent.value.trim()}; workers.list(); });
    nodes.push(filters);
    for (const row of state.rows) {
      const link = document.createElement("a");
      link.className = "worker-row"; link.dataset.workerId = row.id; link.href = `#/workers/${encodeURIComponent(row.id)}`;
      link.append(el("strong", "", row.title || "Not recorded"), el("span", "muted", [terminal ? "terminal" : row.source, row.model, row.status].map((v) => v == null || v === "" ? "Not recorded" : String(v)).join(" · ")));
      const started = row.started_at == null ? "Not recorded" : new Date(row.started_at * 1000).toLocaleString();
      const duration = row.duration_s == null ? "Not recorded" : formatDuration(row.duration_s);
      link.append(el("span", "muted", `${terminal ? "Last turn" : "Started"}: ${started} · Duration: ${duration}${row.parent_run ? ` · Parent: ${row.parent_run}` : ""}`));
      nodes.push(link);
    }
    if (!state.loading && !state.error && !state.rows.length) nodes.push(el("p", "", Object.values(state.filters).some(Boolean) ? "No runs match these filters." : terminal ? "No terminal chats yet." : "No worker runs yet."));
    if (state.nextCursor) nodes.push(button("secondary-button", "Load more", () => workers.list({more: true})));
  }
  const announcement = el("p", "worker-announcement", state.loading ? (state.mode === "list" ? "Loading workers…" : "Loading worker run…") : state.error);
  announcement.setAttribute("role", "status"); announcement.setAttribute("aria-live", "polite");
  nodes.push(announcement);
  if (state.error) {
    nodes.push(button("secondary-button", "Retry", () => state.mode === "list" ? workers.list() : workers.detail(state.selectedId)));
  }
  if (state.mode === "detail" && state.session && state.view) {
    const session = state.session;
    const meta = el("dl", "worker-meta");
    const firstTurn = state.view.turns[0];
    const started = session.started_at == null ? null : new Date(session.started_at * 1000).toLocaleString();
    const duration = session.duration_s ?? firstTurn?.result?.duration_s ?? (firstTurn?.finishedAt && firstTurn?.startedAt ? firstTurn.finishedAt - firstTurn.startedAt : null);
    const fields = terminal
      ? [["Model", session.model], ["Workspace", session.workspace], ["Created", new Date(session.created_at * 1000).toLocaleString()], ["Tokens", session.tokens], ["Status", session.status]]
      : [["Source", session.source], ["Model", session.model], ["Workspace", session.workspace], ["Git branch", session.git_branch], ["Git head", session.git_head], ["Started", started], ["Duration", duration == null ? null : formatDuration(duration)], ["Steps", session.steps], ["Tokens", session.tokens], ["Status", session.status], ["Exit code", session.exit_code], ["Parent run", session.parent_run]];
    for (const [label, value] of fields) {
      meta.append(el("dt", "", label), el("dd", "", value == null || value === "" ? "Not recorded" : String(value)));
    }
    nodes.push(meta, workerStatus.element);
    if (state.unsupported) nodes.push(el("p", "muted", "Some recorded event types cannot be shown."));
    if (state.view.turns.length > 1 && !terminal) nodes.push(el("p", "", "This run has multiple recorded turns."));
    if (!state.view.turns.length) nodes.push(el("p", "", "No recorded turn."));
    if (session.status === "running") nodes.push(el("p", "", terminal ? "A turn is running. New steps appear as they are recorded." : "Run not finished. New steps appear as they are recorded."));
    nodes.push(button("secondary-button", "Refresh", () => workers.detail(state.selectedId)));
    for (const turn of state.view.turns) {
      nodes.push(el("h2", "", terminal ? "You" : "Brief"), el("p", "worker-brief", turn.userText || (terminal ? "No message recorded." : "No brief recorded.")), el("h2", "", "Work"));
      nodes.push(...turnWork(turn, session.id, renderWorkers, state.eventCounts[turn.turn] ? "" : "No recorded events."));
    }
  }
  root.replaceChildren(...nodes);
  if (restoreHeading) title.focus({preventScroll: true});
  syncUI();
}

function setRoute(id) {
  const next = id ? `#/s/${id}` : "#/new";
  try {
    if (globalThis.location && globalThis.location.hash !== next) {
      globalThis.location.hash = next;
    }
  } catch {
    /* A page opened without a location still works. */
  }
}

function hashId() {
  const match = /^#\/s\/([A-Za-z0-9._:-]+)$/.exec(globalThis.location?.hash || "");
  return match ? match[1] : "";
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
  if (mobile && !open && document.activeElement?.closest?.(".sidebar")) {
    $("sidebar-toggle").focus();
  }
}

// Developer mode is the saved [ui] mode; it only adds panels (the drawer's Events
// tab and its run and memory internals) and never changes what a panel does.
let developerMode = false;
async function refreshMode(saved) {
  try { developerMode = (saved ?? await api("/settings")).mode === "developer"; } catch { /* Keep the last known mode. */ }
}
const drawer = setupDrawer({ api, currentId: () => S.activeId, root: $("details-drawer"), trigger: $("details-trigger"), developer: () => developerMode });
const showSettings = setupSettings({
  api,
  onSaved: async (saved) => {
    await refreshMode(saved);
    drawer.modeChanged();
    try {
      setHealth(await api("/health"));
      syncUI();
    } catch (error) {
      notify("Settings saved, but health could not be refreshed: " + error.message, {sticky: true});
    }
  },
});
function openSettings() {
  drawer.close();
  return showSettings();
}
const setupCard = setupModelCard({
  root: $("setup-card"),
  api,
  openSettings,
  connected: async () => {
    setHealth(await api("/health"));
    syncUI();
    input.focus();
  },
});

// ── The clock ──

function tick() {
  if (document.hidden) return;
  const now = nowSeconds();
  for (const [key, entry] of liveNodes) {
    if (!S.view || !S.view.turns.includes(entry.turn) || entry.turn.finishedAt) {
      liveNodes.delete(key);
      continue;
    }
    entry.text.textContent = liveText(entry.turn, now);
    entry.row.dataset.activity = activity(entry.turn, activityOptions(entry.turn, now));
  }
  for (const [item, entry] of runningCards) {
    if (item.status !== "running" || !S.view?.turns.includes(entry.turn)) {
      runningCards.delete(item);
      continue;
    }
    const state = toolState(item, { held: heldOf(entry.turn), now });
    entry.node.textContent = state.text;
    entry.card.dataset.state = state.state;
  }
  refreshStreamNotice();
}

// ── Starting up ──

async function init() {
  const fragment = globalThis.location?.hash || "";
  if (fragment.startsWith("#login=")) {
    // Remove the entire fragment before any request, including a failed exchange.
    try {
      globalThis.history.replaceState(null, "", globalThis.location.pathname + globalThis.location.search);
    } catch {
      lock("Could not clear the sign-in link. Close this tab and try again.");
      return;
    }
    try {
      const params = new URLSearchParams(fragment.slice(1));
      const code = params.get("login") || "";
      // The macOS app passes its launch identifier, which the gateway checks
      // when the app started it. It is a collision check, not authentication.
      const instance = params.get("instance") || "";
      const headers = { "Content-Type": "application/json" };
      if (/^[0-9a-f]{32}$/.test(instance)) headers["X-LocalMelo-Instance"] = instance;
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers,
        body: JSON.stringify({code}),
      });
      if (response.status !== 204) {
        lock("This sign-in link is invalid or expired. Get a new link and try again.");
        return;
      }
    } catch {
      lock("Could not sign in. Get a new link and try again.");
      return;
    }
  }
  try {
    setHealth(await api("/health", {allowLocked: true}));
  } catch (error) {
    lock(error.status === 401 ? "" : "Could not reach the gateway. Open a new sign-in link to try again.");
    return;
  }
  S.authenticated = true;
  $("lock-screen").hidden = true;
  $("app").hidden = false;
  $("app").inert = false;
  S.drafts = stored(KEY + "-drafts", {}, "sessionStorage");
  const savedOutbox = stored(KEY + "-outbox", {}, "sessionStorage");
  S.outbox = {};
  for (const [slot, entry] of Object.entries(savedOutbox)) {
    if (!entry || typeof entry !== "object" || typeof entry.clientRequestId !== "string" || typeof entry.text !== "string") continue;
    if (!["sending", "checking", "failed"].includes(entry.state)) continue;
    S.outbox[slot] = {...entry, state: entry.state === "failed" ? "failed" : "checking"};
  }
  input.value = S.drafts.new || "";
  resizeInput();
  sidebar(false);
  await refreshList();
  if (!S.authenticated) return;
  syncUI();
  // The hash wins: `#/s/<id>` opens that chat, `#/new` a new one. Without
  // one, the last chat this tab opened, else the newest. The hash id is
  // authoritative even when it is outside the list's first page: its 404
  // sends the page back to a new chat, as a gone chat does.
  const fromWorker = workerRoute();
  const fromTask = taskRoute();
  const fromHash = hashId();
  const wantsNew = (globalThis.location?.hash || "") === "#/new";
  let wanted = "";
  if (fromHash) {
    wanted = fromHash;
  } else if (!wantsNew) {
    const saved = stored(KEY + "-view").id;
    wanted = S.sessions.some((session) => session.id === saved)
      ? saved
      : (S.sessions[0]?.id ?? "");
  }
  if (fromTask !== null) {
    showTasks(fromTask);
  } else if (fromWorker !== null) {
    showWorkers(fromWorker);
  } else if (wanted) {
    await openSession(wanted);
  } else {
    setRoute(null);
    renderSessions();
    renderMessages(true);
    syncUI();
  }
  if (!S.authenticated) return;
  for (const [slot, entry] of Object.entries(S.outbox)) {
    if (entry.state === "checking") await checkSubmission(slot, entry.clientRequestId);
    if (!S.authenticated) return;
  }
  setInterval(tick, 1000);
  setInterval(() => {
    if (S.authenticated && !document.hidden) {
      refreshList();
      chatStatus.bump();
      api("/health").then((body) => { setHealth(body); syncUI(); }).catch(() => {});
    }
  }, LIST_POLL_MS);
  S.ready = true;
  syncUI();
  // The first load is done: a screenshot harness (and a curious person) can
  // tell the page is wired before they type into it.
  document.documentElement.dataset.ready = "1";
}

// ── Wiring ──

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    send();
  }
});
input.addEventListener("input", () => {
  presence();
  saveDraft();
  resizeInput();
  syncUI();
});
for (const id of ["new-btn", "settings-btn", "connection-btn", "model-btn"]) {
  const node = $(id);
  if (id === "new-btn") node.addEventListener("click", () => newChat());
  else node.addEventListener("click", openSettings);
}
for (const id of ["close-settings", "cancel-settings"]) $(id).addEventListener("click", () => {
  const dialog = $("settings-dialog");
  if (dialog?.close) dialog.close();
});
$("details-trigger").addEventListener("click", async () => {
  if (workerMode) return;
  if (drawer.close()) return;
  // A config written by hand wins: read the mode again as the drawer opens.
  await refreshMode();
  drawer.open();
});
$("stop-btn").addEventListener("click", stop);
$("reconnect-btn").addEventListener("click", async () => {
  const id = S.activeId;
  if (!id || S.stream.status !== "paused") return;
  if (S.stream.timer !== null) clearTimeout(S.stream.timer);
  S.stream.timer = null;
  try {
    await api(`/sessions/${encodeURIComponent(id)}`);
  } catch (error) {
    if (S.activeId === id && error.status === 404) {
      forgetSession(id);
      notify("This chat expired or the gateway restarted. Create a new chat to continue.", {sticky: true});
      return;
    }
  }
  if (S.activeId === id && S.stream.status === "paused") startStream(id);
});
$("approval-go").addEventListener("click", goToApproval);
$("scroll").addEventListener("scroll", updateApprovalBar);
$("level-btn").addEventListener("click", () => {
  if (!activeSession()) return;
  showLevelMenu(!S.levelsOpen);
});
document.addEventListener("click", (event) => {
  if (S.levelsOpen && !event.target.closest?.(".level-picker")) showLevelMenu(false);
  if (running.isOpen() && !event.target.closest?.(".running-picker")) running.toggle(false);
});
$("search").addEventListener("input", renderSessions);
$("sidebar-toggle").addEventListener("click", () =>
  sidebar(!$("app").classList.contains("sidebar-open")),
);
$("sidebar-backdrop").addEventListener("click", () => sidebar(false));
matchMedia("(max-width: 700px)").addEventListener?.("change", () => sidebar(false));
// Back and Forward move the hash; the chat follows it, as a reload would.
globalThis.addEventListener?.("hashchange", async () => {
  if (!S.authenticated) return;
  const task = taskRoute();
  if (task !== null) { showTasks(task); return; }
  const worker = workerRoute();
  if (worker !== null) { showWorkers(worker); return; }
  const wasWorker = workerMode;
  leaveWorkers();
  const id = hashId();
  if (id) {
    if (wasWorker || S.activeId !== id) await openSession(id);
    return;
  }
  if ((globalThis.location?.hash || "") === "#/new" && (wasWorker || S.activeId !== null)) newChat();
});
document.querySelectorAll("[data-prompt]").forEach((node) =>
  node.addEventListener("click", () => {
    input.value = node.dataset.prompt;
    saveDraft();
    resizeInput();
    syncUI();
    input.focus();
  }),
);
document.addEventListener("keydown", (event) => {
  if (!S.authenticated || workerMode) return;
  if (event.isComposing || event.keyCode === 229) return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    newChat();
    return;
  }
  if (event.key !== "Escape") return;
  // Close what is open first; a stop is the last thing Escape does.
  if (running.isOpen()) {
    running.toggle(false);
    $("running-btn").focus();
    return;
  }
  if (S.levelsOpen) {
    showLevelMenu(false);
    return;
  }
  if ($("settings-dialog")?.open) {
    event.preventDefault();
    $("settings-dialog").close?.();
    return;
  }
  if (drawer.close()) return;
  if ($("app").classList.contains("sidebar-open")) {
    sidebar(false);
    return;
  }
  for (const [id, state] of approvalStates) {
    if (!state.deny || !S.view?.pending.has(id)) continue;
    state.deny = false;
    messageSnapshot = "";
    renderMessages(false);
    document.getElementById(`approval-card-${id}`)?.querySelector("[data-approval-deny]")?.focus();
    return;
  }
  if (runningTurn() !== null) stop();
});
document.addEventListener("visibilitychange", () => {
  if (S.authenticated && !document.hidden) refreshList();
});

init();

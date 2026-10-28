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
import {
  activity,
  applyEvent,
  approvalTag,
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
let noticeTimer = null;
let sessionListSnapshot = "";
let messageSnapshot = "";
const liveNodes = new Map();
// The tool cards still running, so `tick` can move their clocks.
const runningCards = new Map();

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

/** One `/api` call; an API v2 error keeps its code and message. */
async function api(path, { body, method } = {}) {
  const response = await fetch("/api" + path, {
    method: method ?? (body ? "POST" : "GET"),
    ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
  });
  const payload = response.status === 204 ? {} : await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(
      payload?.error?.message || `Request failed (${response.status})`,
    );
    error.status = response.status;
    error.code = payload?.error?.code || "";
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
    !S.online ||
    !!turn ||
    (sending !== null && sending.state !== "failed") ||
    !input.value.trim();
  $("run-status").textContent = turn ? (held ? `Waiting: ${held}` : "Running on your backend") : "";
  $("level-btn").textContent = levelName(session?.level);
  $("level-btn").disabled = !session;
  if (!session && S.levelsOpen) showLevelMenu(false);
  $("new-btn").disabled = false;
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
    close.disabled = !!session.busy || (session.pending_approvals || 0) > 0;
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
  closeStream();
  S.activeId = id;
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
  closeStream();
  S.activeId = null;
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
  if (S.activeId !== id) return;
  closeStream();
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
    noteFrame(frame);
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
    if (S.activeId !== id || S.view === null) return;
    // A stream the browser will retry by itself is left alone; one it gave
    // up on is reopened here, after the same backoff, from our own cursor.
    if (source.readyState === 2) {
      S.stream.status = "paused";
      scheduleReconnect();
    }
  };
}

function closeStream() {
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
  if (!id || S.stream.timer !== null) return;
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
    if (S.activeId !== id) return;
    startStream(id);
  }, delay * 1000);
}

/** Mirror what a frame changed outside the view: title, model, the outbox. */
function applyOutcome(outcome) {
  if (outcome.clientRequestId) {
    for (const [slot, entry] of Object.entries(S.outbox)) {
      if (entry.clientRequestId !== outcome.clientRequestId) continue;
      // A text this page restored to the draft on a failed or unknown send
      // is now a real turn, so it is not a draft any more.
      if (entry.state === "failed" || entry.state === "checking") {
        forgetDraft(slot, entry.text);
      }
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
  if (outcome.model !== undefined || outcome.workspace !== undefined) {
    const session = S.sessions.find((item) => item.id === S.activeId);
    if (session) {
      if (outcome.model !== undefined) session.model = outcome.model;
      if (outcome.workspace !== undefined) session.workspace = outcome.workspace;
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
  if (S.stream.status !== "paused") return;
  // A sticky error outranks the banner; the banner returns once it goes.
  if (S.noticeKind !== "" && S.noticeKind !== "stream") return;
  const seconds = Math.max(0, Math.ceil(S.stream.retryAt - nowSeconds()));
  notify(
    `Live updates paused. Reconnecting in ${seconds}s…`,
    { sticky: true, kind: "stream" },
  );
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
    view?.pending?.size ?? 0,
    turns,
    S.outbox[S.activeId ?? "new"] ?? null,
  ]);
}

function renderMessages(forceScroll = false) {
  const scroller = $("scroll");
  const nearBottom =
    scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 100;
  const oldTop = scroller.scrollTop;
  const snapshot = viewSnapshot();
  if (snapshot === messageSnapshot) {
    if (forceScroll) scroller.scrollTop = scroller.scrollHeight;
    return;
  }
  messageSnapshot = snapshot;
  liveNodes.clear();
  runningCards.clear();
  const view = S.view;
  const turns = view?.turns ?? [];
  const outbox = pendingOutbox();
  const container = $("messages");
  const nodes = turns.map((turn) => turnNode(turn));
  if (outbox && !turns.some((turn) => turn.clientRequestId === outbox.clientRequestId)) {
    nodes.push(outboxNode(outbox, S.activeId ?? "new"));
  }
  container.replaceChildren(...nodes);
  $("welcome").hidden = nodes.length > 0 || !!S.activeId;
  if (forceScroll || nearBottom) scroller.scrollTop = scroller.scrollHeight;
  else scroller.scrollTop = oldTop;
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
      article.append(answerNode(turn.answer));
      article.append(el("p", "turn-caption", "Best effort"));
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
  const text = statusText(turn.status, turn.cause);
  const mark = text.severity === "error" ? "[x]" : text.severity === "ok" ? "[ok]" : "[-]";
  const node = el("p", "turn-status", `${mark} ${text.title}.`);
  node.dataset.severity = text.severity;
  if (turn.error) node.title = turn.error;
  return node;
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
  for (const item of turn.items) group.append(itemNode(item, turn));
  if (!turn.finishedAt) group.append(liveRow(turn));
  return group;
}

function heldOf(turn) {
  if (turn.finishedAt) return "";
  return S.view?.session?.held || "";
}

function itemNode(item, turn) {
  switch (item.kind) {
    case "tool":
      return toolCard(item, turn);
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

function toolCard(item, turn) {
  // A call of a held chat says why the model server is not being asked yet.
  const state = toolState(item, { held: heldOf(turn), now: nowSeconds() });
  const card = el("details", "tool-card");
  card.dataset.state = state.state;
  card.open = item.expanded === true;
  const summary = el("summary", "tool-summary");
  summary.append(el("span", "tool-mark", state.mark));
  summary.append(el("span", "tool-name", item.name));
  if (item.preview) summary.append(el("span", "tool-preview", printable(item.preview)));
  const meta = state.text ? el("span", "tool-meta", state.text) : null;
  if (meta) summary.append(meta);
  if (meta && state.state === "running") {
    runningCards.set(item, {item, turn, node: meta, card});
  }
  card.append(summary);
  card.append(toolBody(item));
  card.addEventListener("toggle", () => {
    item.expanded = card.open;
  });
  return card;
}

function toolBody(item) {
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
  if (item.output) body.append(outputBlock(item));
  for (const image of item.images ?? []) {
    const node = document.createElement("img");
    node.className = "tool-image";
    node.loading = "lazy";
    node.alt = `Image from ${item.name}`;
    node.src = `/api/sessions/${encodeURIComponent(S.activeId ?? "")}/blobs/${encodeURIComponent(image.sha256)}`;
    node.title = `${image.mime_type} · ${formatBytes(image.size_bytes)}`;
    body.append(node);
  }
  return body;
}

function outputBlock(item) {
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
        renderMessages(false);
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
  const text = input.value.trim();
  const slot = S.activeId ?? "new";
  const previous = S.outbox[slot];
  if (!text || runningTurn() !== null) return;
  if (previous && previous.state !== "failed") return;
  // The same text, failed after a check of unknown outcome, keeps its key:
  // the store answers `repeated` instead of running it twice.
  const again = previous && previous.state === "failed" && previous.text === text;
  const key = again ? previous.clientRequestId : newClientRequestId();
  S.outbox[slot] = { clientRequestId: key, text, state: "sending", error: "" };
  input.value = "";
  saveDraft();
  resizeInput();
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
      const created = body.session;
      forgetDraft("new", text);
      delete S.outbox.new;
      delete S.outbox[created.id];
      if (S.activeId === null) {
        S.sessions.unshift(created);
        S.activeId = created.id;
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
      // The outbox is memory only: the chat's saved draft is what survives
      // a reload, and the composer shows it again.
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
    const submission = body.submission ?? {};
    if (submission.state === "accepted") {
      forgetDraft(slot, entry.text);
      delete S.outbox[slot];
      if (sessionId === null && submission.session_id) {
        await refreshList();
        if (S.activeId === null) {
          S.activeId = submission.session_id;
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
    // Nothing in the outbox survives a reload; the draft does.
    keepDraft(slot, entry.text);
  } catch (error) {
    entry.state = "failed";
    entry.error = error.message;
    entry.unknown = true;
    keepDraft(slot, entry.text);
  }
  renderMessages(false);
  syncUI();
}

async function cancelSend(slot, entry) {
  const sessionId = slot === "new" ? null : slot;
  const path = sessionId
    ? `/sessions/${encodeURIComponent(sessionId)}/submissions/${encodeURIComponent(entry.clientRequestId)}/cancel`
    : `/submissions/${encodeURIComponent(entry.clientRequestId)}/cancel`;
  try {
    await api(path, { body: { text: entry.text } });
    delete S.outbox[slot];
    if (!input.value) {
      input.value = entry.text;
      saveDraft();
      resizeInput();
    }
    renderMessages(false);
    syncUI();
  } catch (error) {
    if (error.code === "submission_withdrawn") {
      // The store never took it, so the text comes back to the composer.
      delete S.outbox[slot];
      if (!input.value) {
        input.value = entry.text;
        saveDraft();
        resizeInput();
      }
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
  const id = S.activeId;
  if (!id) return;
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
    request?.catch?.(() => {});
  } catch {
    /* A browser may refuse fetch, or a test may not allow it. */
  }
}

// ── Routing and the sidebar ──

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

function settingsNotice() {
  notify(
    "Connection settings are not on this page yet. Set your model with melo --reconfigure, then reload.",
  );
}

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
  S.drafts = stored(KEY + "-drafts", {}, "sessionStorage");
  input.value = S.drafts.new || "";
  resizeInput();
  sidebar(false);
  const health = api("/health")
    .then((body) => {
      S.health = body;
      return body;
    })
    .catch(() => null);
  await Promise.allSettled([health, refreshList()]);
  if (!S.online) {
    notify("Could not reach the gateway. Check that the gateway is running.", { sticky: true });
  } else if (S.health && S.health.configured === false) {
    notify("No model is configured yet. Run melo --reconfigure to connect one.", { sticky: true });
  } else if (S.health?.sessions && S.health.sessions.store !== "ok") {
    notify(S.health.sessions.store_error || "Chats are unavailable.", { sticky: true });
  }
  // The hash wins: `#/s/<id>` opens that chat, `#/new` a new one. Without
  // one, the last chat this tab opened, else the newest. The hash id is
  // authoritative even when it is outside the list's first page: its 404
  // sends the page back to a new chat, as a gone chat does.
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
  if (wanted) {
    await openSession(wanted);
  } else {
    setRoute(null);
    renderSessions();
    renderMessages(true);
    syncUI();
  }
  setInterval(tick, 1000);
  setInterval(() => {
    if (!document.hidden) refreshList();
  }, LIST_POLL_MS);
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
for (const id of ["new-btn", "settings-btn", "connection-btn", "model-btn", "setup-btn"]) {
  const node = $(id);
  if (id === "new-btn") node.addEventListener("click", () => newChat());
  else node.addEventListener("click", settingsNotice);
}
$("close-settings").addEventListener("click", () => {
  const dialog = $("settings-dialog");
  if (dialog?.close) dialog.close();
});
$("stop-btn").addEventListener("click", stop);
$("level-btn").addEventListener("click", () => {
  if (!activeSession()) return;
  showLevelMenu(!S.levelsOpen);
});
document.addEventListener("click", (event) => {
  if (S.levelsOpen && !event.target.closest?.(".level-picker")) showLevelMenu(false);
});
$("search").addEventListener("input", renderSessions);
$("sidebar-toggle").addEventListener("click", () =>
  sidebar(!$("app").classList.contains("sidebar-open")),
);
$("sidebar-backdrop").addEventListener("click", () => sidebar(false));
matchMedia("(max-width: 700px)").addEventListener?.("change", () => sidebar(false));
// Back and Forward move the hash; the chat follows it, as a reload would.
globalThis.addEventListener?.("hashchange", async () => {
  const id = hashId();
  if (id) {
    if (S.activeId !== id) await openSession(id);
    return;
  }
  if ((globalThis.location?.hash || "") === "#/new" && S.activeId !== null) newChat();
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
  if (event.isComposing || event.keyCode === 229) return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    newChat();
    return;
  }
  if (event.key !== "Escape") return;
  // Close what is open first; a stop is the last thing Escape does.
  if (S.levelsOpen) {
    showLevelMenu(false);
    return;
  }
  if ($("app").classList.contains("sidebar-open")) {
    sidebar(false);
    return;
  }
  if ($("settings-dialog")?.open) {
    $("settings-dialog").close?.();
    return;
  }
  if (runningTurn() !== null) stop();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshList();
});

init();

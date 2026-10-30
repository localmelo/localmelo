/**
 * One session's turns, folded from its stored events, and the words they use.
 *
 * `applyEvent` folds one SSE frame (or one stored event of
 * `GET /api/sessions/{id}/turns`) into the view. Nothing here touches the
 * DOM, so the reducer is tested on its own; `workspace.js` renders what it
 * holds.
 *
 * The words and numbers mirror the terminal's, not a second wording:
 * `support/sessions/display.py` for durations, tokens, sizes, run statuses,
 * approval tags and tool output, and `support/progress_line.py` for the past
 * facts of a first turn. A string that reads differently here is a bug.
 */

const SEPARATOR = " · ";
// `support/cli/render.py`: these two calls end after SESSION_TOOL_TIMEOUT_S.
const LIMITED_TOOLS = new Set(["shell_exec", "python_exec"]);
const SESSION_TOOL_TIMEOUT_S = 600;
const ITEM_MARKS = {done: "✓", current: "◐", skipped: "–", pending: "○"};
const SIMILAR_HINT = "like a similar task:";

// ── Values ──

/** A finite number of *value*, or null. */
function num(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

/** A finite number of *value* cut to an int, or null. */
function int(value) {
  const number = num(value);
  return number === null ? null : Math.trunc(number);
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function objectOf(value) {
  return isObject(value) ? value : {};
}

// ── Text mirrors (support/sessions/display.py, support/progress_line.py) ──

/** `split_thinking`: the reasoning in a `<think>` block, and the text. */
export function splitThinking(text) {
  const value = String(text ?? "");
  const block = /<think>(.*?)<\/think>/is.exec(value);
  if (block !== null) {
    const rest = value.slice(0, block.index) + value.slice(block.index + block[0].length);
    return [block[1].trim(), rest.trim()];
  }
  const opening = /^\s*<think>/i.exec(value);
  if (opening !== null) return [value.slice(opening[0].length).trim(), ""];
  const closing = /<\/think>/i.exec(value);
  if (closing !== null) {
    const before = value.slice(0, closing.index).trim();
    return [before, value.slice(closing.index + closing[0].length).trim()];
  }
  return ["", value.trim()];
}

/** `run_elapsed`: `9s`, `1m03s`, `1h02m`. */
export function runElapsed(seconds) {
  const total = Math.max(0, Math.trunc(Number(seconds) || 0));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m${String(total % 60).padStart(2, "0")}s`;
  return `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, "0")}m`;
}

/** `tool_elapsed`: a clock, `0:41`, `12:03`, `1:02:03`. */
export function toolElapsed(seconds) {
  const total = Math.max(0, Math.trunc(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}:${String(total % 60).padStart(2, "0")}`;
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

/** `format_duration`: tenths below 10 s, then the progress line's run time. */
export function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 10) return `${(Math.floor(value * 10) / 10).toFixed(1)}s`;
  return runElapsed(value);
}

/** `format_tokens`: `412`, `3.4k`, `61k`, `1.2M`; cut, never rounded up. */
export function formatTokens(count) {
  const value = Math.max(0, Math.trunc(Number(count) || 0));
  if (value < 1_000) return String(value);
  if (value < 10_000) return `${(Math.floor(value / 100) / 10).toFixed(1)}k`;
  if (value < 1_000_000) return `${Math.floor(value / 1_000)}k`;
  return `${(Math.floor(value / 100_000) / 10).toFixed(1)}M`;
}

/** `format_bytes`: `412 bytes`, `1.2 KB`, `312 KB`, `12 MB`; decimal, cut. */
export function formatBytes(count) {
  let value = Math.max(0, Math.trunc(Number(count) || 0));
  if (value < 1_000) return `${value} bytes`;
  for (const unit of ["KB", "MB", "GB", "TB"]) {
    value /= 1_000;
    if (value < 1_000 || unit === "TB") {
      if (value < 10) return `${(Math.floor(value * 10) / 10).toFixed(1)} ${unit}`;
      return `${Math.floor(value)} ${unit}`;
    }
  }
  return `${Math.floor(value)} TB`;
}

/** `wall_range`: `40-55s`, `2-3 min`, `45s-2 min`, one value when equal. */
export function wallRange(lo, hi) {
  const low = Math.max(0, Math.min(lo, hi));
  const high = Math.max(0, Math.max(lo, hi));
  if (high < 60) {
    const from = Math.floor(low);
    const to = Math.ceil(high);
    return from === to ? `${from}s` : `${from}-${to}s`;
  }
  const highMinutes = Math.ceil(high / 60);
  if (low < 60) return `${Math.floor(low)}s-${highMinutes} min`;
  const lowMinutes = Math.floor(low / 60);
  return lowMinutes === highMinutes ? `${lowMinutes} min` : `${lowMinutes}-${highMinutes} min`;
}

const stepsWord = (count) => (count === 1 ? "1 step" : `${count} steps`);
const stepRange = (lo, hi) => (lo === hi ? `${lo}` : `${lo}-${hi}`);

/** `printable`: one line, with control and format characters as `?`. */
export function printable(text) {
  const collapsed = String(text ?? "")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ");
  return collapsed.replace(/[\p{Cc}\p{Cf}\p{Cs}\p{Co}]/gu, (character) =>
    character === "‌" || character === "‍" ? character : "?",
  );
}

const ESCAPES =
  /\u001b\[[0-?]*[ -/]*[@-~]|\u001b[\][PX^_][\s\S]*?(?:\u0007|\u001b\\|\u009c)|\u001b[ -/]+[0-~]|\u001b[0-~]|\u009b[0-?]*[ -/]*[@-~]|[\u0090\u0098\u009d\u009e\u009f][\s\S]*?(?:\u0007|\u001b\\|\u009c)/g;
const CONTROLS = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u{E0000}-\u{E007F}]/gu;

/** `clean_output`: what a terminal would leave on screen, with nothing to act on. */
export function cleanOutput(text) {
  const plain = String(text ?? "")
    .replace(/\r\n/g, "\n")
    .replace(ESCAPES, "");
  const lines = plain
    .split("\n")
    .map((line) => line.slice(line.lastIndexOf("\r") + 1));
  return lines.join("\n").replace(CONTROLS, "");
}

/** `items_text`: the steps of past runs, each with its mark. */
export function itemsText(items, level) {
  if (!Array.isArray(items) || items.length === 0) return null;
  const shown = [];
  for (const item of items) {
    if (!isObject(item)) continue;
    let label = printable(String(item.label ?? ""));
    if (!label) continue;
    const repeat = int(item.repeat);
    if (repeat !== null && repeat > 1) label = `${label} (×${repeat})`;
    // Level 1 has no unticked to-do list: a step not matched yet is plain.
    const mark =
      level < 2 && item.state === "pending" ? null : ITEM_MARKS[String(item.state)];
    shown.push(mark ? `${label} ${mark}` : label);
  }
  return shown.length === 0 ? null : shown.join(SEPARATOR);
}

/** The hint a similar match's past facts follow; `_similar_hint`. */
function similarHint(title) {
  const text = typeof title === "string" ? printable(title).trim() : "";
  return text ? `like ${text}:` : SIMILAR_HINT;
}

/** `past_facts`: (steps text, times text, a single past run). */
export function pastFacts(past, level, hint = null) {
  if (!isObject(past)) return [null, null, false];
  const runs = int(past.runs) ?? 0;
  if (level >= 2) {
    const lo = num(past.wall_lo_s);
    const hi = num(past.wall_hi_s);
    return [null, lo === null || hi === null ? null : `past runs ${wallRange(lo, hi)}`, false];
  }
  if (runs <= 1) {
    const steps = int(past.last_steps) ?? int(past.steps_hi);
    const wall = num(past.last_wall_s) ?? num(past.wall_hi_s);
    if (steps === null) return [null, null, false];
    return [
      `${hint || "last time"} ${stepsWord(steps)}`,
      wall === null ? null : runElapsed(wall),
      true,
    ];
  }
  const loSteps = int(past.steps_lo);
  const hiSteps = int(past.steps_hi);
  if (loSteps === null || hiSteps === null) return [null, null, false];
  const lo = num(past.wall_lo_s);
  const hi = num(past.wall_hi_s);
  return [
    `${hint || "past"} ${stepRange(loSteps, hiSteps)} steps`,
    lo === null || hi === null ? null : wallRange(lo, hi),
    false,
  ];
}

/**
 * `format_past`: a `run.progress` payload's past facts and past runs' steps.
 *
 * Level 1 only: `last time 6 steps, 2m10s: Read …`, `past 5-8 steps, 2-3 min
 * · last time: Read …`, or a similar match's `like <title>: …`. Empty below
 * level 1, or with neither facts nor steps. *width* is a character budget:
 * the times go first, then the steps, then the rest is cut with `…`.
 */
export function formatPast(payload, width = Number.POSITIVE_INFINITY) {
  const data = objectOf(payload);
  const level = int(data.level) ?? 0;
  if (level < 1) return "";
  const basis = objectOf(data.basis);
  const similar = basis.match === "similar";
  const hint = similar ? similarHint(basis.title) : null;
  const [pastSteps, pastTimes, single] = pastFacts(data.past, level, hint);
  const items = itemsText(data.items, level);
  let itemsTitle = null;
  if (level === 1) itemsTitle = similar ? (pastSteps ? "its steps:" : hint) : "last time:";
  const merge = single && level === 1;

  const compose = (dropped) => {
    const past = pastSteps || "";
    const withTimes =
      pastTimes && !dropped.has("past_times")
        ? past
          ? `${past}, ${pastTimes}`
          : pastTimes
        : past;
    const shown = dropped.has("items") ? null : items;
    if (shown && merge && withTimes) return `${withTimes}: ${shown}`;
    const parts = [];
    if (withTimes) parts.push(withTimes);
    if (shown) parts.push(itemsTitle ? `${itemsTitle} ${shown}` : shown);
    return parts.join(SEPARATOR);
  };

  const dropped = new Set();
  let text = compose(dropped);
  for (const part of ["past_times", "items"]) {
    if (text.length <= width) break;
    dropped.add(part);
    text = compose(dropped);
  }
  if (text.length <= width) return text;
  const kept = Array.from(text);
  if (width < 1) return "";
  return kept.slice(0, Math.max(0, width - 1)).join("") + "…";
}

// status -> {word, severity, title}; `(status, cause)` rows win.
const STATUS_TEXT = {
  completed: {word: "Done", severity: "ok", title: "Done"},
  cancelled: {word: "Stopped", severity: "stopped", title: "Stopped"},
  failed: {word: "Failed", severity: "warn", title: "A check stopped the run"},
  stuck: {
    word: "Stuck",
    severity: "warn",
    title: "Stopped: the same calls kept repeating",
  },
  gate_stopped: {
    word: "Stopped after reflection",
    severity: "warn",
    title: "Stopped after reflection",
  },
  budget_exhausted: {word: "Out of steps", severity: "warn", title: "Ran out of steps"},
  context_budget_exceeded: {word: "Context too large", severity: "warn", title: "Context budget exceeded"},
  timeout: {word: "Timed out", severity: "warn", title: "Timed out"},
  provider_error: {
    word: "Model server error",
    severity: "error",
    title: "Could not get an answer from the model server",
  },
  setup_error: {
    word: "Tools failed to start",
    severity: "error",
    title: "Tools could not start",
  },
  crashed: {
    word: "Crashed",
    severity: "error",
    title: "Something went wrong inside localmelo",
  },
  "crashed|run": {
    word: "Crashed",
    severity: "error",
    title: "Something went wrong inside localmelo",
  },
  "crashed|crash": {
    word: "Interrupted",
    severity: "warn",
    title: "Interrupted: melo stopped unexpectedly",
  },
  "cancelled|user_stop": {word: "Stopped", severity: "stopped", title: "Stopped by you"},
  "cancelled|crash": {
    word: "Interrupted",
    severity: "warn",
    title: "Interrupted: melo stopped unexpectedly",
  },
  "cancelled|shutdown": {
    word: "Interrupted",
    severity: "warn",
    title: "Interrupted: melo stopped before this turn finished",
  },
  "cancelled|closed": {
    word: "Stopped",
    severity: "stopped",
    title: "Stopped: the session was closed",
  },
  "cancelled|takeover": {
    word: "Stopped",
    severity: "stopped",
    title: "Stopped: another melo window took over this session",
  },
  "cancelled|park": {word: "Parked", severity: "stopped", title: "Parked"},
};

/** `status_text`: the word, severity and title of a run's end. */
export function statusText(status, cause = "") {
  const found =
    STATUS_TEXT[`${status}|${cause}`] ?? STATUS_TEXT[status] ?? null;
  if (found !== null) return found;
  const unknown = status || "Unknown";
  return {word: unknown, severity: "warn", title: unknown};
}

const STATUS_MARK = {ok: "✓", error: "✗", warn: "⚠", stopped: "■"};

/** The glyph of a status line, as the terminal draws it (`support/cli/theme.py`). */
export function statusMark(severity) {
  return STATUS_MARK[severity] ?? STATUS_MARK.warn;
}

// status -> the first thing to do about a run that ended so. A stop or an
// interruption is a neutral line, and `context_budget_exceeded` has no button:
// only a shorter request or a new chat helps, and the hint says so.
const NEXT_ACTION = {
  provider_error: "settings",
  setup_error: "tools",
  crashed: "copy",
  "crashed|crash": null,
  budget_exhausted: "continue",
  timeout: "continue",
  failed: "retry",
  stuck: "retry",
  gate_stopped: "retry",
};

/** The one next step a run's end offers first: `settings`, `tools`, `continue`, `retry`, `copy` or null. */
export function nextAction(status, cause = "") {
  const own = (key) => Object.hasOwn(NEXT_ACTION, key);
  const keyed = `${status}|${cause}`;
  return own(keyed) ? NEXT_ACTION[keyed] : own(status) ? NEXT_ACTION[status] : null;
}

/** `status_line`: one bounded line, without the user's task text. */
export function statusLine(status, error) {
  let reason = String(error ?? "").split(/\s+/).filter(Boolean).join(" ");
  const characters = Array.from(reason);
  if (characters.length > 500) reason = characters.slice(0, 497).join("") + "...";
  return `melo: run ended with status ${status}: ${reason || "no error message"}`;
}

/** Continue keeps tool side effects and is only safe to offer on the newest turn. */
export function canContinue(turn, newest) {
  return newest && !!turn.finishedAt && !turn.undone && Number(turn.result?.n_tool) > 0 &&
    ["budget_exhausted", "context_budget_exceeded", "gate_stopped", "provider_error", "timeout", "cancelled"].includes(turn.status);
}

/** A retry undoes the newest finished turn, never an earlier or undone one. */
export function canRetry(turn, newest) {
  return newest && !turn.undone && !!turn.finishedAt && turn.status !== "completed";
}

/** `approval_tag`: the short and long text of how an approval ended, or null. */
export function approvalTag(status, errorCategory, approval) {
  if (!isObject(approval)) return null;
  const origin = approval.origin;
  const reason = String(approval.reason ?? "")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ");
  if (origin === "withdrawn") {
    if (status !== "cancelled") return null;
    return {short: "not run", long: "Not run: the turn was stopped"};
  }
  if (origin === "error") {
    return {short: "approval failed", long: "Approval failed; the call did not run"};
  }
  if (origin === "timeout") {
    return {short: "not answered in time", long: "Not approved in time"};
  }
  if (approval.allowed) {
    if (origin === "session") {
      return {
        short: "allowed by a session grant",
        long: "Allowed by a grant for this session",
      };
    }
    if (origin === "policy") {
      return {short: "auto-allowed", long: "Allowed: approvals are set to auto"};
    }
    if (approval.remember) {
      return {short: "allowed for the session", long: "Allowed by you for this session"};
    }
    return {short: "allowed", long: "Allowed once by you"};
  }
  if (origin === "policy") {
    return {
      short: reason ? `denied by policy: ${reason}` : "denied by policy",
      long: reason
        ? `Denied by the approval policy: ${reason}`
        : "Denied by the approval policy",
    };
  }
  if (origin === "user" && (status === "blocked" || errorCategory === "approval_denied")) {
    return reason
      ? {short: `denied: ${reason}`, long: `Denied by you: ${reason}`}
      : {short: "denied", long: "Denied by you"};
  }
  return null;
}

// ── The view ──

/** A view of one session before its detail and turns arrive. */
export function newView(id) {
  return {
    id,
    session: null,
    tokens: 0,
    contextTokens: 0,
    grants: [],
    grantedAt: {},
    hasEarlier: false,
    replayFrom: 0,
    lastEventId: 0,
    pending: new Map(),
    resumed: null,
    turns: [],
  };
}

/** One turn from a `turn_json` row (`GET /api/sessions/{id}/turns`). */
export function newTurn(row) {
  const source = objectOf(row);
  const [, answer] = splitThinking(source.answer ?? "");
  const result = isObject(source.result) ? source.result : null;
  return {
    id: String(source.id ?? ""),
    turn: int(source.turn) ?? 0,
    runId: String(source.run_id ?? ""),
    userText: String(source.user_text ?? ""),
    status: String(source.status ?? ""),
    cause: String(source.cause ?? ""),
    answer,
    result,
    undone: source.undone === true,
    startedAt: num(source.started_at) ?? 0,
    finishedAt: num(source.finished_at) ?? 0,
    clientRequestId: String(source.client_request_id ?? ""),
    runningHere: source.running_here === true,
    maxSteps: 0,
    approvalsOn: true,
    step: 0,
    attempt: 0,
    outputTokens: 0,
    heldText: "",
    unstartedCalls: new Map(),
    items: [],
    progress: null,
    retryUntil: 0,
    userToggledGroup: false,
    error: "",
  };
}

/** The turn numbered *number* of *view*, or null. */
export function findTurn(view, number) {
  return view.turns.find((turn) => turn.turn === number) ?? null;
}

function addTurn(view, turn) {
  const index = view.turns.findIndex((other) => other.turn > turn.turn);
  if (index < 0) view.turns.push(turn);
  else view.turns.splice(index, 0, turn);
}

/** The turn a `turn.started` frame names, kept if a skeleton made it already. */
function ensureTurn(view, number, fields = {}) {
  let turn = findTurn(view, number);
  if (turn === null) {
    turn = newTurn({
      id: fields.id ?? "",
      turn: number,
      user_text: fields.userText ?? "",
      client_request_id: fields.clientRequestId ?? "",
      status: "running",
      running_here: true,
    });
    addTurn(view, turn);
  }
  if (fields.id && !turn.id) turn.id = fields.id;
  if (fields.userText && !turn.userText) turn.userText = fields.userText;
  if (fields.clientRequestId && !turn.clientRequestId) {
    turn.clientRequestId = fields.clientRequestId;
  }
  return turn;
}

/** A call the reply listed that never started: a `not run` card, as the CLI prints. */
function unstartedItem(turn, call, step) {
  return {
    kind: "tool",
    key: `${turn.turn}:${step}:${call.id}`,
    callId: String(call.id ?? ""),
    step: int(step) ?? 0,
    name: String(call.name ?? ""),
    arguments: call.arguments ?? {},
    preview: String(call.preview ?? ""),
    status: "not_run",
    startedTs: 0,
    durationS: null,
    output: "",
    outputChars: 0,
    error: "",
    errorCategory: "none",
    images: [],
    approval: null,
    exitCode: null,
    pendingRequestId: null,
  };
}

/** Move the calls a reply listed but no `tool.started` followed into cards. */
function flushUnstarted(turn) {
  if (turn.unstartedCalls.size === 0) return;
  for (const call of turn.unstartedCalls.values()) {
    turn.items.push(unstartedItem(turn, call, call.step ?? turn.step));
  }
  turn.unstartedCalls.clear();
}

function clipped(text, max = 160) {
  const line = String(text ?? "").replace(/\s+/g, " ").trim();
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

/** The one muted line a `note`, `inbox.accepted`, `memory` or `recall` event shows. */
function factLine(type, data) {
  if (type === "note") return `Note${data.kind ? ` (${data.kind})` : ""}: ${clipped(data.text)}`;
  if (type === "inbox.accepted") return `Message accepted: ${clipped(data.text)}`;
  if (type === "memory") return `Memory ${data.action ?? "decision"}${data.kind ? `: ${data.kind}` : ""}`;
  const hits = Array.isArray(data.hits) ? data.hits.length : 0;
  return `Recalled ${hits} ${hits === 1 ? "item" : "items"}${data.trigger ? ` (${data.trigger})` : ""}`;
}

/** The newest tool card of *callId*, made when no `tool.started` came first. */
function newestTool(turn, callId, name = "") {
  for (let index = turn.items.length - 1; index >= 0; index -= 1) {
    const item = turn.items[index];
    if (item.kind === "tool" && String(item.callId) === String(callId)) return item;
  }
  const item = {
    kind: "tool",
    key: `${turn.turn}:${turn.step}:${callId}`,
    callId: String(callId ?? ""),
    step: turn.step,
    name: String(name),
    arguments: {},
    preview: "",
    status: "running",
    startedTs: 0,
    durationS: null,
    output: "",
    outputChars: 0,
    error: "",
    errorCategory: "none",
    images: [],
    approval: null,
    exitCode: null,
    pendingRequestId: null,
  };
  turn.items.push(item);
  return item;
}

/**
 * Fold one event of `GET /api/sessions/{id}/events` into *view*.
 *
 * *frame* is the parsed JSON object of one SSE frame (or of one entry of a
 * turn's `events`): its own `id` is the cursor, and a frame at or below
 * `view.lastEventId` is skipped unless *ignoreCursor* (a turn loaded for
 * `Show earlier turns`). Returns what the page has to mirror outside the
 * view: `{clientRequestId}` for a `turn.started`, `{title}` for
 * `session.title`, and `{model, workspace, workspaceLabel}` for
 * `session.updated`.
 */
export function applyEvent(view, frame, {ignoreCursor = false} = {}) {
  if (!isObject(frame)) return {};
  const id = num(frame.id);
  if (!ignoreCursor && id !== null && id <= view.lastEventId) return {};
  if (id !== null) view.lastEventId = Math.max(view.lastEventId, id);
  const data = objectOf(frame.data);
  const type = String(frame.type ?? "");
  const number = int(frame.turn ?? data.turn) ?? 0;
  const ts = num(frame.ts) ?? 0;
  const turn =
    type === "turn.started" ? ensureTurn(view, number, {
      id: data.turn_id,
      userText: data.user_text,
      clientRequestId: data.client_request_id,
    }) : findTurn(view, number);

  if (type === "turn.started") {
    const finished = Boolean(turn.status) && turn.status !== "running";
    if (!finished) {
      turn.status = "running";
      turn.runningHere = true;
      turn.startedAt = ts || turn.startedAt;
    }
    return {clientRequestId: String(data.client_request_id ?? "")};
  }
  if (type === "session.title") return {title: String(data.title ?? "")};
  if (type === "session.updated") {
    if (view.session !== null) {
      if (data.model !== undefined) view.session.model = data.model;
      if (data.workspace !== undefined) view.session.workspace = data.workspace;
      if (data.workspace_label !== undefined) view.session.workspace_label = data.workspace_label;
    }
    return {model: data.model, workspace: data.workspace, ...(data.workspace_label === undefined ? {} : {workspaceLabel: data.workspace_label})};
  }
  if (turn === null) return {};

  switch (type) {
    case "run.started": {
      turn.runId = String(frame.run_id ?? "");
      turn.maxSteps = int(data.max_steps) ?? 30;
      turn.approvalsOn = data.approval !== false;
      if (data.approval === false) {
        turn.items.unshift({kind: "notice", noticeKind: "yolo", step: 0, message: "No approvals for this turn (--yolo)."});
      }
      break;
    }
    case "step.started": {
      if (turn.heldText) {
        turn.items.push({kind: "assistant_text", text: turn.heldText});
        turn.heldText = "";
      }
      flushUnstarted(turn);
      turn.step = int(data.step) ?? turn.step;
      turn.attempt = int(data.attempt) ?? turn.attempt;
      turn.retryUntil = 0;
      break;
    }
    case "assistant.message": {
      const [splitThinkingText, splitText] = splitThinking(data.content ?? "");
      const thinking = String(data.thinking ?? "").trim()
        ? String(data.thinking).trim()
        : splitThinkingText;
      if (thinking) {
        turn.items.push({
          kind: "thinking",
          text: thinking,
          durationS: num(data.duration_s),
        });
      }
      const usage = objectOf(data.usage);
      const completion = int(usage.completion_tokens);
      if (completion !== null) turn.outputTokens += Math.max(0, completion);
      const prompt = int(usage.prompt_tokens);
      if (prompt !== null && prompt > 0) view.contextTokens = prompt;
      const calls = Array.isArray(data.tool_calls) ? data.tool_calls : [];
      if (calls.length > 0) {
        if (splitText.trim()) {
          turn.items.push({kind: "assistant_text", text: splitText});
        }
        for (const call of calls) {
          if (!isObject(call)) continue;
          turn.unstartedCalls.set(String(call.id ?? ""), {
            id: String(call.id ?? ""),
            name: String(call.name ?? ""),
            arguments: call.arguments ?? {},
            preview: String(call.preview ?? ""),
            step: turn.step,
          });
        }
      } else {
        turn.heldText = splitText;
      }
      break;
    }
    case "tool.started": {
      const callId = String(data.call_id ?? "");
      turn.unstartedCalls.delete(callId);
      turn.items.push({
        kind: "tool",
        key: `${turn.turn}:${int(data.step) ?? turn.step}:${callId}`,
        callId,
        step: int(data.step) ?? turn.step,
        name: String(data.name ?? ""),
        arguments: data.arguments ?? {},
        preview: String(data.preview ?? ""),
        status: "running",
        startedTs: ts,
        durationS: null,
        output: "",
        outputChars: 0,
        error: "",
        errorCategory: "none",
        images: [],
        approval: null,
        exitCode: null,
        pendingRequestId: null,
      });
      break;
    }
    case "approval.requested": {
      const item = newestTool(turn, data.call_id, data.tool_name);
      item.status = "waiting_approval";
      item.pendingRequestId = String(data.request_id ?? "");
      const requestId = String(data.request_id ?? "");
      view.pending.set(requestId, {
        request_id: data.request_id,
        call_id: data.call_id,
        tool_name: data.tool_name,
        arguments: data.arguments ?? {},
        preview: data.preview ?? "",
        workspace_root: data.workspace_root ?? null,
        target: null,
        ...(view.pending.get(requestId) ?? {}),
      });
      break;
    }
    case "approval.resolved": {
      const requestId = String(data.request_id ?? "");
      view.pending.delete(requestId);
      const item = newestTool(turn, data.call_id, data.tool_name);
      const allowed = data.outcome === "allowed";
      if (item.pendingRequestId === requestId) {
        item.pendingRequestId = null;
        // A withdrawn approval never ran the call; a denied one is blocked.
        item.status = allowed ? "running" : data.outcome === "withdrawn" ? "not_run" : "blocked";
      }
      const toolName = String(data.tool_name ?? "");
      if (allowed && data.remember === true && toolName && !view.grants.includes(toolName)) {
        view.grants.push(toolName);
        view.grants.sort();
        view.grantedAt[toolName] = ts;
      }
      break;
    }
    case "tool.finished": {
      const item = newestTool(turn, data.call_id, data.name);
      item.status = String(data.status ?? "");
      item.errorCategory = String(data.error_category ?? "none");
      item.output = String(data.output ?? "");
      item.outputChars = int(data.output_chars) ?? item.output.length;
      item.error = String(data.error ?? "");
      item.durationS = num(data.duration_s);
      item.images = Array.isArray(data.images) ? data.images : [];
      item.approval = isObject(data.approval) ? data.approval : null;
      item.exitCode = int(data.exit_code);
      item.pendingRequestId = null;
      break;
    }
    case "notice": {
      if (data.kind === "retry") {
        const delay = num(objectOf(data.detail).delay_s) ?? 0;
        turn.retryUntil = ts + delay;
        const step = int(data.step) ?? turn.step;
        const retry = [...turn.items]
          .reverse()
          .find(
            (item) =>
              item.kind === "notice" && item.noticeKind === "retry" && item.step === step,
          );
        if (retry) retry.message = String(data.message ?? "");
        else {
          turn.items.push({
            kind: "notice",
            noticeKind: "retry",
            step,
            message: String(data.message ?? ""),
          });
        }
        break;
      }
      turn.items.push({
        kind: "notice",
        noticeKind: String(data.kind ?? ""),
        step: int(data.step) ?? turn.step,
        message: String(data.message ?? ""),
      });
      break;
    }
    case "reflection.started": {
      turn.items.push({
        kind: "reflection",
        attempt: int(data.attempt) ?? turn.attempt,
        status: "running",
        action: null,
        rationale: "",
        summary: "",
        requests: 1,
        readable: true,
        continueRun: false,
        durationS: null,
      });
      break;
    }
    case "reflection": {
      const attempt = int(data.attempt) ?? turn.attempt;
      let item = [...turn.items]
        .reverse()
        .find((other) => other.kind === "reflection" && other.attempt === attempt && other.status === "running");
      if (item === undefined) {
        item = {kind: "reflection", attempt, status: "done"};
        turn.items.push(item);
      }
      item.status = "done";
      item.action = data.action ?? null;
      item.rationale = String(data.rationale ?? "");
      item.summary = String(data.summary ?? "");
      item.requests = int(data.requests) ?? 1;
      item.readable = data.readable !== false;
      item.continueRun = data.continue_run === true;
      item.durationS = num(data.duration_s);
      const completion = int(objectOf(data.usage).completion_tokens);
      if (completion !== null) turn.outputTokens += Math.max(0, completion);
      break;
    }
    case "context_budget":
      // The status row shows the numbers; a line per request would bury the work.
      break;
    case "note":
    case "inbox.accepted":
    case "memory":
    case "recall": {
      turn.items.push({
        kind: "fact",
        factKind: type,
        step: int(data.step) ?? turn.step,
        message: factLine(type, data),
      });
      break;
    }
    case "run.progress": {
      turn.progress = data;
      const step = int(data.step);
      if (step !== null) turn.step = step;
      const attempt = int(data.attempt);
      if (attempt !== null) turn.attempt = attempt;
      break;
    }
    case "run.finished": {
      turn.result = isObject(data.result) ? data.result : null;
      flushUnstarted(turn);
      const [, answer] = splitThinking(turn.result?.result ?? "");
      if (turn.heldText && turn.heldText.trim() !== answer.trim()) {
        turn.items.push({kind: "assistant_text", text: turn.heldText});
      }
      turn.heldText = "";
      turn.answer = answer;
      break;
    }
    case "turn.finished": {
      turn.status = String(data.status ?? "");
      turn.cause = String(data.cause ?? "run");
      turn.runningHere = false;
      turn.finishedAt = ts || turn.finishedAt;
      if (!turn.result && data.error) turn.error = String(data.error);
      for (const item of turn.items) {
        if (item.kind !== "tool") continue;
        if (item.status === "running" || item.status === "waiting_approval") {
          // Its approval, if one was pending, is withdrawn with the turn.
          if (item.pendingRequestId) view.pending.delete(item.pendingRequestId);
          item.status = "interrupted";
          item.pendingRequestId = null;
        }
      }
      break;
    }
    case "turn.undone": {
      turn.undone = true;
      break;
    }
    default:
      break;
  }
  return {};
}

// ── What a turn says ──

/** The tool cards of *turn*, excluding calls that never started. */
export function toolCount(turn) {
  return turn.items.filter((item) => item.kind === "tool" && item.status !== "not_run")
    .length;
}

/** The calls that failed: `error`, `timeout` or `crashed`. */
export function failedCount(turn) {
  return turn.items.filter(
    (item) => item.kind === "tool" && ["error", "timeout", "crashed"].includes(item.status),
  ).length;
}

/** The calls an approval denied. */
export function deniedCount(turn) {
  return turn.items.filter(
    (item) =>
      item.kind === "tool" &&
      item.status === "blocked" &&
      item.errorCategory === "approval_denied",
  ).length;
}

/** A run's own duration, else the turn's own. */
export function turnDuration(turn) {
  const runDuration = num(objectOf(turn.result).duration_s);
  if (runDuration !== null) return runDuration;
  if (turn.finishedAt) return Math.max(0, turn.finishedAt - turn.startedAt);
  return 0;
}

/**
 * The work group's summary line: what the turn did, and how it ended.
 *
 * `Working · 3 tool calls`, `Worked for 14s · 3 tool calls · 1 failed`,
 * `Thought for 1.2s`, `Stopped after 6s · 2 tool calls`. *held* is the
 * session's scheduler hold, when the gate is keeping a model call waiting.
 */
export function workSummary(turn, {held = ""} = {}) {
  const calls = toolCount(turn);
  const callWord = calls === 1 ? "1 tool call" : `${calls} tool calls`;
  if (!turn.finishedAt) {
    const activity = held ? `Waiting: ${held}` : "Working";
    return calls > 0 ? `${activity}${SEPARATOR}${callWord}` : activity;
  }
  const duration = formatDuration(turnDuration(turn));
  if (turn.status === "completed") {
    const parts = calls > 0 ? [`Worked for ${duration}`, callWord] : [`Thought for ${duration}`];
    const failed = failedCount(turn);
    const denied = deniedCount(turn);
    if (failed) parts.push(`${failed} failed`);
    if (denied) parts.push(`${denied} denied`);
    return parts.join(SEPARATOR);
  }
  const word = statusText(turn.status, turn.cause).word;
  const parts = [`${word} after ${duration}`];
  if (calls > 0) parts.push(callWord);
  return parts.join(SEPARATOR);
}

/**
 * The live row's activity, in the order that matters to a person.
 *
 * A held call comes before an approval, an approval before a stop, a stop
 * before a retry, and the model's own work last.
 */
export function activity(turn, {held = "", pending = 0, cancelling = false, now = 0} = {}) {
  if (held) return `Waiting: ${held}`;
  if (pending > 0) return "Waiting for your approval";
  if (cancelling) return "Stopping";
  if (turn.retryUntil > now) {
    return `Retrying in ${Math.ceil(turn.retryUntil - now)}s`;
  }
  const reflection = [...turn.items]
    .reverse()
    .find((item) => item.kind === "reflection" && item.status === "running");
  if (reflection !== undefined) return "Reflecting";
  const tool = [...turn.items]
    .reverse()
    .find((item) => item.kind === "tool" && item.status === "running");
  if (tool !== undefined) return `Running ${tool.name}`;
  if (turn.step > 0) return "Thinking";
  return "Starting";
}

/**
 * The live row's facts after the elapsed time: the `run.progress` wording.
 *
 * `step 4 · attempt 2 · Verify (cycle 1) · tests 3/4`, never `step n/max`
 * once `run.progress` has arrived; before the first one, the step of the
 * turn's own events (`step 2/30`).
 */
export function liveFacts(turn) {
  const parts = [];
  const progress = isObject(turn.progress) ? turn.progress : null;
  if (progress === null) {
    if (turn.step > 0) parts.push(`step ${turn.step}/${turn.maxSteps || 30}`);
    if (turn.attempt >= 2) parts.push(`attempt ${turn.attempt}`);
    return parts;
  }
  const step = int(progress.step) ?? 0;
  const attempt = int(progress.attempt) ?? 0;
  if (step > 0) parts.push(`step ${step}`);
  if (attempt >= 2) parts.push(`attempt ${attempt}`);
  if (progress.phase) {
    const cycle = int(progress.cycle) ?? 0;
    parts.push(cycle > 0 ? `${progress.phase} (cycle ${cycle})` : String(progress.phase));
  }
  const tests = objectOf(progress.tests);
  const passed = int(tests.passed);
  const total = int(tests.total);
  // `progress_line.py` shows a count only when both are there.
  if (passed !== null && total !== null) parts.push(`tests ${passed}/${total}`);
  return parts;
}

/**
 * A tool card's mark, state and short text, as the terminal shows them.
 *
 * `✓` with an `exit 1` warning, `✗ Failed`, `⊘ denied`, `◌ not run`,
 * `● Needs approval`; *held* turns a running call into `Waiting: <held>`. The
 * words beside a mark say what it says, so a colour is never the only signal.
 */
export function toolState(item, {held = "", now = 0} = {}) {
  const duration = item.durationS === null ? null : formatDuration(item.durationS);
  const tag = approvalTag(item.status, item.errorCategory, item.approval);
  switch (item.status) {
    case "running": {
      if (held) return {state: "held", mark: "●", text: `Waiting: ${held}`};
      let clock = item.startedTs ? toolElapsed(now - item.startedTs) : "0:00";
      if (LIMITED_TOOLS.has(item.name)) {
        clock += ` of ${toolElapsed(SESSION_TOOL_TIMEOUT_S)}`;
      }
      return {state: "running", mark: "●", text: `Running${SEPARATOR}${clock}`};
    }
    case "waiting_approval":
      return {state: "waiting_approval", mark: "●", text: "Needs approval"};
    case "success": {
      // The terminal's order: the time, then a non-zero exit, then the tag.
      const extra = [];
      if (duration !== null) extra.push(duration);
      if (item.exitCode !== null && item.exitCode !== 0) {
        extra.push(`exit ${item.exitCode}`);
      }
      if (tag) extra.push(tag.short);
      return {state: "success", mark: "✓", text: extra.join(SEPARATOR)};
    }
    case "error":
    case "timeout":
    case "crashed": {
      const word = {error: "Failed", timeout: "Timed out", crashed: "Crashed"}[item.status];
      return {
        state: item.status,
        mark: "✗",
        text: duration === null ? word : `${word}${SEPARATOR}${duration}`,
      };
    }
    case "blocked": {
      const first = cleanOutput(item.error).trim().split("\n")[0] ?? "";
      const text = tag ? tag.short : first ? `blocked: ${first}` : "blocked";
      return {state: "blocked", mark: "⊘", text};
    }
    case "cancelled": {
      if (item.approval?.origin === "withdrawn") {
        return {state: "cancelled", mark: "◌", text: "not run"};
      }
      return {
        state: "cancelled",
        mark: "⊘",
        text: duration === null ? "stopped" : `stopped after ${duration}`,
      };
    }
    case "interrupted":
      return {
        state: "interrupted",
        mark: "⊘",
        text: "Interrupted: may have partly run",
      };
    case "not_run":
    default:
      return {state: "not_run", mark: "◌", text: "not run"};
  }
}

/** `_first_line`: the first non-blank line of *text*, cleaned and collapsed. */
function firstLine(text) {
  for (const line of cleanOutput(String(text ?? "")).split("\n")) {
    if (line.trim()) return line.trim().split(/\s+/).join(" ");
  }
  return "";
}

/** A reflection's summary line, as `support/cli/render.py` words it. */
export function reflectionLine(item) {
  const attempt = item.attempt;
  if (item.status === "running") return `Reflecting after attempt ${attempt}…`;
  if (item.readable === false) {
    return item.continueRun
      ? `Reflection after attempt ${attempt} could not be read; the run continues.`
      : `Reflection after attempt ${attempt} could not be read, and the run's budget is used up.`;
  }
  // The terminal shows the rationale's first line after the action and the
  // duration in parentheses with a space, not another ` · ` part.
  let line = `Reflected after attempt ${attempt}: ${String(item.action ?? "")}`;
  const first = firstLine(item.rationale);
  if (first) line += `${SEPARATOR}${first}`;
  if (item.durationS !== null) line += ` (${formatDuration(item.durationS)})`;
  return line;
}

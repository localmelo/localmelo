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

import { blockWords } from "./blocked.mjs";
import { t } from "./i18n.mjs";
import { readMemory } from "./memory.mjs";

const SEPARATOR = " · ";
// `support/cli/render.py`: these two calls end after SESSION_TOOL_TIMEOUT_S.
const LIMITED_TOOLS = new Set(["shell_exec", "python_exec"]);
const SESSION_TOOL_TIMEOUT_S = 600;
const ITEM_MARKS = {done: "✓", current: "◐", skipped: "–", pending: "○"};

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

/** The `turn.pictures` event's list, kept to the entries the page can use. */
export function readPictures(list) {
  if (!Array.isArray(list)) return [];
  return list
    .filter((item) => isObject(item) && typeof item.path === "string" && item.path !== "")
    .map((item) => ({
      path: item.path,
      size: num(item.size) ?? 0,
      mtime: num(item.mtime) ?? 0,
      state: typeof item.state === "string" ? item.state : "",
    }));
}

// ── Text mirrors (support/sessions/display.py, support/progress_line.py) ──

/**
 * `split_thinking`: the reasoning in a `<think>` block, and the text. A closed `<thinking>...</thinking>`
 * block that starts the text (after whitespace, any letter case) is the thinking first; a later or an
 * unclosed one stays text. `tests/fixtures/display_vectors.json` holds the cases both sides must read alike.
 */
export function splitThinking(text) {
  const value = String(text ?? "");
  const spelled = /^\s*<thinking>(.*?)<\/thinking>/is.exec(value);
  if (spelled !== null) return [spelled[1].trim(), value.slice(spelled[0].length).trim()];
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
  if (value < 1_000) return t("turns.bytes", {n: value});
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

const stepsWord = (count) => t("turns.steps", {n: count});
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
  return text ? t("turns.past.like", {title: text}) : t("turns.past.likeSimilar");
}

/** `past_facts`: (steps text, times text, a single past run). */
export function pastFacts(past, level, hint = null) {
  if (!isObject(past)) return [null, null, false];
  const runs = int(past.runs) ?? 0;
  if (level >= 2) {
    const lo = num(past.wall_lo_s);
    const hi = num(past.wall_hi_s);
    return [null, lo === null || hi === null ? null : t("turns.past.runs", {range: wallRange(lo, hi)}), false];
  }
  if (runs <= 1) {
    const steps = int(past.last_steps) ?? int(past.steps_hi);
    const wall = num(past.last_wall_s) ?? num(past.wall_hi_s);
    if (steps === null) return [null, null, false];
    return [
      `${hint || t("turns.past.lastTime")} ${stepsWord(steps)}`,
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
    `${hint || t("turns.past.past")} ${t("turns.steps", {n: stepRange(loSteps, hiSteps)})}`,
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
  if (level === 1) itemsTitle = similar ? (pastSteps ? t("turns.past.itsSteps") : hint) : t("turns.past.lastTimeSteps");
  const merge = single && level === 1;

  const compose = (dropped) => {
    const past = pastSteps || "";
    const withTimes =
      pastTimes && !dropped.has("past_times")
        ? past
          ? t("turns.past.withTimes", {steps: past, times: pastTimes})
          : pastTimes
        : past;
    const shown = dropped.has("items") ? null : items;
    if (shown && merge && withTimes) return t("turns.past.merged", {facts: withTimes, steps: shown});
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

// status -> [the catalog's `turns.status.<name>` words, severity]; `(status, cause)` rows win.
const STATUS_TEXT = {
  completed: ["done", "ok"],
  cancelled: ["stopped", "stopped"],
  failed: ["failed", "warn"],
  stuck: ["stuck", "warn"],
  gate_stopped: ["reflected", "warn"],
  budget_exhausted: ["outOfSteps", "warn"],
  context_budget_exceeded: ["context", "warn"],
  timeout: ["timeout", "warn"],
  provider_error: ["provider", "error"],
  setup_error: ["setup", "error"],
  crashed: ["crashed", "error"],
  "crashed|run": ["crashed", "error"],
  "crashed|crash": ["interrupted", "warn"],
  "cancelled|user_stop": ["stoppedByYou", "stopped"],
  "cancelled|crash": ["interrupted", "warn"],
  "cancelled|shutdown": ["shutdown", "warn"],
  "cancelled|closed": ["closed", "stopped"],
  "cancelled|takeover": ["takeover", "stopped"],
  "cancelled|park": ["parked", "stopped"],
};

/** A row of STATUS_TEXT in words: `{word, severity, title}`. */
function statusWords([name, severity]) {
  return {word: t(`turns.status.${name}.word`), severity, title: t(`turns.status.${name}.title`)};
}

// How a provider's error starts when it could not connect at all
// (`UNREACHABLE_PREFIX` in `support/providers/_http.py`; a test keeps the two equal).
const UNREACHABLE_PREFIX = "cannot reach the model server at ";

/** Whether a run ended because the model server could not be reached, not because it answered with an error. */
export function serverUnreachable(status, error) {
  return status === "provider_error" && String(error ?? "").startsWith(UNREACHABLE_PREFIX);
}

/** The address that error names, as the provider wrote it, or "". */
export function unreachableUrl(error) {
  return new RegExp(`^${UNREACHABLE_PREFIX}(\\S+)\\. `).exec(String(error ?? ""))?.[1] ?? "";
}

/** `127.0.0.1:8000` for `http://127.0.0.1:8000/v1`; text that is not an absolute URL comes back as it is. */
export function serverAddress(url) {
  try {
    return new URL(url).host || String(url);
  } catch {
    return String(url ?? "");
  }
}

// The error of a run whose replies stayed unfinished thinking (`UNFINISHED_ANSWER_ERROR` in `melo/schema.py`).
export const UNFINISHED_ANSWER = "melo did not finish an answer";
// The error of a run whose replies wrote a tool call as text (`TOOL_CALL_TEXT_ERROR` in `melo/schema.py`).
export const TOOL_CALL_TEXT = "tool call written as text";

/** `status_text`: the word, severity and title of a run's end. `error` tells an unreachable server apart. */
export function statusText(status, cause = "", error = "") {
  if (serverUnreachable(status, error)) {
    return {...statusWords(STATUS_TEXT.provider_error), title: t("turns.status.unreachable")};
  }
  if (status === "failed" && error === UNFINISHED_ANSWER) {
    return {...statusWords(STATUS_TEXT.failed), title: t("turns.status.unfinished")};
  }
  if (status === "failed" && error === TOOL_CALL_TEXT) {
    return {...statusWords(STATUS_TEXT.failed), title: t("turns.status.toolText")};
  }
  const found =
    STATUS_TEXT[`${status}|${cause}`] ?? STATUS_TEXT[status] ?? null;
  if (found !== null) return statusWords(found);
  const unknown = status || t("turns.status.unknown");
  return {word: unknown, severity: "warn", title: unknown};
}

/**
 * Which of Melo's faces a reply carries: `idle` on every reply but the newest, and on the newest its turn's state
 * as a still picture: `working` while it runs, `waiting` while an approval is open, then `done`, or `error` for any
 * end the console words as a warning or an error. A turn the person stopped is not a failure: `idle`.
 */
export function turnFace(turn, {newest = false, pending = 0} = {}) {
  if (!newest) return "idle";
  if (!turn.finishedAt) return pending > 0 ? "waiting" : "working";
  const {severity} = statusText(turn.status, turn.cause, turn.error);
  return severity === "ok" ? "done" : severity === "stopped" ? "idle" : "error";
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

/**
 * The one next step a run's end offers first: `settings`, `tools`, `continue`, `retry`, `copy` or null.
 * An unreachable server is fixed by starting it, not by a setting: the step is to pick the work up
 * again (`continue`, which the card turns into `retry` when no tool ran).
 */
export function nextAction(status, cause = "", error = "") {
  if (serverUnreachable(status, error)) return "continue";
  const own = (key) => Object.hasOwn(NEXT_ACTION, key);
  const keyed = `${status}|${cause}`;
  return own(keyed) ? NEXT_ACTION[keyed] : own(status) ? NEXT_ACTION[status] : null;
}

/**
 * The items a work box lists. A run that ended on a model-server error leaves out its retry
 * notices: the failure card already says it, and says it once.
 */
export function shownItems(turn) {
  if (!turn.finishedAt || turn.status !== "provider_error") return turn.items;
  return turn.items.filter((item) => !(item.kind === "notice" && item.noticeKind === "retry"));
}

/**
 * `status_line`: one bounded line, without the user's task text. The frame reads in the page's language;
 * the status code and the error's own words stay as they are.
 */
export function statusLine(status, error) {
  let reason = String(error ?? "").split(/\s+/).filter(Boolean).join(" ");
  const characters = Array.from(reason);
  if (characters.length > 500) reason = characters.slice(0, 497).join("") + "...";
  return t("failure.copyLine", {status, reason: reason || t("failure.noMessage")});
}

/** Continue keeps tool side effects and is only safe to offer on the newest turn. */
export function canContinue(turn, newest) {
  return newest && !turn.origin && !!turn.finishedAt && !turn.undone && Number(turn.result?.n_tool) > 0 &&
    ["budget_exhausted", "context_budget_exceeded", "gate_stopped", "provider_error", "timeout", "cancelled"].includes(turn.status);
}

/** A retry undoes the newest finished turn, never an earlier or undone one, nor a task's question. */
export function canRetry(turn, newest) {
  return newest && !turn.origin && !turn.undone && !!turn.finishedAt && turn.status !== "completed";
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
    return {short: t("turns.approval.withdrawn.short"), long: t("turns.approval.withdrawn.long")};
  }
  if (origin === "error") {
    return {short: t("turns.approval.error.short"), long: t("turns.approval.error.long")};
  }
  if (origin === "timeout") {
    return {short: t("turns.approval.timeout.short"), long: t("turns.approval.timeout.long")};
  }
  // Auto mode (#339): a block says why in a few words; a routine call has no tag.
  if (origin === "auto" && !approval.allowed) {
    const words = blockWords(approval);
    return words
      ? {short: t("turns.approval.blocked.shortReason", {reason: words}), long: t("turns.approval.blocked.longReason", {reason: words})}
      : {short: t("turns.approval.blocked.short"), long: t("turns.approval.blocked.long")};
  }
  if (approval.allowed && approval.allowance === "once") {
    return {short: t("turns.approval.allowedOnce"), long: t("turns.approval.once.long")};
  }
  if (origin === "auto") {
    const asked = isObject(approval.review) && Array.isArray(approval.review.categories) && approval.review.categories.length > 0;
    return asked ? {short: t("turns.approval.asked.short"), long: t("turns.approval.asked.long")} : null;
  }
  if (approval.allowed) {
    // A folder outside the workspace (#405) that a grant or the settings let melo read.
    const scope = isObject(approval.folder) ? approval.folder.scope : "";
    if (scope === "always") {
      return {short: t("turns.approval.folderAlways.short"), long: t("turns.approval.folderAlways.long")};
    }
    if (scope === "chat" && origin === "session") {
      return {short: t("turns.approval.folderChat.short"), long: t("turns.approval.folderChat.long")};
    }
    if (origin === "session") {
      return {short: t("turns.approval.grant.short"), long: t("turns.approval.grant.long")};
    }
    // The person's earlier approval of this exact step, found by code in the task's chat (#369).
    if (origin === "earlier") {
      return {short: t("turns.approval.earlier.short"), long: t("turns.approval.earlier.long")};
    }
    if (origin === "policy") {
      return {short: t("turns.approval.auto.short"), long: t("turns.approval.auto.long")};
    }
    if (approval.remember) {
      return {short: t("turns.approval.session.short"), long: t("turns.approval.session.long")};
    }
    return {short: t("turns.approval.once.short"), long: t("turns.approval.once.long")};
  }
  // A step blocked earlier in a task its chat continued on its own does not run again (#411).
  if (origin === "policy" && approval.earlier) {
    return {short: t("turns.approval.earlier.blockedShort"), long: t("turns.approval.earlier.blockedLong")};
  }
  if (origin === "policy") {
    return reason
      ? {short: t("turns.approval.policy.shortReason", {reason}), long: t("turns.approval.policy.longReason", {reason})}
      : {short: t("turns.approval.policy.short"), long: t("turns.approval.policy.long")};
  }
  if (origin === "user" && (status === "blocked" || errorCategory === "approval_denied")) {
    return reason
      ? {short: t("turns.approval.denied.shortReason", {reason}), long: t("turns.approval.denied.longReason", {reason})}
      : {short: t("turns.approval.denied.short"), long: t("turns.approval.denied.long")};
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
    // "ask_owner": the chat's internal turn on a task's question (#369); "" the person's.
    origin: String(source.origin ?? ""),
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
    // The picture files the turn made or changed, or its answer names (`turn.pictures`).
    pictures: [],
    picturesOpen: false,
    // What memory across chats recalled and kept for the message (`turn.memory`), or null.
    memory: null,
    memoryOpen: false,
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
      origin: fields.origin ?? "",
      status: "running",
      running_here: true,
    });
    addTurn(view, turn);
  }
  if (fields.id && !turn.id) turn.id = fields.id;
  if (fields.origin && !turn.origin) turn.origin = fields.origin;
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
  if (type === "note") {
    return data.kind ? t("turns.fact.noteKind", {kind: data.kind, text: clipped(data.text)}) : t("turns.fact.note", {text: clipped(data.text)});
  }
  if (type === "inbox.accepted") return t("turns.fact.accepted", {text: clipped(data.text)});
  if (type === "memory") {
    const action = data.action ?? "decision";
    return data.kind ? t("turns.fact.memoryKind", {action, kind: data.kind}) : t("turns.fact.memory", {action});
  }
  const hits = Array.isArray(data.hits) ? data.hits.length : 0;
  return data.trigger ? t("turns.fact.recalledWhy", {n: hits, trigger: data.trigger}) : t("turns.fact.recalled", {n: hits});
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
      origin: data.origin,
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
        turn.items.unshift({kind: "notice", noticeKind: "yolo", step: 0, message: t("turns.yolo")});
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
      // A tool call written as text did not run and is no answer: the notice says so in its place.
      if (data.kind === "tool_call_as_text") turn.heldText = "";
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
    case "turn.pictures": {
      turn.pictures = readPictures(data.pictures);
      break;
    }
    case "turn.memory": {
      turn.memory = readMemory(data);
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

/** The scheduler's hold for a call that waits on a foreign GPU lock. */
export const LOCK_HOLD = "Reservation";

/** What a held call says: the GPU lock in plain words, as the console's banner does; any other hold keeps its code. */
export function holdText(held) {
  return held === LOCK_HOLD ? t("turns.hold.gpu") : t("turns.hold.other", {held});
}

/**
 * The work group's summary line: what the turn did, and how it ended.
 *
 * `Working · 3 tool calls`, `Worked for 14s · 3 tool calls · 1 failed`,
 * `Thought`, `Stopped after 6s · 2 tool calls`. *held* is the session's
 * scheduler hold, when the gate is keeping a model call waiting.
 *
 * A finished chat turn without tool calls has no line of its own, because the
 * footer under its answer already gives the turn's time: it reads `Thought`
 * when it has reasoning to open (the events carry the step's time, not the
 * reasoning's own, so no time is shown) and is empty when it has none.
 */
export function workSummary(turn, {held = ""} = {}) {
  const calls = toolCount(turn);
  const callWord = t("turns.summary.calls", {n: calls});
  if (!turn.finishedAt) {
    const activity = held ? holdText(held) : t("turns.summary.working");
    return calls > 0 ? `${activity}${SEPARATOR}${callWord}` : activity;
  }
  if (turn.status === "completed" && calls === 0) {
    return turn.items.some((item) => item.kind === "thinking") ? t("turns.summary.thought") : "";
  }
  const duration = formatDuration(turnDuration(turn));
  if (turn.status === "completed") {
    const parts = [t("turns.summary.workedFor", {duration}), callWord];
    const failed = failedCount(turn);
    const denied = deniedCount(turn);
    if (failed) parts.push(t("turns.summary.failed", {n: failed}));
    if (denied) parts.push(t("turns.summary.denied", {n: denied}));
    return parts.join(SEPARATOR);
  }
  const word = statusText(turn.status, turn.cause).word;
  const parts = [t("turns.summary.endedAfter", {word, duration})];
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
  if (held) return holdText(held);
  if (pending > 0) return t("turns.activity.approval");
  if (cancelling) return t("turns.activity.stopping");
  if (turn.retryUntil > now) {
    return t("turns.activity.retrying", {n: Math.ceil(turn.retryUntil - now)});
  }
  const reflection = [...turn.items]
    .reverse()
    .find((item) => item.kind === "reflection" && item.status === "running");
  if (reflection !== undefined) return t("turns.activity.reflecting");
  const tool = [...turn.items]
    .reverse()
    .find((item) => item.kind === "tool" && item.status === "running");
  if (tool !== undefined) return t("turns.activity.running", {tool: tool.name});
  if (turn.step > 0) return t("turns.activity.thinking");
  return t("turns.activity.starting");
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
    if (turn.step > 0) parts.push(t("turns.facts.stepOf", {step: turn.step, max: turn.maxSteps || 30}));
    if (turn.attempt >= 2) parts.push(t("turns.facts.attempt", {n: turn.attempt}));
    return parts;
  }
  const step = int(progress.step) ?? 0;
  const attempt = int(progress.attempt) ?? 0;
  if (step > 0) parts.push(t("turns.facts.step", {step}));
  if (attempt >= 2) parts.push(t("turns.facts.attempt", {n: attempt}));
  if (progress.phase) {
    const cycle = int(progress.cycle) ?? 0;
    parts.push(cycle > 0 ? t("turns.facts.cycle", {phase: progress.phase, n: cycle}) : String(progress.phase));
  }
  const tests = objectOf(progress.tests);
  const passed = int(tests.passed);
  const total = int(tests.total);
  // `progress_line.py` shows a count only when both are there.
  if (passed !== null && total !== null) parts.push(t("turns.facts.tests", {passed, total}));
  return parts;
}

/**
 * A tool card's mark, state and short text, as the terminal shows them.
 *
 * `✓` with an `exit 1` warning, `✗ Failed`, `⊘ denied`, `◌ not run`,
 * `● Needs approval`; *held* turns a running call into `holdText(held)`. The
 * words beside a mark say what it says, so a colour is never the only signal.
 */
export function toolState(item, {held = "", now = 0} = {}) {
  const duration = item.durationS === null ? null : formatDuration(item.durationS);
  const tag = approvalTag(item.status, item.errorCategory, item.approval);
  switch (item.status) {
    case "running": {
      if (held) return {state: "held", mark: "●", text: holdText(held)};
      let clock = item.startedTs ? toolElapsed(now - item.startedTs) : "0:00";
      if (LIMITED_TOOLS.has(item.name)) {
        clock = t("turns.tool.clockOf", {clock, limit: toolElapsed(SESSION_TOOL_TIMEOUT_S)});
      }
      return {state: "running", mark: "●", text: `${t("turns.tool.running")}${SEPARATOR}${clock}`};
    }
    case "waiting_approval":
      return {state: "waiting_approval", mark: "●", text: t("turns.tool.needsApproval")};
    case "success": {
      // The terminal's order: the time, then a non-zero exit, then the tag.
      const extra = [];
      if (duration !== null) extra.push(duration);
      if (item.exitCode !== null && item.exitCode !== 0) {
        extra.push(t("turns.tool.exit", {code: item.exitCode}));
      }
      if (tag) extra.push(tag.short);
      return {state: "success", mark: "✓", text: extra.join(SEPARATOR)};
    }
    case "error":
    case "timeout":
    case "crashed": {
      const word = t(`turns.tool.${item.status}`);
      return {
        state: item.status,
        mark: "✗",
        text: duration === null ? word : `${word}${SEPARATOR}${duration}`,
      };
    }
    case "blocked": {
      const first = cleanOutput(item.error).trim().split("\n")[0] ?? "";
      const text = tag ? tag.short : first ? t("turns.tool.blockedWhy", {reason: first}) : t("turns.tool.blocked");
      return {state: "blocked", mark: "⊘", text};
    }
    case "cancelled": {
      if (item.approval?.origin === "withdrawn") {
        return {state: "cancelled", mark: "◌", text: t("turns.tool.notRun")};
      }
      return {
        state: "cancelled",
        mark: "⊘",
        text: duration === null ? t("turns.tool.stopped") : t("turns.tool.stoppedAfter", {duration}),
      };
    }
    case "interrupted":
      return {
        state: "interrupted",
        mark: "⊘",
        text: t("turns.tool.interrupted"),
      };
    case "not_run":
    default:
      return {state: "not_run", mark: "◌", text: t("turns.tool.notRun")};
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
  if (item.status === "running") return t("turns.reflection.running", {n: attempt});
  if (item.readable === false) {
    return item.continueRun
      ? t("turns.reflection.unreadable", {n: attempt})
      : t("turns.reflection.unreadableEnd", {n: attempt});
  }
  // The terminal shows the rationale's first line after the action and the
  // duration in parentheses with a space, not another ` · ` part.
  let line = t("turns.reflection.done", {n: attempt, action: String(item.action ?? "")});
  const first = firstLine(item.rationale);
  if (first) line += `${SEPARATOR}${first}`;
  if (item.durationS !== null) line += ` (${formatDuration(item.durationS)})`;
  return line;
}

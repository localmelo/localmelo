/**
 * The main chat and its side chats: the words and nodes `workspace.js` draws.
 *
 * The gateway keeps one main chat (`kind: "front"`, `main: true`). Every other console chat is one of
 * its side chats. When a side chat's run ends, the main chat gets a card (an `attention.card` event,
 * never a turn); when a task finishes, fails or waits for an approval, the chat that started it (the
 * main chat or a side chat) does. What a chat did with a task (started, paused, a suggestion, "Start
 * this as a task?") is an echo line (an `attention.echo` event) in that chat; so is "While you were away,
 * melo tidied 3 notes" with its Undo, in the main chat.
 * Nothing here fetches; the page passes what a card's Open and an echo's buttons do.
 */

import {ENDING_MARKS, canFinish, endingName, endingWhy, finishLabel, readEnding} from "./endings.mjs";
import {blockCard, blockFacts, stepLine} from "./blocked.mjs";
import {formatTime, saidText, t} from "./i18n.mjs";
import {LEVELS, levelCode, levelLabel, levelMeaning} from "./levels.mjs";

const DAY_S = 86400;

function startOfDay(ts) {
  const day = new Date(ts * 1000);
  day.setHours(0, 0, 0, 0);
  return day.getTime() / 1000;
}

const daysAgo = (ts, now) => Math.round((startOfDay(now) - startOfDay(ts)) / DAY_S);

/** "Today", "Yesterday", "Oct 28", or "Oct 28, 2025" for another year (in Chinese, 10月28日 and 2025年10月28日). */
export function dayLabel(ts, now) {
  const days = daysAgo(ts, now);
  if (days <= 0) return t("time.today");
  if (days === 1) return t("time.yesterday");
  const day = new Date(ts * 1000);
  return formatTime(day, day.getFullYear() === new Date(now * 1000).getFullYear() ? "day" : "dayYear");
}

/** "7:35 PM" ("下午 7:35"), as the page's language writes a time (the panel's rows write it the same way). */
export function clock(ts) {
  return formatTime(new Date(ts * 1000), "clock");
}

/** A time beside a row: the clock today, else the day. */
export function shortTime(ts, now) {
  if (!ts) return "";
  return daysAgo(ts, now) <= 0 ? clock(ts) : dayLabel(ts, now);
}

export function isMain(session) {
  return session?.main === true || session?.kind === "front";
}

/** A card as the page keeps it, from a `cards` row or an `attention.card` frame; null if unusable. */
export function readCard(row) {
  const data = row?.data && typeof row.data === "object" ? {...row.data, id: row.id, ts: row.ts} : row;
  const id = Number(data?.id);
  if (!Number.isFinite(id) || !data?.source_id) return null;
  const state = ["done", "failed", "stopped", "waiting", "answered", "blocked"].includes(data.state) ? data.state : "done";
  return {
    id,
    at: Number(data.at) || Number(data.ts) || 0,
    state,
    sourceId: String(data.source_id),
    sourceKind: data.source_kind === "task" ? "task" : "chat",
    title: String(data.title || t(data.source_kind === "task" ? "mainChat.aTask" : "mainChat.aSideChat")),
    // A card the gateway worded keeps its words: it reads in the page's language, whoever stored it.
    line: saidText(data.said) ?? String(data.line ?? ""),
    // How an approval was answered (`allowedOnce`, `denied`, …), when the card kept its words.
    ...(String(data.said?.key ?? "").startsWith("cards.answer.") ? {answer: data.said.key.slice(13)} : {}),
    // A waiting card and the answer that follows it share this.
    ...(data.request_id ? {requestId: String(data.request_id)} : {}),
    // The card of a reminder's run: its heading is "Reminder: <the person's words>".
    ...(data.reminder === true ? {reminder: true} : {}),
    // A task's card: how it ended, as code decided it.
    ...(readEnding(data.ending) ? {ending: readEnding(data.ending)} : {}),
    // A step Auto blocked in a task (#369): the block as the task's page has it, with Allow once.
    ...(state === "blocked" ? {
      stepKey: String(data.step_key ?? ""),
      name: String(data.name ?? ""),
      preview: String(data.preview ?? ""),
      review: data.review && typeof data.review === "object" ? data.review : {},
      canAllow: data.can_allow === true,
    } : {}),
  };
}

/** What a blocked card's block says (`blocked.mjs`), or null; Allow once only where the server offers it. */
export function cardBlock(card) {
  if (card?.state !== "blocked") return null;
  const facts = blockFacts({
    approval: {origin: "auto", allowed: false, request_id: card.requestId, step_key: card.stepKey, review: card.review},
    name: card.name,
    preview: card.preview,
  });
  if (!facts || !facts.category) return null;
  return card.canAllow ? facts : {...facts, allow: false};
}

/**
 * *cards* as the person reads them: a waiting card whose approval was answered becomes an answered
 * card in its place, with the answer in words, and the answer's own row goes. An answer whose
 * waiting card is not on the page (yet) shows nothing.
 */
export function foldAnswers(cards) {
  const answers = new Map();
  for (const card of cards) {
    if (card.state === "answered" && card.requestId) answers.set(card.requestId, card);
  }
  return cards
    .filter((card) => card.state !== "answered")
    .map((card) => {
      const answer = card.state === "waiting" && card.requestId ? answers.get(card.requestId) : null;
      return answer ? {...card, state: "answered", line: answer.line, answer: answer.answer} : card;
    });
}

/** *cards* with *more* added once each, oldest first. */
export function mergeCards(cards, more) {
  const byId = new Map(cards.map((card) => [card.id, card]));
  for (const row of more) {
    const card = readCard(row);
    if (card) byId.set(card.id, card);
  }
  return [...byId.values()].sort((left, right) => left.id - right.id);
}

/**
 * The main chat's conversation: its turns, cards and echoes by time, with a divider at each new day.
 * Between two turns, a source shows only its newest card, so a busy side chat is one card; every
 * echo stays, and so does every blocked step (each has its own Allow once). Items: `{kind: "day", label}`, `{kind: "turn", turn}`, `{kind: "card", card}`,
 * `{kind: "echo", echo}`.
 */
export function timeline(turns, cards, now, echoes = []) {
  const at = (item) => (item.kind === "turn" ? Number(item.turn.startedAt) || 0 : item[item.kind].at);
  const items = [
    ...turns.map((turn) => ({kind: "turn", turn})),
    ...foldAnswers(cards).map((card) => ({kind: "card", card})),
    ...echoes.map((echo) => ({kind: "echo", echo})),
  ].sort((left, right) => at(left) - at(right));
  const kept = [];
  let since = new Map();
  for (const item of items) {
    if (item.kind !== "card" || item.card.state === "blocked") {
      if (item.kind === "turn") since = new Map();
      kept.push(item);
      continue;
    }
    const earlier = since.get(item.card.sourceId);
    if (earlier !== undefined) kept[earlier] = null;
    since.set(item.card.sourceId, kept.length);
    kept.push(item);
  }
  const out = [];
  let day = "";
  for (const item of kept) {
    if (item === null) continue;
    const label = at(item) ? dayLabel(at(item), now) : "";
    if (label && label !== day) out.push({kind: "day", label});
    day = label || day;
    out.push(item);
  }
  return out;
}

/** Side chats newest first, under a heading for each day. */
export function byDay(sessions, now) {
  const groups = [];
  for (const session of [...sessions].sort((left, right) => (right.updated_at || 0) - (left.updated_at || 0))) {
    const label = dayLabel(session.updated_at || now, now);
    if (groups.at(-1)?.label !== label) groups.push({label, sessions: []});
    groups.at(-1).sessions.push(session);
  }
  return groups;
}

/** A card's words: its mark, its heading and its action. */
export function cardWords(card) {
  const open = t(card.sourceKind === "task" ? "mainChat.openTask" : "mainChat.openSideChat");
  const title = card.title;
  if (card.state === "waiting") return {mark: "!", heading: t("mainChat.card.waiting", {title}), action: t("common.review")};
  if (card.state === "blocked") return {mark: "⊘", heading: t("mainChat.card.blocked", {title}), action: open};
  // An answered approval has nothing left to review: it says how it was answered (cards stored before
  // they kept their words say it in English).
  if (card.state === "answered") {
    const allowed = card.answer ? card.answer.startsWith("allowed") : card.line.startsWith("Allowed");
    const denied = card.answer ? card.answer === "denied" : card.line.startsWith("Denied");
    return {mark: allowed ? "✓" : denied ? "✕" : "·", heading: t("mainChat.card.answered", {title}), action: open};
  }
  if (card.reminder) return {mark: card.state === "failed" || card.state === "stopped" ? "✕" : "✓", heading: t("mainChat.card.reminder", {title}), action: open};
  // A done task says how code found it ended: ready, done but not finished (Finish it), or needs you.
  const ending = card.state === "done" ? card.ending : null;
  if (ending) {
    const name = endingName(ending);
    // Its own keys: "blocked" cards are steps Auto blocked, a dropped task reads as stopped.
    const key = {blocked: "needsYou", dropped: "stopped"}[name] ?? name;
    const heading = t(`mainChat.card.${key}`, {title});
    return {mark: ENDING_MARKS[ending.state], heading, action: open, ...(canFinish(ending) ? {more: finishLabel(ending)} : {})};
  }
  if (card.state === "failed") return {mark: "✕", heading: t("mainChat.card.failed", {title}), action: open};
  // A task that ran out of steps stopped; it can go on from where it was, so Continue comes first.
  if (card.state === "stopped") return {mark: "■", heading: t("mainChat.card.stopped", {title}), action: open, more: t("failure.continue")};
  return {mark: "✓", heading: t("mainChat.card.done", {title}), action: open};
}

/**
 * One card's node: `{doc, card, open, resume, approval, block}`; `open` gets the card, and
 * `resume(card, button)` is the Continue of a stopped task's card. `approval` is a waiting task's
 * approval card, drawn inside it (its Open then only opens the task). `block` is what a blocked
 * card's Allow once needs (`blockCard`'s options: `allowing`, `busy`, `onAllow`, `error`,
 * `developer`). Its day is the divider above it.
 */
export function cardNode({doc, card, open, resume = null, approval = null, block = null}) {
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const words = cardWords(card);
  const node = el("article", "main-card");
  node.dataset.state = card.state;
  if (card.ending) node.dataset.ending = card.ending.state;
  if (card.state === "answered") node.dataset.answer = words.mark === "✓" ? "allowed" : words.mark === "✕" ? "denied" : "other";
  node.dataset.cardId = String(card.id);
  node.setAttribute("aria-label", words.heading);
  const mark = el("span", "main-card-mark", words.mark);
  mark.setAttribute("aria-hidden", "true");
  const text = el("div", "main-card-text");
  text.append(el("p", "main-card-title", words.heading));
  // Cards written before a card said what it asks kept a line that repeats the heading (in English, when
  // they were stored before cards kept their words).
  const repeats = [t("turns.activity.approval"), "Waiting for your approval"].includes(card.line);
  const facts = cardBlock(card);
  // A blocked step says what and why in its block; its line is for the list and the terminal.
  const line = (card.state === "waiting" && repeats) || facts ? "" : card.line;
  if (line) text.append(el("p", "main-card-line", line));
  // Why it is not ready: what code found missing, or its question.
  const why = card.ending && card.state !== "waiting" ? endingWhy(card.ending) : "";
  if (why) text.append(el("p", "main-card-why", why));
  if (facts) text.append(blockCard(doc, facts, block ?? {}));
  const inline = card.state === "waiting" && approval;
  if (inline) text.append(approval);
  const action = el("button", card.state === "waiting" && !inline ? "outline-button" : "link-button", inline ? t("mainChat.openTask") : words.action);
  action.type = "button";
  // The panel returns focus here when a task opened from this card is closed.
  action.dataset.focusKey = `card:${card.id}`;
  action.addEventListener("click", () => open(card));
  const foot = el("div", "main-card-foot");
  if (words.more && resume) {
    const more = el("button", "outline-button", words.more);
    more.type = "button";
    more.addEventListener("click", () => resume(card, more));
    foot.append(more);
  }
  foot.append(action, el("span", "main-card-time", card.at ? clock(card.at) : ""));
  node.append(mark, text, foot);
  return node;
}

// ── Echoes: what the main chat did with a task ──

const ECHO_KINDS = new Set(["started", "control", "level", "suggest", "ask", "undone", "failed", "sent", "chore", "chore_saved", "checkin", "sleep",
  "task_ask", "task_answered", "task_escalate", "ask_end", "queued", "dequeued", "answer_sent", "answer_delivered", "earlier", "revoked"]);
const MARKS = {started: "+", level: "↕", suggest: "?", ask: "?", undone: "↺", failed: "!", sent: "→", chore: "◷", chore_saved: "◷", checkin: "◷", sleep: "☾",
  stop: "■", pause: "‖", resume: "▶", first: "↑", send: "→", task_ask: "?", task_answered: "↩", task_escalate: "!", ask_end: "·", queued: "…", answer_sent: "↩",
  earlier: "✓", revoked: "↺"};
// A task's question (#369): its seven fields, and the five parts of a question to the person.
const ASK_FIELDS = ["background", "situation", "problem", "likely_causes", "options", "help_needed", "if_unsolvable"];
const PERSON_PARTS = ["background", "what_happened", "tried", "choices", "question"];
const camel = (name) => name.replace(/_(\w)/g, (_, letter) => letter.toUpperCase());
// What the person typed in a reply box, by row: a redraw of the chat keeps it.
const replyDrafts = new Map();

/** Whether *turn* is the chat's internal turn on a task's question: its row stands in for the person's bubble. */
export function askTurn(turn, echoes = []) {
  if (turn?.origin === "ask_owner") return true;
  const text = String(turn?.userText ?? "");
  return echoes.some((echo) => echo.kind === "task_ask" && echo.lead && text.startsWith(echo.lead));
}

/** An echo as the page keeps it, from an `echoes` row or an `attention.echo` frame; null if unusable. */
export function readEcho(row) {
  const data = row?.data && typeof row.data === "object" ? {...row.data, id: row.id, ts: row.ts} : row;
  const id = Number(data?.id);
  if (!Number.isFinite(id) || !ECHO_KINDS.has(data?.echo)) return null;
  return {...data, id, at: Number(data.ts) || 0, kind: data.echo, line: String(data.line ?? "")};
}

/** *echoes* with *more* added once each, oldest first. */
export function mergeEchoes(echoes, more) {
  const byId = new Map(echoes.map((echo) => [echo.id, echo]));
  for (const row of more) {
    const echo = readEcho(row);
    if (echo) byId.set(echo.id, echo);
  }
  return [...byId.values()].sort((left, right) => left.id - right.id);
}

/** The ids of echoes a later one answered (an Undo, an Apply, a Start, a Revoke): their buttons go. */
export function answered(echoes) {
  return new Set(echoes.map((echo) => Number(echo.of)).filter(Number.isFinite));
}

/**
 * The newest "melo tidied" line of *echoes* that no later echo answered, or undefined: `/memory undo`
 * taps it. A line the gateway answered, even with "already put back", is passed over for an older one.
 */
export function unansweredSleep(echoes) {
  const done = answered(echoes);
  return echoes.filter((echo) => echo.kind === "sleep" && !done.has(echo.id)).at(-1);
}

/**
 * Whether *echo*'s buttons go: a later echo answered it (ids in *done*), or its task ended (ids in
 * *ended*). A step that ran on the person's earlier approval keeps Revoke after its task ends: the
 * approval is the chat's, and later tasks could use it.
 */
export function echoDone(echo, done, ended) {
  return done.has(echo.id) || (echo.kind !== "earlier" && ended.has(echo.task_id));
}

/** An echo's buttons: `[label, do]`; a level change opens the level list. */
export function echoActions(echo, done) {
  if (done || !echo.task_id && !["ask", "chore", "chore_saved", "sleep"].includes(echo.kind)) return [];
  const undo = [t("mainChat.echo.undo"), "undo"];
  // A step ran on the person's earlier approval (#369): Revoke takes the approval back.
  if (echo.kind === "earlier") return [[t("approvals.revoke"), "revoke"]];
  // A reminder runs once, at its time: there is no "just once" for it.
  if (echo.kind === "chore" && echo.reminder) return [[t("mainChat.echo.remind"), "save"], [t("common.cancel"), "cancel"]];
  if (echo.kind === "chore") return [[t("mainChat.echo.saveChore"), "save"], [t("mainChat.echo.once"), "once"], [t("common.cancel"), "cancel"]];
  if (echo.kind === "chore_saved" || echo.kind === "sleep") return [undo];
  if (echo.kind === "started" || echo.kind === "level") {
    return echo.kind === "started" ? [[t("mainChat.echo.change"), "level"]] : [undo];
  }
  if (echo.kind === "control") return [undo];
  // A message sent as a task's answer can be taken back until it is delivered.
  if (echo.kind === "answer_sent") return [undo];
  if (echo.kind === "suggest") {
    if (echo.action === "level") return [[t("mainChat.echo.setTo", {level: levelCode(echo.level)}), "apply"]];
    const word = ["stop", "pause", "resume", "first"].includes(echo.action) ? t(`mainChat.echo.${echo.action}`) : String(echo.action);
    return [[word.charAt(0).toUpperCase() + word.slice(1), "apply"]];
  }
  // "When should I remind you?" is answered by typing a time.
  if (echo.kind === "ask") return echo.ask === "when" ? [] : [[t("mainChat.echo.start"), "start"], [t("mainChat.echo.notNow"), "dismiss"]];
  return [];
}

/**
 * One echo's nodes: the person's own message when it answers one, then the line with its buttons.
 * `{doc, echo, done, act}`: `act(echo, do, level)` runs a button; `done` hides them.
 */
export function echoNodes({doc, echo, done = false, act}) {
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const nodes = [];
  // A queued message that ran is its turn's bubble now; the mark that it ran shows nothing.
  if (echo.kind === "dequeued" || echo.kind === "answer_delivered" || (echo.kind === "queued" && done)) return nodes;
  if (echo.typed) {
    const article = el("article", "msg user");
    article.append(el("div", echo.kind === "queued" ? "user-bubble queued" : "user-bubble", echo.typed));
    nodes.push(article);
  }
  const line = el("div", "echo");
  line.dataset.kind = echo.kind;
  line.dataset.echoId = String(echo.id);
  const mark = el("span", "echo-mark", MARKS[echo.action] ?? MARKS[echo.kind] ?? "·");
  mark.setAttribute("aria-hidden", "true");
  const level = Number.isInteger(echo.level) && echo.kind !== "suggest" ? echo.level : null;
  // An echo the gateway worded keeps its words for the page's language; the level is the chip.
  // A step that ran on an earlier approval says its time as the page writes a time.
  const approvedAt = echo.kind === "earlier" ? Number(echo.approved_at) : 0;
  const own = approvedAt ? t("approvals.used", {time: clock(approvedAt)}) : saidText(echo.said);
  const words = level === null || own !== null ? own ?? echo.line : echo.line.replace(/ · [^·]+$/, "").replace(/ → [^→]+$/, " →");
  // The words, the level and the buttons wrap together, under the words.
  const body = el("div", "echo-body");
  body.append(el("p", "echo-text", words));
  line.append(mark, body);
  if (level !== null) {
    // The chip is the code; its meaning is the tooltip and, hidden, part of its accessible name.
    const chip = el("span", "echo-level", levelCode(level));
    chip.dataset.level = String(level);
    chip.title = levelMeaning(level);
    chip.append(el("span", "sr-only", `, ${levelMeaning(level)}`));
    body.append(chip);
  }
  // A chore's proposal names the work it would repeat (a reminder, its words), under the question.
  if (echo.kind === "chore" && (echo.work || echo.text)) body.append(el("p", "echo-sub", echo.work || echo.text));
  // The step that ran on an earlier approval, and the task that ran it.
  if (echo.kind === "earlier" && echo.preview) body.append(el("p", "echo-sub", t("approvals.ran", {title: echo.title ?? "", step: stepLine(echo.preview)})));
  // A task of another chat says whose it is (a side chat's title, or the main chat).
  const owner = echo.from_main ? t("chat.main") : echo.from_chat;
  if (owner) body.append(el("p", "echo-sub", t("chat.fromChat", {title: owner})));
  body.append(...askNodes({el, echo, done, act}));
  const buttons = el("span", "echo-actions");
  for (const [label, does] of echoActions(echo, done)) {
    const button = el("button", ["start", "save"].includes(does) ? "outline-button" : "link-button", label);
    button.type = "button";
    button.addEventListener("click", () => {
      if (does !== "level") return act(echo, does);
      const list = levelList({el, echo, act});
      buttons.replaceChildren(list);
      list.children[echo.level ?? 2]?.focus();
    });
    buttons.append(button);
  }
  if (buttons.children.length) body.append(buttons);
  body.append(...sleepNodes({el, echo}));
  nodes.push(line);
  return nodes;
}

/** What a tidy-up of the notes changed, folded under its row: a word for each change, then the note. */
function sleepNodes({el, echo}) {
  if (echo.kind !== "sleep" || !Array.isArray(echo.items) || !echo.items.length) return [];
  const list = el("dl", "ask-fields");
  for (const item of echo.items) {
    list.append(el("dt", "", t(`mainChat.echo.sleepOp.${item.change}`)), el("dd", "", String(item.text ?? "")));
  }
  const more = Number(echo.more);
  if (more > 0) list.append(el("dt", ""), el("dd", "sleep-more", t("running.more", {n: more})));
  const details = el("details", "ask-details sleep-details");
  details.append(el("summary", "", t("mainChat.echo.sleepChanged")), list);
  return [details];
}

/** A task's question folded under its row, an answer quoted, or a question to the person with its reply box. */
function askNodes({el, echo, done, act}) {
  const fields = (values, names, label) => {
    const list = el("dl", "ask-fields");
    for (const name of names) {
      const value = String(values?.[name] ?? "").trim();
      if (!value) continue;
      const term = el("dt", "", label(camel(name)));
      const text = el("dd", "", value);
      text.dataset.part = name;
      list.append(term, text);
    }
    return list;
  };
  if (echo.kind === "task_ask") {
    const details = el("details", "ask-details");
    details.append(el("summary", "", t("ask.show")), fields(echo.fields, ASK_FIELDS, (name) => t(`ask.field.${name}`)));
    return [details];
  }
  if (echo.kind === "task_answered" || (echo.kind === "ask_end" && echo.text)) return [el("blockquote", "ask-quote", echo.text)];
  if (echo.kind !== "task_escalate") return [];
  const nodes = [fields(echo.parts, PERSON_PARTS, (name) => t(`ask.part.${name}`))];
  const key = `${echo.ask_id ?? ""}:${echo.id}`;
  // How to answer: the box, a plain message when only this task waits, or /answer TASK when several do.
  if (!done) nodes.push(el("p", "echo-sub", Number(echo.waiting) > 1 ? t("ask.answerHintMany", {title: echo.title ?? ""}) : t("ask.answerHint")));
  if (done) {
    replyDrafts.delete(key);
    return nodes;
  }
  const form = el("form", "ask-reply");
  const box = el("textarea", "ask-reply-text");
  box.rows = 2;
  box.placeholder = t("ask.replyHint");
  box.setAttribute("aria-label", t("ask.replyLabel", {title: echo.title ?? ""}));
  box.dataset.replyFor = key;
  box.value = replyDrafts.get(key) ?? "";
  box.addEventListener("input", () => replyDrafts.set(key, box.value));
  const send = el("button", "outline-button", t("ask.send"));
  send.type = "submit";
  form.append(box, send);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = box.value.trim();
    if (!text) return box.focus();
    send.disabled = true;
    let sent = false;
    try {
      sent = (await act(echo, "reply", null, text)) !== false;
    } finally {
      // Not sent: the words stay and Send works again.
      if (sent) replyDrafts.delete(key);
      else send.disabled = false;
    }
  });
  nodes.push(form);
  return nodes;
}

/** The six levels as an option list; arrows move, Enter or a click chooses. */
function levelList({el, echo, act}) {
  const list = el("span", "echo-levels");
  list.setAttribute("role", "radiogroup");
  list.setAttribute("aria-label", t("common.priority"));
  LEVELS.forEach((_, level) => {
    const item = el("button", "echo-level-item", levelLabel(level));
    item.type = "button";
    item.dataset.level = String(level);
    item.setAttribute("role", "radio");
    item.setAttribute("aria-checked", String(level === echo.level));
    item.tabIndex = level === echo.level ? 0 : -1;
    item.addEventListener("click", () => act(echo, "level", levelCode(level)));
    item.addEventListener("keydown", (event) => {
      const step = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[event.key];
      if (!step) return;
      event.preventDefault();
      const items = [...list.children];
      items[(items.indexOf(item) + step + items.length) % items.length].focus();
    });
    list.append(item);
  });
  return list;
}

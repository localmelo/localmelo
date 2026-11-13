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

import {ENDING_MARKS, canFinish, endingLabel, endingWhy, finishLabel, readEnding} from "./endings.mjs";
import {answerLine, blockCard, blockFacts, pageText, ranOut} from "./blocked.mjs";
import {maker, startOfDay} from "./dom.mjs";
import {formatTime, saidText, t} from "./i18n.mjs";
import {LEVELS, levelCode, levelLabel, levelMeaning} from "./levels.mjs";
import {statusText} from "./turns.mjs";

const DAY_S = 86400;

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

// Why a run did not finish, as the page words its end: "Could not get an answer from the model server", "Timed out".
const failureLine = (failure) => (failure?.status ? statusText(String(failure.status), String(failure.cause ?? "")).title : null);

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
    // A card the gateway worded keeps its words: it reads in the page's language, whoever stored it. A run that did not
    // finish says why from its kind, in the page's language; the server's own line stays on the task's page.
    line: failureLine(data.failure) ?? saidText(data.said) ?? String(data.line ?? ""),
    // How an approval was answered (`allowedOnce`, `denied`, …), when the card kept its words.
    ...(String(data.said?.key ?? "").startsWith("cards.answer.") ? {answer: data.said.key.slice(13)} : {}),
    // A waiting card and the answer that follows it share this.
    ...(data.request_id ? {requestId: String(data.request_id)} : {}),
    // The card of a reminder's run: its heading is "Reminder: <the person's words>".
    ...(data.reminder === true ? {reminder: true} : {}),
    // A task's card: how it ended, as code decided it.
    ...(readEnding(data.ending) ? {ending: readEnding(data.ending)} : {}),
    // Its chat follows up on this end (#431): the task's lines keep their buttons until it says what it did.
    ...(data.follow_up === true ? {followUp: true} : {}),
    // Which end of its task this card is (the gateway's event id): a follow-up's row names the end it follows.
    ...(endOf(data.end_id) !== null ? {endId: endOf(data.end_id)} : {}),
    // The start continued its task by itself after a restart: when.
    ...(Number(data.continued_at) > 0 ? {continuedAt: Number(data.continued_at)} : {}),
    // An end that needs the person while exactly one step Auto blocked can still be allowed once: that step.
    // It holds what the step's own card shows, so the card says which step its button allows; a question its
    // tool asked itself (typing on a page, a form sent) also keeps its name and arguments: the card shows its text.
    // Whether it may be allowed, kept, or was answered is its task's to say (`taskBlock`), never the card's copy.
    ...(state === "done" && data.block?.request_id ? {block: {
      requestId: String(data.block.request_id),
      stepKey: String(data.block.step_key ?? ""),
      name: String(data.block.name ?? ""),
      preview: String(data.block.preview ?? ""),
      review: data.block.review && typeof data.block.review === "object" ? data.block.review : {},
      ...questionOf(data.block),
    }} : {}),
    // A step Auto blocked in a task (#369): the block as the task's page has it, with Allow once.
    ...(state === "blocked" ? {
      stepKey: String(data.step_key ?? ""),
      name: String(data.name ?? ""),
      preview: String(data.preview ?? ""),
      review: data.review && typeof data.review === "object" ? data.review : {},
      ...questionOf(data),
    } : {}),
  };
}

/** The question a stored block was (`question`, `arguments`), when its tool asked one itself. */
function questionOf(data) {
  return typeof data.question === "string" && data.question
    ? {question: data.question, arguments: data.arguments && typeof data.arguments === "object" ? data.arguments : {}}
    : {};
}

/**
 * What a blocked card's block says (`blocked.mjs`), or null: as its task's body reads it (*read*, from `taskBlock`),
 * the one record of whether it was answered; until that is read, the step as the card stored it, with no choice.
 */
export function cardBlock(card, read = null) {
  if (card?.state !== "blocked") return null;
  if (read) return read.facts;
  const facts = stepFacts(card);
  return facts?.category ? {...facts, allow: false} : null;
}

/**
 * What a blocked step says (`blocked.mjs`), from its request id, step key, tool, preview and review, and for a question
 * its tool asked itself its name and arguments; null if unusable.
 */
const stepFacts = ({requestId, stepKey, name, preview, review, question, arguments: args}) => blockFacts({
  approval: {origin: "auto", allowed: false, request_id: requestId, step_key: stepKey, review, tool: question, arguments: args},
  name,
  preview,
});

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
 * Whether a card asks the person something now: an approval that waits, a step Auto blocked, or a task that ended
 * "Needs you", while nothing newer of its task came after it (*newest*: the newest card id of each source). Whether
 * its step was answered since is its task's to say, which the order does not wait for.
 */
const asksCard = (card, newest) => {
  const asks = card.state === "waiting" || card.state === "blocked" || (card.state === "done" && card.ending?.state === "blocked");
  return asks && (newest.get(card.sourceId) ?? card.id) <= card.id;
};

/** A finished run that takes part in a fold: a task that ended ready (or says nothing of how it ended), not a reminder. */
const foldable = (card) => card.sourceKind === "task" && card.state === "done" && !card.reminder && (!card.ending || card.ending.state === "ready");

/**
 * One stretch of rows (what stands between two turns), as the person reads it: finished runs of one request (the same
 * title) are one `fold` at the place of their newest run, and what asks the person comes last, so it is the row nearest
 * the message box. Rows keep their time order otherwise. The runs that need the person never fold into the finished ones.
 */
function arrange(stretch, asks) {
  const groups = new Map();
  for (const item of stretch) {
    if (item.kind === "card" && foldable(item.card)) groups.set(item.card.title, [...(groups.get(item.card.title) ?? []), item.card]);
  }
  const rows = [];
  for (const item of stretch) {
    const group = item.kind === "card" && foldable(item.card) ? groups.get(item.card.title) : null;
    if (!group || group.length < 2) rows.push(item);
    else if (item.card === group.at(-1)) rows.push({kind: "fold", fold: {key: String(group[0].id), title: item.card.title, cards: group, at: item.card.at}});
  }
  return [...rows.filter((item) => !asks(item)), ...rows.filter(asks)];
}

/**
 * The main chat's conversation: its turns, cards and echoes by time, with a divider at each new day.
 * Between two turns, a source shows only its newest card, so a busy side chat is one card; every
 * echo stays, and so does every blocked step (each has its own Allow once). Finished runs of one request fold
 * into one row and what asks the person stands last in its stretch (`arrange`). *waiting* is the newest turn, when its
 * approval waits for the person: it stands last, so the rows that came meanwhile stand above it and what waits is
 * the thing nearest the message box; the order of turns never changes. Items: `{kind: "day", label}`, `{kind: "turn", turn}`,
 * `{kind: "card", card}`, `{kind: "fold", fold: {key, title, cards, at}}`, `{kind: "echo", echo}`.
 */
export function timeline(turns, cards, now, echoes = [], {waiting = null} = {}) {
  // melo's follow-up row is written once its turn began (#411): it stands just before that turn.
  const starts = new Map(turns.map((turn) => [String(turn.id ?? ""), Number(turn.startedAt) || 0]));
  const at = (item) => {
    if (item.kind === "turn") return Number(item.turn.startedAt) || 0;
    const lead = item.kind === "echo" && item.echo.kind === "followup" ? starts.get(String(item.echo.turn_id ?? "")) : 0;
    return lead ? lead - 0.001 : item[item.kind].at;
  };
  // The waiting turn, and the row melo wrote for it (#411), come last, in that order.
  const leads = (item) => item.kind === "echo" && item.echo.kind === "followup" && waiting !== null && String(item.echo.turn_id ?? "") === String(waiting.id ?? "");
  const order = (item) => (item.kind === "turn" && item.turn === waiting ? Number.MAX_SAFE_INTEGER : leads(item) ? Number.MAX_SAFE_INTEGER - 1 : at(item));
  const items = [
    ...turns.map((turn) => ({kind: "turn", turn})),
    ...foldAnswers(cards).map((card) => ({kind: "card", card})),
    ...echoes.map((echo) => ({kind: "echo", echo})),
  ].sort((left, right) => order(left) - order(right));
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
  const gone = answered(echoes);
  const ended = endedTasks(cards, echoes);
  const newest = new Map();
  for (const card of cards) newest.set(card.sourceId, Math.max(newest.get(card.sourceId) ?? 0, card.id));
  const asks = (item) => (item.kind === "card" ? asksCard(item.card, newest) : item.kind === "echo" && (leads(item) || (item.echo.kind === "task_escalate" && !echoDone(item.echo, gone, ended))));
  const arranged = [];
  let stretch = [];
  const flush = () => {
    arranged.push(...arrange(stretch, asks));
    stretch = [];
  };
  for (const item of kept) {
    if (item === null) continue;
    if (item.kind === "turn") {
      flush();
      arranged.push(item);
    } else stretch.push(item);
  }
  flush();
  const out = [];
  let day = "";
  // A turn that waited for days and stands last is not a label back in time.
  let latest = 0;
  for (const item of arranged) {
    latest = Math.max(latest, at(item));
    const label = latest ? dayLabel(latest, now) : "";
    if (label && label !== day) out.push({kind: "day", label});
    day = label || day;
    out.push(item);
  }
  return out;
}

/**
 * The `seen` a send carries (#617): the stored id of the newest card or line among *items* (what
 * `timeline` gave the page to draw), kept below the oldest of *held*, the card frames held back
 * while the person types. Only what was on the page counts; 0 for nothing.
 */
export function drawnUpTo(items, held = []) {
  const ids = (list) => list.map(Number).filter(Number.isInteger);
  const drawn = ids(items.map((item) => (item.kind === "card" ? item.card.id : item.kind === "fold" ? Math.max(...item.fold.cards.map((card) => card.id)) : item.kind === "echo" ? item.echo.id : NaN)));
  const waiting = ids(held.map((frame) => frame?.id));
  return Math.max(0, Math.min(Math.max(0, ...drawn), Math.min(Infinity, ...waiting) - 1));
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

/**
 * A card's words: its mark, its heading and its action. *stage* is its task's stage once the step the card stands on
 * was answered, or once its work went on in a newer task (`stepState`, `headStage`): the heading then says where the
 * task is now, never the Needs you it was.
 */
export function cardWords(card, stage = null) {
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
  // A done task says how code found it ended, in the words of every screen: Ready, Done, not finished (Finish it),
  // Needs you…
  const ending = card.state === "done" ? card.ending : null;
  if (ending && stage) return {mark: "✓", heading: t("mainChat.card.ended", {state: t(stage.label, {chat: ""}), title}), action: open};
  if (ending) {
    const heading = t("mainChat.card.ended", {state: endingLabel(ending), title});
    return {mark: ENDING_MARKS[ending.state], heading, action: open, ...(canFinish(ending) ? {more: finishLabel(ending)} : {})};
  }
  if (card.state === "failed") return {mark: "✕", heading: t("mainChat.card.failed", {title}), action: open};
  // A task that ran out of steps stopped; it can go on from where it was, so Continue comes first.
  if (card.state === "stopped") return {mark: "■", heading: t("mainChat.card.stopped", {title}), action: open, more: t("failure.continue")};
  return {mark: "✓", heading: t("mainChat.card.done", {title}), action: open};
}

/**
 * One card's node: `{doc, card, open, resume, approval, block, step}`; `open` gets the card, and
 * `resume(card, button)` is the Continue of a stopped task's card. `approval` is a waiting task's
 * approval card, drawn inside it (its Open then only opens the task). `block` is what a blocked
 * card's Allow once needs (`blockCard`'s options: `allowing`, `busy`, `onAllow`, `error`,
 * `developer`, and `facts`, as `cardBlock` reads them). `step` is the Allow this step of a card that needs the person
 * (`stepState`'s, with `onAllow` and `onKeep`). Its day is the divider above it.
 */
export function cardNode({doc, card, open, resume = null, approval = null, block = null, step = null}) {
  const {node: el} = maker(doc);
  const stage = headStage(step);
  const words = cardWords(card, stage);
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
  const facts = block?.facts ?? cardBlock(card);
  // A blocked step says what and why in its block; its line is for the list and the terminal.
  const line = (card.state === "waiting" && repeats) || facts ? "" : card.line;
  if (line) text.append(el("p", "main-card-line", line));
  // Why it is not ready: what code found missing, or its question (which the step it stands on, shown below, says).
  const why = card.ending && card.state !== "waiting" ? endingWhy(card.ending, {asks: !card.block}) : "";
  if (why) text.append(el("p", "main-card-why", why));
  if (card.continuedAt) text.append(el("p", "main-card-line", t("cards.continued", {when: clock(card.continuedAt)})));
  // Allow this step for a text typed or a form sent shows all of it first, as its blocked card does; the sentence that
  // names the site and the box stands by the button.
  const page = card.block && step && card.block.question ? pageText(card.block.question, card.block.arguments) : null;
  if (page) {
    const shown = el("div", "block-code");
    shown.append(el("code", "block-step", page.text));
    text.append(shown);
  }
  if (facts) text.append(blockCard(doc, facts, block ?? {}));
  // Its step was answered (as its task reads it, on any screen): it asks nothing more.
  if (step?.state === "answered" || stage || (facts && answerLine(facts, block?.allowing ?? {state: "open"}))) node.dataset.answered = "true";
  const inline = card.state === "waiting" && approval;
  if (inline) text.append(approval);
  const action = el("button", card.state === "waiting" && !inline ? "outline-button" : "link-button", inline ? t("mainChat.openTask") : words.action);
  action.type = "button";
  // The panel returns focus here when a task opened from this card is closed.
  action.dataset.focusKey = `card:${card.id}`;
  action.addEventListener("click", () => open(card));
  const foot = el("div", "main-card-foot");
  if (card.block && step) foot.append(...stepNodes(el, step, step.facts ?? stepFacts(card.block)));
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

// The folds the person opened: a redraw of the chat keeps them open.
const openFolds = new Set();

/**
 * One fold's node (`timeline`'s `fold`): the finished runs of one request as one row, with its newest run's mark and
 * words and how many times it ran, opened by a click to a line for each run: what it said, its Open task and its time.
 * `open(card)` is a card's Open task. The number in the row is the length of the list it opens.
 */
export function foldNode({doc, fold, open}) {
  const {node: el} = maker(doc);
  const newest = fold.cards.at(-1);
  const words = cardWords(newest);
  const heading = t("mainChat.fold", {heading: words.heading, n: fold.cards.length});
  const node = el("article", "main-card main-fold");
  node.dataset.state = newest.state;
  if (newest.ending) node.dataset.ending = newest.ending.state;
  node.setAttribute("aria-label", heading);
  const mark = el("span", "main-card-mark", words.mark);
  mark.setAttribute("aria-hidden", "true");
  const details = el("details", "main-fold-details");
  details.open = openFolds.has(fold.key);
  details.addEventListener("toggle", () => {
    if (details.open) openFolds.add(fold.key);
    else openFolds.delete(fold.key);
  });
  const summary = el("summary", "main-card-title", heading);
  summary.append(el("span", "main-card-time", newest.at ? clock(newest.at) : ""));
  const list = el("ul", "main-fold-list");
  for (const card of fold.cards) {
    const run = el("li", "main-fold-run");
    if (card.line) run.append(el("span", "main-card-line", card.line));
    const action = el("button", "link-button", cardWords(card).action);
    action.type = "button";
    // The panel returns focus here when a task opened from this run is closed.
    action.dataset.focusKey = `card:${card.id}`;
    action.addEventListener("click", () => open(card));
    run.append(action, el("span", "main-card-time", card.at ? clock(card.at) : ""));
    list.append(run);
  }
  details.append(summary, list);
  const text = el("div", "main-card-text");
  text.append(details);
  node.append(mark, text);
  return node;
}

/**
 * The Allow this step of a card that needs the person: the step it allows in the words of its own card (*facts*),
 * then the button, how it was answered (the quiet line its task's page shows), or why it did not take. A scheduled
 * task's run (`facts.keepSite`) also offers to keep the answer for its next runs on that site (`onKeep`), as the step's
 * own card does. Until its task's body is read (`reading`), and when nothing may allow it now (`closed`), no button.
 */
function stepNodes(el, {state = "open", line = "", expired = "", error = "", onAllow = null, onKeep = null}, facts = null) {
  const what = [...(facts ? [el("span", "main-card-note", facts.gist?.sentence || facts.title)] : []),
    ...(expired ? [el("span", "main-card-note", expired)] : [])];
  const site = facts?.keepSite ? facts.keepSite.slice(facts.keepSite.indexOf("://") + 3) : "";
  if (state === "answered") return [...what, el("span", "main-card-note", line)];
  if (state === "reading" || state === "closed") return what;
  const buttons = [];
  if (facts?.allow !== false) {
    const allow = el("button", "outline-button", t("mainChat.card.allowStep"));
    allow.type = "button";
    allow.disabled = state === "allowing";
    if (onAllow) allow.addEventListener("click", onAllow);
    buttons.push(allow);
  }
  if (site && !facts.kept) {
    const keep = el("button", "outline-button block-keep", t("auto.blocked.keep", {site}));
    keep.type = "button";
    keep.disabled = state === "allowing";
    if (onKeep) keep.addEventListener("click", onKeep);
    buttons.push(keep);
  }
  return [...what, ...buttons, ...(error ? [alertNote(el, error)] : [])];
}

/** Why a card's step was not allowed, read out as it appears. */
function alertNote(el, error) {
  const problem = el("span", "main-card-note", t("auto.blocked.failed", {problem: error}));
  problem.setAttribute("role", "alert");
  return problem;
}

/**
 * Where a card's Allow this step stands, from its task's body as `taskBlock` reads it (*read*, the record every screen
 * reads): `answered` with its `line` once the step was allowed or kept (here, on the task's page or by the person's typed
 * yes), `closed` when nothing may allow it now, `reading` until the body is read, `allowing` while this page's request
 * is under way, else `open`; `error` is the last failure's text, `facts` the step as read and `stage` its task's.
 */
/** The task's stage a card's heading says: once its step was answered, or whenever the task's work went on in a newer
 * one (it cannot be allowed there, so its step reads `closed`). */
const headStage = (step) => (step?.stage && (step.state === "answered" || step.stage.key === "went_on") ? step.stage : null);

export function stepState(card, {read = null, busy = {}} = {}) {
  const id = card.block?.requestId;
  if (!id) return null;
  if (!read) return {state: "reading"};
  const line = answerLine(read.facts, read.allowing);
  if (line) return {state: "answered", line, facts: read.facts, stage: read.stage};
  // An allowance that ran out unused is no answer: the step asks again, and says so.
  const expired = ranOut(read.allowing);
  if (!read.facts.allow && !read.facts.keepSite) return {state: "closed", facts: read.facts, expired, stage: read.stage};
  const now = busy[id];
  const going = now === "allowing" || now === "keeping";
  return {state: going ? "allowing" : "open", error: now && !going ? now : "", facts: read.facts, expired, stage: read.stage};
}

// ── Echoes: what the main chat did with a task ──

const ECHO_KINDS = new Set(["started", "control", "level", "suggest", "ask", "undone", "failed", "notice", "sent", "chore", "chore_saved", "checkin", "sleep",
  "task_ask", "task_answered", "task_escalate", "ask_end", "queued", "dequeued", "answer_sent", "answer_delivered", "earlier", "revoked",
  "followup", "followed"]);
const MARKS = {started: "+", level: "↕", suggest: "?", ask: "?", undone: "↺", failed: "!", notice: "i", sent: "→", chore: "◷", chore_saved: "◷", checkin: "◷", sleep: "☾",
  stop: "■", pause: "‖", resume: "▶", first: "↑", send: "→", task_ask: "?", task_answered: "↩", task_escalate: "!", ask_end: "·", queued: "…", answer_sent: "↩",
  earlier: "✓", revoked: "↺", followup: "↻"};
// A task's question (#369): its seven fields, and the five parts of a question to the person.
const ASK_FIELDS = ["background", "situation", "problem", "likely_causes", "options", "help_needed", "if_unsolvable"];
const PERSON_PARTS = ["background", "what_happened", "tried", "choices", "question"];
const camel = (name) => name.replace(/_(\w)/g, (_, letter) => letter.toUpperCase());
// What the person typed in a reply box, by row: a redraw of the chat keeps it.
const replyDrafts = new Map();

/**
 * Whether *turn* is a turn melo began itself, on a task's question (#369) or its follow-up on a task that stopped
 * (#411): its row stands in for the person's bubble.
 */
export function askTurn(turn, echoes = []) {
  if (turn?.origin === "ask_owner" || turn?.origin === "followup") return true;
  const text = String(turn?.userText ?? "");
  return echoes.some((echo) => (echo.kind === "task_ask" || echo.kind === "followup") && echo.lead && text.startsWith(echo.lead));
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
 * The tasks nothing more will run for (#431), from *cards* and *echoes* as the page keeps them: those whose
 * newest end is a card no follow-up comes after, or the chat's follow-up on it (`followed`) that did not continue
 * it. A card the gateway marked `followUp` keeps the task live until the follow-up says what it did: continued
 * (the task runs again, and its next end decides), or asked, reported, skipped or stopped (nothing more runs).
 * Newest goes by end (`endId`, and the row's `end_id`) when both name theirs, never by when each was written.
 */
export function endedTasks(cards, echoes = []) {
  const newest = new Map();
  const note = (task, end) => {
    const seen = newest.get(task);
    if (!seen || later(end, seen)) newest.set(task, end);
  };
  for (const card of cards) {
    if (["done", "failed", "stopped"].includes(card.state)) {
      note(card.sourceId, {at: Number(card.at) || 0, end: card.endId ?? null, row: false, live: card.followUp === true});
    }
  }
  // After the cards: a follow-up is told after the end it follows, so it wins a tie.
  for (const echo of echoes) {
    if (echo.kind === "followed" && echo.task_id) {
      note(String(echo.task_id), {at: Number(echo.at) || 0, end: endOf(echo.end_id), row: true, live: echo.did === "continued"});
    }
  }
  return new Set([...newest].filter(([, end]) => !end.live).map(([task]) => task));
}

/** How each task ended, by task id, from its newest card that says (`ending`): what its browser cards say once over. */
export function taskEnds(cards) {
  const ends = new Map();
  for (const card of cards) if (card.sourceKind === "task" && card.ending) ends.set(card.sourceId, card.ending);
  return ends;
}

/** A card's or a follow-up row's end id, or null (rows from before ends were named have none). */
function endOf(value) {
  const id = value === null || value === undefined || value === "" ? NaN : Number(value);
  return Number.isInteger(id) ? id : null;
}

/** Whether *end* comes after *seen*: by end when both name theirs (a follow-up's row after its end's card), else by time. */
function later(end, seen) {
  if (end.end === null || seen.end === null) return end.at >= seen.at;
  return end.end > seen.end || (end.end === seen.end && (end.row || !seen.row));
}

/**
 * Whether *echo*'s buttons go: a later echo answered it (ids in *done*), or its task ended (ids in
 * *ended*, see `endedTasks`). A suggestion to send the person's message keeps its Send: a message reaches
 * a task in any state (it goes on with its job, or a new task starts from its record).
 */
export function echoDone(echo, done, ended) {
  const sends = echo.kind === "suggest" && echo.action === "send";
  return done.has(echo.id) || (!sends && ended.has(echo.task_id));
}

/** A level as a line names it: its P-code for a developer, what it means in words (`today`) in default mode. */
const levelName = (level, plain) => (plain ? levelMeaning(level) : levelCode(level));

/**
 * An echo's buttons: `[label, do]`; a level change opens the level list. *plain* (default mode) leaves out the level
 * list, which names P-codes: a task's priority is a developer's to change, and Stop stays.
 */
export function echoActions(echo, done, plain = false) {
  if (done || !echo.task_id && !["ask", "chore", "chore_saved", "sleep"].includes(echo.kind)) return [];
  const undo = [t("mainChat.echo.undo"), "undo"];
  // A reminder runs once, at its time: there is no "just once" for it.
  if (echo.kind === "chore" && echo.reminder) return [[t("mainChat.echo.remind"), "save"], [t("common.cancel"), "cancel"]];
  // The message started its work already: only the schedule is left to save.
  if (echo.kind === "chore" && echo.ran) return [[t("mainChat.echo.saveChore"), "save"], [t("common.cancel"), "cancel"]];
  if (echo.kind === "chore") return [[t("mainChat.echo.saveChore"), "save"], [t("mainChat.echo.once"), "once"], [t("common.cancel"), "cancel"]];
  if (echo.kind === "chore_saved" || echo.kind === "sleep") return [undo];
  if (echo.kind === "started" || echo.kind === "level") {
    // A task the chat started (#410): its priority, and Stop until it ends.
    return echo.kind === "started" ? [...(plain ? [] : [[t("mainChat.echo.change"), "level"]]), [t("common.stop"), "stop"]] : [undo];
  }
  // A Stop that found nothing left to stop has no Undo: Undo would start the task again.
  if (echo.kind === "control") return echo.undo === null ? [] : [undo];
  // A message sent as a task's answer can be taken back until it is delivered.
  if (echo.kind === "answer_sent") return [undo];
  if (echo.kind === "suggest") {
    if (echo.action === "level") return [[t("mainChat.echo.setTo", {level: levelName(echo.level, plain)}), "apply"]];
    const word = ["stop", "pause", "resume", "first", "send"].includes(echo.action) ? t(`mainChat.echo.${echo.action}`) : String(echo.action);
    return [[word.charAt(0).toUpperCase() + word.slice(1), "apply"]];
  }
  // "When should I remind you?" is answered by typing a time.
  // A "Start this as a task?" card kept from an older version starts nothing now: a chat
  // hands its own work on (#571). It can only be put away.
  if (echo.kind === "ask") return echo.ask === "when" ? [] : [[t("mainChat.echo.notNow"), "dismiss"]];
  return [];
}

/**
 * One echo's nodes: the person's own message when it answers one, then the line with its buttons.
 * `{doc, echo, done, act, plain}`: `act(echo, do, level)` runs a button; `done` hides them; `plain` (default mode) says a
 * changed level in words, never as a P-code, shows none on a task's start and offers no list of them.
 */
export function echoNodes({doc, echo, done = false, act, plain = false}) {
  const {node: el} = maker(doc);
  const nodes = [];
  // A queued message that ran is its turn's bubble now; the mark that it ran shows nothing. What a follow-up did
  // only tells the page whether more runs for its task (#431).
  if (echo.kind === "dequeued" || echo.kind === "answer_delivered" || echo.kind === "followed" || (echo.kind === "queued" && done)) return nodes;
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
  // A step that ran on an earlier approval, a row from before every Allow once was one call, is a plain line.
  const said = plain && echo.said?.params?.level ? {...echo.said, params: {...echo.said.params, level: levelName(echo.level, plain)}} : echo.said;
  const own = saidText(said);
  const words = level === null || own !== null ? own ?? echo.line : echo.line.replace(/ · [^·]+$/, "").replace(/ → [^→]+$/, " →");
  // The words, the level and the buttons wrap together, under the words.
  const body = el("div", "echo-body");
  body.append(el("p", "echo-text", words));
  line.append(mark, body);
  // A task starts at its own level, which default mode never mentions; a level the person changed is said in words.
  if (level !== null && !(plain && echo.kind === "started")) {
    // The chip is the code, its meaning the tooltip and, hidden, part of its accessible name; in default mode it is the meaning.
    const chip = el("span", "echo-level", levelName(level, plain));
    chip.dataset.level = String(level);
    if (!plain) {
      chip.title = levelMeaning(level);
      chip.append(el("span", "sr-only", `, ${levelMeaning(level)}`));
    }
    body.append(chip);
  }
  // A chore's proposal names the work it would repeat (a reminder, its words), under the question.
  if (echo.kind === "chore" && (echo.work || echo.text)) body.append(el("p", "echo-sub", echo.work || echo.text));
  // A follow-up melo began on its own says so, quietly (#411).
  if (echo.kind === "followup") body.append(el("p", "echo-sub", t("followup.marker")));
  // A task of another chat says whose it is (a side chat's title, or the main chat).
  const owner = echo.from_main ? t("chat.main") : echo.from_chat;
  if (owner) body.append(el("p", "echo-sub", t("chat.fromChat", {title: owner})));
  body.append(...askNodes({el, echo, done, act}));
  const buttons = el("span", "echo-actions");
  for (const [label, does] of echoActions(echo, done, plain)) {
    const button = el("button", does === "save" ? "outline-button" : "link-button", label);
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

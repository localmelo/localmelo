/**
 * The main chat and its side chats: the words and nodes `workspace.js` draws.
 *
 * The gateway keeps one main chat (`kind: "front"`, `main: true`). Every other console chat is one of
 * its side chats. When a side chat's run ends, or a task finishes, fails or waits for an approval, the
 * main chat gets a card (an `attention.card` event, never a turn). What the main chat did with a task
 * (started, paused, a suggestion, "Start this as a task?") is an echo line (an `attention.echo` event).
 * Nothing here fetches; the page passes what a card's Open and an echo's buttons do.
 */

/** Levels in words, P0 to P5; P-codes are for developer mode. */
export const LEVELS = ["Now", "Today", "Normal", "Whenever", "If idle", "Later"];

const DAY_S = 86400;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function startOfDay(ts) {
  const day = new Date(ts * 1000);
  day.setHours(0, 0, 0, 0);
  return day.getTime() / 1000;
}

/** "Today", "Yesterday", "Oct 28", or "Oct 28, 2025" for another year. */
export function dayLabel(ts, now) {
  const days = Math.round((startOfDay(now) - startOfDay(ts)) / DAY_S);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  const day = new Date(ts * 1000);
  const label = `${MONTHS[day.getMonth()]} ${day.getDate()}`;
  return day.getFullYear() === new Date(now * 1000).getFullYear() ? label : `${label}, ${day.getFullYear()}`;
}

/** "7:35 PM", as this computer writes a time (the panel's rows write it the same way). */
export function clock(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], {hour: "numeric", minute: "2-digit"});
}

/** A time beside a row: the clock today, else the day. */
export function shortTime(ts, now) {
  if (!ts) return "";
  return dayLabel(ts, now) === "Today" ? clock(ts) : dayLabel(ts, now);
}

export function isMain(session) {
  return session?.main === true || session?.kind === "front";
}

/** A card as the page keeps it, from a `cards` row or an `attention.card` frame; null if unusable. */
export function readCard(row) {
  const data = row?.data && typeof row.data === "object" ? {...row.data, id: row.id, ts: row.ts} : row;
  const id = Number(data?.id);
  if (!Number.isFinite(id) || !data?.source_id) return null;
  const state = ["done", "failed", "waiting"].includes(data.state) ? data.state : "done";
  return {
    id,
    at: Number(data.at) || Number(data.ts) || 0,
    state,
    sourceId: String(data.source_id),
    sourceKind: data.source_kind === "task" ? "task" : "chat",
    title: String(data.title || (data.source_kind === "task" ? "A task" : "A side chat")),
    line: String(data.line ?? ""),
  };
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
 * echo stays. Items: `{kind: "day", label}`, `{kind: "turn", turn}`, `{kind: "card", card}`,
 * `{kind: "echo", echo}`.
 */
export function timeline(turns, cards, now, echoes = []) {
  const at = (item) => (item.kind === "turn" ? Number(item.turn.startedAt) || 0 : item[item.kind].at);
  const items = [
    ...turns.map((turn) => ({kind: "turn", turn})),
    ...cards.map((card) => ({kind: "card", card})),
    ...echoes.map((echo) => ({kind: "echo", echo})),
  ].sort((left, right) => at(left) - at(right));
  const kept = [];
  let since = new Map();
  for (const item of items) {
    if (item.kind !== "card") {
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
  const open = card.sourceKind === "task" ? "Open task" : "Open side chat";
  if (card.state === "waiting") return {mark: "!", heading: `${card.title} needs your approval`, action: "Review"};
  if (card.state === "failed") return {mark: "✕", heading: `Did not finish: ${card.title}`, action: open};
  return {mark: "✓", heading: `Done: ${card.title}`, action: open};
}

/** One card's node: `{doc, card, open}`; `open` gets the card. Its day is the divider above it. */
export function cardNode({doc, card, open}) {
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const words = cardWords(card);
  const node = el("article", "main-card");
  node.dataset.state = card.state;
  node.dataset.cardId = String(card.id);
  node.setAttribute("aria-label", words.heading);
  const mark = el("span", "main-card-mark", words.mark);
  mark.setAttribute("aria-hidden", "true");
  const text = el("div", "main-card-text");
  text.append(el("p", "main-card-title", words.heading));
  if (card.line) text.append(el("p", "main-card-line", card.line));
  const action = el("button", card.state === "waiting" ? "outline-button" : "link-button", words.action);
  action.type = "button";
  action.addEventListener("click", () => open(card));
  const foot = el("div", "main-card-foot");
  foot.append(action, el("span", "main-card-time", card.at ? clock(card.at) : ""));
  node.append(mark, text, foot);
  return node;
}

// ── Echoes: what the main chat did with a task ──

const ECHO_KINDS = new Set(["started", "control", "level", "suggest", "ask", "undone", "failed"]);
const MARKS = {started: "+", level: "↕", suggest: "?", ask: "?", undone: "↺", failed: "!",
  stop: "■", pause: "‖", resume: "▶", first: "↑"};

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

/** The ids of echoes a later one answered (an Undo, an Apply, a Start): their buttons go. */
export function answered(echoes) {
  return new Set(echoes.map((echo) => Number(echo.of)).filter(Number.isFinite));
}

/** An echo's buttons: `[label, do]`; a level change opens the level list. */
export function echoActions(echo, done) {
  if (done || !echo.task_id && echo.kind !== "ask") return [];
  if (echo.kind === "started" || echo.kind === "level") {
    return echo.kind === "started" ? [["Change", "level"]] : [["Undo", "undo"]];
  }
  if (echo.kind === "control") return [["Undo", "undo"]];
  if (echo.kind === "suggest") {
    const word = echo.action === "level" ? `Set to ${LEVELS[echo.level] ?? "Normal"}` : echo.action;
    return [[word.charAt(0).toUpperCase() + word.slice(1), "apply"]];
  }
  if (echo.kind === "ask") return [["Start as a task", "start"], ["Not now", "dismiss"]];
  return [];
}

/**
 * One echo's nodes: the person's own message when it answers one, then the line with its buttons.
 * `{doc, echo, done, act, developer}`: `act(echo, do, level)` runs a button; `done` hides them.
 */
export function echoNodes({doc, echo, done = false, act, developer = false}) {
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const nodes = [];
  if (echo.typed) {
    const article = el("article", "msg user");
    article.append(el("div", "user-bubble", echo.typed));
    nodes.push(article);
  }
  const line = el("div", "echo");
  line.dataset.kind = echo.kind;
  line.dataset.echoId = String(echo.id);
  const mark = el("span", "echo-mark", MARKS[echo.action] ?? MARKS[echo.kind] ?? "·");
  mark.setAttribute("aria-hidden", "true");
  const level = Number.isInteger(echo.level) && echo.kind !== "suggest" ? echo.level : null;
  const words = level === null ? echo.line : echo.line.replace(/ · [^·]+$/, "").replace(/ → [^→]+$/, " →");
  const text = el("p", "echo-text", words);
  line.append(mark, text);
  if (level !== null) {
    const chip = el("span", "echo-level", developer ? `P${level} · ${LEVELS[level]}` : LEVELS[level]);
    chip.dataset.level = String(level);
    line.append(chip);
  }
  const buttons = el("span", "echo-actions");
  for (const [label, does] of echoActions(echo, done)) {
    const button = el("button", does === "start" ? "outline-button" : "link-button", label);
    button.type = "button";
    button.addEventListener("click", () => {
      if (does !== "level") return act(echo, does);
      const list = levelList({el, echo, act, developer});
      buttons.replaceChildren(list);
      list.children[echo.level ?? 2]?.focus();
    });
    buttons.append(button);
  }
  if (buttons.children.length) line.append(buttons);
  nodes.push(line);
  return nodes;
}

/** The six levels as an option list; arrows move, Enter or a click chooses. */
function levelList({el, echo, act, developer}) {
  const list = el("span", "echo-levels");
  list.setAttribute("role", "radiogroup");
  list.setAttribute("aria-label", "Level");
  LEVELS.forEach((word, level) => {
    const item = el("button", "echo-level-item", developer ? `P${level} · ${word}` : word);
    item.type = "button";
    item.dataset.level = String(level);
    item.setAttribute("role", "radio");
    item.setAttribute("aria-checked", String(level === echo.level));
    item.tabIndex = level === echo.level ? 0 : -1;
    item.addEventListener("click", () => act(echo, "level", word));
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

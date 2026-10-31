/**
 * The terminal's slash commands in the console: the "/" menu under the composer, the parsing the
 * terminal uses, and the local rows a command answers with.
 *
 * The list comes from `GET /api/commands` (the terminal's own table, with each command's words in
 * both languages and the page action it maps to), so the terminal and the page never drift.
 * A draft is a command when it starts with `/` and a name of lowercase letters and dashes, then
 * whitespace or its end (`/etc/hosts is broken` and `//text` are messages, as in the terminal; a
 * leading space also makes a message). A command is never sent to the model: its answer is a local
 * row under the conversation, kept per chat until the next message. The actions themselves are in
 * `slash-actions.mjs`; the page passes what they need as `host` (see `workspace.js`).
 */

import {language, t} from './i18n.mjs';
import {createActions} from './slash-actions.mjs';

const COMMAND = /^\/([a-z][a-z-]*)(?:\s+([\s\S]*))?$/;
const TYPING = /^\/([a-z-]*)$/;
const MAX_ROWS = 30;

/** `{name, args}` for a command; null for a message. An unknown name is still a command. */
export function parseCommand(text) {
  const match = COMMAND.exec(String(text).trimEnd());
  return match ? {name: match[1], args: (match[2] ?? '').trim()} : null;
}

/** The name being typed after the `/` (empty for a bare `/`), or null once the draft is more than that. */
export function menuQuery(draft) {
  const match = TYPING.exec(String(draft));
  return match ? match[1] : null;
}

/** The command a name or alias belongs to, or null. */
export function lookup(list, name) {
  return list.find((command) => command.name === name || command.aliases.includes(name)) ?? null;
}

/** A command's words in the page's language. */
export const helpOf = (command) => command.help?.[language()] || command.help?.en || '';

/**
 * The commands a typed name leads to: those whose name or alias starts with it, then those that
 * contain it. Each entry is `{command, spelling}`, the spelling being the name (or alias) that matched.
 */
export function matchCommands(list, query) {
  const typed = query.toLowerCase();
  const found = (test) => list.flatMap((command) => {
    const spelling = [command.name, ...command.aliases].find(test);
    return spelling === undefined ? [] : [{command, spelling}];
  });
  const prefix = found((name) => name.startsWith(typed));
  const rest = found((name) => name.includes(typed)).filter(({command}) => !prefix.some((item) => item.command === command));
  return [...prefix, ...rest];
}

function node(doc, tag, className, text) {
  const made = doc.createElement(tag);
  if (className) made.className = className;
  if (text !== undefined) made.textContent = text;
  return made;
}

/** `/name args  help`, as the menu and `/help` write a command: the name in the accent, the arguments muted. */
function commandLine(doc, className, spelling, command) {
  const line = node(doc, 'div', className);
  line.append(node(doc, 'span', 'slash-name', `/${spelling}`));
  if (command.args) line.append(node(doc, 'span', 'slash-args', command.args));
  line.append(node(doc, 'span', 'slash-help', helpOf(command)));
  return line;
}

function lineNode(doc, line) {
  const made = node(doc, 'div', 'local-line');
  if (line.label) made.append(node(doc, 'span', 'local-label', line.label));
  for (const part of line.parts ?? [{text: line.text, tone: line.tone}]) {
    const span = node(doc, 'span', 'local-text', part.text);
    if (part.tone) span.dataset.tone = part.tone;
    made.append(span);
  }
  if (line.confirm) {
    for (const [label, primary, run] of [[line.confirm.yes, true, line.confirm.onYes], [line.confirm.no, false, line.confirm.onNo]]) {
      const choice = node(doc, 'button', primary ? 'local-choice primary' : 'local-choice', label);
      choice.type = 'button';
      choice.addEventListener('click', () => run?.());
      made.append(choice);
    }
  }
  return made;
}

/** One answer: the command as typed, then its lines (the commands of a `/help` in one aligned table). */
export function rowNode(doc, row) {
  const made = node(doc, 'section', 'local-row');
  made.setAttribute('aria-label', t('slash.rowLabel', {command: row.command}));
  const [head, ...rest] = row.command.split(/\s+/);
  const echo = node(doc, 'div', 'local-command');
  echo.append(node(doc, 'span', 'slash-name', head));
  if (rest.length) echo.append(node(doc, 'span', 'slash-args', rest.join(' ')));
  made.append(echo);
  let table = null;
  for (const line of row.lines) {
    if (!line.help) {
      table = null;
      made.append(lineNode(doc, line));
      continue;
    }
    if (table === null) {
      table = node(doc, 'div', 'local-table');
      made.append(table);
    }
    table.append(commandLine(doc, 'local-help', line.help.spelling, line.help.command));
  }
  return made;
}

/**
 * Wire the "/" menu and the commands. `host` is the page: `doc`, `input` (the composer), `menu` and
 * `rows` (the elements), `api`, `slot()` (the open chat, or "new"), `submit()` (send the composer's
 * draft), `sync()` (the composer changed), `scrollToEnd()` and what `slash-actions.mjs` needs.
 * `key(event)` is the composer's keydown first: true when the menu used the key. `handle(draft)`
 * is for a send: false for a message, true when the draft was a command that ran, "keep" when it
 * was refused and stays in the composer.
 */
export function setupSlash(host) {
  const {doc, input, menu, rows: rowsRoot} = host;
  let loaded = null;
  let loading = null;
  let shown = [];
  let at = 0;
  let typed = '';
  let open = false;
  let version = 0;
  let drawn = '';
  const rows = new Map();

  function load() {
    loading ??= host.api('/commands').then(
      (body) => { loaded = {list: body.commands ?? [], everyday: body.everyday ?? []}; return loaded; },
      (error) => { loading = null; throw error; },
    );
    return loaded ? Promise.resolve(loaded) : loading;
  }

  // ── The menu ──

  function close() {
    if (!open) return;
    open = false;
    menu.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
  }

  function draw() {
    const items = shown.map(({command, spelling}, index) => {
      const item = commandLine(doc, 'slash-item', spelling, command);
      item.id = `slash-opt-${spelling}`;
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', String(index === at));
      item.addEventListener('mousedown', (event) => event.preventDefault?.());
      item.addEventListener('click', () => pick(index, {run: true}));
      return item;
    });
    const query = menuQuery(input.value) ?? '';
    const tail = shown.length === 0
      ? node(doc, 'div', 'slash-empty', t('slash.noMatch', {name: query}))
      : node(doc, 'div', 'slash-foot', t('slash.keys'));
    menu.replaceChildren(...items, tail);
    if (items[at]) {
      input.setAttribute('aria-activedescendant', items[at].id);
      items[at].scrollIntoView?.({block: 'nearest'});
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  /** Open, filter or close the menu for the composer's draft. */
  async function refresh() {
    const query = host.enabled() ? menuQuery(input.value) : null;
    if (query === null) {
      close();
      return;
    }
    let catalog;
    try {
      catalog = await load();
    } catch {
      return; // The list comes back on the next key; a send says why it could not load.
    }
    if (menuQuery(input.value) !== query) return;
    if (query !== typed) at = 0;
    typed = query;
    shown = matchCommands(catalog.list, query);
    open = true;
    menu.hidden = false;
    input.setAttribute('aria-controls', menu.id);
    input.setAttribute('aria-expanded', 'true');
    draw();
  }

  function pick(index, {run = false} = {}) {
    const choice = shown[index];
    if (!choice) return;
    const {command, spelling} = choice;
    close();
    // A no-argument command runs at once; one with arguments waits for them.
    const runs = run && command.args === '';
    input.value = runs ? `/${spelling}` : `/${spelling} `;
    host.sync();
    input.focus();
    if (runs) host.submit();
  }

  /** The composer's keydown: true when the menu took the key. */
  function key(event) {
    if (!open || event.isComposing || event.keyCode === 229) return false;
    const move = {ArrowDown: 1, ArrowUp: -1}[event.key];
    if (move !== undefined && shown.length > 0) {
      event.preventDefault();
      at = (at + move + shown.length) % shown.length;
      draw();
      return true;
    }
    if (event.key === 'Escape') {
      // The menu closes; the composer's own Escape (stopping a turn) waits for the next press.
      event.preventDefault();
      event.stopPropagation?.();
      close();
      return true;
    }
    if ((event.key === 'Enter' && !event.shiftKey) || event.key === 'Tab') {
      const choice = shown[at];
      if (!choice) return false;
      // A name typed in full is already the command: Enter sends it.
      if (event.key === 'Enter' && menuQuery(input.value) === choice.spelling) {
        close();
        return false;
      }
      event.preventDefault();
      pick(at, {run: event.key === 'Enter'});
      return true;
    }
    return false;
  }

  // ── Local rows ──

  function addRow(slot, row) {
    const list = rows.get(slot) ?? [];
    list.push(row);
    rows.set(slot, list.slice(-MAX_ROWS));
    version += 1;
    drawRows();
    host.scrollToEnd();
  }

  /** Forget a chat's rows: the conversation moved on. */
  function clearRows(slot = host.slot()) {
    if (rows.delete(slot)) version += 1;
    drawRows();
  }

  function drawRows() {
    const slot = host.slot();
    const key = `${slot}:${version}`;
    if (key === drawn) return;
    drawn = key;
    const list = rows.get(slot) ?? [];
    rowsRoot.replaceChildren(...list.map((row) => rowNode(doc, row)));
    rowsRoot.hidden = list.length === 0;
  }

  // ── Running a command ──

  async function handle(draft) {
    const parsed = parseCommand(draft);
    if (parsed === null) return false;
    close();
    const text = String(draft).trim();
    const say = (lines) => addRow(host.slot(), {command: text, lines});
    let catalog;
    try {
      catalog = await load();
    } catch (error) {
      say([{text: t('slash.loadFailed', {problem: error.message}), tone: 'error'}]);
      return 'keep';
    }
    const command = lookup(catalog.list, parsed.name);
    if (command === null) {
      say([{text: t('slash.unknown', {name: parsed.name}), tone: 'error'}]);
      return true;
    }
    const allowed = command.during_turn === true || (command.during_turn === 'show' && !parsed.args);
    if (host.state().busy && !allowed) {
      say([{text: t('slash.notDuringTurn', {name: command.name}), tone: 'error'}]);
      return 'keep';
    }
    try {
      const lines = await actions[command.action]?.({command, name: parsed.name, args: parsed.args, text});
      if (lines?.length) say(lines);
    } catch (error) {
      say([{text: error.message, tone: 'error'}]);
    }
    return true;
  }

  const actions = createActions({
    ...host,
    commands: () => loaded?.list ?? [],
    everyday: () => loaded?.everyday ?? [],
    lookup: (name) => lookup(loaded?.list ?? [], name),
    clearRows: () => clearRows(),
  });

  input.addEventListener('input', refresh);
  input.addEventListener('blur', () => {
    // A click on a menu row has already run (mousedown does not take the focus away).
    close();
  });

  return {handle, key, close, clearRows, draw: drawRows, isOpen: () => open, refresh, actions};
}

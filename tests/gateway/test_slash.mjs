/**
 * The terminal's slash commands in the console (`slash.mjs`, `slash-actions.mjs`): how a draft is
 * read, the "/" menu and its keys, and what every command does. The commands are the terminal's own
 * table (`tests/fixtures/slash_commands.json`, kept equal to `GET /api/commands` by
 * `tests/gateway/console/test_commands.py`), so a command added there must have a page action here.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {useLanguage} from '../../support/gateway/static/i18n.mjs';
import {helpOf, lookup, matchCommands, menuQuery, parseCommand, rowNode, setupSlash} from '../../support/gateway/static/slash.mjs';
import {toolCalls} from '../../support/gateway/static/slash-actions.mjs';

const CATALOG = JSON.parse(readFileSync(new URL('../fixtures/slash_commands.json', import.meta.url), 'utf8'));
const LIST = CATALOG.commands;

// ── A DOM double: elements with children, text, attributes and listeners ──

function fakeDoc() {
  const make = (tag) => {
    const node = {
      tag, children: [], attributes: {}, dataset: {}, listeners: {}, hidden: false, className: '', textContent: '', id: '', value: '',
      append(...items) { for (const item of items) node.children.push(typeof item === 'string' ? {textContent: item, children: []} : item); },
      replaceChildren(...items) { node.children = []; node.append(...items); },
      setAttribute(name, value) { node.attributes[name] = String(value); },
      getAttribute(name) { return node.attributes[name] ?? null; },
      removeAttribute(name) { delete node.attributes[name]; },
      addEventListener(type, fn) { (node.listeners[type] ||= []).push(fn); },
      focus() { doc.activeElement = node; },
      click() { for (const fn of node.listeners.click ?? []) fn({preventDefault() {}}); },
    };
    return node;
  };
  const doc = {activeElement: null, createElement: make};
  return {doc, make};
}

const text = (node) => [node.textContent, ...(node.children ?? []).map(text)].filter(Boolean).join(' ');
const find = (node, test_) => (test_(node) ? node : (node.children ?? []).map((child) => find(child, test_)).find(Boolean) ?? null);
const named = (node, name) => find(node, (n) => String(n.className ?? '').split(' ').includes(name));
const all = (node, name) => (String(node.className ?? '').split(' ').includes(name) ? [node] : []).concat((node.children ?? []).flatMap((child) => all(child, name)));

/** A page: the composer, the menu and the rows, a recording host, and the gateway's answers by "METHOD /path". */
function page({routes = {}, state = {}, host: extra = {}} = {}) {
  const {doc, make} = fakeDoc();
  const input = make('textarea');
  input.id = 'input';
  const menu = make('div');
  menu.id = 'slash-menu';
  menu.hidden = true;
  const rows = make('div');
  const calls = [];
  const sent = [];
  const events = [];
  const current = {
    id: 'chat-1', session: {id: 'chat-1', title: 'Trip', approvals: 'ask', workspace_label: '~/notes'}, title: 'Trip', main: false,
    turns: [], busy: false, developer: false, health: {model: {name: 'qwen', backend_label: 'oMLX'}, model_server: {reachable: true}, version: '1.0', uptime_s: 90, sessions: {open: 2, running: 1}},
    ...state,
  };
  const api = async (path, options = {}) => {
    const method = options.method ?? (options.body ? 'POST' : 'GET');
    calls.push({method, path, body: options.body});
    const handler = routes[`${method} ${path}`] ?? (path === '/commands' ? CATALOG : undefined);
    if (handler === undefined) throw new Error(`Unexpected request: ${method} ${path}`);
    const result = typeof handler === 'function' ? handler(options.body) : handler;
    if (result instanceof Error) throw result;
    return result;
  };
  const record = (name) => (...args) => { events.push([name, ...args]); };
  const host = {
    doc, input, menu, rows, api,
    enabled: () => true,
    slot: () => current.id ?? 'new',
    state: () => current,
    submit: record('submit'),
    sync: record('sync'),
    scrollToEnd: record('scroll'),
    send: async (line) => { sent.push(line); },
    newChat: record('newChat'),
    setNewTitle: record('setNewTitle'),
    setTitle: async (title) => { events.push(['setTitle', title]); },
    sessions: () => [{id: 'aaa111', title: 'Trip to Rome'}, {id: 'bbb222', title: 'Taxes'}, {id: 'ccc333', title: 'Trip to Oslo'}],
    pickChat: record('pickChat'),
    openChat: record('openChat'),
    openMain: () => { events.push(['openMain']); return true; },
    openSettings: record('openSettings'),
    openMemory: record('openMemory'),
    openDrawer: record('openDrawer'),
    settingsSaved: async () => { events.push(['settingsSaved']); },
    setMode: async (mode) => { events.push(['setMode', mode]); },
    view: {get: (name) => prefs[name], set: (name, on) => { prefs[name] = on; }},
    reveal: record('reveal'),
    writeText: async (line) => { events.push(['copied', line]); },
    isApp: () => false,
    canQuit: () => false,
    quit: record('quit'),
    ...extra,
  };
  const prefs = {verbose: false, thinking: false, stream: true};
  const slash = setupSlash(host);
  input.menuId = menu.id;
  return {slash, input, menu, rows, calls, sent, events, host, prefs, state: current};
}

/** Type `value` into the composer, as the person does, and let the menu settle. */
async function type(p, value) {
  p.input.value = value;
  for (const fn of p.input.listeners.input ?? []) await fn({});
}

/** A keydown, as the composer's handler receives it. */
function press(p, key, extra = {}) {
  const event = {key, keyCode: 0, isComposing: false, shiftKey: false, stopped: false, prevented: false, preventDefault() { event.prevented = true; }, stopPropagation() { event.stopped = true; }, ...extra};
  event.used = p.slash.key(event);
  return event;
}

const rowTexts = (p) => p.rows.children.map(text);
/** The newest page move of a kind (the page also scrolls and syncs after a row). */
const moved = (p, name) => p.events.filter(([kind]) => kind === name).at(-1);
const lastRow = (p) => text(p.rows.children.at(-1));

// ── Reading a draft ──

test('a draft is a command as the terminal reads it: a lowercase name, then whitespace or the end', () => {
  assert.deepEqual(parseCommand('/model'), {name: 'model', args: ''});
  assert.deepEqual(parseCommand('/model  qwen 3 \n'), {name: 'model', args: 'qwen 3'});
  assert.deepEqual(parseCommand('/approvals forget file_write'), {name: 'approvals', args: 'forget file_write'});
  assert.deepEqual(parseCommand('/side two\nlines'), {name: 'side', args: 'two\nlines'});
  assert.deepEqual(parseCommand('/foo'), {name: 'foo', args: ''}, 'An unknown name is still a command');
  for (const message of ['/etc/hosts is broken', '//text', ' /model', '\t/model', '/Model', '/ model', '/1', 'hi /model', '']) {
    assert.equal(parseCommand(message), null, JSON.stringify(message));
  }
});

test('the menu follows the name being typed: bare slash, letters and dashes, no space yet', () => {
  assert.equal(menuQuery('/'), '');
  assert.equal(menuQuery('/mo'), 'mo');
  assert.equal(menuQuery('/model '), null, 'A space ends the name');
  assert.equal(menuQuery('//'), null);
  assert.equal(menuQuery('/Mo'), null);
  assert.equal(menuQuery('hello'), null);
  assert.equal(menuQuery('/a\nb'), null);
});

test('names and aliases that start with the typed word come first, then those that contain it', () => {
  const names = (query) => matchCommands(LIST, query).map((item) => item.spelling);
  assert.equal(names('').length, LIST.length, 'A bare slash lists every command, in the terminal\'s order');
  assert.deepEqual(names('').slice(0, 3), ['model', 'new', 'status']);
  assert.deepEqual(names('mo'), ['model', 'memory', 'browser'].filter((n) => names('mo').includes(n)).slice(0, 2).concat(names('mo').slice(2)));
  assert.equal(names('mo')[0], 'model');
  assert.ok(names('mo').indexOf('memory') < names('mo').indexOf('main') || !names('mo').includes('main'));
  // Prefix before substring: "ta" starts nothing but "status"... check the order on a word with both kinds.
  const both = names('at');
  const starts = both.filter((n) => n.startsWith('at'));
  assert.deepEqual(both.slice(0, starts.length), starts, 'Prefix matches lead');
  assert.ok(both.length > starts.length, 'Substring matches follow');
  // An alias leads to its command, spelled as typed.
  assert.deepEqual(matchCommands(LIST, 'qu').map((item) => [item.command.name, item.spelling]), [['exit', 'quit']]);
  assert.deepEqual(names('zzz'), []);
  assert.equal(matchCommands(LIST, 'MO')[0].spelling, 'model', 'Case does not matter');
  assert.equal(lookup(LIST, 'quit').name, 'exit');
  assert.equal(lookup(LIST, 'nope'), null);
});

// ── The menu ──

test('typing a slash opens the menu under the composer; it filters as you type and closes after the name', async () => {
  const p = page();
  await type(p, '/');
  assert.equal(p.menu.hidden, false);
  assert.equal(p.input.attributes['aria-expanded'], 'true');
  const items = all(p.menu, 'slash-item');
  assert.equal(items.length, LIST.length);
  assert.equal(items[0].attributes.role, 'option');
  assert.equal(items[0].attributes['aria-selected'], 'true');
  assert.equal(p.input.attributes['aria-activedescendant'], items[0].id);
  // `/name args — help`, the name in the accent and the arguments muted.
  assert.equal(named(items[0], 'slash-name').textContent, '/model');
  assert.equal(named(items[0], 'slash-args').textContent, '[name]');
  assert.equal(named(items[0], 'slash-help').textContent, 'Pick a model, or /model NAME (saved)');
  await type(p, '/ex');
  assert.deepEqual(all(p.menu, 'slash-item').map((node) => named(node, 'slash-name').textContent), ['/exit', '/expand']);
  await type(p, '/exit ');
  assert.equal(p.menu.hidden, true, 'A space ends the name');
  assert.equal(p.input.attributes['aria-expanded'], 'false');
  await type(p, '//x');
  assert.equal(p.menu.hidden, true, 'A double slash is a message');
  await type(p, '/zzz');
  assert.equal(p.menu.hidden, false);
  assert.match(text(p.menu), /No command starts with \/zzz · \/help lists them/);
  assert.equal(all(p.menu, 'slash-item').length, 0);
  await type(p, '');
  assert.equal(p.menu.hidden, true);
});

test('arrows move, Enter or Tab pick, and a command without arguments runs on Enter', async () => {
  const p = page();
  await type(p, '/');
  let event = press(p, 'ArrowDown');
  assert.ok(event.used && event.prevented);
  assert.equal(p.menu.children[1].attributes['aria-selected'], 'true');
  assert.equal(p.input.attributes['aria-activedescendant'], 'slash-opt-new');
  press(p, 'ArrowUp');
  press(p, 'ArrowUp');
  assert.equal(p.input.attributes['aria-activedescendant'], `slash-opt-${LIST.at(-1).name}`, 'Up wraps to the end');
  // /model takes arguments: Enter fills the name and keeps typing.
  await type(p, '/mod');
  event = press(p, 'Enter');
  assert.ok(event.used && event.prevented);
  assert.equal(p.input.value, '/model ');
  assert.equal(p.menu.hidden, true);
  assert.deepEqual(p.events.filter(([name]) => name === 'submit'), []);
  // /new takes none: Enter runs it.
  await type(p, '/ne');
  press(p, 'Enter');
  assert.equal(p.input.value, '/new');
  assert.deepEqual(p.events.filter(([name]) => name === 'submit'), [['submit']]);
  // Tab only fills, whatever the command.
  p.events.length = 0;
  await type(p, '/ne');
  press(p, 'Tab');
  assert.equal(p.input.value, '/new ');
  assert.deepEqual(p.events.filter(([name]) => name === 'submit'), []);
  // An alias fills as typed.
  await type(p, '/qu');
  press(p, 'Tab');
  assert.equal(p.input.value, '/quit ');
});

test('a name typed in full is sent by Enter; Shift+Enter and an empty menu leave the key alone', async () => {
  const p = page();
  await type(p, '/model');
  const event = press(p, 'Enter');
  assert.equal(event.used, false, 'The composer sends it');
  assert.equal(event.prevented, false);
  assert.equal(p.menu.hidden, true);
  await type(p, '/mod');
  assert.equal(press(p, 'Enter', {shiftKey: true}).used, false, 'Shift+Enter is a new line');
  await type(p, '/zzz');
  assert.equal(press(p, 'Enter').used, false, 'Nothing to pick: the line is sent and answered as unknown');
  assert.equal(press(p, 'ArrowDown').used, false);
});

test('Escape closes the menu and keeps the draft, without stopping the turn behind it', async () => {
  const p = page();
  await type(p, '/mo');
  const event = press(p, 'Escape');
  assert.ok(event.used && event.stopped && event.prevented, 'The page\'s own Escape does not run');
  assert.equal(p.menu.hidden, true);
  assert.equal(p.input.value, '/mo');
  assert.equal(press(p, 'Escape').used, false, 'The next Escape is the page\'s');
});

test('while an input method composes, no key acts', async () => {
  const p = page();
  await type(p, '/');
  for (const extra of [{isComposing: true}, {keyCode: 229}]) {
    for (const key of ['ArrowDown', 'Enter', 'Tab', 'Escape']) {
      const event = press(p, key, extra);
      assert.equal(event.used, false, `${key} ${JSON.stringify(extra)}`);
      assert.equal(event.prevented, false);
    }
  }
  assert.equal(p.menu.hidden, false);
  assert.equal(p.input.value, '/');
});

test('a click on a row picks it and the composer keeps the focus', async () => {
  const p = page();
  await type(p, '/mo');
  const item = all(p.menu, 'slash-item')[0];
  let prevented = false;
  for (const fn of item.listeners.mousedown) fn({preventDefault() { prevented = true; }});
  assert.ok(prevented, 'The mouse does not take the focus from the composer');
  item.click();
  assert.equal(p.input.value, '/model ');
  assert.equal(p.menu.hidden, true);
  assert.equal(p.host.doc.activeElement, p.input);
  await type(p, '/ne');
  all(p.menu, 'slash-item')[0].click();
  assert.deepEqual(p.events.filter(([name]) => name === 'submit'), [['submit']], 'A click on /new runs it');
});

test('the menu stays shut where commands do not run, and when the list cannot be read', async () => {
  const off = page({host: {enabled: () => false}});
  await type(off, '/');
  assert.equal(off.menu.hidden, true);
  const broken = page({routes: {'GET /commands': new Error('down')}});
  await type(broken, '/');
  assert.equal(broken.menu.hidden, true);
  assert.equal(await broken.slash.handle('/help'), 'keep', 'A send says why, and keeps the draft');
  assert.match(lastRow(broken), /\/help.*Could not load the commands: down/);
});

// ── Running a command ──

test('a message is not a command: handle leaves it to the composer', async () => {
  const p = page();
  for (const message of ['hello', '//text', ' /model', '/etc/hosts is broken']) assert.equal(await p.slash.handle(message), false, message);
  assert.deepEqual(p.calls, []);
});

test('an unknown command is answered by a local row and never sent', async () => {
  const p = page();
  assert.equal(await p.slash.handle('/foo bar'), true);
  assert.deepEqual(p.sent, []);
  assert.equal(p.rows.hidden, false);
  assert.match(lastRow(p), /^\/foo bar Unknown command \/foo — \/help lists them$/);
  assert.equal(named(p.rows.children[0], 'local-text').dataset.tone, 'error');
  assert.equal(named(p.rows.children[0], 'slash-name').textContent, '/foo');
  assert.equal(named(p.rows.children[0], 'slash-args').textContent, 'bar');
});

test('a command that is not for a turn in progress says so and keeps the draft', async () => {
  const p = page({state: {busy: true}});
  assert.equal(await p.slash.handle('/new'), 'keep');
  assert.match(lastRow(p), /\/new is not available while a turn runs \(Esc stops it\)\./);
  assert.deepEqual(p.events.filter(([name]) => name === 'newChat'), []);
  // Some run during a turn; /approvals does only to show.
  assert.equal(await p.slash.handle('/help'), true);
  assert.equal(await p.slash.handle('/approvals ask'), 'keep');
  assert.match(lastRow(p), /\/approvals is not available while a turn runs/);
  assert.equal(await p.slash.handle('/approvals'), true);
  assert.match(lastRow(p), /Approvals: Ask · grants: none/);
});

test('the rows of a chat go when the conversation moves on, and a chat keeps its own', async () => {
  const p = page();
  await p.slash.handle('/help');
  assert.equal(p.rows.children.length, 1);
  p.state.id = 'chat-2';
  p.slash.draw();
  assert.equal(p.rows.hidden, true, 'Another chat shows none of them');
  await p.slash.handle('/foo');
  p.slash.clearRows('chat-2');
  assert.equal(p.rows.hidden, true);
  p.state.id = 'chat-1';
  p.slash.draw();
  assert.equal(p.rows.children.length, 1, 'The first chat still has its row');
  p.slash.clearRows('chat-1');
  assert.equal(p.rows.hidden, true);
});

// ── Every command ──

/** What a command needs from the gateway to run in the test. */
const ROUTES = {
  'GET /settings': {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws', revision: 'r1', browser: false, sections: {mlc: {chat_url: 'http://127.0.0.1:9/v1', chat_model: 'm'}}},
  'PUT /settings': (body) => ({...body, browser: body.browser ?? false}),
  'GET /sessions/chat-1/status': {parts: [{key: 'status.ctx', params: {used: '1.2k', limit: '8.0k'}}], steps: {active: true, step: 3, attempt: 1, phase: 'acting'}, context: {measured: 1200, limit: 8000}},
  'GET /sessions/chat-1': {grants: ['file_write']},
  'GET /sessions/chat-1/memory': {working: [{}, {}], working_max: 20, long_term: {enabled: true, backend: 'ollama'}},
  'POST /sessions/chat-1/retry': {},
  'POST /sessions/chat-1/undo': {},
  'DELETE /sessions/chat-1/grants': {},
  'DELETE /sessions/chat-1/grants/file_write': {},
};
const TURN = {id: 't1', turn: 1, userText: 'Plan a trip to Rome', answer: 'Here is a plan.', finishedAt: 100, startedAt: 90, status: 'completed', undone: false, outputTokens: 1500, items: [{kind: 'tool', name: 'file_read', status: 'success'}], result: {}};

test('every command of the terminal has a page action, and runs without reaching the model', async () => {
  assert.ok(LIST.length >= 30);
  const actions = page().slash.actions;
  for (const command of LIST) {
    assert.ok(command.action, `/${command.name} names its page action`);
    assert.equal(typeof actions[command.action], 'function', `/${command.name} -> ${command.action}`);
  }
  const argument = {copy: '', model: 'qwen3', backend: 'http://127.0.0.1:9/v1', side: 'Rome', title: 'Rome', expand: '', memory: '', approvals: 'ask', browser: 'on', verbose: 'on', thinking: 'on', stream: 'off', resume: 'Rome', sessions: 'Rome', help: 'all', status: 'all', task: 'plan a trip', answer: 't1 yes'};
  for (const command of LIST) {
    const p = page({routes: ROUTES, state: {turns: [TURN], main: false}});
    const line = `/${command.name}${argument[command.name] ? ` ${argument[command.name]}` : ''}`;
    const done = await p.slash.handle(line);
    assert.equal(done, true, line);
    assert.ok(!p.rows.children.some((row) => /Unknown command|Could not load/.test(text(row))), `${line}: ${rowTexts(p)}`);
    // Nothing goes to the model: only /task and /answer are lines a chat's own rules read.
    assert.deepEqual(p.sent, ['task', 'answer'].includes(command.name) ? [line] : [], line);
    assert.ok(!p.calls.some((call) => call.path.endsWith('/turns')), line);
  }
});

test('commands that only look answer in a local row', async () => {
  const p = page({routes: ROUTES, state: {turns: [TURN]}});
  await p.slash.handle('/status');
  assert.match(lastRow(p), /model qwen · oMLX\s+✓ answers/);
  assert.match(lastRow(p), /context ctx 1\.2k\/8\.0k/);
  assert.match(lastRow(p), /folder ~\/notes/);
  assert.match(lastRow(p), /chat “Trip” · 1 turn/);
  assert.match(lastRow(p), /approvals Approvals: Ask · grants: file_write/);
  assert.doesNotMatch(lastRow(p), /version/, 'The short status leaves the gateway rows out');
  await p.slash.handle('/status all');
  assert.match(lastRow(p), /version melo 1\.0/);
  assert.match(lastRow(p), /gateway 2 chats open, 1 running · up 1m/);
  await p.slash.handle('/usage');
  assert.match(lastRow(p), /1 turns · 1 tool calls · 1\.5k tokens written/);
  await p.slash.handle('/progress');
  assert.match(lastRow(p), /step 3 · attempt 1 · acting/);
  assert.match(lastRow(p), /context 1\.2k of 8\.0k/);
  await p.slash.handle('/mcp');
  assert.match(lastRow(p), /MCP servers are not part of the console yet/);
  await p.slash.handle('/memory window');
  assert.match(lastRow(p), /holds 2 of 20 messages · long-term memory: ollama/);
  await p.slash.handle('/exit');
  assert.match(lastRow(p), /Close this tab\. melo keeps running\./);
  assert.deepEqual(p.sent, []);
});

test('/help lists the everyday commands, every command with "all", or one command\'s details', async () => {
  const p = page();
  await p.slash.handle('/help');
  const shown = (row) => all(row, 'local-help').map((node) => named(node, 'slash-name').textContent);
  assert.deepEqual(shown(p.rows.children[0]), CATALOG.everyday.map((name) => `/${name}`));
  assert.match(lastRow(p), /\/help all lists every command\./);
  await p.slash.handle('/help all');
  assert.equal(shown(p.rows.children[1]).length, LIST.length);
  assert.equal(all(p.rows.children[1], 'local-table').length, 1, 'One table, so the columns line up');
  await p.slash.handle('/help exit');
  assert.match(lastRow(p), /^\/help exit \/exit Quit also \/quit, ctrl\+d During a turn: yes$/);
  await p.slash.handle('/help /approvals');
  assert.match(lastRow(p), /During a turn: only without arguments/);
  await p.slash.handle('/help nope');
  assert.match(lastRow(p), /Unknown command \/nope — \/help lists them/);
});

test('/new, /clear and /side open a new chat; /side names it and /clear forgets what was shown', async () => {
  const p = page();
  await p.slash.handle('/foo');
  await p.slash.handle('/new');
  await p.slash.handle('/side Rome trip');
  await p.slash.handle('/clear');
  assert.deepEqual(p.events.filter(([name]) => name === 'newChat'), [['newChat', ''], ['newChat', 'Rome trip'], ['newChat', '']]);
  assert.equal(p.rows.hidden, true, '/clear emptied the chat\'s rows');
  assert.equal(p.rows.children.length, 0);
});

test('/chats, /sessions and /resume pick a chat; /main goes back to the main chat', async () => {
  const p = page();
  await p.slash.handle('/chats');
  await p.slash.handle('/sessions Rome');
  await p.slash.handle('/resume');
  assert.deepEqual(p.events.filter(([name]) => name === 'pickChat'), [['pickChat', ''], ['pickChat', 'Rome'], ['pickChat', '']]);
  await p.slash.handle('/resume bbb');
  await p.slash.handle('/resume taxes');
  assert.deepEqual(p.events.filter(([name]) => name === 'openChat'), [['openChat', 'bbb222'], ['openChat', 'bbb222']]);
  await p.slash.handle('/resume trip');
  assert.match(lastRow(p), /2 chats match “trip”\. Pick one in the list\./);
  assert.deepEqual(moved(p, 'pickChat'), ['pickChat', 'trip']);
  await p.slash.handle('/resume paris');
  assert.match(lastRow(p), /No chat matches “paris”\./);
  await p.slash.handle('/main');
  assert.deepEqual(moved(p, 'openMain'), ['openMain']);
  const none = page({host: {openMain: () => false}});
  await none.slash.handle('/main');
  assert.match(lastRow(none), /This gateway has no main chat\./);
});

test('/task and /answer go to the open chat as the line its own rules read; a chat not sent yet has none to own', async () => {
  const main = page({state: {main: true}});
  await main.slash.handle('/task make slides');
  assert.deepEqual(main.sent, ['/task make slides']);
  const side = page();
  await side.slash.handle('/answer t1 use Postgres');
  assert.deepEqual(side.sent, ['/answer t1 use Postgres']);
  const fresh = page({state: {id: null}});
  await fresh.slash.handle('/task make slides');
  assert.deepEqual(fresh.sent, []);
  assert.match(lastRow(fresh), /Send this chat's first message first, then \/task …/);
});

test('/model and /backend save through the settings route, or open the model panel without a value', async () => {
  const p = page({routes: ROUTES});
  await p.slash.handle('/model');
  await p.slash.handle('/backend');
  assert.deepEqual(p.events.filter(([name]) => name === 'openSettings'), [['openSettings', 'model'], ['openSettings', 'model']]);
  await p.slash.handle('/model qwen3');
  assert.deepEqual(p.calls.at(-1), {method: 'PUT', path: '/settings', body: {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws', revision: 'r1', sections: {mlc: {chat_model: 'qwen3'}}}});
  assert.match(lastRow(p), /Model set to qwen3 \(saved\)\./);
  await p.slash.handle('/backend http://user:secret@127.0.0.1:9/v1');
  assert.deepEqual(p.calls.at(-1).body.sections, {mlc: {chat_url: 'http://user:secret@127.0.0.1:9/v1'}});
  assert.match(lastRow(p), /Model server set to http:\/\/127\.0\.0\.1:9\/v1 \(saved\)\./, 'The row shows no password');
  assert.equal(p.events.filter(([name]) => name === 'settingsSaved').length, 2, 'The page follows the saved settings');
  const cloud = page({routes: {...ROUTES, 'GET /settings': {...ROUTES['GET /settings'], chat_backend: 'openai', sections: {openai: {base_url: 'https://x', chat_model: 'm'}}}}});
  await cloud.slash.handle('/backend https://api.example/v1');
  assert.deepEqual(cloud.calls.at(-1).body.sections, {openai: {base_url: 'https://api.example/v1'}});
  const refused = page({routes: {...ROUTES, 'PUT /settings': Object.assign(new Error('chat_model: Must not be empty'), {status: 400})}});
  assert.equal(await refused.slash.handle('/model x'), true);
  assert.match(lastRow(refused), /Must not be empty/);
  assert.equal(named(refused.rows.children[0], 'local-text').dataset.tone, 'error');
});

test('/retry and /undo act on the newest finished turn', async () => {
  const p = page({routes: ROUTES, state: {turns: [{...TURN, id: 'old', turn: 1, undone: true}, TURN]}});
  await p.slash.handle('/retry');
  assert.deepEqual(p.calls.at(-1), {method: 'POST', path: '/sessions/chat-1/retry', body: {turn_id: 't1'}});
  assert.match(lastRow(p), /↻ Retrying: Plan a trip to Rome/);
  await p.slash.handle('/undo');
  assert.deepEqual(p.calls.at(-1), {method: 'POST', path: '/sessions/chat-1/undo', body: {turn_id: 't1'}});
  assert.match(lastRow(p), /Forgot the last turn\. Its file changes stay\./);
  const empty = page({routes: ROUTES});
  await empty.slash.handle('/retry');
  await empty.slash.handle('/undo');
  assert.deepEqual(empty.rows.children.map(text), ['/retry Nothing to retry.', '/undo Nothing to undo.']);
  const fresh = page({routes: ROUTES, state: {id: null, turns: [TURN]}});
  await fresh.slash.handle('/retry');
  assert.match(lastRow(fresh), /This chat has no messages yet\./);
});

test('/copy puts a stored answer on the clipboard', async () => {
  const second = {...TURN, id: 't2', turn: 2, answer: 'The second answer.'};
  const p = page({state: {turns: [TURN, second, {...TURN, id: 't3', turn: 3, answer: '', finishedAt: 0}]}});
  await p.slash.handle('/copy');
  await p.slash.handle('/copy 1');
  assert.deepEqual(p.events.filter(([name]) => name === 'copied'), [['copied', 'The second answer.'], ['copied', 'Here is a plan.']]);
  assert.match(lastRow(p), /Copied the answer of turn 1\./);
  await p.slash.handle('/copy 3');
  assert.match(lastRow(p), /Turn 3 has no answer to copy\./);
  await p.slash.handle('/copy x');
  assert.match(lastRow(p), /Use \/copy or \/copy TURN\./);
  const denied = page({state: {turns: [TURN]}, host: {writeText: async () => { throw new Error('denied'); }}});
  await denied.slash.handle('/copy');
  assert.match(lastRow(denied), /Clipboard unavailable/);
  const none = page();
  await none.slash.handle('/copy');
  assert.match(lastRow(none), /No answer to copy yet\./);
});

test('/title shows or sets the chat\'s name; before the chat exists the name waits for it', async () => {
  const p = page({state: {turns: [TURN]}});
  await p.slash.handle('/title');
  assert.match(lastRow(p), /This chat is called “Trip”\./);
  await p.slash.handle('/title Rome plans');
  assert.deepEqual(moved(p, 'setTitle'), ['setTitle', 'Rome plans']);
  assert.match(lastRow(p), /Renamed to “Rome plans”\./);
  const fresh = page({state: {id: null}});
  await fresh.slash.handle('/title Rome plans');
  assert.deepEqual(moved(fresh, 'setNewTitle'), ['setNewTitle', 'Rome plans']);
  assert.match(lastRow(fresh), /The new chat will be called “Rome plans”\./);
  const main = page({state: {main: true}});
  await main.slash.handle('/title x');
  assert.match(lastRow(main), /The main chat keeps its name\./);
});

test('/memory opens the Memory tab; /memory import leaves a hook for the import screens', async () => {
  const p = page({routes: ROUTES});
  await p.slash.handle('/memory');
  assert.deepEqual(moved(p, 'openMemory'), ['openMemory']);
  assert.equal(p.rows.children.length, 0);
  await p.slash.handle('/memory import');
  assert.match(lastRow(p), /Importing notes is not on this page yet\./);
  const ready = page({host: {memoryImport: () => [{text: 'Import opened.'}]}});
  await ready.slash.handle('/memory import');
  assert.match(lastRow(ready), /Import opened\./);
  await p.slash.handle('/memory nope');
  assert.match(lastRow(p), /Use \/memory, or \/memory window/);
  const developer = page({routes: ROUTES, state: {developer: true}});
  await developer.slash.handle('/memory window');
  assert.deepEqual(moved(developer, 'openDrawer'), ['openDrawer', 'memory']);
});

test('/approvals switches the chat\'s mode, forgets a grant, and shows the mode and the grants', async () => {
  const p = page({routes: ROUTES});
  await p.slash.handle('/approvals');
  assert.match(lastRow(p), /Approvals: Ask · grants: file_write/);
  await p.slash.handle('/approvals auto');
  assert.deepEqual(moved(p, 'setMode'), ['setMode', 'auto']);
  await p.slash.handle('/approvals forget file_write');
  assert.deepEqual(p.calls.at(-1), {method: 'DELETE', path: '/sessions/chat-1/grants/file_write', body: undefined});
  assert.match(lastRow(p), /Forgot the grant for file_write\./);
  await p.slash.handle('/approvals forget');
  assert.deepEqual(p.calls.at(-1).path, '/sessions/chat-1/grants');
  assert.match(lastRow(p), /Forgot every grant in this chat\./);
  await p.slash.handle('/approvals deny');
  assert.match(lastRow(p), /The console has Ask and Auto\. Deny is for the terminal\./);
  await p.slash.handle('/approvals maybe');
  assert.match(lastRow(p), /Use \/approvals ask, auto, or forget \[tool\]\./);
  const missing = page({routes: {...ROUTES, 'DELETE /sessions/chat-1/grants/nope': Object.assign(new Error('x'), {status: 404})}});
  await missing.slash.handle('/approvals forget nope');
  assert.match(lastRow(missing), /No grant for nope in this chat\./);
});

test('/browser shows whether melo may browse, and /browser on or off saves it through the settings route', async () => {
  const p = page({routes: ROUTES});
  await p.slash.handle('/browser');
  assert.match(lastRow(p), /Browsing is off\. \/browser on lets melo read web pages in Safari\./);
  await p.slash.handle('/browser on');
  assert.deepEqual(p.calls.at(-1).body, {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws', revision: 'r1', sections: {}, browser: true});
  assert.match(lastRow(p), /Browsing is on/);
  assert.equal(p.events.filter(([name]) => name === 'settingsSaved').length, 1);
  await p.slash.handle('/browser maybe');
  assert.match(lastRow(p), /Use \/browser, \/browser on or \/browser off\./);
});

test('/verbose, /thinking and /stream set how the page shows a chat', async () => {
  const p = page();
  await p.slash.handle('/verbose');
  assert.equal(p.prefs.verbose, true, 'Without a word it flips');
  assert.match(lastRow(p), /Verbose rendering on\./);
  await p.slash.handle('/verbose off');
  assert.equal(p.prefs.verbose, false);
  await p.slash.handle('/thinking on');
  assert.equal(p.prefs.thinking, true);
  await p.slash.handle('/stream');
  assert.equal(p.prefs.stream, false, 'Streaming starts on, so a bare /stream turns it off');
  assert.match(lastRow(p), /Streaming off\. A turn shows when it is complete\./);
  await p.slash.handle('/stream maybe');
  assert.match(lastRow(p), /Use \/stream on or \/stream off\./);
});

test('/expand opens a tool call whole, by turn and call number', async () => {
  const call = (name) => ({kind: 'tool', name, status: 'success', output: 'x'});
  const turns = [
    {...TURN, id: 't1', turn: 1, items: [call('file_read'), {kind: 'thinking', text: 'hm'}, call('shell')]},
    {...TURN, id: 't2', turn: 2, items: [call('web')]},
    {...TURN, id: 't9', turn: 9, undone: true, items: [call('gone')]},
  ];
  assert.deepEqual(toolCalls(turns).map((item) => item.ref), ['1.1', '1.2', '2.1']);
  const p = page({state: {turns}});
  await p.slash.handle('/expand');
  assert.equal(moved(p, 'reveal')[1].name, 'web', 'The newest call');
  await p.slash.handle('/expand 1.2');
  assert.equal(moved(p, 'reveal')[1].name, 'shell');
  assert.match(lastRow(p), /Opened call 1\.2 \(shell\) with its whole output\./);
  await p.slash.handle('/expand 1');
  assert.equal(moved(p, 'reveal')[1].name, 'web', 'A bare number is a call of the newest turn');
  await p.slash.handle('/expand 1.7');
  assert.match(lastRow(p), /Turn 1 has 2 tool calls\./);
  await p.slash.handle('/expand 4.1');
  assert.match(lastRow(p), /No tool calls yet\./);
  await p.slash.handle('/expand x');
  assert.match(lastRow(p), /Use \/expand or \/expand TURN\.CALL\./);
  const none = page();
  await none.slash.handle('/expand');
  assert.match(lastRow(none), /No tool calls yet\./);
});

test('/exit quits the app after a confirm, and in a browser tab says to close the tab', async () => {
  const app = page({host: {isApp: () => true, canQuit: () => true}});
  await app.slash.handle('/quit');
  const row = app.rows.children[0];
  assert.match(text(row), /Quit melo\? Quit Cancel/);
  const [yes, no] = all(row, 'local-choice');
  assert.equal(yes.className, 'local-choice primary');
  no.click();
  assert.deepEqual(app.events.filter(([name]) => name === 'quit'), [], 'Cancel quits nothing');
  yes.click();
  assert.deepEqual(app.events.filter(([name]) => name === 'quit'), [['quit']]);
  const old = page({host: {isApp: () => true, canQuit: () => false}});
  await old.slash.handle('/exit');
  assert.match(lastRow(old), /Quit melo from its menu bar item, or press ⌘Q\./);
});

test('the console takes no files yet, and says so for each of the three commands', async () => {
  const p = page();
  for (const line of ['/attach notes.txt', '/attachments', '/detach all']) {
    await p.slash.handle(line);
    assert.match(lastRow(p), /This page cannot take files yet\. Attach one in a terminal chat with \/attach PATH\./);
  }
  assert.deepEqual(p.sent, []);
});

test('a row is a command echo, then lines whose colors say what they mean', () => {
  const {doc} = fakeDoc();
  const row = rowNode(doc, {command: '/model qwen', lines: [{text: 'Saved.', tone: 'ok'}, {label: 'model', parts: [{text: 'qwen'}, {text: ' ✗ down', tone: 'error'}]}, {text: 'quiet', tone: 'muted'}]});
  assert.equal(row.attributes['aria-label'], 'Output of /model qwen');
  assert.equal(named(row, 'slash-name').textContent, '/model');
  assert.equal(named(row, 'slash-args').textContent, 'qwen');
  assert.deepEqual(all(row, 'local-text').map((node) => node.dataset.tone ?? ''), ['ok', '', 'error', 'muted']);
  assert.equal(named(row, 'local-label').textContent, 'model');
});

// ── In Chinese ──

test('in Chinese the menu, the rows and the help read in Chinese', async () => {
  await useLanguage('zh-Hans');
  try {
    const p = page();
    await type(p, '/mo');
    assert.equal(named(all(p.menu, 'slash-item')[0], 'slash-help').textContent, helpOf(LIST[0]));
    assert.match(helpOf(LIST[0]), /模型/);
    assert.match(text(p.menu), /↑ ↓ 选择 · Enter 或 Tab 选定 · Esc 关闭/);
    await p.slash.handle('/foo');
    assert.match(lastRow(p), /未知命令 \/foo — \/help 列出全部命令/);
    await type(p, '/zzz');
    assert.match(text(p.menu), /没有以 \/zzz 开头的命令/);
    const busy = page({state: {busy: true}});
    await busy.slash.handle('/new');
    assert.match(lastRow(busy), /一轮运行时不能用 \/new/);
    await p.slash.handle('/help exit');
    assert.match(lastRow(p), /一轮运行时：可用/);
  } finally {
    await useLanguage('en');
  }
});

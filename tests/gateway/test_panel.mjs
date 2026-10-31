/**
 * The panel beside a chat (panel.mjs) in a DOM double: Melo's header, the tabs and their keys, the day's
 * activity and its calm state, the panel following the open chat, the approvals folded under a row, the Workers
 * tab, a row opened to a task's live steps on a fake event stream, its actions, and the sheet a phone opens from
 * the face chip.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {headline, setupPanel, stepWords} from '../../support/gateway/static/panel.mjs';

function dom() {
  const doc = {activeElement: null, listeners: {}, visibilityState: 'visible',
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    removeEventListener() {}};
  class Element {
    constructor(tag) {
      Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, handlers: {}, dataset: {}, style: {},
        textContent: '', className: '', hidden: false, disabled: false, tabIndex: 0, parent: null, ownerDocument: doc});
      const classes = () => new Set(this.className.split(/\s+/).filter(Boolean));
      this.classList = {
        toggle: (name, on) => { const set = classes(); if (on ?? !set.has(name)) set.add(name); else set.delete(name); this.className = [...set].join(' '); },
        contains: (name) => classes().has(name),
      };
    }
    get firstChild() { return this.children[0] ?? null; }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    getAttribute(key) { return this.attributes[key] ?? null; }
    append(...nodes) { for (const item of nodes) { item.parent = this; this.children.push(item); } }
    prepend(...nodes) { for (const item of nodes) item.parent = this; this.children.unshift(...nodes); }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
    addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
    dispatch(type, event = {}) { for (const fn of this.handlers[type] ?? []) fn({target: this, preventDefault() {}, ...event}); }
    click() { if (!this.disabled) this.dispatch('click'); }
    focus() { doc.activeElement = this; }
    querySelectorAll(selector) {
      const tag = selector.toUpperCase();
      return this.children.flatMap((child) => [...(child.tagName === tag ? [child] : []), ...child.querySelectorAll(selector)]);
    }
    text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' ').replace(/\s+/g, ' ').trim(); }
    find(test) { if (test(this)) return this; for (const child of this.children) { const hit = child.find(test); if (hit) return hit; } return null; }
    findAll(test) { return [...(test(this) ? [this] : []), ...this.children.flatMap((child) => child.findAll(test))]; }
  }
  doc.createElement = (tag) => new Element(tag);
  doc.body = new Element('body');
  return {doc, Element};
}

class FakeSource {
  constructor(url) { this.url = url; this.closed = false; }
  close() { this.closed = true; }
  emit(frame) { this.onmessage?.({data: JSON.stringify(frame)}); }
}

const settle = () => new Promise((resolve) => setImmediate(resolve));
const health = (status = {}, extra = {}) => ({configured: true, model: {name: 'Qwen3.8 Flash', backend_label: 'oMLX'},
  model_server: {reachable: true}, status: {running: {chats: 0, tasks: 0, runs: 0, terminal: 0}, waiting: [], items: [], recent: null, ...status}, ...extra});
const row = (extra) => ({kind: 'task', id: 't1', title: 'Slides for the 3 pm meeting', state: 'done', activity: '', icon: 'done', result: '', at: 1000, ...extra});
const started = {id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'Make slides'}};
const tool = (id, call, type, data = {}) => ({id, turn: 1, ts: id, type, data: {step: 1, call_id: call, ...data}});

function harness({rows = [], turns = null, storage = null, failActivity = false, wide = null, openChat = '', developer = false} = {}) {
  const {doc, Element} = dom();
  const root = new Element('aside');
  const drawer = new Element('section');
  drawer.className = 'details-drawer';
  root.append(drawer);
  const chip = new Element('button');
  const backdrop = new Element('button');
  const chat = new Element('main');
  const h = {calls: [], sources: [], timers: [], rows, turns: turns ?? [started], doc, root, chip, backdrop, chat, drawer,
    open: openChat, scope: 'all', gate: {}, answerError: null};
  const api = async (path, options = {}) => {
    h.calls.push([options.method ?? 'GET', path, options.body]);
    if (path.startsWith('/activity')) {
      if (failActivity) throw Object.assign(new Error('down'), {status: 503});
      const id = new URL(path, 'http://x').searchParams.get('session_id') ?? '';
      await h.gate[id];
      return {since: 0, scope: id ? h.scope : 'all', rows: typeof h.rows === 'function' ? h.rows(id) : h.rows, more: false, unavailable: []};
    }
    if (path.startsWith('/approvals/')) {
      if (h.answerError) throw h.answerError;
      return {outcome: 'allowed'};
    }
    if (path.startsWith('/sessions/')) return {turns: [{turn: 1, status: 'running', events: h.turns}], has_earlier: false};
    if (path.startsWith('/tasks/')) return {task: {}};
    if (path === '/memory/records') return {on: h.memoryOn !== false, records: [], documents: 0, embedding_model: '', last_error: ''};
    if (path === '/memory/import/scan') return {batch_id: 'b1', sources: [], partial: false, home: '/h'};
    if (path.startsWith('/chores')) {
      const id = new URL(path, 'http://x').searchParams.get('session_id') ?? '';
      return {chores: h.chores?.(id) ?? [], checkin: {on: true, every_min: 30, next_at: 86400 * 400 + 1800, last_at: null}, scope: id ? 'chat' : 'all'};
    }
    throw new Error(`unexpected ${path}`);
  };
  h.panel = setupPanel({
    root, chip, backdrop, api, cover: () => [chat], chat: () => h.open, developer: () => developer, storage: storage ?? {getItem: () => null, setItem: () => {}},
    clock: () => 86400 * 1000 * 400,
    wide,
    live: {
      doc,
      open: (url) => { const source = new FakeSource(url); h.sources.push(source); return source; },
      schedule: (fn, ms) => { const timer = {fn, ms}; h.timers.push(timer); return timer; },
      cancel: () => {},
    },
  });
  h.body = () => root.find((node) => node.attributes.role === 'tabpanel');
  h.tabs = () => root.findAll((node) => node.attributes.role === 'tab');
  h.heads = () => root.findAll((node) => node.className === 'activity-head');
  h.activityCalls = () => h.calls.filter(([, path]) => path.startsWith('/activity')).map(([, path]) => path.replace(/since=\d+/, 'since=S'));
  h.click = (label) => root.find((node) => node.tagName === 'BUTTON' && node.textContent === label)?.click();
  return h;
}

test('a step reads as the server words it, in the present while it runs and the past once done', () => {
  // The same cases as tests/sessions/test_activity_words.py, present tense.
  const cases = [
    ['file_read', {path: '/w/talk/notes.md'}, 'Reading notes.md'],
    ['file_write', {path: 'slides.md'}, 'Writing slides.md'],
    ['shell_exec', {command: 'uv run pytest -q'}, 'Running the tests'],
    ['shell_exec', {command: 'npm test'}, 'Running the tests'],
    ['shell_exec', {command: 'cat pytest.ini'}, 'Running a command'],
    ['python_exec', {code: 'print(1)'}, 'Running Python'],
    ['browser_open', {url: 'https://www.example.com/a?b'}, 'Reading example.com'],
    ['browser_follow', {ref: 'e1'}, 'Following a link'],
    ['recall', {}, 'Remembering'],
    ['remember', {text: 'You prefer tea.', topic: 'preferences'}, 'Remembering'],
    ['mcp__files__file_read', {path: 'a/b.txt'}, 'Reading b.txt'],
    ['mcp__calendar__list_events', {}, 'Using a tool'],
  ];
  for (const [name, args, words] of cases) assert.equal(stepWords(name, args), words, name);
  assert.equal(stepWords('file_read', {path: 'notes.md'}, true), 'Read notes.md');
  assert.equal(stepWords('shell_exec', {command: 'cargo test'}, true), 'Ran the tests');
  assert.equal(stepWords('list_files', {}, true), 'Looked through the folder');
  assert.equal(stepWords('remember', {text: 'You prefer tea.'}, true), 'Remembered');
  const secret = 'sk-' + 'a1b2c3'.repeat(4);
  assert.ok(!stepWords('shell_exec', {command: `curl -H ${secret}`}).includes(secret));
});

test('the header says the model, what runs, a wait, or what to fix', () => {
  assert.deepEqual(headline(health()), {text: 'Connected · Qwen3.8 Flash on oMLX', mood: 'idle', fix: false, count: 0});
  const busy = headline(health({running: {chats: 1, tasks: 1, runs: 0, terminal: 0}}));
  assert.equal(busy.text, 'Working on 2 things');
  assert.equal(busy.mood, 'working');
  const waiting = headline(health({running: {chats: 1, tasks: 0, runs: 0, terminal: 0}, waiting: [{kind: 'chat', id: 'a'}]}));
  assert.deepEqual([waiting.text, waiting.mood], ['Waiting for you', 'waiting']);
  assert.deepEqual(headline(health({}, {model_server: {reachable: false}})), {text: 'Model server not reachable', mood: 'error', fix: true});
  assert.equal(headline(health({}, {configured: false})).text, 'No model set up yet');
  assert.equal(headline(health({recent: {outcome: 'done', ended_s_ago: 2}})).mood, 'done');
  assert.equal(headline(health({recent: {outcome: 'done', ended_s_ago: 30}})).mood, 'idle');
});

test('the portrait, the status line and the face chip follow health', async () => {
  const h = harness();
  await h.panel.update(health({running: {chats: 0, tasks: 2, runs: 0, terminal: 0}, waiting: [{kind: 'task', id: 't1'}]}));
  assert.equal(h.root.dataset.mood, 'waiting');
  const picture = h.root.find((node) => node.tagName === 'IMG');
  assert.equal(picture.getAttribute('src'), '/assets/akita/sit-waiting.png');
  assert.match(h.root.find((node) => node.attributes.role === 'status').text(), /Waiting for you/);
  assert.equal(h.chip.children[1].textContent, '2 · !');
  assert.equal(h.chip.attributes['aria-label'], 'Activity: Waiting for you');
  await h.panel.update(health());
  assert.equal(picture.getAttribute('src'), '/assets/akita/sit-idle.png');
  assert.ok(h.root.find((node) => node.className.includes('blinks')), 'only the resting portrait blinks');
});

test('calm: nothing running, and today newest first with each kind and result', async () => {
  const today = [
    row({id: 'c1', kind: 'chat', title: 'Read the readme', icon: 'read', result: 'One script.', at: 2000}),
    row({id: 'r1', kind: 'run', title: 'Check the build', icon: 'run', result: 'All 6 tests passed.', at: 1500}),
  ];
  const h = harness({rows: today});
  await h.panel.update(health());
  const text = h.body().text();
  assert.match(text, /Now Nothing running Today/);
  assert.match(text, /Read the readme One script\..*Check the build All 6 tests passed\./);
  const icons = h.root.findAll((node) => node.className === 'activity-icon');
  assert.deepEqual(icons.map((icon) => [icon.dataset.icon, icon.textContent]), [['read', '▤'], ['run', '>_']]);
  assert.deepEqual(h.root.findAll((node) => node.tagName === 'A'), [], 'worker history is reached from the Workers tab');
  assert.equal(h.calls.filter(([, path]) => path.startsWith('/activity')).length, 1);
  // The same health again reads nothing; a change in what runs reads the day again.
  await h.panel.update(health());
  assert.equal(h.calls.filter(([, path]) => path.startsWith('/activity')).length, 1);
  await h.panel.update(health({items: [{kind: 'chat', id: 'c1', title: 'x', activity: 'Thinking', icon: 'think'}]}));
  assert.equal(h.calls.filter(([, path]) => path.startsWith('/activity')).length, 2);
});

test('a long day folds behind "Earlier today", and an unreadable day says so and asks again', async () => {
  const many = Array.from({length: 11}, (_, n) => row({id: `t${n}`, title: `Task ${n}`, at: 1000 - n}));
  const h = harness({rows: many});
  await h.panel.update(health());
  assert.equal(h.heads().length, 8);
  h.root.find((node) => node.textContent === 'Earlier today (3)').click();
  assert.equal(h.heads().length, 11);
  const down = harness({failActivity: true});
  await down.panel.update(health());
  assert.match(down.body().text(), /Could not read today/);
  await down.panel.update(health());
  assert.equal(down.calls.filter(([, path]) => path.startsWith('/activity')).length, 2, 'a failed read is tried at the next poll');
});

test('the tabs are a tablist: arrows, Home and End move and select, and the choice is remembered', async () => {
  const saved = new Map();
  const storage = {getItem: (key) => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value)};
  const h = harness({storage});
  await h.panel.update(health());
  const tabs = h.tabs();
  assert.deepEqual(tabs.map((tab) => tab.text()), ['≡ Activity', '⚙ Workers', '◷ Scheduled', '◎ Memory']);
  assert.deepEqual(tabs.map((tab) => [tab.attributes['aria-selected'], tab.tabIndex]), [['true', 0], ['false', -1], ['false', -1], ['false', -1]]);
  const list = h.root.find((node) => node.attributes.role === 'tablist');
  list.dispatch('keydown', {key: 'ArrowRight'});
  assert.equal(h.doc.activeElement, tabs[1]);
  assert.equal(tabs[1].attributes['aria-selected'], 'true');
  assert.match(h.body().text(), /No workers yet today/);
  assert.equal(h.body().attributes['aria-labelledby'], 'panel-tab-workers');
  list.dispatch('keydown', {key: 'End'});
  assert.equal(h.doc.activeElement, tabs[3]);
  await settle();
  assert.match(h.body().text(), /Nothing remembered yet/, 'the Memory tab lists what melo keeps (memory-tab.mjs has its own tests)');
  list.dispatch('keydown', {key: 'ArrowRight'});
  assert.equal(h.doc.activeElement, tabs[0], 'the arrows wrap around');
  list.dispatch('keydown', {key: 'ArrowLeft'});
  list.dispatch('keydown', {key: 'Home'});
  assert.equal(tabs[0].attributes['aria-selected'], 'true');
  tabs[2].click();
  assert.equal(saved.get('localmelo-panel-tab'), 'scheduled');
  const again = harness({storage});
  assert.equal(again.tabs()[2].attributes['aria-selected'], 'true', 'a reload opens the last tab');
});

const ask = (id, what, answer, at = 1000) => ({id, what, answer, at});
const task = (extra) => row({kind: 'task', id: 't1', title: 'Slides for the 3 pm meeting', ...extra});

test('the panel follows the open chat: its own rows, then the day for the main chat or none', async () => {
  const mine = [row({id: 'c1', kind: 'chat', title: 'Slides chat', state: 'waiting', activity: 'Waiting for your approval', icon: 'wait'})];
  const day = [...mine, row({id: 'c2', kind: 'chat', title: 'Taxes chat', result: 'Filed.'}), row({id: 'r1', kind: 'run', title: 'Check the build'})];
  const h = harness({rows: (id) => (id === 'c1' ? mine : day)});
  // The console as a whole is busy with another chat's approval.
  const busy = health({running: {chats: 1, tasks: 0, runs: 0, terminal: 0}, waiting: [{kind: 'chat', id: 'c9'}]});
  await h.panel.update(busy);
  assert.deepEqual(h.activityCalls(), ['/activity?since=S']);
  assert.match(h.body().text(), /Slides chat.*Taxes chat.*Check the build/);
  assert.equal(h.root.dataset.mood, 'waiting', 'with no chat open the header is the whole console');
  // A side chat opens: the old rows go at once, the chat's own are asked for, and the header reads them.
  h.open = 'c1';
  h.scope = 'chat';
  const moved = h.panel.follow();
  assert.match(h.body().text(), /Loading today/, 'another chat\'s rows are never shown for this one');
  await moved;
  assert.deepEqual(h.activityCalls().at(-1), '/activity?since=S&session_id=c1');
  assert.doesNotMatch(h.body().text(), /Taxes chat|Check the build/);
  assert.match(h.body().text(), /Slides chat/);
  const said = (hh) => hh.root.find((node) => node.attributes.role === 'status').text();
  assert.match(said(h), /Waiting for you/, 'its state line follows the chat: it waits');
  assert.deepEqual([h.chip.attributes['aria-label'], h.chip.children[1].textContent], ['Activity: Waiting for you', '!'], 'so does the phone sheet\'s face chip');
  mine[0] = row({id: 'c1', kind: 'chat', title: 'Slides chat', result: 'Done.'});
  await h.panel.refresh();
  assert.match(said(h), /Connected/, 'the chat is calm although the console had something running');
  // The main chat, or no chat, is the whole day again; a health poll notices the change without a hash event.
  h.open = '';
  h.scope = 'all';
  await h.panel.update(busy);
  assert.equal(h.activityCalls().at(-1), '/activity?since=S');
  assert.match(h.body().text(), /Taxes chat.*Check the build/);
  assert.match(said(h), /Waiting for you/);
});

test('a read for the chat just left never lands on the chat now open', async () => {
  let release;
  const h = harness({rows: (id) => [row({id: id || 'all', kind: 'chat', title: id ? `Chat ${id}` : 'Everything'})]});
  h.gate.a = new Promise((resolve) => { release = resolve; });
  h.open = 'a';
  h.scope = 'chat';
  const slow = h.panel.follow();
  h.open = 'b';
  await h.panel.follow();
  assert.match(h.body().text(), /Chat b/);
  release();
  await slow;
  assert.match(h.body().text(), /Chat b/);
  assert.doesNotMatch(h.body().text(), /Chat a/);
});

test('approvals fold under the row that asked: counted, opened to each answer, never a tab', async () => {
  const asked = [ask('q1', 'Write slides.md', 'allowed once', 3600), ask('q2', 'Run the tests', 'allowed for this chat', 3700),
    ask('q3', 'Run a command', 'denied', 3800), ask('q4', 'Write notes.md', 'no answer', 3900)];
  const h = harness({rows: [task({approvals: asked}), row({id: 'c1', kind: 'chat', title: 'Plain', approvals: [ask('p1', 'Run Python', 'allowed once')]}),
    row({id: 'c2', kind: 'chat', title: 'Quiet', approvals: []})]});
  await h.panel.update(health());
  assert.ok(!h.tabs().some((tab) => /Approvals/.test(tab.text())));
  const folds = () => h.root.findAll((node) => node.className === 'approval-toggle');
  assert.deepEqual(folds().map((fold) => [fold.text(), fold.attributes['aria-expanded']]), [['4 approvals', 'false'], ['1 approval', 'false']]);
  assert.equal(h.root.findAll((node) => node.className === 'approval-list').length, 0, 'folded: no request is listed');
  folds()[0].click();
  const items = () => h.root.findAll((node) => node.className === 'approval-item').map((item) => item.text().replace(/\d+:\d+\s?[AP]M/, 'T'));
  assert.deepEqual(items(), ['Write slides.md T Allowed once', 'Run the tests T Allowed for this chat', 'Run a command T Denied',
    'Write notes.md T No answer in time']);
  assert.equal(h.root.findAll((node) => node.className === 'approval-choices').length, 0, 'an answered request offers nothing');
  assert.deepEqual(h.root.findAll((node) => node.className === 'approval-item').map((item) => item.dataset.answer), ['allowed', 'allowed', 'denied', 'other']);
  folds()[0].click();
  assert.equal(h.root.findAll((node) => node.className === 'approval-list').length, 0);
  // The fold and the row open independently, and the fold survives the list being read again.
  folds()[0].click();
  await h.panel.refresh();
  assert.equal(items().length, 4);
});

test('a waiting approval is answered from its fold with the calls the chat card makes', async () => {
  const waiting = [ask('q1', 'Write slides.md', 'allowed once'), ask('q2', 'Run a command', 'waiting')];
  const h = harness({rows: [row({id: 'c9', kind: 'chat', title: 'Add the date', state: 'waiting', activity: 'Waiting for your approval', icon: 'wait', approvals: waiting})]});
  await h.panel.update(health());
  const fold = () => h.root.find((node) => node.className === 'approval-toggle');
  assert.equal(fold().text(), '2 approvals, 1 waiting');
  assert.equal(h.root.find((node) => node.className === 'approval-fold').dataset.waiting, 'true');
  fold().click();
  const choices = () => h.root.find((node) => node.className === 'approval-choices');
  assert.deepEqual(choices().children.map((button) => button.textContent), ['Allow once', 'Deny', 'Review']);
  assert.equal(choices().children[2].href, '#/s/c9', 'the full card is one link away');
  h.rows = [row({id: 'c9', kind: 'chat', title: 'Add the date', approvals: [waiting[0], ask('q2', 'Run a command', 'allowed once')]})];
  h.click('Allow once');
  await settle();
  assert.deepEqual(h.calls.find(([, path]) => path.startsWith('/approvals/')), ['GET', '/approvals/q2', {choice: 'once'}]);
  assert.equal(fold().text(), '2 approvals', 'the row is read again and nothing waits');
  assert.equal(h.root.findAll((node) => node.className === 'approval-choices').length, 0);
  // Deny, and an answer that came too late.
  h.rows = [row({id: 'c9', kind: 'chat', title: 'Add the date', state: 'waiting', approvals: [ask('q3', 'Write a.md', 'waiting')]})];
  await h.panel.refresh();
  h.answerError = Object.assign(new Error('This approval was answered already.'), {status: 409});
  h.click('Deny');
  await settle();
  assert.deepEqual(h.calls.filter(([, path]) => path === '/approvals/q3').map(([, , body]) => body), [{choice: 'deny'}]);
  assert.match(h.body().text(), /answered already/);
});

test('a request of a worker run or a terminal chat is listed but answered where it runs', async () => {
  const h = harness({rows: [row({id: 'r1', kind: 'run', title: 'Check the build', state: 'waiting', activity: 'Waiting for your approval', icon: 'wait',
    approvals: [ask('q1', 'Run the tests', 'waiting')]})]});
  await h.panel.update(health());
  h.root.find((node) => node.className === 'approval-toggle').click();
  assert.match(h.body().text(), /Run the tests.*Waiting for you/);
  const choices = h.root.find((node) => node.className === 'approval-choices');
  assert.deepEqual(choices.children.map((node) => node.textContent), ['Review'], 'no answer buttons: the run asks in its own terminal');
  assert.equal(choices.children[0].href, '#/workers/r1');
});

test('Workers lists the tasks and runs of the scope, running first, each with its state in words', async () => {
  const day = [
    task({id: 't1', title: 'Slides', state: 'running', activity: 'Writing slides.md', icon: 'write', level: 2, at: 4000}),
    task({id: 't2', title: 'Taxes', state: 'waiting', activity: 'Waiting for your approval', icon: 'wait', at: 3000}),
    row({id: 'r1', kind: 'run', title: 'Check the build', state: 'done', result: 'All 6 tests passed.', icon: 'run', at: 2000}),
    task({id: 't3', title: 'Trip plan', state: 'failed', result: 'No disk.', icon: 'failed', at: 1500}),
    row({id: 'c1', kind: 'chat', title: 'A chat', result: 'Hello.', at: 5000}),
    row({id: 'x1', kind: 'terminal', title: 'Terminal talk', result: 'Hi.', at: 4500}),
  ];
  const h = harness({rows: day});
  await h.panel.update(health());
  h.tabs()[1].click();
  const titles = () => h.heads().map((head) => head.find((node) => node.tagName === 'STRONG').textContent);
  assert.deepEqual(titles(), ['Slides', 'Taxes', 'Check the build', 'Trip plan'], 'no chat, no terminal chat; running before done');
  const lines = h.root.findAll((node) => node.className === 'activity-line').map((line) => line.textContent);
  assert.deepEqual(lines, ['Working · Writing slides.md', 'Waiting for your approval', 'Done · All 6 tests passed.', 'Failed · No disk.']);
  assert.doesNotMatch(h.heads()[0].text(), /\d:\d\d/, 'a worker that runs shows no time, its row is live');
  assert.match(h.heads()[2].text(), /\d:\d\d/, 'a finished one shows when');
  assert.deepEqual(h.root.findAll((node) => node.tagName === 'A').map((link) => link.href), ['#/tasks', '#/workers']);
  // A worker opens to its live steps and its actions, as an Activity row does.
  h.heads()[0].click();
  await settle();
  assert.equal(h.sources[0].url.startsWith('/api/sessions/t1/events'), true);
  const actions = h.root.find((node) => node.className === 'activity-actions');
  assert.deepEqual(actions.children.map((node) => node.tagName), ['BUTTON', 'BUTTON', 'SELECT', 'A']);
  // Leaving for a tab that lists no rows stops following it.
  h.tabs()[3].click();
  assert.equal(h.sources[0].closed, true);
});

test('a task names its folder in Activity and in Workers; a row without one adds nothing', async () => {
  const h = harness({rows: [
    task({id: 't1', title: 'Sort photos', state: 'failed', result: 'Could not use its folder', icon: 'failed', at: 2000, folder: '/Volumes/Photos'}),
    task({id: 't2', title: 'Notes', state: 'done', result: 'Done.', at: 1000, folder: '~/notes'}),
    row({id: 'c1', kind: 'chat', title: 'A chat', result: 'Hello.', at: 500}),
  ]});
  await h.panel.update(health());
  const folders = () => h.root.findAll((node) => node.className === 'activity-folder').map((node) => node.textContent);
  assert.deepEqual(folders(), ['In /Volumes/Photos', 'In ~/notes']);
  h.tabs()[1].click();
  assert.deepEqual(folders(), ['In /Volumes/Photos', 'In ~/notes']);
});

test('a task a side chat started says so in the whole day\'s list', async () => {
  const h = harness({rows: [
    task({id: 't1', title: 'Find hotels', state: 'running', at: 2000, from: 'Trip plan'}),
    task({id: 't2', title: 'Book a table', state: 'done', result: 'Done.', at: 1000}),
  ]});
  await h.panel.update(health());
  const chips = () => h.root.findAll((node) => node.className === 'activity-folder activity-from').map((node) => node.textContent);
  assert.deepEqual(chips(), ['From Trip plan']);
});

test('a held worker says why in the server\'s words, a plain hold keeps the state words', async () => {
  const day = [
    task({id: 't1', title: 'Slides', state: 'held', activity: 'Waiting for a higher-priority task', icon: 'held', at: 4000}),
    task({id: 't2', title: 'Taxes', state: 'held', activity: '', icon: 'held', at: 3000}),
  ];
  const h = harness({rows: day});
  await h.panel.update(health());
  h.tabs()[1].click();
  const lines = h.root.findAll((node) => node.className === 'activity-line').map((line) => line.textContent);
  assert.deepEqual(lines, ['Waiting for a higher-priority task', 'Waiting for the model']);
});

test('Workers says when workers appear: for a side chat and for the main chat, and keeps its links in developer mode', async () => {
  const chats = [row({id: 'c1', kind: 'chat', title: 'A chat', result: 'Hello.'})];
  const side = harness({rows: chats, openChat: 'c1'});
  side.scope = 'chat';
  await side.panel.update(health());
  side.tabs()[1].click();
  assert.match(side.body().text(), /This chat has not started any workers\. When it hands a task to one, the worker is listed here/);
  const all = harness({rows: chats});
  await all.panel.update(health());
  all.tabs()[1].click();
  assert.match(all.body().text(), /No workers yet today\. When Melo hands a task to a worker, or you run one yourself, it is listed here/);
  assert.deepEqual(all.root.findAll((node) => node.tagName === 'A'), [], 'no worker, no history link');
  const dev = harness({rows: chats, developer: true});
  await dev.panel.update(health());
  dev.tabs()[1].click();
  assert.deepEqual(dev.root.findAll((node) => node.tagName === 'A').map((link) => link.href), ['#/tasks', '#/workers']);
});

test('a running task opens to its live steps, which follow the task stream until it ends', async () => {
  const live = row({state: 'running', activity: 'Reading notes.md', icon: 'read', level: 2, at: 900});
  const h = harness({rows: [live], turns: [started, tool(2, 'c1', 'tool.started', {name: 'file_read', arguments: {path: 'notes.md'}})]});
  await h.panel.update(health({running: {chats: 0, tasks: 1, runs: 0, terminal: 0}}));
  const [head] = h.heads();
  assert.equal(head.attributes['aria-expanded'], 'false');
  assert.match(head.text(), /Slides for the 3 pm meeting Reading notes\.md/);
  head.click();
  await settle();
  assert.equal(h.heads()[0].attributes['aria-expanded'], 'true');
  const steps = () => h.root.findAll((node) => node.className === 'activity-step').map((step) => [step.dataset.state, step.text()]);
  assert.deepEqual(steps(), [['running', '● Reading notes.md now']]);
  assert.equal(h.sources.length, 1);
  assert.equal(h.sources[0].url, '/api/sessions/t1/events?after=2');
  assert.deepEqual(h.calls.filter(([, path]) => path.includes('/turns')).map(([, path]) => path),
    ['/sessions/t1/turns?limit=1&events=true'], 'only the newest turn is read');
  h.sources[0].emit(tool(3, 'c1', 'tool.finished', {name: 'file_read', status: 'success', output: 'Meeting notes\n- one\n- two\n- three', duration_s: 0.2}));
  h.sources[0].emit(tool(4, 'c2', 'tool.started', {name: 'file_write', arguments: {path: 'slides.md', content: 'x'}}));
  h.timers.at(-1).fn();
  assert.deepEqual(steps(), [['done', '✓ Read notes.md 0.2s'], ['running', '● Writing slides.md now']]);
  assert.match(h.root.find((node) => node.className === 'activity-output').textContent, /- one\n- two\n- three/);
  // The actions are the terminal's: Pause, Stop and the priority as P-codes with their meaning, and the link to the task.
  const actions = h.root.find((node) => node.className === 'activity-actions');
  assert.deepEqual(actions.children.map((node) => node.tagName), ['BUTTON', 'BUTTON', 'SELECT', 'A']);
  assert.deepEqual(actions.children[2].children.map((option) => option.textContent), [
    'P0 · runs first; others wait', 'P1 · today', 'P2 · normal', 'P3 · when there is time', 'P4 · only when the computer is idle',
  ]);
  assert.equal(actions.children[3].href, '#/tasks/t1');
  actions.children[0].click();
  await settle();
  assert.deepEqual(h.calls.find(([method]) => method === 'PATCH'), ['PATCH', '/tasks/t1', {state: 'pause'}]);
  // Once the day says it ended, the stream closes and the session is read once more.
  h.rows = [row({state: 'done', icon: 'write', result: 'Four slides.'})];
  h.turns = [started, tool(2, 'c1', 'tool.started', {name: 'file_read', arguments: {path: 'notes.md'}}),
    tool(3, 'c1', 'tool.finished', {name: 'file_read', status: 'success', output: 'x', duration_s: 0.2})];
  await h.panel.refresh();
  await settle();
  assert.equal(h.sources[0].closed, true);
  assert.deepEqual(steps(), [['done', '✓ Read notes.md 0.2s']]);
  assert.equal(h.root.find((node) => node.className === 'activity-actions').children.length, 1, 'a finished task offers only its link');
});

test('a task that ran out of steps reads Stopped with what is left, and its row offers Continue', async () => {
  const stopped = row({state: 'stopped', icon: 'stopped', result: 'Remaining: run the tests and commit.', can_continue: true});
  const failed = row({id: 't2', title: 'Broke', state: 'failed', icon: 'failed', result: 'no disk'});
  const h = harness({rows: [stopped, failed]});
  await h.panel.update(health());
  h.tabs()[1].click();
  const lines = h.root.findAll((node) => node.className === 'activity-line').map((node) => node.textContent);
  assert.deepEqual(lines, ['Stopped · Remaining: run the tests and commit.', 'Failed · no disk']);
  h.heads()[0].click();
  await settle();
  const actions = h.root.find((node) => node.className === 'activity-actions');
  assert.deepEqual(actions.children.map((node) => node.textContent), ['Continue', 'Open task']);
  actions.children[0].click();
  await settle();
  assert.deepEqual(h.calls.find(([, path]) => path.endsWith('/continue')), ['POST', '/tasks/t1/continue', undefined]);
  // A real failure has no Continue.
  h.heads()[1].click();
  await settle();
  assert.deepEqual(h.root.find((node) => node.className === 'activity-actions').children.map((node) => node.textContent), ['Open task']);
});

test('a task code found not finished reads so in amber, and its row offers Finish it', async () => {
  const draft = row({state: 'done', icon: 'write', result: 'Fixed it.', can_continue: true,
    ending: {state: 'draft', reasons: ['tests_failed'], question: '', tests: []}});
  const ready = row({id: 't2', title: 'Notes', state: 'done', result: 'Wrote notes.', ending: {state: 'ready', reasons: [], tests: []}});
  const h = harness({rows: [draft, ready]});
  await h.panel.update(health());
  const lines = h.root.findAll((node) => node.className === 'activity-line').map((node) => node.textContent);
  assert.deepEqual(lines, ['Done, not finished · The last test run failed', 'Wrote notes.'], 'a ready task keeps its result');
  assert.equal(h.root.find((node) => node.className === 'activity-row').dataset.ending, 'draft');
  h.tabs()[1].click();
  const workers = h.root.findAll((node) => node.className === 'activity-line').map((node) => node.textContent);
  assert.deepEqual(workers, ['Done, not finished · The last test run failed', 'Ready']);
  h.heads()[0].click();
  await settle();
  const actions = h.root.find((node) => node.className === 'activity-actions');
  assert.deepEqual(actions.children.map((node) => node.textContent), ['Finish it: run the tests', 'Open task']);
  actions.children[0].click();
  await settle();
  assert.deepEqual(h.calls.find(([, path]) => path.endsWith('/continue')), ['POST', '/tasks/t1/continue', undefined]);
});

test('a task held on an approval says so in the row, a slot wait keeps its words', async () => {
  const asks = row({state: 'held', icon: 'wait', activity: 'Needs your approval'});
  const slot = row({id: 't2', title: 'Slides', state: 'held', icon: 'held', activity: 'Waiting for a free model slot'});
  const h = harness({rows: [asks, slot]});
  await h.panel.update(health());
  assert.deepEqual(h.root.findAll((node) => node.className === 'activity-line').map((node) => node.textContent),
    ['Needs your approval', 'Waiting for a free model slot']);
});

test('between calls the open row ends with what it is doing; a waiting row offers Review', async () => {
  const thinking = row({state: 'running', activity: 'Thinking', icon: 'think'});
  const waiting = row({id: 'c9', kind: 'chat', title: 'Add the date', state: 'waiting', activity: 'Waiting for your approval', icon: 'wait'});
  const h = harness({rows: [thinking, waiting], turns: [started, tool(2, 'c1', 'tool.started', {name: 'list_files', arguments: {}}),
    tool(3, 'c1', 'tool.finished', {name: 'list_files', status: 'success', output: '', duration_s: 0.1})]});
  await h.panel.update(health());
  h.heads()[0].click();
  await settle();
  assert.deepEqual(h.root.findAll((node) => node.className === 'activity-step').map((step) => step.text()),
    ['✓ Looked through the folder 0.1s', '● Thinking now']);
  h.heads()[1].click();
  await settle();
  assert.equal(h.heads()[0].attributes['aria-expanded'], 'false', 'one row is open at a time');
  const link = h.root.find((node) => node.className === 'activity-open');
  assert.deepEqual([link.textContent, link.href], ['Review', '#/s/c9']);
});

test('the phone sheet opens from the chip, covers the page, and closes on the backdrop, the handle or a drag', async () => {
  const h = harness();
  assert.equal(h.chip.hidden, false);
  h.chip.click();
  assert.equal(h.root.classList.contains('sheet-open'), true);
  assert.deepEqual([h.backdrop.hidden, h.chip.attributes['aria-expanded'], h.chat.inert], [false, 'true', true]);
  const handle = h.root.find((node) => node.className === 'panel-handle');
  assert.equal(h.doc.activeElement, handle);
  assert.equal(h.panel.closeSheet(), true);
  assert.deepEqual([h.root.classList.contains('sheet-open'), h.backdrop.hidden, h.chat.inert], [false, true, false]);
  assert.equal(h.doc.activeElement, h.chip, 'focus returns to the chip');
  assert.equal(h.panel.closeSheet(), false, 'Escape goes on to the next thing when the sheet is closed');
  h.chip.click();
  h.backdrop.click();
  assert.equal(h.root.classList.contains('sheet-open'), false);
  h.chip.click();
  handle.dispatch('pointerdown', {clientY: 100, pointerId: 1});
  handle.dispatch('pointermove', {clientY: 130});
  assert.equal(h.root.style.transform, 'translateY(30px)');
  handle.dispatch('pointerup');
  assert.equal(h.root.classList.contains('sheet-open'), true, 'a short drag springs back');
  handle.dispatch('pointerdown', {clientY: 100, pointerId: 1});
  handle.dispatch('pointermove', {clientY: 200});
  handle.dispatch('pointerup');
  assert.equal(h.root.classList.contains('sheet-open'), false);
  assert.equal(h.root.style.transform, '');
});

test('Details stays in the panel: its section is kept after the header and the tabs', () => {
  const h = harness();
  assert.equal(h.root.children.at(-1), h.drawer);
  assert.deepEqual(h.root.children.slice(0, 3).map((node) => node.className), ['panel-head', 'panel-tabs', 'panel-body']);
});

test("a window grown past the sheet's width closes the sheet, so the page is not left covered", () => {
  const wide = {matches: false, listeners: [], addEventListener(type, fn) { this.listeners.push(fn); }};
  const h = harness({wide});
  h.chip.click();
  assert.equal(h.chat.inert, true);
  wide.matches = true;
  for (const fn of wide.listeners) fn();
  assert.deepEqual([h.root.classList.contains('sheet-open'), h.chat.inert, h.backdrop.hidden], [false, false, true]);
});

test("an edit open on Scheduled does not follow the panel to another chat", async () => {
  const sample = (id, title) => ({id, title, request: title, schedule: {kind: 'daily', at: '09:00'}, when: 'every day at 9:00', level: 2, level_word: 'Normal',
    enabled: true, revision: 1, next_run: 86400 * 400 + 7200, last_run: null, last: null});
  const h = harness({openChat: 'side-1'});
  h.chores = (id) => (id === 'side-2' ? [sample('b', 'Second chore')] : [sample('a', 'First chore')]);
  await h.panel.update(health());
  h.tabs()[2].click();
  await settle();
  h.body().find((node) => node.className === 'sched-head').click();
  h.body().find((node) => node.attributes?.['aria-label'] === 'Edit time: First chore').click();
  assert.ok(h.body().find((node) => node.className === 'sched-form'), 'the first chat\'s chore is being edited');
  h.open = 'side-2';
  await h.panel.follow();
  await settle();
  assert.match(h.body().text(), /Second chore/);
  assert.ok(!h.body().text().includes('First chore'));
  assert.equal(h.body().find((node) => node.className === 'sched-form'), null, 'the editor did not stay');
});

test('Scheduled lists the check-in and the chores of the open chat, and reads them again as the panel refreshes', async () => {
  const sample = (id, title) => ({id, title, request: title, schedule: {kind: 'daily', at: '09:00'}, when: 'every day at 9:00', level: 2, level_word: 'Normal',
    enabled: true, revision: 1, next_run: 86400 * 400 + 7200, last_run: null, last: null});
  const h = harness();
  h.chores = (id) => (id ? [sample('b', 'Side chore')] : [sample('a', 'Main chore')]);
  await h.panel.update(health());
  h.tabs()[2].click();
  await settle();
  assert.match(h.body().text(), /Check-in/);
  assert.match(h.body().text(), /Main chore every day at 9:00/);
  assert.deepEqual(h.calls.filter(([, path]) => path.startsWith('/chores')).map(([, path]) => path), ['/chores']);
  // The panel follows the chat: the side chat's own chores replace the main chat's.
  h.open = 'side-1';
  await h.panel.follow();
  await settle();
  assert.match(h.body().text(), /Side chore/);
  assert.ok(!h.body().text().includes('Main chore'));
  assert.deepEqual(h.calls.filter(([, path]) => path.startsWith('/chores')).at(-1), ['GET', '/chores?session_id=side-1', undefined]);
  // Another tab drops it; back on Scheduled it reads the list again.
  h.tabs()[0].click();
  h.tabs()[2].click();
  await settle();
  assert.equal(h.calls.filter(([, path]) => path.startsWith('/chores')).length, 3);
});

test('lookForNotes opens the Memory tab\'s "Look for notes" (the console\'s /memory import), and says no when memory is off', async () => {
  const h = harness();
  h.panel.select('memory');
  assert.equal(await h.panel.lookForNotes(), true);
  assert.ok(h.calls.some(([method, path]) => method === 'GET' && path === '/memory/import/scan'), 'the files were looked at');
  assert.match(h.body().text(), /From other assistants.*No notes found on this Mac\./);
  const off = harness();
  off.memoryOn = false;
  off.panel.select('memory');
  assert.equal(await off.panel.lookForNotes(), false);
  assert.ok(!off.calls.some(([, path]) => path === '/memory/import/scan'), 'with memory off nothing is scanned');
});


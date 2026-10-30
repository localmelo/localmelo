/**
 * The panel beside a chat (panel.mjs) in a DOM double: Melo's header, the tabs and their keys, the day's
 * activity and its calm state, a row opened to a task's live steps on a fake event stream, its actions, and the
 * sheet a phone opens from the face chip.
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

function harness({rows = [], turns = null, storage = null, failActivity = false, wide = null} = {}) {
  const {doc, Element} = dom();
  const root = new Element('aside');
  const drawer = new Element('section');
  drawer.className = 'details-drawer';
  root.append(drawer);
  const chip = new Element('button');
  const backdrop = new Element('button');
  const chat = new Element('main');
  const h = {calls: [], sources: [], timers: [], rows, turns: turns ?? [started], doc, root, chip, backdrop, chat, drawer};
  const api = async (path, options = {}) => {
    h.calls.push([options.method ?? 'GET', path, options.body]);
    if (path.startsWith('/activity')) {
      if (failActivity) throw Object.assign(new Error('down'), {status: 503});
      return {since: 0, rows: h.rows, more: false, unavailable: []};
    }
    if (path.startsWith('/sessions/')) return {turns: [{turn: 1, status: 'running', events: h.turns}], has_earlier: false};
    if (path.startsWith('/tasks/')) return {task: {}};
    if (path === '/memory/records') return {on: true, records: [], documents: 0, embedding_model: '', last_error: ''};
    throw new Error(`unexpected ${path}`);
  };
  h.panel = setupPanel({
    root, chip, backdrop, api, cover: () => [chat], storage: storage ?? {getItem: () => null, setItem: () => {}},
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
    ['mcp__files__file_read', {path: 'a/b.txt'}, 'Reading b.txt'],
    ['mcp__calendar__list_events', {}, 'Using a tool'],
  ];
  for (const [name, args, words] of cases) assert.equal(stepWords(name, args), words, name);
  assert.equal(stepWords('file_read', {path: 'notes.md'}, true), 'Read notes.md');
  assert.equal(stepWords('shell_exec', {command: 'cargo test'}, true), 'Ran the tests');
  assert.equal(stepWords('list_files', {}, true), 'Looked through the folder');
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
  const links = h.root.findAll((node) => node.tagName === 'A').map((link) => link.href);
  assert.deepEqual(links, ['#/workers'], 'a day with no task does not offer All tasks');
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
  const tabs = h.tabs();
  assert.deepEqual(tabs.map((tab) => tab.text()), ['≡ Activity', '✓ Approvals', '◷ Scheduled', '◎ Memory']);
  assert.deepEqual(tabs.map((tab) => [tab.attributes['aria-selected'], tab.tabIndex]), [['true', 0], ['false', -1], ['false', -1], ['false', -1]]);
  const list = h.root.find((node) => node.attributes.role === 'tablist');
  list.dispatch('keydown', {key: 'ArrowRight'});
  assert.equal(h.doc.activeElement, tabs[1]);
  assert.equal(tabs[1].attributes['aria-selected'], 'true');
  assert.match(h.body().text(), /how you answered/);
  assert.equal(h.body().attributes['aria-labelledby'], 'panel-tab-approvals');
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
  // The actions are the terminal's: Pause, Stop and the priority in words, and the link to the task.
  const actions = h.root.find((node) => node.className === 'activity-actions');
  assert.deepEqual(actions.children.map((node) => node.tagName), ['BUTTON', 'BUTTON', 'SELECT', 'A']);
  assert.deepEqual(actions.children[2].children.map((option) => option.textContent), ['Now', 'Today', 'Normal', 'Whenever', 'If idle']);
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

/**
 * The welcome's setup card (setup-card.mjs) in a DOM double: the list of what this Mac can run, the one OK that
 * starts the job, its steps drawn in place while detect carries them, Stop, the connection a finished job leads
 * to, and what a stopped, failed or unavailable setup says. The detect, plan, start and cancel answers are the
 * gateway's shapes (tests/gateway/console/test_setup_flow.py runs the real ones).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {setupModelCard} from '../../support/gateway/static/setup-card.mjs';

class Element {
  constructor(tag) {
    Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, handlers: {}, dataset: {}, textContent: '',
      className: '', hidden: false, disabled: false, checked: false, value: '', type: '', tabIndex: 0, parent: null});
  }
  get ownerDocument() { return doc; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key] ?? null; }
  removeAttribute(key) { delete this.attributes[key]; }
  append(...nodes) { for (const item of nodes) { item.parent = this; this.children.push(item); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  dispatch(type) { for (const fn of this.handlers[type] ?? []) fn({target: this}); }
  click() { if (!this.disabled) this.dispatch('click'); }
  toggle(checked) { this.checked = checked; this.dispatch('change'); }
  focus() { doc.activeElement = this; }
  contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
  text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' ').replace(/\s+/g, ' ').trim(); }
  findAll(test) { return [...(test(this) ? [this] : []), ...this.children.flatMap((child) => child.findAll(test))]; }
  find(test) { return this.findAll(test)[0] ?? null; }
  button(label) { return this.find((node) => node.tagName === 'BUTTON' && node.text() === label); }
  rows(className) { return this.findAll((node) => node.className === className); }
}
const doc = {activeElement: null, hidden: false, createElement: (tag) => new Element(tag)};
const settle = () => new Promise((resolve) => setImmediate(resolve));

const start = {name: 'oMLX', commands: ['brew install jundot/omlx/omlx'], note: 'Put a model in ~/models first.', guide: 'https://example.test/guide'};
const none = (job = null) => ({state: 'none', name: '', backend: '', url: '', models: [], loaded: [], suggested: null, problem: '', start, job});
const found = (job = null) => ({...none(job), state: 'found', backend: 'melo_serve', url: 'http://127.0.0.1:8321/v1', models: ['Qwen3-8B-4bit'], loaded: ['Qwen3-8B-4bit'], suggested: 'Qwen3-8B-4bit'});

const row = (id, name, kind, size, where, state, off = true) => ({id, name, kind, size, where, state, off});
const PLAN = {
  available: true, reason: '', hint: '',
  pick: {id: 'qwen3-8b-omlx', name: 'Qwen3 8B', backend: 'omlx', backend_name: 'oMLX'},
  mac: 'This Mac: Apple M5 Pro · 32 GB · macOS 27.2 · 83 GB free',
  intro: 'Recommended: Qwen3 8B on oMLX. Nothing is installed until you say yes.',
  rows: [
    row('melo_serve', 'melo_serve', 'model service', 'about 30 MB', 'uv tool', 'todo', false),
    row('backend:omlx', 'oMLX 0.7.3', 'chat backend', '881 MB', 'pinned download → ~/Applications/oMLX.app', 'todo'),
    row('model:qwen3-8b-omlx', 'Qwen3 8B', 'chat model', '4.9 GB', 'Hugging Face → ~/models/huggingface', 'todo'),
    row('model:embed', 'Qwen3 Embedding 0.6B', 'memory model', '788 MB', '0.5 of 1.3 GB already here', 'partial'),
    row('service:melo_serve', 'Start at login', 'melo_serve', '', 'macOS shows a "Login Items" notice', 'todo'),
  ],
  download: 'Download about 6.6 GB.', free: '83 GB free.', notes: [], action: 'Set up Qwen3 8B (6.6 GB)', can_set_up: true,
};
const WITHOUT_EMBED = {
  ...PLAN,
  rows: PLAN.rows.map((r) => (r.id === 'model:embed' ? {...r, size: '', where: 'skipped', state: 'skipped'} : r)),
  download: 'Download about 5.8 GB.', notes: ['Without Qwen3 Embedding 0.6B, memory goes by words only.'], action: 'Set up Qwen3 8B (5.8 GB)',
};

const step = (id, name, state, line = '', done = 0, total = 0, progress = '') => ({id, name, state, line, done_bytes: done, total_bytes: total, progress});
const job = (state, rows, extra = {}) => ({id: 1, state, stopping: false, error: '', pick: 'qwen3-8b-omlx', name: 'Qwen3 8B', saved: false, rows, ...extra});
const WAITING = [step('melo_serve', 'melo_serve', 'waiting'), step('backend:omlx', 'oMLX', 'waiting'), step('save', 'Saving the settings', 'waiting')];
const GOING = [step('melo_serve', 'melo_serve', 'done', 'installed melo_serve 0.4.0'), step('backend:omlx', 'oMLX', 'running', 'downloading', 310e6, 881e6, '310 of 881 MB'), step('save', 'Saving the settings', 'waiting')];

/** The gateway as the card sees it: what detect, plan, start and cancel answer now; every call is recorded. */
function harness({detect = none(), plan = PLAN, answers = {}} = {}) {
  const h = {calls: [], timers: [], detect, plan, connected: 0, settings: 0, answers};
  h.api = async (path, options = {}) => {
    const method = options.method ?? (options.body ? 'POST' : 'GET');
    h.calls.push([method, path, options.body]);
    const own = h.answers[`${method} ${path}`];
    if (own) { const reply = typeof own === 'function' ? own(options.body) : own; if (reply instanceof Error) throw reply; return reply; }
    if (path === '/setup/detect') return h.detect;
    if (path.startsWith('/setup/plan')) { if (h.plan instanceof Error) throw h.plan; return h.plan; }
    if (path === '/settings') return method === 'PUT' ? {} : {embedding_backend: 'melo_serve', workspace: '~/ws', revision: 'r1'};
    if (path === '/settings/test') return {ok: true};
    throw new Error(`Unexpected request: ${method} ${path}`);
  };
  h.root = new Element('section');
  h.card = setupModelCard({
    root: h.root, api: h.api, openSettings: (section) => { h.opened = section; }, connected: async () => { h.connected += 1; },
    schedule: (fn, ms) => { const timer = {fn, ms, cleared: false}; h.timers.push(timer); return timer; },
    cancel: (timer) => { timer.cleared = true; },
  });
  h.text = () => h.root.text();
  h.paths = () => h.calls.map(([method, path]) => `${method} ${path}`);
  /** Run the look that is waiting (the card asks again every few seconds). */
  h.look = async () => { const waiting = h.timers.filter((timer) => !timer.cleared); assert.equal(waiting.length, 1, 'one look waits'); const [timer] = waiting; timer.cleared = true; await timer.fn(); await settle(); };
  h.waiting = () => h.timers.filter((timer) => !timer.cleared);
  h.open = async () => { h.card.show(true); await settle(); };
  return h;
}

test('with nothing found the card shows the terminal\'s list with its sizes and places, and one button that says what it sets up', async () => {
  const h = harness();
  await h.open();
  assert.deepEqual(h.paths(), ['GET /setup/detect', 'GET /setup/plan']);
  assert.match(h.text(), /Set up a model on this Mac.*This Mac: Apple M5 Pro.*Recommended: Qwen3 8B on oMLX\. Nothing is installed until you say yes\./);
  const rows = h.root.rows('setup-row');
  assert.equal(rows.length, 5);
  assert.match(rows[1].text(), /oMLX 0\.7\.3 881 MB chat backend pinned download → ~\/Applications\/oMLX\.app/);
  assert.equal(rows[3].dataset.state, 'partial');
  assert.match(h.text(), /Download about 6\.6 GB\. 83 GB free\./);
  const boxes = rows.map((r) => r.findAll((n) => n.type === 'checkbox')[0]);
  assert.deepEqual(boxes.map((box) => box?.disabled), [true, false, false, false, false], 'a row that has to be set up has its box on and locked');
  assert.ok(h.root.button('Set up Qwen3 8B (6.6 GB)'));
  assert.match(h.text(), /Or use an account or a key/, 'the other way stays');
  assert.ok(h.root.button('Enter a server address…'));
  assert.equal(h.waiting().length, 1, 'a server started by hand is still found');
});

test('turning a row off asks for the list worded for it and redraws it in place, keeping the focused box', async () => {
  const h = harness({answers: {}});
  await h.open();
  const rows = h.root.rows('setup-row');
  const box = rows[3].findAll((n) => n.type === 'checkbox')[0];
  assert.equal(box.checked, true);
  assert.equal(box.getAttribute('aria-label'), 'Set up Qwen3 Embedding 0.6B');
  box.focus();
  h.plan = WITHOUT_EMBED;
  box.toggle(false);
  await settle();
  assert.deepEqual(h.calls.at(-1), ['GET', '/setup/plan?skip=model%3Aembed', undefined]);
  assert.deepEqual(h.root.rows('setup-row'), rows, 'the same rows, drawn again in place');
  assert.equal(rows[3].dataset.state, 'skipped');
  assert.match(h.text(), /Download about 5\.8 GB\..*Without Qwen3 Embedding 0\.6B, memory goes by words only\./);
  assert.ok(h.root.button('Set up Qwen3 8B (5.8 GB)'));
  assert.equal(doc.activeElement, box, 'a keyboard user stays on the box');
  h.plan = PLAN;
  box.toggle(true);
  await settle();
  assert.deepEqual(h.calls.at(-1), ['GET', '/setup/plan', undefined]);
  assert.equal(h.root.rows('setup-note').length, 0);
});

test('one OK starts the job with the pick and what was left out, and the steps are drawn from detect, in place', async () => {
  const h = harness({answers: {'POST /setup/start': {job: job('running', WAITING)}}});
  await h.open();
  h.plan = WITHOUT_EMBED;
  h.root.rows('setup-row')[3].findAll((n) => n.type === 'checkbox')[0].toggle(false);
  await settle();
  await h.root.button('Set up Qwen3 8B (5.8 GB)').click();
  await settle();
  assert.deepEqual(h.calls.find(([, path]) => path === '/setup/start'), ['POST', '/setup/start', {pick: 'qwen3-8b-omlx', skip: ['model:embed']}]);
  assert.match(h.text(), /Setting up Qwen3 8B.*Stop keeps what is already downloaded\./);
  assert.equal(h.root.rows('setup-row').length, 0, 'the list gave way to the steps');
  const steps = h.root.rows('setup-step');
  assert.deepEqual(steps.map((s) => s.dataset.state), ['waiting', 'waiting', 'waiting']);
  assert.match(h.text(), /Or use an account or a key/, 'the other way stays while it runs');
  assert.equal(h.waiting()[0].ms, 1500, 'a job is looked at often');
  // The next look: a step done, another downloading.
  h.detect = none(job('running', GOING));
  await h.look();
  assert.deepEqual(h.root.rows('setup-step'), steps, 'the same steps, updated in place');
  assert.deepEqual(steps.map((s) => s.dataset.state), ['done', 'running', 'waiting']);
  assert.match(steps[1].text(), /oMLX.*310 of 881 MB/);
  const bar = steps[1].find((n) => n.tagName === 'PROGRESS');
  assert.equal(bar.value, 35);
  assert.equal(bar.hidden, false);
  assert.equal(steps[0].find((n) => n.tagName === 'PROGRESS').hidden, true);
  assert.equal(steps[0].text(), '✓ melo_serve done installed melo_serve 0.4.0');
  assert.equal(h.waiting().length, 1, 'and it goes on looking');
});

test('Stop asks the gateway to stop, says it is stopping, and keeps the focus on its button', async () => {
  const h = harness({detect: none(job('running', GOING))});
  await h.open();
  const stop = h.root.button('Stop');
  stop.focus();
  h.answers['POST /setup/cancel'] = {job: job('running', GOING, {stopping: true})};
  await stop.click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['POST', '/setup/cancel', undefined]);
  assert.equal(stop.text(), 'Stopping…');
  assert.equal(stop.disabled, true);
  assert.equal(doc.activeElement, stop);
});

test('a job seen running that ends done connects to the model it set up, as Use this model does', async () => {
  const h = harness({detect: none(job('running', GOING))});
  await h.open();
  h.detect = found(job('done', GOING.map((s) => ({...s, state: 'done'})), {saved: true}));
  await h.look();
  assert.deepEqual(h.paths().slice(-3), ['GET /settings', 'PUT /settings', 'POST /settings/test']);
  const put = h.calls.find(([method, path]) => method === 'PUT' && path === '/settings')[2];
  assert.deepEqual(put, {chat_backend: 'melo_serve', embedding_backend: 'melo_serve', workspace: '~/ws', sections: {melo_serve: {chat_url: 'http://127.0.0.1:8321/v1', chat_model: 'Qwen3-8B-4bit'}}, revision: 'r1'});
  assert.equal(h.connected, 1);
  assert.equal(h.waiting().length, 0);
});

test('a job that ended done before this card saw it run is not connected on its own: the server is offered as always', async () => {
  const h = harness({detect: found(job('done', WAITING, {saved: true}))});
  await h.open();
  assert.match(h.text(), /Found a model server on this Mac/);
  assert.ok(h.root.button('Use this model'));
  assert.equal(h.connected, 0);
  assert.equal(h.paths().includes('PUT /settings'), false);
});

test('a stopped job says so above the list, which is asked again and goes on from what is here', async () => {
  const h = harness({detect: none(job('running', GOING))});
  await h.open();
  h.plan = {...PLAN, action: 'Continue (5.7 GB left)'};
  h.detect = none(job('cancelled', GOING));
  await h.look();
  const plans = h.paths().filter((path) => path.startsWith('GET /setup/plan'));
  assert.equal(plans.length, 1, 'the list is asked again once the job is over');
  assert.match(h.text(), /Set up a model on this Mac.*Setup stopped What was downloaded is kept\. Continue when you are ready\./);
  assert.equal(h.root.find((n) => n.className === 'setup-banner').dataset.kind, 'stopped');
  assert.ok(h.root.button('Continue (5.7 GB left)'));
  h.answers['POST /setup/start'] = {job: {...job('running', WAITING), id: 2}};
  await h.root.button('Continue (5.7 GB left)').click();
  await settle();
  assert.match(h.text(), /Setting up Qwen3 8B/);
});

test('a job that failed says why, and one that left the chat model out says what to do', async () => {
  const failed = harness({detect: none(job('failed', GOING, {error: 'failed: network is down; what was downloaded is kept'}))});
  await failed.open();
  const banner = failed.root.find((n) => n.className === 'setup-banner');
  assert.equal(banner.dataset.kind, 'failed');
  assert.match(banner.text(), /Setup did not finish failed: network is down; what was downloaded is kept/);
  assert.ok(failed.root.button('Set up Qwen3 8B (6.6 GB)'), 'the list offers to try again');
  const left = harness({detect: none(job('done', GOING, {saved: false}))});
  await left.open();
  assert.match(left.root.find((n) => n.className === 'setup-banner').text(), /The rest is set up, but there is no chat model yet Set up again with the chat model turned on to finish\./);
});

test('a start the gateway refuses is said under the list, which stays', async () => {
  const refused = Object.assign(new Error('A setup is already running.'), {status: 409});
  const h = harness({answers: {'POST /setup/start': refused}});
  await h.open();
  await h.root.button('Set up Qwen3 8B (6.6 GB)').click();
  await settle();
  assert.match(h.text(), /A setup is already running\./);
  assert.equal(h.root.rows('setup-row').length, 5);
  assert.equal(h.root.button('Set up Qwen3 8B (6.6 GB)').disabled, false, 'it can be tried again');
});

test('a list that does not fit the disk cannot be started until something is left out', async () => {
  const h = harness({plan: {...PLAN, can_set_up: false, notes: ['There is not enough free space for this. Another model may fit.']}});
  await h.open();
  assert.match(h.text(), /not enough free space/);
  assert.equal(h.root.button('Set up Qwen3 8B (6.6 GB)').disabled, true);
});

test('with no plan the commands that start a server by hand stay, with why there is no setup', async () => {
  const h = harness({plan: {available: false, reason: 'uv is not installed (https://docs.astral.sh/uv/)', hint: ''}});
  await h.open();
  assert.match(h.text(), /No model server is running on this Mac.*Automatic setup is not available: uv is not installed.*To start one with oMLX, run this in a terminal:.*brew install/);
  assert.equal(h.root.rows('setup-row').length, 0);
  const quiet = harness({plan: new Error('Unexpected request')});
  await quiet.open();
  assert.match(quiet.text(), /No model server is running on this Mac.*brew install/, 'a plan that cannot be asked leaves the commands, with no words about it');
  assert.doesNotMatch(quiet.text(), /Automatic setup/);
});

test('while the plan is asked the card says so, and a server found meanwhile takes its place', async () => {
  let release;
  const h = harness({answers: {'GET /setup/plan': () => new Promise((resolve) => { release = () => resolve(PLAN); })}});
  await h.open();
  assert.match(h.text(), /Checking what this Mac can run…/);
  assert.equal(h.root.attributes['aria-busy'], 'true');
  h.detect = found();
  await h.look();
  assert.match(h.text(), /Found a model server on this Mac/);
  release();
  await settle();
  assert.match(h.text(), /Found a model server on this Mac/, 'the plan that comes late draws nothing');
});

test('hiding the card forgets the job on screen and asks nothing more', async () => {
  const h = harness({detect: none(job('running', GOING))});
  await h.open();
  h.card.show(false);
  assert.equal(h.waiting().length, 0);
  assert.equal(h.card.holding(), false);
  const asked = h.calls.length;
  await h.look.call(h).catch(() => {});
  assert.equal(h.calls.length, asked);
});

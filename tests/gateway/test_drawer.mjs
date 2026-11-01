import test from 'node:test';
import assert from 'node:assert/strict';
import {setupDrawer} from '../../support/gateway/static/drawer.mjs';

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
}
function harness(routes, mode = {on: false}) {
  const doc = {activeElement: null, handlers: {},
    addEventListener(type, handler) { this.handlers[type] = handler; },
    removeEventListener(type, handler) { if (this.handlers[type] === handler) delete this.handlers[type]; }};
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.handlers = {}; this.textContent = ''; this.ownerDocument = doc; this.tabIndex = tag === 'button' ? 0 : -1; }
    setAttribute(key, value) { this.attributes[key] = value; }
    contains(element) { return this === element || this.children.some((child) => child.contains(element)); }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { if (this.contains(doc.activeElement) && this !== doc.activeElement) doc.activeElement = null; this.children = nodes; }
    addEventListener(type, handler) { this.handlers[type] = handler; }
    focus() { doc.activeElement = this; doc.handlers.focusin?.({target: this}); }
    querySelectorAll(tag) { const match = (c) => tag.includes(',') ? c.tag === 'button' : c.tag === tag; return this.children.flatMap((c) => [...(match(c) ? [c] : []), ...c.querySelectorAll(tag)]); }
    click() { if (!this.disabled) return this.handlers.click?.({target: this}); }
  }
  doc.createElement = (tag) => new Element(tag);
  const root = new Element('section');
  const trigger = new Element('button');
  trigger.focus();
  let id = 's/1';
  const calls = [];
  const api = async (path, options = {}) => {
    calls.push([path, options.method || 'GET']);
    const route = routes[`${options.method || 'GET'} ${path}`];
    if (route === undefined) throw new Error(`Unexpected request ${path}`);
    return typeof route === 'function' ? route() : route;
  };
  const drawer = setupDrawer({api, currentId: () => id, root, trigger, developer: () => mode.on});
  function text(element = root) { return [element.textContent, ...element.children.map(text)].join(' '); }
  function find(label) { return root.querySelectorAll('button').find((item) => item.textContent === label); }
  return {drawer, root, trigger, calls, text, find, setId(value) { id = value; }, doc};
}
const tools = {tools: [{name: '<shell/a>', source: 'builtin', description: '<script>bad</script>', granted: true}, {name: 'read', source: 'builtin', granted: false}], mcp_config_label: '', mcp_error: 'MCP is not built in the console yet'};
const emptyTools = {tools: [], mcp_config_label: '', mcp_error: ''};
const memory = {agent: 'open', working: [{role: 'assistant', text: '<img src=x>', tool_calls: ['read'], tool_call_id: 'c', images: 1}], working_max: 20, long_term: {enabled: true, backend: 'mlx'}};
const run = (turn) => ({turn, turn_id: `t${turn}`, status: 'completed', started_at: 10, finished_at: 12, result: {result: 'secret answer', n_tool: 2}});

test('tools render safely and revoke one or all through encoded paths', async () => {
  let count = 0;
  const h = harness({
    'GET /sessions/s%2F1/tools': () => ++count === 1 ? tools : emptyTools,
    'DELETE /sessions/s%2F1/grants/%3Cshell%2Fa%3E': {},
    'DELETE /sessions/s%2F1/grants': {},
  });
  await h.drawer.open();
  assert.match(h.text(), /<script>bad<\/script>/);
  assert.equal(h.root.querySelectorAll('script').length, 0);
  assert.equal(h.trigger.attributes['aria-expanded'], 'true');
  assert.equal(h.find('Revoke').attributes['aria-label'], 'Revoke <shell/a>');
  await h.find('Revoke').click();
  assert.deepEqual(h.calls.slice(1), [['/sessions/s%2F1/grants/%3Cshell%2Fa%3E', 'DELETE'], ['/sessions/s%2F1/tools', 'GET']]);
  count = 0;
  await h.drawer.selectTab('tools');
  await h.find('Revoke all grants').click();
  assert.deepEqual(h.calls.at(-2), ['/sessions/s%2F1/grants', 'DELETE']);
  assert.equal(h.drawer.close(), true);
  assert.equal(h.root.hidden, true);
  assert.equal(h.doc.activeElement, h.trigger);
});

test('memory and paginated runs use actual response shapes', async () => {
  const h = harness({'GET /sessions/s%2F1/tools': emptyTools, 'GET /sessions/s%2F1/memory': memory,
    'GET /sessions/s%2F1/runs': {runs: [run(3), run(4)], has_earlier: true},
    'GET /sessions/s%2F1/runs?before=3': {runs: [run(1), run(2)], has_earlier: false}});
  await h.drawer.open();
  await h.drawer.selectTab('memory');
  assert.match(h.text(), /Working memory: 1 \/ 20/);
  assert.match(h.text(), /<img src=x>/);
  assert.equal(h.root.querySelectorAll('img').length, 0);
  await h.drawer.selectTab('runs');
  assert.match(h.text(), /Duration: 2.0s/);
  assert.doesNotMatch(h.text(), /secret answer/);
  await h.find('Earlier runs').click();
  assert.ok(h.text().indexOf('Turn 1') < h.text().indexOf('Turn 3'));
  assert.equal(h.find('Earlier runs'), undefined);
});

test('errors expose retry and failed revokes preserve controls', async () => {
  let load = 0;
  const h = harness({'GET /sessions/s%2F1/tools': () => { if (!load++) throw new Error('offline'); return tools; },
    'DELETE /sessions/s%2F1/grants/%3Cshell%2Fa%3E': () => { throw new Error('denied'); }});
  await h.drawer.open();
  assert.match(h.text(), /offline/);
  h.find('Retry').focus();
  await h.find('Retry').click();
  assert.equal(h.doc.activeElement, h.find('Close details'));
  await h.find('Revoke').click();
  assert.match(h.text(), /denied/);
  assert.equal(h.find('Revoke').disabled, false);
});

test('late loads cannot overwrite a new tab, session, or closed drawer', async () => {
  const first = deferred(), second = deferred(), third = deferred();
  let n = 0;
  const h = harness({'GET /sessions/s%2F1/tools': () => [first, second, third][n++].promise,
    'GET /sessions/s%2F1/memory': memory, 'GET /sessions/new/tools': emptyTools});
  const a = h.drawer.open();
  await h.drawer.selectTab('memory');
  first.resolve(tools); await a;
  assert.match(h.text(), /Working memory/);
  const b = h.drawer.selectTab('tools');
  h.setId('new'); await h.drawer.sessionChanged();
  second.resolve(tools); await b;
  assert.match(h.text(), /No tools available/);
  h.setId('s/1');
  const c = h.drawer.sessionChanged();
  h.drawer.close(); third.resolve(tools); await c;
  assert.equal(h.root.hidden, true);
  assert.doesNotMatch(h.text(), /shell/);
});

test('revoke finishing after tab return refreshes grants', async () => {
  const mutation = deferred();
  let count = 0;
  const h = harness({'GET /sessions/s%2F1/tools': () => ++count < 3 ? tools : emptyTools,
    'DELETE /sessions/s%2F1/grants/%3Cshell%2Fa%3E': () => mutation.promise,
    'GET /sessions/s%2F1/memory': memory});
  await h.drawer.open();
  const pending = h.find('Revoke').click();
  await h.drawer.selectTab('memory');
  await h.drawer.selectTab('tools');
  mutation.resolve({}); await pending;
  assert.match(h.text(), /No tools available/);
});

test('Tab and Shift+Tab wrap focus while open and ignore disabled buttons', async () => {
  const h = harness({'GET /sessions/s%2F1/tools': tools});
  await h.drawer.open();
  const buttons = h.root.querySelectorAll('button');
  const first = buttons[0], last = buttons.at(-1);
  let prevented = false;
  const key = (shiftKey) => h.root.handlers.keydown({key: 'Tab', shiftKey, preventDefault() { prevented = true; }});
  first.focus(); key(true);
  assert.equal(prevented, true);
  assert.equal(h.doc.activeElement, last);
  prevented = false; key(false);
  assert.equal(prevented, true);
  assert.equal(h.doc.activeElement, first);
  last.disabled = true;
  prevented = false; key(true);
  assert.equal(h.doc.activeElement, buttons.at(-2));
  assert.equal(prevented, true);
  h.drawer.close();
  prevented = false; key(true);
  assert.equal(prevented, false);
});

test('tools hint explains the Agent and open(tab) loads only the requested tab', async () => {
  const h = harness({'GET /sessions/s%2F1/tools': emptyTools, 'GET /sessions/s%2F1/memory': memory});
  await h.drawer.open('memory');
  assert.deepEqual(h.calls, [['/sessions/s%2F1/memory', 'GET']]);
  await h.drawer.open('tools');
  assert.deepEqual(h.calls, [['/sessions/s%2F1/memory', 'GET'], ['/sessions/s%2F1/tools', 'GET']]);
  assert.match(h.text(), /Viewing tools opens the Agent/);
  h.drawer.close();
  assert.equal(h.doc.activeElement, h.trigger);
});

test('404 revoke refreshes stale grants and preserves the notice', async () => {
  let count = 0;
  const missing = Object.assign(new Error('Grant already revoked'), {status: 404});
  const h = harness({'GET /sessions/s%2F1/tools': () => ++count === 1 ? tools : emptyTools,
    'DELETE /sessions/s%2F1/grants/%3Cshell%2Fa%3E': () => { throw missing; }});
  await h.drawer.open();
  await h.find('Revoke').click();
  assert.equal(count, 2);
  assert.match(h.text(), /Grant already revoked/);
  assert.match(h.text(), /No tools available/);
  assert.equal(h.find('Revoke'), undefined);
});

test('focus stays in the drawer across removal, external focus, and close', async () => {
  const pending = deferred();
  let count = 0;
  const h = harness({'GET /sessions/s%2F1/tools': () => ++count === 1 ? tools : pending.promise,
    'DELETE /sessions/s%2F1/grants/%3Cshell%2Fa%3E': {}});
  await h.drawer.open();
  h.find('Revoke').focus();
  const refresh = h.find('Revoke').click();
  await Promise.resolve();
  assert.equal(h.doc.activeElement, h.find('Close details'));
  h.trigger.focus();
  assert.equal(h.doc.activeElement, h.find('Close details'));
  h.drawer.close();
  assert.equal(h.doc.handlers.focusin, undefined);
  const outside = h.doc.createElement('button');
  outside.focus();
  pending.resolve(emptyTools); await refresh;
  assert.equal(h.doc.activeElement, outside);
});

test('rendering an appended page restores focus removed with the old content', async () => {
  const h = harness({'GET /sessions/s%2F1/runs': {runs: [run(3)], has_earlier: true},
    'GET /sessions/s%2F1/runs?before=3': {runs: [run(1)], has_earlier: false}});
  await h.drawer.open('runs');
  h.find('Earlier runs').focus();
  await h.find('Earlier runs').click();
  assert.equal(h.doc.activeElement, h.find('Close details'));
});

test('empty filtered page with earlier turns explains unavailable cursor', async () => {
  const h = harness({'GET /sessions/s%2F1/tools': emptyTools, 'GET /sessions/s%2F1/runs': {runs: [], has_earlier: true}});
  await h.drawer.open(); await h.drawer.selectTab('runs');
  assert.match(h.text(), /no finished runs/);
});

const richRun = {turn: 1, turn_id: 't1', status: 'completed', started_at: 10, finished_at: 12,
  result: {n_llm: 3, usage: {prompt_tokens: 9}, prefix_cache: {hits: 2}}};

test('developer mode adds the Events tab, memory internals and run counters; off hides them', async () => {
  const mode = {on: false};
  const h = harness({'GET /sessions/s%2F1/tools': emptyTools, 'GET /sessions/s%2F1/memory': memory,
    'GET /sessions/s%2F1/runs': {runs: [richRun], has_earlier: false},
    'GET /sessions/s%2F1/turns?limit=1&events=true': {turns: [{turn: 7, events: [{id: 5, type: 'run.finished', usage: {prompt_tokens: 9}}]}]}}, mode);
  await h.drawer.open();
  assert.equal(h.find('Events').hidden, true);
  assert.equal(await h.drawer.selectTab('events'), undefined);
  await h.drawer.selectTab('memory');
  assert.doesNotMatch(h.text(), /Agent: open|\(mlx\)/);
  await h.drawer.selectTab('runs');
  assert.match(h.text(), /Duration: 2.0s/);
  assert.doesNotMatch(h.text(), /n_llm|prefix_cache/);
  mode.on = true;
  await h.drawer.modeChanged();
  assert.equal(h.find('Events').hidden, false);
  assert.match(h.text(), /n_llm: 3/);
  assert.match(h.text(), /prefix_cache: {"hits":2}/);
  await h.drawer.selectTab('memory');
  assert.match(h.text(), /Agent: open/);
  assert.match(h.text(), /\(mlx\)/);
  await h.drawer.selectTab('events');
  assert.match(h.text(), /Raw events of turn: 7/);
  assert.match(h.text(), /#5 run.finished/);
  mode.on = false;
  await h.drawer.modeChanged();
  assert.equal(h.find('Events').hidden, true);
  assert.match(h.text(), /No tools available/);
});

test('the Tools tab lists the steps the person approved in this chat, and Revoke takes one back', async () => {
  let listed = [{id: 'e1', tool: 'python_exec', folder: '/w', verdict: 'routine', categories: [], preview: 'print(1)\nprint(2)', at: 1_800_000_000}];
  const h = harness({
    'GET /sessions/s%2F1/tools': () => ({...emptyTools, approvals: listed}),
    'DELETE /sessions/s%2F1/approvals/e1': () => { listed = []; return {}; },
  });
  await h.drawer.open();
  assert.match(h.text(), /Steps you approved here.*run these exact steps again in the same folder without asking.*High-impact steps always ask/);
  assert.match(h.text(), /print\(1\) print\(2\).*python_exec · approved at/);
  const revoke = h.find('Revoke');
  assert.equal(revoke.attributes['aria-label'], 'Revoke your approval of print(1) print(2)');
  await revoke.click();
  assert.deepEqual(h.calls.slice(1), [['/sessions/s%2F1/approvals/e1', 'DELETE'], ['/sessions/s%2F1/tools', 'GET']]);
  assert.doesNotMatch(h.text(), /Steps you approved here/);
});

test('the folders this chat may read are listed, and Revoke takes one back by its path', async () => {
  let count = 0;
  const withFolder = {...emptyTools, folders: [{path: '/Users/me/developer/career', label: '~/developer/career'}]};
  const h = harness({
    'GET /sessions/s%2F1/tools': () => ++count === 1 ? withFolder : emptyTools,
    'DELETE /sessions/s%2F1/folders?path=%2FUsers%2Fme%2Fdeveloper%2Fcareer': {},
  });
  await h.drawer.open();
  assert.match(h.text(), /Folders this chat may read/);
  assert.match(h.text(), /~\/developer\/career/);
  assert.ok(h.find('Revoke all grants'), 'a folder is a grant the chat holds');
  const revoke = h.find('Revoke');
  assert.equal(revoke.attributes['aria-label'], 'Revoke ~/developer/career');
  await revoke.click();
  assert.deepEqual(h.calls.slice(1), [['/sessions/s%2F1/folders?path=%2FUsers%2Fme%2Fdeveloper%2Fcareer', 'DELETE'], ['/sessions/s%2F1/tools', 'GET']]);
  assert.doesNotMatch(h.text(), /Folders this chat may read/);
});

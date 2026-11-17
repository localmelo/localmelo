import assert from 'node:assert/strict';
import test from 'node:test';
import {setupServe} from '../../support/gateway/static/serve.mjs';

function node(tag, doc) {
  const item = {
    tagName: tag.toUpperCase(), hidden: false, textContent: '', className: '', type: '', dataset: {}, children: [],
    listeners: {}, attributes: {},
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name] ?? null; },
    addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); },
    focus() { doc.activeElement = this; },
    click() { return this.listeners.click?.[0]?.({target: this}); },
  };
  return item;
}

const text = (item) => [item.textContent, ...item.children.map(text)].join('');
const find = (item, test) => (test(item) ? item : item.children.map((child) => find(child, test)).find(Boolean) ?? null);

const PICKS = [
  {id: '27b', name: 'Qwen3.8 27B', current: true, loadable: true, words: 'Splash · needs 21 GB · fits', fit: 'fits'},
  {id: '8b', name: 'Qwen3 8B', current: false, loadable: true, words: 'oMLX · needs 5 GB · fits', fit: 'fits'},
  {id: '4b', name: 'Qwen3 4B', current: false, loadable: true, words: 'oMLX · needs 2.5 GB · tight fit', fit: 'tight fit', tight: true},
  {id: '70b', name: 'Qwen3 70B', current: false, loadable: false, words: 'Splash · needs 48 GB, 31 GB free', fit: ''},
];

function rig({groups = null, backend = 'melo_serve', refuse = '', keyed = groups !== null} = {}) {
  const doc = {activeElement: null};
  doc.createElement = (tag) => node(tag, doc);
  const health = {model: {backend, name: 'Qwen3.8-27B', backend_label: 'DeepSeek'}, switch: null};
  const calls = [];
  const choiceCalls = [];
  const modelSet = [];
  const said = [];
  const waiting = {id: 1, pick: '8b', name: 'Qwen3 8B', state: 'waiting', title: 'Switching to Qwen3 8B…', detail: 'Waiting for 1 chat to finish', error: ''};
  const api = async (path, options = {}) => {
    if (path.startsWith('/chat-models')) {
      choiceCalls.push([path, options.body ?? null]);
      if (path === '/chat-models/switch' && refuse) throw new Error(refuse);
      if (path === '/chat-models/switch') return {ok: true, model: {backend: 'openai', name: options.body.model, backend_label: options.body.group}};
      return {hosted: groups !== null, groups: groups ?? []};
    }
    calls.push([path, options.body ?? null]);
    if (path === '/serve') return {picks: PICKS, foot: 'Every chat on this Mac follows the switch.'};
    if (path === '/serve/switch') return {switch: waiting};
    return {switch: {...waiting, state: 'cancelled'}};
  };
  const openers = [0, 1].map(() => ({button: node('button', doc), menu: node('div', doc)}));
  const serve = setupServe({
    doc, api, state: () => ({health, keyed: keyed ? ['deepseek'] : []}), notify: (words) => said.push(words), openers,
    changed: (sw) => { health.switch = sw; },
    modelChanged: (model) => modelSet.push(model),
  });
  return {doc, health, calls, choiceCalls, modelSet, said, openers, serve, menu: openers[0].menu};
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const rows = (menu) => menu.children.filter((child) => /mode-item/.test(child.className));

test('the chip opens the picks: the current one checked, a fit word coloured, one that cannot load disabled', async () => {
  const t = rig();
  assert.equal(t.serve.click(0), true);
  await settle();
  assert.equal(t.menu.hidden, false);
  assert.equal(t.openers[0].button.attributes['aria-expanded'], 'true');
  const [current, eight, four, big] = rows(t.menu);
  assert.equal(current.attributes['aria-checked'], 'true');
  assert.equal(text(eight), '✓Qwen3 8BoMLX · needs 5 GB · fits');
  assert.equal(find(eight, (item) => item.className === 'pick-fit').textContent, 'fits');
  assert.equal(find(four, (item) => item.className === 'pick-fit').dataset.fit, 'tight');
  assert.equal(big.attributes['aria-disabled'], 'true');
  assert.equal(find(big, (item) => item.className === 'pick-fit'), null);  // dimmed, its reason in words
  assert.equal(t.menu.children.at(-1).textContent, 'Every chat on this Mac follows the switch.');
  big.click();  // never chosen
  assert.deepEqual(t.calls, [['/serve', null]]);
  assert.equal(t.doc.activeElement, current);  // the arrows start on the current pick
});

test('a pick switches every chat; while it waits the others are disabled and Cancel ends the wait', async () => {
  const t = rig();
  t.serve.click(0);
  await settle();
  await rows(t.menu)[1].click();
  assert.deepEqual(t.calls.at(-1), ['/serve/switch', {pick: '8b'}]);
  const [current, waiting, four] = rows(t.menu);
  assert.equal(current.attributes['aria-disabled'], 'true');
  assert.equal(four.attributes['aria-disabled'], 'true');
  assert.equal(waiting.className, 'mode-item pick-waiting');
  assert.match(text(waiting), /Qwen3 8BWaiting for 1 chat to finish/);
  assert.deepEqual(t.serve.chip(), {name: 'Switching to Qwen3 8B…', backend: ''});
  await find(waiting, (item) => /pick-cancel/.test(item.className)).click();
  assert.deepEqual(t.calls.at(-1), ['/serve/switch/cancel', null]);
  // A switch that fails is said once, when the health brings its end.
  t.health.switch = {id: 2, pick: '4b', name: 'Qwen3 4B', state: 'loading'};
  t.serve.update();
  t.health.switch = {...t.health.switch, state: 'failed', error: 'it did not start'};
  t.serve.update();
  t.serve.update();
  assert.deepEqual(t.said, ['The switch to Qwen3 4B did not finish: it did not start']);
  // Another model server: the buttons open Settings, as before.
  t.health.model.backend = 'ollama';
  t.serve.close();
  assert.equal(t.serve.click(1), false);
});

test('a switch that fails before the next poll is said once', async () => {
  const t = rig();
  await t.serve.switchTo('8b');  // the 202 says waiting; no poll saw it
  t.health.switch = {id: 1, pick: '8b', name: 'Qwen3 8B', state: 'failed', error: 'melo_serve is busy'};
  t.serve.update();
  t.serve.update();
  assert.deepEqual(t.said, ['The switch to Qwen3 8B did not finish: melo_serve is busy']);
});

// --- hosted services: the groups of the list ---------------------------------------------------------------

const GROUPS = [
  {id: 'local', label: 'oMLX', kind: 'local', serves: false, models: ['qwen-small', 'qwen-big'], current: '', problem: ''},
  {id: 'openai', label: 'OpenAI', kind: 'hosted', serves: false, models: ['gpt-4o', 'gpt-4o-mini'], current: '', problem: ''},
  {id: 'deepseek', label: 'DeepSeek', kind: 'hosted', serves: false, models: ['deepseek-flash', 'deepseek-v4-pro'], current: 'deepseek-flash', problem: ''},
  {id: 'gemini', label: 'Google Gemini', kind: 'hosted', serves: false, models: [], current: '', problem: 'The service did not accept the key (401). Paste the key again.'},
];
const headings = (menu) => menu.children.filter((child) => child.className === 'mode-menu-title').map((child) => child.textContent);

test('with no hosted key and another model server there is no list: the buttons open Settings, as before', async () => {
  const t = rig({backend: 'ollama'});
  t.serve.update();
  await settle();
  assert.equal(t.serve.serves(), false);
  assert.equal(t.serve.click(0), false);
  assert.deepEqual(t.choiceCalls, [], 'no new request: nothing is asked while no hosted service has a key');
  await t.serve.reloadChoices();  // a settings save with no key known asks nothing either
  assert.deepEqual(t.choiceCalls, []);
});

test('a key saved in Settings puts its service on the list at once, asking only for the groups', async () => {
  const t = rig({groups: GROUPS, backend: 'ollama', keyed: false});
  assert.equal(t.serve.serves(), false);
  await t.serve.reloadChoices(true);
  assert.deepEqual(t.choiceCalls, [['/chat-models?models=0', null]]);
  assert.equal(t.serve.serves(), true);
});

test('with a key the chip opens one list: a group per server and service, the one in use marked, a failing one as a row that says why', async () => {
  const t = rig({groups: GROUPS, backend: 'openai'});
  t.serve.update();
  await settle();
  assert.equal(t.serve.serves(), true);
  assert.deepEqual(t.serve.chip(), {name: 'Qwen3.8-27B', backend: 'DeepSeek'});
  assert.equal(t.serve.click(0), true);
  await settle();
  assert.equal(t.choiceCalls.at(-1)[0], '/chat-models', 'the models are asked when the list opens');
  assert.deepEqual(headings(t.menu).slice(0, 4), ['oMLX', 'OpenAI', 'DeepSeek', 'Google Gemini']);
  const items = rows(t.menu);
  assert.deepEqual(items.map((item) => text(item)), ['✓qwen-small', '✓qwen-big', '✓gpt-4o', '✓gpt-4o-mini', '✓deepseek-flash', '✓deepseek-v4-pro', 'The service did not accept the key (401). Paste the key again.']);
  assert.deepEqual(items.map((item) => item.attributes['aria-checked']).slice(0, 6), ['false', 'false', 'false', 'false', 'true', 'false']);
  assert.equal(items.at(-1).attributes['aria-disabled'], 'true', 'the failing service is no choice and hides no other');
  assert.equal(t.doc.activeElement, items[4], 'the arrows start on the model in use');
  assert.equal(t.calls.length, 0, 'melo_serve is not asked');
});

test('a model picked from a group is switched to at once and named by the chip, no key, no test', async () => {
  const t = rig({groups: GROUPS, backend: 'openai'});
  t.serve.update();
  await settle();
  t.serve.click(0);
  await settle();
  await rows(t.menu)[3].click();
  assert.deepEqual(t.choiceCalls.find(([path]) => path === '/chat-models/switch'), ['/chat-models/switch', {group: 'openai', model: 'gpt-4o-mini'}]);
  assert.equal(t.modelSet.at(-1).name, 'gpt-4o-mini');
  assert.equal(t.menu.hidden, true, 'the list closes');
  assert.deepEqual(t.said, []);
  // The model in use is not switched to again.
  t.serve.click(0);
  await settle();
  const before = t.choiceCalls.length;
  await rows(t.menu)[4].click();
  assert.equal(t.choiceCalls.length, before);
});

test('/model names a model of any group, or service:model', async () => {
  const t = rig({groups: GROUPS, backend: 'openai'});
  t.serve.update();
  await settle();
  await t.serve.load();
  assert.equal(t.serve.named('gpt-4o').model, 'gpt-4o');
  assert.equal(t.serve.named('deepseek:deepseek-v4-pro').group, 'deepseek');
  assert.equal(t.serve.named('nothing'), null);
  assert.equal(t.serve.unknown('nothing'), 'No listed model is called nothing.');
  const started = await t.serve.switchTo(t.serve.named('qwen-big').id);
  assert.deepEqual(t.choiceCalls.find(([path]) => path === '/chat-models/switch'), ['/chat-models/switch', {group: 'local', model: 'qwen-big'}]);
  assert.equal(started.title, 'Model set to qwen-big (saved).');
});

test('a refused switch is said and the list stays open on the model in use', async () => {
  const words = 'The service has no model called gpt-4o. Choose one from the list.';
  const t = rig({groups: GROUPS, backend: 'openai', refuse: words});
  t.serve.update();
  await settle();
  t.serve.click(0);
  await settle();
  await rows(t.menu)[2].click();
  assert.deepEqual(t.said, [words]);
  assert.deepEqual(t.modelSet, []);
  assert.equal(t.menu.hidden, false);
});

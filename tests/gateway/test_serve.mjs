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

function rig() {
  const doc = {activeElement: null};
  doc.createElement = (tag) => node(tag, doc);
  const health = {model: {backend: 'melo_serve', name: 'Qwen3.8-27B'}, switch: null};
  const calls = [];
  const said = [];
  const waiting = {id: 1, pick: '8b', name: 'Qwen3 8B', state: 'waiting', title: 'Switching to Qwen3 8B…', detail: 'Waiting for 1 chat to finish', error: ''};
  const api = async (path, options = {}) => {
    calls.push([path, options.body ?? null]);
    if (path === '/serve') return {picks: PICKS, foot: 'Every chat on this Mac follows the switch.'};
    if (path === '/serve/switch') return {switch: waiting};
    return {switch: {...waiting, state: 'cancelled'}};
  };
  const openers = [0, 1].map(() => ({button: node('button', doc), menu: node('div', doc)}));
  const serve = setupServe({
    doc, api, state: () => ({health}), notify: (words) => said.push(words), openers,
    changed: (sw) => { health.switch = sw; },
  });
  return {doc, health, calls, said, openers, serve, menu: openers[0].menu};
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

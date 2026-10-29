import assert from 'node:assert/strict';
import test from 'node:test';
import {describe, setupRunning} from '../../support/gateway/static/running.mjs';

function node(tag, doc) {
  const item = {
    tag, hidden: false, textContent: '', className: '', href: '', children: [], listeners: {}, attributes: {},
    ownerDocument: doc,
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); },
    contains(other) { return other === this || this.children.some((child) => child.contains?.(other)); },
    focus() { doc.focused = this; },
  };
  return item;
}

function rig() {
  const doc = {listeners: {}, focused: null};
  doc.createElement = (tag) => node(tag, doc);
  doc.addEventListener = (type, handler) => { (doc.listeners[type] ||= []).push(handler); };
  const root = node('div', doc);
  const button = node('button', doc);
  const menu = node('div', doc);
  root.append(button, menu);
  menu.hidden = true;
  const running = setupRunning({root, button, menu});
  const rows = () => menu.children.map((child) => [child.className, ...child.children.map((part) => part.textContent), child.href]);
  return {doc, root, button, menu, running, rows, fire: (target, type, event = {}) => target.listeners[type]?.forEach((fn) => fn({target, ...event}))};
}

const status = (running, items = []) => ({running: {chats: 0, tasks: 0, runs: 0, terminal: 0, ...running}, tasks: {}, items});

test('the chip is hidden at 0 and says how many run, by kind', () => {
  const t = rig();
  t.running.update(status({}));
  assert.equal(t.root.hidden, true);
  t.running.update(status({chats: 1, tasks: 2, terminal: 1}));
  assert.equal(t.root.hidden, false);
  assert.equal(t.button.textContent, '4 running');
  assert.equal(t.button.attributes['aria-label'], '4 running: 1 chat, 2 tasks, 1 terminal chat');
  t.running.update(undefined);
  assert.equal(t.root.hidden, true);
});

test('describe words each kind in the singular and the plural', () => {
  assert.equal(describe({chats: 1, runs: 1}), '1 chat, 1 run');
  assert.equal(describe({chats: 2, tasks: 3, runs: 4, terminal: 5}), '2 chats, 3 tasks, 4 runs, 5 terminal chats');
  assert.equal(describe({}), '');
});

test('the list links each kind to the view that shows it', () => {
  const t = rig();
  t.running.update(status({chats: 1, tasks: 1, runs: 1, terminal: 1}, [
    {kind: 'chat', id: 'c 1', title: 'Fix the build'},
    {kind: 'task', id: 't1', title: 'Index'},
    {kind: 'run', id: 'w1', title: ''},
    {kind: 'terminal', id: 'k1', title: 'Terminal work'},
  ]));
  assert.equal(t.menu.hidden, true);
  t.fire(t.button, 'click');
  assert.equal(t.menu.hidden, false);
  assert.equal(t.button.attributes['aria-expanded'], 'true');
  assert.deepEqual(t.rows(), [
    ['running-item', 'Chat', 'Fix the build', '#/s/c%201'],
    ['running-item', 'Task', 'Index', '#/tasks/t1'],
    ['running-item', 'Run', 'Untitled', '#/workers/w1'],
    ['running-item', 'Terminal chat', 'Terminal work', '#/workers/k1'],
  ]);
});

test('a capped list says how many more run', () => {
  const t = rig();
  t.running.update(status({runs: 25}, Array.from({length: 20}, (_, n) => ({kind: 'run', id: `w${n}`, title: `run ${n}`}))));
  t.fire(t.button, 'click');
  assert.equal(t.menu.children.length, 21);
  assert.equal(t.menu.children[20].textContent, 'and 5 more');
});

test('an open list follows the poll, an unchanged poll leaves it alone, and 0 closes it', () => {
  const t = rig();
  t.running.update(status({tasks: 1}, [{kind: 'task', id: 't1', title: 'One'}]));
  t.fire(t.button, 'click');
  const first = t.menu.children[0];
  t.running.update(status({tasks: 1}, [{kind: 'task', id: 't1', title: 'One'}]));
  assert.equal(t.menu.children[0], first);
  t.running.update(status({tasks: 2}, [{kind: 'task', id: 't1', title: 'One'}, {kind: 'task', id: 't2', title: 'Two'}]));
  assert.deepEqual(t.rows().map((row) => row[2]), ['One', 'Two']);
  t.running.update(status({}));
  assert.equal(t.running.isOpen(), false);
  assert.equal(t.menu.hidden, true);
  assert.equal(t.root.hidden, true);
});

test('choosing a row closes the list, and the page can close it', () => {
  const t = rig();
  t.running.update(status({chats: 1}, [{kind: 'chat', id: 'c', title: 'Chat'}]));
  t.fire(t.button, 'click');
  assert.equal(t.running.isOpen(), true);
  t.fire(t.menu, 'click');
  assert.equal(t.running.isOpen(), false);
  assert.equal(t.button.attributes['aria-expanded'], 'false');
  t.fire(t.button, 'click');
  t.running.toggle(false);
  assert.equal(t.menu.hidden, true);
  t.fire(t.button, 'click');
  t.fire(t.button, 'click');
  assert.equal(t.running.isOpen(), false);
});

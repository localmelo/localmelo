/** The element makers every console module shares. */
import assert from 'node:assert/strict';
import test from 'node:test';

import {maker, reachable, startOfDay, titleWhenCut} from '../../support/gateway/static/dom.mjs';

const doc = {
  createElement: (tag) => ({
    tag, className: '', textContent: '', attributes: {}, listeners: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(type, handler) { this.listeners[type] = handler; },
  }),
};

test('node sets the class and the text it is given, and nothing else', () => {
  const {node} = maker(doc);
  assert.deepEqual([node('p').className, node('p').textContent], ['', '']);
  assert.equal(node('p', 'lead', 'Hello').className, 'lead');
  assert.equal(node('p', '', 5).textContent, '5');
  assert.equal(node('p', 'lead', null).textContent, '');
});

test('button is a type=button with its handler, and a label or disabled only when asked', () => {
  const {button} = maker(doc);
  const clicks = [];
  const plain = button('quiet-button', 'Stop', () => clicks.push('stop'));
  assert.deepEqual([plain.tag, plain.type, plain.className, plain.textContent], ['button', 'button', 'quiet-button', 'Stop']);
  assert.deepEqual([plain.attributes['aria-label'], plain.disabled], [undefined, undefined]);
  plain.listeners.click();
  assert.deepEqual(clicks, ['stop']);
  const labelled = button('', 'Edit', () => {}, {label: 'Edit this note', disabled: true});
  assert.deepEqual([labelled.attributes['aria-label'], labelled.disabled], ['Edit this note', true]);
});

test('startOfDay is the local midnight in seconds', () => {
  const noon = new Date(2026, 9, 28, 12, 30, 15).getTime() / 1000;
  assert.equal(startOfDay(noon), new Date(2026, 9, 28).getTime() / 1000);
});

// A browser tells a box when its size changes; here the test does, through the one observer the helpers share.
const watching = new Map();
globalThis.ResizeObserver = class {
  constructor(callback) { this.callback = callback; }
  observe(target) { watching.set(target, this); }
  unobserve(target) { watching.delete(target); }
};
const resize = (box) => [...new Set(watching.values())][0].callback([{target: box}]);

const box = (tag, size) => ({
  localName: tag, isConnected: true, title: '', attributes: {}, ...size,
  setAttribute(name, value) { this.attributes[name] = String(value); },
  removeAttribute(name) { delete this.attributes[name]; },
  hasAttribute(name) { return name in this.attributes; },
});

test('a scrolling box is a named tab stop while its content overflows, and neither once it fits', () => {
  const output = box('pre', {scrollHeight: 400, clientHeight: 260, scrollWidth: 300, clientWidth: 300});
  assert.equal(reachable(output, 'Output'), output, 'the box comes back, to be placed where it goes');
  resize(output);
  assert.deepEqual(output.attributes, {tabindex: '0', 'aria-label': 'Output', role: 'group'});
  output.scrollHeight = 200;
  resize(output);
  assert.deepEqual(output.attributes, {});
  const code = box('pre', {scrollHeight: 20, clientHeight: 20, scrollWidth: 900, clientWidth: 600});
  resize(reachable(code, 'Code'));
  assert.equal(code.attributes.tabindex, '0', 'a line that runs sideways needs the keys too');
});

test('a table that scrolls keeps its own role', () => {
  const wide = box('table', {scrollHeight: 30, clientHeight: 30, scrollWidth: 900, clientWidth: 600});
  resize(reachable(wide, 'Table'));
  assert.deepEqual(wide.attributes, {tabindex: '0', 'aria-label': 'Table'});
});

test('a box that left the page is let go', () => {
  const gone = box('pre', {scrollHeight: 400, clientHeight: 260, scrollWidth: 1, clientWidth: 1});
  reachable(gone, 'Output');
  gone.isConnected = false;
  resize(gone);
  assert.deepEqual([gone.attributes, watching.has(gone)], [{}, false]);
});

test('words cut with an ellipsis are the box\'s tooltip, and words that fit have none', () => {
  const chip = box('span', {scrollWidth: 160, clientWidth: 120});
  titleWhenCut(chip, 'localmelo-workspace');
  assert.equal(chip.title, '', 'the chip is read when it is laid out, not while the page redraws');
  resize(chip);
  assert.equal(chip.title, 'localmelo-workspace');
  chip.scrollWidth = 120;
  resize(chip);
  assert.equal(chip.title, '');
});

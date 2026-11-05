/** The element makers every console module shares. */
import assert from 'node:assert/strict';
import test from 'node:test';

import {maker, startOfDay} from '../../support/gateway/static/dom.mjs';

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

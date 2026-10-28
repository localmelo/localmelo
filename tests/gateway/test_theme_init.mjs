import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const root = new URL('../../support/gateway/static/', import.meta.url);
const script = readFileSync(new URL('theme-init.js', root), 'utf8');
const page = readFileSync(new URL('index.html', root), 'utf8');

test('theme script is synchronous and precedes the stylesheet', () => {
  const tag = '<script src="/assets/theme-init.js"></script>';
  assert.ok(page.includes(tag));
  assert.ok(page.indexOf(tag) < page.indexOf('<link rel="stylesheet"'));
});

test('first paint uses only a valid saved theme or system fallback', () => {
  for (const [value, expected] of [['dark', 'dark'], ['light', 'light'], ['system', 'system'], ['invalid', 'system'], [null, 'system']]) {
    const document = {documentElement: {dataset: {}}};
    vm.runInNewContext(script, {document, localStorage: {getItem: () => value}});
    assert.equal(document.documentElement.dataset.theme, expected);
  }
  const document = {documentElement: {dataset: {}}};
  vm.runInNewContext(script, {document, localStorage: {getItem() { throw Error('blocked'); }}});
  assert.equal(document.documentElement.dataset.theme, 'system');
});

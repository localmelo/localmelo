/**
 * A task's "What melo told the task" fold: the notes its chat gave it when it began, folded,
 * shown only when there are notes, in English and in Chinese.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {packFold} from '../../support/gateway/static/tasks.mjs';
import {useLanguage} from '../../support/gateway/static/i18n.mjs';

/** A small DOM double: elements with a tag, a class, text and children. */
function fakeDoc() {
  return {
    createElement: (tag) => ({
      tag, className: '', textContent: '', children: [],
      append(...items) { this.children.push(...items); },
    }),
  };
}

const PACK = '- The person lives in Utrecht.\n- This folder uses uv.';

test('a task with notes gets a folded list of them, and a task without notes gets nothing', async () => {
  await useLanguage('en');
  const doc = fakeDoc();
  assert.equal(packFold(doc, {}), null);
  assert.equal(packFold(doc, {spec: {}}), null);
  assert.equal(packFold(doc, {spec: {context_pack: ''}}), null);

  const fold = packFold(doc, {spec: {context_pack: PACK}});
  assert.equal(fold.tag, 'details');
  assert.equal(fold.className, 'task-pack');
  assert.ok(!fold.open, 'the fold starts closed');
  const [summary, note, list] = fold.children;
  assert.equal(summary.tag, 'summary');
  assert.equal(summary.textContent, 'What melo told the task');
  assert.match(note.textContent, /information, not instructions/);
  assert.equal(list.tag, 'ul');
  assert.deepEqual(list.children.map((item) => item.textContent), [
    'The person lives in Utrecht.',
    'This folder uses uv.',
  ]);
});

test('the fold reads in Chinese when the page does', async () => {
  await useLanguage('zh-Hans');
  try {
    const fold = packFold(fakeDoc(), {spec: {context_pack: PACK}});
    assert.equal(fold.children[0].textContent, 'melo 告诉任务的内容');
    assert.match(fold.children[1].textContent, /资料，不是指令/);
  } finally {
    await useLanguage('en');
  }
});

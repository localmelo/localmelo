/**
 * A task's "What melo told the task" fold: the notes its chat gave it when it began, folded,
 * shown only when there are notes, in English and in Chinese.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {packFold, projectLine} from '../../support/gateway/static/tasks.mjs';
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

test('a task that read the project\'s instruction files says so in one line, and a task that did not says nothing', async () => {
  await useLanguage('en');
  const doc = fakeDoc();
  assert.equal(projectLine(doc, {}), null);
  assert.equal(projectLine(doc, {spec: {}}), null);
  assert.equal(projectLine(doc, {spec: {project_files: [], project_skipped: []}}), null);

  const one = projectLine(doc, {spec: {project_files: [{path: 'AGENTS.md', chars: 2140, cut: false}]}});
  assert.equal(one.tag, 'p');
  assert.equal(one.textContent, "Read the project's instructions: AGENTS.md");

  const many = projectLine(doc, {spec: {project_files: [
    {path: 'CLAUDE.md', chars: 850, cut: false},
    {path: '../AGENTS.md', chars: 4100, cut: true},
  ]}});
  assert.equal(many.textContent,
    "Read the project's instructions: CLAUDE.md, ../AGENTS.md (cut)");

  const big = projectLine(doc, {spec: {project_skipped: ['AGENTS.md']}});
  assert.equal(big.textContent, 'Not read, too large: AGENTS.md');
  const both = projectLine(doc, {spec: {
    project_files: [{path: 'CLAUDE.md', chars: 850, cut: false}], project_skipped: ['AGENTS.md']}});
  assert.equal(both.textContent,
    "Read the project's instructions: CLAUDE.md · Not read, too large: AGENTS.md");
});

test('the project line reads in Chinese when the page does', async () => {
  await useLanguage('zh-Hans');
  try {
    const line = projectLine(fakeDoc(), {spec: {project_files: [{path: 'AGENTS.md', chars: 2140, cut: true}]}});
    assert.equal(line.textContent, '已读取项目说明：AGENTS.md（已截断）');
  } finally {
    await useLanguage('en');
  }
});

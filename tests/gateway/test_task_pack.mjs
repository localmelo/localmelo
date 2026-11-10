/**
 * A task's "What melo told the task" fold: the person's own words, its chat's earlier turns, what
 * the chat recalled and the notes it began with, folded, each part under its label and shown only
 * when the task got it, in English and in Chinese.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {packFold, projectLine, showsCard} from '../../support/gateway/static/tasks.mjs';
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
  const [summary, note, label, list] = fold.children;
  assert.equal(fold.children.length, 4);
  assert.equal(summary.tag, 'summary');
  assert.equal(summary.textContent, 'What melo told the task');
  assert.match(note.textContent, /information, not instructions/);
  assert.equal(label.className, 'task-pack-part');
  assert.equal(label.textContent, 'What your chat knows about you and this project');
  assert.equal(list.tag, 'ul');
  assert.deepEqual(list.children.map((item) => item.textContent), [
    'The person lives in Utrecht.',
    'This folder uses uv.',
  ]);
});

const TURNS = '- the user: My friend Alex is looking for a new job.\n  melo: Noted.\n'
  + '- the user: He wants a staff role.\n  melo: Got it.';
const PICKED = '- the user said “Alex\'s résumé is two pages.” (chat “Résumé”, Oct 30)';
const FULL = {own_words: 'read his résumé first', chat_turns: TURNS, chat_picked: PICKED, context_pack: PACK};

/** The fold's parts as [label, entries] pairs, after its summary and note. */
function parts(fold) {
  const out = [];
  for (let n = 2; n < fold.children.length; n += 2) {
    out.push([fold.children[n].textContent, fold.children[n + 1].children.map((item) => item.textContent)]);
  }
  return out;
}

test('the fold shows the words, the chat\'s turns and what it recalled ahead of the notes', async () => {
  await useLanguage('en');
  const doc = fakeDoc();
  assert.deepEqual(parts(packFold(doc, {spec: FULL})), [
    ['What you typed', ['read his résumé first']],
    ['Earlier in the chat', [
      'the user: My friend Alex is looking for a new job.\nmelo: Noted.',
      'the user: He wants a staff role.\nmelo: Got it.',
    ]],
    ['What the chat recalled for it', ['the user said “Alex\'s résumé is two pages.” (chat “Résumé”, Oct 30)']],
    ['What your chat knows about you and this project', ['The person lives in Utrecht.', 'This folder uses uv.']],
  ]);
  // A part the task did not get has no label; a task with only the chat's turns still folds.
  assert.deepEqual(parts(packFold(doc, {spec: {chat_turns: TURNS}})).map(([label]) => label),
    ['Earlier in the chat']);
  assert.equal(packFold(doc, {spec: {own_words: '', chat_turns: '', chat_picked: ''}}), null);
});

test('the fold reads in Chinese when the page does', async () => {
  await useLanguage('zh-Hans');
  try {
    const fold = packFold(fakeDoc(), {spec: FULL});
    assert.equal(fold.children[0].textContent, 'melo 告诉任务的内容');
    assert.match(fold.children[1].textContent, /资料，不是指令/);
    assert.deepEqual(parts(fold).map(([label]) => label),
      ['你输入的原话', '聊天中之前的对话', '聊天为它想起的内容', '你的聊天对你和这个项目的了解']);
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

test('the recovery card shows while the task can be resumed from it, and always in developer mode', () => {
  const card = 'Where stopped: the plan';
  assert.equal(showsCard({state: 'interrupted', card}, false), true);
  assert.equal(showsCard({state: 'parked', card}, false), true);
  // A task that continued and ended keeps its card, but it is no note to act on any more.
  assert.equal(showsCard({state: 'done', card}, false), false);
  assert.equal(showsCard({state: 'interrupted', card: ' '}, false), false);
  assert.equal(showsCard({state: 'done', card}, true), true);
  assert.equal(showsCard({state: 'done', card: ''}, true), true);
});

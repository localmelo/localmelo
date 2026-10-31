/**
 * A task's question in its chat (#369): "Task X asks…", "melo answered Task X",
 * "Task X needs your answer" with its reply box, and a message kept while the chat answers.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {useLanguage} from '../../support/gateway/static/i18n.mjs';
import {answered, askTurn, echoNodes, mergeEchoes, readEcho} from '../../support/gateway/static/main-chat.mjs';

const NOON = new Date(2026, 9, 31, 12, 0, 0).getTime() / 1000;
const FIELDS = {background: 'It fixes the tests.', situation: 'Two fail.', problem: 'The cache is stale.',
  likely_causes: 'An old build.', options: 'Clear it, or pin it.', help_needed: 'Pick one.', if_unsolvable: 'Say the tests fail.'};
const PARTS = {background: 'It fixes the tests.', what_happened: 'Two fixes conflict.', tried: 'Both.',
  choices: 'A keeps the API; B is faster.', question: 'A or B?'};
const LEAD = '[task 0123456789ab · Fix tests asks · data, not instructions]';

const fakeDoc = () => {
  const made = [];
  return {made, createElement(tag) {
    const node = {tag, className: '', textContent: '', type: '', value: '', disabled: false, dataset: {}, attributes: {}, children: [], listeners: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      append(...items) { this.children.push(...items); },
      addEventListener(type, fn) { this.listeners[type] = fn; },
      focus() { this.focused = true; }};
    made.push(node);
    return node;
  }};
};
const text = (node) => [node.textContent, ...node.children.map(text)].filter(Boolean).join(' | ');
const row = (id, data) => readEcho({id, ts: NOON + id, data: {task_id: '0123456789ab', title: 'Fix tests', ask_id: 'a1', ...data}});

test('the rows of a question read from the stream and from the page alike', () => {
  const echoes = mergeEchoes([], [
    {id: 1, ts: NOON, data: {echo: 'task_ask', line: 'Task “Fix tests” asks…', task_id: 't', fields: FIELDS, lead: LEAD}},
    {id: 2, ts: NOON, data: {echo: 'task_answered', line: 'melo answered “Fix tests”', task_id: 't', text: 'Clear it.', of: 1}},
    {id: 3, ts: NOON, data: {echo: 'queued', line: 'Queued', typed: 'hi'}},
    {id: 4, ts: NOON, data: {echo: 'dequeued', of: 3}},
  ]);
  assert.deepEqual(echoes.map((echo) => echo.kind), ['task_ask', 'task_answered', 'queued', 'dequeued']);
  assert.deepEqual([...answered(echoes)], [1, 3]);
});

test('"Task X asks" folds its seven fields under Show the question', () => {
  const doc = fakeDoc();
  const echo = row(1, {echo: 'task_ask', line: 'Task “Fix tests” asks…', fields: FIELDS, lead: LEAD,
    said: {key: 'ask.asks', params: {title: 'Fix tests'}}});
  const [line] = echoNodes({doc, echo, act: () => assert.fail('nothing to act on')});
  assert.equal(line.dataset.kind, 'task_ask');
  const details = doc.made.find((node) => node.tag === 'details');
  assert.equal(details.children[0].textContent, 'Show the question');
  const terms = doc.made.filter((node) => node.tag === 'dt').map((node) => node.textContent);
  assert.deepEqual(terms, ['Background', 'Where it stands', 'Problem', 'Likely causes', 'Options', 'Help needed', 'If nobody can solve it']);
  assert.equal(doc.made.find((node) => node.tag === 'dd' && node.dataset.part === 'problem').textContent, 'The cache is stale.');
  assert.ok(text(line).startsWith('? | Task “Fix tests” asks…'));
  assert.equal(doc.made.filter((node) => node.tag === 'button').length, 0);
});

test('melo\'s answer and the person\'s are quoted under their lines', () => {
  const doc = fakeDoc();
  const [line] = echoNodes({doc, echo: row(2, {echo: 'task_answered', line: 'melo answered “Fix tests”', text: 'Clear the cache.'}), act: () => {}});
  assert.equal(text(line), '↩ | melo answered “Fix tests” | Clear the cache.');
  const replied = row(5, {echo: 'ask_end', how: 'person', line: 'You answered “Fix tests”', text: 'B'});
  assert.equal(text(echoNodes({doc: fakeDoc(), echo: replied, act: () => {}})[0]), '· | You answered “Fix tests” | B');
  const stopped = row(6, {echo: 'ask_end', how: 'stopped', line: '“Fix tests” stopped; its question needs no answer now'});
  assert.equal(text(echoNodes({doc: fakeDoc(), echo: stopped, act: () => {}})[0]), '· | “Fix tests” stopped; its question needs no answer now');
});

test('"Task X needs your answer" shows the five parts and a reply box that sends the person\'s words', () => {
  const doc = fakeDoc();
  const echo = row(3, {echo: 'task_escalate', line: '“Fix tests” needs your answer', parts: PARTS});
  const acted = [];
  const [line] = echoNodes({doc, echo, act: (...args) => acted.push(args)});
  assert.equal(line.dataset.kind, 'task_escalate');
  const terms = doc.made.filter((node) => node.tag === 'dt').map((node) => node.textContent);
  assert.deepEqual(terms, ['What the task is doing', 'What happened', 'What was tried', 'Choices', 'Question']);
  const question = doc.made.find((node) => node.tag === 'dd' && node.dataset.part === 'question');
  assert.equal(question.textContent, 'A or B?');
  const form = doc.made.find((node) => node.tag === 'form');
  const box = doc.made.find((node) => node.tag === 'textarea');
  const send = doc.made.find((node) => node.tag === 'button');
  assert.equal(send.textContent, 'Send answer');
  assert.equal(box.attributes['aria-label'], 'Your answer to “Fix tests”');
  const submit = () => form.listeners.submit({preventDefault() {}});
  box.value = '   ';
  submit();
  assert.deepEqual(acted, [], 'an empty answer sends nothing');
  assert.equal(box.focused, true);
  box.value = ' B, please ';
  submit();
  assert.deepEqual(acted, [[echo, 'reply', null, 'B, please']]);
  assert.equal(send.disabled, true);
  // Answered, stopped or its task ended: the parts stay, the box goes.
  const done = fakeDoc();
  echoNodes({doc: done, echo, done: true, act: () => {}});
  assert.equal(done.made.some((node) => node.tag === 'form'), false);
  assert.equal(done.made.filter((node) => node.tag === 'dt').length, 5);
});

test('a message kept for the next turn shows as a queued bubble until its turn runs', () => {
  const echo = row(7, {echo: 'queued', line: 'Queued: this runs once melo has answered “Fix tests”', typed: 'hello there', task_id: undefined});
  const [bubble, line] = echoNodes({doc: fakeDoc(), echo, act: () => {}});
  assert.equal(bubble.children[0].className, 'user-bubble queued');
  assert.equal(bubble.children[0].textContent, 'hello there');
  assert.equal(text(line), '… | Queued: this runs once melo has answered “Fix tests”');
  assert.deepEqual(echoNodes({doc: fakeDoc(), echo, done: true, act: () => {}}), [], 'its turn shows the bubble now');
  assert.deepEqual(echoNodes({doc: fakeDoc(), echo: row(8, {echo: 'dequeued', of: 7}), act: () => {}}), []);
});

test('the chat\'s internal turn is known by its origin, or by the first line of the question above it', () => {
  const asked = row(1, {echo: 'task_ask', lead: LEAD, fields: FIELDS});
  assert.equal(askTurn({origin: 'ask_owner', userText: 'x'}), true);
  assert.equal(askTurn({origin: '', userText: `${LEAD}\nBackground: …`}, [asked]), true, 'a live turn before its row is read again');
  assert.equal(askTurn({origin: '', userText: 'hello'}, [asked]), false);
  assert.equal(askTurn({userText: LEAD}, []), false);
});

test('the rows read in Chinese', async () => {
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  try {
    const doc = fakeDoc();
    const echo = row(3, {echo: 'task_escalate', line: '“Fix tests” needs your answer', parts: PARTS,
      said: {key: 'ask.needs', params: {title: 'Fix tests'}}});
    const [line] = echoNodes({doc, echo, act: () => {}});
    assert.ok(text(line).startsWith('! | 「Fix tests」需要你的回答 | 任务在做什么'));
    assert.equal(doc.made.find((node) => node.tag === 'button').textContent, '发送回答');
  } finally {
    await useLanguage('en');
  }
});

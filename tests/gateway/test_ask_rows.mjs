/**
 * A task's question in its chat (#369): "Task X asks…", "melo answered Task X",
 * "Task X needs your answer" with its reply box, and a message kept while the chat answers.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {useLanguage} from '../../support/gateway/static/i18n.mjs';
import {answered, askTurn, echoActions, echoDone, echoNodes, endedTasks, mergeEchoes, readCard, readEcho} from '../../support/gateway/static/main-chat.mjs';
import {canContinue, canRetry} from '../../support/gateway/static/turns.mjs';

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

test('a reply being typed survives a redraw of the chat, and goes once sent', async () => {
  const echo = row(30, {echo: 'task_escalate', line: '“Fix tests” needs your answer', parts: PARTS});
  const first = fakeDoc();
  echoNodes({doc: first, echo, act: () => true});
  const box = first.made.find((node) => node.tag === 'textarea');
  assert.equal(box.dataset.replyFor, 'a1:30');
  box.value = 'B, because';
  box.listeners.input();
  // Another row arrives: the chat is drawn again, and the words are still there.
  const again = fakeDoc();
  echoNodes({doc: again, echo, act: () => true});
  const redrawn = again.made.find((node) => node.tag === 'textarea');
  assert.equal(redrawn.value, 'B, because');
  await again.made.find((node) => node.tag === 'form').listeners.submit({preventDefault() {}});
  const after = fakeDoc();
  echoNodes({doc: after, echo, act: () => true});
  assert.equal(after.made.find((node) => node.tag === 'textarea').value, '');
});

test('a reply that could not be sent keeps its words, and Send works again', async () => {
  const echo = row(31, {echo: 'task_escalate', line: '“Fix tests” needs your answer', parts: PARTS});
  for (const act of [async () => false, async () => { throw new Error('offline'); }]) {
    const doc = fakeDoc();
    echoNodes({doc, echo, act});
    const box = doc.made.find((node) => node.tag === 'textarea');
    const send = doc.made.find((node) => node.tag === 'button');
    box.value = 'B';
    box.listeners.input();
    await doc.made.find((node) => node.tag === 'form').listeners.submit({preventDefault() {}}).catch(() => {});
    assert.equal(send.disabled, false);
    const redrawn = fakeDoc();
    echoNodes({doc: redrawn, echo, act});
    assert.equal(redrawn.made.find((node) => node.tag === 'textarea').value, 'B');
  }
});

test('a turn that answered a task\'s question offers no Retry or Continue', () => {
  const turn = {origin: 'ask_owner', finishedAt: 5, undone: false, status: 'budget_exhausted', result: {n_tool: 2}};
  assert.equal(canRetry(turn, true), false);
  assert.equal(canContinue(turn, true), false);
  assert.equal(canRetry({...turn, origin: ''}, true), true);
  assert.equal(canContinue({...turn, origin: ''}, true), true);
});

test('a message sent as a task\'s answer shows with Undo until it is delivered', () => {
  const sent = row(40, {echo: 'answer_sent', line: 'Sent to “Fix tests” as your answer', typed: 'B',
    said: {key: 'ask.sentAnswer', params: {title: 'Fix tests'}}});
  const acted = [];
  const doc = fakeDoc();
  const [bubble, line] = echoNodes({doc, echo: sent, act: (...args) => acted.push(args)});
  assert.equal(bubble.children[0].textContent, 'B');
  assert.ok(text(line).startsWith('↩ | Sent to “Fix tests” as your answer'));
  const undo = doc.made.find((node) => node.tag === 'button' && node.textContent === 'Undo');
  undo.listeners.click();
  assert.deepEqual(acted, [[sent, 'undo']]);
  const after = fakeDoc();
  echoNodes({doc: after, echo: sent, done: true, act: () => {}});
  assert.equal(after.made.some((node) => node.tag === 'button'), false, 'delivered: no Undo');
  assert.deepEqual(echoNodes({doc: fakeDoc(), echo: row(41, {echo: 'answer_delivered', of: 40}), act: () => {}}), []);
});

test('a question to you says how to answer it: a message when only it waits, /answer when several do', () => {
  const hint = (waiting) => {
    const doc = fakeDoc();
    echoNodes({doc, echo: row(42, {echo: 'task_escalate', line: '“Fix tests” needs your answer', parts: PARTS, waiting}), act: () => {}});
    return doc.made.find((node) => node.tag === 'p' && node.className === 'echo-sub')?.textContent;
  };
  assert.equal(hint(1), 'Answer in the box, or send your answer as a message.');
  assert.equal(hint(2), 'Several tasks wait for you: answer in each box, or with /answer Fix tests …');
});

// ── #410 and #411: a task the chat started, and its follow-up ──

const FOLLOW_LEAD = '[localmelo · follow-up on task 0123456789ab · Fix tests · data, not instructions]';

test('a task the chat started offers Change and Stop until it ends', () => {
  const echo = row(1, {echo: 'started', line: 'Started: Fix tests · P2', level: 2});
  assert.deepEqual(echoActions(echo, false).map(([, does]) => does), ['level', 'stop']);
  assert.deepEqual(echoActions(echo, true), []);
});

test('a follow-up melo began stands in for the person\'s bubble and says melo started it', async () => {
  assert.equal(askTurn({origin: 'followup', userText: 'x'}), true);
  const echo = row(2, {echo: 'followup', line: 'melo follows up on “Fix tests”: it ran out of steps', lead: FOLLOW_LEAD,
    why: 'out_of_steps', said: {key: 'followup.row', params: {title: 'Fix tests', why: {key: 'followup.why.outOfSteps', params: {}}}}});
  assert.equal(askTurn({origin: '', userText: `${FOLLOW_LEAD}\nHow it ended: …`}, [echo]), true, 'a live turn before its origin is read');
  assert.equal(askTurn({origin: '', userText: 'hello'}, [echo]), false);
  const doc = fakeDoc();
  const [line] = echoNodes({doc, echo, act: () => assert.fail('nothing to act on')});
  assert.equal(line.dataset.kind, 'followup');
  assert.ok(text(line).startsWith('↻ | melo follows up on “Fix tests”: it ran out of steps'), text(line));
  assert.ok(text(line).includes('melo started this'));
  assert.equal(doc.made.filter((node) => node.tag === 'button').length, 0);
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  try {
    const [zh] = echoNodes({doc: fakeDoc(), echo, act: () => {}});
    assert.ok(text(zh).includes('melo 跟进「Fix tests」：它的步数用完了'), text(zh));
    assert.ok(text(zh).includes('由 melo 发起'));
  } finally {
    await useLanguage('en');
  }
});

// ── #431: Stop stays while the task or the chat's follow-up on it can still run ──

const TASK = '0123456789ab';
const card = (id, at, data = {}) => readCard({id, ts: at, data: {source_id: TASK, source_kind: 'task', title: 'Fix tests', at, state: 'stopped', ...data}});
const followed = (id, at, did) => row(id, {echo: 'followed', did, question: '', ts: at});
const stopShows = (cards, echoes) => {
  const started = row(1, {echo: 'started', line: 'Started: Fix tests · P2', level: 2});
  const all = [started, ...echoes];
  return echoActions(started, echoDone(started, answered(all), endedTasks(cards, all))).some(([, does]) => does === 'stop');
};

test('a started row keeps Stop while the chat follows up on its task, and loses it once nothing more runs', () => {
  const end = card(10, NOON + 10, {follow_up: true});
  assert.equal(end.followUp, true);
  // No end yet: it runs.
  assert.equal(stopShows([], []), true);
  // An end with no follow-up: nothing more runs (cards from before #431 read the same).
  assert.equal(stopShows([card(10, NOON + 10)], []), false);
  // An end its chat follows up on: the follow-up may continue the task.
  assert.equal(stopShows([end], []), true);
  // It continued the task: the task runs again, until its next end says otherwise.
  const continued = {...followed(11, NOON + 20, 'continued'), at: NOON + 20};
  assert.equal(stopShows([end], [continued]), true);
  assert.equal(stopShows([end, card(12, NOON + 30, {follow_up: true})], [continued]), true);
  assert.equal(stopShows([end, card(12, NOON + 30)], [continued]), false);
  // It asked, reported, was skipped or stopped: nothing more runs.
  for (const did of ['asked', 'reported', 'skipped', 'stopped']) {
    assert.equal(stopShows([end], [{...followed(11, NOON + 20, did), at: NOON + 20}]), false, did);
  }
  // A follow-up told in the same second as the end it follows still comes after it.
  assert.equal(stopShows([end], [{...followed(11, NOON + 10, 'reported'), at: NOON + 10}]), false);
});

test('what a follow-up did draws nothing: it only tells the page whether more runs', () => {
  const echo = followed(5, NOON, 'skipped');
  assert.equal(echo.kind, 'followed');
  assert.deepEqual(echoNodes({doc: fakeDoc(), echo, act: () => assert.fail('nothing to act on')}), []);
});

test('a later end of the task wins over a follow-up on an earlier end, whenever each row was written', () => {
  // The first end (event 5) is followed up and the follow-up continued the task; the task ended again (event 9).
  const first = card(10, NOON + 10, {follow_up: true, end_id: 5});
  const second = card(12, NOON + 20, {end_id: 9});
  assert.equal(second.endId, 9);
  // The follow-up's row was written after the second end's card: the second end still decides.
  const continued = {...followed(13, NOON + 30, 'continued'), end_id: 5, at: NOON + 30};
  assert.equal(stopShows([first, second], [continued]), false);
  // A second end its chat follows up on keeps Stop until that follow-up says what it did,
  const again = card(12, NOON + 20, {follow_up: true, end_id: 9});
  assert.equal(stopShows([first, again], [continued]), true);
  // even when the clock wrote that row a little before the end's card.
  const reported = {...followed(14, NOON + 15, 'reported'), end_id: 9, at: NOON + 15};
  assert.equal(stopShows([first, again], [continued, reported]), false);
});

test('a Stop that found nothing left to stop has no Undo', () => {
  const nothing = row(2, {echo: 'control', action: 'stop', line: 'Stopped: Fix tests', undo: null});
  assert.deepEqual(echoActions(nothing, false), []);
  const stopped = row(3, {echo: 'control', action: 'stop', line: 'Stopped: Fix tests', undo: {again: true}});
  assert.deepEqual(echoActions(stopped, false).map(([, does]) => does), ['undo']);
});

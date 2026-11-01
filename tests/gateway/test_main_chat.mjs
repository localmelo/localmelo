/**
 * The main chat's words and cards (`main-chat.mjs`), without the page.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  byDay, cardNode, cardWords, dayLabel, foldAnswers, isMain, mergeCards, readCard, shortTime, timeline,
} from '../../support/gateway/static/main-chat.mjs';
import * as mainChat from '../../support/gateway/static/main-chat.mjs';

// Noon on a fixed day, on this computer's clock, so the day maths never crosses midnight.
const NOON = new Date(2026, 9, 30, 12, 0, 0).getTime() / 1000;
const HOUR = 3600;
const DAY = 86400;

test('days read as people say them', () => {
  assert.equal(dayLabel(NOON - HOUR, NOON), 'Today');
  assert.equal(dayLabel(NOON - DAY, NOON), 'Yesterday');
  assert.equal(dayLabel(NOON - 2 * DAY, NOON), 'Oct 28');
  assert.equal(dayLabel(NOON - 365 * DAY, NOON), 'Oct 30, 2025');
  const time = (ts) => new Date(ts * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  assert.equal(shortTime(NOON - HOUR, NOON), time(NOON - HOUR), 'Today reads as the clock, as the panel writes it');
  assert.equal(shortTime(NOON - DAY, NOON), 'Yesterday');
  assert.equal(shortTime(0, NOON), '');
});

test('the main chat is the front session', () => {
  assert.equal(isMain({kind: 'front'}), true);
  assert.equal(isMain({kind: 'chat', main: true}), true);
  assert.equal(isMain({kind: 'chat', main: false}), false);
  assert.equal(isMain(null), false);
});

test('a card reads from a page row and from a stream frame alike, once each', () => {
  const row = {id: 3, ts: NOON, at: NOON - 5, state: 'done', source_id: 'a', source_kind: 'chat', title: 'Trip', line: 'Planned.'};
  const frame = {id: 3, session_id: 'm', turn: null, type: 'attention.card', ts: NOON, data: {...row, id: undefined, ts: undefined}};
  assert.deepEqual(readCard(frame), readCard(row));
  assert.deepEqual(readCard(row), {id: 3, at: NOON - 5, state: 'done', sourceId: 'a', sourceKind: 'chat', title: 'Trip', line: 'Planned.'});
  assert.equal(readCard({id: 4}), null, 'A card without a source opens nothing and is left out');
  assert.equal(readCard({id: 5, source_id: 't', source_kind: 'task', state: 'odd'}).state, 'done');
  const merged = mergeCards([readCard(row)], [frame, {...row, id: 1}]);
  assert.deepEqual(merged.map((card) => card.id), [1, 3]);
});

test('the timeline orders turns and cards by time, with a divider per day and one card per source between turns', () => {
  const turn = (startedAt) => ({turn: 1, startedAt});
  const card = (id, at, sourceId) => ({id, at, sourceId, state: 'done', sourceKind: 'chat', title: '', line: ''});
  const items = timeline(
    [turn(NOON - HOUR)],
    [card(1, NOON - DAY, 'a'), card(2, NOON - 2 * HOUR, 'a'), card(3, NOON - 90 * 60, 'b'), card(4, NOON - 10, 'a')],
    NOON,
  );
  assert.deepEqual(
    items.map((item) => item.kind === 'day' ? item.label : item.kind === 'card' ? item.card.id : 'turn'),
    ['Today', 2, 3, 'turn', 4],
    'Yesterday\'s card from "a" folds into its newer one before the turn',
  );
  assert.deepEqual(timeline([], [], NOON), []);
});

test('side chats group newest first under their day', () => {
  const groups = byDay([
    {id: 'old', updated_at: NOON - 2 * DAY},
    {id: 'new', updated_at: NOON - 10},
    {id: 'mid', updated_at: NOON - HOUR},
  ], NOON);
  assert.deepEqual(groups.map((group) => [group.label, group.sessions.map((s) => s.id)]), [['Today', ['new', 'mid']], ['Oct 28', ['old']]]);
});

test('a card says what came back in words, never by colour alone', () => {
  const base = {id: 1, at: NOON, sourceId: 'a', sourceKind: 'chat', title: 'Fix CI', line: ''};
  assert.deepEqual(cardWords({...base, state: 'done'}), {mark: '✓', heading: 'Done: Fix CI', action: 'Open side chat'});
  assert.deepEqual(cardWords({...base, state: 'failed', sourceKind: 'task'}), {mark: '✕', heading: 'Did not finish: Fix CI', action: 'Open task'});
  assert.deepEqual(cardWords({...base, state: 'waiting'}), {mark: '!', heading: 'Needs your approval: Fix CI', action: 'Review'});
  // An answered approval has nothing to review: its words are the answer, and it opens its source.
  assert.deepEqual(cardWords({...base, state: 'answered', line: 'Allowed once'}), {mark: '✓', heading: 'Approval for Fix CI', action: 'Open side chat'});
  assert.deepEqual(cardWords({...base, state: 'answered', line: 'Denied', sourceKind: 'task'}), {mark: '✕', heading: 'Approval for Fix CI', action: 'Open task'});
  assert.equal(cardWords({...base, state: 'answered', line: 'No answer in time'}).mark, '·');
});

test('a task that ran out of steps is a stopped card that offers Continue before Open task', () => {
  const base = {id: 2, at: NOON, sourceId: 't9', sourceKind: 'task', title: 'Fix CI', line: 'Remaining: run the tests'};
  assert.equal(readCard({id: 2, source_id: 't9', source_kind: 'task', state: 'stopped'}).state, 'stopped');
  assert.deepEqual(cardWords({...base, state: 'stopped'}), {mark: '■', heading: 'Stopped: Fix CI', action: 'Open task', more: 'Continue'});
  const made = [];
  const doc = {createElement(tag) {
    const node = {tag, className: '', textContent: '', type: '', dataset: {}, attributes: {}, children: [], listeners: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      append(...items) { this.children.push(...items); },
      addEventListener(type, fn) { this.listeners[type] = fn; }};
    made.push(node);
    return node;
  }};
  const opened = [];
  const resumed = [];
  const card = {...base, state: 'stopped'};
  const node = cardNode({doc, card, open: (value) => opened.push(value), resume: (value, button) => resumed.push([value, button.textContent])});
  assert.equal(node.dataset.state, 'stopped');
  assert.equal(node.attributes['aria-label'], 'Stopped: Fix CI');
  assert.equal(made.find((item) => item.className === 'main-card-line').textContent, 'Remaining: run the tests');
  const foot = made.find((item) => item.className === 'main-card-foot');
  const buttons = foot.children.filter((item) => item.tag === 'button');
  assert.deepEqual(buttons.map((item) => [item.className, item.textContent]), [['outline-button', 'Continue'], ['link-button', 'Open task']]);
  buttons[0].listeners.click();
  buttons[1].listeners.click();
  assert.deepEqual(resumed, [[card, 'Continue']]);
  assert.deepEqual(opened, [card]);
  assert.equal(buttons[1].dataset.focusKey, `card:${card.id}`, 'the panel returns focus to Open when the task it showed is closed');
  // Without a way to continue, the card still opens the task.
  made.length = 0;
  cardNode({doc, card, open: () => {}});
  assert.deepEqual(made.filter((item) => item.tag === 'button').map((item) => item.textContent), ['Open task']);
});

test('a done task card says how code found it ended: ready, not finished with Finish it, or needs you', () => {
  const base = {id: 5, at: NOON, state: 'done', sourceId: 't5', sourceKind: 'task', title: 'Fix CI', line: 'Fixed the check.'};
  const draft = {state: 'draft', reasons: ['tests_missing', 'commit_missing'], question: ''};
  const card = readCard({id: 5, source_id: 't5', source_kind: 'task', state: 'done', title: 'Fix CI', ending: draft});
  assert.deepEqual(card.ending, {...draft, proposed: null, tests: []});
  assert.deepEqual(cardWords({...base, ending: {state: 'ready', reasons: [], question: '', tests: []}}),
    {mark: '✓', heading: 'Ready: Fix CI', action: 'Open task'});
  assert.deepEqual(cardWords(card), {mark: '◐', heading: 'Done, not finished: Fix CI', action: 'Open task', more: 'Finish it: run the tests and commit'});
  const blocked = cardWords({...base, ending: {state: 'blocked', reasons: ['question'], question: 'Which port?', tests: []}});
  assert.equal(blocked.heading, 'Needs you: Fix CI');
  assert.equal(readCard({id: 6, source_id: 't', state: 'done', ending: {state: 'finished'}}).ending, undefined, 'an unknown ending is no ending');
  const made = [];
  const doc = {createElement(tag) {
    const node = {tag, className: '', textContent: '', type: '', dataset: {}, attributes: {}, children: [], listeners: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      append(...items) { this.children.push(...items); },
      addEventListener(type, fn) { this.listeners[type] = fn; }};
    made.push(node);
    return node;
  }};
  const finished = [];
  const node = cardNode({doc, card: {...card, at: NOON}, open: () => {}, resume: (value, button) => finished.push(button.textContent)});
  assert.equal(node.dataset.ending, 'draft');
  assert.equal(made.find((item) => item.className === 'main-card-why').textContent, 'No test ran after the last change · The changes are not committed');
  made.find((item) => item.textContent === 'Finish it: run the tests and commit').listeners.click();
  assert.deepEqual(finished, ['Finish it: run the tests and commit'], 'Finish it is the continue the stopped card uses');
});

test('a card node opens its source', () => {
  const made = [];
  const doc = {createElement(tag) {
    const node = {tag, className: '', textContent: '', type: '', dataset: {}, attributes: {}, children: [], listeners: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      append(...items) { this.children.push(...items); },
      addEventListener(type, fn) { this.listeners[type] = fn; }};
    made.push(node);
    return node;
  }};
  const opened = [];
  const card = {id: 7, at: NOON, state: 'waiting', sourceId: 'a', sourceKind: 'chat', title: 'Fix CI', line: 'Wants to run a command'};
  const node = cardNode({doc, card, open: (value) => opened.push(value)});
  assert.equal(node.dataset.state, 'waiting');
  assert.equal(node.attributes['aria-label'], 'Needs your approval: Fix CI');
  const action = made.find((item) => item.tag === 'button');
  assert.equal(action.textContent, 'Review');
  action.listeners.click();
  assert.deepEqual(opened, [card]);
  const noon = new Date(NOON * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  assert.equal(made.find((item) => item.className === 'main-card-time').textContent, noon);
});

test('echoes stay in the timeline, never folded like cards, and a later echo answers an earlier one', () => {
  const now = 1_800_000_000;
  const echoes = mainChat.mergeEchoes([], [
    {id: 2, ts: now - 50, data: {echo: 'started', task_id: 't', title: 'Slides', level: 2, line: 'Started: Slides · P2'}},
    {id: 3, ts: now - 40, data: {echo: 'control', action: 'pause', task_id: 't', line: 'Paused: Slides'}},
    {id: 4, ts: now - 30, data: {echo: 'undone', task_id: 't', of: 3, line: 'Undone: Slides is going again'}},
    {id: 5, ts: now - 20, data: {echo: 'nonsense'}},
  ]);
  assert.deepEqual(echoes.map((echo) => echo.id), [2, 3, 4]);
  const card = mainChat.readCard({id: 9, ts: now - 10, data: {state: 'done', source_id: 't', source_kind: 'task', title: 'Slides', at: now - 10}});
  const items = mainChat.timeline([], [card], now, echoes);
  assert.deepEqual(items.map((item) => item.kind), ['day', 'echo', 'echo', 'echo', 'card']);
  assert.deepEqual([...mainChat.answered(echoes)], [3]);
  assert.deepEqual(mainChat.echoActions(echoes[1], false), [['Undo', 'undo']]);
  assert.deepEqual(mainChat.echoActions(echoes[1], true), []);
  assert.deepEqual(mainChat.echoActions({kind: 'suggest', action: 'level', level: 1, task_id: 't'}, false), [['Set to P1', 'apply']]);
});

test('a message sent to a task is an echo line with no buttons; a suggested one sends on a tap', () => {
  const [sent] = mainChat.mergeEchoes([], [
    {id: 7, ts: 1_800_000_000, data: {echo: 'sent', how: 'continued', task_id: 't', title: 'Old report', line: 'Continued: Old report'}},
  ]);
  assert.equal(sent.kind, 'sent');
  assert.deepEqual(mainChat.echoActions(sent, false), []);
  assert.deepEqual(mainChat.echoActions({kind: 'suggest', action: 'send', task_id: 't'}, false), [['Send', 'apply']]);
});

// ── An approval card follows its approval ──

const asking = (id, request, at = NOON) => ({id, ts: at, data: {state: 'waiting', source_id: 't', source_kind: 'task', title: 'Fix CI', line: 'Wants to run a command', request_id: request, at}});
const answer = (id, request, line, at = NOON + 5) => ({id, ts: at, data: {state: 'answered', source_id: 't', source_kind: 'task', title: 'Fix CI', line, request_id: request, at}});
const fakeDoc = () => {
  const made = [];
  return {made, createElement(tag) {
    const node = {tag, className: '', textContent: '', type: '', dataset: {}, attributes: {}, children: [], listeners: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      append(...items) { this.children.push(...items); },
      addEventListener(type, fn) { this.listeners[type] = fn; }};
    made.push(node);
    return node;
  }};
};

test('a card reads its request id, and an answered card is a state of its own', () => {
  const card = readCard(asking(3, 'r1'));
  assert.equal(card.requestId, 'r1');
  assert.equal(card.state, 'waiting');
  assert.equal(readCard(answer(4, 'r1', 'Denied')).state, 'answered');
  assert.equal('requestId' in readCard({id: 5, state: 'done', source_id: 'a'}), false, 'a card without one keeps its old shape');
});

test('an answer turns its waiting card into an answered one, in place, and is not a card itself', () => {
  const cards = mergeCards([], [asking(3, 'r1'), asking(4, 'r2', NOON + 1), answer(5, 'r1', 'Allowed for this task')]);
  assert.deepEqual(cards.map((card) => card.state), ['waiting', 'waiting', 'answered'], 'the page keeps what it was told');
  const folded = foldAnswers(cards);
  assert.deepEqual(folded.map((card) => [card.id, card.state, card.line]), [[3, 'answered', 'Allowed for this task'], [4, 'waiting', 'Wants to run a command']]);
  assert.equal(folded[0].at, NOON, 'the card keeps the place and time it had');
  // An answer arriving before its card was read (an older page) shows nothing until the card is there.
  assert.deepEqual(foldAnswers(mergeCards([], [answer(5, 'r9', 'Denied')])), []);
  assert.deepEqual(foldAnswers(mergeCards(mergeCards([], [answer(5, 'r9', 'Denied')]), [asking(2, 'r9')])).map((card) => card.state), ['answered']);
  // Folding changes no card it was given.
  assert.equal(cards[0].state, 'waiting');
});

test('the timeline shows one answered card, with no Review, and a later card of the task replaces it', () => {
  const now = NOON + 60;
  const cards = mergeCards([], [asking(3, 'r1'), answer(4, 'r1', 'Allowed once')]);
  const items = timeline([], cards, now);
  assert.deepEqual(items.map((item) => item.kind), ['day', 'card']);
  assert.equal(items[1].card.state, 'answered');
  const doc = fakeDoc();
  const node = cardNode({doc, card: items[1].card, open: () => {}});
  assert.equal(node.dataset.state, 'answered');
  assert.equal(node.dataset.answer, 'allowed');
  const labels = doc.made.filter((item) => item.tag === 'button').map((item) => item.textContent);
  assert.deepEqual(labels, ['Open task']);
  assert.equal(doc.made.find((item) => item.className === 'main-card-line').textContent, 'Allowed once');
  // The task's Done card comes later: between two turns a source shows only its newest card.
  const done = {id: 9, ts: NOON + 30, data: {state: 'done', source_id: 't', source_kind: 'task', title: 'Fix CI', line: 'Changed the rule.\nTests: 12 passed.', at: NOON + 30}};
  const later = timeline([], mergeCards(cards, [done]), now);
  assert.deepEqual(later.map((item) => item.kind === 'card' ? item.card.state : item.kind), ['day', 'done']);
  // Two approvals of one task: the newest card stands, answered or not.
  const two = mergeCards([], [asking(3, 'r1'), asking(4, 'r2', NOON + 1), answer(5, 'r1', 'Allowed once')]);
  assert.deepEqual(timeline([], two, now).filter((item) => item.kind === 'card').map((item) => [item.card.id, item.card.state]), [[4, 'waiting']]);
});

test('a card written before cards said what they ask does not repeat its heading', () => {
  const doc = fakeDoc();
  const card = readCard({...asking(3, 'r1'), data: {...asking(3, 'r1').data, line: 'Waiting for your approval'}});
  cardNode({doc, card, open: () => {}});
  assert.equal(doc.made.find((item) => item.className === 'main-card-line'), undefined);
});

test('a done card keeps the lines of the answer, and the page lets a line break show', () => {
  const doc = fakeDoc();
  const card = readCard({id: 9, ts: NOON, data: {state: 'done', source_id: 't', source_kind: 'task', title: 'Fix CI', line: 'Changed the rule.\nTests: 12 passed.'}});
  assert.equal(card.line, 'Changed the rule.\nTests: 12 passed.');
  cardNode({doc, card, open: () => {}});
  assert.equal(doc.made.find((item) => item.className === 'main-card-line').textContent, 'Changed the rule.\nTests: 12 passed.');
});

test('a chore proposal offers Save chore, Just once and Cancel until a later line answers it', () => {
  const echoes = mainChat.mergeEchoes([], [
    {id: 2, ts: NOON - 30, data: {echo: 'chore', line: 'Run this every weekday at 8:30?', title: 'Brief', text: 'brief me on my calendar', level: 2,
      when: 'every weekday at 8:30', schedule: {kind: 'weekdays', at: '08:30'}, typed: 'every weekday at 8:30, brief me on my calendar'}},
    {id: 3, ts: NOON - 20, data: {echo: 'chore_saved', chore_id: 'c1', title: 'Brief', of: 2, line: 'Saved: Brief · every weekday at 8:30'}},
    {id: 4, ts: NOON - 10, data: {echo: 'checkin', items: 2, line: 'Your brief is ready.'}},
  ]);
  assert.deepEqual(echoes.map((echo) => echo.kind), ['chore', 'chore_saved', 'checkin']);
  // None of the three names a task: the proposal and the saved line still have their buttons.
  assert.deepEqual(mainChat.echoActions(echoes[0], false), [['Save chore', 'save'], ['Just once', 'once'], ['Cancel', 'cancel']]);
  assert.deepEqual(mainChat.echoActions(echoes[1], false), [['Undo', 'undo']]);
  assert.deepEqual(mainChat.echoActions(echoes[2], false), [], 'the check-in is only a note');
  assert.deepEqual([...mainChat.answered(echoes)], [2]);
  assert.deepEqual(mainChat.echoActions(echoes[0], mainChat.answered(echoes).has(2)), []);
});

test('a chore proposal is drawn as the question, the work it would repeat and three buttons that act', () => {
  const made = [];
  const doc = {createElement: (tag) => {
    const node = {tag, className: '', textContent: '', dataset: {}, children: [], handlers: {}, attributes: {},
      setAttribute(key, value) { this.attributes[key] = value; }, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; }, addEventListener(type, fn) { this.handlers[type] = fn; }};
    made.push(node);
    return node;
  }};
  const echo = mainChat.readEcho({id: 2, ts: NOON, data: {echo: 'chore', line: 'Run this every weekday at 8:30?', title: 'Brief',
    text: 'brief me on my calendar', level: 2, typed: 'every weekday at 8:30, brief me on my calendar'}});
  const acted = [];
  const [bubble, line] = mainChat.echoNodes({doc, echo, act: (...args) => acted.push(args.slice(1))});
  assert.equal(bubble.children[0].textContent, 'every weekday at 8:30, brief me on my calendar');
  const text = (node) => [node.textContent, ...node.children.map(text)].filter(Boolean).join(' | ');
  assert.equal(text(line), '◷ | Run this every weekday at 8:30? | P2 | , normal | brief me on my calendar | Save chore | Just once | Cancel');
  const buttons = made.filter((node) => node.tag === 'button');
  assert.deepEqual(buttons.map((node) => node.className), ['outline-button', 'link-button', 'link-button']);
  buttons.forEach((node) => node.handlers.click());
  assert.deepEqual(acted, [['save'], ['once'], ['cancel']]);
});

test('a reminder proposal names the time and the words, offers Remind me and Cancel, and a "when" question has no buttons', () => {
  const made = [];
  const doc = {createElement: (tag) => {
    const node = {tag, className: '', textContent: '', dataset: {}, children: [], handlers: {}, attributes: {},
      setAttribute(key, value) { this.attributes[key] = value; }, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; }, addEventListener(type, fn) { this.handlers[type] = fn; }};
    made.push(node);
    return node;
  }};
  const echo = mainChat.readEcho({id: 2, ts: NOON, data: {echo: 'chore', reminder: true, line: 'Remind you on Fri Oct 30 at 9:00?',
    title: 'send the report', text: 'Reminder: send the report', work: 'send the report', when: 'once, Fri Oct 30 at 9:00',
    schedule: {kind: 'once', at: '2026-10-30T09:00'}, typed: 'remind me to send the report on Friday at 9'}});
  assert.deepEqual(mainChat.echoActions(echo, false), [['Remind me', 'save'], ['Cancel', 'cancel']]);
  assert.deepEqual(mainChat.echoActions(echo, true), []);
  const acted = [];
  const [bubble, line] = mainChat.echoNodes({doc, echo, act: (...args) => acted.push(args.slice(1))});
  assert.equal(bubble.children[0].textContent, 'remind me to send the report on Friday at 9');
  const text = (node) => [node.textContent, ...node.children.map(text)].filter(Boolean).join(' | ');
  // The question carries the absolute time; under it, the person's words, not the wrapped request.
  assert.equal(text(line), '◷ | Remind you on Fri Oct 30 at 9:00? | send the report | Remind me | Cancel');
  made.filter((node) => node.tag === 'button').forEach((node) => node.handlers.click());
  assert.deepEqual(acted, [['save'], ['cancel']]);
  const ask = mainChat.readEcho({id: 3, ts: NOON, data: {echo: 'ask', ask: 'when', work: 'send the report', line: 'When should I remind you?'}});
  assert.deepEqual(mainChat.echoActions(ask, false), [], 'it is answered by typing a time');
  assert.deepEqual(mainChat.echoActions({kind: 'ask', text: 'x'}, false), [['Start as a task', 'start'], ['Not now', 'dismiss']]);
});

// A tidy-up of the notes while the person was away: one line, what changed, and one Undo.
const sleepRow = (id, data = {}) => ({id, ts: NOON, data: {
  echo: 'sleep', run_id: 'r1', count: 3, variant: 'rules', line: 'While you were away, melo tidied 3 notes',
  items: [{change: 'merged', text: 'You live in Utrecht.'}, {change: 'added', text: 'You prefer short answers.'}], ...data,
}});
const fakeNodes = () => {
  const made = [];
  const doc = {createElement: (tag) => {
    const node = {tag, className: '', textContent: '', dataset: {}, children: [], handlers: {}, attributes: {},
      setAttribute(key, value) { this.attributes[key] = value; }, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; }, addEventListener(type, fn) { this.handlers[type] = fn; }};
    made.push(node);
    return node;
  }};
  return {doc, made};
};
const flat = (node) => [node.textContent, ...node.children.map(flat)].filter(Boolean).join(' | ');

test('/memory undo goes to the newest tidy-up line no line has answered, also past one that was put back already', () => {
  const rows = [sleepRow(5, {run_id: 'a'}), sleepRow(7, {run_id: 'b'})];
  const echoes = (more = []) => mainChat.mergeEchoes([], [...rows, ...more]);
  assert.equal(mainChat.unansweredSleep(echoes()).id, 7);
  // The gateway answered the newest with "already put back": it is settled, and the older one is next.
  const gone = {id: 8, ts: NOON + 9, data: {echo: 'undone', of: 7, run_id: 'b', line: 'Those notes were already put back'}};
  assert.equal(mainChat.unansweredSleep(echoes([gone])).id, 5);
  const both = {id: 9, ts: NOON + 12, data: {echo: 'undone', of: 5, run_id: 'a', line: 'Put back the 3 notes melo tidied'}};
  assert.equal(mainChat.unansweredSleep(echoes([gone, both])), undefined);
  assert.equal(mainChat.unansweredSleep([]), undefined);
});

test('a tidy-up line offers Undo until a later line answers it, and Undo answers with an undone line', () => {
  const echoes = mainChat.mergeEchoes([], [
    sleepRow(5),
    {id: 6, ts: NOON + 5, data: {echo: 'undone', of: 5, run_id: 'r1', line: 'Put back the 3 notes melo tidied'}},
  ]);
  assert.deepEqual(echoes.map((echo) => echo.kind), ['sleep', 'undone']);
  const done = mainChat.answered(echoes);
  assert.deepEqual([...done], [5]);
  // It names no task, and still has its button.
  assert.deepEqual(mainChat.echoActions(echoes[0], false), [['Undo', 'undo']]);
  assert.deepEqual(mainChat.echoActions(echoes[0], done.has(5)), [], 'undone: no button');
  assert.deepEqual(mainChat.echoActions(echoes[1], false), []);
  assert.equal(mainChat.echoDone(echoes[0], done, new Set()), true);
});

test('a tidy-up line is the sentence, the dot and Undo, with what changed folded under it', () => {
  const {doc, made} = fakeNodes();
  const echo = mainChat.readEcho(sleepRow(5, {more: 2}));
  const acted = [];
  const [line] = mainChat.echoNodes({doc, echo, act: (...args) => acted.push(args.slice(1))});
  assert.equal(line.dataset.kind, 'sleep');
  assert.equal(
    flat(line),
    '☾ | While you were away, melo tidied 3 notes | Undo | What changed | Merged | You live in Utrecht. | Added | You prefer short answers. | and 2 more',
  );
  const details = made.find((node) => node.tag === 'details');
  assert.equal(details.className, 'ask-details sleep-details');
  made.filter((node) => node.tag === 'button').forEach((node) => node.handlers.click());
  assert.deepEqual(acted, [['undo']]);
});

test('an undone tidy-up line keeps what changed and loses Undo; its answer is a quiet undone line', () => {
  const {doc, made} = fakeNodes();
  const echo = mainChat.readEcho(sleepRow(5));
  const [line] = mainChat.echoNodes({doc, echo, done: true, act: () => assert.fail('nothing to act on')});
  assert.equal(made.some((node) => node.tag === 'button'), false);
  assert.match(flat(line), /^☾ \| While you were away, melo tidied 3 notes \| What changed \|/);
  const undone = mainChat.readEcho({id: 6, ts: NOON, data: {echo: 'undone', of: 5, line: 'Put back the 3 notes melo tidied'}});
  const [answer] = mainChat.echoNodes({doc: fakeNodes().doc, echo: undone, act: () => {}});
  assert.equal(flat(answer), '↺ | Put back the 3 notes melo tidied');
});

test('a tidy-up line with nothing listed has no folded part', () => {
  const {doc, made} = fakeNodes();
  const [line] = mainChat.echoNodes({doc, echo: mainChat.readEcho(sleepRow(5, {items: []})), act: () => {}});
  assert.equal(made.some((node) => node.tag === 'details'), false);
  assert.equal(flat(line), '☾ | While you were away, melo tidied 3 notes | Undo');
});

test('the card of a reminder says Reminder: and the persons words, done or not', () => {
  const base = {id: 1, at: NOON, sourceId: 'a', sourceKind: 'task', title: 'send the report', line: ''};
  const done = readCard({id: 5, ts: NOON, data: {state: 'done', source_id: 't', source_kind: 'task', title: 'send the report', reminder: true}});
  assert.equal(done.reminder, true);
  assert.equal(readCard({id: 6, ts: NOON, data: {state: 'done', source_id: 't', source_kind: 'task', title: 'x'}}).reminder, undefined);
  assert.deepEqual(cardWords({...base, state: 'done', reminder: true}), {mark: '✓', heading: 'Reminder: send the report', action: 'Open task'});
  assert.deepEqual(cardWords({...base, state: 'failed', reminder: true}), {mark: '✕', heading: 'Reminder: send the report', action: 'Open task'});
  // An approval a reminder's run asks for is still an approval.
  assert.equal(cardWords({...base, state: 'waiting', reminder: true}).heading, 'Needs your approval: send the report');
});

// ── A task's approvals in the chat that owns it ──

const blockedRow = (id, request, at = NOON) => ({id, ts: at, data: {
  state: 'blocked', source_id: 't', source_kind: 'task', title: 'Tidy the repo', line: 'Blocked: deletes files', at,
  request_id: request, step_key: `key-${request}`, name: 'python_exec', preview: "import shutil\nshutil.rmtree('.git')",
  review: {verdict: 'high', categories: ['delete'], reasons: ['Deletes .git.']}, can_allow: true,
}});

test('a step Auto blocked in a task is a card of its own, with the block and Allow once', () => {
  const card = readCard(blockedRow(9, 'r9'));
  assert.deepEqual([card.state, card.requestId, card.stepKey, card.name, card.canAllow], ['blocked', 'r9', 'key-r9', 'python_exec', true]);
  // Every block stays between turns, each with its own Allow once; a later card of the task does not fold them.
  const done = readCard({id: 11, ts: NOON + 2, data: {state: 'done', source_id: 't', source_kind: 'task', title: 'Tidy the repo', at: NOON + 2}});
  const cards = [card, readCard(blockedRow(10, 'r10', NOON + 1)), done];
  assert.deepEqual(timeline([], cards, NOON + 5).filter((item) => item.kind === 'card').map((item) => item.card.id), [9, 10, 11]);
  assert.deepEqual(cardWords(card), {mark: '⊘', heading: 'A step was blocked: Tidy the repo', action: 'Open task'});
  const doc = fakeDoc();
  const allowed = [];
  const node = cardNode({doc, card, open: () => {}, block: {onAllow: () => allowed.push(card.requestId)}});
  assert.equal(node.dataset.state, 'blocked');
  // The block says what and why; the stored line is for the list and the terminal.
  assert.equal(doc.made.find((item) => item.className === 'main-card-line'), undefined);
  // The step as the task's page shows it: the call's preview, as code wrote it.
  assert.equal(doc.made.find((item) => item.className === 'block-step').textContent, "import shutil\nshutil.rmtree('.git')");
  const allow = doc.made.find((item) => item.tag === 'button' && item.textContent === 'Allow once');
  allow.listeners.click();
  assert.deepEqual(allowed, ['r9']);
  // Where the server offers no Allow once, there is none.
  assert.equal(mainChat.cardBlock({...card, canAllow: false}).allow, false);
  const none = fakeDoc();
  cardNode({doc: none, card: {...card, canAllow: false}, open: () => {}});
  assert.equal(none.made.some((item) => item.textContent === 'Allow once'), false);
  assert.equal(mainChat.cardBlock(done), null);
});

test('a waiting task card holds its approval card, and its own button only opens the task', () => {
  const doc = fakeDoc();
  const approval = {tag: 'section', className: 'approval-card', children: []};
  const opened = [];
  const card = readCard(asking(3, 'r1'));
  cardNode({doc, card, open: (value) => opened.push(value), approval});
  const text = doc.made.find((item) => item.className === 'main-card-text');
  assert.equal(text.children.at(-1), approval);
  const buttons = doc.made.filter((item) => item.tag === 'button');
  assert.deepEqual(buttons.map((item) => [item.className, item.textContent]), [['link-button', 'Open task']]);
  buttons[0].listeners.click();
  assert.deepEqual(opened, [card]);
  // An answered card shows no approval card, whatever it is given.
  const answered = fakeDoc();
  cardNode({doc: answered, card: {...card, state: 'answered', answer: 'allowedOnce', line: 'Allowed once'}, open: () => {}, approval});
  assert.equal(answered.made.find((item) => item.className === 'main-card-text').children.includes(approval), false);
});

test('a step that ran on an earlier approval is a row with Revoke, which outlives its task', () => {
  const now = 1_800_000_000;
  const echoes = mainChat.mergeEchoes([], [
    {id: 20, ts: now - 30, data: {echo: 'earlier', task_id: 't', title: 'Check the notes', entry_id: 'e1', approved_at: now - 600, preview: 'print(1)\n',
      line: 'Used your approval from 14:02', said: {key: 'approvals.used', params: {time: '14:02'}}}},
  ]);
  const [used] = echoes;
  assert.equal(used.kind, 'earlier');
  assert.deepEqual(mainChat.echoActions(used, false), [['Revoke', 'revoke']]);
  // The task ended: its other rows lose their buttons, this one keeps Revoke.
  const ended = new Set(['t']);
  assert.equal(mainChat.echoDone(used, new Set(), ended), false);
  assert.equal(mainChat.echoDone({id: 21, kind: 'control', task_id: 't'}, new Set(), ended), true);
  const doc = fakeDoc();
  const acted = [];
  mainChat.echoNodes({doc, echo: used, act: (echo, does) => acted.push([echo.id, does])});
  // The time as the page writes one, not the gateway's stored words.
  const approved = new Date((now - 600) * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  assert.equal(doc.made.find((item) => item.className === 'echo-text').textContent, `Used your approval from ${approved}`);
  assert.equal(doc.made.find((item) => item.className === 'echo-sub').textContent, 'Task “Check the notes” ran print(1)');
  doc.made.find((item) => item.tag === 'button').listeners.click();
  assert.deepEqual(acted, [[20, 'revoke']]);
  // Revoked: the row that answers it takes its button away.
  const both = mainChat.mergeEchoes(echoes, [{id: 22, ts: now, data: {echo: 'revoked', of: 20, task_id: 't', line: 'Revoked: that step asks you again next time'}}]);
  const done = mainChat.answered(both);
  assert.equal(mainChat.echoDone(used, done, new Set()), true);
  assert.deepEqual(mainChat.echoActions(both[1], false), []);
});

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

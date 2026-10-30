/**
 * The main chat's words and cards (`main-chat.mjs`), without the page.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  byDay, cardNode, cardWords, dayLabel, isMain, mergeCards, readCard, shortTime, timeline,
} from '../../support/gateway/static/main-chat.mjs';

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
  assert.deepEqual(cardWords({...base, state: 'waiting'}), {mark: '!', heading: 'Fix CI needs your approval', action: 'Review'});
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
  const card = {id: 7, at: NOON, state: 'waiting', sourceId: 'a', sourceKind: 'chat', title: 'Fix CI', line: 'Waiting for your approval'};
  const node = cardNode({doc, card, open: (value) => opened.push(value)});
  assert.equal(node.dataset.state, 'waiting');
  assert.equal(node.attributes['aria-label'], 'Fix CI needs your approval');
  const action = made.find((item) => item.tag === 'button');
  assert.equal(action.textContent, 'Review');
  action.listeners.click();
  assert.deepEqual(opened, [card]);
  const noon = new Date(NOON * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  assert.equal(made.find((item) => item.className === 'main-card-time').textContent, noon);
});

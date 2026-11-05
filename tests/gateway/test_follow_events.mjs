/**
 * `followEvents` (turns.mjs): the one follower of a session's events. The task
 * card, the worker view and the chat all use it, so its rules are tested here
 * and only the wiring is tested in each of them.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {followEvents, newView} from '../../support/gateway/static/turns.mjs';

class FakeSource {
  constructor(url) { this.url = url; this.closed = false; }
  close() { this.closed = true; }
  emit(frame) { this.onmessage?.({data: typeof frame === 'string' ? frame : JSON.stringify(frame)}); }
  fail() { this.onerror?.({}); }
}

const step = (id) => ({id, turn: 1, type: 'step.started', ts: id, data: {step: 1, attempt: 1}});
const settle = () => new Promise((resolve) => setImmediate(resolve));

function fixture(options = {}) {
  const f = {sources: [], timers: [], frames: [], draws: 0, statuses: [], clock: 100, view: newView('s')};
  f.timer = (ms) => f.timers.find((timer) => timer.ms === ms && !timer.cleared);
  f.fire = (ms) => { const timer = f.timer(ms); timer.cleared = true; return timer.fn(); };
  f.follower = followEvents({
    id: 's/1',
    view: f.view,
    onFrame: (frame, outcome) => f.frames.push([frame, outcome]),
    onStatus: () => f.statuses.push(f.follower.status),
    open: (url) => { const source = new FakeSource(url); f.sources.push(source); return source; },
    schedule: (fn, ms) => { const timer = {fn, ms, cleared: false}; f.timers.push(timer); return timer; },
    cancel: (timer) => { timer.cleared = true; },
    now: () => f.clock,
    ...options,
  });
  return f;
}

test('it opens after the view\'s last id, and asks to be counted as a viewer only when told to', async () => {
  const f = fixture();
  assert.equal(f.sources[0].url, '/api/sessions/s%2F1/events?after=0');
  f.sources[0].emit(step(7));
  f.sources[0].fail();
  await f.fire(1000);
  assert.equal(f.sources[1].url, '/api/sessions/s%2F1/events?after=7', 'a reopened stream resumes after the last id folded');
  assert.equal(fixture({watch: true}).sources[0].url, '/api/sessions/s%2F1/events?after=0&watch=true');
});

test('each new frame is folded into the view and handed on once; unreadable and repeated frames are dropped', () => {
  const f = fixture();
  const [source] = f.sources;
  source.emit(step(2));
  source.emit(step(2));
  source.emit(step(1));
  source.emit('not json');
  source.emit('7');
  source.emit({type: 'session.title', data: {title: 'Named'}});
  assert.deepEqual(f.frames.map(([frame]) => frame.id), [2, undefined]);
  assert.equal(f.view.lastEventId, 2);
  assert.deepEqual(f.frames[1][1], {title: 'Named'}, 'a frame without an id is folded, and the outcome goes with it');
});

test('new frames are drawn together, and only when asked for', () => {
  const draws = [];
  const f = fixture({onDraw: () => draws.push(1)});
  f.sources[0].emit(step(2));
  f.sources[0].emit(step(3));
  assert.equal(draws.length, 0);
  assert.equal(f.timers.filter((timer) => timer.ms === 150).length, 1, 'one timer for the two frames');
  f.fire(150);
  assert.equal(draws.length, 1);
  f.sources[0].emit(step(4));
  assert.ok(f.timer(150), 'the next frame starts the next draw');
  assert.equal(fixture().timers.length, 0, 'with no onDraw nothing is scheduled');
});

test('a page that is hidden keeps following: the page\'s visibility is not consulted', () => {
  const listeners = [];
  const before = globalThis.document;
  globalThis.document = {visibilityState: 'hidden', hidden: true, addEventListener: (type, fn) => listeners.push(fn)};
  try {
    const f = fixture();
    for (const fn of listeners) fn();
    f.sources[0].emit(step(2));
    assert.equal(f.sources[0].closed, false);
    assert.equal(f.frames.length, 1);
  } finally {
    globalThis.document = before;
  }
});

test('a stream that ends is opened again after 1, 2, 4, 8, 16, 30 and then 30 seconds; an open stream starts over', async () => {
  const f = fixture();
  const waited = [];
  for (let attempt = 0; attempt < 7; attempt += 1) {
    f.sources.at(-1).fail();
    assert.equal(f.sources.at(-1).closed, true);
    const timer = f.timers.findLast((entry) => !entry.cleared);
    waited.push(timer.ms / 1000);
    timer.cleared = true;
    await timer.fn();
  }
  assert.deepEqual(waited, [1, 2, 4, 8, 16, 30, 30]);
  assert.equal(f.sources.length, 8);
  f.sources.at(-1).onopen();
  f.sources.at(-1).fail();
  assert.ok(f.timer(1000), 'once a stream has opened, the next wait is the shortest again');
});

test('while it waits it says so, and when it will try again', async () => {
  const f = fixture();
  assert.equal(f.follower.status, 'live');
  f.sources[0].fail();
  assert.equal(f.follower.status, 'paused');
  assert.equal(f.follower.retryAt, 101);
  assert.deepEqual(f.statuses, ['paused']);
  f.fire(1000);
  await settle();
  assert.equal(f.follower.status, 'live', 'a stream being opened is live');
  f.sources[1].onopen();
  assert.deepEqual(f.statuses, ['paused', 'live', 'live']);
});

test('the caller is asked at each end: false stops the follow at once', async () => {
  const f = fixture({onEnd: async () => false});
  f.sources[0].fail();
  const wait = f.timer(1000);
  await settle();
  assert.equal(wait.cleared, true, 'the wait is cancelled');
  assert.equal(f.sources.length, 1);
});

test('a new stream waits for the caller\'s answer, and a caller that fails to answer does not stop the follow', async () => {
  let release;
  let answer = new Promise((resolve) => { release = resolve; });
  const f = fixture({onEnd: () => answer});
  f.sources[0].fail();
  const reopening = f.fire(1000);
  assert.equal(f.sources.length, 1, 'nothing is opened before the caller has answered');
  release(true);
  await reopening;
  assert.equal(f.sources.length, 2);

  answer = Promise.reject(new Error('could not tell'));
  f.sources[1].fail();
  await f.fire(2000);
  assert.equal(f.sources.length, 3);
  answer = () => { throw new Error('threw'); };
  const throwing = fixture({onEnd: answer});
  throwing.sources[0].fail();
  await throwing.fire(1000);
  assert.equal(throwing.sources.length, 2);
});

test('retryNow opens at once, once, and only while it waits', async () => {
  const f = fixture();
  await f.follower.retryNow();
  assert.equal(f.sources.length, 1, 'a live stream is left alone');
  f.sources[0].fail();
  const wait = f.timer(1000);
  await f.follower.retryNow();
  assert.equal(wait.cleared, true);
  assert.equal(f.sources.length, 2);
  await f.follower.retryNow();
  assert.equal(f.sources.length, 2);
});

test('closing stops everything: the stream, the wait, the draw, and what a late callback would do', async () => {
  const f = fixture({onDraw: () => {}});
  const [first] = f.sources;
  first.emit(step(2));
  const draw = f.timer(150);
  first.fail();
  const wait = f.timer(1000);
  f.follower.close();
  assert.equal(draw.cleared, true);
  assert.equal(wait.cleared, true);
  await wait.fn();
  first.emit(step(3));
  first.fail();
  assert.equal(f.sources.length, 1, 'nothing is opened after a close');
  assert.equal(f.frames.length, 1, 'a frame of a closed stream is dropped');

  const g = fixture();
  g.follower.close();
  assert.equal(g.sources[0].closed, true);
  g.sources[0].emit(step(2));
  assert.equal(g.frames.length, 0);
  g.follower.close();
});

test('where there is no EventSource it returns null, so the caller reads instead', () => {
  assert.equal(followEvents({id: 's', view: newView('s'), onFrame: () => {}, open: () => null}), null);
});

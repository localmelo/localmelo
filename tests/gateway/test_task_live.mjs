import assert from 'node:assert/strict';
import test from 'node:test';
import {taskLive} from '../../support/gateway/static/task-live.mjs';

class FakeSource {
  constructor(url) { this.url = url; this.closed = false; }
  close() { this.closed = true; }
  emit(frame) { this.onmessage?.({data: JSON.stringify(frame)}); }
  fail() { this.onerror?.({}); }
}

const started = {id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'Objective'}};
const task = (extra = {}) => ({task_id: 't', state: 'running', started_at: 5, ...extra});
const settle = () => new Promise((resolve) => setImmediate(resolve));

function fixture({events = [started], noSource = false} = {}) {
  const f = {sources: [], timers: [], calls: [], draws: 0, listeners: new Set(), events, missing: false, status: 'running'};
  f.doc = {
    visibilityState: 'visible',
    addEventListener: (type, fn) => f.listeners.add(fn),
    removeEventListener: (type, fn) => f.listeners.delete(fn),
  };
  f.show = (state) => { f.doc.visibilityState = state; for (const fn of f.listeners) fn(); };
  f.timer = (ms) => f.timers.find((t) => t.ms === ms && !t.cleared);
  f.bumps = 0;
  f.live = taskLive({
    api: async (path) => {
      f.calls.push(path);
      if (f.missing) throw Object.assign(new Error('not found'), {status: 404});
      if (f.broken) throw new Error('down');
      return {turns: [{turn: 1, status: f.status, events: f.events}], has_earlier: false};
    },
    changed: () => { f.draws++; },
    open: noSource ? () => null : (url) => { const source = new FakeSource(url); f.sources.push(source); return source; },
    schedule: (fn, ms) => { const timer = {fn, ms, cleared: false}; f.timers.push(timer); return timer; },
    cancel: (timer) => { timer.cleared = true; },
    statusRow: {bump: () => { f.bumps++; }},
    doc: f.doc,
  });
  return f;
}

const step = (id) => ({id, turn: 1, type: 'step.started', ts: id, data: {step: 1, attempt: 1}});
const toolStarted = (id, call = 'c') => ({id, turn: 1, type: 'tool.started', ts: id, data: {step: 1, call_id: call, name: 'shell_exec', arguments: {}, preview: 'ls'}});
const toolFinished = (id, call = 'c') => ({id, turn: 1, type: 'tool.finished', ts: id, data: {step: 1, call_id: call, name: 'shell_exec', status: 'success', output: 'ok'}});

test('a task that never started shows nothing live and reads nothing', async () => {
  const f = fixture();
  f.live.sync(task({state: 'queued', started_at: null}));
  await settle();
  assert.deepEqual(f.calls, []);
  assert.equal(f.live.state.view, null);
  assert.equal(f.sources.length, 0);
});

test('a running task is read once, then a step and a tool call are folded in without another read', async () => {
  const f = fixture();
  f.live.sync(task());
  await settle();
  assert.deepEqual(f.calls, ['/sessions/t/turns?limit=50&events=true']);
  assert.equal(f.sources.length, 1);
  assert.equal(f.sources[0].url, '/api/sessions/t/events?after=1', 'the stream resumes after the last event read');
  const [source] = f.sources;
  const drawn = f.draws;
  source.emit(step(2));
  source.emit(toolStarted(3));
  source.emit(toolStarted(3));
  const items = f.live.state.view.turns[0].items;
  assert.deepEqual(items.map((item) => item.kind), ['tool'], 'a repeated id is the same event');
  assert.equal(f.draws, drawn, 'frames are drawn together, not one by one');
  f.timer(150).fn();
  assert.equal(f.draws, drawn + 1);
  source.emit(toolFinished(4));
  assert.equal(items[0].status, 'success', 'the open tool call is completed in place');
  assert.equal(f.calls.length, 1, 'no second read of the task');
  assert.equal(f.bumps, 3, 'each new frame asks the status row for a refetch');
  f.live.sync(task());
  assert.equal(f.sources.length, 1, 'the next poll does not open a second stream');
});

test('a task waiting for approval keeps its stream, and a frame from an old stream is dropped after the task is left', async () => {
  const f = fixture();
  f.live.sync(task({state: 'held'}));
  await settle();
  assert.equal(f.sources.length, 1);
  f.sources[0].emit(toolStarted(2));
  f.live.leave();
  assert.equal(f.sources[0].closed, true);
  assert.equal(f.listeners.size, 0);
  f.sources[0].emit(toolStarted(3, 'late'));
  assert.equal(f.live.state.view, null);
});

test('a finished task is read once and never followed', async () => {
  for (const state of ['done', 'failed', 'cancelled', 'expired', 'interrupted', 'parked']) {
    const f = fixture();
    f.live.sync(task({state}));
    await settle();
    f.live.sync(task({state}));
    await settle();
    assert.equal(f.calls.length, 1, state);
    assert.equal(f.sources.length, 0, state);
    assert.equal(f.listeners.size, 0, state);
    assert.equal(f.live.state.view.turns[0].userText, 'Objective');
  }
});

test('a task that finishes closes the stream and reads its record once more', async () => {
  const f = fixture();
  f.live.sync(task());
  await settle();
  f.sources[0].emit(step(2));
  f.events = [started, step(2), toolStarted(3), toolFinished(4)];
  f.live.sync(task({state: 'done'}));
  assert.equal(f.sources[0].closed, true);
  assert.equal(f.listeners.size, 0);
  await settle();
  assert.equal(f.calls.length, 2, 'one catch-up read, so frames the stream had not delivered are not lost');
  assert.equal(f.live.state.view.turns[0].items.length, 1);
  assert.equal(f.live.state.view.lastEventId, 4);
  assert.equal(f.sources.length, 1);
});

test('a task that stops and is resumed is followed again after the last id', async () => {
  const f = fixture();
  f.live.sync(task());
  await settle();
  f.sources[0].emit(step(2));
  f.events = [started, step(2)];
  f.live.sync(task({state: 'parked'}));
  await settle();
  assert.equal(f.sources[0].closed, true);
  f.live.sync(task({state: 'running'}));
  assert.equal(f.sources.length, 2);
  assert.equal(f.sources[1].url, '/api/sessions/t/events?after=2');
});

test('a hidden tab stops following and a shown one catches up from the last id', async () => {
  const f = fixture();
  f.live.sync(task());
  await settle();
  f.sources[0].emit(step(2));
  f.show('hidden');
  assert.equal(f.sources[0].closed, true);
  f.show('visible');
  assert.equal(f.sources.length, 2);
  assert.equal(f.sources[1].url, '/api/sessions/t/events?after=2');
  f.show('visible');
  assert.equal(f.sources.length, 2, 'a stream that is open is not opened twice');
});

test('a stream that fails is opened again by the next poll, after the last id', async () => {
  const f = fixture();
  f.live.sync(task());
  await settle();
  f.sources[0].emit(step(2));
  f.sources[0].fail();
  assert.equal(f.sources[0].closed, true);
  f.live.sync(task());
  assert.equal(f.sources.length, 2);
  assert.equal(f.sources[1].url, '/api/sessions/t/events?after=2');
});

test('leaving or opening another task drops a read that answers late', async () => {
  const f = fixture();
  f.live.sync(task());
  f.live.leave();
  await settle();
  assert.equal(f.live.state.view, null);
  assert.equal(f.sources.length, 0, 'no stream is opened for a task that was left');
  f.live.sync(task({task_id: 'a'}));
  f.live.sync(task({task_id: 'b'}));
  await settle();
  assert.equal(f.live.state.id, 'b');
  assert.equal(f.live.state.view.id, 'b');
  assert.deepEqual(f.sources.map((source) => source.url), ['/api/sessions/b/events?after=1']);
});

test('a task without a session shows nothing, and a failed read says so and is tried again', async () => {
  const f = fixture();
  f.missing = true;
  f.live.sync(task());
  await settle();
  assert.equal(f.live.state.view, null);
  assert.equal(f.live.state.error, '');
  f.missing = false; f.broken = true;
  f.live.sync(task());
  await settle();
  assert.equal(f.live.state.error, 'Could not load the live view.');
  f.broken = false;
  f.live.sync(task());
  await settle();
  assert.equal(f.live.state.error, '');
  assert.equal(f.live.state.view.turns.length, 1);
});

test('without an EventSource a running task is read again on each poll', async () => {
  const f = fixture({noSource: true});
  f.live.sync(task());
  await settle();
  f.events = [started, step(2)];
  f.live.sync(task());
  await settle();
  assert.equal(f.calls.length, 2, 'one read for the view and one for the poll that follows');
  assert.equal(f.live.state.view.lastEventId, 2);
});

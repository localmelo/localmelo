import assert from 'node:assert/strict';
import test from 'node:test';
import {foldWorkerTurns, workerHistory} from '../../support/gateway/static/worker-history.mjs';

test('folds stored events exactly once and keeps a stored answer', () => {
  const view = foldWorkerTurns('worker', [{turn: 1, user_text: '', answer: 'Saved', status: 'completed', finished_at: 2, events: [
    {id: 3, turn: 1, type: 'assistant.message', data: {thinking: 'Second'}},
    {id: 2, turn: 1, type: 'assistant.message', data: {thinking: 'First'}},
    {id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'Brief'}},
  ]}]);
  assert.equal(view.turns.length, 1);
  assert.equal(view.turns[0].userText, 'Brief');
  assert.equal(view.turns[0].answer, 'Saved');
  assert.deepEqual(view.turns[0].items.map((item) => item.text), ['First', 'Second']);
});

test('filters and cursor are sent server-side, duplicate rows are suppressed', async () => {
  const calls = [];
  const history = workerHistory(async (path) => {
    calls.push(path);
    return calls.length === 1 ? {sessions: [{id: 'a'}], next_cursor: 'opaque'} : {sessions: [{id: 'a'}, {id: 'b'}], next_cursor: null};
  }, () => {});
  history.state.filters = {status: 'completed', source: 'dsh', parent: 'r 1'};
  await history.list();
  await history.list({more: true});
  const query = new URLSearchParams(calls[1].split('?')[1]);
  assert.equal(query.get('cursor'), 'opaque');
  assert.equal(query.get('parent_run'), 'r 1');
  assert.equal(query.get('status'), 'completed');
  assert.deepEqual(history.state.rows.map((row) => row.id), ['a', 'b']);
});

test('leaving invalidates late worker responses', async () => {
  let resolve;
  const history = workerHistory(() => new Promise((done) => { resolve = done; }), () => {});
  const request = history.list();
  history.leave();
  resolve({sessions: [{id: 'stale'}]});
  await request;
  assert.deepEqual(history.state.rows, []);
});

test('detail rejects chats and distinguishes missing and unavailable history', async () => {
  for (const [result, expected] of [
    [{session: {kind: 'chat'}}, 'Worker run not found.'],
    [Object.assign(new Error('gone'), {status: 404}), 'Worker run not found.'],
    [Object.assign(new Error('unavailable'), {status: 503}), 'Worker history is unavailable.'],
  ]) {
    const history = workerHistory(async (path) => {
      if (path.includes('/turns?')) return {turns: []};
      if (result instanceof Error) throw result;
      return result;
    }, () => {});
    await history.detail('w');
    assert.equal(history.state.error, expected);
    assert.equal(history.state.view, null);
  }
});

test('new selection wins over a slow detail response', async () => {
  let oldDetail;
  const history = workerHistory((path) => {
    if (path === '/sessions/old') return new Promise((resolve) => { oldDetail = resolve; });
    if (path.includes('/turns?')) return Promise.resolve({turns: []});
    return Promise.resolve({session: {id: 'new', kind: 'worker'}});
  }, () => {});
  const old = history.detail('old');
  await history.detail('new');
  oldDetail({session: {id: 'old', kind: 'worker'}});
  await old;
  assert.equal(history.state.session.id, 'new');
});

test('the open list and an unfinished run are read again on the scheduler tick, a finished run is not', async () => {
  let ticks = [];
  const every = (fn, ms) => { const tick = {fn, ms, stopped: false}; ticks.push(tick); return () => { tick.stopped = true; }; };
  const calls = [];
  let run = {id: 'w', kind: 'worker', status: 'running'};
  const history = workerHistory(async (path) => {
    calls.push(path);
    if (path.startsWith('/sessions?')) return {sessions: [run], next_cursor: null};
    if (path.includes('/turns')) return {turns: [{turn: 1, status: run.status, events: []}], has_earlier: false};
    return {session: run};
  }, () => {}, every);
  await history.list();
  assert.equal(ticks.length, 1);
  assert.equal(ticks[0].ms, 3000);
  run = {...run, status: 'completed'};
  await ticks[0].fn();
  assert.equal(history.state.rows[0].status, 'completed', 'the list follows the run');
  await history.detail('w');
  assert.equal(ticks.length, 1, 'one scheduler for the whole view');
  const reads = calls.length;
  await ticks[0].fn();
  assert.equal(calls.length, reads, 'a finished run is not read again');
  run = {...run, status: 'running'};
  await history.detail('w');
  run = {...run, status: 'completed'};
  await ticks[0].fn();
  assert.equal(history.state.session.status, 'completed', 'an unfinished run follows the store');
  history.leave();
  assert.equal(ticks[0].stopped, true);
});

test('a poll that fails or answers late draws nothing', async () => {
  let ticks = [];
  const every = (fn) => { ticks.push(fn); return () => {}; };
  let fail = false;
  let release;
  let draws = 0;
  const history = workerHistory(async () => {
    if (fail) throw new Error('down');
    if (release) await new Promise((resolve) => { release = resolve; });
    return {sessions: [{id: 'a', status: 'running'}], next_cursor: null};
  }, () => { draws++; }, every);
  await history.list();
  const drawn = draws;
  fail = true;
  await ticks[0]();
  assert.equal(draws, drawn);
  assert.equal(history.state.error, '', 'a failed poll shows no error; the next tick tries again');
  fail = false;
  release = () => {};
  const late = ticks[0]();
  history.leave();
  release();
  await late;
  assert.equal(draws, drawn, 'a poll answered after leaving is dropped');
});

// ── Following an unfinished run ──

class FakeSource {
  constructor(url) { this.url = url; this.closed = false; }
  close() { this.closed = true; }
  emit(frame) { this.onmessage?.({data: JSON.stringify(frame)}); }
  end() { this.onerror?.({}); }
}

function followFixture({status = 'running', busy = true, events = [{id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'Brief'}}]} = {}) {
  const f = {sources: [], timers: [], calls: [], draws: 0, status, busy, listeners: new Set(), statusCalls: []};
  f.doc = {
    visibilityState: 'visible',
    addEventListener: (type, fn) => f.listeners.add(fn),
    removeEventListener: (type, fn) => f.listeners.delete(fn),
  };
  f.show = (state) => { f.doc.visibilityState = state; for (const fn of f.listeners) fn(); };
  f.timer = (ms) => f.timers.find((t) => t.ms === ms && !t.cleared);
  f.statusRow = {follow: (id) => f.statusCalls.push(['follow', id]), bump: () => f.statusCalls.push(['bump']), clear: () => f.statusCalls.push(['clear'])};
  f.history = workerHistory(async (path) => {
    f.calls.push(path);
    if (f.gone && path === '/sessions/w') throw Object.assign(new Error('gone'), {status: 404});
    if (path.includes('/turns')) return {turns: [{turn: 1, status: f.status, events}], has_earlier: false};
    return {session: {id: 'w', kind: 'worker', status: f.status, busy: f.busy}};
  }, () => { f.draws++; }, () => () => {}, {
    open: (url) => { const source = new FakeSource(url); f.sources.push(source); return source; },
    schedule: (fn, ms) => { const timer = {fn, ms, cleared: false}; f.timers.push(timer); return timer; },
    cancel: (timer) => { timer.cleared = true; },
    statusRow: f.statusRow, doc: f.doc,
  });
  return f;
}

const message = (id, extra = {}) => ({id, turn: 1, type: 'assistant.message', ts: id, data: {step: 1, content: 'text ' + id, tool_calls: [], usage: {}}, ...extra});

test('an unfinished run is read once, then new events are folded in without another read', async () => {
  const f = followFixture();
  await f.history.detail('w');
  const reads = f.calls.length;
  assert.equal(f.sources.length, 1);
  assert.equal(f.sources[0].url, '/api/sessions/w/events?after=1', 'the stream resumes after the last event read');
  const [source] = f.sources;
  const drawn = f.draws;
  source.emit({id: 2, turn: 1, type: 'step.started', ts: 2, data: {step: 1, attempt: 1}});
  source.emit({id: 3, turn: 1, type: 'tool.started', ts: 3, data: {step: 1, call_id: 'c', name: 'shell_exec', arguments: {}, preview: 'ls'}});
  source.emit({id: 3, turn: 1, type: 'tool.started', ts: 3, data: {step: 1, call_id: 'c', name: 'shell_exec', arguments: {}, preview: 'ls'}});
  const items = f.history.state.view.turns[0].items;
  assert.deepEqual(items.map((item) => item.kind), ['tool'], 'a repeated id is the same event');
  assert.equal(f.history.state.eventCounts[1], 3);
  assert.equal(f.draws, drawn, 'frames are drawn together, not one by one');
  f.timer(150).fn();
  assert.equal(f.draws, drawn + 1);
  source.emit({id: 4, turn: 1, type: 'tool.finished', ts: 4, data: {step: 1, call_id: 'c', name: 'shell_exec', status: 'success', output: 'x'}});
  assert.equal(items[0].status, 'success', 'the open tool call is completed in place');
  assert.equal(f.calls.length, reads, 'no read of the run');
  assert.ok(f.statusCalls.some(([name, id]) => name === 'follow' && id === 'w'));
  assert.ok(f.statusCalls.filter(([name]) => name === 'bump').length >= 3, 'each new frame asks for a status refetch');
});

test('a new type of event marks the view and a known one does not', async () => {
  const f = followFixture();
  await f.history.detail('w');
  const [source] = f.sources;
  for (const [index, type] of ['context_budget', 'note', 'inbox.accepted', 'memory', 'recall'].entries()) {
    source.emit({id: 2 + index, turn: 1, type, ts: 1, data: {}});
  }
  assert.equal(f.history.state.unsupported, false);
  source.emit({id: 20, turn: 1, type: 'future.event', ts: 1, data: {}});
  assert.equal(f.history.state.unsupported, true);
});

test('the end of the stream reads the record once and stops when the run is finished', async () => {
  const f = followFixture();
  await f.history.detail('w');
  f.status = 'completed'; f.busy = false;
  const [source] = f.sources;
  source.end();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(source.closed, true);
  assert.equal(f.history.state.session.status, 'completed');
  assert.equal(f.sources.length, 1, 'no stream is opened for a finished run');
  assert.equal(f.timer(2000), undefined);
  assert.equal(f.listeners.size, 0, 'the visibility handler is gone');
});

test('a stream that ends while the run still holds its lease is opened again after the last id', async () => {
  const f = followFixture();
  await f.history.detail('w');
  f.sources[0].emit(message(2));
  f.sources[0].end();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.sources.length, 1);
  f.timer(2000).fn();
  assert.equal(f.sources.length, 2);
  assert.equal(f.sources[1].url, '/api/sessions/w/events?after=2');
});

test('a run that was deleted ends the follow instead of retrying', async () => {
  const f = followFixture();
  await f.history.detail('w');
  f.gone = true;
  f.sources[0].end();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.timer(2000), undefined);
  assert.equal(f.listeners.size, 0);
});

test('a hidden tab stops following and a shown one catches up from the last id', async () => {
  const f = followFixture();
  await f.history.detail('w');
  f.sources[0].emit(message(2));
  f.show('hidden');
  assert.equal(f.sources[0].closed, true);
  f.show('visible');
  assert.equal(f.sources.length, 2);
  assert.equal(f.sources[1].url, '/api/sessions/w/events?after=2');
  f.show('visible');
  assert.equal(f.sources.length, 2, 'a stream that is open is not opened twice');
});

test('a run that is already finished, or has no live lease, is not followed', async () => {
  for (const options of [{status: 'completed', busy: false}, {status: 'running', busy: false}]) {
    const f = followFixture(options);
    await f.history.detail('w');
    assert.equal(f.sources.length, 0);
    assert.equal(f.listeners.size, 0);
  }
});

test('leaving or choosing another run closes the stream, the timers and the status row', async () => {
  const f = followFixture();
  await f.history.detail('w');
  f.sources[0].emit(message(2));
  const pendingDraw = f.timer(150);
  f.history.leave();
  assert.equal(f.sources[0].closed, true);
  assert.equal(pendingDraw.cleared, true);
  assert.equal(f.listeners.size, 0);
  assert.deepEqual(f.statusCalls.at(-1), ['clear']);
  f.sources[0].emit({id: 3, turn: 1, type: 'tool.started', ts: 3, data: {step: 1, call_id: 'c', name: 'shell_exec', arguments: {}}});
  assert.equal(f.history.state.view.turns[0].items.length, 0, 'a frame from a closed stream is dropped');
  await f.history.detail('w');
  await f.history.list();
  assert.equal(f.sources[1].closed, true);
});

test('without an EventSource the unfinished run is read again on the poll', async () => {
  const ticks = [];
  const calls = [];
  const history = workerHistory(async (path) => {
    calls.push(path);
    if (path.includes('/turns')) return {turns: [{turn: 1, status: 'running', events: []}], has_earlier: false};
    return {session: {id: 'w', kind: 'worker', status: 'running'}};
  }, () => {}, (fn) => { ticks.push(fn); return () => {}; }, {open: () => null, doc: undefined});
  await history.detail('w');
  const reads = calls.length;
  await ticks[0]();
  assert.ok(calls.length > reads);
});

test('the fold shows a muted line for notes, accepted messages, memory and recall, and none for the budget', () => {
  const rows = [{turn: 1, status: 'running', events: [
    {id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'Brief'}},
    {id: 2, turn: 1, type: 'context_budget', data: {measured_prompt_tokens: 10, input_limit: 100}},
    {id: 3, turn: 1, type: 'note', data: {step: 2, kind: 'plan', text: 'First\n  read   the file'}},
    {id: 4, turn: 1, type: 'inbox.accepted', data: {step: 2, text: 'also check b'}},
    {id: 5, turn: 1, type: 'memory', data: {action: 'admit', kind: 'digest'}},
    {id: 6, turn: 1, type: 'recall', data: {step: 3, trigger: 'step', hits: [{}, {}]}},
    {id: 7, turn: 1, type: 'recall', data: {step: 3, hits: [{}]}},
  ]}];
  const items = foldWorkerTurns('w', rows).turns[0].items;
  assert.deepEqual(items.map((item) => item.message), [
    'Note (plan): First read the file', 'Message accepted: also check b', 'Memory admit: digest',
    'Recalled 2 items (step)', 'Recalled 1 item',
  ]);
  assert.ok(items.every((item) => item.kind === 'fact'));
  assert.equal(items[0].step, 2);
});

test('a terminal chat is listed under its own kind, opens, and follows while its turn runs', async () => {
  const calls = [];
  const f = followFixture();
  const history = workerHistory(async (path) => {
    calls.push(path);
    if (path.startsWith('/sessions?')) return {sessions: [{id: 't', kind: 'terminal'}], next_cursor: null};
    if (path.includes('/turns')) return {turns: [{turn: 1, status: 'running', events: [{id: 1, turn: 1, type: 'turn.started', data: {turn: 1, user_text: 'hi'}}]}], has_earlier: false};
    return {session: {id: 't', kind: 'terminal', status: 'running', busy: true}};
  }, () => {}, () => () => {}, {
    open: (url) => { const source = new FakeSource(url); f.sources.push(source); return source; },
    schedule: (fn, ms) => ({fn, ms}), cancel: () => {}, doc: f.doc,
  });
  history.state.kind = 'terminal';
  await history.list();
  assert.deepEqual(new URLSearchParams(calls[0].split('?')[1]).get('kind'), 'terminal');
  await history.detail('t');
  assert.equal(history.state.error, '');
  assert.equal(history.state.kind, 'terminal');
  assert.equal(f.sources.at(-1).url, '/api/sessions/t/events?after=1');
  history.leave();
});

test('opening a run keeps the list on worker runs, and a console chat is still refused', async () => {
  const history = workerHistory(async () => ({session: {id: 'w', kind: 'worker', status: 'completed'}, turns: [], has_earlier: false}), () => {}, () => () => {});
  history.state.kind = 'terminal';
  await history.detail('w');
  assert.equal(history.state.kind, 'worker');
  const chat = workerHistory(async () => ({session: {id: 'c', kind: 'chat'}}), () => {}, () => () => {});
  await chat.detail('c');
  assert.equal(chat.state.error, 'Worker run not found.');
});

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

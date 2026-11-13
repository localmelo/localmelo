import assert from 'node:assert/strict';
import test from 'node:test';
import {fingerprint, LIST_BASE_MS, LIST_CEILING_MS, listBusy, nextListDelay} from '../../support/gateway/static/rest.mjs';
import {MOOD_WINDOW_S} from '../../support/gateway/static/panel.mjs';

test('the list refresh asks at 4 s while it must and backs off in steps to 16 s while nothing changes', () => {
  assert.equal(LIST_BASE_MS, 4000);
  assert.equal(LIST_CEILING_MS, 16000);
  const cases = [
    // [what the last refresh found, the interval it ran at, the next interval]
    ['nothing new', {}, 4000, 8000],
    ['nothing new, second time', {}, 8000, 16000],
    ['nothing new at the ceiling', {}, 16000, 16000],
    ['something new', {changed: true}, 16000, 4000],
    ['the server did not answer', {failed: true}, 16000, 4000],
    ['something runs or waits', {busy: true}, 16000, 4000],
    ['the person acted or came back', {acted: true}, 16000, 4000],
    ['everything at once', {changed: true, failed: true, busy: true, acted: true}, 8000, 4000],
  ];
  for (const [name, found, at, next] of cases) assert.equal(nextListDelay(at, found), next, name);
  assert.equal(nextListDelay(4000), 8000, 'a refresh with no facts found nothing new');
});

test('anything that runs or waits for the person keeps the page asking at the base interval', () => {
  const quiet = {running: {chats: 0, tasks: 0, runs: 0, terminal: 0}, tasks: {running: 0, queued: 0, held: 0}, items: [], waiting: [], control: [], recent: null};
  const row = (extra = {}) => ({id: 'a', busy: false, held: '', pending_approvals: 0, ...extra});
  const cases = [
    ['nothing at all', null, null, false],
    ['a quiet list and health', {sessions: [row()], main: row({id: 'm'})}, {status: quiet}, false],
    ['a chat turn runs', {sessions: [row({busy: true})]}, {status: quiet}, true],
    ['the main chat runs', {main: row({busy: true})}, {status: quiet}, true],
    ['a turn is held', {sessions: [row({held: 'Resources'})]}, {status: quiet}, true],
    ['an approval waits', {sessions: [row({pending_approvals: 1})]}, {status: quiet}, true],
    ['a task waits in a chat', {sessions: [row({waiting_tasks: 1})]}, {status: quiet}, true],
    ['a task waits in a side chat', {main: row({waiting_in_side: 2})}, {status: quiet}, true],
    ['the health counts a run', {sessions: [row()]}, {status: {...quiet, running: {...quiet.running, tasks: 1}}}, true],
    ['the health lists a wait', null, {status: {...quiet, waiting: [{kind: 'task', id: 't'}]}}, true],
    ['the person holds a page', null, {status: {...quiet, control: [{kind: 'chat', id: 'a'}]}}, true],
    ['a task is queued', null, {status: {...quiet, tasks: {queued: 1, held: 0}}}, true],
    ['a task is held', null, {status: {...quiet, tasks: {queued: 0, held: 1}}}, true],
    ['a run that just ended: the header still shows how', null, {status: {...quiet, recent: {kind: 'chat', id: 'a', outcome: 'done', ended_s_ago: 1}}}, true],
    ['a run that ended long ago is nothing to wait for', null, {status: {...quiet, recent: {kind: 'chat', id: 'a', outcome: 'done', ended_s_ago: MOOD_WINDOW_S + 1}}}, false],
    ['a stopped run shows no ending', null, {status: {...quiet, recent: {kind: 'chat', id: 'a', outcome: 'stopped', ended_s_ago: 1}}}, false],
  ];
  for (const [name, list, health, busy] of cases) assert.equal(listBusy(list, health), busy, name);
});

test('two reads of an unchanged server read the same, and any other difference is something new', () => {
  const health = (seconds) => ({
    ok: true, uptime_s: seconds, model_server: {reachable: true, checked_s_ago: seconds % 30},
    status: {running: {chats: 0}, recent: {kind: 'chat', id: 'a', outcome: 'done', ended_s_ago: seconds}},
  });
  assert.equal(fingerprint(health(10)), fingerprint(health(14.5)));
  const list = {sessions: [{id: 'a', title: 'One', busy: false}], machine: null};
  assert.equal(fingerprint(list), fingerprint(structuredClone(list)));
  assert.notEqual(fingerprint(list), fingerprint({...list, sessions: [{id: 'a', title: 'Two', busy: false}]}));
  assert.notEqual(fingerprint(health(10)), fingerprint({...health(10), model_server: {reachable: false, checked_s_ago: 1}}));
  assert.notEqual(fingerprint(health(10)), fingerprint(null));
  assert.equal(fingerprint(undefined), fingerprint(null));
});

test('what the header derives from the age of a run\'s end is part of the comparison: the window opening and closing is something new', () => {
  const body = (ended) => ({ok: true, status: {running: {chats: 0}, recent: {kind: 'chat', id: 'a', outcome: 'done', ended_s_ago: ended}}});
  assert.equal(fingerprint(body(1)), fingerprint(body(MOOD_WINDOW_S - 1)), 'two readings inside the window read the same');
  assert.equal(fingerprint(body(MOOD_WINDOW_S)), fingerprint(body(60)), 'two readings outside it too');
  assert.notEqual(fingerprint(body(MOOD_WINDOW_S - 1)), fingerprint(body(MOOD_WINDOW_S)), 'the window closing is a change');
  assert.notEqual(fingerprint(body(1)), fingerprint({ok: true, status: {running: {chats: 0}, recent: null}}));
});

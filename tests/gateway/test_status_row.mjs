import assert from 'node:assert/strict';
import test from 'node:test';
import {setupStatusRow} from '../../support/gateway/static/status-row.mjs';

function rig(answers) {
  const calls = [];
  const timers = [];
  const root = {
    hidden: true, children: [],
    ownerDocument: {createElement: () => ({className: '', textContent: ''})},
    replaceChildren(...items) { this.children = items; },
  };
  let now = 10_000;
  const row = setupStatusRow({
    api: async (path) => {
      calls.push(path);
      const answer = answers.shift();
      if (answer instanceof Error) throw answer;
      return typeof answer === 'function' ? answer() : answer;
    },
    root,
    schedule: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
    cancel: (id) => { timers[id - 1].cancelled = true; },
    clock: () => now,
  });
  const text = () => root.children.map((child) => child.textContent);
  const tick = () => new Promise((resolve) => setImmediate(resolve));
  return {row, root, calls, timers, text, tick, advance: (ms) => { now += ms; }};
}

test('draws the segments the server sends, in order, and hides an empty row', async () => {
  const t = rig([{segments: ['cache --', 'ctx 1.0k/8.0k', 'step 3']}, {segments: []}]);
  t.row.follow('a b');
  await t.tick();
  assert.deepEqual(t.calls, ['/sessions/a%20b/status']);
  assert.deepEqual(t.text(), ['cache --', 'ctx 1.0k/8.0k', 'step 3']);
  assert.equal(t.root.hidden, false);
  t.advance(1000);
  t.row.bump();
  await t.tick();
  assert.equal(t.root.hidden, true);
});

test('frames within a second collapse into one fetch at the end of the gap', async () => {
  const t = rig([{segments: ['one']}, {segments: ['two']}]);
  t.row.follow('a');
  await t.tick();
  t.advance(300);
  for (let i = 0; i < 5; i++) t.row.bump();
  await t.tick();
  assert.equal(t.calls.length, 1);
  assert.equal(t.timers.length, 1);
  assert.equal(t.timers[0].ms, 700);
  t.advance(700);
  t.timers[0].fn();
  await t.tick();
  assert.equal(t.calls.length, 2);
  assert.deepEqual(t.text(), ['two']);
});

test('a bump while a fetch is in flight waits for it, and never overlaps', async () => {
  let release;
  const t = rig([() => new Promise((resolve) => { release = () => resolve({segments: ['slow']}); }), {segments: ['next']}]);
  t.row.follow('a');
  t.advance(2000);
  t.row.bump();
  await t.tick();
  assert.equal(t.calls.length, 1);
  t.advance(2000);
  release();
  await t.tick();
  assert.equal(t.calls.length, 2);
  assert.deepEqual(t.text(), ['next']);
});

test('an answer for a session left in the meantime is dropped', async () => {
  let release;
  const t = rig([() => new Promise((resolve) => { release = () => resolve({segments: ['old']}); })]);
  t.row.follow('a');
  t.row.clear();
  release();
  await t.tick();
  assert.deepEqual(t.text(), []);
  assert.equal(t.root.hidden, true);
  assert.equal(t.row.current(), null);
});

test('a failed fetch keeps the row, and clear cancels a waiting fetch', async () => {
  const t = rig([{segments: ['kept']}, new Error('offline')]);
  t.row.follow('a');
  await t.tick();
  t.advance(1000);
  t.row.bump();
  await t.tick();
  assert.deepEqual(t.text(), ['kept']);
  t.row.bump();
  assert.equal(t.timers.length, 1);
  t.row.clear();
  assert.equal(t.timers[0].cancelled, true);
});

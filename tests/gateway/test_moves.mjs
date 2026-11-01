/**
 * Melo's moves (moves.mjs): the timeline of a state, the cross-fade times, the rest-or-play rule, and the player on
 * a fake canvas and clock: attention interrupts from the frame on screen, the latest request wins over an older
 * load or completion, the same state never restarts, startup plays once and holds, reduced motion shows posters,
 * and a hidden page draws nothing and goes on where it stopped. The shipped manifest is read as the page reads it.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {
  MOVES, PLAY_AFTER_MS, PLAY_EVERY_MS, cellOf, createPlayer, fadeMs, frameAt, idleMove, startMoves,
} from '../../support/gateway/static/moves.mjs';

const MANIFEST = JSON.parse(readFileSync(new URL('../../support/gateway/static/moves/manifest.json', import.meta.url)));
const settle = () => new Promise((resolve) => setImmediate(resolve));

function canvasDouble(label = 'screen') {
  const calls = [];
  const ctx = {
    globalAlpha: 1,
    clearRect() { calls.push(['clear']); },
    drawImage(image, ...args) { calls.push(['draw', image.label ?? image, this.globalAlpha, ...args]); },
  };
  return {label, width: 0, height: 0, calls, getContext: () => ctx};
}

/** A player on a fake clock: `tick(ms)` moves time and runs the frame callbacks and timers that are due. */
function rig({still = false, slow = {}, broken = []} = {}) {
  let now = 1000;
  let next = 1;
  const pending = new Map();
  const schedule = (fn, ms) => { const id = next++; pending.set(id, {fn, at: now + ms}); return id; };
  const cancel = (id) => pending.delete(id);
  const loads = [];
  const completions = [];
  const canvas = canvasDouble();
  const player = createPlayer({
    canvas, manifest: MANIFEST, still, base: '/m/',
    clock: () => now,
    frame: (fn) => schedule(fn, 16), cancelFrame: cancel,
    wait: schedule, cancelWait: cancel,
    makeCanvas: () => canvasDouble('snapshot'),
    loadImage: (src) => {
      loads.push(src);
      const sheet = {label: src, width: 1024};
      if (broken.some((name) => src.endsWith(`${name}.webp`))) return Promise.reject(new Error(`Cannot load ${src}`));
      const key = Object.keys(slow).find((name) => src.endsWith(`${name}.webp`));
      return key ? slow[key].then(() => sheet) : Promise.resolve(sheet);
    },
    onComplete: (event) => completions.push(event.state),
  });
  async function tick(ms) {
    const end = now + ms;
    for (;;) {
      const due = [...pending.entries()].filter(([, job]) => job.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      pending.delete(due[0]);
      now = Math.max(now, due[1].at);
      due[1].fn(now);
      await settle();
    }
    now = end;
  }
  const lastDraw = () => canvas.calls.filter((call) => call[0] === 'draw').at(-1);
  return {player, canvas, loads, completions, tick, lastDraw, pending, at: () => now};
}

test('the shipped manifest has the six states, one duration per frame and the fade times', () => {
  assert.deepEqual(Object.keys(MANIFEST.states), MOVES);
  assert.deepEqual(MANIFEST.fade_ms, {same_pose: 160, pose_change: 220, attention: 120, reduced_motion: 80});
  for (const [key, entry] of Object.entries(MANIFEST.states)) {
    assert.ok(entry.durations_ms.length <= MANIFEST.sheet.columns * MANIFEST.sheet.rows, key);
    assert.ok(entry.poster >= 0 && entry.poster < entry.durations_ms.length, key);
    assert.equal(entry.playback, key === 'startup' ? 'once' : 'loop', key);
  }
  // Startup's poster is its last, sitting frame: where it holds.
  assert.equal(MANIFEST.states.startup.poster, MANIFEST.states.startup.durations_ms.length - 1);
});

test('a frame follows its durations, a loop wraps and a once state ends on its last frame', () => {
  const entry = {durations_ms: [100, 50, 200], playback: 'loop'};
  assert.deepEqual(frameAt(entry, 0), {index: 0, complete: false, next: 100});
  assert.deepEqual(frameAt(entry, 120), {index: 1, complete: false, next: 30});
  assert.deepEqual(frameAt(entry, 349), {index: 2, complete: false, next: 1});
  assert.deepEqual(frameAt(entry, 350), {index: 0, complete: false, next: 100});
  assert.deepEqual(frameAt(entry, 350, true), {index: 2, complete: true, next: Infinity});
  assert.deepEqual(frameAt({...entry, playback: 'once'}, 9999), {index: 2, complete: true, next: Infinity});
  assert.deepEqual(cellOf(MANIFEST, 6, 1024), {x: 512, y: 256, size: 256});
});

test('fades: 220 ms between poses, 160 ms within one, 120 ms into attention, 80 ms when still, none at first', () => {
  assert.equal(fadeMs(MANIFEST, null, 'startup'), 0);
  assert.equal(fadeMs(MANIFEST, 'working', 'idle_rest'), 220);
  assert.equal(fadeMs(MANIFEST, 'startup', 'thinking'), 160);
  assert.equal(fadeMs(MANIFEST, 'attention', 'thinking'), 160);
  assert.equal(fadeMs(MANIFEST, 'idle_play', 'attention'), 120);
  assert.equal(fadeMs(MANIFEST, 'thinking', 'attention'), 120);
  assert.equal(fadeMs(MANIFEST, 'working', 'attention', true), 80);
});

test('rest plays once after 20 s of quiet and at most once every 5 minutes', () => {
  assert.equal(PLAY_AFTER_MS, 20_000);
  assert.equal(PLAY_EVERY_MS, 300_000);
  assert.equal(idleMove(0, Infinity), 'idle_rest');
  assert.equal(idleMove(19_999, Infinity), 'idle_rest');
  assert.equal(idleMove(20_000, Infinity), 'idle_play');
  assert.equal(idleMove(600_000, 299_999), 'idle_rest');
  assert.equal(idleMove(600_000, 300_000), 'idle_play');
});

test('startup plays once, holds its last frame and stops drawing; asking for it again does not restart it', async () => {
  const {player, completions, tick, lastDraw, pending} = rig();
  assert.deepEqual(await player.set('startup'), {changed: true, durationMs: 0});
  assert.equal(lastDraw()[1], '/m/startup.webp');
  await tick(1000);
  assert.ok(player.inspect().frame > 0 && player.inspect().frame < 15);
  assert.deepEqual(await player.set('startup'), {changed: false});
  await tick(3500);
  assert.deepEqual(completions, ['startup']);
  assert.equal(player.inspect().frame, 15);
  assert.equal(player.inspect().complete, true);
  assert.equal(pending.size, 0, 'nothing is scheduled while it holds');
  // The last cell of the 4 x 4 sheet.
  assert.deepEqual(lastDraw().slice(3, 7), [768, 768, 256, 256]);
});

test('attention interrupts at once: the frame on screen fades out over 120 ms from where it was', async () => {
  const {player, canvas, tick, lastDraw} = rig();
  await player.set('idle_play');
  await tick(1300);
  const frameBefore = player.inspect().frame;
  assert.ok(frameBefore > 0, 'the play is under way');
  canvas.calls.length = 0;
  assert.deepEqual(await player.set('attention'), {changed: true, durationMs: 120});
  const first = canvas.calls.filter((call) => call[0] === 'draw');
  // The frozen screen is drawn under the new state, which starts at its first frame.
  assert.equal(first[0][1], 'snapshot');
  assert.equal(first[0][2], 1);
  assert.equal(first[1][1], '/m/attention.webp');
  assert.deepEqual(first[1].slice(3, 5), [0, 0]);
  assert.equal(player.inspect().fading, true);
  await tick(130);
  assert.equal(player.inspect().fading, false);
  assert.equal(lastDraw()[2], 1);
  assert.equal(player.inspect().shown, 'attention');
});

test('the latest request wins: an older load that ends later and a late completion never bring their state back', async () => {
  let release;
  const slow = {working: new Promise((resolve) => { release = resolve; })};
  const {player, tick, completions} = rig({slow});
  await player.set('startup');
  const asked = player.set('working');
  const urgent = await player.set('attention');
  assert.equal(urgent.changed, true);
  release();
  assert.deepEqual(await asked, {superseded: true});
  assert.equal(player.inspect().shown, 'attention');
  await tick(10_000);
  // Startup was cut before its end: its completion never comes.
  assert.deepEqual(completions, []);
  assert.equal(player.inspect().shown, 'attention');
});

test('a change in the middle of a fade starts from the mix on screen, and the same state asked again keeps its loop', async () => {
  const {player, canvas, tick} = rig();
  await player.set('working');
  await tick(500);
  await player.set('idle_rest');
  await tick(50);
  assert.equal(player.inspect().fading, true);
  canvas.calls.length = 0;
  await player.set('thinking');
  // One snapshot of the screen, then the new state: the old fade does not go on behind it.
  const draws = canvas.calls.filter((call) => call[0] === 'draw');
  assert.deepEqual(draws.map((call) => call[1]), ['snapshot', '/m/thinking.webp']);
  await tick(2000);
  const frame = player.inspect().frame;
  assert.deepEqual(await player.set('thinking'), {changed: false});
  assert.equal(player.inspect().frame, frame);
  await tick(100);
  assert.equal(player.inspect().shown, 'thinking');
});

test('a looping state asked to play once holds its last frame and says it ended', async () => {
  const {player, tick, completions} = rig();
  await player.set('idle_play', {once: true});
  await tick(6100);
  assert.deepEqual(completions, ['idle_play']);
  assert.equal(player.inspect().frame, 15);
});

test('reduced motion: each state is its poster, a change is an 80 ms fade, and startup ends at once', async () => {
  const {player, tick, lastDraw, completions, pending} = rig({still: true});
  await player.set('startup');
  assert.deepEqual(completions, ['startup']);
  assert.deepEqual(lastDraw().slice(3, 5), [768, 768]);
  assert.deepEqual(await player.set('working'), {changed: true, durationMs: 80});
  await tick(100);
  assert.equal(pending.size, 0, 'a poster draws once');
  // Turning reduced motion off starts the state moving again.
  await player.setStill(false);
  assert.equal(player.inspect().still, false);
  assert.ok(pending.size > 0);
});

test('a hidden page draws nothing and the timeline goes on where it stopped', async () => {
  const {player, canvas, tick, pending} = rig();
  await player.set('thinking');
  await tick(1000);
  const frame = player.inspect().frame;
  player.setVisible(false);
  assert.equal(pending.size, 0);
  canvas.calls.length = 0;
  await tick(60_000);
  assert.equal(canvas.calls.length, 0);
  player.setVisible(true);
  assert.equal(player.inspect().frame, frame);
  // Two reasons to hide: it moves again only once both are gone.
  player.setVisible(false, 'page');
  player.setVisible(false, 'portrait');
  player.setVisible(true, 'page');
  assert.equal(player.inspect().paused, true);
  player.setVisible(true, 'portrait');
  assert.equal(player.inspect().paused, false);
});

test('a state asked for while hidden starts when the page shows, without a fade', async () => {
  const {player, tick} = rig();
  await player.set('working');
  player.setVisible(false);
  assert.deepEqual(await player.set('attention'), {changed: true, durationMs: 0});
  await tick(5000);
  player.setVisible(true);
  assert.equal(player.inspect().shown, 'attention');
  assert.equal(player.inspect().frame, 0);
  assert.equal(player.inspect().fading, false);
});

test('a page without a canvas or without the files keeps its pictures', async () => {
  assert.equal(await startMoves({}), null);
  const canvas = canvasDouble();
  assert.equal(await startMoves(canvas, {fetchJson: async () => { throw new Error('404'); }, doc: null}), null);
  const failing = await startMoves(canvas, {fetchJson: async () => MANIFEST, doc: null, media: null,
    loadImage: async () => { throw new Error('gone'); }});
  assert.equal(failing, null);
});

test('a page follows its visibility and the reduced-motion setting, and lets go of both when disposed', async () => {
  const listeners = {};
  const doc = {visibilityState: 'visible', addEventListener: (type, fn) => { listeners[type] = fn; },
    removeEventListener: (type, fn) => { if (listeners[type] === fn) delete listeners[type]; }};
  const media = {matches: true, addEventListener: (type, fn) => { listeners.motion = fn; },
    removeEventListener: (type, fn) => { if (listeners.motion === fn) delete listeners.motion; }};
  const player = await startMoves(canvasDouble(), {fetchJson: async () => MANIFEST, doc, media: () => media,
    loadImage: async (src) => ({label: src, width: 1024}), clock: () => 0, frame: () => 1, cancelFrame() {},
    wait: () => 1, cancelWait() {}, makeCanvas: () => canvasDouble('snapshot')});
  assert.equal(player.inspect().still, true);
  doc.visibilityState = 'hidden';
  listeners.visibilitychange();
  assert.equal(player.inspect().paused, true);
  listeners.motion({matches: false});
  assert.equal(player.inspect().still, false);
  player.dispose();
  assert.deepEqual(Object.keys(listeners), []);
});

test('a sheet that cannot load leaves the state shown going on where it was', async () => {
  const r = rig({broken: ['working']});
  r.player.set('idle_rest');
  await settle();
  await r.tick(500);
  const frame = r.player.inspect().frame;
  const result = await r.player.set('working');
  assert.ok(result.error);
  const after = r.player.inspect();
  assert.equal(after.wanted, 'idle_rest');
  assert.equal(after.shown, 'idle_rest');
  assert.equal(after.drawing, true, 'its timeline goes on');
  assert.equal(after.frame, frame);
  await r.tick(2000);
  assert.notEqual(r.player.inspect().frame, frame);
});

test('one chain of frames at a time, whatever was asked in between', async () => {
  const r = rig();
  r.player.set('working');
  await settle();
  r.player.setVisible(false, 'page');
  r.player.setVisible(true, 'page');
  r.player.setVisible(true, 'page');
  r.player.set('attention');
  await settle();
  r.player.set('attention', {restart: true});
  await settle();
  assert.ok(r.pending.size <= 1, `${r.pending.size} chains`);
  await r.tick(100);
  assert.ok(r.pending.size <= 1);
});

test('a state may cut in with no fade when the canvas was not on show', async () => {
  const r = rig();
  r.player.set('idle_rest');
  await settle();
  await r.tick(300);
  const result = await r.player.set('idle_play', {cut: true});
  assert.equal(result.durationMs, 0);
  assert.equal(r.player.inspect().fading, false);
});

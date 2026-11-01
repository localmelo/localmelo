/**
 * Melo's moves: the animated states of the portrait beside a chat. The desktop pet of the macOS app plays the same
 * files with the same rules (PetMoves.swift).
 *
 * The files are /assets/moves (scripts/build_moves.py): per state, a sheet of its frames, COLUMNS x ROWS cells of
 * one square canvas, left to right and top to bottom; manifest.json gives each state its frame durations, whether
 * it plays once or loops, its poster frame and its pose, and the cross-fade times.
 *
 * The rules, here and in the app:
 * - A change cross-fades from what is on screen (the frame shown, or the mix of a fade still going): 160 ms between
 *   two states of one pose, 220 ms between two poses, 120 ms into `attention`, which interrupts anything at once.
 * - The latest request wins: an older load or a completion that comes late never brings back its state.
 * - The same state asked again does not restart. `startup` plays once and holds its last frame.
 * - With reduced motion each state is its poster, and a change is an 80 ms fade.
 * - While the page is hidden nothing is drawn and the timeline stops; it goes on where it was when shown again.
 *
 * Between frames the player waits on a timer for the next frame's start (a frame lasts 40 ms to 1.6 s); only a
 * cross-fade draws on every animation frame.
 */
export const MOVES_BASE = '/assets/moves/';

/** The states of the pack. */
export const MOVES = ['startup', 'thinking', 'working', 'attention', 'idle_rest', 'idle_play'];

/** While nothing runs: Melo rests, and plays once (one loop of `idle_play`) after PLAY_AFTER_MS of rest, at most
 * once every PLAY_EVERY_MS. The desktop pet uses the same two numbers (`petPlayAfter`, `petPlayEvery`). */
export const PLAY_AFTER_MS = 20_000;
export const PLAY_EVERY_MS = 300_000;

/** `idle_play` once rest has lasted PLAY_AFTER_MS and the last play began PLAY_EVERY_MS ago or more; else
 * `idle_rest`. */
export function idleMove(restedMs, sincePlayMs) {
  return restedMs >= PLAY_AFTER_MS && sincePlayMs >= PLAY_EVERY_MS ? 'idle_play' : 'idle_rest';
}

/** Milliseconds of the cross-fade from `from` to `to` (0 for the first state). */
export function fadeMs(manifest, from, to, still = false) {
  if (!from) return 0;
  const fade = manifest.fade_ms;
  if (still) return fade.reduced_motion;
  if (to === 'attention') return fade.attention;
  return manifest.states[from]?.pose === manifest.states[to]?.pose ? fade.same_pose : fade.pose_change;
}

/**
 * The frame of a state `elapsed` ms into its timeline: `{index, complete, next}`, with `next` the ms until the
 * frame changes (Infinity once a state that plays once has ended, on its last frame). `once` overrides the
 * manifest's playback.
 */
export function frameAt(entry, elapsed, once = entry.playback === 'once') {
  const durations = entry.durations_ms;
  const total = durations.reduce((sum, ms) => sum + ms, 0);
  const at = Math.max(0, elapsed);
  if (once && at >= total) return {index: durations.length - 1, complete: true, next: Infinity};
  let phase = at % total;
  for (let index = 0; index < durations.length; index += 1) {
    if (phase < durations[index]) return {index, complete: false, next: durations[index] - phase};
    phase -= durations[index];
  }
  return {index: 0, complete: false, next: durations[0]};
}

/** The cell of frame `index` on a sheet `width` pixels wide: `{x, y, size}` in the sheet's pixels. */
export function cellOf(manifest, index, width) {
  const {columns} = manifest.sheet;
  const size = width / columns;
  return {x: (index % columns) * size, y: Math.floor(index / columns) * size, size};
}

const smooth = (t) => t * t * (3 - 2 * t);

/**
 * A player that draws the moves on `canvas` (a 2D canvas of the canvas size, shown at any size). The manifest is
 * read already; the sheets load on `preload()` or when a state is first asked for. `onComplete({state})` hears a
 * state that played once reach its end (only the state still shown). The timing and the images come from the
 * options, so it runs without a browser.
 */
export function createPlayer({
  canvas, manifest, base = MOVES_BASE, still = false, onComplete = () => {}, onError = () => {},
  clock = () => globalThis.performance.now(),
  frame = (fn) => globalThis.requestAnimationFrame(fn), cancelFrame = (id) => globalThis.cancelAnimationFrame(id),
  wait = (fn, ms) => setTimeout(fn, ms), cancelWait = (id) => clearTimeout(id),
  makeCanvas = () => canvas.ownerDocument.createElement('canvas'),
  loadImage = (src) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => (image.decode ? image.decode().then(() => resolve(image), () => resolve(image)) : resolve(image));
    image.onerror = () => reject(new Error(`Cannot load ${src}`));
    image.src = src;
  }),
}) {
  const ctx = canvas.getContext('2d');
  const side = manifest.canvas.size;
  canvas.width = side;
  canvas.height = side;
  const cache = new Map();
  const my = {
    wanted: null, shown: null, once: false, sheet: null, started: 0, pausedAt: null, generation: 0, blend: null,
    complete: false, index: 0, still, handle: null, timer: null, disposed: false, hiders: new Set(),
  };

  // A decoded image's own width (an <img> may report a laid-out one); the grid's when it says nothing.
  const sheetWidth = (sheet) => sheet.naturalWidth || sheet.width || side * manifest.sheet.columns;
  function load(key) {
    if (!cache.has(key)) {
      const promise = loadImage(`${base}${manifest.states[key].file}`);
      cache.set(key, promise);
      promise.catch(() => { if (cache.get(key) === promise) cache.delete(key); });
    }
    return cache.get(key);
  }
  function stop() {
    if (my.handle !== null) cancelFrame(my.handle);
    if (my.timer !== null) cancelWait(my.timer);
    my.handle = null;
    my.timer = null;
  }
  // What is on the screen now, frozen: the next state fades from it, so nothing snaps back to a first frame.
  function snapshot() {
    if (!my.shown) return null;
    const copy = makeCanvas();
    copy.width = side;
    copy.height = side;
    copy.getContext('2d').drawImage(canvas, 0, 0);
    return copy;
  }

  function draw(token) {
    if (token !== my.generation || my.disposed || my.pausedAt !== null || !my.shown) return;
    my.handle = null;
    my.timer = null;
    const now = clock();
    const entry = manifest.states[my.shown];
    // A still state that plays once has ended as soon as it shows: its poster is where it holds.
    const sample = my.still ? {index: entry.poster, complete: my.once, next: Infinity}
      : frameAt(entry, now - my.started, my.once);
    my.index = sample.index;
    const cell = cellOf(manifest, sample.index, sheetWidth(my.sheet));
    ctx.clearRect(0, 0, side, side);
    let mix = 1;
    if (my.blend) {
      mix = my.blend.duration > 0 ? Math.min(1, Math.max(0, (now - my.blend.started) / my.blend.duration)) : 1;
      mix = smooth(mix);
      ctx.globalAlpha = 1 - mix;
      ctx.drawImage(my.blend.image, 0, 0, side, side);
    }
    ctx.globalAlpha = mix;
    ctx.drawImage(my.sheet, cell.x, cell.y, cell.size, cell.size, 0, 0, side, side);
    ctx.globalAlpha = 1;
    if (mix >= 1) my.blend = null;
    if (sample.complete && !my.complete) {
      my.complete = true;
      onComplete({state: my.shown});
      // The handler may have asked for another state: that one draws now.
      if (token !== my.generation) return;
    }
    if (my.blend) my.handle = frame(() => draw(token));
    else if (Number.isFinite(sample.next)) my.timer = wait(() => draw(token), Math.max(1, Math.ceil(sample.next)));
  }

  /**
   * Show `key`. The same state as the one asked for last does nothing (unless `restart`). `once` plays a looping
   * state a single time and holds its last frame. Resolves `{changed}`; a request overtaken by a later one
   * resolves `{superseded: true}`.
   */
  async function set(key, {once = false, restart = false} = {}) {
    if (my.disposed) return {changed: false};
    if (!manifest.states[key]) throw new Error(`Unknown move: ${key}`);
    if (!restart && my.wanted === key) return {changed: false};
    const token = ++my.generation;
    my.wanted = key;
    // The old timeline stops at once, including a fade still going; the screen is copied as it is.
    stop();
    const frozen = my.pausedAt === null ? snapshot() : null;
    const previous = my.shown;
    let sheet;
    try {
      sheet = await load(key);
    } catch (error) {
      if (token !== my.generation || my.disposed) return {superseded: true};
      my.wanted = my.shown;
      onError(error);
      return {changed: false, error};
    }
    if (token !== my.generation || my.disposed) return {superseded: true};
    const now = clock();
    Object.assign(my, {shown: key, sheet, once: once || manifest.states[key].playback === 'once', complete: false,
      started: now});
    const duration = fadeMs(manifest, previous, key, my.still);
    my.blend = frozen && duration > 0 && previous ? {image: frozen, started: now, duration} : null;
    if (my.pausedAt !== null) {
      // Hidden: the new state starts when the page is shown again; its first frame is ready for that moment.
      my.pausedAt = now;
      my.blend = null;
      const entry = manifest.states[key];
      const cell = cellOf(manifest, my.still ? entry.poster : 0, sheetWidth(sheet));
      ctx.clearRect(0, 0, side, side);
      ctx.drawImage(sheet, cell.x, cell.y, cell.size, cell.size, 0, 0, side, side);
      return {changed: true, durationMs: 0};
    }
    draw(token);
    return {changed: true, durationMs: duration};
  }

  /**
   * The canvas cannot be seen for `reason` (the page is hidden, a still picture covers it): nothing draws and the
   * timeline stops; once no reason is left it goes on where it stopped.
   */
  function setVisible(visible, reason = 'page') {
    if (my.disposed) return;
    const now = clock();
    if (!visible) {
      my.hiders.add(reason);
      if (my.pausedAt !== null) return;
      stop();
      my.pausedAt = now;
      return;
    }
    my.hiders.delete(reason);
    if (my.pausedAt === null || my.hiders.size) return;
    const gone = now - my.pausedAt;
    my.pausedAt = null;
    my.started += gone;
    if (my.blend) my.blend.started += gone;
    draw(my.generation);
  }

  /** Reduced motion on or off: the state shown starts again, as its poster or moving, after a short fade. */
  function setStill(value) {
    if (my.still === Boolean(value)) return Promise.resolve({changed: false});
    my.still = Boolean(value);
    return my.wanted ? set(my.wanted, {restart: true, once: my.once && manifest.states[my.wanted].playback !== 'once'}) : Promise.resolve({changed: false});
  }

  function preload() {
    return Promise.all(Object.keys(manifest.states).map((key) => load(key)));
  }

  function dispose() {
    my.disposed = true;
    my.generation += 1;
    stop();
    cache.clear();
  }

  /** What a test or a look asks: the state asked for and shown, the frame, and whether it fades, ended or waits. */
  function inspect() {
    return {wanted: my.wanted, shown: my.shown, frame: my.index, fading: Boolean(my.blend), complete: my.complete,
      still: my.still, paused: my.pausedAt !== null, once: my.once, drawing: my.handle !== null || my.timer !== null};
  }

  return {set, setVisible, setStill, preload, dispose, inspect};
}

/**
 * The player for a page: it reads the manifest, preloads every sheet (so `attention` never waits on the network),
 * follows the page's visibility and the reduced-motion setting, and keeps the latest `set` until it is ready.
 * Resolves to null where there is no canvas or the files cannot be read, so the caller keeps its pictures.
 */
export async function startMoves(canvas, {
  base = MOVES_BASE, onComplete = () => {}, doc = canvas.ownerDocument, media = globalThis.matchMedia,
  fetchJson = async (url) => {
    const response = await fetch(url, {credentials: 'same-origin'});
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return response.json();
  },
  ...options
} = {}) {
  if (typeof canvas?.getContext !== 'function' || !canvas.getContext('2d')) return null;
  let manifest;
  try {
    manifest = await fetchJson(`${base}manifest.json`);
  } catch {
    return null;
  }
  const reduce = media?.('(prefers-reduced-motion: reduce)');
  const player = createPlayer({canvas, manifest, base, still: Boolean(reduce?.matches), onComplete, ...options});
  try {
    await player.preload();
  } catch {
    player.dispose();
    return null;
  }
  reduce?.addEventListener?.('change', (event) => player.setStill(event.matches));
  if (doc) {
    const follow = () => player.setVisible(doc.visibilityState !== 'hidden');
    doc.addEventListener?.('visibilitychange', follow);
    follow();
  }
  return player;
}

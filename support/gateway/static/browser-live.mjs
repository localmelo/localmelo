import {t} from './i18n.mjs';
import {createStreams, parseMeta} from './browser-stream.mjs';
import {ACTIVE, stateWord} from './browser-card.mjs';

export {RETRY_S, STATES, parseMeta, retryDelay, socketUrl} from './browser-stream.mjs';

/**
 * The Browser panel: a live window onto the page melo is working in, which the person can take over.
 *
 * It sits beside the chat and never opens by itself: the **Open browser** button of the browser card in the
 * conversation (and of the dock above it) opens it on that card's source (`open(card)`), and the chat's header chip,
 * which names the state of the chat's browser, shows or hides it. Hiding it only stops watching: the page, melo's
 * run and the card stay as they are, and the panel is never Done or Stop. Whether the chat can be watched
 * (`available`: its browser is Chrome and the live view is on) is asked when the chat is followed, on `recheck()`
 * (this page saved Settings), and again when a chat marked unwatchable has a new browser card or is asked to open one
 * (another page may have saved Settings since). The pictures come from the page's one stream of the card's source
 * (browser-stream.mjs): a panel that is hidden holds no watcher, and the card, the dock and the panel of one source
 * share one socket and one decoder.
 *
 * `follow(id)` / `release()` bind the panel to the chat that is open, `card(card)` tells it the state of the chat's
 * newest browser card (the chip's words, and the source the chip opens), and a page that is hidden for a while, or
 * kept by the browser for Back, lets go of its sockets (`suspend`) and asks again when it is back (`resume`).
 * `#/live/<id>` (`liveRoute`, `liveUrl`) is the panel alone, which **Pop out** opens large: in its own window in the
 * macOS app, in a window of the browser elsewhere. The pure helpers (`parseMeta`, `view`, `layout`, `toPage`,
 * `addressParts`, `control` and the input messages) are what the tests check.
 *
 * Take over: **Take over** (or the "melo needs you" card, `takeOver()`) hands the tab to the person (`meta.owner` is
 * then `you`) and melo is paused, with no timer. Where they have it is `meta.place`: `window`, a real Chrome window
 * in front of them, of which this panel is a live copy (**Show Chrome window** brings it back to the front); or
 * `app`, this panel, when the page has a form they began (`form`: **Open in Chrome, clear the form** moves it to a
 * window) or the window could not open (`failed`; `busy`: another chat's window is open). In the panel their clicks, drags and wheel on the picture, and
 * their keys, input-method text and pastes in a hidden text box over it, go to the gateway as small JSON messages
 * (`mouseMessage`, `wheelMessage`, `keyMessage`, `composeMessage`, `commitMessage`, `pasteMessage`) and from there to
 * melo's browser as its own input. A point on the picture becomes a point on the page by the frame's own metadata
 * (`toPage`: its size, the space above it and its zoom). The page's boxes are answered in the panel. **Done** gives the
 * page back (closing the Chrome window does too). Nothing typed is kept here either: the text box is emptied after
 * every key.
 */

// The most characters one paste or input-method commit sends (the gateway takes no more).
export const MAX_PASTE = 1000;
// The modifier bits of the browser's input events.
export const ALT = 1;
export const CTRL = 2;
export const META = 4;
export const SHIFT = 8;
// Keys sent by name; any other key is sent only when it types one character.
const NAMED = new Set(['Backspace', 'Tab', 'Enter', 'Escape', 'PageUp', 'PageDown', 'End', 'Home', 'ArrowLeft',
  'ArrowUp', 'ArrowRight', 'ArrowDown', 'Delete', 'Shift', 'Control', 'Alt', 'Meta',
  ...Array.from({length: 12}, (_, n) => `F${n + 1}`)]);
const BUTTONS = ['left', 'middle', 'right', 'back', 'forward'];
// What one wheel step is in page pixels, by the event's delta mode (pixels, lines, pages).
const WHEEL_UNIT = [1, 40, 800];
// The hidden text box keeps one space, so a "select all" from the app's menu shows as a selection of it.
const SENTINEL = ' ';

/** The chat id of a `#/live/<id>` address, or ''. */
export const liveRoute = (hash) => /^#\/live\/([A-Za-z0-9._:-]+)$/.exec(hash ?? '')?.[1] ?? '';

/** The page that shows one chat's Browser panel alone (`#/live/<id>`) on this page's own origin: Pop out opens it. */
export const liveUrl = (location, session) => `${location.origin}/#/live/${encodeURIComponent(session)}`;

/** A page address as two parts: the site (`host`, which the panel stresses) and what follows it. */
export function addressParts(address) {
  const text = String(address ?? '').trim();
  if (!text) return {host: '', rest: ''};
  try {
    const url = new URL(text);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return {host: text.slice(0, 80), rest: ''};
    const rest = `${url.pathname === '/' ? '' : url.pathname}${url.search}`;
    return {host: url.host, rest};
  } catch {
    return {host: text.slice(0, 80), rest: ''};
  }
}

/**
 * Where the picture sits in a box of *box* size: the largest copy of the page view that fits, centred across and
 * flush with the top (a page reads from its top). `k` is the box pixels per page pixel. Null until a picture's
 * size is known.
 */
export function layout(meta, box) {
  const w = meta?.w;
  const h = meta?.h;
  if (!(w > 0 && h > 0 && box?.width > 0 && box?.height > 0)) return null;
  const k = Math.min(box.width / w, box.height / h);
  const width = Math.max(1, Math.floor(w * k));
  const height = Math.max(1, Math.floor(h * k));
  return {left: Math.floor((box.width - width) / 2), top: 0, width, height, k};
}

/**
 * The page point under *point* (box pixels, from the box's top left), by the frame's own metadata: the picture shows
 * the page's view, `w` x `h` pixels, `offsetTop` of them the space the browser keeps above the page, at the page's
 * zoom `scale` (a pinch zoom: the browser's input is in page CSS pixels, so the point is divided by it). Null outside
 * the picture; with `clamp`, the nearest point on it (a drag that leaves the picture still ends on the page).
 */
export function toPage(meta, box, point, {clamp = false} = {}) {
  const fit = layout(meta, box);
  if (!fit || !Number.isFinite(point?.x) || !Number.isFinite(point?.y)) return null;
  let x = (point.x - fit.left) / fit.k;
  let y = (point.y - fit.top) / fit.k - (meta.offsetTop || 0);
  if (clamp) {
    x = Math.min(Math.max(x, 0), meta.w);
    y = Math.min(Math.max(y, 0), meta.h);
  } else if (x < 0 || y < 0 || x > meta.w || y > meta.h) return null;
  const scale = meta.scale > 0 ? meta.scale : 1;
  return {x: Math.round((x / scale) * 100) / 100, y: Math.round((y / scale) * 100) / 100};
}

// ── The person's input ──

/** The modifier bits held in *event* (a key, pointer or wheel event). */
export function modsOf(event) {
  return (event?.altKey ? ALT : 0) | (event?.ctrlKey ? CTRL : 0) | (event?.metaKey ? META : 0) | (event?.shiftKey ? SHIFT : 0);
}

const isChar = (key) => typeof key === 'string' && [...key].length === 1;

/**
 * The message for a key going *action* (`down` or `up`), or null for one the panel does not send: an input method's
 * key (its text comes from the composition), a dead key, or the paste shortcut (the paste comes as text, from the
 * paste itself). A key that types one character carries its key code; the gateway decides what it types.
 */
export function keyMessage(event, action) {
  const key = String(event?.key ?? '');
  if (!key || key === 'Unidentified' || key === 'Dead' || key === 'Process' || event?.isComposing || event?.keyCode === 229) return null;
  const mods = modsOf(event);
  if ((mods & (META | CTRL)) && key.toLowerCase() === 'v') return null;
  if (!isChar(key) && !NAMED.has(key)) return null;
  const code = /^[A-Za-z0-9]{0,32}$/.test(String(event.code ?? '')) ? String(event.code ?? '') : '';
  const message = {type: 'key', action, key, code, mods};
  if (isChar(key)) {
    const keyCode = Number(event.keyCode);
    message.keyCode = Number.isInteger(keyCode) && keyCode >= 0 && keyCode <= 255 ? keyCode : 0;
  }
  if (event.repeat === true) message.repeat = true;
  return message;
}

/** The button a held *buttons* mask presses first, as the browser names it. */
function heldButton(buttons) {
  if (buttons & 1) return 'left';
  if (buttons & 2) return 'right';
  if (buttons & 4) return 'middle';
  return 'none';
}

/**
 * The message for the pointer going *action* (`move`, `down` or `up`) at *point* on the picture, or null off it. A
 * press or release that started on the picture is kept on it (`clamp`), so a drag always ends on the page.
 */
export function mouseMessage(meta, box, point, event, action, {clamp = false} = {}) {
  const at = toPage(meta, box, point, {clamp});
  if (!at) return null;
  const buttons = Number.isInteger(event?.buttons) ? event.buttons & 31 : 0;
  const message = {type: 'mouse', action, x: at.x, y: at.y, buttons, mods: modsOf(event)};
  if (action === 'move') message.button = heldButton(buttons);
  else {
    message.button = BUTTONS[event?.button] ?? 'left';
    message.clicks = Math.min(Math.max(Number(event?.detail) || 1, 1), 3);
  }
  return message;
}

/** The message for a scroll of *dx*, *dy* page pixels at *point* on the picture, or null off it. */
export function wheelMessage(meta, box, point, dx, dy, mods = 0) {
  const at = toPage(meta, box, point);
  if (!at || !Number.isFinite(dx) || !Number.isFinite(dy) || (dx === 0 && dy === 0)) return null;
  const k = layout(meta, box)?.k || 1;
  // A wheel step scrolls the page as far as it scrolls the picture, so a step reads the same here as on the page.
  return {type: 'wheel', x: at.x, y: at.y, dx: Math.round(dx / k), dy: Math.round(dy / k), mods};
}

/** A wheel event's movement in pixels, whatever unit it came in. */
export const wheelPixels = (event) => {
  const unit = WHEEL_UNIT[event?.deltaMode] ?? 1;
  return {dx: (Number(event?.deltaX) || 0) * unit, dy: (Number(event?.deltaY) || 0) * unit};
};

/** An input method's text being composed (shown underlined on the page until it is committed). */
export function composeMessage(text) {
  const value = String(text ?? '').slice(0, 200);
  return {type: 'compose', text: value, start: value.length, end: value.length};
}

/** An input method's committed text (or text from another source, such as the emoji picker); null when empty. */
export function commitMessage(text) {
  const value = String(text ?? '');
  return value ? {type: 'text', text: value.slice(0, MAX_PASTE)} : null;
}

/** A paste, as plain text; null when empty or longer than the gateway takes (the panel says so). */
export function pasteMessage(text) {
  const value = String(text ?? '').replace(/\r\n?/g, '\n');
  if (!value || value.length > MAX_PASTE) return null;
  return {type: 'paste', text: value};
}

/**
 * Who has the page, in what the panel shows: `you` once the person took it over (with the words of the control bar
 * and the page's box, if it shows one), `canTake` when they may take it, `finishing` while melo ends a step it was in
 * when they took it (their input waits). `place` is where they have it; `input` whether their input goes through
 * this panel (not while the page is in a Chrome window); `window` the words of the button that shows the window, or ''
 * for none. A *touch* screen has no Control-Esc, so its note leaves that out.
 */
export function control(meta, {focused = false, link = 'open', touch = false} = {}) {
  const live = Boolean(meta) && ['live', 'loading', 'stale'].includes(meta.state) && link === 'open';
  const you = live && meta.owner === 'you';
  const finishing = you && meta.browsing === true;
  const place = you ? meta.place || 'app' : '';
  let hint = '';
  if (place === 'window') hint = `${t('browser.paused')} ${t('browser.liveCopy')}`;
  else if (you) {
    const why = meta.busy ? t('browser.busy') : meta.failed ? t('browser.failed') : meta.form ? t('browser.formHere') : '';
    const keys = finishing ? t('browser.finishing') : focused ? t(touch ? 'browser.inControlNoteTouch' : 'browser.inControlNote') : t('browser.clickToType');
    hint = [t('browser.pausedHere'), why, keys].filter(Boolean).join(' ');
  }
  return {
    you,
    canTake: live && !you && meta.takeover === true,
    finishing,
    place,
    input: place === 'app' && !finishing,
    window: place === 'window' ? t('browser.showWindow') : place === 'app' && meta.form ? t('browser.openWindow') : '',
    status: you ? t('browser.inControl') : '',
    hint,
    dialog: you ? meta.dialog ?? null : null,
  };
}

/**
 * What the panel shows, from the latest `meta` (null before the first), the link to the gateway (`off`,
 * `connecting`, `open` or `retry`) and whether a picture is on the canvas. The states are calm and each says what
 * is going on and whether anything is needed (nothing ever is): `idle` (no page), `loading` (opening, or waiting
 * for the first picture), `live`, `stale` (the picture is kept but not updating, or the link dropped and is being
 * made again) and `closed` (the tab, the browser or the chat is gone).
 */
export function view(meta, link, {picture = false, known = false} = {}) {
  let state = meta?.state ?? (known ? 'loading' : 'idle');
  if (!meta && link === 'retry') state = known ? 'stale' : 'idle';
  if (meta && link === 'retry' && (state === 'live' || state === 'loading')) state = picture ? 'stale' : 'loading';
  const browsing = Boolean(meta?.browsing) && (state === 'live' || state === 'loading');
  const parts = addressParts(meta?.url);
  // A page with no title is called by its address, which the line below shows: say it has none.
  const bare = (text) => String(text ?? '').trim().replace(/^https?:\/\//, '').replace(/\/$/, '');
  const named = meta?.title && bare(meta.title) !== bare(meta.url) ? meta.title : '';
  const title = named || (state === 'idle' ? t('browser.idle.title') : t('browser.untitled'));
  let note = '';
  let status = '';
  if (state === 'idle') note = t('browser.idle.note');
  else if (state === 'loading') {
    note = t('browser.loading.note');
    status = browsing ? t('browser.browsing') : t('browser.loading.title');
  } else if (state === 'live') status = browsing ? t('browser.browsing') : t('browser.live');
  else if (state === 'stale') {
    note = link === 'retry' ? t('browser.reconnecting') : t('browser.stale.note');
    status = link === 'retry' ? t('browser.reconnecting') : t('browser.stale.title');
  } else note = t(`browser.closed.${meta?.reason || 'tab'}`);
  const shows = picture && (state === 'live' || state === 'stale');
  return {state, browsing, title, host: parts.host, rest: parts.rest, note, status, picture: shows};
}

/** A text node's words, written only when they change (a status that is read aloud must not repeat itself). */
function setText(node, text) {
  if (node && node.textContent !== text) node.textContent = text;
}

/** Listen to *type* on *target* when it can be listened to (a test's double may not). */
function on(target, type, fn, options) {
  if (target && typeof target.addEventListener === 'function') target.addEventListener(type, fn, options);
}

/**
 * The panel's behaviour. *els* are its parts: `root`, `chip` (and `chipText`), `title`, `host`, `rest`, `status`,
 * `viewport` (the box the picture is fitted in), `canvas`, `note`, `hide` and `pop`; for taking over, `take`, `give`,
 * `viewOnly`, `hint`, `keys` (the hidden text box keys go to) and the box's `dialog`, `dialogTitle`, `dialogText`,
 * `dialogInput`, `dialogOk` and `dialogCancel`. *api* reads paths below /api. *streams* is the page's one transport
 * (`createStreams`: the card and the dock watch the same sources); without it the panel makes its own from
 * *openSocket* (a WebSocket, so a test can give it a fake), *decode* (a picture's bytes to a bitmap the canvas draws),
 * *schedule*, *cancel* and *denied()* (the gateway refused the stream for the token's sake). *popOut* is set only
 * inside the macOS app. *changed({shown})* tells the page the panel opened or closed (it makes room for it).
 * *frame* runs a function before the next paint (pointer moves and wheel steps are sent at most once a frame).
 */
export function setupBrowser({
  api,
  els,
  streams = null,
  openSocket,
  decode,
  location = globalThis.location,
  schedule = (fn, ms) => setTimeout(fn, ms),
  cancel = (id) => clearTimeout(id),
  wide = () => true,
  popOut = null,
  changed = () => {},
  denied = () => {},
  observe = null,
  frame = (fn) => (globalThis.requestAnimationFrame ? globalThis.requestAnimationFrame(fn) : setTimeout(fn, 16)),
}) {
  const hub = streams ?? createStreams({openSocket, decode, location, schedule, cancel, denied});
  const S = {
    session: '',
    liveOnly: false,
    available: null, // whether this chat's browser can be watched at all (null: the gateway has not said)
    known: false, // the chat has opened a page: the chip may show
    shown: false,
    meta: null, // what the stream of the shown source says (browser-stream.mjs), mirrored here
    link: 'off',
    picture: false,
    unwatch: null, // lets go of the shown source
    source: null, // the source being watched: the chat's, pinned to a card's run and generation when it has one
    pin: null, // the `{gen, run}` of the chat's newest active browser card: the source the chip and Open browser show
    card: null, // that card's `{gen, run, state}`, or the newest card's once it ended (the chip's words)
    asking: false, // a question to the gateway is on its way
    generation: 0,
    focused: false, // the hidden text box has the keys (take over)
    wantFocus: false, // the person asked to take over: the keys go to the page once it is theirs
    pressed: new Set(), // keys sent down and not yet up
    composing: false,
    move: null, // the newest pointer move not sent yet, and the wheel steps summed since the last frame
    wheel: null,
    framed: false,
    tooLong: false, // a paste was too long: the hint says so until the next key
    committed: null, // the text an input method just committed (sent once, whatever follows it)
  };

  const working = () => S.card?.state === 'working';
  // A phone or a tablet has no Control key to leave the page with.
  const touchScreen = () => globalThis.matchMedia?.('(pointer: coarse)')?.matches === true;

  function render() {
    const v = view(S.meta, S.link, {picture: S.picture, known: S.known});
    const c = control(S.meta, {focused: S.focused, link: S.link, touch: touchScreen()});
    // The card says melo is at work before the stream does: show it on the panel too.
    const browsing = v.browsing || (working() && S.shown && v.state !== 'closed' && v.state !== 'idle');
    const {root, chip} = els;
    root.hidden = !S.shown;
    root.dataset.state = v.state;
    root.dataset.browsing = String(browsing && !c.you);
    root.dataset.owner = c.you ? 'you' : 'melo';
    root.dataset.place = c.place;
    root.dataset.keys = String(c.input && S.focused);
    setText(els.title, v.title);
    setText(els.host, v.host);
    setText(els.rest, v.rest);
    root.dataset.picture = String(v.picture);
    setText(els.status, c.you ? c.status : browsing && !v.status ? t('browser.browsing') : v.status);
    renderControl(c);
    setText(els.note, v.note);
    els.note.hidden = v.picture && v.state === 'live';
    els.note.dataset.over = String(v.picture);
    els.canvas.hidden = !v.picture;
    els.canvas.setAttribute('aria-label', `${t('browser.picture')}: ${v.title}`);
    if (els.pop) els.pop.hidden = !(popOut && S.shown && !S.liveOnly && v.state !== 'closed');
    if (els.hide) els.hide.hidden = S.liveOnly;
    if (chip) {
      chip.hidden = S.liveOnly || !(S.available && (S.known || S.shown));
      chip.dataset.browsing = String(!c.you && (browsing || (working() && !S.shown)));
      chip.setAttribute('aria-expanded', String(S.shown));
      chip.setAttribute('aria-label', S.shown ? t('browser.hide') : t('browser.show'));
      // One word for the state, the same on the card and the dock: the chat's browser while no run is using it.
      setText(els.chipText, c.you ? c.status : S.card && ACTIVE.has(S.card.state) ? stateWord(S.card.state) : t('browser.title'));
    }
    relayout();
  }

  /** The take-over parts: the buttons, the hint under them, the hidden text box and the page's box. */
  function renderControl(c) {
    if (els.take) els.take.hidden = !c.canTake;
    if (els.give) els.give.hidden = !c.you;
    if (els.window) {
      els.window.hidden = !c.window;
      setText(els.window, c.window);
    }
    if (els.viewOnly) els.viewOnly.hidden = c.canTake || c.you;
    if (els.hint) {
      setText(els.hint, S.tooLong && c.you ? t('browser.pasteTooLong', {n: MAX_PASTE}) : c.hint);
      els.hint.hidden = !c.you;
    }
    if (els.keys) {
      els.keys.disabled = !c.input;
      els.keys.tabIndex = c.input ? 0 : -1;
    }
    if (els.canvas) els.canvas.dataset.input = String(c.input);
    if (!c.input && S.focused && els.keys?.blur) els.keys.blur();
    if (c.input && S.wantFocus) {
      S.wantFocus = false;
      focusKeys(null);
    }
    renderDialog(c.dialog);
  }

  function renderDialog(box) {
    if (!els.dialog) return;
    const was = !els.dialog.hidden;
    els.dialog.hidden = !box;
    if (!box) return;
    const leaving = box.kind === 'beforeunload';
    setText(els.dialogTitle, leaving ? t('browser.dialog.leave') : t('browser.dialog.title'));
    setText(els.dialogText, box.message);
    if (els.dialogInput) {
      els.dialogInput.hidden = box.kind !== 'prompt';
      if (!was && box.kind === 'prompt') els.dialogInput.value = box.default ?? '';
    }
    if (els.dialogCancel) els.dialogCancel.hidden = box.kind === 'alert';
    setText(els.dialogOk, leaving ? t('browser.dialog.leaveOk') : t('browser.dialog.ok'));
    setText(els.dialogCancel, leaving ? t('browser.dialog.stay') : t('browser.dialog.cancel'));
    if (!was) (box.kind === 'prompt' ? els.dialogInput : els.dialogOk)?.focus?.();
  }

  function relayout() {
    const box = {width: els.viewport?.clientWidth ?? 0, height: els.viewport?.clientHeight ?? 0};
    const fit = layout(S.meta, box);
    if (!fit) return;
    Object.assign(els.canvas.style, {
      left: fit.left + 'px',
      top: fit.top + 'px',
      width: fit.width + 'px',
      height: fit.height + 'px',
    });
  }
  if (observe && els.viewport) observe(els.viewport, relayout);

  // ── The stream ──

  /** What the shown source's stream says, as the stream tells it. */
  const sink = {
    canvas: els.canvas,
    update(state) {
      S.meta = state.meta;
      S.link = state.link;
      S.picture = state.picture;
      if (state.link !== 'open') S.pressed.clear();
      if (state.meta && state.meta.state !== 'idle') S.known = true;
      render();
    },
  };

  /** The source the panel shows: the chat's browser, pinned to a card's run and generation when there is one. */
  const sourceOf = () => ({session: S.session, ...(S.pin ?? {})});

  function watch() {
    if (S.unwatch || !S.shown || !S.session) return;
    S.source = sourceOf();
    S.unwatch = hub.watch(S.source, sink);
  }

  function unwatch() {
    const stop = S.unwatch;
    S.unwatch = null;
    S.link = 'off';
    S.meta = null;
    S.picture = false;
    stop?.();
  }

  // ── Taking over ──

  /** Send one message to the gateway; whether it went. */
  const send = (message) => S.unwatch !== null && S.source !== null && hub.send(S.source, message);

  const inControl = () => control(S.meta, {link: S.link}).input;
  const box = () => ({width: els.viewport?.clientWidth ?? 0, height: els.viewport?.clientHeight ?? 0});

  /** Where *event* is in the picture's box. */
  function pointOf(event) {
    const rect = els.viewport?.getBoundingClientRect?.() ?? {left: 0, top: 0};
    return {x: event.clientX - rect.left, y: event.clientY - rect.top};
  }

  /** Give the keys to the page: the hidden text box is focused, at *point* so an input method's window opens there. */
  function focusKeys(point) {
    const keys = els.keys;
    if (!keys || typeof keys.focus !== 'function') return;
    if (point && keys.style) Object.assign(keys.style, {left: Math.round(point.x) + 'px', top: Math.round(point.y) + 'px'});
    resetKeys();
    keys.focus({preventScroll: true});
  }

  /** Empty the hidden text box: nothing typed stays in it. */
  function resetKeys() {
    const keys = els.keys;
    if (!keys) return;
    keys.value = SENTINEL;
    keys.setSelectionRange?.(SENTINEL.length, SENTINEL.length);
  }

  function flush() {
    S.framed = false;
    if (S.move) send(S.move);
    if (S.wheel) {
      const {point, dx, dy, mods} = S.wheel;
      const message = wheelMessage(S.meta, box(), point, dx, dy, mods);
      if (message) send(message);
    }
    S.move = null;
    S.wheel = null;
  }

  function soon() {
    if (S.framed) return;
    S.framed = true;
    frame(flush);
  }

  function pointer(event, action) {
    if (!inControl()) return;
    const point = pointOf(event);
    if (action === 'move') {
      const message = mouseMessage(S.meta, box(), point, event, 'move', {clamp: event.buttons > 0});
      if (message) {
        S.move = message;
        soon();
      }
      return;
    }
    event.preventDefault?.();
    if (action === 'down') {
      focusKeys(point);
      els.canvas?.setPointerCapture?.(event.pointerId);
    }
    flush(); // a move before the press arrives before it
    const message = mouseMessage(S.meta, box(), point, event, action, {clamp: action === 'up'});
    if (message) send(message);
  }

  function key(event, action) {
    S.committed = null;
    if (!inControl()) return;
    if (action === 'down' && event.key === 'Escape' && event.ctrlKey) {
      // Control-Esc leaves the page: the keys go back to the console.
      event.preventDefault?.();
      els.keys?.blur?.();
      els.give?.focus?.();
      return;
    }
    const message = keyMessage(event, action);
    if (!message) return;
    const id = `${message.code}|${message.key}`;
    if (action === 'up') {
      if (!S.pressed.delete(id) && !S.pressed.delete(`${message.code}|${message.key.toLowerCase()}`)) return;
    } else S.pressed.add(id);
    event.preventDefault?.();
    if (S.tooLong) {
      S.tooLong = false;
      render();
    }
    send(message);
  }

  /** A shortcut that arrived as an editing action (the macOS app's menu) rather than as a key: sent as its key. */
  function shortcut(letter, mods) {
    const code = `Key${letter.toUpperCase()}`;
    const keyCode = letter.toUpperCase().charCodeAt(0);
    send({type: 'key', action: 'down', key: letter, code, keyCode, mods});
    send({type: 'key', action: 'up', key: letter, code, keyCode, mods});
  }

  const keys = els.keys;
  on(els.canvas, 'pointerdown', (event) => pointer(event, 'down'));
  on(els.canvas, 'pointermove', (event) => pointer(event, 'move'));
  on(els.canvas, 'pointerup', (event) => pointer(event, 'up'));
  on(els.canvas, 'contextmenu', (event) => { if (inControl()) event.preventDefault?.(); });
  on(els.canvas, 'wheel', (event) => {
    if (!inControl()) return;
    event.preventDefault?.();
    const {dx, dy} = wheelPixels(event);
    const point = pointOf(event);
    S.wheel = S.wheel ? {...S.wheel, point, dx: S.wheel.dx + dx, dy: S.wheel.dy + dy} : {point, dx, dy, mods: modsOf(event)};
    soon();
  }, {passive: false});
  on(keys, 'keydown', (event) => key(event, 'down'));
  on(keys, 'keyup', (event) => key(event, 'up'));
  on(keys, 'focus', () => { S.focused = true; render(); });
  on(keys, 'blur', () => {
    S.focused = false;
    // Keys held when the box lost the keys are let go on the page too.
    for (const id of S.pressed) {
      const [code, name] = id.split('|');
      send({type: 'key', action: 'up', key: name, code, mods: 0, keyCode: 0});
    }
    S.pressed.clear();
    render();
  });
  on(keys, 'compositionstart', () => { S.composing = true; });
  on(keys, 'compositionupdate', (event) => { if (inControl()) send(composeMessage(event.data)); });
  on(keys, 'compositionend', (event) => {
    S.composing = false;
    const message = commitMessage(event.data);
    if (inControl()) send(message ?? composeMessage(''));
    // Some browsers say the commit again as an input right after: it is sent once.
    S.committed = message ? message.text : null;
    resetKeys();
  });
  on(keys, 'input', (event) => {
    if (S.composing || event.isComposing || /Composition/.test(String(event.inputType ?? ''))) return;
    const again = S.committed !== null && event.data === S.committed;
    S.committed = null;
    // Text that came by no key: the emoji picker, dictation, an input method that does not compose.
    if (!again && inControl() && typeof event.data === 'string' && event.inputType?.startsWith('insert')) {
      const message = commitMessage(event.data);
      if (message) send(message);
    }
    resetKeys();
  });
  on(keys, 'beforeinput', (event) => {
    // Undo and redo from the app's menu reach the box as editing actions, not keys.
    if (event.inputType === 'historyUndo' || event.inputType === 'historyRedo') {
      event.preventDefault?.();
      if (inControl()) shortcut('z', META | (event.inputType === 'historyRedo' ? SHIFT : 0));
    }
  });
  on(keys, 'select', () => {
    // "Select all" from the app's menu selects the box's one space: it is the page's select all.
    if (S.composing || !keys || keys.selectionStart !== 0 || keys.selectionEnd !== keys.value.length || !keys.value) return;
    if (inControl()) shortcut('a', META);
    resetKeys();
  });
  on(keys, 'cut', (event) => {
    event.preventDefault?.();
    if (inControl()) shortcut('x', META);
  });
  on(keys, 'copy', (event) => event.preventDefault?.());
  on(keys, 'paste', (event) => {
    event.preventDefault?.();
    if (!inControl()) return;
    const text = event.clipboardData?.getData?.('text/plain') ?? '';
    const message = pasteMessage(text);
    S.tooLong = !message && text.length > MAX_PASTE;
    if (message) send(message);
    render();
  });
  on(els.take, 'click', () => takeOver({ask: true}));
  on(els.give, 'click', () => {
    send({type: 'give'});
    els.keys?.blur?.();
  });
  on(els.window, 'click', () => showWindow());
  const answer = (accept) => {
    const text = els.dialogInput && !els.dialogInput.hidden ? String(els.dialogInput.value ?? '') : '';
    send({type: 'dialog', accept, text});
    if (els.dialogInput) els.dialogInput.value = '';
    focusKeys(null);
  };
  on(els.dialogOk, 'click', () => answer(true));
  on(els.dialogCancel, 'click', () => answer(false));
  on(els.dialogInput, 'keydown', (event) => {
    if (event.key === 'Enter' && !event.isComposing) {
      event.preventDefault?.();
      answer(true);
    }
  });

  /** Take the page over: from the panel's button (*ask*: the gateway is asked), or after the card was answered. */
  function takeOver({ask = false} = {}) {
    if (!S.session) return;
    S.available = S.available ?? true;
    S.wantFocus = true;
    show();
    if (ask) send({type: 'take'});
    render();
  }

  /** The page in a real Chrome window: moved there from the panel, or that window brought to the front. */
  function showWindow() {
    if (!S.session) return;
    const query = new URLSearchParams(S.source ? {...(S.source.gen ? {gen: S.source.gen} : {}), ...(S.source.run ? {run: S.source.run} : {})} : {}).toString();
    api(`/sessions/${encodeURIComponent(S.session)}/browser/window${query ? `?${query}` : ''}`, {method: 'POST'}).catch(() => {});
  }

  // ── Showing and hiding ──

  const sameSource = (a, b) => a?.session === b?.session && (a?.gen ?? '') === (b?.gen ?? '') && (a?.run ?? '') === (b?.run ?? '');

  /** Show the panel (never by itself: the person asked); one that shows another source moves to the one asked for. */
  function show() {
    if (S.shown) {
      if (S.unwatch && !sameSource(S.source, sourceOf())) {
        unwatch();
        watch();
      }
      return;
    }
    S.shown = true;
    changed({shown: true});
    watch();
    render();
  }

  function hide() {
    if (!S.shown) return;
    S.shown = false;
    unwatch();
    changed({shown: false});
    render();
  }

  function reset() {
    unwatch();
    S.generation += 1;
    S.known = false;
    S.available = null;
    S.pin = null;
    S.card = null;
    S.asking = false;
    const was = S.shown;
    S.shown = false;
    if (was) changed({shown: false});
  }

  async function probe() {
    const id = S.session;
    const generation = S.generation;
    let state = null;
    S.asking = true;
    try {
      state = await api(`/sessions/${encodeURIComponent(id)}/browser`);
    } catch {
      state = null;
    }
    if (S.session !== id || S.generation !== generation) return;
    S.asking = false;
    S.available = state?.available === true;
    if (state?.open) S.known = true;
    if (S.liveOnly) {
      // The pop-out window is the panel alone: it shows, and says so when there is nothing to watch.
      show();
      if (!S.available) S.meta = parseMeta('{"type":"meta","state":"closed","reason":"unavailable"}');
      render();
    } else if (!S.available && S.shown) {
      // The browser now chosen cannot be watched: the panel goes, and it was not the person who hid it.
      hide();
    } else render();
  }

  return {
    /** Bind the panel to chat *id* (the same chat again keeps what it has). `liveOnly`: the pop-out window's panel. */
    async follow(id, {liveOnly = false} = {}) {
      if (!id) return;
      if (id === S.session && S.liveOnly === liveOnly) {
        if (S.shown && !S.unwatch) watch();
        return;
      }
      reset();
      S.session = id;
      S.liveOnly = liveOnly;
      render();
      await probe();
    },

    /** The chat is left (for chat *next*, or for none): the panel goes, unless *next* is the chat it already follows. */
    release(next = '') {
      if (!S.session || (next && next === S.session)) return;
      reset();
      S.session = '';
      render();
    },

    /**
     * The state of the chat's newest browser card (`{gen, run, state}`, or null when it has none): the chip says it in
     * the card's words, and while the card is active its source is the one the chip (and a panel not asked for a
     * card) shows. A panel that shows the chat's browser moves to the card's source when a new card begins.
     */
    card(card) {
      if (!S.session || S.liveOnly) return;
      const live = Boolean(card) && ACTIVE.has(card.state);
      const pin = live ? {gen: card.gen, run: card.run} : null;
      const moved = pin !== null && !sameSource({session: S.session, ...pin}, {session: S.session, ...(S.pin ?? {})});
      S.card = card ? {gen: card.gen, run: card.run, state: card.state} : null;
      S.known = S.known || S.card !== null;
      S.pin = pin;
      if (moved && S.shown) show();
      render();
      // A chat last found unwatchable may browse in another browser now (Settings saved elsewhere): ask once for the run.
      if (moved && S.available === false) probe();
    },

    /**
     * Ask the gateway again whether the chat can be watched (this page saved Settings: the browser may be another
     * now). The chip and the panel follow the answer; nothing opens.
     */
    async recheck() {
      if (!S.session || S.liveOnly) return;
      S.known = S.card !== null;
      await probe();
    },

    /** Open browser, on a card or the dock: the panel shows that card's source (`{gen, run}`). */
    async open(card) {
      if (!S.session || S.liveOnly) return;
      const id = S.session;
      if (S.available === false) await probe(); // the browser may have changed since this page last asked
      if (S.session !== id || S.available === false) return;
      S.pin = {gen: card.gen, run: card.run};
      S.available = S.available ?? true;
      show();
    },

    /** The header chip: show the panel, or hide it. Hiding only stops watching. */
    toggle() {
      if (S.shown) hide();
      else if (S.session) {
        S.available = S.available ?? true;
        show();
      }
    },

    hide() {
      if (!S.shown || S.liveOnly) return;
      hide();
    },

    /** Escape on a phone: the sheet closes. True when it did. */
    closeSheet() {
      if (!S.shown || S.liveOnly || wide()) return false;
      this.hide();
      return true;
    },

    /** The pop-out button: the app shows this chat's panel alone in a window. */
    popOut() {
      if (popOut && S.session && S.shown && !S.liveOnly) popOut(S.session);
    },

    /**
     * The page is away: hidden for a while, or kept by the browser for Back. Nobody watches, so the sockets close
     * and the gateway stops the streams; the panel stays as it is.
     */
    suspend() {
      hub.suspend();
      if (S.shown) render();
    },

    /** The page is back: the streams are asked for again. */
    resume: () => hub.resume(),

    /** The person took the page over on the "melo needs you" card: the panel opens and the keys go to the page. */
    takeOver: () => takeOver(),

    /** The panel's Take over button. */
    take: () => takeOver({ask: true}),

    /** The panel's Done button. */
    give: () => send({type: 'give'}),

    /** Show Chrome window, or Open in Chrome (the card's choice for a page with a form). */
    showWindow,

    relayout,
    isShown: () => S.shown,
    state: () => ({...S, socket: S.unwatch !== null, unwatch: undefined}),
  };
}

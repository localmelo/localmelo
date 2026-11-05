import {t} from './i18n.mjs';

/**
 * The Browser panel: a live window onto the page melo is working in, which the person can take over.
 *
 * It sits beside the chat and opens by itself while melo browses: when a `browser_*` tool call starts in the chat's
 * event stream, or when the chat is opened in the middle of a browser step or with a page the person took over. When
 * the turn ends it goes again, unless the person opened it or has the page; the chat's header keeps a chip that
 * brings it back, and a panel the person hid does not open by itself again for that chat. Whether the chat can be
 * watched (`available`: its browser is Chrome and the live view is on) is asked when the chat is followed, again when
 * melo browses in a chat that could not be watched (the browser was changed in Settings elsewhere), and on
 * `recheck()` (this page saved Settings). The pictures come from one WebSocket per open panel,
 * `/api/sessions/<id>/browser/live` (support/gateway/console/routes_browser.py): JSON `meta` messages say the state
 * (`idle`, `loading`, `live`, `stale`, `closed`), the tab, its address and title, whether melo is in the middle of a
 * browser step, and the size of the page view; binary messages are JPEG pictures, of which only the newest is
 * drawn. A panel that is hidden holds no socket, so the gateway stops the stream and melo's browser costs nothing.
 *
 * `follow(id)` / `release()` bind the panel to the chat that is open, `note(frame)` reads that chat's events, and
 * A page that is hidden for a while, or kept by the browser for Back, lets go of its socket (`suspend`) and asks
 * again when it is back (`resume`). `#/live/<id>` (`liveRoute`, `liveUrl`) is the panel alone, which **Pop out** opens
 * large: in its own window in the macOS app, in a window of the browser elsewhere. The pure helpers
 * (`parseMeta`, `view`, `layout`, `toPage`, `addressParts`, `control` and the input messages) are what the tests check.
 *
 * Take over: **Take over** (or the "melo needs you" card, `takeOver()`) hands the tab to the person (`meta.owner` is
 * then `you`). Their clicks, drags and wheel on the picture, and their keys, input-method text and pastes in a hidden
 * text box over it, go to the gateway as small JSON messages (`mouseMessage`, `wheelMessage`, `keyMessage`,
 * `composeMessage`, `commitMessage`, `pasteMessage`) and from there to melo's browser as its own input. A point on the
 * picture becomes a point on the page by the frame's own metadata (`toPage`: its size, the space above it and its
 * zoom). The page's boxes are answered in the panel. **Give back to melo** returns the tab. Nothing typed is kept here
 * either: the text box is emptied after every key.
 */

export const RETRY_S = [1, 2, 4, 8, 16, 30];
export const STATES = ['idle', 'loading', 'live', 'stale', 'closed'];
const REASONS = new Set(['tab', 'browser', 'chat', 'unavailable']);
const DIALOGS = new Set(['alert', 'confirm', 'prompt', 'beforeunload']);
// Close codes the gateway uses: a refusal (policy), and a store that is down (try again).
const REFUSED = 1008;
const MAX_TEXT = 2048;
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

/** The WebSocket address of one chat's browser on this page's own origin, optionally pinned to a tab. */
export function socketUrl(location, session, tab = '') {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const query = tab ? `?tab=${encodeURIComponent(tab)}` : '';
  return `${scheme}//${location.host}/api/sessions/${encodeURIComponent(session)}/browser/live${query}`;
}

/** Seconds to wait before the next try, by how many tries failed in a row. */
export const retryDelay = (attempt) => RETRY_S[Math.min(Math.max(attempt, 0), RETRY_S.length - 1)];

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

const finite = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const words = (value) => (typeof value === 'string' ? value.slice(0, MAX_TEXT) : '');

/** The `meta` message in *text*, cleaned, or null when it is not one. Nothing in it is trusted beyond its shape. */
export function parseMeta(text) {
  let raw;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!raw || typeof raw !== 'object' || raw.type !== 'meta' || !STATES.includes(raw.state)) return null;
  const meta = {
    state: raw.state,
    tab: typeof raw.tab === 'string' && /^t[0-9]{1,6}$/.test(raw.tab) ? raw.tab : '',
    url: words(raw.url),
    title: words(raw.title),
    browsing: raw.browsing === true,
    owner: raw.owner === 'you' ? 'you' : 'melo',
    reason: REASONS.has(raw.reason) ? raw.reason : '',
    w: 0,
    h: 0,
    offsetTop: 0,
    scale: 1,
  };
  const w = finite(raw.w);
  const h = finite(raw.h);
  if (w !== null && h !== null && w > 0 && h > 0) {
    meta.w = w;
    meta.h = h;
    meta.offsetTop = Math.max(0, finite(raw.offsetTop) ?? 0);
    const scale = finite(raw.scale) ?? 1;
    meta.scale = scale > 0 ? scale : 1;
  }
  meta.takeover = raw.takeover === true;
  meta.grace = meta.owner === 'you' ? Math.max(0, Math.round(finite(raw.grace) ?? 0)) : 0;
  const box = raw.dialog;
  meta.dialog = meta.owner === 'you' && box && typeof box === 'object' && DIALOGS.has(box.kind)
    ? {kind: box.kind, message: words(box.message).slice(0, 300), default: words(box.default).slice(0, 300)}
    : null;
  return meta;
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
 * when they took it (their input waits).
 */
export function control(meta, {focused = false, link = 'open'} = {}) {
  const live = Boolean(meta) && ['live', 'loading', 'stale'].includes(meta.state) && link === 'open';
  const you = live && meta.owner === 'you';
  const finishing = you && meta.browsing === true;
  let hint = '';
  if (you) {
    hint = finishing ? t('browser.finishing') : focused ? t('browser.inControlNote') : t('browser.clickToType');
    if (meta.grace > 0) hint += ` ${t('browser.grace', {minutes: Math.max(1, Math.round(meta.grace / 60))})}`;
  }
  return {
    you,
    canTake: live && !you && meta.takeover === true,
    finishing,
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
 * `dialogInput`, `dialogOk` and `dialogCancel`. *api* reads paths below /api and *openSocket* opens a WebSocket (so a
 * test can give it a fake). *decode* turns a picture's bytes into a bitmap the canvas draws. *popOut* is set only
 * inside the macOS app. *changed({shown})* tells the page the panel opened or closed (it makes room for it), and
 * *denied()* that the gateway refused the stream for the token's sake. *frame* runs a function before the next paint
 * (pointer moves and wheel steps are sent at most once a frame).
 */
export function setupBrowser({
  api,
  els,
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
  const S = {
    session: '',
    liveOnly: false,
    available: null, // whether this chat's browser can be watched at all (null: the gateway has not said)
    known: false, // the chat has opened a page: the chip may show
    shown: false,
    suspended: false, // the page is away (hidden, or kept for Back): the stream is not asked for
    meta: null,
    link: 'off',
    attempt: 0,
    socket: null,
    timer: null,
    latest: null,
    drawing: false,
    picture: false,
    calls: new Set(), // browser tool calls of the chat that are running (from its events)
    dismissed: new Set(), // chats whose panel the person hid: it does not open by itself again
    auto: false, // the panel opened by itself: it goes again when the browsing ends
    stepped: false, // melo browsed in the turn that runs
    asked: false, // the gateway was asked again in this turn (a chat that could not be watched)
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

  const hint = () => S.calls.size > 0;

  function render() {
    const v = view(S.meta, S.link, {picture: S.picture, known: S.known});
    const c = control(S.meta, {focused: S.focused, link: S.link});
    // Melo is mid-step before the stream says so (the event came first): show it on the chip and the panel.
    const browsing = v.browsing || (hint() && S.shown && v.state !== 'closed' && v.state !== 'idle');
    const {root, chip} = els;
    root.hidden = !S.shown;
    root.dataset.state = v.state;
    root.dataset.browsing = String(browsing && !c.you);
    root.dataset.owner = c.you ? 'you' : 'melo';
    root.dataset.keys = String(c.you && S.focused);
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
      chip.dataset.browsing = String(browsing || (hint() && !S.shown));
      chip.setAttribute('aria-expanded', String(S.shown));
      chip.setAttribute('aria-label', S.shown ? t('browser.hide') : t('browser.show'));
      setText(els.chipText, browsing || (hint() && !S.shown) ? t('browser.browsing') : t('browser.title'));
    }
    relayout();
  }

  /** The take-over parts: the buttons, the hint under them, the hidden text box and the page's box. */
  function renderControl(c) {
    if (els.take) els.take.hidden = !c.canTake;
    if (els.give) els.give.hidden = !c.you;
    if (els.viewOnly) els.viewOnly.hidden = c.canTake || c.you;
    if (els.hint) {
      setText(els.hint, S.tooLong && c.you ? t('browser.pasteTooLong', {n: MAX_PASTE}) : c.hint);
      els.hint.hidden = !c.you;
    }
    if (els.keys) {
      els.keys.disabled = !c.you;
      els.keys.tabIndex = c.you ? 0 : -1;
    }
    if (els.canvas) els.canvas.dataset.input = String(c.you && !c.finishing);
    if (!c.you && S.focused && els.keys?.blur) els.keys.blur();
    if (c.you && S.wantFocus && !c.finishing) {
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

  // ── The socket ──

  function dropSocket() {
    if (S.timer !== null) {
      cancel(S.timer);
      S.timer = null;
    }
    const socket = S.socket;
    S.socket = null;
    S.link = 'off';
    S.latest = null;
    if (socket) {
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      try {
        socket.close(1000);
      } catch {
        /* A socket that is already closed needs nothing. */
      }
    }
  }

  function retry() {
    if (!S.shown || S.suspended || S.timer !== null || !S.session) return;
    S.link = 'retry';
    const wait = retryDelay(S.attempt);
    S.attempt += 1;
    S.timer = schedule(() => {
      S.timer = null;
      connect();
    }, wait * 1000);
    render();
  }

  function connect() {
    if (S.socket || !S.shown || S.suspended || !S.session) return;
    const generation = S.generation;
    let socket;
    try {
      socket = openSocket(socketUrl(location, S.session));
    } catch {
      retry();
      return;
    }
    socket.binaryType = 'arraybuffer';
    S.socket = socket;
    S.link = 'connecting';
    socket.onopen = () => {
      if (S.socket !== socket || S.generation !== generation) return;
      S.link = 'open';
      S.attempt = 0;
      render();
    };
    socket.onmessage = (event) => {
      if (S.socket !== socket || S.generation !== generation) return;
      if (typeof event.data === 'string') {
        const meta = parseMeta(event.data);
        if (!meta) return;
        if (meta.tab !== (S.meta?.tab ?? '') || meta.state === 'idle' || meta.state === 'closed') S.picture = false;
        S.meta = meta;
        if (meta.state !== 'idle') S.known = true;
        render();
        return;
      }
      S.latest = event.data;
      pump(generation);
    };
    socket.onclose = (event) => {
      if (S.socket !== socket || S.generation !== generation) return;
      S.socket = null;
      S.link = 'off';
      S.pressed.clear();
      if (event?.code === REFUSED) {
        if (event.reason === 'token_rotated') {
          denied();
          retry();
        } else {
          S.meta = {...(S.meta ?? parseMeta('{"type":"meta","state":"closed"}')), state: 'closed', reason: 'unavailable', browsing: false};
          S.picture = false;
          render();
        }
        return;
      }
      // The gateway ended a closed view on purpose: it stays as it was told, with nothing to retry.
      if (event?.code === 1000 && (S.meta?.state === 'closed')) {
        render();
        return;
      }
      retry();
    };
    socket.onerror = () => {};
    render();
  }

  /** Draw the newest picture; one that arrives while another is being decoded takes its place. */
  async function pump(generation) {
    if (S.drawing) return;
    S.drawing = true;
    try {
      while (S.latest !== null) {
        const data = S.latest;
        S.latest = null;
        let bitmap;
        try {
          bitmap = await decode(data);
        } catch {
          continue;
        }
        if (S.generation !== generation) {
          bitmap?.close?.();
          return;
        }
        const canvas = els.canvas;
        if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
        if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
        canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
        bitmap.close?.();
        S.picture = true;
        render();
      }
    } finally {
      S.drawing = false;
    }
  }

  // ── Taking over ──

  /** Send one message to the gateway; whether it went. */
  function send(message) {
    const socket = S.socket;
    if (!socket || S.link !== 'open' || typeof socket.send !== 'function') return false;
    try {
      socket.send(JSON.stringify(message));
      return true;
    } catch {
      return false;
    }
  }

  const inControl = () => {
    const c = control(S.meta, {link: S.link});
    return c.you && !c.finishing;
  };
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
    S.dismissed.delete(S.session);
    S.available = S.available ?? true;
    S.wantFocus = true;
    S.auto = false;
    show();
    if (ask) send({type: 'take'});
    render();
  }

  // ── Showing and hiding ──

  function show() {
    if (S.shown) return;
    S.shown = true;
    changed({shown: true});
    connect();
    render();
  }

  function hide() {
    if (!S.shown) return;
    S.shown = false;
    dropSocket();
    changed({shown: false});
    render();
  }

  /** Whether the panel may open by itself: not for a chat whose panel was hidden, and below 900 px only on request. */
  const wantsOpen = () => S.liveOnly || (!S.dismissed.has(S.session) && wide());

  /** Open the panel by itself while melo browses (a step runs, or *awake*: the gateway says so or the page is the person's). */
  function reveal(awake = false) {
    if (!S.session) return;
    if (S.liveOnly) {
      show();
      return;
    }
    if (S.available === true && S.known && !S.shown && wantsOpen() && (hint() || awake)) {
      S.auto = true;
      show();
    } else render();
  }

  /** The browsing ended (the turn did): a panel that opened by itself goes, and the chip stays to bring it back. */
  function settle() {
    const browsed = S.stepped;
    S.stepped = false;
    S.asked = false;
    S.calls.clear();
    if (browsed && S.auto && S.shown && S.meta?.owner !== 'you') {
      S.auto = false;
      hide();
    } else render();
  }

  function reset() {
    dropSocket();
    S.generation += 1;
    S.meta = null;
    S.picture = false;
    S.known = false;
    S.available = null;
    S.calls.clear();
    S.auto = false;
    S.stepped = false;
    S.asked = false;
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
    if (S.liveOnly && !S.available) {
      S.meta = parseMeta('{"type":"meta","state":"closed","reason":"unavailable"}');
    } else if (!S.available && S.shown && !S.liveOnly) {
      // The browser now chosen cannot be watched: the panel goes, and it was not the person who hid it.
      S.auto = false;
      hide();
    }
    reveal(state?.browsing === true || state?.yours === true);
  }

  return {
    /** Bind the panel to chat *id* (the same chat again keeps what it has). `liveOnly`: the pop-out window's panel. */
    async follow(id, {liveOnly = false} = {}) {
      if (!id) return;
      if (id === S.session && S.liveOnly === liveOnly) {
        if (S.shown && !S.socket) connect();
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
     * Read one event of the followed chat: a browser tool call starting is melo beginning to browse, and the turn
     * finishing is the browsing ending. In a chat that could not be watched, the first browser step of a turn asks
     * the gateway again: the browser may have been changed in Settings since the chat was followed.
     */
    note(frame) {
      if (!S.session || S.liveOnly) return;
      const type = String(frame?.type ?? '');
      const data = frame?.data ?? {};
      if (type === 'tool.started' && String(data.name ?? '').startsWith('browser_')) {
        S.calls.add(String(data.call_id ?? ''));
        S.known = true;
        S.stepped = true;
        if (S.available === false && !S.asked && !S.asking) {
          S.asked = true;
          probe();
        }
        reveal();
      } else if (type === 'tool.finished') {
        if (S.calls.delete(String(data.call_id ?? ''))) render();
      } else if (type === 'turn.finished') {
        settle();
      }
    },

    /**
     * Ask the gateway again whether the chat can be watched (this page saved Settings: the browser may be another
     * now). The chip and the panel follow the answer; nothing opens unless melo is browsing.
     */
    async recheck() {
      if (!S.session || S.liveOnly) return;
      S.known = S.calls.size > 0;
      await probe();
    },

    /** The header chip: show the panel, or hide it (a hidden panel stays hidden for this chat until asked). */
    toggle() {
      S.auto = false;
      if (S.shown) {
        S.dismissed.add(S.session);
        hide();
      } else if (S.session) {
        S.dismissed.delete(S.session);
        S.available = S.available ?? true;
        show();
      }
    },

    hide() {
      if (!S.shown || S.liveOnly) return;
      S.auto = false;
      S.dismissed.add(S.session);
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
     * The page is away: hidden for a while, or kept by the browser for Back. Nobody watches, so the socket closes
     * and the gateway stops the stream; the panel stays as it is. A page that is kept would otherwise hold its
     * socket open, and the stream with it, until the browser lets the page go.
     */
    suspend() {
      S.suspended = true;
      dropSocket();
      if (S.shown) render();
    },

    /** The page is back: the stream is asked for again if the panel is shown. */
    resume() {
      if (!S.suspended) return;
      S.suspended = false;
      connect();
    },

    /** The person took the page over on the "melo needs you" card: the panel opens and the keys go to the page. */
    takeOver: () => takeOver(),

    /** The panel's Take over button. */
    take: () => takeOver({ask: true}),

    /** The panel's Give back to melo button. */
    give: () => send({type: 'give'}),

    relayout,
    isShown: () => S.shown,
    state: () => ({...S, calls: [...S.calls], dismissed: [...S.dismissed], socket: S.socket !== null, latest: undefined}),
  };
}

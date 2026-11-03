import {t} from './i18n.mjs';

/**
 * The Browser panel: a live, view-only window onto the page melo is working in.
 *
 * It sits beside the chat, opens by itself when melo starts browsing (a `browser_*` tool call in the chat's event
 * stream) or when the chat already has a page open, and can be hidden; the chat's header then keeps a chip that
 * brings it back. The pictures come from one WebSocket per open panel,
 * `/api/sessions/<id>/browser/live` (support/gateway/console/routes_browser.py): JSON `meta` messages say the state
 * (`idle`, `loading`, `live`, `stale`, `closed`), the tab, its address and title, whether melo is in the middle of a
 * browser step, and the size of the page view; binary messages are JPEG pictures, of which only the newest is
 * drawn. A panel that is hidden holds no socket, so the gateway stops the stream and melo's browser costs nothing.
 *
 * `follow(id)` / `release()` bind the panel to the chat that is open, `note(frame)` reads that chat's events, and
 * A page that is hidden for a while, or kept by the browser for Back, lets go of its socket (`suspend`) and asks
 * again when it is back (`resume`). `#/live/<id>` (`liveRoute`) is the panel alone, which the macOS app's pop-out window opens. The pure helpers
 * (`parseMeta`, `view`, `layout`, `toPage`, `addressParts`) are what the tests check; `layout` and `toPage` are the
 * pair a later take-over uses to turn a point on the picture into a point on the page.
 */

export const RETRY_S = [1, 2, 4, 8, 16, 30];
export const STATES = ['idle', 'loading', 'live', 'stale', 'closed'];
const REASONS = new Set(['tab', 'browser', 'chat', 'unavailable']);
// Close codes the gateway uses: a refusal (policy), and a store that is down (try again).
const REFUSED = 1008;
const MAX_TEXT = 2048;

/** The chat id of a `#/live/<id>` address, or ''. */
export const liveRoute = (hash) => /^#\/live\/([A-Za-z0-9._:-]+)$/.exec(hash ?? '')?.[1] ?? '';

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
    meta.scale = (finite(raw.scale) ?? 1) || 1;
  }
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
 * The page point under *point* (box pixels, from the box's top left), by the frame's own metadata: the page view is
 * `w` x `h` CSS pixels and `offsetTop` is the space the browser keeps above the page. Null outside the picture.
 * The browser's zoom (`scale`) is not folded in: melo's browser is never pinch-zoomed.
 */
export function toPage(meta, box, point) {
  const fit = layout(meta, box);
  if (!fit || !Number.isFinite(point?.x) || !Number.isFinite(point?.y)) return null;
  const x = (point.x - fit.left) / fit.k;
  const y = (point.y - fit.top) / fit.k - (meta.offsetTop || 0);
  if (x < 0 || y < 0 || x > meta.w || y > meta.h) return null;
  return {x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100};
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

/**
 * The panel's behaviour. *els* are its parts: `root`, `chip` (and `chipText`), `title`, `host`, `rest`, `status`,
 * `viewport` (the box the picture is fitted in), `canvas`, `note`, `hide` and `pop`. *api* reads paths below /api
 * and *openSocket* opens a WebSocket (so a test can give it a fake). *decode* turns a picture's bytes into a bitmap
 * the canvas draws. *popOut* is set only inside the macOS app. *changed({shown})* tells the page the panel opened
 * or closed (it makes room for it), and *denied()* that the gateway refused the stream for the token's sake.
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
    generation: 0,
  };

  const hint = () => S.calls.size > 0;

  function render() {
    const v = view(S.meta, S.link, {picture: S.picture, known: S.known});
    // Melo is mid-step before the stream says so (the event came first): show it on the chip and the panel.
    const browsing = v.browsing || (hint() && S.shown && v.state !== 'closed' && v.state !== 'idle');
    const {root, chip} = els;
    root.hidden = !S.shown;
    root.dataset.state = v.state;
    root.dataset.browsing = String(browsing);
    setText(els.title, v.title);
    setText(els.host, v.host);
    setText(els.rest, v.rest);
    root.dataset.picture = String(v.picture);
    setText(els.status, browsing && !v.status ? t('browser.browsing') : v.status);
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

  function reveal() {
    if (!S.session) return;
    if (S.liveOnly) {
      show();
      return;
    }
    if (S.available === true && S.known && !S.shown && wantsOpen()) show();
    else render();
  }

  function reset() {
    dropSocket();
    S.generation += 1;
    S.meta = null;
    S.picture = false;
    S.known = false;
    S.available = null;
    S.calls.clear();
    const was = S.shown;
    S.shown = false;
    if (was) changed({shown: false});
  }

  async function probe() {
    const id = S.session;
    const generation = S.generation;
    let state = null;
    try {
      state = await api(`/sessions/${encodeURIComponent(id)}/browser`);
    } catch {
      state = null;
    }
    if (S.session !== id || S.generation !== generation) return;
    S.available = state?.available === true;
    if (state?.open) S.known = true;
    if (S.liveOnly && !S.available) {
      S.meta = parseMeta('{"type":"meta","state":"closed","reason":"unavailable"}');
    }
    reveal();
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

    /** Read one event of the followed chat: a browser tool call starting is melo beginning to browse. */
    note(frame) {
      if (!S.session || S.liveOnly) return;
      const type = String(frame?.type ?? '');
      const data = frame?.data ?? {};
      if (type === 'tool.started' && String(data.name ?? '').startsWith('browser_')) {
        S.calls.add(String(data.call_id ?? ''));
        S.known = true;
        reveal();
      } else if (type === 'tool.finished') {
        if (S.calls.delete(String(data.call_id ?? ''))) render();
      } else if (type === 'turn.finished' && S.calls.size > 0) {
        S.calls.clear();
        render();
      }
    },

    /** The header chip: show the panel, or hide it (a hidden panel stays hidden for this chat until asked). */
    toggle() {
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

    relayout,
    isShown: () => S.shown,
    state: () => ({...S, calls: [...S.calls], dismissed: [...S.dismissed], socket: S.socket !== null, latest: undefined}),
  };
}

/**
 * The transport of melo's live browser view: one WebSocket and one picture decoder for each source the page shows,
 * whoever shows it.
 *
 * A source is one chat's browser, optionally pinned to the run and the browser generation of a card
 * (`{session, gen, run, tab}`: `/api/sessions/<id>/browser/live`, support/gateway/console/routes_browser.py, refuses a
 * source that is not the chat's now). The card in the conversation, the dock above it and the Browser panel each
 * `watch(source, sink)` while they are visible and let go when they are not; the first watcher opens the socket and
 * the last one closes it, so a hidden view holds nothing and the gateway stops its stream. JSON `meta` messages say
 * the state of the view (`idle`, `loading`, `live`, `stale`, `closed`), the tab, its address and title, whether melo is
 * in the middle of a browser step and the size of the page view; binary messages are JPEG pictures, of which only the
 * newest is decoded, once, and painted straight onto the canvas of every sink: nothing is redrawn at the stream's rate.
 *
 * A sink is `{canvas, update(state)}`; `state` is `{meta, link, picture}`: the latest `meta` (null before the first),
 * the link to the gateway (`off`, `connecting`, `open` or `retry`) and whether a picture has been painted since the
 * source was first watched. A lost link is tried again, later and later; a refusal, and a view the gateway ended on
 * purpose, are not. A picture is painted only if what it belongs to still shows when it is decoded: one that arrives
 * after its source was let go, or whose tab, view or connection changed while it was decoding, is dropped.
 */

export const RETRY_S = [1, 2, 4, 8, 16, 30];
export const STATES = ['idle', 'loading', 'live', 'stale', 'closed'];
const REASONS = new Set(['tab', 'browser', 'chat', 'unavailable']);
const DIALOGS = new Set(['alert', 'confirm', 'prompt', 'beforeunload']);
// Close codes the gateway uses: a refusal (policy), and a store that is down (try again).
const REFUSED = 1008;
const MAX_TEXT = 2048;

const finite = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const words = (value) => (typeof value === 'string' ? value.slice(0, MAX_TEXT) : '');

/**
 * The WebSocket address of a source on this page's own origin. *pin* is a tab id, or what else the source names:
 * `{tab, gen, run}` (a card's browser generation and run).
 */
export function socketUrl(location, session, pin = '') {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const named = typeof pin === 'string' ? {tab: pin} : pin ?? {};
  const query = new URLSearchParams();
  for (const key of ['tab', 'gen', 'run']) if (named[key]) query.set(key, named[key]);
  const text = query.toString();
  return `${scheme}//${location.host}/api/sessions/${encodeURIComponent(session)}/browser/live${text ? `?${text}` : ''}`;
}

/** Seconds to wait before the next try, by how many tries failed in a row. */
export const retryDelay = (attempt) => RETRY_S[Math.min(Math.max(attempt, 0), RETRY_S.length - 1)];

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
  const yours = meta.owner === 'you';
  meta.place = yours ? (raw.place === 'window' ? 'window' : 'app') : '';
  meta.form = yours && raw.form === true;
  meta.failed = yours && raw.failed === true;
  meta.busy = meta.failed && raw.busy === true;
  const box = raw.dialog;
  meta.dialog = meta.owner === 'you' && box && typeof box === 'object' && DIALOGS.has(box.kind)
    ? {kind: box.kind, message: words(box.message).slice(0, 300), default: words(box.default).slice(0, 300)}
    : null;
  return meta;
}

const keyOf = (source) => [source.session, source.gen ?? '', source.run ?? '', source.tab ?? ''].join('|');

/**
 * The page's streams. *openSocket* opens a WebSocket (so a test can give it a fake) and *decode* turns a picture's
 * bytes into a bitmap a canvas draws; *schedule* and *cancel* are the clock of the retries; *denied()* says the
 * gateway ended a stream for the token's sake (the page checks it, and the stream is tried again).
 */
export function createStreams({
  openSocket,
  decode,
  location = globalThis.location,
  schedule = (fn, ms) => setTimeout(fn, ms),
  cancel = (id) => clearTimeout(id),
  denied = () => {},
}) {
  const entries = new Map();
  // The page is away (hidden for a while, or kept by the browser for Back): nothing is connected.
  let away = false;

  const snapshot = (entry) => ({meta: entry.meta, link: entry.link, picture: entry.picture});

  function notify(entry) {
    const state = snapshot(entry);
    for (const sink of [...entry.sinks]) sink.update?.(state);
  }

  /** What a picture still being decoded belongs to is over (another tab, a closed view, another connection). */
  function fence(entry) {
    entry.epoch += 1;
    entry.latest = null;
  }

  function dropSocket(entry) {
    fence(entry);
    if (entry.timer !== null) {
      cancel(entry.timer);
      entry.timer = null;
    }
    const socket = entry.socket;
    entry.socket = null;
    entry.link = 'off';
    if (socket) {
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      try {
        socket.close(1000);
      } catch {
        /* A socket that is already closed needs nothing. */
      }
    }
  }

  function retry(entry) {
    if (away || entry.timer !== null || !entry.alive) return;
    entry.link = 'retry';
    const wait = retryDelay(entry.attempt);
    entry.attempt += 1;
    entry.timer = schedule(() => {
      entry.timer = null;
      connect(entry);
    }, wait * 1000);
    notify(entry);
  }

  function connect(entry) {
    if (entry.socket || entry.timer !== null || away || !entry.alive) return;
    let socket;
    try {
      socket = openSocket(socketUrl(location, entry.source.session, entry.source));
    } catch {
      retry(entry);
      return;
    }
    socket.binaryType = 'arraybuffer';
    entry.socket = socket;
    entry.link = 'connecting';
    socket.onopen = () => {
      if (entry.socket !== socket) return;
      entry.link = 'open';
      entry.attempt = 0;
      notify(entry);
    };
    socket.onmessage = (event) => {
      if (entry.socket !== socket) return;
      if (typeof event.data === 'string') {
        const meta = parseMeta(event.data);
        if (!meta) return;
        if (meta.tab !== (entry.meta?.tab ?? '') || meta.state === 'idle' || meta.state === 'closed') {
          entry.picture = false;
          fence(entry);
        }
        entry.meta = meta;
        notify(entry);
        return;
      }
      entry.latest = event.data;
      pump(entry);
    };
    socket.onclose = (event) => {
      if (entry.socket !== socket) return;
      entry.socket = null;
      entry.link = 'off';
      fence(entry);
      if (event?.code === REFUSED) {
        if (event.reason === 'token_rotated') {
          denied();
          retry(entry);
        } else {
          // A source that is not the chat's now (its run ended, its browser was rebuilt) reads as a stopped browser.
          const reason = event.reason === 'stale_source' ? 'browser' : 'unavailable';
          entry.meta = {...(entry.meta ?? parseMeta('{"type":"meta","state":"closed"}')), state: 'closed', reason, browsing: false};
          entry.picture = false;
          notify(entry);
        }
        return;
      }
      // The gateway ended a closed view on purpose: it stays as it was told, with nothing to retry.
      if (event?.code === 1000 && entry.meta?.state === 'closed') {
        notify(entry);
        return;
      }
      retry(entry);
    };
    socket.onerror = () => {};
    notify(entry);
  }

  /** Decode the newest picture and paint it on every sink; one that arrives while another decodes takes its place. */
  async function pump(entry) {
    if (entry.drawing) return;
    entry.drawing = true;
    try {
      while (entry.latest !== null) {
        const data = entry.latest;
        const epoch = entry.epoch;
        entry.latest = null;
        let bitmap;
        try {
          bitmap = await decode(data);
        } catch {
          continue;
        }
        // The source was let go while the picture was being decoded: it is not painted on whatever shows now.
        if (!entry.alive) {
          bitmap?.close?.();
          return;
        }
        // Nor is a picture of a page that changed meanwhile: only a picture that came after the change is of what shows.
        if (entry.epoch !== epoch) {
          bitmap?.close?.();
          continue;
        }
        for (const sink of entry.sinks) paint(sink.canvas, bitmap);
        bitmap.close?.();
        entry.picture = true;
        notify(entry);
      }
    } finally {
      entry.drawing = false;
    }
  }

  function paint(canvas, bitmap) {
    if (!canvas) return;
    if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
    if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
    canvas.getContext?.('2d')?.drawImage(bitmap, 0, 0);
  }

  /** A sink that joins a source already showing a picture gets a copy of it: the page may not change for a while. */
  function copyTo(entry, sink) {
    const from = [...entry.sinks].find((other) => other !== sink && other.canvas && entry.picture);
    if (!from || !sink.canvas) return;
    paint(sink.canvas, from.canvas);
  }

  return {
    /**
     * Show *source* on *sink* until the returned function is called. The first watcher of a source opens its
     * socket, the last one to let go closes it.
     */
    watch(source, sink) {
      const key = keyOf(source);
      let entry = entries.get(key);
      if (!entry) {
        entry = {
          key, source: {...source}, sinks: new Set(), meta: null, link: 'off', attempt: 0, socket: null, timer: null,
          latest: null, drawing: false, picture: false, alive: true, epoch: 0,
        };
        entries.set(key, entry);
      }
      entry.sinks.add(sink);
      const shown = entry;
      copyTo(entry, sink);
      sink.update?.(snapshot(entry));
      connect(entry);
      let released = false;
      return () => {
        if (released) return;
        released = true;
        shown.sinks.delete(sink);
        if (shown.sinks.size > 0) return;
        shown.alive = false;
        dropSocket(shown);
        if (entries.get(key) === shown) entries.delete(key);
      };
    },

    /** Send one message to a source's gateway end; whether it went (only while the link is open). */
    send(source, message) {
      const entry = entries.get(keyOf(source));
      const socket = entry?.socket;
      if (!entry || !socket || entry.link !== 'open' || typeof socket.send !== 'function') return false;
      try {
        socket.send(JSON.stringify(message));
        return true;
      } catch {
        return false;
      }
    },

    /**
     * The page is away: hidden for a while, or kept by the browser for Back. Nobody watches, so every socket closes
     * and the gateway stops its streams; the watchers stay as they are. A page that is kept would otherwise hold
     * its sockets open until the browser lets the page go.
     */
    suspend() {
      away = true;
      for (const entry of entries.values()) {
        dropSocket(entry);
        notify(entry);
      }
    },

    /** The page is back: each source that is watched is asked for again. */
    resume() {
      if (!away) return;
      away = false;
      for (const entry of entries.values()) connect(entry);
    },

    /** How many sources are open and how many watchers there are (for a test). */
    size: () => ({sources: entries.size, sinks: [...entries.values()].reduce((sum, entry) => sum + entry.sinks.size, 0)}),
  };
}

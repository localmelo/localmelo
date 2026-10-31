import {saidText} from './i18n.mjs';

// The keys of the developer's segments (`status.*` in the catalogs): the cache, the step and the queue.
const DEVELOPER_KEYS = /^status\.(cache|step|tasks)/;

/**
 * A new user's row: the cache share, the step and phase and the queue counts
 * are for developers. A sentence such as "1 task waits for you" is not (the
 * server says it only for a task that holds on the person's approval), so it
 * stays. A segment is its words (`{key, params}`, as the server sends them) or,
 * from a server that sends only text, the terminal's English text.
 */
export function plainSegments(segments) {
  return segments.filter((segment) => (typeof segment === 'string'
    ? !segment.startsWith('cache') && !segment.startsWith('step ') && !segment.startsWith('tasks ')
    : !DEVELOPER_KEYS.test(segment.key)));
}

/**
 * The status row: the terminal's status-bar segments for one session, as
 * `GET /api/sessions/{id}/status` folds them on the server (cache share with
 * `--` when the server reports none, context, step, tasks), so the console
 * and the terminal show the same numbers. The server sends each segment as
 * words (`parts`: a catalog key and its parameters), which the row writes in
 * the page's language, and as the terminal's English text (`segments`), which
 * it draws when a server sends no parts. `api` takes paths relative to /api.
 * `bump()` asks for a refetch when a frame or a poll arrives; fetches are at
 * least `minGapMs` apart and never overlap, and a late answer for another
 * session is dropped. `schedule`/`cancel` and `clock` are injected for tests.
 * The segments are spans in one clipped line, so a narrow width drops the
 * rightmost first (see `.status-row` in workspace.css). `filter(segments)`
 * picks the ones to draw (the words, or the English texts): default mode
 * draws `plainSegments`.
 */
export function setupStatusRow({api, root, schedule = setTimeout, cancel = clearTimeout, clock = Date.now, minGapMs = 1000, filter = (segments) => segments}) {
  const doc = root.ownerDocument;
  let id = null;
  let epoch = 0;
  let lastFetch = -Infinity;
  let timer = null;
  let inflight = false;
  let dirty = false;

  function draw(segments) {
    root.replaceChildren(...segments.map((segment) => {
      const span = doc.createElement('span');
      span.className = 'status-seg';
      span.textContent = typeof segment === 'string' ? segment : saidText(segment) ?? '';
      return span;
    }));
    root.hidden = segments.length === 0;
  }

  async function run() {
    dirty = false;
    inflight = true;
    lastFetch = clock();
    const mine = epoch;
    let body = null;
    try {
      body = await api(`/sessions/${encodeURIComponent(id)}/status`);
    } catch {
      // Keep the last row: the next frame or poll asks again.
    }
    inflight = false;
    const rows = Array.isArray(body?.parts) ? body.parts : body?.segments;
    if (mine === epoch && Array.isArray(rows)) draw(filter(rows));
    pump();
  }

  function pump() {
    if (!id || !dirty || inflight || timer !== null) return;
    const wait = lastFetch + minGapMs - clock();
    if (wait > 0) {
      timer = schedule(() => { timer = null; pump(); }, wait);
      return;
    }
    run();
  }

  function bump() {
    dirty = true;
    pump();
  }

  /** Show the row of session `next`; the same session again is a bump. */
  function follow(next) {
    if (next !== id) {
      clear();
      id = next;
      root.hidden = true;
    }
    bump();
  }

  function clear() {
    epoch++;
    id = null;
    dirty = false;
    lastFetch = -Infinity;
    if (timer !== null) cancel(timer);
    timer = null;
    root.replaceChildren();
    root.hidden = true;
  }

  return {follow, bump, clear, element: root, current: () => id};
}

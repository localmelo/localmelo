// The scheduler's counts, e.g. "tasks 0 running 0 queued 2 held"; the held ones wait for the user.
const QUEUE = /^tasks \d+ running \d+ queued(?: (\d+) held)?$/;

/**
 * A new user's row: the cache share, the step and phase and the queue counts
 * are for developers. What waits for the user is their business, so a held
 * task stays in the row — in plain words ("2 tasks wait for you"), never as
 * the scheduler's counts.
 */
export function plainSegments(segments) {
  const plain = [];
  for (const text of segments) {
    if (text.startsWith('cache') || text.startsWith('step ')) continue;
    const queue = QUEUE.exec(text);
    if (queue) {
      const held = Number(queue[1] ?? 0);
      if (held > 0) plain.push(held === 1 ? '1 task waits for you' : `${held} tasks wait for you`);
      continue;
    }
    plain.push(text);
  }
  return plain;
}

/**
 * The status row: the terminal's status-bar segments for one session, as
 * `GET /api/sessions/{id}/status` folds them on the server (cache share with
 * `--` when the server reports none, context, step, tasks), so the console
 * and the terminal show the same numbers. `api` takes paths relative to /api.
 * `bump()` asks for a refetch when a frame or a poll arrives; fetches are at
 * least `minGapMs` apart and never overlap, and a late answer for another
 * session is dropped. `schedule`/`cancel` and `clock` are injected for tests.
 * The segments are spans in one clipped line, so a narrow width drops the
 * rightmost first (see `.status-row` in workspace.css). `filter(segments)`
 * picks the ones to draw: default mode draws `plainSegments`.
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
    root.replaceChildren(...segments.map((text) => {
      const span = doc.createElement('span');
      span.className = 'status-seg';
      span.textContent = text;
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
    if (mine === epoch && Array.isArray(body?.segments)) draw(filter(body.segments));
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

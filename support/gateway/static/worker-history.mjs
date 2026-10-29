import {applyEvent, newTurn, newView} from './turns.mjs';

const SUPPORTED_EVENTS = new Set([
  'turn.started', 'session.title', 'session.updated', 'run.started', 'step.started',
  'assistant.message', 'tool.started', 'approval.requested', 'approval.resolved',
  'tool.finished', 'notice', 'reflection.started', 'reflection', 'run.progress',
  'run.finished', 'turn.finished', 'turn.undone', 'context_budget', 'note',
  'inbox.accepted', 'memory', 'recall',
]);

/** Fold stored events once into a separate, non-streaming worker view. */
export function foldWorkerTurns(id, rows) {
  const view = newView(id);
  view.turns = rows.map((row) => newTurn(row));
  const frames = rows.flatMap((row) => row.events ?? []).slice().sort((a, b) => a.id - b.id);
  for (const frame of frames) applyEvent(view, frame);
  return view;
}

const POLL_MS = 3000;
const PAGE = 50;
// New frames are drawn together, at most this often; a failed stream is opened again after RETRY_MS.
const DRAW_MS = 150;
const RETRY_MS = 2000;

function defaultEvery(fn, ms) {
  const timer = setInterval(fn, ms);
  timer?.unref?.();
  return () => clearInterval(timer);
}

function defaultOpen(url) {
  return typeof EventSource === 'function' ? new EventSource(url) : null;
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// A run is still being written while its status says so and its lease is live.
const live = (session) => session?.status === 'running' && session?.busy !== false;

/**
 * Worker requests have their own epoch and never modify the active chat. While
 * the list is open it is read again every POLL_MS and drawn only when it
 * changed. An unfinished run is read once, then followed on the events route:
 * `open(url)` makes the EventSource (null where there is none, and the run is
 * then read again every POLL_MS), frames are folded into the open view and
 * drawn together, a hidden tab closes the stream and a shown one resumes after
 * the last id, and the end of the stream reads the run's record once more.
 * `every(fn, ms)` schedules the poll and returns a function that stops it;
 * `schedule`/`cancel` time the draws and retries; `statusRow` is an optional
 * status-row controller (status-row.mjs) that follows the open run.
 */
export function workerHistory(api, changed, every = defaultEvery, {
  open = defaultOpen, schedule = setTimeout, cancel = clearTimeout, statusRow = null,
  doc = globalThis.document,
} = {}) {
  const state = {
    mode: 'list', filters: {status: '', source: '', parent: ''}, rows: [],
    nextCursor: null, selectedId: null, view: null, session: null,
    loading: false, error: '', epoch: 0, loaded: false, eventCounts: {}, unsupported: false,
  };
  const notify = () => changed(state);
  let stopPolling = null;
  const follow = {id: null, source: null, retry: null, draw: null};
  const fail = (error, detail) => {
    if (error.status === 503) return 'Worker history is unavailable.';
    if (detail && error.status === 404) return 'Worker run not found.';
    if (!detail && ([404, 405].includes(error.status) || (error.status === 400 && /kind/i.test(error.message || '')))) return 'This server does not support worker history yet.';
    return detail ? 'Could not load this worker run. Try again.' : 'Could not load workers. Try again.';
  };
  function listPath(cursor) {
    const params = new URLSearchParams({kind: 'worker', limit: String(PAGE)});
    for (const [key, value] of Object.entries(state.filters)) {
      if (value) params.set(key === 'parent' ? 'parent_run' : key, value);
    }
    if (cursor) params.set('cursor', cursor);
    return `/sessions?${params}`;
  }
  async function list({more = false} = {}) {
    stopPolling ??= every(poll, POLL_MS);
    stopFollow();
    statusRow?.clear();
    const epoch = ++state.epoch;
    state.mode = 'list'; state.selectedId = null; state.view = null; state.session = null; state.loading = true; state.error = '';
    if (!more) { state.rows = []; state.nextCursor = null; state.loaded = false; }
    notify();
    try {
      const page = await api(listPath(more ? state.nextCursor : null));
      if (epoch !== state.epoch) return;
      const seen = new Set(state.rows.map((row) => row.id));
      for (const row of page.sessions ?? []) {
        if (!seen.has(row.id)) { state.rows.push(row); seen.add(row.id); }
      }
      state.nextCursor = page.next_cursor ?? null;
      state.loaded = true;
    } catch (error) {
      if (epoch !== state.epoch) return;
      state.error = fail(error, false);
    } finally {
      if (epoch === state.epoch) { state.loading = false; notify(); }
    }
  }
  // The run and all its turns, with their events.
  async function readRun(id) {
    const path = `/sessions/${encodeURIComponent(id)}`;
    const [detail, first] = await Promise.all([api(path), api(`${path}/turns?limit=50&events=true`)]);
    let page = first;
    let rows = page.turns ?? [];
    while (page.has_earlier && rows.length) {
      const before = rows[0].turn;
      page = await api(`${path}/turns?limit=50&events=true&before=${encodeURIComponent(before)}`);
      const older = page.turns ?? [];
      if (!older.length || older[0].turn >= before) break;
      rows = [...older, ...rows];
    }
    return {session: detail.session, rows};
  }
  function showRun(id, {session, rows}) {
    state.session = session;
    state.eventCounts = Object.fromEntries(rows.map((row) => [row.turn, (row.events ?? []).length]));
    state.unsupported = rows.some((row) => (row.events ?? []).some((event) => !SUPPORTED_EVENTS.has(event?.type)));
    state.view = foldWorkerTurns(id, rows);
    state.raw = {session, rows};
  }
  async function detail(id) {
    stopPolling ??= every(poll, POLL_MS);
    stopFollow();
    statusRow?.follow(id);
    const epoch = ++state.epoch;
    state.mode = 'detail'; state.selectedId = id; state.view = null; state.session = null;
    state.loading = true; state.error = ''; notify();
    try {
      const run = await readRun(id);
      if (epoch !== state.epoch) return;
      if (run.session?.kind !== 'worker') { state.error = 'Worker run not found.'; return; }
      showRun(id, run);
      startFollow(id);
    } catch (error) {
      if (epoch !== state.epoch) return;
      state.error = fail(error, true);
    } finally {
      if (epoch === state.epoch) { state.loading = false; notify(); }
    }
  }
  // Follow an unfinished run on the events route, from the last id folded.
  function startFollow(id) {
    if (!live(state.session)) return;
    follow.id = id;
    doc?.addEventListener?.('visibilitychange', onVisibility);
    if (doc?.visibilityState !== 'hidden') connect();
  }
  function connect() {
    const id = follow.id;
    if (id === null || follow.source !== null) return;
    const source = open(`/api/sessions/${encodeURIComponent(id)}/events?after=${state.view.lastEventId}`);
    if (!source) { stopFollow(); return; }
    follow.source = source;
    source.onmessage = (message) => frame(source, message);
    source.onerror = () => end(source);
  }
  function frame(source, message) {
    if (follow.source !== source) return;
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    if (!event || typeof event !== 'object' || event.id <= state.view.lastEventId) return;
    applyEvent(state.view, event);
    state.eventCounts[event.turn] = (state.eventCounts[event.turn] ?? 0) + 1;
    if (!SUPPORTED_EVENTS.has(event.type)) state.unsupported = true;
    statusRow?.bump();
    follow.draw ??= schedule(() => { follow.draw = null; notify(); }, DRAW_MS);
  }
  // The server ends the stream when the run is no longer live: read the
  // record once more, then stop, or open the stream again if it still runs.
  async function end(source) {
    if (follow.source !== source) return;
    source.close(); follow.source = null;
    const {id} = follow, epoch = state.epoch;
    let gone = false;
    try {
      const {session} = await api(`/sessions/${encodeURIComponent(id)}`);
      if (epoch === state.epoch && session?.kind === 'worker') state.session = session;
    } catch (error) { gone = error?.status === 404; }
    if (epoch !== state.epoch || follow.id !== id) return;
    if (!gone && live(state.session)) follow.retry = schedule(() => { follow.retry = null; connect(); }, RETRY_MS);
    else stopFollow();
    statusRow?.bump();
    notify();
  }
  function onVisibility() {
    if (follow.id === null) return;
    if (doc.visibilityState === 'hidden') { closeStream(); return; }
    connect();
  }
  function closeStream() {
    follow.source?.close(); follow.source = null;
    if (follow.retry !== null) cancel(follow.retry);
    follow.retry = null;
  }
  function stopFollow() {
    closeStream();
    if (follow.draw !== null) cancel(follow.draw);
    follow.draw = null; follow.id = null;
    doc?.removeEventListener?.('visibilitychange', onVisibility);
  }
  // Read the open view again and draw only a change; an error or a late answer
  // for another view is dropped, and the next tick tries again.
  async function poll() {
    const doc = globalThis.document;
    if (state.loading || state.error || doc?.visibilityState === 'hidden') return;
    // A filter being typed is not redrawn away.
    if (state.mode === 'list' && doc?.activeElement?.closest?.('.worker-filters')) return;
    const epoch = state.epoch;
    try {
      if (state.mode === 'list') {
        if (!state.loaded) return;
        const page = await api(listPath(null));
        if (epoch !== state.epoch || state.loading) return;
        const fresh = page.sessions ?? [];
        // Rows past the first page came from "Load more" and stay as they are.
        const ids = new Set(fresh.map((row) => row.id));
        const later = state.rows.slice(PAGE).filter((row) => !ids.has(row.id));
        const rows = [...fresh, ...later];
        const cursor = state.rows.length > PAGE ? state.nextCursor : (page.next_cursor ?? null);
        if (same(rows, state.rows) && cursor === state.nextCursor) return;
        state.rows = rows; state.nextCursor = cursor;
      } else if (state.mode === 'detail' && follow.id !== null) {
        // Frames arrive on the stream; the row's idle time moves on its own.
        statusRow?.bump();
        return;
      } else if (state.mode === 'detail' && live(state.session)) {
        const run = await readRun(state.selectedId);
        if (epoch !== state.epoch || state.loading || run.session?.kind !== 'worker') return;
        if (same(run, state.raw)) return;
        showRun(state.selectedId, run);
      } else return;
    } catch {
      return;
    }
    notify();
  }
  function leave() {
    state.epoch++; state.loading = false;
    stopPolling?.(); stopPolling = null;
    stopFollow();
    statusRow?.clear();
  }
  return {state, list, detail, leave};
}

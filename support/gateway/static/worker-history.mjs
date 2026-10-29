import {applyEvent, newTurn, newView} from './turns.mjs';

const SUPPORTED_EVENTS = new Set([
  'turn.started', 'session.title', 'session.updated', 'run.started', 'step.started',
  'assistant.message', 'tool.started', 'approval.requested', 'approval.resolved',
  'tool.finished', 'notice', 'reflection.started', 'reflection', 'run.progress',
  'run.finished', 'turn.finished', 'turn.undone',
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

function defaultEvery(fn, ms) {
  const timer = setInterval(fn, ms);
  timer?.unref?.();
  return () => clearInterval(timer);
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Worker requests have their own epoch and never modify the active chat. While
 * the list is open, or a run that has not finished, it is read again every
 * POLL_MS and drawn only when it changed. `every(fn, ms)` schedules that and
 * returns a function that stops it.
 */
export function workerHistory(api, changed, every = defaultEvery) {
  const state = {
    mode: 'list', filters: {status: '', source: '', parent: ''}, rows: [],
    nextCursor: null, selectedId: null, view: null, session: null,
    loading: false, error: '', epoch: 0, loaded: false, eventCounts: {}, unsupported: false,
  };
  const notify = () => changed(state);
  let stopPolling = null;
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
    const epoch = ++state.epoch;
    state.mode = 'detail'; state.selectedId = id; state.view = null; state.session = null;
    state.loading = true; state.error = ''; notify();
    try {
      const run = await readRun(id);
      if (epoch !== state.epoch) return;
      if (run.session?.kind !== 'worker') { state.error = 'Worker run not found.'; return; }
      showRun(id, run);
    } catch (error) {
      if (epoch !== state.epoch) return;
      state.error = fail(error, true);
    } finally {
      if (epoch === state.epoch) { state.loading = false; notify(); }
    }
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
      } else if (state.mode === 'detail' && state.session?.status === 'running') {
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
  }
  return {state, list, detail, leave};
}

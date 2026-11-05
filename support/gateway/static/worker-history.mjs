import {t} from './i18n.mjs';
import {applyEvent, followEvents, newTurn, newView} from './turns.mjs';

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

/** All turns of a session with their events, oldest first, read a page at a time. */
export async function readTurns(api, id) {
  const path = `/sessions/${encodeURIComponent(id)}/turns?limit=50&events=true`;
  let page = await api(path);
  let rows = page.turns ?? [];
  while (page.has_earlier && rows.length) {
    const before = rows[0].turn;
    page = await api(`${path}&before=${encodeURIComponent(before)}`);
    const older = page.turns ?? [];
    if (!older.length || older[0].turn >= before) break;
    rows = [...older, ...rows];
  }
  return rows;
}

const POLL_MS = 3000;
const PAGE = 50;

function defaultEvery(fn, ms) {
  const timer = setInterval(fn, ms);
  timer?.unref?.();
  return () => clearInterval(timer);
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// A run is still being written while its status says so and its lease is live.
const live = (session) => session?.status === 'running' && session?.busy !== false;
// The two kinds this view reads: one-shot runs and chats of the terminal.
const readable = (session) => session?.kind === 'worker' || session?.kind === 'terminal';

/**
 * Worker requests have their own epoch and never modify the active chat. While
 * the list is open it is read again every POLL_MS and drawn only when it
 * changed. An unfinished run is read once, then followed by `followEvents`
 * (turns.mjs): `open(url)` makes the EventSource (null where there is none, and
 * the run is then read again every POLL_MS), frames are folded into the open
 * view and drawn together, and the end of a stream reads the run's record once
 * more, to follow on or to stop. `every(fn, ms)` schedules the poll and returns
 * a function that stops it; `schedule`/`cancel` time the draws and retries;
 * `statusRow` is an optional status-row controller (status-row.mjs) that
 * follows the open run.
 */
export function workerHistory(api, changed, every = defaultEvery, {open, schedule, cancel, statusRow = null} = {}) {
  const state = {
    mode: 'list', kind: 'worker', filters: {status: '', source: '', parent: ''}, rows: [],
    nextCursor: null, selectedId: null, view: null, session: null,
    loading: false, error: '', epoch: 0, loaded: false, eventCounts: {}, unsupported: false,
  };
  const notify = () => changed(state);
  let stopPolling = null;
  let follower = null;
  const fail = (error, detail) => {
    if (error.status === 503) return t('workers.unavailable');
    if (detail && error.status === 404) return t('workers.notFound');
    if (!detail && ([404, 405].includes(error.status) || (error.status === 400 && /kind/i.test(error.message || '')))) {
      return t(state.kind === 'terminal' ? 'workers.noTerminal' : 'workers.noHistory');
    }
    return t(detail ? 'workers.runFailed' : 'workers.listFailed');
  };
  function listPath(cursor) {
    const params = new URLSearchParams({kind: state.kind, limit: String(PAGE)});
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
    const [detail, rows] = await Promise.all([api(`/sessions/${encodeURIComponent(id)}`), readTurns(api, id)]);
    return {session: detail.session, rows};
  }
  function showRun(id, {session, rows}) {
    state.session = session;
    state.kind = session?.kind === 'terminal' ? 'terminal' : 'worker';
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
      if (!readable(run.session)) { state.error = t('workers.notFound'); return; }
      showRun(id, run);
      startFollow(id);
    } catch (error) {
      if (epoch !== state.epoch) return;
      state.error = fail(error, true);
    } finally {
      if (epoch === state.epoch) { state.loading = false; notify(); }
    }
  }
  // Follow an unfinished run on the events route, from the last id folded;
  // null where there is no EventSource, and the poll reads the run instead.
  function startFollow(id) {
    if (!live(state.session)) return;
    follower = followEvents({
      id, view: state.view, open, schedule, cancel,
      onFrame: (event) => {
        state.eventCounts[event.turn] = (state.eventCounts[event.turn] ?? 0) + 1;
        if (!SUPPORTED_EVENTS.has(event.type)) state.unsupported = true;
        statusRow?.bump();
      },
      onDraw: notify,
      onEnd: () => ended(id),
    });
  }
  // The server ends the stream when the run is no longer live: read the
  // record once more, then stop, or follow on if it still runs.
  async function ended(id) {
    const epoch = state.epoch;
    let gone = false;
    try {
      const {session} = await api(`/sessions/${encodeURIComponent(id)}`);
      if (epoch === state.epoch && readable(session)) state.session = session;
    } catch (error) { gone = error?.status === 404; }
    if (epoch !== state.epoch) return false;
    const more = !gone && live(state.session);
    if (!more) stopFollow();
    statusRow?.bump();
    notify();
    return more;
  }
  function stopFollow() {
    follower?.close();
    follower = null;
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
      } else if (state.mode === 'detail' && follower !== null) {
        // Frames arrive on the stream; the row's idle time moves on its own.
        statusRow?.bump();
        return;
      } else if (state.mode === 'detail' && (live(state.session) || state.session?.kind === 'terminal')) {
        // An idle chat of the terminal can start another turn: its record is
        // read first, and the whole chat only once it has changed.
        const wasLive = live(state.session);
        if (!wasLive) {
          const {session} = await api(`/sessions/${encodeURIComponent(state.selectedId)}`);
          if (epoch !== state.epoch || state.loading || !readable(session)) return;
          if (!live(session) && session.updated_at === state.session.updated_at) return;
        }
        const run = await readRun(state.selectedId);
        if (epoch !== state.epoch || state.loading || !readable(run.session)) return;
        if (same(run, state.raw)) return;
        showRun(state.selectedId, run);
        if (!wasLive) startFollow(state.selectedId);
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

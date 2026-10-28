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

/** Worker requests have their own epoch and never modify the active chat. */
export function workerHistory(api, changed) {
  const state = {
    mode: 'list', filters: {status: '', source: '', parent: ''}, rows: [],
    nextCursor: null, selectedId: null, view: null, session: null,
    loading: false, error: '', epoch: 0, loaded: false, eventCounts: {}, unsupported: false,
  };
  const notify = () => changed(state);
  const fail = (error, detail) => {
    if (error.status === 503) return 'Worker history is unavailable.';
    if (detail && error.status === 404) return 'Worker run not found.';
    if (!detail && ([404, 405].includes(error.status) || (error.status === 400 && /kind/i.test(error.message || '')))) return 'This server does not support worker history yet.';
    return detail ? 'Could not load this worker run. Try again.' : 'Could not load workers. Try again.';
  };
  async function list({more = false} = {}) {
    const epoch = ++state.epoch;
    state.mode = 'list'; state.selectedId = null; state.view = null; state.session = null; state.loading = true; state.error = '';
    if (!more) { state.rows = []; state.nextCursor = null; state.loaded = false; }
    const params = new URLSearchParams({kind: 'worker', limit: '50'});
    for (const [key, value] of Object.entries(state.filters)) {
      if (value) params.set(key === 'parent' ? 'parent_run' : key, value);
    }
    if (more && state.nextCursor) params.set('cursor', state.nextCursor);
    notify();
    try {
      const page = await api(`/sessions?${params}`);
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
  async function detail(id) {
    const epoch = ++state.epoch;
    state.mode = 'detail'; state.selectedId = id; state.view = null; state.session = null;
    state.loading = true; state.error = ''; notify();
    try {
      const path = `/sessions/${encodeURIComponent(id)}`;
      const [detail, first] = await Promise.all([api(path), api(`${path}/turns?limit=50&events=true`)]);
      if (epoch !== state.epoch) return;
      if (detail.session?.kind !== 'worker') { state.error = 'Worker run not found.'; return; }
      let page = first;
      let rows = page.turns ?? [];
      while (page.has_earlier && rows.length) {
        const before = rows[0].turn;
        page = await api(`${path}/turns?limit=50&events=true&before=${encodeURIComponent(before)}`);
        if (epoch !== state.epoch) return;
        const older = page.turns ?? [];
        if (!older.length || older[0].turn >= before) break;
        rows = [...older, ...rows];
      }
      state.session = detail.session;
      state.eventCounts = Object.fromEntries(rows.map((row) => [row.turn, (row.events ?? []).length]));
      state.unsupported = rows.some((row) => (row.events ?? []).some((event) => !SUPPORTED_EVENTS.has(event?.type)));
      state.view = foldWorkerTurns(id, rows);
    } catch (error) {
      if (epoch !== state.epoch) return;
      state.error = fail(error, true);
    } finally {
      if (epoch === state.epoch) { state.loading = false; notify(); }
    }
  }
  function leave() { state.epoch++; state.loading = false; }
  return {state, list, detail, leave};
}

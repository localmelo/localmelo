import {applyEvent} from './turns.mjs';
import {foldWorkerTurns, readTurns} from './worker-history.mjs';

// New frames are drawn together, at most this often.
const DRAW_MS = 150;
// A task's process writes events only in these states; any other state ends the stream.
const STREAMING = new Set(['running', 'held']);

function defaultOpen(url) {
  return typeof EventSource === 'function' ? new EventSource(url) : null;
}

/**
 * The live part of a task card: the task's session, folded like a chat. The
 * card calls `sync(task)` with every read of the task. A started task is read
 * once (`/sessions/{id}/turns?events=true`); while it runs or waits for an
 * approval its events are followed on `/events?after=<last id>`, drawn together
 * via `changed()`. A hidden tab closes the stream and a shown one resumes after
 * the last id; a stream that failed is opened again by the next `sync`. When the
 * task stops (done, failed, cancelled, interrupted, parked) the stream closes
 * and the session is read once more, so no frame is lost. A task that never
 * started shows nothing. `open(url)` makes the EventSource (null where there is
 * none, and a running task is then read again on each `sync`); `statusRow` is
 * told about every frame.
 */
export function taskLive({api, changed, open = defaultOpen, schedule = setTimeout, cancel = clearTimeout, statusRow = null, doc = globalThis.document}) {
  const state = {id: null, view: null, loading: false, error: ''};
  const follow = {on: false, source: null, draw: null};
  let wants = false;
  let reads = 0;

  async function load() {
    const id = state.id, mine = ++reads;
    state.loading = true;
    try {
      const view = foldWorkerTurns(id, await readTurns(api, id));
      if (mine !== reads) return;
      Object.assign(state, {view, error: ''});
    } catch (error) {
      if (mine !== reads) return;
      // A task that never wrote a session has nothing to show.
      state.error = error?.status === 404 ? '' : 'Could not load the live view.';
    }
    state.loading = false;
    if (wants && state.view) { startFollow(); connect(); }
    changed();
  }
  // True when a stream is open or is waiting for the tab; false where there is no EventSource.
  function connect() {
    if (follow.source || !wants || doc?.visibilityState === 'hidden') return true;
    const source = open(`/api/sessions/${encodeURIComponent(state.id)}/events?after=${state.view.lastEventId}`);
    if (!source) return false;
    follow.source = source;
    source.onmessage = (message) => frame(source, message);
    source.onerror = () => { if (follow.source === source) closeStream(); };
    return true;
  }
  function frame(source, message) {
    if (follow.source !== source) return;
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    if (!event || typeof event !== 'object' || event.id <= state.view.lastEventId) return;
    applyEvent(state.view, event);
    statusRow?.bump();
    follow.draw ??= schedule(() => { follow.draw = null; changed(); }, DRAW_MS);
  }
  function onVisibility() {
    if (doc.visibilityState === 'hidden') closeStream();
    else connect();
  }
  function startFollow() {
    if (follow.on) return;
    follow.on = true;
    doc?.addEventListener?.('visibilitychange', onVisibility);
  }
  function closeStream() {
    follow.source?.close();
    follow.source = null;
  }
  function stopFollow() {
    closeStream();
    if (follow.draw !== null) cancel(follow.draw);
    follow.draw = null; follow.on = false;
    doc?.removeEventListener?.('visibilitychange', onVisibility);
  }
  function reset(id) {
    reads++;
    stopFollow();
    wants = false;
    Object.assign(state, {id, view: null, loading: false, error: ''});
  }

  function sync(task) {
    if (state.id !== task.task_id) reset(task.task_id);
    if (task.started_at == null) return;
    const was = wants;
    wants = STREAMING.has(task.state);
    if (!state.view) {
      if (!state.loading) load();
    } else if (wants) {
      startFollow();
      if (!connect()) load();
    } else if (was || follow.on) {
      stopFollow();
      load();
    }
  }
  function leave() { reset(null); }

  return {state, sync, leave};
}

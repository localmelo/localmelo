import {t} from './i18n.mjs';
import {followEvents} from './turns.mjs';
import {foldWorkerTurns, readTurns} from './worker-history.mjs';

// A task's process writes events only in these states; any other state ends the stream.
const STREAMING = new Set(['running', 'held']);

/**
 * The live part of a task card: the task's session, folded like a chat. The
 * card calls `sync(task)` with every read of the task. A started task is read
 * once (`/sessions/{id}/turns?events=true`); while it runs or waits for an
 * approval its events are followed by `followEvents` (turns.mjs), drawn
 * together via `changed()`; that follower keeps going in a hidden tab and opens
 * a failed stream again. When the task stops (done, failed, cancelled,
 * interrupted, parked) the stream closes and the session is read once more, so
 * no frame is lost. A task that never started shows nothing. `open(url)` makes
 * the EventSource (null where there is none, and a running task is then read
 * again on each `sync`); `statusRow` is told about every frame.
 */
export function taskLive({api, changed, open, schedule, cancel, statusRow = null}) {
  const state = {id: null, view: null, loading: false, error: ''};
  let follower = null;
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
      state.error = error?.status === 404 ? '' : t('taskLive.loadFailed');
    }
    state.loading = false;
    if (wants && state.view) connect();
    changed();
  }
  // True when the task is followed; false where there is no EventSource.
  function connect() {
    // `watch`: the person sees the task here, so its card may ask (#465).
    follower ??= followEvents({
      id: state.id, view: state.view, watch: true, open, schedule, cancel,
      onFrame: () => statusRow?.bump(), onDraw: changed,
    });
    return follower !== null;
  }
  function stopFollow() {
    follower?.close();
    follower = null;
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
      if (!connect()) load();
    } else if (was || follower !== null) {
      stopFollow();
      load();
    }
  }
  function leave() { reset(null); }

  return {state, sync, leave};
}

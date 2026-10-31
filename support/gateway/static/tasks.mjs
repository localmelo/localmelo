import {taskLive} from './task-live.mjs';

const RESUMABLE = new Set(['parked', 'interrupted']);
const FINISHED = new Set(['done', 'failed', 'cancelled', 'expired']);
// "Live" is for a task whose process is writing now; the work of any other task is its steps.
const RUNNING = new Set(['running', 'held']);
const MISSING = 'Resume unavailable: recovery information is missing.';
const WARNING = 'A process from before the restart may still be running. Resuming may repeat effects. Confirmation does not prove the process stopped or effects happened only once.';

/**
 * Mount the read-and-control view of scheduler tasks. `api` takes paths
 * relative to /api and rejects with an Error carrying status, code and
 * details. `approvalCard(request, view)` draws the console's approval card
 * (view: rerender, answer, reload, workspace). `id` is a fresh request or
 * operation id. Every response is checked against an epoch, so a late one
 * never draws over a newer view or a task the user left. While the view is
 * open it is read again every POLL_MS and drawn only when it changed, so a
 * task that starts waiting for an approval shows its card. `every(fn, ms)`
 * schedules that and returns a function that stops it. `statusRow` is an
 * optional status-row controller (status-row.mjs): the card shows it under
 * its facts, and each poll bumps it. `live` (optional) adds the Live section
 * (task-live.mjs): `live.nodes(view, redraw)` draws the folded turns of the
 * task's session, and `open`, `schedule` and `cancel` are passed on to follow
 * its events. The section is redrawn on its own, never the whole card, so a
 * reason being typed in an approval card keeps its focus. A task that ended
 * opens with its answer: `live.result(view, redraw)` gives `{answer, steps}`,
 * the answer (and, if it did not complete, why) above the facts and the work
 * folded under "Steps"; "Live" heads the work only while the task runs.
 * `developer()` says whether developer mode is on: the scheduler's priority
 * (P2) and its words are shown only then.
 * `chooseFolder()` (optional) asks the macOS app's folder panel for a path and
 * resolves it, or null; without it, Choose a folder opens a path field. A task
 * that could not use its folder offers it, and starts again in the folder chosen.
 */
const POLL_MS = 3000;

function defaultEvery(fn, ms) {
  const timer = setInterval(fn, ms);
  timer?.unref?.();
  return () => clearInterval(timer);
}

export function setupTasks({api, root, newId, navigate, approvalCard, every = defaultEvery, statusRow = null, live: liveOptions = null, developer = () => true, chooseFolder = null}) {
  const doc = root.ownerDocument;
  const state = {mode: 'list', epoch: 0, tasks: [], task: null, error: '', notice: '',
    busy: '', evidence: '', dialog: null, pending: null, again: null, stepsOpen: false, folderField: false, folderText: ''};
  let active = false;
  let stopPolling = null;
  const liveBox = doc.createElement('section');
  liveBox.className = 'task-live';
  const answerBox = doc.createElement('section');
  answerBox.className = 'task-answer';
  const {nodes: liveNodes, result: liveResult, ...follow} = liveOptions ?? {};
  const live = liveOptions ? taskLive({api, changed: drawLive, statusRow, doc, ...follow}) : null;

  function node(tag, text, className) {
    const element = doc.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }
  function button(text, action, disabled = false) {
    const element = node('button', text);
    element.type = 'button';
    element.disabled = disabled;
    element.addEventListener('click', action);
    return element;
  }
  const path = (id) => `/tasks/${encodeURIComponent(id)}`;
  const when = (value) => {
    const date = value == null ? null : new Date(value * 1000);
    return date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : '—';
  };
  const errorText = (error) => error?.message || 'Request failed. Please try again.';
  const hasCard = (task) => typeof task.card === 'string' && task.card.trim() !== '';
  // A started task without a card cannot resume; the server refuses it too.
  const cardMissing = (task) => state.evidence === 'missing_card' || (task.started_at != null && !hasCard(task));

  const pendingOf = (task) => (Array.isArray(task.pending_approvals) ? task.pending_approvals : []);
  const stateText = (task) => (pendingOf(task).length ? 'Waiting for your approval' : task.state);

  function evidenceText(task) {
    if (state.evidence === 'unverified_processes') return 'Earlier processes could not be verified as stopped; confirmation is required to resume.';
    if (state.evidence === 'missing_card') return 'Recovery information is missing.';
    if (state.evidence) return `Resume was refused: ${state.evidence.replaceAll('_', ' ')}.`;
    if (RESUMABLE.has(task.state)) return 'Checked when you press Resume.';
    return '';
  }

  function list() {
    const rows = [...state.tasks].sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
    const intro = developer() ? 'Scheduler tasks. Open one to see its recovery card and actions.' : 'Open a task to see what it did.';
    const nodes = [node('p', intro, 'muted')];
    nodes.push(button('Refresh', () => show('')));
    if (!rows.length && !state.error) nodes.push(node('p', 'No tasks yet.'));
    for (const task of rows) {
      const link = node('a', undefined, 'worker-row task-row');
      link.href = `#/tasks/${encodeURIComponent(task.task_id)}`;
      link.append(node('strong', task.title || task.task_id),
        node('span', [stateText(task), developer() ? `P${task.level_eff ?? task.level}` : '', `updated ${when(task.updated_at)}`, task.workspace_label ? `in ${task.workspace_label}` : ''].filter(Boolean).join(' · '), 'muted'));
      nodes.push(link);
    }
    return nodes;
  }

  function card() {
    const task = state.task;
    if (!task) return [];
    const back = node('a', 'Back to tasks');
    back.href = '#/tasks';
    const meta = node('dl', undefined, 'worker-meta');
    const row = (label, value) => meta.append(node('dt', label), node('dd', value));
    row('State', stateText(task));
    if (developer()) row('Priority', `P${task.level_eff ?? task.level}`);
    row('Updated', when(task.updated_at));
    // A refused folder says why in its own box below, in words.
    if (task.wait_reason && !task.folder_problem) row('Waiting', task.wait_reason);
    if (task.workspace_label) row('Workspace', task.workspace_label);
    // Evidence is shown when there is some to show, never as "not needed".
    const evidence = evidenceText(task);
    if (evidence) row('Evidence', evidence);
    const nodes = [back];
    if (live) { drawLive(); nodes.push(answerBox); }
    nodes.push(meta);
    if (task.folder_problem) nodes.push(folderBox(task));
    if (task.not_committed) nodes.push(node('p', task.not_committed, 'task-note'));
    if (statusRow) nodes.push(statusRow.element);
    const view = {rerender: render, answer: answerApproval, reload: () => refresh(), workspace: task.workspace_label, scope: 'task'};
    for (const request of pendingOf(task)) nodes.push(approvalCard(request, view));
    if (live) nodes.push(liveBox);
    nodes.push(node('h2', 'Recovery card'));
    nodes.push(hasCard(task) ? node('pre', task.card, 'worker-brief task-card-text') : node('p', 'No recovery card is saved.', 'muted'));
    const actions = node('div', undefined, 'task-actions');
    if (RESUMABLE.has(task.state) && !cardMissing(task)) {
      actions.append(button(state.busy === 'resume' ? 'Resuming…' : 'Resume', () => resume(), Boolean(state.busy)));
    }
    if (!FINISHED.has(task.state)) {
      actions.append(button(state.busy === 'stop' ? 'Stopping…' : 'Stop', () => stop(), state.busy === 'stop'));
    }
    nodes.push(actions);
    if (RESUMABLE.has(task.state) && cardMissing(task)) {
      const box = node('div', undefined, 'task-missing');
      box.setAttribute('role', 'alert');
      box.append(node('p', MISSING),
        node('p', 'Earlier effects of this task are unknown. A new task starts from the original objective alone and does not undo them.'),
        button(state.busy === 'again' ? 'Starting…' : 'Start again as a new task', () => startAgain(), Boolean(state.busy)));
      nodes.push(box);
    }
    return nodes;
  }

  // The task's session as the page shows it. A task that ended opens with its answer (above the
  // facts) and folds its work under Steps; any other task heads its work Live while it runs, else Steps.
  function drawLive() {
    const view = live.state.view;
    const task = state.task;
    const open = Boolean(view) && state.mode === 'card';
    let top = [];
    let shown = [];
    if (open && FINISHED.has(task?.state) && liveResult) {
      const {answer, steps} = liveResult(view, drawLive);
      top = answer;
      if (steps.length) shown.push(stepsFold(steps));
    } else if (open) {
      const turns = liveNodes(view, drawLive);
      if (turns.length) shown = [node('h2', RUNNING.has(task?.state) ? 'Live' : 'Steps'), ...turns];
    }
    if (live.state.error && state.mode === 'card') shown.push(node('p', live.state.error, 'muted'));
    answerBox.hidden = !top.length;
    answerBox.replaceChildren(...top);
    liveBox.hidden = !shown.length;
    liveBox.replaceChildren(...shown);
  }

  function stepsFold(steps) {
    const fold = node('details', undefined, 'task-steps');
    fold.open = state.stepsOpen;
    fold.addEventListener('toggle', () => { state.stepsOpen = fold.open; });
    fold.append(node('summary', 'Steps'), ...steps);
    return fold;
  }

  function dialog() {
    const open = state.dialog;
    const element = node('dialog', undefined, 'task-dialog');
    element.setAttribute('aria-labelledby', 'task-dialog-title');
    element.setAttribute('aria-describedby', 'task-dialog-text');
    const title = node('h2', 'Resume this task?');
    title.id = 'task-dialog-title';
    const text = node('p', open.warning || WARNING);
    text.id = 'task-dialog-text';
    const confirm = button(state.busy === 'resume' ? 'Resuming…' : 'Resume anyway', () => resume(open.token), Boolean(state.busy));
    const cancel = button('Cancel', () => closeDialog(), state.busy === 'resume');
    const actions = node('div', undefined, 'task-actions');
    actions.append(confirm, cancel);
    element.append(title, text, actions);
    element.addEventListener('cancel', (event) => { event.preventDefault?.(); if (state.busy !== 'resume') closeDialog(); });
    element.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault?.();
      if (state.busy !== 'resume') closeDialog();
    });
    return {element, cancel};
  }

  function render() {
    if (!active) return;
    const title = node('h1', state.mode === 'list' ? 'Tasks' : state.task?.title || 'Task');
    title.tabIndex = -1;
    const nodes = [title];
    for (const [text, role] of [[state.error, 'alert'], [state.notice, 'status']]) {
      if (!text) continue;
      const line = node('p', text, role === 'alert' ? 'task-error' : 'muted');
      line.setAttribute('role', role);
      nodes.push(line);
    }
    nodes.push(...(state.mode === 'list' ? list() : card()));
    let shown = null;
    if (state.dialog) {
      shown = dialog();
      nodes.push(shown.element);
    }
    const hadFocus = root.contains(doc.activeElement);
    root.replaceChildren(...nodes);
    if (shown) {
      if (shown.element.showModal) shown.element.showModal();
      else shown.element.hidden = false;
      shown.cancel.focus();
    } else if (hadFocus && !root.contains(doc.activeElement)) {
      title.focus();
    }
  }

  function closeDialog() {
    if (!state.dialog) return false;
    state.dialog = null;
    render();
    return true;
  }

  async function show(id) {
    active = true;
    stopPolling ??= every(poll, POLL_MS);
    const epoch = ++state.epoch;
    live?.leave();
    Object.assign(state, {mode: id ? 'card' : 'list', error: '', notice: '', busy: '', evidence: '',
      dialog: null, pending: null, again: null, task: null, stepsOpen: false});
    root.hidden = false;
    render();
    if (id) statusRow?.follow(id);
    else statusRow?.clear();
    try {
      const data = await api(id ? path(id) : '/tasks');
      if (epoch !== state.epoch) return;
      if (id) { state.task = data.task; live?.sync(data.task); }
      else state.tasks = data.tasks || [];
    } catch (error) {
      if (epoch !== state.epoch) return;
      state.error = errorText(error);
    }
    render();
  }

  // One request at a time per task; a late answer for an old view is dropped.
  async function act(kind, request) {
    if (state.busy) return null;
    const epoch = state.epoch;
    const id = state.task.task_id;
    state.busy = kind;
    state.error = '';
    render();
    try {
      return {data: await request(id)};
    } catch (error) {
      return {error};
    } finally {
      if (epoch === state.epoch && state.task?.task_id === id) state.busy = '';
    }
  }

  // Read the task again; a late answer for another view or task is dropped.
  async function refresh() {
    const epoch = state.epoch;
    const id = state.task?.task_id;
    if (!id) return;
    try {
      const data = await api(path(id));
      if (epoch === state.epoch && state.task?.task_id === id) { state.task = data.task; live?.sync(data.task); }
    } catch (error) {
      if (epoch === state.epoch) state.error = errorText(error);
    }
    render();
  }

  // Read the open view again; draw only what changed, so typing in a card is
  // not disturbed while the task waits. A dialog or an action in flight wins.
  async function poll() {
    if (!active || state.busy || state.dialog || doc.visibilityState === 'hidden') return;
    const id = state.mode === 'card' ? state.task?.task_id : null;
    if (state.mode === 'card' && (!id || FINISHED.has(state.task.state))) return;
    const epoch = state.epoch;
    let data;
    if (id) statusRow?.bump();
    try {
      data = await api(id ? path(id) : '/tasks');
    } catch {
      return;
    }
    if (epoch !== state.epoch || state.busy || state.dialog) return;
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    if (id) {
      if (state.task?.task_id !== id) return;
      live?.sync(data.task);
      if (same(data.task, state.task)) return;
      state.task = data.task;
    } else {
      const tasks = data.tasks || [];
      if (same(tasks, state.tasks)) return;
      state.tasks = tasks;
    }
    render();
  }

  // The answer goes to the approvals route, as in a chat; the task moves on
  // by itself, and its next state is read back.
  async function answerApproval(request, card, choice) {
    const id = String(request.request_id);
    const complete = request.arguments_complete === true && !(request.tool_name === 'file_write' && !request.target);
    if (card.submitting || (choice !== 'deny' && !complete)) return;
    card.submitting = true;
    render();
    const epoch = state.epoch;
    try {
      await api(`/approvals/${encodeURIComponent(id)}`,
        {body: {choice, ...(choice === 'deny' ? {reason: card.reason} : {})}});
    } catch (error) {
      card.submitting = false;
      if (epoch === state.epoch) state.error = `Could not answer approval: ${errorText(error)}`;
      render();
      return;
    }
    await refresh();
  }

  async function resume(token) {
    const epoch = state.epoch;
    if (!state.pending) state.pending = newId();
    const operation = state.pending;
    const outcome = await act('resume', (id) => api(`${path(id)}/resume`,
      {method: 'POST', body: {operation_id: operation, ...(token ? {confirmation_token: token} : {})}}));
    if (!outcome || epoch !== state.epoch) return;
    if (outcome.data) {
      Object.assign(state, {task: outcome.data.task, dialog: null, pending: null, evidence: '', notice: `Resume accepted. The task is ${outcome.data.task.state}.`});
      live?.sync(outcome.data.task);
    } else {
      const error = outcome.error;
      if (error.code === 'confirmation_required' && error.details?.confirmation_token) {
        // Not a refusal: the same operation id is retried with the token.
        Object.assign(state, {evidence: 'unverified_processes', dialog: {token: error.details.confirmation_token, warning: error.details.warning}});
      } else {
        // A definite answer ends the operation; a network failure keeps its id
        // so the retry cannot start a second turn.
        if (error.status) state.pending = null;
        state.dialog = null;
        state.evidence = error.details?.reason || '';
        state.error = errorText(error);
      }
    }
    render();
  }

  async function stop() {
    const epoch = state.epoch;
    const outcome = await act('stop', (id) => api(path(id), {method: 'PATCH', body: {state: 'stop'}}));
    if (!outcome || epoch !== state.epoch) return;
    if (outcome.data) {
      Object.assign(state, {task: outcome.data.task, dialog: null, pending: null, evidence: '', notice: `Stop accepted. The task is ${outcome.data.task.state}.`});
      live?.sync(outcome.data.task);
    } else state.error = errorText(outcome.error);
    render();
  }

  // A task refused its folder never ran: it says why and starts again in a folder the person chooses.
  function folderBox(task) {
    const box = node('div', undefined, 'task-missing task-folder');
    box.setAttribute('role', 'alert');
    const where = task.workspace_label || 'its folder';
    box.append(node('p', `This task did not start: melo cannot work in ${where}. ${String(task.folder_problem).replace(/\.?$/, '.')}`),
      node('p', 'It never runs in another folder instead. Choose a folder to start it there.', 'muted'));
    const busy = Boolean(state.busy);
    if (state.folderField) {
      const field = node('input', undefined, 'task-folder-path');
      field.type = 'text';
      field.value = state.folderText;
      field.placeholder = '~/projects/my-app';
      field.setAttribute('aria-label', 'Folder path');
      field.addEventListener('input', () => { state.folderText = field.value; });
      field.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); startIn(field.value); }
      });
      box.append(field, button(state.busy === 'folder' ? 'Starting…' : 'Start here', () => startIn(field.value), busy));
    } else {
      box.append(button('Choose a folder', () => pickFolder(), busy));
    }
    return box;
  }

  async function pickFolder() {
    const picked = chooseFolder ? await chooseFolder().catch(() => null) : undefined;
    if (typeof picked === 'string' && picked.startsWith('/')) return startIn(picked);
    if (picked === undefined) {
      state.folderField = true;
      render();
      root.querySelector('.task-folder-path')?.focus();
    }
  }

  async function startIn(path) {
    const text = String(path ?? '').trim();
    if (!/^(\/|~\/)\S/.test(text)) {
      state.error = 'Type a full path, such as ~/projects/my-app.';
      render();
      return;
    }
    const epoch = state.epoch;
    if (!state.again) state.again = newId();
    const task = state.task;
    const spec = task.spec || {};
    const body = {objective: spec.objective, title: spec.title, workspace: text, request_key: state.again};
    if (task.front_id) body.front_id = task.front_id;
    const outcome = await act('folder', () => api('/tasks', {method: 'POST', body}));
    if (!outcome || epoch !== state.epoch) return;
    if (outcome.data) {
      Object.assign(state, {again: null, folderField: false, folderText: ''});
      navigate(`#/tasks/${encodeURIComponent(outcome.data.task.task_id)}`);
      return;
    }
    if (outcome.error.status) state.again = null;
    state.error = errorText(outcome.error);
    render();
  }

  async function startAgain() {
    const epoch = state.epoch;
    if (!state.again) state.again = newId();
    const key = state.again;
    const spec = state.task.spec || {};
    const body = {objective: spec.objective, title: spec.title, request_key: key};
    if (spec.workspace) body.workspace = spec.workspace;
    const outcome = await act('again', () => api('/tasks', {method: 'POST', body}));
    if (!outcome || epoch !== state.epoch) return;
    if (outcome.data) {
      state.again = null;
      navigate(`#/tasks/${encodeURIComponent(outcome.data.task.task_id)}`);
      return;
    }
    if (outcome.error.status) state.again = null;
    state.error = errorText(outcome.error);
    render();
  }

  function leave() {
    active = false;
    stopPolling?.();
    stopPolling = null;
    state.epoch++;
    state.dialog = null;
    live?.leave();
    statusRow?.clear();
    root.hidden = true;
    root.replaceChildren();
  }

  return {show, leave, state, closeDialog, isActive: () => active};
}

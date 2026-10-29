const RESUMABLE = new Set(['parked', 'interrupted']);
const FINISHED = new Set(['done', 'failed', 'cancelled', 'expired']);
const MISSING = 'Resume unavailable: recovery information is missing.';
const WARNING = 'A process from before the restart may still be running. Resuming may repeat effects. Confirmation does not prove the process stopped or effects happened only once.';

/**
 * Mount the read-and-control view of scheduler tasks. `api` takes paths
 * relative to /api and rejects with an Error carrying status, code and
 * details. `approvalCard(request, view)` draws the console's approval card
 * (view: rerender, answer, reload, workspace). `id` is a fresh request or
 * operation id. Every response is checked against an epoch, so a late one
 * never draws over a newer view or a task the user left.
 */
export function setupTasks({api, root, newId, navigate, approvalCard}) {
  const doc = root.ownerDocument;
  const state = {mode: 'list', epoch: 0, tasks: [], task: null, error: '', notice: '',
    busy: '', evidence: '', dialog: null, pending: null, again: null};
  let active = false;

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
    return 'Not needed in this state.';
  }

  function list() {
    const rows = [...state.tasks].sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
    const nodes = [node('p', 'Scheduler tasks. Open one to see its recovery card and actions.', 'muted')];
    nodes.push(button('Refresh', () => show('')));
    if (!rows.length && !state.error) nodes.push(node('p', 'No tasks yet.'));
    for (const task of rows) {
      const link = node('a', undefined, 'worker-row task-row');
      link.href = `#/tasks/${encodeURIComponent(task.task_id)}`;
      link.append(node('strong', task.title || task.task_id),
        node('span', `${stateText(task)} · P${task.level_eff ?? task.level} · updated ${when(task.updated_at)}`, 'muted'));
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
    row('Priority', `P${task.level_eff ?? task.level}`);
    row('Updated', when(task.updated_at));
    if (task.wait_reason) row('Waiting', task.wait_reason);
    if (task.workspace_label) row('Workspace', task.workspace_label);
    row('Evidence', evidenceText(task));
    const nodes = [back, meta];
    const view = {rerender: render, answer: answerApproval, reload: () => refresh(), workspace: task.workspace_label};
    for (const request of pendingOf(task)) nodes.push(approvalCard(request, view));
    nodes.push(node('h2', 'Recovery card'));
    nodes.push(hasCard(task) ? node('pre', task.card, 'worker-brief task-card-text') : node('p', 'No recovery card is saved.', 'muted'));
    const actions = node('div', undefined, 'task-actions');
    if (RESUMABLE.has(task.state) && !cardMissing(task)) {
      actions.append(button(state.busy === 'resume' ? 'Resuming…' : 'Resume', () => resume(), Boolean(state.busy)));
    }
    actions.append(button('Refresh', () => refresh(), Boolean(state.busy)));
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
    const epoch = ++state.epoch;
    Object.assign(state, {mode: id ? 'card' : 'list', error: '', notice: '', busy: '', evidence: '',
      dialog: null, pending: null, again: null, task: null});
    root.hidden = false;
    render();
    try {
      const data = await api(id ? path(id) : '/tasks');
      if (epoch !== state.epoch) return;
      if (id) state.task = data.task;
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
      if (epoch === state.epoch && state.task?.task_id === id) state.task = data.task;
    } catch (error) {
      if (epoch === state.epoch) state.error = errorText(error);
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
    if (outcome.data) Object.assign(state, {task: outcome.data.task, dialog: null, pending: null, evidence: '', notice: `Stop accepted. The task is ${outcome.data.task.state}.`});
    else state.error = errorText(outcome.error);
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
    state.epoch++;
    state.dialog = null;
    root.hidden = true;
    root.replaceChildren();
  }

  return {show, leave, state, closeDialog, isActive: () => active};
}

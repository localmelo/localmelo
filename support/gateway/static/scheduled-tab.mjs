/**
 * The panel's Scheduled tab: the check-in and the chores that run on a schedule.
 *
 * `GET /api/chores` lists the chores (the main chat shows all of them, a side chat its own, by `chat()`) with the
 * check-in's state. The check-in's switch is `PUT /api/checkin`; a chore's switch and "Edit time" are `PATCH
 * /api/chores/{id}` with the revision that was read (a 409 reloads the list); "Run now" is `POST
 * /api/chores/{id}/run`; "Delete" asks first, then calls `DELETE`. `root` is mounted in the panel's body while the
 * tab is open; `show()` reads the list again and `reload()` does so quietly, unless an edit is open.
 */
export const EMPTY = "Nothing scheduled. Ask in the main chat: 'every weekday at 8:30, brief me on my calendar'.";
export const CHECKIN = 'Every 30 min while localmelo runs · uses the model only if something is due';
const KINDS = [['daily', 'Every day'], ['weekdays', 'Every weekday'], ['hours', 'Every N hours']];
// How a chore's last run reads, by the state of its task.
const RAN = {done: 'Done', failed: 'Did not finish', cancelled: 'Stopped', expired: 'Expired', queued: 'Starting', running: 'Running now',
  held: 'Waiting for the model', waiting: 'Waiting for you', parked: 'Paused', interrupted: 'Stopped by a restart'};

/** ``today at 8:30 AM``, ``tomorrow at 8:30 AM``, ``Friday at 8:30 AM``, or ``Nov 2 at 8:30 AM``. */
export function whenWords(at, now) {
  if (!Number.isFinite(at)) return '';
  const time = new Date(at * 1000);
  const day = (stamp) => { const d = new Date(stamp * 1000); d.setHours(0, 0, 0, 0); return d.getTime() / 1000; };
  const ahead = Math.round((day(at) - day(now)) / 86400);
  const clock = time.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  if (ahead <= 0) return `today at ${clock}`;
  if (ahead === 1) return `tomorrow at ${clock}`;
  const label = ahead < 7 ? time.toLocaleDateString([], {weekday: 'long'}) : time.toLocaleDateString([], {month: 'short', day: 'numeric'});
  return `${label} at ${clock}`;
}

/** The lines under a row's title: when it runs next, then how the last run went. */
export function rowLines(chore, now) {
  const lines = [chore.enabled ? `Next ${whenWords(chore.next_run, now)}` : 'Off'];
  if (chore.last) lines.push(`Last: ${[RAN[chore.last.state] ?? chore.last.state, chore.last.line].filter(Boolean).join(' · ')}`);
  return lines;
}

export function setupScheduledTab({doc, api, chat = () => '', developer = () => false, clock = Date.now}) {
  const state = {loaded: false, error: '', chores: [], checkin: null, scope: 'all', open: null, edit: null, ask: null, notice: '', busy: false};
  let reads = 0;
  function node(tag, className, text) {
    const element = doc.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }
  function button(className, text, action, label) {
    const element = node('button', className, text);
    element.type = 'button';
    if (label) element.setAttribute('aria-label', label);
    element.addEventListener('click', action);
    return element;
  }
  function toggle(label, on, action) {
    const element = node('button', 'sched-switch');
    element.type = 'button';
    element.setAttribute('role', 'switch');
    element.setAttribute('aria-checked', String(on));
    element.setAttribute('aria-label', label);
    element.append(node('span', 'sched-knob'));
    element.addEventListener('click', action);
    return element;
  }
  const root = node('div', 'sched-tab');
  const now = () => clock() / 1000;

  async function load({quiet = false} = {}) {
    const mine = ++reads;
    const asked = chat();
    try {
      const page = await api(`/chores${asked ? `?session_id=${encodeURIComponent(asked)}` : ''}`);
      if (mine !== reads) return;
      Object.assign(state, {chores: page.chores ?? [], checkin: page.checkin ?? null, scope: page.scope === 'chat' ? 'chat' : 'all', loaded: true, error: ''});
      if (state.open && !state.chores.some((item) => item.id === state.open)) Object.assign(state, {open: null, edit: null, ask: null});
    } catch (error) {
      if (mine !== reads || quiet) return;
      state.error = error?.message || 'Could not read the schedule.';
    }
    draw();
  }
  // One write at a time; a chore that changed meanwhile (409) is read again, and the person is told.
  async function call(path, options, failure) {
    if (state.busy) return null;
    state.busy = true;
    state.notice = '';
    try {
      return await api(path, options);
    } catch (error) {
      const stale = error?.code === 'stale_revision';
      state.notice = stale ? 'This chore changed. The list is up to date again, try once more.' : error?.message || failure;
      if (stale) await load({quiet: true});
      return null;
    } finally {
      state.busy = false;
    }
  }
  const put = (chore, change) => call(`/chores/${encodeURIComponent(chore.id)}`, {method: 'PATCH', body: {revision: chore.revision, ...change}}, 'Could not change the chore. Try again.');
  const replace = (chore) => { state.chores = state.chores.map((item) => (item.id === chore.id ? chore : item)); };

  function drawCheckin() {
    const box = node('section', 'sched-checkin');
    if (!state.checkin) return box;
    const on = state.checkin.on !== false;
    const row = node('div', 'sched-switchrow');
    row.append(node('strong', '', 'Check-in'), toggle('Check-in', on, async () => {
      const done = await call('/checkin', {method: 'PUT', body: {on: !on}}, 'Could not change the check-in. Try again.');
      if (done) state.checkin = done.checkin;
      draw();
    }));
    box.append(row, node('p', 'sched-line', CHECKIN), node('p', 'sched-line', on && state.checkin.next_at ? `Next check-in ${whenWords(state.checkin.next_at, now())}` : 'Off'));
    return box;
  }

  function timeForm(chore) {
    const form = node('div', 'sched-form');
    const kind = node('select', 'sched-input');
    kind.setAttribute('aria-label', 'How often');
    for (const [value, label] of KINDS) {
      const option = node('option', '', label);
      option.value = value;
      option.selected = value === state.edit.kind;
      kind.append(option);
    }
    kind.addEventListener('change', () => { state.edit.kind = kind.value; draw(); });
    const field = node('input', 'sched-input');
    if (state.edit.kind === 'hours') {
      Object.assign(field, {type: 'number', min: '1', max: '24', value: String(state.edit.every)});
      field.setAttribute('aria-label', 'Hours between runs');
      field.addEventListener('input', () => { state.edit.every = field.value; });
    } else {
      Object.assign(field, {type: 'time', value: state.edit.at});
      field.setAttribute('aria-label', 'Time of day');
      field.addEventListener('input', () => { state.edit.at = field.value; });
    }
    const save = async () => {
      const {kind: how, at, every} = state.edit;
      const schedule = how === 'hours' ? {kind: how, every: Number(every)} : {kind: how, at};
      const done = await put(chore, {schedule});
      if (done) { replace(done.chore); state.edit = null; }
      draw();
    };
    const actions = node('div', 'sched-actions');
    actions.append(button('outline-button', 'Save', save), button('quiet-button', 'Cancel', () => { state.edit = null; draw(); }));
    form.append(kind, field, actions);
    return form;
  }

  function detail(chore) {
    const box = node('div', 'sched-detail');
    box.append(node('p', 'sched-request', chore.request));
    const level = developer() ? `P${chore.level} · ${chore.level_word}` : chore.level_word;
    box.append(node('p', 'sched-line', `Priority: ${level}`));
    if (state.edit?.id === chore.id) { box.append(timeForm(chore)); return box; }
    const actions = node('div', 'sched-actions');
    if (state.ask === chore.id) {
      actions.append(node('span', 'sched-ask', 'Delete this chore?'), button('quiet-button', 'Keep it', () => { state.ask = null; draw(); }),
        button('outline-button danger', 'Delete', () => remove(chore)));
    } else {
      actions.append(
        button('outline-button', 'Run now', () => runNow(chore), `Run now: ${chore.title}`),
        button('quiet-button', 'Edit time', () => {
          const at = chore.schedule.at ?? '09:00';
          state.edit = {id: chore.id, kind: chore.schedule.kind, at, every: chore.schedule.every ?? 4};
          draw();
        }, `Edit time: ${chore.title}`),
        button('quiet-button', 'Delete', () => { state.ask = chore.id; draw(); }, `Delete: ${chore.title}`));
    }
    box.append(actions);
    return box;
  }
  function rowNode(chore) {
    const item = node('li', 'sched-row');
    item.dataset.id = chore.id;
    item.dataset.enabled = String(chore.enabled);
    const open = state.open === chore.id;
    const head = node('div', 'sched-main');
    const text = button('sched-head', '', () => { state.open = open ? null : chore.id; state.edit = state.ask = null; draw(); });
    text.setAttribute('aria-expanded', String(open));
    const words = node('span', 'sched-text');
    words.append(node('strong', 'sched-title', chore.title), node('span', 'sched-when', chore.when),
      ...rowLines(chore, now()).map((line) => node('span', 'sched-line', line)));
    text.append(words);
    head.append(text, toggle(`${chore.title} on`, chore.enabled, async () => {
      const done = await put(chore, {enabled: !chore.enabled});
      if (done) replace(done.chore);
      draw();
    }));
    item.append(head);
    if (open) item.append(detail(chore));
    return item;
  }
  async function runNow(chore) {
    const done = await call(`/chores/${encodeURIComponent(chore.id)}/run`, {method: 'POST', body: {}}, 'Could not start the chore. Try again.');
    if (done) { replace(done.chore); state.notice = `Started: ${chore.title}. Its result shows in Activity and in the main chat.`; }
    draw();
  }
  async function remove(chore) {
    const done = await call(`/chores/${encodeURIComponent(chore.id)}?revision=${chore.revision}`, {method: 'DELETE'}, 'Could not delete the chore. Try again.');
    if (done) { state.chores = state.chores.filter((item) => item.id !== chore.id); state.open = null; }
    state.ask = null;
    draw();
  }

  function draw() {
    if (!state.loaded) { root.replaceChildren(node('p', 'panel-note', state.error || 'Reading the schedule…')); return; }
    const nodes = [drawCheckin()];
    if (state.scope === 'chat') nodes.push(node('p', 'panel-note', 'This chat’s own chores. The main chat lists all of them.'));
    if (state.chores.length) {
      const list = node('ul', 'sched-list');
      list.append(...state.chores.map(rowNode));
      nodes.push(list);
    } else {
      nodes.push(node('p', 'panel-soon', EMPTY));
    }
    if (state.notice) nodes.push(node('p', 'sched-fault', state.notice));
    root.replaceChildren(...nodes);
  }

  draw();
  return {root, state, show: () => load(), reload: () => (state.edit === null && !state.busy ? load({quiet: true}) : undefined)};
}

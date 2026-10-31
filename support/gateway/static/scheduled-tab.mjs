/**
 * The panel's Scheduled tab: the check-in and the chores that run on a schedule.
 *
 * `GET /api/chores` lists the chores (the main chat shows all of them, a side chat its own, by `chat()`) with the
 * check-in's state. The check-in's switch is `PUT /api/checkin`; a chore's switch and "Edit time" are `PATCH
 * /api/chores/{id}` with the revision that was read (a 409 reloads the list); "Run now" is `POST
 * /api/chores/{id}/run`; "Delete" asks first, then calls `DELETE`. `root` is mounted in the panel's body while the
 * tab is open; `show()` reads the list again and `reload()` does so quietly, unless an edit is open.
 */
import {language, t} from './i18n.mjs';
import {levelLabel} from './levels.mjs';

// Read once, in the page's language (a new language reloads the page).
export const EMPTY = t('sched.empty');
export const CHECKIN = t('sched.checkinLine', {n: 30});
const KINDS = ['daily', 'weekdays', 'hours'];
// How a chore's last run reads, by the state of its task (`sched.ran.<state>`).
const RAN = new Set(['done', 'failed', 'cancelled', 'expired', 'queued', 'running', 'held', 'waiting', 'parked', 'interrupted']);

/** ``today at 8:30 AM``, ``tomorrow at 8:30 AM``, ``Friday at 8:30 AM``, or ``Nov 2 at 8:30 AM``. */
export function whenWords(at, now) {
  if (!Number.isFinite(at)) return '';
  const time = new Date(at * 1000);
  const day = (stamp) => { const d = new Date(stamp * 1000); d.setHours(0, 0, 0, 0); return d.getTime() / 1000; };
  const ahead = Math.round((day(at) - day(now)) / 86400);
  // The day and the time as the page's language writes them (in English, as this computer does).
  const names = language() === 'en' ? [] : 'zh-CN';
  const clock = time.toLocaleTimeString(names, {hour: 'numeric', minute: '2-digit'});
  if (ahead <= 0) return t('sched.when.today', {clock});
  if (ahead === 1) return t('sched.when.tomorrow', {clock});
  const label = ahead < 7 ? time.toLocaleDateString(names, {weekday: 'long'}) : time.toLocaleDateString(names, {month: 'short', day: 'numeric'});
  return t('sched.when.day', {day: label, clock});
}

/** The lines under a row's title: when it runs next, then how the last run went. */
export function rowLines(chore, now) {
  const lines = [chore.enabled ? t('sched.next', {when: whenWords(chore.next_run, now)}) : t('common.off')];
  const ran = RAN.has(chore.last?.state) ? t(`sched.ran.${chore.last.state}`) : chore.last?.state;
  if (chore.last) lines.push(t('sched.last', {what: [ran, chore.last.line].filter(Boolean).join(' · ')}));
  return lines;
}

export function setupScheduledTab({doc, api, chat = () => '', clock = Date.now}) {
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
      state.error = error?.message || t('sched.readFailed');
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
      state.notice = stale ? t('sched.stale') : error?.message || failure;
      if (stale) await load({quiet: true});
      return null;
    } finally {
      state.busy = false;
    }
  }
  const put = (chore, change) => call(`/chores/${encodeURIComponent(chore.id)}`, {method: 'PATCH', body: {revision: chore.revision, ...change}}, t('sched.changeFailed'));
  const replace = (chore) => { state.chores = state.chores.map((item) => (item.id === chore.id ? chore : item)); };

  function drawCheckin() {
    const box = node('section', 'sched-checkin');
    if (!state.checkin) return box;
    const on = state.checkin.on !== false;
    const row = node('div', 'sched-switchrow');
    row.append(node('strong', '', t('sched.checkin')), toggle(t('sched.checkin'), on, async () => {
      const done = await call('/checkin', {method: 'PUT', body: {on: !on}}, t('sched.checkinFailed'));
      if (done) state.checkin = done.checkin;
      draw();
    }));
    box.append(row, node('p', 'sched-line', CHECKIN), node('p', 'sched-line', on && state.checkin.next_at ? t('sched.nextCheckin', {when: whenWords(state.checkin.next_at, now())}) : t('common.off')));
    return box;
  }

  function timeForm(chore) {
    const form = node('div', 'sched-form');
    const kind = node('select', 'sched-input');
    kind.setAttribute('aria-label', t('sched.howOften'));
    for (const value of KINDS) {
      const option = node('option', '', t(`sched.kind.${value}`));
      option.value = value;
      option.selected = value === state.edit.kind;
      kind.append(option);
    }
    kind.addEventListener('change', () => { state.edit.kind = kind.value; draw(); });
    const field = node('input', 'sched-input');
    if (state.edit.kind === 'hours') {
      Object.assign(field, {type: 'number', min: '1', max: '24', value: String(state.edit.every)});
      field.setAttribute('aria-label', t('sched.hoursBetween'));
      field.addEventListener('input', () => { state.edit.every = field.value; });
    } else {
      Object.assign(field, {type: 'time', value: state.edit.at});
      field.setAttribute('aria-label', t('sched.timeOfDay'));
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
    actions.append(button('outline-button', t('common.save'), save), button('quiet-button', t('common.cancel'), () => { state.edit = null; draw(); }));
    form.append(kind, field, actions);
    return form;
  }

  function detail(chore) {
    const box = node('div', 'sched-detail');
    box.append(node('p', 'sched-request', chore.request));
    box.append(node('p', 'sched-line', t('sched.priority', {level: levelLabel(chore.level)})));
    if (state.edit?.id === chore.id) { box.append(timeForm(chore)); return box; }
    const actions = node('div', 'sched-actions');
    if (state.ask === chore.id) {
      actions.append(node('span', 'sched-ask', t('sched.askDelete')), button('quiet-button', t('sched.keep'), () => { state.ask = null; draw(); }),
        button('outline-button danger', t('common.delete'), () => remove(chore)));
    } else {
      actions.append(
        button('outline-button', t('sched.runNow'), () => runNow(chore), t('sched.runNowLabel', {title: chore.title})),
        button('quiet-button', t('sched.editTime'), () => {
          const at = chore.schedule.at ?? '09:00';
          state.edit = {id: chore.id, kind: chore.schedule.kind, at, every: chore.schedule.every ?? 4};
          draw();
        }, t('sched.editTimeLabel', {title: chore.title})),
        button('quiet-button', t('common.delete'), () => { state.ask = chore.id; draw(); }, t('sched.deleteLabel', {title: chore.title})));
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
    head.append(text, toggle(t('sched.switchLabel', {title: chore.title}), chore.enabled, async () => {
      const done = await put(chore, {enabled: !chore.enabled});
      if (done) replace(done.chore);
      draw();
    }));
    item.append(head);
    if (open) item.append(detail(chore));
    return item;
  }
  async function runNow(chore) {
    const done = await call(`/chores/${encodeURIComponent(chore.id)}/run`, {method: 'POST', body: {}}, t('sched.startFailed'));
    if (done) { replace(done.chore); state.notice = t('sched.started', {title: chore.title}); }
    draw();
  }
  async function remove(chore) {
    const done = await call(`/chores/${encodeURIComponent(chore.id)}?revision=${chore.revision}`, {method: 'DELETE'}, t('sched.deleteFailed'));
    if (done) { state.chores = state.chores.filter((item) => item.id !== chore.id); state.open = null; }
    state.ask = null;
    draw();
  }

  function draw() {
    if (!state.loaded) { root.replaceChildren(node('p', 'panel-note', state.error || t('sched.reading'))); return; }
    const nodes = [drawCheckin()];
    if (state.scope === 'chat') nodes.push(node('p', 'panel-note', t('sched.chatScope')));
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

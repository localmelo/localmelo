/**
 * The Permissions section of Settings: what the person allowed, and a way to take each back.
 *
 * `GET /api/grants` lists every grant in display order: the ones that count everywhere first (newest first), then
 * the ones that count in one chat only, grouped by chat. `DELETE /api/grants/{id}` revokes one at once; a 404
 * means it is gone already, which is what the person wanted. The rows keep the server's order, so the numbers the
 * `/grants` command shows are the same ones. `root` is mounted in the section's panel; `show()` reads the list
 * (quietly when there is one already), `reset()` forgets it, and `count()` is the number of grants, or null
 * before the list was read.
 *
 * `grantWords`, `grantChat`, `grantPlace` and `grantWho` are the words of one grant, shared with the `/grants`
 * command so the two agree.
 */
import {formatTime, t} from './i18n.mjs';

const SEPARATOR = ' · ';
const DAY_S = 86400;
// The tools a grant names that have plain words (`permissions.tool.<name>`); any other reads "Use the tool X".
const TOOLS = new Set(['shell_exec', 'python_exec', 'file_write', 'file_edit']);
// What each other kind of grant names its target by: `permissions.what.<action>` takes {site}, {folder} or {app}.
const TARGETS = {open: 'site', act: 'site', read: 'site', folder: 'folder', window: 'app'};

/** What a grant lets melo do, in words: "Run shell commands", "Open example.com", "Read files in ~/Documents". */
export function grantWords(grant) {
  const target = String(grant?.target ?? '');
  const action = grant?.action;
  if (action === 'tool') return TOOLS.has(target) ? t(`permissions.tool.${target}`) : t('permissions.tool.other', {name: target});
  if (typeof action === 'string' && Object.hasOwn(TARGETS, action)) return t(`permissions.what.${action}`, {[TARGETS[action]]: target});
  // A kind of grant this page does not know: the gateway's own words.
  return String(grant?.label ?? '') || target;
}

/** The chat a grant was made in: "Main chat", its title, "Task: …", or a word for a chat without a title. */
export function grantChat(grant) {
  const kind = grant?.session_kind;
  const title = String(grant?.chat_title ?? '').trim();
  if (kind === 'main') return t('chat.main');
  if (kind === 'task') return title ? t('permissions.chat.task', {title}) : t('permissions.chat.taskUntitled');
  if (title) return title;
  return t(kind === 'terminal' ? 'permissions.chat.terminalUntitled' : 'permissions.chat.sideUntitled');
}

/** Where a grant counts: "everywhere", or the chat's name. */
export function grantPlace(grant) {
  return grant?.scope === 'global' ? t('permissions.place.everywhere') : grantChat(grant);
}

/** Who decided: "You", or "Auto" for what Auto mode allowed. */
export function grantWho(grant) {
  return t(grant?.by === 'auto' ? 'permissions.by.auto' : 'permissions.by.person');
}

const startOfDay = (seconds) => {
  const day = new Date(seconds * 1000);
  day.setHours(0, 0, 0, 0);
  return day.getTime() / 1000;
};

/** "just now", "12 min ago", "today at 9:05 AM", "yesterday at 9:05 AM", then the day: "Oct 28", "Oct 28, 2025". */
export function grantedWords(at, now) {
  const when = Number(at);
  if (!Number.isFinite(when)) return '';
  const ago = now - when;
  if (ago < 60) return t('permissions.when.now');
  if (ago < 3600) return t('permissions.when.minutes', {n: Math.floor(ago / 60)});
  const date = new Date(when * 1000);
  const days = Math.round((startOfDay(now) - startOfDay(when)) / DAY_S);
  if (days <= 0) return t('permissions.when.today', {clock: formatTime(date, 'clock')});
  if (days === 1) return t('permissions.when.yesterday', {clock: formatTime(date, 'clock')});
  return formatTime(date, date.getFullYear() === new Date(now * 1000).getFullYear() ? 'day' : 'dayYear');
}

export function setupPermissionsTab({doc, api, clock = Date.now, onChange = () => {}}) {
  const state = {loaded: false, grants: [], say: '', fault: false, busy: new Set()};
  let reads = 0; // a read that is not the newest one is dropped
  let epoch = 0; // reset() moves it on: what an old visit started changes nothing
  let controls = []; // the Revoke buttons as drawn, in order, with the grant each one belongs to
  function node(tag, className, text) {
    const element = doc.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }
  const root = node('div', 'perm-tab');
  root.tabIndex = -1; // the focus goes here when the last Revoke button is gone
  const status = node('p', 'perm-status');
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  const body = node('div', 'perm-body');
  root.append(status, body);
  const now = () => clock() / 1000;

  function say(text = '', fault = false) {
    state.say = text;
    state.fault = fault;
  }

  async function load() {
    const mine = ++reads;
    const visit = epoch;
    try {
      const page = await api('/grants');
      if (mine !== reads || visit !== epoch) return;
      Object.assign(state, {grants: Array.isArray(page?.grants) ? page.grants : [], loaded: true});
      say();
    } catch (error) {
      if (mine !== reads || visit !== epoch) return;
      say(error?.message || t('permissions.readFailed'), true);
    }
    draw();
    onChange();
  }

  async function revoke(grant) {
    if (state.busy.has(grant.id)) return;
    const visit = epoch;
    const what = grantWords(grant);
    const buttonOf = () => controls.find((item) => item.grant.id === grant.id)?.button;
    const place = controls.findIndex((item) => item.grant.id === grant.id);
    const focused = doc.activeElement !== undefined && doc.activeElement === buttonOf();
    // The button stays where it is (and keeps the focus) while the request runs; a redraw during that
    // time draws it disabled too.
    state.busy.add(grant.id);
    say();
    paintStatus();
    if (buttonOf()) buttonOf().disabled = true;
    // A 404 says the grant is gone already: the row goes, with no complaint.
    let gone = false;
    let failure = '';
    try {
      await api(`/grants/${encodeURIComponent(grant.id)}`, {method: 'DELETE'});
      gone = true;
      say(t('permissions.revoked', {what}));
    } catch (error) {
      if (error?.status === 404 || error?.code === 'not_found') gone = true;
      else failure = error?.message || t('permissions.revokeFailed');
    }
    if (visit !== epoch) return;
    state.busy.delete(grant.id);
    if (!gone) {
      say(failure, true);
      paintStatus();
      if (buttonOf()) buttonOf().disabled = false;
      return;
    }
    reads += 1; // a read that began before the revoke could bring the row back
    state.grants = state.grants.filter((item) => item.id !== grant.id);
    draw();
    if (focused) (controls[Math.min(place, controls.length - 1)]?.button ?? root).focus?.();
    onChange();
  }

  function rowNode(grant, fromMain) {
    const what = grantWords(grant);
    const item = node('li', 'st-row');
    item.dataset.id = grant.id;
    const text = node('div', 'st-text');
    const facts = [fromMain ? t('permissions.fromMain') : '', grantWho(grant), grantedWords(grant.granted_at, now())].filter(Boolean);
    text.append(node('span', 'st-label', what), node('p', 'st-hint', facts.join(SEPARATOR)));
    const button = node('button', 'quiet-button', t('permissions.revoke'));
    button.type = 'button';
    button.disabled = state.busy.has(grant.id);
    button.setAttribute('aria-label', t('permissions.revokeLabel', {what}));
    button.addEventListener('click', () => revoke(grant));
    controls.push({grant, button});
    const actions = node('div', 'st-actions');
    actions.append(button);
    item.append(text, actions);
    return item;
  }
  function listNode(grants, fromMain, labelledBy) {
    const list = node('ul', 'st-group');
    if (labelledBy) list.setAttribute('aria-labelledby', labelledBy);
    list.append(...grants.map((grant) => rowNode(grant, fromMain)));
    return list;
  }
  function heading(tag, className, text, id) {
    const element = node(tag, className, text);
    element.id = id;
    return element;
  }

  function groups() {
    const everywhere = state.grants.filter((grant) => grant.scope === 'global');
    // Each chat's grants together, the chats in the order the gateway listed them.
    const chats = new Map();
    for (const grant of state.grants) {
      if (grant.scope === 'global') continue;
      const key = grant.session_id ?? '';
      if (!chats.has(key)) chats.set(key, []);
      chats.get(key).push(grant);
    }
    const nodes = [];
    if (everywhere.length) {
      nodes.push(heading('h4', 'st-group-title', t('permissions.groupEverywhere'), 'perm-everywhere'), listNode(everywhere, true, 'perm-everywhere'));
    }
    if (chats.size) {
      nodes.push(heading('h4', 'st-group-title', t('permissions.groupChats'), 'perm-chats'));
      [...chats.values()].forEach((grants, index) => {
        const id = `perm-chat-${index}`;
        const box = node('div', 'perm-chat');
        box.dataset.chat = grants[0].session_id ?? '';
        box.setAttribute('role', 'group');
        box.setAttribute('aria-labelledby', id);
        box.append(heading('h5', 'perm-chat-title', grantChat(grants[0]), id), listNode(grants, false, id));
        nodes.push(box);
      });
    }
    return nodes;
  }

  function paintStatus() {
    status.textContent = state.say;
    status.dataset.state = state.fault ? 'error' : '';
  }
  function draw() {
    paintStatus();
    controls = [];
    if (!state.loaded) {
      if (!state.fault) { body.replaceChildren(node('p', 'st-hint', t('permissions.reading'))); return; }
      const retry = node('button', 'outline-button', t('permissions.retry'));
      retry.type = 'button';
      retry.addEventListener('click', () => { say(); draw(); load(); });
      body.replaceChildren(retry);
      return;
    }
    if (!state.grants.length) {
      const list = node('ul', 'st-group');
      list.append(node('li', 'st-empty', t('permissions.empty')));
      body.replaceChildren(list);
      return;
    }
    body.replaceChildren(...groups());
  }

  // A new visit to the dialog: what was listed belongs to the last one.
  function reset() {
    reads += 1;
    epoch += 1;
    Object.assign(state, {loaded: false, grants: [], say: '', fault: false, busy: new Set()});
    draw();
  }

  draw();
  return {root, state, show: () => load(), reset, count: () => (state.loaded ? state.grants.length : null)};
}

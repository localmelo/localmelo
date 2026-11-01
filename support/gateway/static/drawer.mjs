import {stepLine} from './blocked.mjs';
import {formatTime, t} from './i18n.mjs';
import {formatDuration, statusText} from './turns.mjs';

/**
 * Mount a session details region. `api` takes paths relative to /api and
 * `currentId` returns the active session id (or null). The host calls
 * sessionChanged() whenever that id changes, and close() from its Escape
 * handler before considering turn cancellation. `developer()` says whether
 * developer mode is on: it adds panels (the Events tab, memory internals, the
 * request and cache counts of a run) and never changes what a panel does.
 */
export function setupDrawer({api, currentId, root, trigger, developer = () => false}) {
  const doc = root.ownerDocument;
  let open = false;
  let id = null;
  let tab = 'tools';
  let generation = 0;
  let runs = [];
  let earlier = false;
  let busy = false;
  let returnFocus = null;

  function node(tag, text, className) {
    const element = doc.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }
  function button(text, action) {
    const element = node('button', text);
    element.type = 'button';
    element.addEventListener('click', action);
    return element;
  }
  const heading = node('h2', t('drawer.title'));
  const closeButton = button(t('drawer.close'), () => close());
  const tabs = node('div', undefined, 'drawer-tabs');
  const content = node('div', undefined, 'drawer-content');
  const message = node('p', '', 'drawer-message');
  message.setAttribute('role', 'status');
  message.setAttribute('aria-live', 'polite');
  root.setAttribute('role', 'region');
  root.setAttribute('aria-label', t('drawer.label'));
  root.hidden = true;
  root.replaceChildren(heading, closeButton, tabs, message, content);
  const tabButtons = {};
  const TABS = ['tools', 'memory', 'runs', 'events'];
  const available = (name) => name !== 'events' || developer();
  for (const name of TABS) {
    const item = button(t(`drawer.tab.${name}`), () => selectTab(name));
    tabButtons[name] = item;
    tabs.append(item);
  }
  function updateControls() {
    for (const [name, item] of Object.entries(tabButtons)) {
      item.hidden = !available(name);
      item.setAttribute('aria-pressed', String(name === tab));
    }
    if (trigger) trigger.setAttribute('aria-expanded', String(open));
  }
  function valid(token, session, selected) {
    return open && token === generation && id === session && currentId() === session && tab === selected;
  }
  function path(session, suffix) {
    return `/sessions/${encodeURIComponent(session)}/${suffix}`;
  }
  function errorText(error) {
    return error?.message || t('drawer.requestFailed');
  }
  function line(parent, label, value) {
    parent.append(node('p', t('drawer.line', {label, value})));
  }
  function toolView(data) {
    content.append(node('p', t('drawer.tools.opens')));
    const tools = data.tools || [];
    if (!tools.length) content.append(node('p', t('drawer.tools.none')));
    for (const tool of tools) {
      const row = node('div', undefined, 'drawer-tool');
      row.append(node('strong', tool.name), node('span', ` ${tool.source || ''} · ${t(tool.granted ? 'drawer.tools.allowed' : 'drawer.tools.notAllowed')}`));
      if (tool.description) row.append(node('p', tool.description));
      if (tool.granted) {
        const revoke = button(t('drawer.tools.revoke'), () => revokeGrants(tool.name));
        revoke.setAttribute('aria-label', t('drawer.tools.revokeOne', {tool: tool.name}));
        row.append(revoke);
      }
      content.append(row);
    }
    const folders = Array.isArray(data.folders) ? data.folders : [];
    if (tools.some((tool) => tool.granted) || folders.length) content.append(button(t('drawer.tools.revokeAll'), () => revokeGrants()));
    foldersView(folders);
    approvalsView(data.approvals || []);
    if (data.mcp_config_label) content.append(node('p', data.mcp_config_label));
    if (data.mcp_error) content.append(node('p', data.mcp_error));
  }
  // The folders outside the workspace this chat may read until it ends (#405), each revocable.
  function foldersView(folders) {
    if (!folders.length) return;
    content.append(node('h3', t('drawer.folders.title'), 'drawer-subhead'), node('p', t('drawer.folders.note')));
    for (const folder of folders) {
      const label = String(folder.label || folder.path || '');
      const row = node('div', undefined, 'drawer-tool drawer-approval');
      const where = node('code', label, 'drawer-approval-step');
      where.title = String(folder.path ?? '');
      const revoke = button(t('drawer.tools.revoke'), () => revokeAt(`folders?path=${encodeURIComponent(String(folder.path ?? ''))}`));
      revoke.setAttribute('aria-label', t('drawer.folders.revokeOne', {folder: label}));
      row.append(where, revoke);
      content.append(row);
    }
  }
  // The steps the person allowed in this chat, which its tasks run again without asking (#369).
  function approvalsView(approvals) {
    if (!approvals.length) return;
    content.append(node('h3', t('drawer.approvals.title'), 'drawer-subhead'), node('p', t('drawer.approvals.note')));
    for (const entry of approvals) {
      const row = node('div', undefined, 'drawer-tool drawer-approval');
      const step = node('code', stepLine(entry.preview), 'drawer-approval-step');
      step.title = String(entry.preview ?? '');
      const at = formatTime(new Date(entry.at * 1000), 'clock');
      row.append(step, node('p', t('drawer.approvals.when', {tool: entry.tool, time: at})));
      const revoke = button(t('approvals.revoke'), () => revokeAt(`approvals/${encodeURIComponent(entry.id)}`));
      revoke.setAttribute('aria-label', t('approvals.revokeStep', {step: stepLine(entry.preview, 60)}));
      row.append(revoke);
      content.append(row);
    }
  }
  function memoryView(data) {
    line(content, t('drawer.memory.working'), `${(data.working || []).length} / ${data.working_max}`);
    const longTerm = data.long_term || {};
    line(content, t('drawer.memory.longTerm'), `${t(longTerm.enabled ? 'drawer.memory.on' : 'drawer.memory.off')}${developer() ? ` (${longTerm.backend || 'none'})` : ''}`);
    if (developer() && data.agent) line(content, t('drawer.memory.agent'), typeof data.agent === 'string' ? data.agent : JSON.stringify(data.agent));
    if (!data.working?.length) content.append(node('p', t('drawer.memory.empty')));
    for (const entry of data.working || []) {
      const row = node('div', undefined, 'drawer-memory');
      row.append(node('strong', entry.role), node('p', entry.text));
      if (entry.tool_calls?.length) line(row, t('drawer.memory.tools'), entry.tool_calls.join(', '));
      if (entry.tool_call_id) line(row, t('drawer.memory.toolCall'), entry.tool_call_id);
      if (entry.images) line(row, t('drawer.memory.images'), entry.images);
      content.append(row);
    }
  }
  function timestamp(value) {
    if (value == null) return '—';
    const date = new Date(value * 1000);
    return Number.isNaN(date.getTime()) ? String(value) : formatTime(date, 'full');
  }
  function runsView() {
    if (!runs.length) content.append(node('p', t('drawer.runs.none')));
    for (const run of runs) {
      const row = node('div', undefined, 'drawer-run');
      const status = statusText(run.status);
      row.append(node('strong', t('drawer.runs.turn', {turn: run.turn, status: status.word})));
      line(row, t('drawer.runs.started'), timestamp(run.started_at));
      line(row, t('drawer.runs.finished'), timestamp(run.finished_at));
      if (typeof run.started_at === 'number' && typeof run.finished_at === 'number') {
        line(row, t('drawer.runs.duration'), formatDuration(run.finished_at - run.started_at));
      }
      if (developer() && run.result && typeof run.result === 'object') {
        for (const [key, value] of Object.entries(run.result)) {
          if (key !== 'result') line(row, key, typeof value === 'object' ? JSON.stringify(value) : String(value));
        }
      }
      content.append(row);
    }
    if (earlier && runs.length) content.append(button(t('drawer.runs.earlier'), () => load(true)));
    else if (earlier) content.append(node('p', t('drawer.runs.noEarlier')));
  }
  function eventsView(data) {
    const turn = (data.turns || []).at(-1);
    const events = turn?.events || [];
    if (!events.length) content.append(node('p', t('drawer.events.none')));
    else line(content, t('drawer.events.ofTurn'), turn.turn);
    for (const event of events) {
      const row = node('div', undefined, 'drawer-event');
      row.append(node('strong', `#${event.id} ${event.type}`), node('pre', JSON.stringify(event, null, 1)));
      content.append(row);
    }
  }
  function clearContent(...children) {
    if (open && content.contains(doc.activeElement)) closeButton.focus();
    content.replaceChildren(...children);
  }
  function render(data) {
    clearContent();
    if (tab === 'tools') toolView(data);
    else if (tab === 'memory') memoryView(data);
    else if (tab === 'events') eventsView(data);
    else runsView();
  }
  async function load(append = false, notice = '') {
    if (!open || !id || currentId() !== id || busy) return;
    const session = id;
    const selected = tab;
    const token = ++generation;
    busy = true;
    message.textContent = t('drawer.loading');
    if (!append) clearContent();
    try {
      const cursor = append ? `?before=${encodeURIComponent(Math.min(...runs.map((run) => run.turn)))}` : '';
      const data = await api(path(session, selected === 'events' ? 'turns?limit=1&events=true' : selected + cursor));
      if (!valid(token, session, selected)) return;
      if (selected === 'runs') {
        runs = append ? [...data.runs, ...runs] : data.runs;
        earlier = data.has_earlier;
      }
      message.textContent = notice;
      render(data);
    } catch (error) {
      if (!valid(token, session, selected)) return;
      message.textContent = notice ? `${notice} ${errorText(error)}` : errorText(error);
      if (!append) {
        clearContent(button(t('drawer.retry'), () => load(false, notice)));
      }
    } finally {
      if (valid(token, session, selected)) busy = false;
    }
  }
  function revokeGrants(tool) {
    return revokeAt(`grants${tool === undefined ? '' : '/' + encodeURIComponent(tool)}`);
  }
  async function revokeAt(suffix) {
    if (busy || !open || currentId() !== id) return;
    const session = id;
    const token = ++generation;
    busy = true;
    message.textContent = t('drawer.revoking');
    if (content.contains(doc.activeElement)) closeButton.focus();
    for (const item of content.querySelectorAll('button')) item.disabled = true;
    try {
      await api(path(session, suffix), {method: 'DELETE'});
      if (!open || id !== session || currentId() !== session || tab !== 'tools') return;
      // A tab may have been reopened while the DELETE was pending. Refresh
      // even then: an earlier GET may have observed the old grant.
      busy = false;
      await load();
    } catch (error) {
      if (error?.status === 404 && open && id === session && currentId() === session && tab === 'tools') {
        busy = false;
        await load(false, errorText(error));
        return;
      }
      if (!valid(token, session, 'tools')) return;
      message.textContent = errorText(error);
      for (const item of content.querySelectorAll('button')) item.disabled = false;
      busy = false;
    }
  }
  function selectTab(next) {
    if (!TABS.includes(next) || !available(next) || !open) return;
    generation++;
    busy = false;
    tab = next;
    runs = [];
    earlier = false;
    updateControls();
    return load();
  }
  function guardFocus(event) {
    if (open && !root.contains(event.target)) closeButton.focus();
  }
  function close() {
    if (!open) return false;
    generation++;
    open = false;
    doc.removeEventListener('focusin', guardFocus);
    busy = false;
    root.hidden = true;
    content.replaceChildren();
    message.textContent = '';
    updateControls();
    returnFocus?.focus();
    return true;
  }
  function show(initialTab) {
    const session = currentId();
    if (!session || (initialTab !== undefined && !(TABS.includes(initialTab) && available(initialTab)))) return false;
    if (initialTab !== undefined) tab = initialTab;
    if (!available(tab)) tab = 'tools';
    generation++;
    busy = false;
    id = session;
    runs = [];
    earlier = false;
    if (!open) {
      returnFocus = doc.activeElement;
      open = true;
      doc.addEventListener('focusin', guardFocus);
    }
    root.hidden = false;
    updateControls();
    closeButton.focus();
    return load();
  }
  function modeChanged() {
    if (!open) return;
    if (!available(tab)) tab = 'tools';
    updateControls();
    return load();
  }
  function sessionChanged() {
    if (!open) return;
    const session = currentId();
    if (!session) return close();
    id = session;
    generation++;
    busy = false;
    runs = [];
    earlier = false;
    return load();
  }
  root.addEventListener('keydown', (event) => {
    if (!open || event.key !== 'Tab') return;
    const focusable = [...root.querySelectorAll('button, [href], input, select, textarea, [tabindex]')]
      .filter((item) => !item.disabled && !item.hidden && item.tabIndex >= 0);
    if (!focusable.length) {
      event.preventDefault();
      return;
    }
    const index = focusable.indexOf(doc.activeElement);
    if (event.shiftKey && index <= 0) {
      event.preventDefault();
      focusable[focusable.length - 1].focus();
    } else if (!event.shiftKey && (index === -1 || index === focusable.length - 1)) {
      event.preventDefault();
      focusable[0].focus();
    }
  });
  updateControls();
  return {open: show, close, selectTab, sessionChanged, modeChanged, isOpen: () => open};
}

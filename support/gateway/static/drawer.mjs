import {formatDuration, statusText} from './turns.mjs';

/**
 * Mount a session details region. `api` takes paths relative to /api and
 * `currentId` returns the active session id (or null). The host calls
 * sessionChanged() whenever that id changes, and close() from its Escape
 * handler before considering turn cancellation.
 */
export function setupDrawer({api, currentId, root, trigger}) {
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
  const heading = node('h2', 'Details');
  const closeButton = button('Close details', () => close());
  const tabs = node('div', undefined, 'drawer-tabs');
  const content = node('div', undefined, 'drawer-content');
  const message = node('p', '', 'drawer-message');
  message.setAttribute('role', 'status');
  message.setAttribute('aria-live', 'polite');
  root.setAttribute('role', 'region');
  root.setAttribute('aria-label', 'Session details');
  root.hidden = true;
  root.replaceChildren(heading, closeButton, tabs, message, content);
  const tabButtons = {};
  for (const name of ['tools', 'memory', 'runs']) {
    const item = button(name[0].toUpperCase() + name.slice(1), () => selectTab(name));
    tabButtons[name] = item;
    tabs.append(item);
  }
  function updateControls() {
    for (const [name, item] of Object.entries(tabButtons)) {
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
    return error?.message || 'Request failed. Please try again.';
  }
  function line(parent, label, value) {
    parent.append(node('p', `${label}: ${value}`));
  }
  function toolView(data) {
    const tools = data.tools || [];
    if (!tools.length) content.append(node('p', 'No tools available.'));
    for (const tool of tools) {
      const row = node('div', undefined, 'drawer-tool');
      row.append(node('strong', tool.name), node('span', ` ${tool.source || ''} · ${tool.granted ? 'Allowed' : 'Not allowed'}`));
      if (tool.description) row.append(node('p', tool.description));
      if (tool.granted) {
        const revoke = button('Revoke', () => revokeGrants(tool.name));
        revoke.setAttribute('aria-label', `Revoke ${tool.name}`);
        row.append(revoke);
      }
      content.append(row);
    }
    if (tools.some((tool) => tool.granted)) content.append(button('Revoke all grants', () => revokeGrants()));
    if (data.mcp_config_label) content.append(node('p', data.mcp_config_label));
    if (data.mcp_error) content.append(node('p', data.mcp_error));
  }
  function memoryView(data) {
    line(content, 'Working memory', `${(data.working || []).length} / ${data.working_max}`);
    const longTerm = data.long_term || {};
    line(content, 'Long-term memory', `${longTerm.enabled ? 'Enabled' : 'Disabled'} (${longTerm.backend || 'none'})`);
    if (data.agent) line(content, 'Agent', typeof data.agent === 'string' ? data.agent : JSON.stringify(data.agent));
    if (!data.working?.length) content.append(node('p', 'No working memory.'));
    for (const entry of data.working || []) {
      const row = node('div', undefined, 'drawer-memory');
      row.append(node('strong', entry.role), node('p', entry.text));
      if (entry.tool_calls?.length) line(row, 'Tools', entry.tool_calls.join(', '));
      if (entry.tool_call_id) line(row, 'Tool call', entry.tool_call_id);
      if (entry.images) line(row, 'Images', entry.images);
      content.append(row);
    }
  }
  function timestamp(value) {
    if (value == null) return '—';
    const date = new Date(value * 1000);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }
  function runsView() {
    if (!runs.length) content.append(node('p', 'No finished runs.'));
    for (const run of runs) {
      const row = node('div', undefined, 'drawer-run');
      const status = statusText(run.status);
      row.append(node('strong', `Turn ${run.turn} · ${status.word}`));
      line(row, 'Started', timestamp(run.started_at));
      line(row, 'Finished', timestamp(run.finished_at));
      if (typeof run.started_at === 'number' && typeof run.finished_at === 'number') {
        line(row, 'Duration', formatDuration(run.finished_at - run.started_at));
      }
      if (run.result && typeof run.result === 'object') {
        for (const [key, value] of Object.entries(run.result)) {
          if (key !== 'result') line(row, key, typeof value === 'object' ? JSON.stringify(value) : String(value));
        }
      }
      content.append(row);
    }
    if (earlier && runs.length) content.append(button('Earlier runs', () => load(true)));
    else if (earlier) content.append(node('p', 'Earlier runs cannot be loaded because this page has no finished runs.'));
  }
  function render(data) {
    content.replaceChildren();
    if (tab === 'tools') toolView(data);
    else if (tab === 'memory') memoryView(data);
    else runsView();
  }
  async function load(append = false) {
    if (!open || !id || currentId() !== id || busy) return;
    const session = id;
    const selected = tab;
    const token = ++generation;
    busy = true;
    message.textContent = 'Loading…';
    if (!append) content.replaceChildren();
    try {
      const cursor = append ? `?before=${encodeURIComponent(Math.min(...runs.map((run) => run.turn)))}` : '';
      const data = await api(path(session, selected + cursor));
      if (!valid(token, session, selected)) return;
      if (selected === 'runs') {
        runs = append ? [...data.runs, ...runs] : data.runs;
        earlier = data.has_earlier;
      }
      message.textContent = '';
      render(data);
    } catch (error) {
      if (!valid(token, session, selected)) return;
      message.textContent = errorText(error);
      if (!append) {
        content.replaceChildren(button('Retry', () => load()));
      }
    } finally {
      if (valid(token, session, selected)) busy = false;
    }
  }
  async function revokeGrants(tool) {
    if (busy || !open || currentId() !== id) return;
    const session = id;
    const token = ++generation;
    busy = true;
    message.textContent = 'Revoking…';
    for (const item of content.querySelectorAll('button')) item.disabled = true;
    try {
      await api(path(session, `grants${tool === undefined ? '' : '/' + encodeURIComponent(tool)}`), {method: 'DELETE'});
      if (!open || id !== session || currentId() !== session || tab !== 'tools') return;
      // A tab may have been reopened while the DELETE was pending. Refresh
      // even then: an earlier GET may have observed the old grant.
      busy = false;
      await load();
    } catch (error) {
      if (!valid(token, session, 'tools')) return;
      message.textContent = errorText(error);
      for (const item of content.querySelectorAll('button')) item.disabled = false;
      busy = false;
    }
  }
  function selectTab(next) {
    if (!['tools', 'memory', 'runs'].includes(next) || !open) return;
    generation++;
    busy = false;
    tab = next;
    runs = [];
    earlier = false;
    updateControls();
    return load();
  }
  function close() {
    if (!open) return false;
    generation++;
    open = false;
    busy = false;
    root.hidden = true;
    content.replaceChildren();
    message.textContent = '';
    updateControls();
    returnFocus?.focus();
    return true;
  }
  function show() {
    const session = currentId();
    if (!session) return false;
    generation++;
    busy = false;
    id = session;
    runs = [];
    earlier = false;
    returnFocus = doc.activeElement;
    open = true;
    root.hidden = false;
    updateControls();
    closeButton.focus();
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
  return {open: show, close, selectTab, sessionChanged, isOpen: () => open};
}

/**
 * The header's "N running" chip and the list it opens: what runs now across
 * chats, tasks, worker runs and terminal chats, from `status` of
 * `GET /api/health`. `update(status)` is called on every health poll; the chip
 * is hidden at 0 and the list redraws while it is open. The page closes the
 * list on Escape and on a click outside (`toggle(false)`), as it does the level
 * menu. Each row links to the view that shows it: the chat, the task's card,
 * or the Workers detail.
 */
const KINDS = {
  chat: {label: 'Chat', plural: 'chats', href: (id) => `#/s/${id}`},
  task: {label: 'Task', plural: 'tasks', href: (id) => `#/tasks/${id}`},
  run: {label: 'Run', plural: 'runs', href: (id) => `#/workers/${id}`},
  terminal: {label: 'Terminal chat', plural: 'terminal chats', href: (id) => `#/workers/${id}`},
};
const COUNTS = {chats: 'chat', tasks: 'task', runs: 'run', terminal: 'terminal'};

/** `"2 tasks, 1 chat"`: the counts of `running` that are above 0. */
export function describe(running) {
  const parts = [];
  for (const [field, kind] of Object.entries(COUNTS)) {
    const n = running?.[field] ?? 0;
    if (n > 0) parts.push(`${n} ${n === 1 ? KINDS[kind].label.toLowerCase() : KINDS[kind].plural}`);
  }
  return parts.join(', ');
}

export function setupRunning({root, button, menu}) {
  const doc = root.ownerDocument;
  let status = null;
  let seen = '';
  let open = false;

  const total = () => Object.keys(COUNTS).reduce((sum, field) => sum + (status?.running?.[field] ?? 0), 0);

  function row(item) {
    const kind = KINDS[item.kind] ?? KINDS.run;
    const link = doc.createElement('a');
    link.className = 'running-item';
    link.href = kind.href(encodeURIComponent(item.id));
    const label = doc.createElement('span');
    label.className = 'running-kind';
    label.textContent = kind.label;
    const title = doc.createElement('span');
    title.className = 'running-title';
    title.textContent = item.title || 'Untitled';
    link.append(label, title);
    return link;
  }

  function draw() {
    const count = total();
    root.hidden = count === 0;
    if (count === 0) set(false);
    button.textContent = `${count} running`;
    button.setAttribute('aria-label', `${count} running: ${describe(status?.running)}`);
    if (!open) return;
    const items = status?.items ?? [];
    const nodes = items.map(row);
    if (count > items.length) {
      const more = doc.createElement('div');
      more.className = 'running-more';
      more.textContent = `and ${count - items.length} more`;
      nodes.push(more);
    }
    menu.replaceChildren(...nodes);
  }

  function set(next) {
    open = next;
    menu.hidden = !next;
    button.setAttribute('aria-expanded', String(next));
    if (next) draw();
  }

  button.addEventListener('click', () => set(!open));
  menu.addEventListener('click', () => set(false));

  return {
    update(next) {
      const json = JSON.stringify(next ?? null);
      if (json === seen) return;
      seen = json;
      status = next ?? null;
      draw();
    },
    toggle: set,
    isOpen: () => open,
  };
}

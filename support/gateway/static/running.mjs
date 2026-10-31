/**
 * The header's "N running" chip and the list it opens: what runs now across
 * chats, tasks, worker runs and terminal chats, from `status` of
 * `GET /api/health`. `update(status)` is called on every health poll; the chip
 * is hidden at 0 and the list redraws while it is open. The page closes the
 * list on Escape and on a click outside (`toggle(false)`), as it does the level
 * menu. Each row links to the view that shows it: the chat, the task's card,
 * or the Workers detail.
 */
import {t} from './i18n.mjs';

// Each kind's words are the catalog's `running.kind.<kind>` and `running.count.<kind>`.
const KINDS = {
  chat: {href: (id) => `#/s/${id}`},
  task: {href: (id) => `#/tasks/${id}`},
  run: {href: (id) => `#/workers/${id}`},
  terminal: {href: (id) => `#/workers/${id}`},
};
const COUNTS = {chats: 'chat', tasks: 'task', runs: 'run', terminal: 'terminal'};

/** `"2 tasks, 1 chat"`: the counts of `running` that are above 0. */
export function describe(running) {
  const parts = [];
  for (const [field, kind] of Object.entries(COUNTS)) {
    const n = running?.[field] ?? 0;
    if (n > 0) parts.push(t(`running.count.${kind}`, {n}));
  }
  return parts.join(t('common.listSeparator'));
}

export function setupRunning({root, button, menu}) {
  const doc = root.ownerDocument;
  let status = null;
  let seen = '';
  let open = false;

  const total = () => Object.keys(COUNTS).reduce((sum, field) => sum + (status?.running?.[field] ?? 0), 0);

  function row(item) {
    const name = KINDS[item.kind] ? item.kind : 'run';
    const kind = KINDS[name];
    const link = doc.createElement('a');
    link.className = 'running-item';
    link.href = kind.href(encodeURIComponent(item.id));
    const label = doc.createElement('span');
    label.className = 'running-kind';
    label.textContent = t(`running.kind.${name}`);
    const title = doc.createElement('span');
    title.className = 'running-title';
    title.textContent = item.title || t('common.untitled');
    link.append(label, title);
    return link;
  }

  function draw() {
    const count = total();
    root.hidden = count === 0;
    if (count === 0) set(false);
    button.textContent = t('running.chip', {n: count});
    button.setAttribute('aria-label', t('running.chipLabel', {n: count, what: describe(status?.running)}));
    if (!open) return;
    const items = status?.items ?? [];
    const nodes = items.map(row);
    if (count > items.length) {
      const more = doc.createElement('div');
      more.className = 'running-more';
      more.textContent = t('running.more', {n: count - items.length});
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

import {cleanOutput, formatDuration} from './turns.mjs';
import {taskLive} from './task-live.mjs';
import {setupMemoryTab} from './memory-tab.mjs';
import {setupScheduledTab} from './scheduled-tab.mjs';

/**
 * The panel beside a chat: Melo's portrait and status, then tabs. It follows
 * the open chat (`chat()`, by default the `#/s/<id>` of the address): a side
 * chat's panel lists its own activity and workers, the main chat's (or none)
 * the whole day. Activity lists what runs now and what ran today
 * (`GET /api/activity`); a row opens in place to its live steps (task-live.mjs,
 * which follows any session's events) with the actions the terminal's task
 * list offers, and carries the approvals it asked for folded under it, where
 * one that waits is answered. Workers lists the tasks and runs of the same
 * scope. Scheduled (scheduled-tab.mjs) lists the check-in and the chores that run on a schedule; Memory
 * (memory-tab.mjs) lists the notes melo keeps. `update(health)` is called on every health poll: the
 * header follows it, and the list is read again when what runs changed, the
 * chat changed, or every REFRESH_MS. Below 900 px the panel is a bottom sheet
 * behind `chip`, closed by Escape (`closeSheet()`), the backdrop, or a drag
 * down on its handle. `api` takes paths relative to /api.
 */
export const GLYPHS = {
  wait: '!', held: '◷', queued: '◷', paused: '‖', read: '▤', write: '✎', run: '>_', web: '⊕',
  memory: '◎', think: '…', tool: '⚙', done: '✓', failed: '✗', stopped: '⊘', running: '●',
};
const TABS = [['activity', 'Activity', '≡'], ['workers', 'Workers', '⚙'], ['scheduled', 'Scheduled', '◷'], ['memory', 'Memory', '◎']];
const EMPTY_WORKERS = {
  chat: 'This chat has not started any workers. When it hands a task to one, the worker is listed here, running first.',
  all: 'No workers yet today. When Melo hands a task to a worker, or you run one yourself, it is listed here, running first.',
};
// A row's state in words, and how an approval was answered.
const STATE_WORDS = {running: 'Working', waiting: 'Waiting for your approval', held: 'Waiting for the model', queued: 'Waiting to start',
  paused: 'Paused', done: 'Done', failed: 'Failed', stopped: 'Stopped'};
const ANSWERS = {waiting: 'Waiting for you', 'allowed once': 'Allowed once', 'allowed for this chat': 'Allowed for this chat',
  'allowed automatically': 'Allowed automatically', denied: 'Denied', 'no answer': 'No answer in time', withdrawn: 'Not run: it stopped first'};
const LIVE = new Set(['running', 'waiting', 'held', 'queued', 'paused']);
const LEVELS = ['Now', 'Today', 'Normal', 'Whenever', 'If idle'];
const HREF = {chat: '#/s/', task: '#/tasks/', run: '#/workers/', terminal: '#/workers/'};
const OPEN = {chat: 'Open chat', task: 'Open task', run: 'Open run', terminal: 'Open terminal chat'};
const SHOWN = 8;
const STEPS = 8;
const REFRESH_MS = 30000;
const KEY = 'localmelo-panel-tab';
const TESTS = /(?:^|[\s;&|(])(?:pytest|py\.test|tox|nox|jest|vitest|(?:npm|pnpm|yarn|bun)(?: run)? test|(?:cargo|go|swift|deno|mix) test|python3? -m (?:pytest|unittest)|make (?:test|check))(?:$|[\s;&|)])/;

const chatOfAddress = () => /^#\/s\/([A-Za-z0-9._:-]+)$/.exec(globalThis.location?.hash ?? '')?.[1] ?? '';
const base = (path) => String(path ?? '').replace(/\/+$/, '').split('/').pop();

/** A step's plain words, as the server's `activity_text` says them: `Reading notes.md`, or `Read notes.md` once done. */
export function stepWords(name, args, done = false) {
  const tool = String(name ?? '').startsWith('mcp__') ? String(name).split('__').pop() : String(name ?? '');
  const a = args && typeof args === 'object' ? args : {};
  const say = (now, past, what = '') => `${done ? past : now}${what ? ` ${what}` : ''}`;
  if (tool === 'file_read') return say('Reading', 'Read', base(a.path) || 'a file');
  if (tool === 'file_write') return say('Writing', 'Wrote', base(a.path) || 'a file');
  if (tool === 'list_files') return say('Looking through', 'Looked through', base(a.path) || 'the folder');
  if (tool === 'shell_exec') return TESTS.test(String(a.command ?? '').replace(/\s+/g, ' ')) ? say('Running', 'Ran', 'the tests') : say('Running', 'Ran', 'a command');
  if (tool === 'python_exec') return say('Running', 'Ran', 'Python');
  if (tool === 'recall' || tool === 'search_memory' || tool === 'remember') return say('Remembering', 'Remembered');
  if (tool === 'browser_open') {
    let host = '';
    try { host = new URL(String(a.url)).hostname.replace(/^www\./, ''); } catch { /* not a page address */ }
    return host ? say('Reading', 'Read', host) : say('Opening', 'Opened', 'a page');
  }
  if (tool === 'browser_follow') return say('Following', 'Followed', 'a link');
  if (tool === 'browser_screenshot') return say('Looking at', 'Looked at', 'the page');
  if (tool.startsWith('browser_')) return say('Reading', 'Read', 'the page');
  return say('Using', 'Used', 'a tool');
}

/**
 * What the header says, and the link it offers: the model's state first, then what runs. For a side chat,
 * `scoped` is that chat's rows and what runs is read from them instead of the whole console.
 */
export function headline(health, scoped = null) {
  if (!health) return {text: 'Connecting…', mood: 'idle', fix: false};
  if (health.configured === false) return {text: 'No model set up yet', mood: 'error', fix: true};
  if (health.model_server?.reachable === false) return {text: 'Model server not reachable', mood: 'error', fix: true};
  const status = health.status ?? {};
  const count = scoped ? scoped.filter((row) => row.state === 'running' || row.state === 'held').length
    : Object.values(status.running ?? {}).reduce((sum, n) => sum + (Number(n) || 0), 0);
  if (scoped ? scoped.some((row) => row.state === 'waiting') : status.waiting?.length) return {text: 'Waiting for you', mood: 'waiting', fix: false, count};
  if (count > 0) return {text: `Working on ${count === 1 ? '1 thing' : `${count} things`}`, mood: 'working', fix: false, count};
  const recent = scoped && !scoped.some((row) => row.id === status.recent?.id) ? null : status.recent;
  const mood = recent && recent.ended_s_ago < 8 && recent.outcome !== 'stopped' ? (recent.outcome === 'done' ? 'done' : 'error') : 'idle';
  const model = health.model?.name;
  const server = health.model?.backend_label;
  return {text: model ? `Connected · ${model}${server ? ` on ${server}` : ''}` : 'Connected', mood, fix: false, count: 0};
}

export function setupPanel({
  root, chip, backdrop, api, developer = () => false, openSettings = () => {}, cover = () => [],
  storage = globalThis.localStorage, clock = Date.now, live: stream = {}, chat = chatOfAddress,
  wide = globalThis.matchMedia?.('(min-width: 900px)'),
}) {
  const doc = root.ownerDocument;
  const state = {health: null, rows: [], more: false, unavailable: [], loaded: false, error: '', notice: '', noticeFor: '',
    tab: 'activity', open: null, all: false, sheet: false, busy: '', fetchedAt: 0, seen: null, day: 0,
    chat: chat(), scope: 'all', folds: new Set()};
  try { if (TABS.some(([id]) => id === storage?.getItem(KEY))) state.tab = storage.getItem(KEY); } catch { /* no storage */ }
  // A row's steps are its session's newest turn: a long chat is not read back to its first turn.
  const newest = (path, options) => (/\/turns\?/.test(path)
    ? api(path.replace(/limit=\d+/, 'limit=1'), options).then((page) => ({...page, has_earlier: false}))
    : api(path, options));
  const memory = setupMemoryTab({doc, api, developer});
  const scheduled = setupScheduledTab({doc, api, developer, clock, chat: () => state.chat});
  const live = taskLive({api: newest, changed: () => drawBody(), doc, ...stream});
  let reads = 0;

  function node(tag, className, text) {
    const element = doc.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }
  function button(className, text, action) {
    const element = node('button', className, text);
    element.type = 'button';
    element.addEventListener('click', action);
    return element;
  }

  // The header: the portrait (the one bold thing), the name and one status line.
  const ring = node('div', 'panel-portrait');
  const frame = node('span', 'panel-picture');
  const picture = node('img');
  picture.alt = '';
  picture.width = 108;
  picture.height = 140;
  frame.append(picture);
  ring.append(frame);
  const status = node('p', 'panel-status');
  status.setAttribute('role', 'status');
  const head = node('header', 'panel-head');
  const handle = button('panel-handle', '', () => setSheet(false));
  handle.setAttribute('aria-label', 'Close activity');
  head.append(handle, ring, node('h2', 'panel-name', 'Melo'), status);
  const tablist = node('div', 'panel-tabs');
  tablist.setAttribute('role', 'tablist');
  tablist.setAttribute('aria-label', 'Melo');
  const tabs = TABS.map(([id, label, glyph]) => {
    const tab = button('panel-tab', '', () => select(id));
    tab.id = `panel-tab-${id}`;
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', 'panel-body');
    tab.dataset.tab = id;
    tab.append(node('span', 'panel-tab-glyph', glyph), node('span', '', label));
    tab.firstChild.setAttribute('aria-hidden', 'true');
    return tab;
  });
  tablist.append(...tabs);
  tablist.addEventListener('keydown', (event) => {
    const at = TABS.findIndex(([id]) => id === state.tab);
    const to = {ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: TABS.length - 1}[event.key];
    if (to === undefined) return;
    event.preventDefault();
    select(TABS[(to + TABS.length) % TABS.length][0], {focus: true});
  });
  const body = node('div', 'panel-body');
  body.id = 'panel-body';
  body.setAttribute('role', 'tabpanel');
  body.tabIndex = 0;
  root.prepend(head, tablist, body);

  function select(id, {focus = false} = {}) {
    state.tab = id;
    // Rows are listed on these two tabs only: on another, what was open stops being followed.
    if (!['activity', 'workers'].includes(id) && state.open) closeRow();
    try { storage?.setItem(KEY, id); } catch { /* the choice is only for this view */ }
    drawTabs();
    drawBody();
    if (focus) tabs.find((tab) => tab.dataset.tab === id)?.focus();
  }
  function drawTabs() {
    for (const tab of tabs) {
      const on = tab.dataset.tab === state.tab;
      tab.setAttribute('aria-selected', String(on));
      tab.tabIndex = on ? 0 : -1;
    }
    body.setAttribute('aria-labelledby', `panel-tab-${state.tab}`);
  }

  function drawHeader() {
    const said = headline(state.health, state.scope === 'chat' && state.loaded ? state.rows : null);
    root.dataset.mood = said.mood;
    const src = `/assets/akita/sit-${said.mood}.png`;
    if (picture.getAttribute?.('src') !== src) picture.setAttribute('src', src);
    frame.classList.toggle('blinks', said.mood === 'idle');
    const nodes = [node('span', 'panel-dot'), node('span', '', said.text)];
    if (said.fix) nodes.push(button('link-button', 'Open settings', () => openSettings()));
    status.replaceChildren(...nodes);
    const waiting = said.mood === 'waiting';
    chip.dataset.mood = said.mood;
    chip.replaceChildren(Object.assign(node('img'), {src: `/assets/akita/face-${said.mood}.png`, alt: '', width: 24, height: 24}),
      node('span', '', [said.count || '', waiting ? '!' : ''].filter(Boolean).join(' · ')));
    chip.setAttribute('aria-label', `Activity: ${said.text}`);
  }

  // ── Activity ──

  const key = (row) => `${row.kind}:${row.id}`;
  const timeOf = (at) => new Date(at * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  // What the live view follows for a row: a running, waiting or held session streams; a queued one never started.
  const liveState = (row) => ({task_id: row.id, state: ['running', 'waiting', 'held'].includes(row.state) ? 'running' : 'done',
    started_at: row.state === 'queued' ? null : row.at});

  function icon(name) {
    const mark = node('span', 'activity-icon', GLYPHS[name] ?? GLYPHS.tool);
    mark.dataset.icon = name in GLYPHS ? name : 'tool';
    mark.setAttribute('aria-hidden', 'true');
    return mark;
  }
  // A worker's second line says its state in words, then what it does or how it ended.
  // A held worker says why in the server's words (`Waiting for a free model slot`).
  const workerLine = (row) => (row.state === 'held' && row.activity ? row.activity
    : [STATE_WORDS[row.state], row.state === 'running' ? row.activity : LIVE.has(row.state) ? '' : row.result].filter(Boolean).join(' · '));
  function rowNode(row, worker = false) {
    const item = node('li', 'activity-row');
    item.dataset.state = row.state;
    const open = state.open === key(row);
    const headButton = button('activity-head', '', () => toggle(row));
    headButton.dataset.focusKey = key(row);
    headButton.setAttribute('aria-expanded', String(open));
    const text = node('span', 'activity-text');
    text.append(node('strong', 'activity-title', row.title), node('span', 'activity-line', worker ? workerLine(row) : row.activity || row.result || ''));
    headButton.append(icon(row.icon), text);
    if (!LIVE.has(row.state)) headButton.append(node('time', 'activity-time', timeOf(row.at)));
    item.append(headButton);
    if (row.approvals?.length) item.append(approvalFold(row));
    if (open) item.append(detail(row));
    return item;
  }
  // The approvals a row asked for, folded under its result line: "2 approvals, 1 waiting".
  function approvalFold(row) {
    const asked = row.approvals;
    const waiting = asked.filter((item) => item.answer === 'waiting').length;
    const open = state.folds.has(key(row));
    const box = node('div', 'approval-fold');
    box.dataset.waiting = String(waiting > 0);
    const toggler = button('approval-toggle', `${asked.length} approval${asked.length === 1 ? '' : 's'}${waiting ? `, ${waiting} waiting` : ''}`, () => {
      if (!state.folds.delete(key(row))) state.folds.add(key(row));
      drawBody();
    });
    toggler.dataset.focusKey = `fold:${key(row)}`;
    toggler.setAttribute('aria-expanded', String(open));
    box.append(toggler);
    if (!open) return box;
    const list = node('ul', 'approval-list');
    for (const item of asked) {
      const entry = node('li', 'approval-item');
      entry.dataset.answer = item.answer === 'waiting' ? 'waiting' : item.answer.startsWith('allowed') ? 'allowed' : item.answer === 'denied' ? 'denied' : 'other';
      entry.append(node('span', 'approval-what', item.what), node('time', 'approval-time', timeOf(item.at)),
        node('span', 'approval-answer', ANSWERS[item.answer] ?? item.answer));
      if (item.answer === 'waiting') {
        // Only the gateway's own chats and tasks are answered here; the full card, with the content, is one link away.
        const mine = ['chat', 'task'].includes(row.kind);
        const busy = state.busy === `approval:${item.id}`;
        const choices = mine ? [button('outline-button', 'Allow once', () => answer(row, item, 'once')), button('outline-button', 'Deny', () => answer(row, item, 'deny'))] : [];
        for (const choice of choices) choice.disabled = busy;
        const review = node('a', 'activity-open', 'Review');
        review.href = `${HREF[row.kind]}${encodeURIComponent(row.id)}`;
        review.addEventListener('click', () => setSheet(false));
        const choose = node('span', 'approval-choices');
        choose.append(...choices, review);
        entry.append(choose);
      }
      list.append(entry);
    }
    box.append(list);
    if (state.notice && state.noticeFor === key(row)) box.append(node('p', 'panel-note', state.notice));
    return box;
  }
  // The same call the chat's approval card makes; the row is read again to show the answer.
  async function answer(row, item, choice) {
    if (state.busy) return;
    state.busy = `approval:${item.id}`;
    state.notice = '';
    drawBody();
    try {
      await api(`/approvals/${encodeURIComponent(item.id)}`, {body: {choice}});
    } catch (error) {
      state.noticeFor = key(row);
      state.notice = error?.status === 409 ? 'It was answered already.' : error?.message || 'Could not send the answer. Try again.';
    }
    state.busy = '';
    await refresh();
  }
  function stepNode(step) {
    const running = step.status === 'running';
    const failed = ['error', 'timeout', 'crashed'].includes(step.status);
    const item = node('li', 'activity-step');
    item.dataset.state = running ? 'running' : failed ? 'failed' : 'done';
    item.append(node('span', 'activity-mark', running ? '●' : failed ? '✗' : '✓'),
      node('span', 'activity-step-label', step.label ?? stepWords(step.name, step.arguments, !running)),
      node('span', 'activity-step-time', running ? 'now' : step.durationS == null ? '' : formatDuration(step.durationS)));
    item.firstChild.setAttribute('aria-hidden', 'true');
    return item;
  }
  function detail(row) {
    const box = node('div', 'activity-detail');
    const turn = live.state.view?.turns?.at(-1);
    const steps = turn ? turn.items.filter((item) => item.kind === 'tool') : [];
    if (live.state.loading && !turn) box.append(node('p', 'panel-note', 'Loading the steps…'));
    else if (steps.length) {
      if (steps.length > STEPS) box.append(node('p', 'panel-note', `${steps.length - STEPS} earlier steps`));
      const list = node('ol', 'activity-steps');
      list.append(...steps.slice(-STEPS).map(stepNode));
      // Between calls a running turn is thinking or waiting: its row's words close the list.
      if (LIVE.has(row.state) && row.activity && !steps.some((step) => step.status === 'running')) {
        list.append(stepNode({status: 'running', label: row.activity}));
      }
      box.append(list);
      const output = cleanOutput(String([...steps].reverse().find((step) => step.output)?.output ?? '')).trimEnd();
      if (output) box.append(node('pre', 'activity-output', output.split('\n').slice(-3).join('\n')));
    } else if (turn || row.state === 'queued') {
      box.append(node('p', 'panel-note', row.state === 'queued' ? 'Not started yet.' : LIVE.has(row.state) ? 'Thinking…' : 'Answered without steps.'));
    }
    if (live.state.error) box.append(node('p', 'panel-note', live.state.error));
    const actions = node('div', 'activity-actions');
    if (row.kind === 'task' && LIVE.has(row.state)) actions.append(...taskActions(row));
    const link = node('a', 'activity-open', row.state === 'waiting' ? 'Review' : OPEN[row.kind]);
    link.href = `${HREF[row.kind]}${encodeURIComponent(row.id)}`;
    link.addEventListener('click', () => setSheet(false));
    actions.append(link);
    box.append(actions);
    if (state.notice && state.noticeFor === key(row)) box.append(node('p', 'panel-note', state.notice));
    return box;
  }
  function taskActions(row) {
    const paused = row.state === 'paused';
    const busy = state.busy === key(row);
    const nodes = [
      button('outline-button', paused ? 'Resume' : 'Pause', () => act(row, {state: paused ? 'resume' : 'pause'})),
      button('outline-button', 'Stop', () => act(row, {state: 'stop'})),
    ];
    if (!paused && Number.isInteger(row.level) && row.level < LEVELS.length) {
      const menu = node('select', 'activity-level');
      menu.setAttribute('aria-label', 'Priority');
      LEVELS.forEach((word, level) => {
        const option = node('option', '', developer() ? `P${level} · ${word}` : word);
        option.value = String(level);
        option.selected = level === row.level;
        menu.append(option);
      });
      menu.addEventListener('change', () => act(row, {level: Number(menu.value)}));
      nodes.push(menu);
    }
    for (const control of nodes) control.disabled = busy;
    return nodes;
  }
  async function act(row, change) {
    if (state.busy) return;
    state.busy = key(row);
    state.notice = '';
    drawBody();
    try {
      await api(`/tasks/${encodeURIComponent(row.id)}`, {method: 'PATCH', body: change});
    } catch (error) {
      state.noticeFor = key(row);
      state.notice = error?.message || 'Could not change the task. Try again.';
    }
    state.busy = '';
    await refresh();
  }
  function toggle(row) {
    if (state.open === key(row)) closeRow();
    else { state.open = key(row); state.notice = ''; live.leave(); live.sync(liveState(row)); }
    drawBody();
  }
  function closeRow() {
    state.open = null;
    live.leave();
  }

  // The links to the full lists, for the kinds this scope has (all of them in developer mode).
  function footLinks(links) {
    const kinds = new Set(state.rows.map((row) => row.kind));
    return links.filter(([want]) => developer() || want.some((kind) => kinds.has(kind))).map(([, text, href]) => {
      const link = node('a', 'panel-link', text);
      link.href = href;
      link.addEventListener('click', () => setSheet(false));
      return link;
    });
  }
  // Today's finished rows, folded after SHOWN; the foot says how to see the rest.
  function dayNodes(rows, worker, links) {
    const now = rows.filter((row) => LIVE.has(row.state));
    const done = rows.filter((row) => !LIVE.has(row.state));
    const list = (these) => { const element = node('ul', 'activity-list'); element.append(...these.map((row) => rowNode(row, worker))); return element; };
    const nodes = [node('h3', 'panel-section', 'Now')];
    nodes.push(now.length ? list(now) : node('p', 'panel-calm', worker ? 'No worker is running' : 'Nothing running'));
    nodes.push(node('h3', 'panel-section', 'Today'));
    const shown = state.all ? done : done.slice(0, SHOWN);
    nodes.push(done.length ? list(shown) : node('p', 'panel-note', 'Nothing finished yet today.'));
    const foot = node('div', 'panel-foot');
    if (done.length > shown.length) foot.append(button('link-button', `Earlier today (${done.length - shown.length})`, () => { state.all = true; drawBody(); }));
    foot.append(...footLinks(links));
    if (foot.children.length) nodes.push(foot);
    if (state.more) nodes.push(node('p', 'panel-note', 'More ran today than this list shows.'));
    if (state.unavailable.length) nodes.push(node('p', 'panel-note', 'Part of today could not be read.'));
    return nodes;
  }
  function activityNodes() {
    if (!state.loaded) return [node('p', 'panel-note', state.error || 'Loading today…')];
    return dayNodes(state.rows, false, []);
  }
  // The workers of this scope: the tasks and the worker runs, running ones first. Terminal chats are not workers.
  function workersNodes() {
    if (!state.loaded) return [node('p', 'panel-note', state.error || 'Loading the workers…')];
    const workers = state.rows.filter((row) => row.kind === 'task' || row.kind === 'run');
    const links = [[['task'], 'All tasks', '#/tasks'], [['run', 'terminal'], 'All worker runs', '#/workers']];
    if (workers.length) return dayNodes(workers, true, links);
    const foot = node('div', 'panel-foot');
    foot.append(...footLinks(links));
    return [node('p', 'panel-soon', EMPTY_WORKERS[state.scope]), ...(foot.children.length ? [foot] : [])];
  }

  function drawBody() {
    // A priority being chosen is not drawn away.
    if (body.contains(doc.activeElement) && doc.activeElement?.tagName === 'SELECT') return;
    const own = {memory, scheduled}[state.tab];
    if (own) {
      // These tabs keep their own state (a note being reworded, a time being edited is not drawn away).
      if (body.firstChild !== own.root) { body.replaceChildren(own.root); own.show(); }
      return;
    }
    const focused = doc.activeElement?.dataset?.focusKey;
    const draw = {activity: activityNodes, workers: workersNodes};
    const nodes = draw[state.tab]();
    body.replaceChildren(...nodes);
    if (focused && body.contains(doc.activeElement) === false) {
      const again = [...body.querySelectorAll('button')].find((item) => item.dataset.focusKey === focused);
      again?.focus();
    }
  }

  function midnight() {
    const day = new Date(clock());
    day.setHours(0, 0, 0, 0);
    return Math.floor(day.getTime() / 1000);
  }
  async function refresh() {
    const mine = ++reads;
    const since = midnight();
    const asked = state.chat;
    try {
      const page = await api(`/activity?since=${since}${asked ? `&session_id=${encodeURIComponent(asked)}` : ''}`);
      if (mine !== reads) return;
      Object.assign(state, {rows: page.rows ?? [], more: Boolean(page.more), unavailable: page.unavailable ?? [],
        scope: page.scope === 'chat' ? 'chat' : 'all', loaded: true, error: '', fetchedAt: clock(), day: since});
    } catch {
      if (mine !== reads) return;
      if (!state.loaded) state.error = 'Could not read today’s activity. It is tried again shortly.';
      state.seen = null; // the next health poll asks again
    }
    const open = state.rows.find((row) => key(row) === state.open);
    if (open) live.sync(liveState(open));
    else if (state.open) closeRow();
    if (state.tab === 'memory') memory.reload();
    if (state.tab === 'scheduled') scheduled.reload();
    drawHeader();
    drawBody();
  }

  // The panel shows the chat that is open: when it changes, the old chat's rows go and the new chat's are read.
  function follow() {
    const now = chat();
    if (now === state.chat) return undefined;
    Object.assign(state, {chat: now, rows: [], more: false, unavailable: [], loaded: false, error: '', scope: 'all', notice: '', all: false});
    state.folds.clear();
    closeRow();
    drawHeader();
    drawBody();
    return refresh();
  }
  globalThis.addEventListener?.('hashchange', follow);

  function update(health) {
    state.health = health;
    const moved = follow();
    if (moved) return moved;
    drawHeader();
    const status = health?.status ?? {};
    const seen = JSON.stringify([status.running, status.tasks, status.items, status.waiting, status.recent?.id, status.recent?.outcome]);
    if (seen !== state.seen || clock() - state.fetchedAt > REFRESH_MS || midnight() !== state.day) {
      state.seen = seen;
      return refresh();
    }
    return undefined;
  }

  // ── The sheet below 900 px ──

  function setSheet(open) {
    if (state.sheet === open) return false;
    state.sheet = open;
    root.classList.toggle('sheet-open', open);
    root.style.transform = '';
    backdrop.hidden = !open;
    chip.setAttribute('aria-expanded', String(open));
    for (const element of cover()) element.inert = open;
    if (open) handle.focus();
    else if (root.contains(doc.activeElement) || doc.activeElement === doc.body) chip.focus();
    return true;
  }
  chip.addEventListener('click', () => setSheet(!state.sheet));
  // A window grown past the sheet's width has no chip to close it with: it closes, and the page is usable again.
  wide?.addEventListener?.('change', () => { if (wide.matches) setSheet(false); });
  backdrop.addEventListener('click', () => setSheet(false));
  let drag = null;
  handle.addEventListener('pointerdown', (event) => { drag = {y: event.clientY, moved: 0}; handle.setPointerCapture?.(event.pointerId); });
  handle.addEventListener('pointermove', (event) => {
    if (!drag) return;
    drag.moved = Math.max(0, event.clientY - drag.y);
    root.style.transform = `translateY(${drag.moved}px)`;
  });
  handle.addEventListener('pointerup', () => {
    const moved = drag?.moved ?? 0;
    drag = null;
    root.style.transform = '';
    if (moved > 60) setSheet(false);
  });
  chip.hidden = false;
  drawHeader();
  drawTabs();
  drawBody();

  return {update, refresh, follow, select, state, closeSheet: () => setSheet(false), openSheet: () => setSheet(true)};
}

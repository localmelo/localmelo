/**
 * The panel's Memory tab: what melo remembers, to read, reword and delete.
 *
 * `GET /api/memory/records` lists the notes newest first, each with where it came from (a chat, or "a
 * deleted chat": a note outlives its chat). A note is reworded with `PATCH /api/memory/records/{id}` and
 * deleted with `DELETE` on the same path; "Forget everything" names what goes before it calls
 * `DELETE /api/memory/records?all=true`; the switch is `PUT /api/memory`. The search box filters the loaded
 * list. `root` is mounted in the panel's body while the tab is open; `show()` reads the notes again.
 * `askAboutNotes` is the question a deleted chat with notes asks.
 */
export const TOPICS = {about: 'About you', preferences: 'Preferences', people: 'People', projects: 'Projects', decisions: 'Decisions'};
const EMPTY = 'Nothing remembered yet.';
const HOW = 'melo keeps a note when you tell it something about yourself, such as “I prefer short answers”, or ask it to remember something. Notes show up here, and you can reword or delete them.';
const NEVER = 'Never kept: passwords, keys, codes and page contents.';

const capital = (text) => String(text ?? '').charAt(0).toUpperCase() + String(text ?? '').slice(1);
const plural = (count, one, many) => `${count} ${count === 1 ? one : many}`;

/** How the search finds a note: every word typed is in its text, topic or source. */
export function matches(row, query) {
  const words = String(query ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  const hay = `${row.text} ${TOPICS[row.topic] ?? ''} ${row.from ?? ''}`.toLowerCase();
  return words.every((word) => hay.includes(word));
}

/** What melo says about how it searches: by meaning with the embedding model, else by words only. */
export function searchLine(on, model, retriever = 'auto') {
  if (!on) return 'Off. Notes stay on this computer but are not used in chats.';
  if (retriever === 'bm25') return 'Searching by words only, as the memory retriever setting says.';
  if (retriever === 'llm') return 'Searching by words, and your chat model picks the notes that matter.';
  return model ? `Searching by meaning with ${model}, and by words.` : 'Searching by words only. An embedding model in Settings adds search by meaning.';
}

/** What "Forget everything" will remove, in the confirm's words. */
export function forgetLine(notes, turns) {
  const parts = [plural(notes, 'note', 'notes')];
  if (turns) parts.push(`melo's search index of ${plural(turns, 'chat turn', 'chat turns')}`);
  return `This removes ${parts.join(' and ')}. Your chats stay. It cannot be undone.`;
}

export function setupMemoryTab({doc, api, developer = () => false}) {
  const state = {loaded: false, error: '', on: true, rows: [], documents: 0, model: '', retriever: 'auto', query: '', edit: null, draft: '', fault: '',
    ask: null, forgetting: false, notice: '', busy: false, focus: false};
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
  const root = node('div', 'memory-tab');
  const head = node('div', 'memory-head');
  const search = node('input', 'memory-search');
  search.type = 'search';
  search.placeholder = 'Search what melo remembers';
  search.setAttribute('aria-label', 'Search what melo remembers');
  search.addEventListener('input', () => { state.query = search.value; drawList(); });
  const searchBox = node('div', 'memory-searchbox');
  searchBox.append(search);
  const list = node('div', 'memory-rows');
  const foot = node('div', 'memory-foot');
  root.append(head, searchBox, list, foot);

  async function load({quiet = false} = {}) {
    const mine = ++reads;
    try {
      const page = await api('/memory/records');
      if (mine !== reads) return;
      Object.assign(state, {rows: page.records ?? [], on: page.on !== false, documents: page.documents ?? 0,
        model: page.embedding_model ?? '', retriever: page.retriever ?? 'auto', loaded: true, error: ''});
    } catch (error) {
      if (mine !== reads || quiet) return;
      state.error = error?.message || 'Could not read what melo remembers.';
    }
    draw();
  }
  async function call(path, options, failure) {
    if (state.busy) return null;
    state.busy = true;
    state.notice = '';
    try {
      return await api(path, options);
    } catch (error) {
      state.notice = error?.message || failure;
      return null;
    } finally {
      state.busy = false;
    }
  }

  function drawHead() {
    if (!state.loaded) { head.replaceChildren(); return; }
    const row = node('div', 'memory-switchrow');
    const toggle = node('button', 'memory-switch');
    toggle.type = 'button';
    toggle.setAttribute('role', 'switch');
    toggle.setAttribute('aria-checked', String(state.on));
    toggle.setAttribute('aria-label', 'Remember across chats');
    toggle.append(node('span', 'memory-knob'));
    toggle.addEventListener('click', async () => {
      const done = await call('/memory', {method: 'PUT', body: {on: !state.on}}, 'Could not change this. Try again.');
      if (done) state.on = done.on !== false;
      draw();
    });
    row.append(node('strong', '', 'Remember across chats'), toggle);
    head.replaceChildren(row, node('p', 'memory-line', searchLine(state.on, state.model, state.retriever)));
  }
  function noteNode(row) {
    const item = node('li', 'memory-note');
    item.dataset.id = String(row.id);
    const editing = state.edit === row.id;
    if (editing) {
      const area = node('textarea', 'memory-edit');
      area.rows = 3;
      area.maxLength = 300;
      area.value = state.draft;
      area.setAttribute('aria-label', 'The note, in your words');
      area.addEventListener('input', () => { state.draft = area.value; });
      area.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') { event.stopPropagation(); state.edit = null; draw(); }
      });
      item.append(area);
      if (state.fault) item.append(node('p', 'memory-fault', state.fault));
      const buttons = node('div', 'memory-actions');
      buttons.append(button('outline-button', 'Save', () => save(row)), button('quiet-button', 'Cancel', () => { state.edit = null; draw(); }));
      item.append(buttons);
      return item;
    }
    item.append(node('p', 'memory-text', row.text));
    const meta = node('p', 'memory-meta');
    if (TOPICS[row.topic]) meta.append(node('span', 'memory-topic', TOPICS[row.topic]));
    meta.append(node('span', row.chat_gone ? 'memory-from gone' : 'memory-from', capital(row.from) + (row.edited ? ' · edited' : '')));
    item.append(meta);
    const actions = node('div', 'memory-actions');
    if (state.ask === row.id) {
      actions.append(node('span', 'memory-ask', 'Forget this note?'),
        button('quiet-button', 'Keep it', () => { state.ask = null; draw(); }),
        button('outline-button danger', 'Forget it', () => forget(row)));
    } else {
      actions.append(button('quiet-button', 'Edit', () => { Object.assign(state, {edit: row.id, draft: row.text, fault: '', focus: true}); draw(); }, `Edit: ${row.text}`),
        button('quiet-button', 'Delete', () => { state.ask = row.id; draw(); }, `Delete: ${row.text}`));
    }
    item.append(actions);
    return item;
  }
  function drawList() {
    if (!state.loaded) { list.replaceChildren(node('p', 'panel-note', state.error || 'Reading what melo remembers…')); return; }
    const shown = state.rows.filter((row) => matches(row, state.query));
    if (!state.rows.length) {
      const empty = node('div', 'memory-empty');
      empty.append(node('p', 'memory-empty-title', EMPTY), node('p', 'panel-note', HOW));
      list.replaceChildren(empty);
      return;
    }
    if (!shown.length) { list.replaceChildren(node('p', 'panel-note', `No note matches “${state.query.trim()}”.`)); return; }
    const notes = node('ul', 'memory-notes-list');
    notes.append(...shown.map(noteNode));
    list.replaceChildren(notes);
    if (state.focus) { state.focus = false; list.querySelector?.('textarea')?.focus?.(); }
  }
  function drawFoot() {
    const nodes = [node('p', 'memory-never', NEVER)];
    if (state.notice) nodes.push(node('p', 'memory-fault', state.notice));
    if (state.forgetting) {
      const box = node('div', 'memory-forget');
      box.append(node('strong', '', 'Forget everything?'), node('p', '', forgetLine(state.rows.length, state.documents)));
      const actions = node('div', 'memory-actions');
      actions.append(button('outline-button', 'Keep everything', () => { state.forgetting = false; draw(); }),
        button('outline-button danger', 'Forget everything', forgetAll));
      box.append(actions);
      nodes.push(box);
    } else if (state.rows.length || state.documents) {
      const links = node('div', 'panel-foot');
      links.append(button('link-button danger', 'Forget everything…', () => { state.forgetting = true; draw(); }));
      if (developer()) links.append(button('link-button', 'Rebuild search index', reindex));
      nodes.push(links);
    }
    foot.replaceChildren(...nodes);
  }
  function draw() {
    drawHead();
    drawList();
    drawFoot();
    searchBox.hidden = !state.rows.length;
  }

  async function save(row) {
    const done = await call(`/memory/records/${encodeURIComponent(row.id)}`, {method: 'PATCH', body: {text: state.draft}}, 'Could not save the note. Try again.');
    if (done) { state.rows = state.rows.map((item) => (item.id === row.id ? done : item)); state.edit = null; state.fault = ''; }
    else state.fault = state.notice;
    state.notice = '';
    draw();
  }
  async function forget(row) {
    const done = await call(`/memory/records/${encodeURIComponent(row.id)}`, {method: 'DELETE'}, 'Could not forget the note. Try again.');
    if (done !== null) state.rows = state.rows.filter((item) => item.id !== row.id);
    state.ask = null;
    draw();
  }
  async function forgetAll() {
    const done = await call('/memory/records?all=true', {method: 'DELETE'}, 'Could not forget everything. Try again.');
    if (done) {
      Object.assign(state, {rows: [], documents: 0, forgetting: false, edit: null, ask: null, query: '', notice: ''});
      search.value = '';
    }
    draw();
  }
  async function reindex() {
    const done = await call('/memory/reindex', {method: 'POST', body: {}}, 'Could not rebuild the index. Try again.');
    if (done) { state.notice = `Rebuilt: ${plural(done.turns, 'chat turn', 'chat turns')}.`; await load({quiet: true}); }
    draw();
  }

  draw();
  return {root, state, show: () => load(), reload: () => (state.edit === null && !state.busy ? load({quiet: true}) : undefined)};
}

/**
 * The question a deleted chat with notes asks: keep them (the first answer) or forget them too.
 * Resolves to `keep`, `forget`, or `cancel` (Escape, or "Don't delete the chat").
 */
export function askAboutNotes({doc, count, title = ''}) {
  return new Promise((resolve) => {
    const dialog = doc.createElement('dialog');
    dialog.className = 'notes-ask';
    dialog.setAttribute('aria-labelledby', 'notes-ask-title');
    const heading = doc.createElement('h2');
    heading.id = 'notes-ask-title';
    heading.textContent = `Also forget the ${plural(count, 'note', 'notes')} from this chat?`;
    const about = doc.createElement('p');
    about.textContent = `${title ? `“${title}” is deleted either way. ` : ''}Notes are facts melo learned about you there; they can stay after the chat is gone.`;
    const answer = (value) => { dialog.close?.(); dialog.remove?.(); resolve(value); };
    const choose = (text, value, className) => {
      const item = doc.createElement('button');
      item.type = 'button';
      item.className = className;
      item.textContent = text;
      item.addEventListener('click', () => answer(value));
      return item;
    };
    const keep = choose('Keep the notes', 'keep', 'outline-button');
    dialog.append(heading, about, keep, choose(`Forget the ${plural(count, 'note', 'notes')} too`, 'forget', 'outline-button danger'),
      choose('Don’t delete the chat', 'cancel', 'quiet-button'));
    dialog.addEventListener('cancel', (event) => { event.preventDefault(); answer('cancel'); });
    (doc.body ?? doc.documentElement).append(dialog);
    dialog.showModal?.();
    keep.focus?.();
  });
}

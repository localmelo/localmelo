/**
 * The panel's Memory tab: what melo remembers, to read, reword and delete.
 *
 * `GET /api/memory/records` lists the notes newest first, each with where it came from (a chat, or "a
 * deleted chat": a note outlives its chat). A note is reworded with `PATCH /api/memory/records/{id}` and
 * deleted with `DELETE` on the same path; "Forget everything" names what goes before it calls
 * `DELETE /api/memory/records?all=true`; the switch is `PUT /api/memory`. The search box filters the loaded
 * list. `root` is mounted in the panel's body while the tab is open; `show()` reads the notes again.
 * `askAboutNotes` is the question a deleted chat with notes asks. "From other assistants" (import-card.mjs)
 * looks for notes other assistants keep, brings in the checked files, and takes a batch back.
 */
import {t} from './i18n.mjs';
import {importSection} from './import-card.mjs';

export const TOPICS = ['about', 'preferences', 'people', 'projects', 'decisions'];

/** A topic's name in the page's language, or '' for a topic the page does not know. */
export function topicName(topic) {
  return TOPICS.includes(topic) ? t(`memoryTab.topic.${topic}`) : '';
}

const capital = (text) => String(text ?? '').charAt(0).toUpperCase() + String(text ?? '').slice(1);

/** How the search finds a note: every word typed is in its text, topic or source. */
export function matches(row, query) {
  const words = String(query ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  const hay = `${row.text} ${topicName(row.topic)} ${row.from ?? ''}`.toLowerCase();
  return words.every((word) => hay.includes(word));
}

/** What melo says about how it searches: by meaning with the embedding model, else by words only. */
export function searchLine(on, model, retriever = 'auto') {
  if (!on) return t('memoryTab.search.off');
  if (retriever === 'bm25') return t('memoryTab.search.words');
  if (retriever === 'llm') return t('memoryTab.search.model');
  return model ? t('memoryTab.search.meaning', {model}) : t('memoryTab.search.wordsOnly');
}

/** What "Forget everything" will remove, in the confirm's words. */
export function forgetLine(notes, turns) {
  const what = t('memoryTab.notes', {n: notes});
  return turns
    ? t('memoryTab.forgetLineIndex', {notes: what, index: t('memoryTab.index', {n: turns})})
    : t('memoryTab.forgetLine', {notes: what});
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
  search.placeholder = t('memoryTab.searchBox');
  search.setAttribute('aria-label', t('memoryTab.searchBox'));
  search.addEventListener('input', () => { state.query = search.value; drawList(); });
  const searchBox = node('div', 'memory-searchbox');
  searchBox.append(search);
  const list = node('div', 'memory-rows');
  const foot = node('div', 'memory-foot');
  const importing = importSection({doc, api, rows: () => state.rows, reload: () => load({quiet: true}), redraw: () => draw()});
  root.append(head, searchBox, list, importing.root, foot);

  async function load({quiet = false} = {}) {
    const mine = ++reads;
    try {
      const page = await api('/memory/records');
      if (mine !== reads) return;
      Object.assign(state, {rows: page.records ?? [], on: page.on !== false, documents: page.documents ?? 0,
        model: page.embedding_model ?? '', retriever: page.retriever ?? 'auto', loaded: true, error: ''});
    } catch (error) {
      if (mine !== reads || quiet) return;
      state.error = error?.message || t('memoryTab.readFailed');
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
    toggle.setAttribute('aria-label', t('memoryTab.switch'));
    toggle.append(node('span', 'memory-knob'));
    toggle.addEventListener('click', async () => {
      const done = await call('/memory', {method: 'PUT', body: {on: !state.on}}, t('memoryTab.switchFailed'));
      if (done) state.on = done.on !== false;
      draw();
    });
    row.append(node('strong', '', t('memoryTab.switch')), toggle);
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
      area.setAttribute('aria-label', t('memoryTab.editLabel'));
      area.addEventListener('input', () => { state.draft = area.value; });
      area.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') { event.stopPropagation(); state.edit = null; draw(); }
      });
      item.append(area);
      if (state.fault) item.append(node('p', 'memory-fault', state.fault));
      const buttons = node('div', 'memory-actions');
      buttons.append(button('outline-button', t('memoryTab.save'), () => save(row)), button('quiet-button', t('memoryTab.cancel'), () => { state.edit = null; draw(); }));
      item.append(buttons);
      return item;
    }
    item.append(node('p', 'memory-text', row.text));
    const meta = node('p', 'memory-meta');
    if (topicName(row.topic)) meta.append(node('span', 'memory-topic', topicName(row.topic)));
    meta.append(node('span', row.chat_gone ? 'memory-from gone' : 'memory-from', capital(row.from) + (row.edited ? ` · ${t('memoryTab.edited')}` : '')));
    item.append(meta);
    const actions = node('div', 'memory-actions');
    if (state.ask === row.id) {
      actions.append(node('span', 'memory-ask', t('memoryTab.forgetOne')),
        button('quiet-button', t('memoryTab.keepIt'), () => { state.ask = null; draw(); }),
        button('outline-button danger', t('memoryTab.forgetIt'), () => forget(row)));
    } else {
      actions.append(button('quiet-button', t('memoryTab.edit'), () => { Object.assign(state, {edit: row.id, draft: row.text, fault: '', focus: true}); draw(); }, t('memoryTab.editNote', {note: row.text})),
        button('quiet-button', t('memoryTab.delete'), () => { state.ask = row.id; draw(); }, t('memoryTab.deleteNote', {note: row.text})));
    }
    item.append(actions);
    return item;
  }
  function drawList() {
    if (!state.loaded) { list.replaceChildren(node('p', 'panel-note', state.error || t('memoryTab.reading'))); return; }
    const shown = state.rows.filter((row) => matches(row, state.query));
    if (!state.rows.length) {
      const empty = node('div', 'memory-empty');
      empty.append(node('p', 'memory-empty-title', t('memoryTab.empty')), node('p', 'panel-note', t('memoryTab.how')));
      list.replaceChildren(empty);
      return;
    }
    if (!shown.length) { list.replaceChildren(node('p', 'panel-note', t('memoryTab.noMatch', {query: state.query.trim()}))); return; }
    const notes = node('ul', 'memory-notes-list');
    notes.append(...shown.map(noteNode));
    list.replaceChildren(notes);
    if (state.focus) { state.focus = false; list.querySelector?.('textarea')?.focus?.(); }
  }
  function drawFoot() {
    const nodes = [node('p', 'memory-never', t('memoryTab.never'))];
    if (state.notice) nodes.push(node('p', 'memory-fault', state.notice));
    if (state.forgetting) {
      const box = node('div', 'memory-forget');
      box.append(node('strong', '', t('memoryTab.forgetAllAsk')), node('p', '', forgetLine(state.rows.length, state.documents)));
      const actions = node('div', 'memory-actions');
      actions.append(button('outline-button', t('memoryTab.keepAll'), () => { state.forgetting = false; draw(); }),
        button('outline-button danger', t('memoryTab.forgetAll'), forgetAll));
      box.append(actions);
      nodes.push(box);
    } else if (state.rows.length || state.documents) {
      const links = node('div', 'panel-foot');
      links.append(button('link-button danger', t('memoryTab.forgetAllLink'), () => { state.forgetting = true; draw(); }));
      if (developer()) links.append(button('link-button', t('memoryTab.reindex'), reindex));
      nodes.push(links);
    }
    foot.replaceChildren(...nodes);
  }
  function draw() {
    drawHead();
    drawList();
    drawFoot();
    importing.root.hidden = !state.loaded || !state.on;
    if (!importing.root.hidden) importing.draw();
    searchBox.hidden = !state.rows.length;
  }

  async function save(row) {
    const done = await call(`/memory/records/${encodeURIComponent(row.id)}`, {method: 'PATCH', body: {text: state.draft}}, t('memoryTab.saveFailed'));
    if (done) { state.rows = state.rows.map((item) => (item.id === row.id ? done : item)); state.edit = null; state.fault = ''; }
    else state.fault = state.notice;
    state.notice = '';
    draw();
  }
  async function forget(row) {
    const done = await call(`/memory/records/${encodeURIComponent(row.id)}`, {method: 'DELETE'}, t('memoryTab.forgetFailed'));
    if (done !== null) state.rows = state.rows.filter((item) => item.id !== row.id);
    state.ask = null;
    draw();
  }
  async function forgetAll() {
    const done = await call('/memory/records?all=true', {method: 'DELETE'}, t('memoryTab.forgetAllFailed'));
    if (done) {
      Object.assign(state, {rows: [], documents: 0, forgetting: false, edit: null, ask: null, query: '', notice: ''});
      search.value = '';
    }
    draw();
  }
  async function reindex() {
    const done = await call('/memory/reindex', {method: 'POST', body: {}}, t('memoryTab.reindexFailed'));
    if (done) { state.notice = t('memoryTab.rebuilt', {n: done.turns}); await load({quiet: true}); }
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
    heading.textContent = t('memoryTab.ask.title', {n: count});
    const about = doc.createElement('p');
    about.textContent = title ? t('memoryTab.ask.aboutTitled', {title}) : t('memoryTab.ask.about');
    const answer = (value) => { dialog.close?.(); dialog.remove?.(); resolve(value); };
    const choose = (text, value, className) => {
      const item = doc.createElement('button');
      item.type = 'button';
      item.className = className;
      item.textContent = text;
      item.addEventListener('click', () => answer(value));
      return item;
    };
    const keep = choose(t('memoryTab.ask.keep'), 'keep', 'outline-button');
    dialog.append(heading, about, keep, choose(t('memoryTab.ask.forget', {n: count}), 'forget', 'outline-button danger'),
      choose(t('memoryTab.ask.cancel'), 'cancel', 'quiet-button'));
    dialog.addEventListener('cancel', (event) => { event.preventDefault(); answer('cancel'); });
    (doc.body ?? doc.documentElement).append(dialog);
    dialog.showModal?.();
    keep.focus?.();
  });
}

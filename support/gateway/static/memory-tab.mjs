/**
 * The panel's Memory tab: what melo remembers, to read, reword and delete.
 *
 * `GET /api/memory/records` lists the notes newest first, each with where it came from (a chat, or "a
 * deleted chat": a note outlives its chat). A note is reworded with `PATCH /api/memory/records/{id}` and
 * deleted with `DELETE` on the same path; "Forget everything" names what goes before it calls
 * `DELETE /api/memory/records?all=true`; the switch is `PUT /api/memory`. The search box filters the loaded
 * list. `GET /api/memory/items` adds the memory items melo wrote from the chats to the same list (nothing
 * when there are none), each marked as melo's, with its chat as a link. An item is deleted at once with
 * `DELETE /api/memory/items/{id}`; its place then holds "Deleted … Undo" (`POST …/restore`) for as long as
 * the server says it can be put back. A replaced item shows in developer mode only. Item text is always
 * set as text. `root` is mounted in the panel's body while the tab is open; `show()` reads the notes again.
 * `askAboutNotes` is the question a deleted chat with notes asks. "From other assistants" (import-card.mjs)
 * looks for notes other assistants keep, brings in the checked files, and takes a batch back.
 */
import {t} from './i18n.mjs';
import {maker} from './dom.mjs';
import {importSection} from './import-card.mjs';

export const TOPICS = ['about', 'preferences', 'people', 'projects', 'decisions'];
const ITEM_KINDS = ['profile', 'semantic', 'episodic'];

/** A topic's name in the page's language, or '' for a topic the page does not know. */
export function topicName(topic) {
  return TOPICS.includes(topic) ? t(`memoryTab.topic.${topic}`) : '';
}

const capital = (text) => String(text ?? '').charAt(0).toUpperCase() + String(text ?? '').slice(1);

/** How the search finds a note: every word typed is in its text, topic or source. */
export function matches(row, query) {
  const words = String(query ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  const hay = `${row.title ?? ''} ${row.text} ${topicName(row.topic)} ${row.from ?? ''}`.toLowerCase();
  return words.every((word) => hay.includes(word));
}

/**
 * What melo says about how it searches. The line follows the way that answers now (`answering`), and says why
 * when that is not what the `retriever` setting names: no embedding model sends `auto` to the chat model's
 * choice, or to words alone when no chat model is set up either.
 */
export function searchLine(on, model, retriever = 'auto', answering = 'bm25') {
  if (!on) return t('memoryTab.search.off');
  if (answering === 'llm') return t(retriever === 'llm' ? 'memoryTab.search.model' : 'memoryTab.search.noEmbedding');
  if (answering === 'embedding' && model) return t('memoryTab.search.meaning', {model});
  return t(retriever === 'bm25' ? 'memoryTab.search.words' : 'memoryTab.search.wordsOnly');
}

/** What "Forget everything" will remove, in the confirm's words. */
export function forgetLine(notes, turns, items = 0) {
  const what = items ? t('memoryTab.notesAndItems', {notes: t('memoryTab.notes', {n: notes}), items: t('memoryTab.items', {n: items})})
    : t('memoryTab.notes', {n: notes});
  return turns
    ? t('memoryTab.forgetLineIndex', {notes: what, index: t('memoryTab.index', {n: turns})})
    : t('memoryTab.forgetLine', {notes: what});
}

export function setupMemoryTab({doc, api, developer = () => false, clock = Date.now}) {
  const state = {loaded: false, error: '', on: true, rows: [], items: [], removed: [], documents: 0, model: '', retriever: 'auto', answering: 'bm25', query: '', edit: null,
    draft: '', fault: '', ask: null, forgetting: false, notice: '', busy: false, focus: false};
  let reads = 0;
  const {node, button} = maker(doc);
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
      const [page, written] = await Promise.all([api('/memory/records'), api('/memory/items')]);
      if (mine !== reads) return;
      Object.assign(state, {rows: page.records ?? [], items: written.items ?? [], on: page.on !== false, documents: page.documents ?? 0,
        model: page.embedding_model ?? '', retriever: page.retriever ?? 'auto', answering: page.answering ?? 'bm25', loaded: true, error: ''});
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
    head.replaceChildren(row, node('p', 'memory-line', searchLine(state.on, state.model, state.retriever, state.answering)));
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
      actions.append(button('quiet-button', t('memoryTab.edit'), () => { Object.assign(state, {edit: row.id, draft: row.text, fault: '', focus: true}); draw(); }, {label: t('memoryTab.editNote', {note: row.text})}),
        button('quiet-button', t('memoryTab.delete'), () => { state.ask = row.id; draw(); }, {label: t('memoryTab.deleteNote', {note: row.text})}));
    }
    item.append(actions);
    return item;
  }
  /** What melo wrote from a chat: its title and text as text, whose chat it is, and what the person can act on. */
  function itemNode(row) {
    const item = node('li', 'memory-note memory-item');
    item.dataset.item = String(row.id);
    if (row.title) item.append(node('p', 'memory-item-title', row.title));
    item.append(node('p', 'memory-text', row.text));
    const meta = node('p', 'memory-meta');
    meta.append(node('span', 'memory-topic memory-by', t('memoryTab.item.by')));
    if (developer() && ITEM_KINDS.includes(row.kind)) meta.append(node('span', 'memory-topic', t(`memoryTab.item.kind.${row.kind}`)));
    const from = capital(row.from);
    if (row.link && !row.chat_gone) {
      const chat = node('a', 'memory-from', from);
      chat.href = row.link;
      meta.append(chat);
    } else meta.append(node('span', row.chat_gone ? 'memory-from gone' : 'memory-from', from));
    // A disagreement is the one status a person can settle (delete the wrong item); a replaced item is history.
    if (row.status === 'conflict') meta.append(node('span', 'memory-status', t('memoryTab.item.conflict')));
    if (row.status === 'replaced') meta.append(node('span', 'memory-status', t(row.replaced_by ? 'memoryTab.item.replaced' : 'memoryTab.item.replacedGone')));
    item.append(meta);
    const actions = node('div', 'memory-actions');
    actions.append(button('quiet-button', t('memoryTab.delete'), () => removeItem(row), {label: t('memoryTab.deleteNote', {note: row.title || row.text})}));
    item.append(actions);
    return item;
  }
  /** The place of an item just deleted: what went, and Undo while the server can still put it back. */
  function goneNode(gone) {
    const item = node('li', 'memory-note memory-gone');
    item.dataset.item = String(gone.row.id);
    const line = node('p', 'memory-gone-line');
    line.append(node('span', 'memory-gone-text', t('memoryTab.item.deleted', {item: gone.row.title || gone.row.text})),
      button('quiet-button', t('memory.undo'), () => putBack(gone)));
    item.append(line);
    return item;
  }
  /** The notes and the items worth showing in one list, newest first; with no items it is the notes as sent. */
  function listed() {
    state.removed = state.removed.filter((gone) => clock() - gone.at < gone.window);
    const items = state.items.filter((row) => row.status !== 'replaced' || developer());
    if (!items.length && !state.removed.length) return state.rows.map((row) => ({draw: noteNode, row}));
    return [...state.rows.map((row) => ({draw: noteNode, row, at: row.created_at})),
      ...items.map((row) => ({draw: itemNode, row, at: row.created_at})),
      ...state.removed.map((gone) => ({draw: goneNode, row: gone, at: gone.row.created_at, always: true}))].sort((a, b) => b.at - a.at);
  }
  function drawList() {
    if (!state.loaded) { list.replaceChildren(node('p', 'panel-note', state.error || t('memoryTab.reading'))); return; }
    const entries = listed();
    const shown = entries.filter((entry) => entry.always || matches(entry.row, state.query));
    if (!entries.length) {
      const empty = node('div', 'memory-empty');
      empty.append(node('p', 'memory-empty-title', t('memoryTab.empty')), node('p', 'panel-note', t('memoryTab.how')));
      list.replaceChildren(empty);
      return;
    }
    if (!shown.length) { list.replaceChildren(node('p', 'panel-note', t('memoryTab.noMatch', {query: state.query.trim()}))); return; }
    const notes = node('ul', 'memory-notes-list');
    notes.append(...shown.map((entry) => entry.draw(entry.row)));
    list.replaceChildren(notes);
    if (state.focus) { state.focus = false; list.querySelector?.('textarea')?.focus?.(); }
  }
  function drawFoot() {
    const nodes = [node('p', 'memory-never', t('memoryTab.never'))];
    if (state.notice) nodes.push(node('p', 'memory-fault', state.notice));
    if (state.forgetting) {
      const box = node('div', 'memory-forget');
      box.append(node('strong', '', t('memoryTab.forgetAllAsk')), node('p', '', forgetLine(state.rows.length, state.documents, state.items.length)));
      const actions = node('div', 'memory-actions');
      actions.append(button('outline-button', t('memoryTab.keepAll'), () => { state.forgetting = false; draw(); }),
        button('outline-button danger', t('memoryTab.forgetAll'), forgetAll));
      box.append(actions);
      nodes.push(box);
    } else if (state.rows.length || state.items.length || state.documents) {
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
    searchBox.hidden = !state.rows.length && !state.items.length;
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
  async function removeItem(row) {
    const done = await call(`/memory/items/${encodeURIComponent(row.id)}`, {method: 'DELETE'}, t('memoryTab.item.deleteFailed'));
    if (done) {
      state.items = state.items.filter((item) => item.id !== row.id);
      state.removed.push({row, at: clock(), window: (Number(done.undo_seconds) || 0) * 1000});
    }
    draw();
  }
  async function putBack(gone) {
    const done = await call(`/memory/items/${encodeURIComponent(gone.row.id)}/restore`, {method: 'POST', body: {}}, t('memoryTab.item.undoFailed'));
    if (done !== null) {
      state.removed = state.removed.filter((other) => other !== gone);
      await load({quiet: true});
    }
    draw();
  }
  async function forgetAll() {
    const done = await call('/memory/records?all=true', {method: 'DELETE'}, t('memoryTab.forgetAllFailed'));
    if (done) {
      Object.assign(state, {rows: [], items: [], removed: [], documents: 0, forgetting: false, edit: null, ask: null, query: '', notice: ''});
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
  /** `/memory import`: the notes are read, then "Look for notes" is pressed; false when memory is off (no section). */
  async function lookForNotes() {
    if (!state.loaded) await load();
    if (!state.on) return false;
    await importing.look();
    return true;
  }

  return {root, state, show: () => load(), lookForNotes, reload: () => (state.edit === null && !state.busy ? load({quiet: true}) : undefined)};
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

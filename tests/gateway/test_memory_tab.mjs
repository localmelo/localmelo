/**
 * The panel's Memory tab (memory-tab.mjs) in a DOM double: the notes newest first with where each came from,
 * the search box, rewording and deleting a note, "Forget everything" with a confirm that names what goes, the
 * switch, the empty state, and the question a deleted chat with notes asks. Memory items melo wrote from chats
 * share the list: marked, linked to their chat, deleted at once with an Undo, and always shown as text.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {askAboutNotes, forgetLine, matches, searchLine, setupMemoryTab} from '../../support/gateway/static/memory-tab.mjs';

class Element {
  constructor(tag) {
    Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, handlers: {}, dataset: {}, textContent: '',
      className: '', hidden: false, disabled: false, value: '', parent: null});
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  append(...nodes) { for (const item of nodes) { item.parent = this; this.children.push(item); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  dispatch(type, event = {}) { for (const fn of this.handlers[type] ?? []) fn({target: this, stopPropagation() {}, preventDefault() {}, ...event}); }
  click() { if (!this.disabled) this.dispatch('click'); }
  focus() { doc.activeElement = this; }
  remove() { this.removed = true; }
  close() { this.closed = true; }
  showModal() { this.opened = true; }
  querySelector(selector) { return this.findAll((node) => node.tagName === selector.toUpperCase())[0] ?? null; }
  text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' ').replace(/\s+/g, ' ').trim(); }
  findAll(test) { return [...(test(this) ? [this] : []), ...this.children.flatMap((child) => child.findAll(test))]; }
  button(label) { return this.findAll((node) => node.tagName === 'BUTTON' && node.text() === label)[0] ?? null; }
}
const doc = {activeElement: null, createElement: (tag) => new Element(tag)};
doc.body = new Element('body');

const settle = () => new Promise((resolve) => setImmediate(resolve));
const note = (id, text, extra = {}) => ({id, text, topic: 'preferences', quote: '', source: '', from: 'from chat “Planning”, Oct 30',
  chat_gone: false, edited: false, session_id: 's1', created_at: id, ...extra});
const item = (id, title, text, extra = {}) => ({id, title, text, kind: 'semantic', status: 'active', replaced_by: null, conflicts: [],
  from: 'from chat “Garden”, Oct 31', chat_gone: false, link: '#/s/s2', session_id: 's2', created_at: id, ...extra});
const page = (records, extra = {}) => ({on: true, records, documents: 4, embedding_model: '', retriever: 'auto', answering: 'bm25', last_error: '', ...extra});

function harness({records = [], items = [], developer = false, fail = {}, extra = {}} = {}) {
  const h = {calls: [], records, items, trash: [], now: 0};
  const api = async (path, options = {}) => {
    const method = options.method ?? 'GET';
    h.calls.push([method, path, options.body]);
    if (fail[`${method} ${path}`]) throw new Error(fail[`${method} ${path}`]);
    if (path === '/memory/records' && method === 'GET') return page(h.records, extra);
    if (path === '/memory/items') return {items: h.items};
    if (method === 'DELETE' && path.startsWith('/memory/items/')) {
      const id = Number(path.split('/').pop());
      h.trash = h.items.filter((row) => row.id === id);
      h.items = h.items.filter((row) => row.id !== id);
      return {undo_seconds: 600};
    }
    if (method === 'POST' && path.endsWith('/restore')) { h.items = [...h.items, ...h.trash]; return {}; }
    if (method === 'PATCH') return {...h.records.find((item) => `/memory/records/${item.id}` === path), text: options.body.text, edited: true};
    if (path === '/memory/records?all=true') return {notes: h.records.length, turns: 4};
    if (path === '/memory') return {on: options.body.on};
    if (path === '/memory/reindex') return {notes: 0, turns: 4};
    return {};
  };
  h.tab = setupMemoryTab({doc, api, developer: () => developer, clock: () => h.now});
  h.text = () => h.tab.root.text();
  return h;
}

test('the words: what a search matches, how memory searches, and what forgetting removes', () => {
  const row = note(1, 'You prefer short answers.', {from: 'from a deleted chat, Oct 30'});
  assert.ok(matches(row, 'short ANSWERS') && matches(row, 'preferences deleted') && matches(row, ''));
  assert.ok(!matches(row, 'tea'));
  assert.match(searchLine(false, 'x'), /^Off\./);
  // The line follows the way that answers now, and says why when the setting names another.
  const model = 'Qwen3-Embedding-0.6B';
  const meaning = 'Searching by meaning with Qwen3-Embedding-0.6B, and by words.';
  assert.equal(searchLine(true, model, 'auto', 'embedding'), meaning);
  assert.equal(searchLine(true, model, 'embedding', 'embedding'), meaning);
  const picks = 'Searching by words, and your chat model picks the notes that matter.';
  assert.equal(searchLine(true, '', 'llm', 'llm'), picks);
  assert.equal(searchLine(true, '', 'auto', 'llm'), `${picks} An embedding model in Settings adds search by meaning.`);
  assert.equal(searchLine(true, '', 'bm25', 'bm25'), 'Searching by words only, as the memory retriever setting says.');
  assert.equal(searchLine(true, '', 'auto', 'bm25'), 'Searching by words only. An embedding model in Settings adds search by meaning.');
  assert.equal(searchLine(true, '', 'embedding', 'bm25'), 'Searching by words only. An embedding model in Settings adds search by meaning.');
  assert.equal(forgetLine(2, 5), "This removes 2 notes and melo's search index of 5 chat turns. Your chats stay. It cannot be undone.");
  assert.equal(forgetLine(1, 0), 'This removes 1 note. Your chats stay. It cannot be undone.');
  assert.equal(forgetLine(2, 5, 3), "This removes 2 notes, 3 memory items and melo's search index of 5 chat turns. Your chats stay. It cannot be undone.");
  assert.ok(matches(item(1, 'Tomato garden stakes', 'The user uses bamboo.'), 'tomato BAMBOO'), 'an item is found by its title and its text');
});

test('the notes are listed as the server sent them, each with its source, and a gone chat says so', async () => {
  const h = harness({records: [note(2, 'Your dentist is Lee Park.', {topic: 'about', from: 'from a deleted chat, Oct 30', chat_gone: true}),
    note(1, 'You prefer short answers.', {edited: true})]});
  await h.tab.show();
  const rows = h.tab.root.findAll((node) => node.className === 'memory-note');
  assert.deepEqual(rows.map((item) => item.text()), [
    'Your dentist is Lee Park. About you From a deleted chat, Oct 30 Edit Delete',
    'You prefer short answers. Preferences From chat “Planning”, Oct 30 · edited Edit Delete',
  ]);
  assert.ok(h.tab.root.findAll((node) => node.className === 'memory-from gone').length === 1);
  assert.match(h.text(), /Never kept: passwords, keys, codes and page contents\./);
});

test('with nothing remembered it says how memory starts and offers nothing to forget', async () => {
  const h = harness({extra: {documents: 0}});
  await h.tab.show();
  assert.match(h.text(), /Nothing remembered yet\. melo keeps a note when you tell it something about yourself/);
  assert.equal(h.tab.root.button('Forget everything…'), null);
  assert.equal(h.tab.root.findAll((node) => node.tagName === 'INPUT')[0].parent.hidden, true, 'no search box without notes');
});

test('the search filters the loaded notes and says when none match', async () => {
  const h = harness({records: [note(2, 'Your dentist is Lee Park.', {topic: 'about'}), note(1, 'You prefer short answers.')]});
  await h.tab.show();
  const box = h.tab.root.findAll((node) => node.tagName === 'INPUT')[0];
  box.value = 'dentist';
  box.dispatch('input');
  assert.equal(h.tab.root.findAll((node) => node.className === 'memory-note').length, 1);
  box.value = 'tea';
  box.dispatch('input');
  assert.match(h.text(), /No note matches “tea”\./);
});

test('rewording a note sends the new text, shows it, and keeps the editor open when the server refuses', async () => {
  const h = harness({records: [note(1, 'You prefer short answers.')], fail: {}});
  await h.tab.show();
  h.tab.root.button('Edit').click();
  const area = h.tab.root.findAll((node) => node.tagName === 'TEXTAREA')[0];
  assert.equal(area.value, 'You prefer short answers.');
  assert.equal(doc.activeElement, area, 'the editor takes the focus');
  area.value = 'You prefer long answers.';
  area.dispatch('input');
  h.tab.root.button('Save').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PATCH', '/memory/records/1', {text: 'You prefer long answers.'}]);
  assert.match(h.text(), /You prefer long answers\. Preferences .* · edited/);
  // A refusal says why, in the editor.
  const refused = harness({records: [note(1, 'You prefer short answers.')], fail: {'PATCH /memory/records/1': 'That looks like a password or key, so melo does not keep it.'}});
  await refused.tab.show();
  refused.tab.root.button('Edit').click();
  refused.tab.root.button('Save').click();
  await settle();
  assert.match(refused.text(), /That looks like a password or key/);
  assert.ok(refused.tab.root.findAll((node) => node.tagName === 'TEXTAREA').length === 1);
  refused.tab.root.button('Cancel').click();
  assert.equal(refused.tab.root.findAll((node) => node.tagName === 'TEXTAREA').length, 0);
});

test('deleting a note asks first, keeps it on "Keep it" and forgets it on "Forget it"', async () => {
  const h = harness({records: [note(2, 'Your dentist is Lee Park.', {topic: 'about'}), note(1, 'You prefer short answers.')]});
  await h.tab.show();
  h.tab.root.findAll((node) => node.attributes['aria-label'] === 'Delete: Your dentist is Lee Park.')[0].click();
  assert.match(h.text(), /Forget this note\?/);
  h.tab.root.button('Keep it').click();
  assert.equal(h.tab.root.findAll((node) => node.className === 'memory-note').length, 2);
  assert.ok(!h.calls.some(([method]) => method === 'DELETE'));
  h.tab.root.findAll((node) => node.attributes['aria-label'] === 'Delete: Your dentist is Lee Park.')[0].click();
  h.tab.root.button('Forget it').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/memory/records/2', undefined]);
  assert.deepEqual(h.tab.root.findAll((node) => node.className === 'memory-text').map((item) => item.text()), ['You prefer short answers.']);
});

test('forget everything names what goes, keeps first, and empties the tab once confirmed', async () => {
  const h = harness({records: [note(2, 'Your dentist is Lee Park.', {topic: 'about'}), note(1, 'You prefer short answers.')]});
  await h.tab.show();
  h.tab.root.button('Forget everything…').click();
  assert.match(h.text(), /This removes 2 notes and melo's search index of 4 chat turns\. Your chats stay\. It cannot be undone\./);
  const buttons = h.tab.root.findAll((node) => node.tagName === 'BUTTON' && /everything/.test(node.text())).map((item) => item.text());
  assert.deepEqual(buttons, ['Keep everything', 'Forget everything'], 'keeping comes first');
  h.tab.root.button('Keep everything').click();
  assert.ok(!h.calls.some(([, path]) => path === '/memory/records?all=true'));
  h.tab.root.button('Forget everything…').click();
  h.tab.root.button('Forget everything').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/memory/records?all=true', undefined]);
  assert.match(h.text(), /Nothing remembered yet\./);
});

test('the switch turns memory across chats off and on, and says how it searches', async () => {
  const h = harness({records: [note(1, 'You prefer short answers.')], extra: {embedding_model: 'Qwen3-Embedding-0.6B', answering: 'embedding'}});
  await h.tab.show();
  const toggle = h.tab.root.findAll((node) => node.attributes.role === 'switch')[0];
  assert.equal(toggle.attributes['aria-checked'], 'true');
  assert.match(h.text(), /Searching by meaning with Qwen3-Embedding-0.6B/);
  toggle.click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['PUT', '/memory', {on: false}]);
  assert.equal(h.tab.root.findAll((node) => node.attributes.role === 'switch')[0].attributes['aria-checked'], 'false');
  assert.match(h.text(), /Off\. Notes stay on this computer/);
});

test('the line under the switch follows the way that answers, as the page reads it', async () => {
  const lines = {
    embedding: [{embedding_model: 'Qwen3-Embedding-0.6B', answering: 'embedding'}, /^Remember across chats Searching by meaning with Qwen3-Embedding-0.6B, and by words\./],
    chat: [{answering: 'llm'}, /Searching by words, and your chat model picks the notes that matter\. An embedding model in Settings/],
    named: [{retriever: 'llm', answering: 'llm'}, /Searching by words, and your chat model picks the notes that matter\. You prefer short answers/],
    neither: [{}, /Searching by words only\. An embedding model in Settings adds search by meaning\./],
    words: [{retriever: 'bm25'}, /Searching by words only, as the memory retriever setting says\./],
  };
  for (const [extra, shown] of Object.values(lines)) {
    const h = harness({records: [note(1, 'You prefer short answers.')], extra});
    await h.tab.show();
    assert.match(h.text(), shown);
  }
});

test('rebuilding the search index is a developer-mode link', async () => {
  const plain = harness({records: [note(1, 'x y')]});
  await plain.tab.show();
  assert.equal(plain.tab.root.button('Rebuild search index'), null);
  const dev = harness({records: [note(1, 'x y')], developer: true});
  await dev.tab.show();
  dev.tab.root.button('Rebuild search index').click();
  await settle();
  assert.deepEqual(dev.calls.find(([, path]) => path === '/memory/reindex'), ['POST', '/memory/reindex', {}]);
  assert.match(dev.text(), /Rebuilt: 4 chat turns\./);
});

test('an unreadable memory says so instead of an empty list', async () => {
  const h = harness({fail: {'GET /memory/records': 'Chats are unavailable.'}});
  await h.tab.show();
  assert.match(h.text(), /Chats are unavailable\./);
  assert.doesNotMatch(h.text(), /Nothing remembered yet/);
});

test('a deleted chat with notes asks: keep them first, forget them too, or leave the chat alone', async () => {
  const answers = [];
  for (const [label, want] of [['Keep the notes', 'keep'], ['Forget the 2 notes too', 'forget'], ['Don’t delete the chat', 'cancel']]) {
    const asked = askAboutNotes({doc, count: 2, title: 'Planning'});
    const dialog = doc.body.children.at(-1);
    assert.match(dialog.text(), /Also forget the 2 notes from this chat\?/);
    assert.deepEqual(dialog.findAll((node) => node.tagName === 'BUTTON').map((item) => item.text()),
      ['Keep the notes', 'Forget the 2 notes too', 'Don’t delete the chat']);
    assert.equal(doc.activeElement.text(), 'Keep the notes', 'keeping is focused first');
    dialog.button(label).click();
    answers.push(await asked);
    assert.equal(dialog.removed, true);
    assert.equal(want, answers.at(-1));
  }
  const escaped = askAboutNotes({doc, count: 1});
  const dialog = doc.body.children.at(-1);
  assert.match(dialog.text(), /Also forget the 1 note from this chat\?/);
  dialog.dispatch('cancel');
  assert.equal(await escaped, 'cancel', 'Escape leaves the chat alone');
});

// ── Memory items: what melo wrote from the chats, in the notes' list ──

const rowsOf = (h) => h.tab.root.findAll((node) => node.tagName === 'LI').map((row) => row.text());

test('the items share the notes\' list newest first, marked as melo\'s, with the chat as a link', async () => {
  const h = harness({
    records: [note(4, 'You prefer short answers.'), note(1, 'Your dentist is Lee Park.', {topic: 'about'})],
    items: [item(3, 'Tomato garden stakes', 'The user stakes the tomato garden with bamboo.'),
      item(2, 'Old chat', 'The user planted peas.', {from: 'from a deleted chat, Oct 30', chat_gone: true, link: ''})],
  });
  await h.tab.show();
  assert.deepEqual(rowsOf(h), [
    'You prefer short answers. Preferences From chat “Planning”, Oct 30 Edit Delete',
    'Tomato garden stakes The user stakes the tomato garden with bamboo. Written by melo From chat “Garden”, Oct 31 Delete',
    'Old chat The user planted peas. Written by melo From a deleted chat, Oct 30 Delete',
    'Your dentist is Lee Park. About you From chat “Planning”, Oct 30 Edit Delete',
  ]);
  const links = h.tab.root.findAll((node) => node.tagName === 'A');
  assert.deepEqual(links.map((link) => [link.href, link.text()]), [['#/s/s2', 'From chat “Garden”, Oct 31']]);
  assert.equal(h.tab.root.findAll((node) => node.className === 'memory-from gone').length, 1, 'a gone chat is plain words, not a link');
  assert.equal(h.tab.root.findAll((node) => node.tagName === 'LI' && node.text().includes('Edit')).length, 2, 'only notes can be edited');
  assert.equal(h.tab.root.findAll((node) => node.tagName === 'INPUT')[0].parent.hidden, false, 'items alone give the search box too');
});

test('with no items the tab is the notes as sent: no mark, and the same words as before', async () => {
  const h = harness({records: [note(1, 'You prefer short answers.')]});
  await h.tab.show();
  assert.deepEqual(rowsOf(h), ['You prefer short answers. Preferences From chat “Planning”, Oct 30 Edit Delete']);
  assert.doesNotMatch(h.text(), /Written by melo/);
  h.tab.root.button('Forget everything…').click();
  assert.match(h.text(), /^.*This removes 1 note and melo's search index of 4 chat turns\. Your chats stay\./);
});

test('an item\'s title and text are set as text, whatever characters they hold', async () => {
  const markup = '<img src=x onerror=alert(1)> & <b>bold</b>';
  const h = harness({items: [item(1, markup, '"><script>alert(2)</script> `code`')]});
  await h.tab.show();
  const [title] = h.tab.root.findAll((node) => node.className === 'memory-item-title');
  const [text] = h.tab.root.findAll((node) => node.className === 'memory-text');
  assert.equal(title.textContent, markup);
  assert.equal(text.textContent, '"><script>alert(2)</script> `code`');
  assert.deepEqual([title.children.length, text.children.length], [0, 0], 'no element is made from them');
  assert.ok(h.tab.root.findAll((node) => node.attributes['aria-label'] === `Delete: ${markup}`).length === 1);
});

test('only what a person can settle is named: a disagreement shows, history only for a developer', async () => {
  const items = [item(4, 'Spice', 'The user likes spicy food.', {status: 'conflict', conflicts: [5], kind: 'profile'}),
    item(3, 'Home', 'The user lives in Pittsburgh.', {status: 'replaced', replaced_by: 2}),
    item(1, 'Sync', 'The sync is on Wednesday.', {status: 'replaced', replaced_by: null})];
  const plain = harness({items});
  await plain.tab.show();
  assert.deepEqual(rowsOf(plain), ['Spice The user likes spicy food. Written by melo From chat “Garden”, Oct 31 Disagrees with another item Delete']);
  const dev = harness({items, developer: true});
  await dev.tab.show();
  assert.deepEqual(rowsOf(dev), [
    'Spice The user likes spicy food. Written by melo About you From chat “Garden”, Oct 31 Disagrees with another item Delete',
    'Home The user lives in Pittsburgh. Written by melo Fact From chat “Garden”, Oct 31 Replaced by a newer item Delete',
    'Sync The sync is on Wednesday. Written by melo Fact From chat “Garden”, Oct 31 Replaced; the newer item was deleted Delete',
  ]);
});

test('deleting an item takes effect at once and leaves an Undo that puts it back', async () => {
  const h = harness({records: [note(1, 'You prefer short answers.')], items: [item(3, 'Tomato garden stakes', 'The user uses bamboo.'), item(2, 'Peas', 'The user planted peas.')]});
  await h.tab.show();
  h.tab.root.findAll((node) => node.attributes['aria-label'] === 'Delete: Tomato garden stakes')[0].click();
  await settle();
  assert.deepEqual(h.calls.find(([method]) => method === 'DELETE'), ['DELETE', '/memory/items/3', undefined], 'no question first');
  assert.deepEqual(rowsOf(h), [
    'Deleted “Tomato garden stakes”. Undo',
    'Peas The user planted peas. Written by melo From chat “Garden”, Oct 31 Delete',
    'You prefer short answers. Preferences From chat “Planning”, Oct 30 Edit Delete',
  ]);
  h.tab.root.button('Undo').click();
  await settle();
  assert.deepEqual(h.calls.find(([method]) => method === 'POST'), ['POST', '/memory/items/3/restore', {}]);
  assert.deepEqual(rowsOf(h).slice(0, 2), [
    'Tomato garden stakes The user uses bamboo. Written by melo From chat “Garden”, Oct 31 Delete',
    'Peas The user planted peas. Written by melo From chat “Garden”, Oct 31 Delete',
  ]);
  assert.equal(h.tab.root.button('Undo'), null);
});

test('an Undo line goes when the server\'s time is up, stays when an undo fails, and a refused delete says so', async () => {
  const h = harness({items: [item(3, 'Tomato garden stakes', 'The user uses bamboo.')], fail: {'POST /memory/items/3/restore': 'That item can no longer be put back.'}});
  await h.tab.show();
  h.tab.root.button('Delete').click();
  await settle();
  h.now = 599_000;
  h.tab.root.button('Undo').click();
  await settle();
  assert.match(h.text(), /That item can no longer be put back\./);
  assert.ok(h.tab.root.button('Undo'), 'the line stays so the person can try again');
  h.now = 600_000;
  h.tab.reload();
  await settle();
  assert.equal(h.tab.root.button('Undo'), null, 'after the time the line is gone');
  assert.match(h.text(), /Nothing remembered yet\./);
  const refused = harness({items: [item(3, 'Tomato garden stakes', 'The user uses bamboo.')], fail: {'DELETE /memory/items/3': 'That item is not in melo\'s memory.'}});
  await refused.tab.show();
  refused.tab.root.button('Delete').click();
  await settle();
  assert.match(refused.text(), /That item is not in melo's memory\./);
  assert.equal(rowsOf(refused).length, 1, 'the item stays listed');
});

test('forget everything names the items too and empties them with the notes', async () => {
  const h = harness({records: [note(1, 'You prefer short answers.')], items: [item(2, 'Peas', 'The user planted peas.')]});
  await h.tab.show();
  h.tab.root.button('Forget everything…').click();
  assert.match(h.text(), /This removes 1 note, 1 memory item and melo's search index of 4 chat turns\./);
  h.tab.root.button('Forget everything').click();
  await settle();
  assert.match(h.text(), /Nothing remembered yet\./);
  assert.doesNotMatch(h.text(), /Peas/);
});

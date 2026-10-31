/**
 * Bringing in what other assistants know (import-card.mjs) in a DOM double: the main chat's first-run card
 * (offer, working, done with Undo, "Not now", an empty scan, a partial one, refusals) and the Memory tab's
 * "From other assistants" section (look, bring in, batches with Undo, the "from <file>" label of a note).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {batches, homePath, leftOut, readScan, setupImportCard} from '../../support/gateway/static/import-card.mjs';
import {setupMemoryTab} from '../../support/gateway/static/memory-tab.mjs';

class Element {
  constructor(tag) {
    Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, handlers: {}, dataset: {}, textContent: '',
      className: '', hidden: false, disabled: false, checked: false, value: '', parent: null});
  }
  get ownerDocument() { return doc; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  append(...nodes) { for (const item of nodes) { if (typeof item === 'string') { const text = new Element('#text'); text.textContent = item; this.append(text); } else { item.parent = this; this.children.push(item); } } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  dispatch(type, event = {}) { for (const fn of this.handlers[type] ?? []) fn({target: this, stopPropagation() {}, preventDefault() {}, ...event}); }
  click() { if (!this.disabled) this.dispatch('click'); }
  toggle(checked) { this.checked = checked; this.dispatch('change'); }
  focus() { doc.activeElement = this; }
  contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
  text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' ').replace(/\s+/g, ' ').trim(); }
  findAll(test) { return [...(test(this) ? [this] : []), ...this.children.flatMap((child) => child.findAll(test))]; }
  button(label) { return this.findAll((node) => node.tagName === 'BUTTON' && node.text() === label)[0] ?? null; }
  boxes() { return this.findAll((node) => node.tagName === 'INPUT' && node.type === 'checkbox'); }
}
const doc = {activeElement: null, createElement: (tag) => new Element(tag)};
const settle = () => new Promise((resolve) => setImmediate(resolve));

const HOME = '/Users/me';
const CLAUDE = {id: 'claude-35bc414c', harness: 'claude', path: `${HOME}/.claude/CLAUDE.md`, scope: 'user', workspace: '', items_found: 5, items_kept: 2,
  filtered: {secret: 1, third_party: 1, wrong_owner: 0, duplicate: 0, mechanics: 0}, sample: ['Use uv for new Python projects.', 'Keep commits small.']};
const HERMES = {id: 'hermes-2e22e64e', harness: 'hermes', path: `${HOME}/.hermes/memories/USER.md`, scope: 'user', workspace: '', items_found: 2, items_kept: 2,
  filtered: {secret: 0, third_party: 0, wrong_owner: 0, duplicate: 0, mechanics: 0}, sample: ['Prefers concise replies.']};
const scan = (sources = [CLAUDE, HERMES], extra = {}) => ({batch_id: 'b1', sources, partial: false, home: HOME, ...extra});

/** A gateway double: the offer, the scan and the writes; `fail` makes a call refuse, `hold` keeps a call pending. */
function harness({pending = true, scanned = scan(), fail = {}, written = [1, 2, 3, 4]} = {}) {
  const h = {calls: [], pending, scanned, hold: {}};
  h.api = async (path, options = {}) => {
    const method = options.method ?? 'GET';
    h.calls.push([method, path, options.body]);
    if (h.hold[`${method} ${path}`]) await h.hold[`${method} ${path}`];
    if (fail[`${method} ${path}`]) throw Object.assign(new Error(fail[`${method} ${path}`].message), {status: fail[`${method} ${path}`].status});
    if (path === '/memory/import/offer') return {pending: method === 'GET' ? h.pending : false};
    if (path === '/memory/import/scan') return h.scanned;
    if (path === '/memory/import') return {batch_id: options.body.batch_id, written_ids: written, skipped: 0};
    if (path.startsWith('/memory/import/')) return {batch_id: path.split('/').pop(), removed: written.length};
    return {};
  };
  h.root = new Element('section');
  h.card = setupImportCard({root: h.root, api: h.api});
  h.text = () => h.root.text();
  h.paths = () => h.calls.map(([method, path]) => `${method} ${path}`);
  return h;
}

test('a path shows the home folder as ~, and what a file left out is said by reason', () => {
  assert.equal(homePath('/Users/me/.claude/CLAUDE.md', '/Users/me'), '~/.claude/CLAUDE.md');
  assert.equal(homePath('/Users/me/.claude/CLAUDE.md', '/Users/me/'), '~/.claude/CLAUDE.md');
  assert.equal(homePath('/Users/mel/x.md', '/Users/me'), '/Users/mel/x.md', 'a longer name is not the home');
  assert.equal(homePath('/opt/x.md', ''), '/opt/x.md');
  assert.equal(leftOut({secret: 1, third_party: 1}), '2 left out: secrets, other people’s details');
  assert.equal(leftOut({mechanics: 3, secret: 0}), '3 left out: how the other tool works');
  assert.equal(leftOut({secret: 0}), '');
  const read = readScan(scan());
  assert.deepEqual(read.sources.map((source) => [source.path, source.kept]), [['~/.claude/CLAUDE.md', 2], ['~/.hermes/memories/USER.md', 2]]);
  assert.equal(read.batchId, 'b1');
});

test('a hidden card asks nothing; showing it asks whether the offer is open before it looks', async () => {
  const h = harness();
  h.card.show(false);
  await settle();
  assert.deepEqual(h.calls, []);
  assert.equal(h.root.hidden, true);
  h.card.show(true);
  await settle();
  assert.deepEqual(h.paths(), ['GET /memory/import/offer', 'GET /memory/import/scan']);
  assert.equal(h.root.hidden, false);
  h.card.show(true);
  h.card.show(false);
  assert.equal(h.root.hidden, true, 'hiding keeps the card for when the main chat is back');
  h.card.show(true);
  await settle();
  assert.equal(h.calls.length, 2, 'asked once');
  assert.equal(h.root.hidden, false);
});

test('the offer is one row per file with a checked box, the notes, and a disclosure of the samples and what was left out', async () => {
  const h = harness();
  h.card.show(true);
  await settle();
  assert.match(h.text(), /^melo found notes on this Mac from other assistants/);
  assert.match(h.text(), /Pick what melo may learn from/);
  const rows = h.root.findAll((node) => node.className === 'import-source');
  assert.deepEqual(rows.map((row) => row.findAll((node) => node.className === 'import-pick')[0].text()), [
    '~/.claude/CLAUDE.md 2 notes',
    '~/.hermes/memories/USER.md 2 notes',
  ]);
  assert.deepEqual(h.root.boxes().map((box) => box.checked), [true, true]);
  assert.match(rows[0].text(), /See what is in it Use uv for new Python projects\. Keep commits small\. 2 left out: secrets, other people’s details/);
  assert.doesNotMatch(rows[1].text(), /left out/);
  assert.doesNotMatch(h.text().replace(/~\/\S+/g, ''), /claude|hermes|codex|gemini/i, 'only paths name the files');
  assert.ok(h.root.button('Bring them in') && h.root.button('Not now'));
  assert.deepEqual(h.paths(), ['GET /memory/import/offer', 'GET /memory/import/scan'], 'looking wrote nothing');
});

test('"Bring them in" sends only the checked files, shows the work, then "Brought in N notes · Undo"', async () => {
  const h = harness({written: [1, 2]});
  let release;
  h.hold['POST /memory/import'] = new Promise((resolve) => { release = resolve; });
  h.card.show(true);
  await settle();
  h.root.boxes()[1].toggle(false);
  h.root.button('Bring them in').click();
  await settle();
  assert.equal(h.root.attributes['aria-busy'], 'true');
  assert.ok(h.root.button('Bringing them in…').disabled, 'a second press is not possible');
  assert.ok(h.root.button('Not now').disabled);
  release();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['POST', '/memory/import', {batch_id: 'b1', source_ids: ['claude-35bc414c']}]);
  assert.equal(h.root.text(), 'Brought in 2 notes · Undo Close');
  assert.equal(h.root.attributes['aria-busy'], 'false');
});

test('Undo takes the batch back and says so; Close hides the card', async () => {
  const h = harness();
  h.card.show(true);
  await settle();
  h.root.button('Bring them in').click();
  await settle();
  h.root.button('Undo').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/memory/import/b1', undefined]);
  assert.equal(h.root.text(), 'Taken back. Close');
  h.root.button('Close').click();
  assert.equal(h.root.hidden, true);
  assert.ok(!h.paths().includes('DELETE /memory/import/offer'), 'closing does not decline anything');
});

test('"Not now" records the offer as made and the card does not come back', async () => {
  const h = harness();
  h.card.show(true);
  await settle();
  h.root.button('Not now').click();
  await settle();
  assert.equal(h.root.hidden, true);
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/memory/import/offer', undefined]);
  assert.ok(!h.paths().includes('POST /memory/import'));
  h.card.show(false);
  h.card.show(true);
  await settle();
  assert.equal(h.root.hidden, true);
  assert.equal(h.calls.filter(([method]) => method === 'GET').length, 2, 'it did not ask again');
});

test('no card when the offer was made, when the scan finds nothing, or when asking fails', async () => {
  const made = harness({pending: false});
  made.card.show(true);
  await settle();
  assert.equal(made.root.hidden, true);
  assert.deepEqual(made.paths(), ['GET /memory/import/offer'], 'an offer made once is not scanned for');

  const empty = harness({scanned: scan([])});
  empty.card.show(true);
  await settle();
  assert.equal(empty.root.hidden, true);
  assert.ok(!empty.paths().includes('DELETE /memory/import/offer'), 'an empty scan leaves the offer open');

  const nothing = harness({scanned: scan([{...CLAUDE, items_kept: 0, sample: []}])});
  nothing.card.show(true);
  await settle();
  assert.equal(nothing.root.hidden, true, 'files with no note to bring in are no offer');

  const broken = harness({fail: {'GET /memory/import/scan': {message: 'boom'}}});
  broken.card.show(true);
  await settle();
  assert.equal(broken.root.hidden, true);
  broken.card.show(true);
  await settle();
  assert.equal(broken.calls.length, 2, 'a failed look is not repeated until the page reloads');
});

test('a partial scan says so, a file with no note cannot be checked, and nothing checked cannot be brought in', async () => {
  const h = harness({scanned: scan([CLAUDE, {...HERMES, items_kept: 0, sample: []}], {partial: true})});
  h.card.show(true);
  await settle();
  assert.match(h.text(), /The search stopped early, so a few files may be missing\./);
  const [first, second] = h.root.boxes();
  assert.equal(first.checked, true);
  assert.equal(second.checked, false);
  assert.equal(second.disabled, true);
  first.toggle(false);
  assert.equal(h.root.button('Bring them in').disabled, true);
  first.toggle(true);
  assert.equal(h.root.button('Bring them in').disabled, false);
});

test('a refusal is said in the server’s words and the card stays; a forgotten scan is read again', async () => {
  const h = harness({fail: {'POST /memory/import': {message: 'That scan is gone: scan again.', status: 404}}});
  h.card.show(true);
  await settle();
  h.root.button('Bring them in').click();
  await settle();
  assert.match(h.text(), /That scan is gone: scan again\./);
  assert.ok(h.root.button('Bring them in'), 'the offer is still there');
  assert.equal(h.paths().filter((call) => call === 'GET /memory/import/scan').length, 2, 'the scan was read again');
  const failing = harness({fail: {'DELETE /memory/import/b1': {message: ''}}});
  failing.card.show(true);
  await settle();
  failing.root.button('Bring them in').click();
  await settle();
  failing.root.button('Undo').click();
  await settle();
  assert.match(failing.text(), /Brought in 4 notes · Undo.*Could not take the notes back\. Try again\./);
});

test('notes that were all known already say so and offer no Undo', async () => {
  const h = harness({written: []});
  h.card.show(true);
  await settle();
  h.root.button('Bring them in').click();
  await settle();
  assert.equal(h.root.text(), 'Nothing new: melo already knows these. Close');
});

test('the batches are the imported notes grouped, newest first', () => {
  const note = (id, batch, at, imported = true) => ({id, imported, batch_id: batch, created_at: at});
  assert.deepEqual(batches([note(1, 'a', 10), note(2, 'a', 30), note(3, 'b', 50), note(4, '', 90, false), note(5, '', 95)]), [
    {id: 'b', count: 1, at: 50},
    {id: 'a', count: 2, at: 30},
  ]);
});

// ── The Memory tab's section ──

const row = (id, text, extra = {}) => ({id, text, topic: 'preferences', quote: '', source: '', from: 'from the file CLAUDE.md, Oct 31', imported: true,
  batch_id: 'b1', chat_gone: false, edited: false, session_id: '', created_at: 1_761_900_000 + id, ...extra});

function tab({records = [], on = true, ...options} = {}) {
  const h = harness(options);
  h.records = records;
  const api = async (path, call = {}) => {
    if (path === '/memory/records') return {on, records: h.records, documents: 0, embedding_model: '', last_error: ''};
    const answer = await h.api(path, call);
    if (call.method === 'POST') h.records = [row(7, 'Use uv for new Python projects.'), row(8, 'Keep commits small.')];
    if (call.method === 'DELETE') h.records = [];
    return answer;
  };
  h.tab = setupMemoryTab({doc, api});
  h.text = () => h.tab.root.text();
  return h;
}

test('the Memory tab: "Look for notes" shows the files, "Bring them in" writes the checked ones and lists the batch with Undo', async () => {
  const h = tab();
  await h.tab.show();
  assert.match(h.text(), /From other assistants Bring in what other assistants on this Mac already know about you/);
  assert.deepEqual(h.paths(), [], 'the tab reads nothing until asked');
  h.tab.root.button('Look for notes').click();
  await settle();
  assert.deepEqual(h.paths(), ['GET /memory/import/scan']);
  assert.match(h.text(), /~\/\.claude\/CLAUDE\.md 2 notes.*~\/\.hermes\/memories\/USER\.md 2 notes/);
  h.tab.root.boxes()[1].toggle(false);
  h.tab.root.button('Bring them in').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['POST', '/memory/import', {batch_id: 'b1', source_ids: ['claude-35bc414c']}]);
  assert.match(h.text(), /Brought in 4 notes/);
  assert.match(h.text(), /2 notes brought in · /, 'the batch is listed from the notes the tab loaded');
  assert.match(h.text(), /Use uv for new Python projects\. Preferences From the file CLAUDE\.md, Oct 31 Edit Delete/, 'an imported note carries its file');
  h.tab.root.button('Undo').click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/memory/import/b1', undefined]);
  assert.match(h.text(), /Taken back\./);
  assert.doesNotMatch(h.text(), /brought in ·/);
});

test('the Memory tab: nothing found, a refused look, past batches without a look, and memory switched off', async () => {
  const none = tab({scanned: scan([])});
  await none.tab.show();
  none.tab.root.button('Look for notes').click();
  await settle();
  assert.match(none.text(), /No notes found on this Mac\./);

  const known = tab({scanned: scan([{...CLAUDE, items_kept: 0, sample: []}])});
  await known.tab.show();
  known.tab.root.button('Look for notes').click();
  await settle();
  assert.match(known.text(), /Nothing new: melo already knows these\./, 'files whose notes are all known are not "no notes"');

  const refused = tab({fail: {'GET /memory/import/scan': {message: ''}}});
  await refused.tab.show();
  refused.tab.root.button('Look for notes').click();
  await settle();
  assert.match(refused.text(), /Could not look for notes\. Try again\./);
  assert.ok(refused.tab.root.button('Look for notes'), 'it can look again');

  const past = tab({records: [row(1, 'Use uv.'), row(2, 'Keep commits small.', {batch_id: 'b2'})]});
  await past.tab.show();
  assert.equal(past.tab.root.findAll((node) => node.className === 'import-batch').length, 2);

  const off = tab({on: false});
  await off.tab.show();
  assert.doesNotMatch(off.text(), /From other assistants/, 'with memory off there is nothing to bring notes into');
});

import assert from 'node:assert/strict';
import test from 'node:test';
import {useLanguage} from '../../support/gateway/static/i18n.mjs';
import {digestParts, inline, memoryNodes, readMemory, usedLabel} from '../../support/gateway/static/memory.mjs';
import {applyEvent, findTurn, newView} from '../../support/gateway/static/turns.mjs';

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.handlers = {}; this.className = ''; this.textContent = ''; this.type = ''; }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...nodes) { this.children.push(...nodes); }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  text() { return this.textContent + this.children.map((c) => c.text()).join(''); }
  all(tag) { return this.children.flatMap((c) => [...(c.tag === tag ? [c] : []), ...c.all(tag)]); }
}
const doc = {createElement: (tag) => new Node(tag)};
const source = 'you said it · chat “Planning”, Oct 30';
const data = {
  turn: 2,
  turn_id: 't2',
  used: [{kind: 'record', id: 4, text: 'Your group meeting is on Thursdays.', source, session_id: 's1', turn: 1, at: 1}],
  remembered: [{id: 7, text: 'You prefer short answers.', topic: 'preferences'}],
  refused: 'it looked like a password or key',
};

test('readMemory keeps what the page can use and is null for nothing', () => {
  assert.equal(readMemory({}), null);
  assert.equal(readMemory({used: 'x', remembered: [null, {id: 0, text: 'x'}], refused: 3}), null);
  const memory = readMemory(data);
  assert.deepEqual(memory.used, [{text: 'Your group meeting is on Thursdays.', source}]);
  assert.deepEqual(memory.remembered, [{id: 7, text: 'You prefer short answers.', undone: false}]);
  assert.equal(memory.refused, 'it looked like a password or key');
});

test('the words: one memory, two memories, a note inside a sentence', () => {
  assert.equal(usedLabel(1), 'Used 1 memory');
  assert.equal(usedLabel(2), 'Used 2 memories');
  assert.equal(inline('You prefer short answers.'), 'you prefer short answers.');
  assert.equal(inline('Your group meeting is on Thursdays.'), 'your group meeting is on Thursdays.');
  assert.equal(inline('Sam has the lab keys.'), 'Sam has the lab keys.');
  assert.equal(inline('Yoga is on Mondays.'), 'Yoga is on Mondays.');
});

test('a turn.memory event lands on its turn', () => {
  const view = newView();
  applyEvent(view, {id: 1, type: 'turn.started', turn: 2, data: {turn: 2, turn_id: 't2', user_text: 'when is my group meeting?'}});
  applyEvent(view, {id: 2, type: 'turn.memory', turn: 2, data});
  assert.equal(findTurn(view, 2).memory.used.length, 1);
  assert.equal(findTurn(view, 2).memoryOpen, false);
});

test('closed, it says how many; open, it lists each with its source', () => {
  const memory = readMemory(data);
  const toggles = [];
  const undone = [];
  const shut = memoryNodes({doc, memory, open: false, toggle: () => toggles.push(1), undo: (note) => undone.push(note.id)});
  assert.equal(shut.length, 1);
  const [toggle, undo] = shut[0].all('button');
  assert.equal(toggle.textContent, 'Used 1 memory ▾');
  assert.equal(toggle.attributes['aria-expanded'], 'false');
  assert.equal(shut[0].all('li').length, 0);
  toggle.handlers.click();
  assert.deepEqual(toggles, [1]);
  const lines = shut[0].all('p').map((p) => p.text());
  assert.deepEqual(lines, [
    'Remembered: you prefer short answers. · Undo',
    'Not remembered: it looked like a password or key',
  ]);
  undo.handlers.click();
  assert.deepEqual(undone, [7]);

  const [open] = memoryNodes({doc, memory, open: true, toggle() {}, undo() {}});
  assert.equal(open.all('button')[0].textContent, 'Used 1 memory ▴');
  assert.deepEqual(open.all('li').map((li) => li.text()), [`Your group meeting is on Thursdays. (${source})`]);
});

test('an undone note says so and offers no Undo; no memory draws nothing', () => {
  const memory = readMemory({remembered: [{id: 7, text: 'You go by Ray.'}]});
  memory.remembered[0].undone = true;
  const [box] = memoryNodes({doc, memory, open: false, toggle() {}, undo() {}});
  assert.deepEqual(box.all('p').map((p) => p.text()), ['Forgotten: you go by Ray.']);
  assert.equal(box.all('button').length, 0);
  assert.deepEqual(memoryNodes({doc, memory: null, open: false, toggle() {}, undo() {}}), []);
});

// A new chat's first message carried a digest of the user's other chats (#452).
const digestData = {
  turn: 1,
  turn_id: 't1',
  used: [],
  remembered: [],
  refused: '',
  digest: {
    minutes: 30,
    items: [
      {session_id: 'main1', turn: 4, line: 'Main chat, 22:45 · the user: What else could Alex do besides UX research? · melo: Alex could move into product work.'},
      {session_id: 'cli9', turn: 1, line: 'terminal chat “notes”, 22:50 · the user: go on'},
    ],
  },
};

test('readMemory keeps the digest, and a digest alone is something to show', () => {
  const memory = readMemory(digestData);
  assert.deepEqual(memory.digest, {
    minutes: 30,
    items: [
      {sessionId: 'main1', turn: 4, line: digestData.digest.items[0].line},
      {sessionId: 'cli9', turn: 1, line: digestData.digest.items[1].line},
    ],
  });
  assert.equal(readMemory({...digestData, digest: {minutes: 0, items: digestData.digest.items}}), null);
  assert.equal(readMemory({...digestData, digest: {minutes: 30, items: [{session_id: 'x', line: ''}]}}), null);
  assert.equal(readMemory(data).digest, null);
});

test('a digest line splits into the chat it names and what was said there', () => {
  assert.deepEqual(digestParts('Main chat, 22:45 · the user: hi'), {where: 'Main chat, 22:45', said: ' · the user: hi'});
  assert.deepEqual(digestParts('no parts here'), {where: '', said: 'no parts here'});
});

test('closed, the group names the window; open, each line reads as the model read it', () => {
  const memory = readMemory(digestData);
  const toggles = [];
  const [shut] = memoryNodes({doc, memory, open: false, toggle() {}, undo() {}, toggleDigest: () => toggles.push(1)});
  const [toggle] = shut.all('button');
  assert.equal(toggle.textContent, 'From your other chats (last 30 min) ▾');
  assert.equal(toggle.attributes['aria-expanded'], 'false');
  assert.equal(shut.all('li').length, 0);
  toggle.handlers.click();
  assert.deepEqual(toggles, [1]);

  const opened = [];
  const openChat = (id) => (id === 'main1' ? () => opened.push(id) : null);
  const [open] = memoryNodes({doc, memory, open: false, toggle() {}, undo() {}, digestOpen: true, openChat});
  assert.equal(open.all('button')[0].textContent, 'From your other chats (last 30 min) ▴');
  const rows = open.all('li');
  assert.deepEqual(rows.map((li) => li.text()), digestData.digest.items.map((item) => item.line));
  // The main chat's line opens it; the terminal chat's names it without a link. No Undo.
  const [link] = rows[0].all('button');
  assert.equal(link.textContent, 'Main chat, 22:45');
  link.handlers.click();
  assert.deepEqual(opened, ['main1']);
  assert.equal(rows[1].all('button').length, 0);
  assert.equal(rows[1].children[0].className, 'memory-source');
  assert.ok(!open.text().includes('Undo'));
});

test('the group reads in Chinese on a Chinese page', async () => {
  await useLanguage('zh-Hans');
  try {
    const [box] = memoryNodes({doc, memory: readMemory(digestData), open: false, toggle() {}, undo() {}});
    assert.equal(box.all('button')[0].textContent, '来自你的其他聊天（最近 30 分钟） ▾');
  } finally {
    await useLanguage('en');
  }
});

test('a turn.memory event with a digest lands on its turn, closed', () => {
  const view = newView();
  applyEvent(view, {id: 1, type: 'turn.started', turn: 1, data: {turn: 1, turn_id: 't1', user_text: 'read his résumé first'}});
  applyEvent(view, {id: 2, type: 'turn.memory', turn: 1, data: digestData});
  assert.equal(findTurn(view, 1).memory.digest.items.length, 2);
  assert.equal(findTurn(view, 1).digestOpen, false);
});

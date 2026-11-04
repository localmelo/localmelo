/**
 * The composer's pictures and files (`composer-media.mjs`): what a file is, what the strip holds, and how
 * the words a message was stored with are read back for the bubble. The page's drop, paste and send flows
 * are in `test_workspace.mjs`.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {
  MAX_FILES, MAX_PICTURES, isStandIn, kindOf, pictureItems, requestOf, setupMedia, splitAttachments, turnPictures, uploadType,
} from '../../support/gateway/static/composer-media.mjs';

const COMPOSED = JSON.parse(readFileSync(new URL('../fixtures/composed_attachments.json', import.meta.url), 'utf8'));

test('a file is a picture, text or other by its type, and by its name when the type is missing', () => {
  const kinds = (...files) => files.map(([name, type]) => kindOf({name, type}));
  assert.deepEqual(kinds(['a.png', 'image/png'], ['b.JPG', ''], ['c.heic', 'image/heic'], ['d.webp', 'image/webp'], ['e.gif', 'image/gif']),
    ['picture', 'picture', 'picture', 'picture', 'picture']);
  assert.deepEqual(kinds(['a.txt', 'text/plain'], ['b.md', ''], ['c.py', 'text/x-python'], ['d.json', 'application/json'], ['e.swift', '']),
    ['text', 'text', 'text', 'text', 'text']);
  assert.deepEqual(kinds(['a.pdf', 'application/pdf'], ['b.zip', 'application/zip'], ['c.mp4', 'video/mp4'], ['README', '']),
    ['other', 'other', 'other', 'other']);
  // An SVG is markup the model reads as text, not a picture the gateway could scale.
  assert.equal(kindOf({name: 'logo.svg', type: 'image/svg+xml'}), 'text');
  assert.equal(kindOf({name: 'logo.svg', type: ''}), 'text');
});

test('an upload goes out with the picture\'s own type, or a generic one the gateway reads by the bytes', () => {
  assert.equal(uploadType({name: 'a.png', type: 'image/png'}), 'image/png');
  assert.equal(uploadType({name: 'a.heic', type: ''}), 'application/octet-stream');
  assert.equal(uploadType({name: 'a.jpg', type: 'image/x-weird'}), 'application/octet-stream');
});

test('the limits are the ones the gateway keeps', () => {
  assert.equal(MAX_PICTURES, 4);
  assert.equal(MAX_FILES, 8);
});

test('the text files of a stored message are read back from the gateway\'s own composition', () => {
  const both = splitAttachments(COMPOSED.with_words.composed);
  assert.equal(both.words, 'see these');
  assert.deepEqual(both.files.map((file) => file.name), ['notes.md', 'a "b" & <c>.txt'], 'names come back unescaped');
  assert.deepEqual(both.files.map((file) => file.chars), [16, 39]);
  // A file that holds the closing tag does not end the block early: the gateway escapes it.
  assert.equal(both.files.length, COMPOSED.files.length);
  const alone = splitAttachments(COMPOSED.stand_in.composed);
  assert.equal(alone.words, COMPOSED.stand_in.words);
  assert.equal(isStandIn(alone.words), true, 'a message sent with no words shows only its files');
  assert.deepEqual(splitAttachments('plain words'), {words: 'plain words', files: []});
});

test('words that merely look like attachments stay the person\'s words', () => {
  const lookalike = 'hello\n\n<attachment name="x" chars="3">\nnot closed';
  assert.deepEqual(splitAttachments(lookalike), {words: lookalike, files: []});
  assert.equal(isStandIn('[attached: notes.md]'), true);
  assert.equal(isStandIn('[2 pictures]'), true);
  assert.equal(isStandIn('[1 picture] [attached: a.txt, b.md]'), true);
  assert.equal(isStandIn('what is [1 picture]?'), false);
  assert.equal(isStandIn(''), false);
});

test('a request carries only the pictures and files it has', () => {
  assert.deepEqual(requestOf([]), {});
  const taken = [
    {kind: 'picture', uploadId: 'u1'},
    {kind: 'file', name: 'n.md', text: 'héllo'},
    {kind: 'picture', uploadId: 'u2'},
  ];
  assert.deepEqual(requestOf(taken), {images: ['u1', 'u2'], attachments: [{name: 'n.md', text: 'héllo', chars: 5}]});
});

test('the pictures of a turn are served from the chat\'s blobs', () => {
  const [one, two] = turnPictures('chat 1', [{sha256: 'ab'.repeat(32), size_bytes: 10}, {sha256: 'cd'.repeat(32), size_bytes: 20}]);
  assert.equal(one.url, `/api/sessions/chat%201/blobs/${'ab'.repeat(32)}`);
  assert.equal(one.name, 'Picture 1');
  assert.equal(two.name, 'Picture 2');
  assert.deepEqual(turnPictures('c', undefined), []);
});

// ── The strip, over a minimal document ──

function fakeDocument() {
  const make = (tag) => {
    const listeners = {};
    const node = {
      tag, className: '', textContent: '', children: [], dataset: {}, attributes: {}, hidden: false,
      append(...items) { node.children.push(...items); },
      replaceChildren(...items) { node.children = items; },
      setAttribute(name, value) { node.attributes[name] = value; },
      addEventListener(type, handler) { (listeners[type] ||= []).push(handler); },
      listeners,
      click() { listeners.click?.forEach((handler) => handler()); },
    };
    return node;
  };
  return {doc: {createElement: make}, make};
}

function strip({upload, notify = [], discarded = [], changes = [], refuse} = {}) {
  const {doc, make} = fakeDocument();
  const parts = {tray: make('div'), warning: make('p'), fileInput: make('input')};
  const media = setupMedia({
    doc, ...parts,
    upload: upload ?? (async (file) => ({id: `id-${file.name}`, size_bytes: file.size, width: 8, height: 8, model_reads: true})),
    discard: (id) => discarded.push(id),
    notify: (message) => notify.push(message),
    changed: () => changes.push(1),
    openSettings: () => {},
    open: () => {},
    ...(refuse ? {refuse} : {}),
  });
  return {media, ...parts, notify, discarded, changes};
}
const picture = (name, size = 100) => ({name, type: 'image/png', size, arrayBuffer: async () => new ArrayBuffer(size)});
const textual = (name, body) => {
  const bytes = new TextEncoder().encode(body);
  return {name, type: 'text/plain', size: bytes.length, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length)};
};
const later = () => new Promise((resolve) => setImmediate(resolve));

test('a picture shows a waiting tile while it uploads, and a thumbnail when it has arrived', async () => {
  let arrive;
  const held = new Promise((resolve) => { arrive = resolve; });
  const s = strip({upload: () => held});
  const added = s.media.add([picture('a.png')]);
  assert.equal(s.media.busy(), true, 'Send waits for the picture');
  assert.equal(s.media.has(), true);
  assert.equal(s.tray.children[0].dataset.state, 'adding');
  arrive({id: 'u1', size_bytes: 10, width: 1, height: 1, model_reads: true});
  await added;
  assert.equal(s.media.busy(), false);
  assert.equal(s.tray.children[0].dataset.state, 'ready');
});

test('a picture removed while it uploads is forgotten at the gateway as soon as it arrives', async () => {
  let arrive;
  const held = new Promise((resolve) => { arrive = resolve; });
  const s = strip({upload: () => held});
  const added = s.media.add([picture('a.png')]);
  s.media.remove(s.media.items()[0].id);
  assert.equal(s.media.has(), false);
  arrive({id: 'late', size_bytes: 10, width: 1, height: 1, model_reads: true});
  await added;
  assert.deepEqual(s.discarded, ['late']);
  assert.equal(s.media.has(), false);
});

test('take empties the strip of what arrived and leaves what is still on its way; restore puts it back first', async () => {
  const s = strip();
  await s.media.add([picture('a.png'), textual('n.txt', 'hi')]);
  const taken = s.media.take();
  assert.deepEqual(taken.map((item) => item.name), ['a.png', 'n.txt']);
  assert.equal(s.media.has(), false);
  await s.media.add([textual('later.txt', 'x')]);
  s.media.restore(taken);
  assert.deepEqual(s.media.items().map((item) => item.name), ['a.png', 'n.txt', 'later.txt']);
});

test('a failed upload leaves no tile, says why, and counts nothing against the limit', async () => {
  const s = strip({upload: async () => { throw new Error('too big for me'); }});
  await s.media.add([picture('a.png')]);
  assert.equal(s.media.has(), false);
  assert.match(s.notify[0], /Could not add a\.png: too big for me/);
});

test('the warning appears with a text-only model and goes when the picture does', async () => {
  const s = strip({upload: async () => ({id: 'u', size_bytes: 1, width: 1, height: 1, model_reads: false})});
  await s.media.add([picture('a.png')]);
  assert.equal(s.warning.hidden, false);
  s.media.remove(s.media.items()[0].id);
  assert.equal(s.warning.hidden, true);
});

test('a refusal for pictures warns about the strip until the gateway says the model reads them', async () => {
  let reads = true;
  const s = strip({upload: async () => ({id: 'u', size_bytes: 1, width: 1, height: 1, model_reads: reads})});
  s.media.refusedPictures();
  assert.equal(s.warning.hidden, true, 'nothing waits to be warned about');
  await s.media.add([picture('a.png')]);
  assert.equal(s.warning.hidden, true, 'the gateway says the model reads them');
  s.media.refusedPictures();
  assert.equal(s.warning.hidden, false);
  reads = false;
  await s.media.add([picture('b.png')]);
  assert.equal(s.warning.hidden, false, 'still text only');
  s.media.remove(s.media.items()[0].id);
  s.media.remove(s.media.items()[0].id);
  reads = true;
  await s.media.add([picture('c.png')]);
  assert.equal(s.warning.hidden, true, 'the gateway answered that the model reads them: the refusal is over');
});

test('swap shows another chat\'s strip and returns this one; discard forgets what it held', async () => {
  const s = strip();
  await s.media.add([picture('a.png')]);
  const mine = s.media.swap([]);
  assert.equal(s.media.has(), false);
  assert.equal(mine.length, 1);
  s.media.swap(mine);
  assert.equal(s.media.has(), true);
  s.media.swap([], {discard: true});
  assert.deepEqual(s.discarded, ['id-a.png']);
  assert.equal(s.media.has(), false);
});

test('eight text files fit; the ninth, a file over 256 KB and a pile over 512 KB are each turned away with the limit', async () => {
  const s = strip();
  await s.media.add(Array.from({length: 9}, (_, n) => textual(`f${n}.txt`, 'x')));
  assert.equal(s.media.items().length, 8);
  assert.match(s.notify.at(-1), /at most|can carry 8 text files/);
  const t = strip();
  await t.media.add([textual('big.txt', 'x'.repeat(256 * 1024 + 1))]);
  assert.match(t.notify[0], /over 256 KB/);
  const u = strip();
  await u.media.add([textual('a.txt', 'x'.repeat(250 * 1024)), textual('b.txt', 'x'.repeat(250 * 1024)), textual('c.txt', 'x'.repeat(100 * 1024))]);
  assert.equal(u.media.items().length, 2);
  assert.match(u.notify[0], /total 512 KB/);
});

test('the file input\'s change adds what was chosen and clears the input', async () => {
  const s = strip();
  s.fileInput.files = [picture('a.png')];
  s.fileInput.value = 'C:\\fakepath\\a.png';
  s.fileInput.listeners.change[0]();
  await later();
  assert.equal(s.fileInput.value, '');
  assert.equal(s.media.items().length, 1);
});

test('a host can refuse a kind of file for now: it is told why, and the pictures beside it are still added', async () => {
  const kinds = [];
  const s = strip({refuse: (file, kind) => { kinds.push([file.name, kind]); return kind === 'text' ? 'Text files cannot join now.' : null; }});
  await s.media.add([picture('a.png'), textual('n.txt', 'hi')]);
  assert.deepEqual(kinds, [['a.png', 'picture'], ['n.txt', 'text']]);
  assert.deepEqual(s.notify, ['Text files cannot join now.']);
  assert.deepEqual(s.media.items().map((item) => item.name), ['a.png']);
  assert.equal(s.media.hasFiles(), false);
  assert.equal(s.media.hasPictures(), true);
});

test('hasFiles says whether a text file waits in the strip', async () => {
  const s = strip();
  await s.media.add([picture('a.png')]);
  assert.equal(s.media.hasFiles(), false);
  await s.media.add([textual('n.txt', 'hi')]);
  assert.equal(s.media.hasFiles(), true);
});

test('pictures the gateway staged again are strip items that go back out as picture ids', () => {
  const items = pictureItems([
    {id: 'up9', name: 'shot.png', mime_type: 'image/png', size_bytes: 3000, width: 640, height: 480, model_reads: false},
    {id: 'up8', mime_type: 'image/png', size_bytes: 10, width: 1, height: 1},
  ]);
  assert.deepEqual(items.map((item) => [item.kind, item.state, item.uploadId, item.url, item.modelReads]), [
    ['picture', 'ready', 'up9', '/api/uploads/up9', false],
    ['picture', 'ready', 'up8', '/api/uploads/up8', null],
  ]);
  assert.notEqual(items[0].id, items[1].id);
  assert.equal(items[0].name, 'shot.png');
  assert.deepEqual(requestOf(items), {images: ['up9', 'up8']});
  assert.deepEqual(pictureItems(undefined), []);
});

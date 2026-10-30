import assert from 'node:assert/strict';
import test from 'node:test';
import {
  PICTURES_SHOWN, failureNote, pictureFacts, pictureName, pictureRow, pictureUrl,
  picturesNode, setupPictureBox,
} from '../../support/gateway/static/pictures.mjs';
import {applyEvent, newView, readPictures} from '../../support/gateway/static/turns.mjs';

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.handlers = {}; this.className = ''; this.textContent = ''; }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  all(tag) { return this.children.flatMap((c) => [...(c.tag === tag ? [c] : []), ...c.all(tag)]); }
  byClass(name) { return this.children.flatMap((c) => [...(c.className.split(' ').includes(name) ? [c] : []), ...c.byClass(name)]); }
}
const doc = {createElement: (tag) => new Node(tag)};
const pic = (path, extra = {}) => ({path, size: 84_000, mtime: 1_761_000_000.5, state: 'new', ...extra});
const many = (count) => Array.from({length: count}, (_, index) => pic(`out/p${index}.png`));
const tick = () => new Promise((resolve) => setImmediate(resolve));

test('readPictures keeps usable entries and drops the rest', () => {
  assert.deepEqual(readPictures(undefined), []);
  assert.deepEqual(readPictures('x'), []);
  const kept = readPictures([pic('a.png'), null, 7, {path: ''}, {path: 3}, {path: 'b.svg', size: 'big', state: 4}]);
  assert.deepEqual(kept.map((item) => item.path), ['a.png', 'b.svg']);
  assert.deepEqual(kept[1], {path: 'b.svg', size: 0, mtime: 0, state: ''});
});

test('a name is the last part of the path and the url encodes the path', () => {
  assert.equal(pictureName(pic('charts/q 1/ai_capex.png')), 'ai_capex.png');
  assert.equal(pictureName(pic('a.png')), 'a.png');
  const url = pictureUrl('s/1', pic('charts/a&b #1.png'));
  assert.equal(url, '/api/sessions/s%2F1/pictures?path=charts%2Fa%26b%20%231.png&v=1761000000500');
});

test('the row shows four pictures and counts the rest', () => {
  assert.equal(PICTURES_SHOWN, 4);
  assert.deepEqual(pictureRow(many(3)).more, 0);
  assert.equal(pictureRow(many(4)).more, 0);
  const row = pictureRow(many(7));
  assert.equal(row.shown.length, 4);
  assert.equal(row.more, 3);
  assert.equal(pictureRow(many(7), true).shown.length, 7);
  assert.equal(pictureRow(many(7), true).more, 0);
});

test('facts are the size, and say when a picture was updated', () => {
  assert.equal(pictureFacts(pic('a.png')), '84 KB');
  assert.equal(pictureFacts(pic('a.png', {state: 'modified'})), '84 KB · updated');
});

test('failure notes say why a picture is not there', () => {
  assert.equal(failureNote(404), 'No longer in the folder');
  assert.equal(failureNote(413), 'Too large to preview');
  assert.equal(failureNote(415), 'Not a picture file');
  assert.equal(failureNote(403), "Outside the chat's folder");
  assert.equal(failureNote(0), 'Cannot be shown');
});

test('a turn.pictures event fills the turn, also when a reopened chat replays it', () => {
  const view = newView('s1');
  applyEvent(view, {id: 1, turn: 1, type: 'turn.started', ts: 1, data: {turn: 1, turn_id: 't1', user_text: 'chart', client_request_id: null}});
  assert.deepEqual(view.turns[0].pictures, []);
  applyEvent(view, {id: 2, turn: 1, type: 'turn.pictures', ts: 2, data: {turn: 1, turn_id: 't1', pictures: [pic('a.png'), {path: 7}]}});
  assert.deepEqual(view.turns[0].pictures.map((item) => item.path), ['a.png']);
  const replay = newView('s1');
  replay.turns = [{...view.turns[0], pictures: []}];
  applyEvent(replay, {id: 2, turn: 1, type: 'turn.pictures', ts: 2, data: {pictures: [pic('b.svg')]}}, {ignoreCursor: true});
  assert.equal(replay.turns[0].pictures[0].path, 'b.svg');
});

function rig(pictures, options = {}) {
  const opened = [];
  let toggled = 0;
  const node = picturesNode({
    doc, pictures, sessionId: 's1', open: (picture, url) => opened.push([picture.path, url]), toggle: () => { toggled += 1; },
    ...options,
  });
  return {node, opened, toggled: () => toggled};
}

test('no pictures draw nothing', () => {
  assert.equal(picturesNode({doc, pictures: [], sessionId: 's1', open() {}, toggle() {}}), null);
});

test('previews are buttons with the name and size, and a click opens one', () => {
  const t = rig([pic('charts/fcf.png'), pic('plot.svg', {size: 4_100})]);
  assert.equal(t.node.attributes['aria-label'], 'Pictures from this turn');
  const tiles = t.node.all('button');
  assert.equal(tiles.length, 2);
  assert.ok(tiles.every((tile) => tile.type === 'button' && tile.className === 'picture'));
  const names = t.node.byClass('picture-name').map((item) => item.textContent);
  assert.deepEqual(names, ['fcf.png', 'plot.svg']);
  assert.deepEqual(t.node.byClass('picture-facts').map((item) => item.textContent), ['84 KB', '4.1 KB']);
  const image = t.node.all('img')[0];
  assert.equal(image.alt, '');
  assert.equal(image.src, '/api/sessions/s1/pictures?path=charts%2Ffcf.png&v=1761000000500');
  tiles[1].handlers.click();
  assert.deepEqual(t.opened, [['plot.svg', '/api/sessions/s1/pictures?path=plot.svg&v=1761000000500']]);
  assert.equal(t.node.byClass('picture-more').length, 0);
});

test('a single picture is named in the singular', () => {
  assert.equal(rig([pic('a.png')]).node.attributes['aria-label'], 'Picture from this turn');
});

test('more than four show four and a +N more button that toggles', () => {
  const t = rig(many(7));
  assert.equal(t.node.all('img').length, 4);
  const [more] = t.node.byClass('picture-more');
  assert.equal(more.textContent, '+3 more');
  assert.equal(more.attributes['aria-label'], 'Show 3 more pictures');
  assert.equal(more.attributes['aria-expanded'], 'false');
  more.handlers.click();
  assert.equal(t.toggled(), 1);
  const open = rig(many(7), {expanded: true});
  assert.equal(open.node.all('img').length, 7);
  const [fewer] = open.node.byClass('picture-more');
  assert.equal(fewer.textContent, 'Show fewer');
  assert.equal(fewer.attributes['aria-expanded'], 'true');
  assert.equal(rig(many(5)).node.byClass('picture-more')[0].attributes['aria-label'], 'Show 1 more picture');
});

test('a picture that is gone is replaced by a note, not a broken image', async () => {
  const asked = [];
  const t = rig([pic('gone.png')], {status: async (url) => { asked.push(url); return 404; }});
  const [tile] = t.node.all('button');
  await tile.children[0].children[0].handlers.error();
  assert.equal(asked.length, 1);
  assert.equal(tile.disabled, true);
  assert.equal(tile.children[0].children[0].textContent, 'No longer in the folder');
  assert.equal(t.node.byClass('picture-facts')[0].textContent, '');
});

test('a gateway that does not answer gives the generic note', async () => {
  const t = rig([pic('a.png')], {status: async () => { throw new Error('down'); }});
  const [tile] = t.node.all('button');
  await tile.children[0].children[0].handlers.error();
  await tick();
  assert.equal(tile.children[0].children[0].textContent, 'Cannot be shown');
});

function box() {
  const dialog = new Node('dialog');
  dialog.ownerDocument = doc;
  dialog.open = false;
  dialog.showModal = () => { dialog.open = true; };
  dialog.close = () => { dialog.open = false; };
  const open = setupPictureBox({dialog});
  const [image, bar] = dialog.children;
  const [caption, close] = bar.children;
  return {dialog, image, caption, close, open, handlers: dialog.handlers};
}

test('the light box shows the picture at full size and closes three ways', () => {
  const b = box();
  b.open(pic('charts/fcf.png', {state: 'modified'}), '/u');
  assert.equal(b.dialog.open, true);
  assert.equal(b.image.src, '/u');
  assert.equal(b.image.alt, 'fcf.png');
  assert.equal(b.image.className, '');
  assert.equal(b.caption.textContent, 'fcf.png · 84 KB · updated');
  b.close.handlers.click();
  assert.equal(b.dialog.open, false);
  b.open(pic('a.png'), '/u');
  b.handlers.click({target: b.image});
  assert.equal(b.dialog.open, true, 'a click on the picture keeps it open');
  b.handlers.click({target: b.dialog});
  assert.equal(b.dialog.open, false, 'a click outside the picture closes it');
});

test('an SVG in the light box gets the vector size', () => {
  const b = box();
  b.open(pic('plot.SVG'), '/u');
  assert.equal(b.image.className, 'vector');
});

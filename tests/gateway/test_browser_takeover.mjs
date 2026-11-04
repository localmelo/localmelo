/**
 * Taking over melo's browser in the Browser panel (`browser-live.mjs`): a point on the picture becomes a point on the
 * page by the frame's metadata (its size, the space above it and its zoom), keys, an input method's composition and
 * a paste become the gateway's messages, and the panel shows who is in control. The panel runs on a fake socket and a
 * DOM double that dispatches events; nothing typed stays in the panel's text box.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ALT,
  META,
  SHIFT,
  MAX_PASTE,
  commitMessage,
  composeMessage,
  control,
  keyMessage,
  mouseMessage,
  parseMeta,
  pasteMessage,
  setupBrowser,
  toPage,
  wheelMessage,
  wheelPixels,
} from '../../support/gateway/static/browser-live.mjs';

const message = (extra = {}) => JSON.stringify({type: 'meta', state: 'live', tab: 't1', url: 'https://example.com/', title: 'Example', browsing: false, owner: 'melo', w: 1280, h: 713, offsetTop: 0, scale: 1, takeover: true, ...extra});

// ── Mapping a point ──

test('a point on the picture maps to the page by its size, the space above it and its zoom', () => {
  const box = {width: 640, height: 400}; // the picture is drawn at half size: 640 x 356.5
  assert.deepEqual(toPage(parseMeta(message()), box, {x: 320, y: 100}), {x: 640, y: 200});
  // The browser keeps 40 pixels above the page.
  assert.deepEqual(toPage(parseMeta(message({offsetTop: 40, h: 753})), {width: 640, height: 400}, {x: 10, y: 60}), {x: 20, y: 80});
  // The page is zoomed in twice: the same point on the picture is half as far into the page.
  assert.deepEqual(toPage(parseMeta(message({scale: 2})), box, {x: 320, y: 100}), {x: 320, y: 100});
  assert.deepEqual(toPage(parseMeta(message({scale: 1.5, offsetTop: 30, h: 743})), {width: 1280, height: 743}, {x: 300, y: 330}), {x: 200, y: 200});
});

test('a drag that leaves the picture still ends on the page', () => {
  const meta = parseMeta(message());
  const box = {width: 640, height: 400};
  assert.equal(toPage(meta, box, {x: 700, y: 100}), null);
  assert.deepEqual(toPage(meta, box, {x: 700, y: -20}, {clamp: true}), {x: 1280, y: 0});
  const up = mouseMessage(meta, box, {x: 900, y: 500}, {button: 0, buttons: 0, detail: 1}, 'up', {clamp: true});
  assert.deepEqual([up.x, up.y, up.action, up.button], [1280, 713, 'up', 'left']);
});

test('a press, a drag and a wheel step are the gateway\'s mouse messages', () => {
  const meta = parseMeta(message());
  const box = {width: 1280, height: 713};
  assert.deepEqual(mouseMessage(meta, box, {x: 10, y: 20}, {button: 0, buttons: 1, detail: 2, shiftKey: true}, 'down'),
    {type: 'mouse', action: 'down', x: 10, y: 20, buttons: 1, mods: SHIFT, button: 'left', clicks: 2});
  assert.deepEqual(mouseMessage(meta, box, {x: 11, y: 20}, {buttons: 1}, 'move'),
    {type: 'mouse', action: 'move', x: 11, y: 20, buttons: 1, mods: 0, button: 'left'});
  assert.equal(mouseMessage(meta, box, {x: 11, y: 20}, {buttons: 0}, 'move').button, 'none');
  assert.equal(mouseMessage(meta, box, {x: 1, y: 1}, {button: 2, buttons: 2, detail: 9}, 'down').clicks, 3);
  // A wheel step scrolls the page as far as it scrolls the picture.
  assert.deepEqual(wheelMessage(meta, {width: 640, height: 400}, {x: 10, y: 10}, 0, 100), {type: 'wheel', x: 20, y: 20, dx: 0, dy: 200, mods: 0});
  assert.equal(wheelMessage(meta, box, {x: 10, y: 10}, 0, 0), null);
  assert.deepEqual(wheelPixels({deltaMode: 1, deltaX: 0, deltaY: 3}), {dx: 0, dy: 120});
});

// ── Keys, an input method and a paste ──

test('a key that types, a named key and a shortcut are key messages; the paste shortcut and an input method\'s keys are not', () => {
  assert.deepEqual(keyMessage({key: 'a', code: 'KeyA', keyCode: 65}, 'down'), {type: 'key', action: 'down', key: 'a', code: 'KeyA', mods: 0, keyCode: 65});
  assert.deepEqual(keyMessage({key: 'Enter', code: 'Enter', keyCode: 13}, 'down'), {type: 'key', action: 'down', key: 'Enter', code: 'Enter', mods: 0});
  assert.equal(keyMessage({key: 'a', code: 'KeyA', metaKey: true, keyCode: 65}, 'down').mods, META);
  assert.equal(keyMessage({key: 'v', code: 'KeyV', metaKey: true}, 'down'), null, 'a paste comes as text');
  assert.equal(keyMessage({key: 'n', code: 'KeyN', isComposing: true}, 'down'), null);
  assert.equal(keyMessage({key: 'Process', keyCode: 229}, 'down'), null);
  assert.equal(keyMessage({key: 'Dead', code: 'KeyE', altKey: true}, 'down'), null);
  assert.equal(keyMessage({key: 'MediaPlayPause'}, 'down'), null);
  assert.equal(keyMessage({key: 'å', code: 'KeyA', altKey: true, keyCode: 65, repeat: true}, 'down').repeat, true);
  assert.equal(keyMessage({key: 'å', code: 'KeyA', altKey: true}, 'down').mods, ALT);
});

test('an input method composes, then commits its text; a paste is plain text up to the cap', () => {
  assert.deepEqual(composeMessage('nihao'), {type: 'compose', text: 'nihao', start: 5, end: 5});
  assert.deepEqual(commitMessage('你好'), {type: 'text', text: '你好'});
  assert.equal(commitMessage(''), null);
  assert.deepEqual(pasteMessage('a\r\nb'), {type: 'paste', text: 'a\nb'});
  assert.equal(pasteMessage(''), null);
  assert.equal(pasteMessage('x'.repeat(MAX_PASTE + 1)), null);
  assert.equal(new TextEncoder().encode(JSON.stringify(pasteMessage('密'.repeat(MAX_PASTE)))).length < 4096, true, 'a full paste fits the gateway\'s message cap');
});

// ── Who is in control ──

test('the panel offers Take over while melo has the page and says who is in control', () => {
  const melo = control(parseMeta(message()));
  assert.deepEqual([melo.you, melo.canTake, melo.status, melo.hint], [false, true, '', '']);
  assert.equal(control(parseMeta(message({takeover: false}))).canTake, false);
  assert.equal(control(parseMeta(message()), {link: 'retry'}).canTake, false);
  const you = control(parseMeta(message({owner: 'you', grace: 120})), {focused: false});
  assert.equal(you.you, true);
  assert.equal(you.status, 'You are in control');
  assert.equal(you.hint, 'Click the page to type. If you leave this panel, melo gets the page back after 2 min.');
  assert.match(control(parseMeta(message({owner: 'you'})), {focused: true}).hint, /Control-Esc/);
  const finishing = control(parseMeta(message({owner: 'you', browsing: true})));
  assert.equal(finishing.finishing, true);
  assert.match(finishing.hint, /finishing a step/);
  const box = control(parseMeta(message({owner: 'you', dialog: {kind: 'prompt', message: 'Name?', default: 'Alex'}})));
  assert.deepEqual(box.dialog, {kind: 'prompt', message: 'Name?', default: 'Alex'});
  assert.equal(parseMeta(message({dialog: {kind: 'prompt', message: 'x'}})).dialog, null, 'only the person sees a box');
  assert.equal(parseMeta(message({owner: 'you', dialog: {kind: 'exec', message: 'x'}})).dialog, null);
});

// ── The panel ──

function node(extra = {}) {
  const attributes = {};
  const listeners = {};
  return {
    hidden: false,
    textContent: '',
    dataset: {},
    style: {},
    value: '',
    setAttribute(key, value) { attributes[key] = String(value); },
    getAttribute(key) { return attributes[key] ?? null; },
    addEventListener(type, fn) { (listeners[type] ??= []).push(fn); },
    fire(type, event = {}) {
      const full = {defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...event};
      for (const fn of listeners[type] ?? []) fn(full);
      return full;
    },
    focus() { this.focused = true; this.fire('focus'); },
    blur() { if (this.focused) { this.focused = false; this.fire('blur'); } },
    getBoundingClientRect() { return {left: 100, top: 50}; },
    setPointerCapture() {},
    setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; },
    attributes,
    ...extra,
  };
}

class FakeSocket {
  static all = [];
  constructor(url) {
    this.url = url;
    this.sent = [];
    FakeSocket.all.push(this);
  }
  close() {}
  send(text) { this.sent.push(JSON.parse(text)); }
  open() { this.onopen?.(); }
  say(data) { this.onmessage?.({data}); }
}

async function panel() {
  FakeSocket.all = [];
  const names = ['root', 'chip', 'chipText', 'title', 'host', 'rest', 'status', 'note', 'hide', 'pop', 'take', 'give', 'viewOnly', 'hint', 'dialog', 'dialogTitle', 'dialogText', 'dialogInput', 'dialogOk', 'dialogCancel'];
  const els = Object.fromEntries(names.map((name) => [name, node()]));
  els.viewport = node({clientWidth: 640, clientHeight: 400});
  els.canvas = node({width: 300, height: 150, getContext() { return {drawImage() {}}; }});
  els.keys = node();
  const frames = [];
  const browser = setupBrowser({
    api: async () => ({available: true, open: true, browsing: true}),
    els,
    openSocket: (url) => new FakeSocket(url),
    decode: () => Promise.resolve({width: 1280, height: 713, close() {}}),
    location: {protocol: 'http:', host: 'gw.test'},
    schedule: () => 0,
    cancel: () => {},
    frame: (fn) => frames.push(fn),
  });
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.open();
  socket.say(message());
  return {browser, els, socket, frames, sent: () => socket.sent};
}

test('Take over asks the gateway, and the keys go to the page once it is the person\'s', async () => {
  const {els, socket} = await panel();
  assert.equal(els.take.hidden, false);
  assert.equal(els.viewOnly.hidden, true);
  assert.equal(els.give.hidden, true);
  els.take.fire('click');
  assert.deepEqual(socket.sent, [{type: 'take'}]);
  socket.say(message({owner: 'you', grace: 120}));
  assert.equal(els.root.dataset.owner, 'you');
  assert.equal(els.status.textContent, 'You are in control');
  assert.equal(els.take.hidden, true);
  assert.equal(els.give.hidden, false);
  assert.equal(els.keys.focused, true, 'the keys went to the page');
  assert.equal(els.root.dataset.keys, 'true');
  assert.match(els.hint.textContent, /Control-Esc/);
  els.give.fire('click');
  assert.deepEqual(socket.sent.at(-1), {type: 'give'});
  socket.say(message());
  assert.equal(els.root.dataset.owner, 'melo');
  assert.equal(els.keys.focused, false);
});

test('nothing is sent while melo has the page', async () => {
  const {els, socket} = await panel();
  els.canvas.fire('pointerdown', {clientX: 120, clientY: 60, button: 0, buttons: 1, detail: 1});
  els.keys.fire('keydown', {key: 'a', code: 'KeyA', keyCode: 65});
  els.keys.fire('paste', {clipboardData: {getData: () => 'secret'}});
  assert.deepEqual(socket.sent, []);
});

test('a click on the picture is a click on the page, and moves are sent once a frame', async () => {
  const {els, socket, frames} = await panel();
  socket.say(message({owner: 'you'}));
  // The picture is drawn at half size in the 640 x 400 box, which starts at (100, 50) on the screen.
  const down = els.canvas.fire('pointerdown', {clientX: 420, clientY: 150, button: 0, buttons: 1, detail: 1, pointerId: 1});
  assert.equal(down.defaultPrevented, true);
  els.canvas.fire('pointermove', {clientX: 421, clientY: 150, buttons: 1});
  els.canvas.fire('pointermove', {clientX: 422, clientY: 151, buttons: 1});
  assert.equal(frames.length, 1, 'one frame for both moves');
  frames.shift()();
  els.canvas.fire('pointerup', {clientX: 422, clientY: 151, button: 0, buttons: 0, detail: 1});
  assert.deepEqual(socket.sent.map((m) => [m.type, m.action, m.x, m.y]), [
    ['mouse', 'down', 640, 200],
    ['mouse', 'move', 644, 202],
    ['mouse', 'up', 644, 202],
  ]);
  assert.equal(els.keys.style.left, '320px', 'the text box sits where the person clicked, for an input method\'s window');
});

test('typing sends each key and keeps nothing in the text box', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you'}));
  els.canvas.fire('pointerdown', {clientX: 120, clientY: 60, button: 0, buttons: 1, detail: 1});
  socket.sent.length = 0;
  for (const key of ['h', 'i']) {
    const down = els.keys.fire('keydown', {key, code: `Key${key.toUpperCase()}`, keyCode: key.toUpperCase().charCodeAt(0)});
    assert.equal(down.defaultPrevented, true);
    els.keys.fire('keyup', {key, code: `Key${key.toUpperCase()}`});
  }
  assert.deepEqual(socket.sent.map((m) => `${m.action}:${m.key}`), ['down:h', 'up:h', 'down:i', 'up:i']);
  const paste = els.keys.fire('keydown', {key: 'v', code: 'KeyV', metaKey: true});
  assert.equal(paste.defaultPrevented, false, 'the paste shortcut is left to the browser');
  assert.equal(els.keys.value, ' ', 'only the panel\'s own space');
});

test('an input method\'s composition goes as it is composed, and its commit as text', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you'}));
  els.canvas.fire('pointerdown', {clientX: 120, clientY: 60, button: 0, buttons: 1, detail: 1});
  socket.sent.length = 0;
  els.keys.fire('compositionstart', {data: ''});
  for (const data of ['n', 'ni', 'nih', 'niha', 'nihao']) {
    els.keys.fire('keydown', {key: 'Process', keyCode: 229, isComposing: true});
    els.keys.fire('compositionupdate', {data});
    els.keys.value = ` ${data}`;
    els.keys.fire('input', {data, inputType: 'insertCompositionText', isComposing: true});
  }
  els.keys.fire('compositionend', {data: '你好'});
  els.keys.fire('input', {data: '你好', inputType: 'insertText', isComposing: false});
  // No key of the input method is sent, and the commit that the browser says again as an input goes once.
  assert.deepEqual(socket.sent, [
    ...['n', 'ni', 'nih', 'niha', 'nihao'].map((text) => ({type: 'compose', text, start: text.length, end: text.length})),
    {type: 'text', text: '你好'},
  ]);
  assert.equal(els.keys.value, ' ');
  // Text that comes by no key (the emoji picker) is sent as text.
  els.keys.fire('input', {data: '🙂', inputType: 'insertText', isComposing: false});
  assert.deepEqual(socket.sent.at(-1), {type: 'text', text: '🙂'});
});

test('a cancelled composition clears what was composed on the page', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you'}));
  socket.sent.length = 0;
  els.keys.fire('compositionstart', {});
  els.keys.fire('compositionupdate', {data: 'ni'});
  els.keys.fire('compositionend', {data: ''});
  assert.deepEqual(socket.sent, [{type: 'compose', text: 'ni', start: 2, end: 2}, {type: 'compose', text: '', start: 0, end: 0}]);
});

test('a paste is sent as plain text, and one too long is refused with a word', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you'}));
  socket.sent.length = 0;
  const paste = els.keys.fire('paste', {clipboardData: {getData: (kind) => (kind === 'text/plain' ? 'Alex\r\nexample.com' : '<b>no</b>')}});
  assert.equal(paste.defaultPrevented, true);
  assert.deepEqual(socket.sent, [{type: 'paste', text: 'Alex\nexample.com'}]);
  els.keys.fire('paste', {clipboardData: {getData: () => 'x'.repeat(MAX_PASTE + 1)}});
  assert.equal(socket.sent.length, 1);
  assert.equal(els.hint.textContent, `Paste up to ${MAX_PASTE} characters at a time.`);
});

test('the macOS app\'s menu shortcuts reach the page as their keys', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you'}));
  socket.sent.length = 0;
  els.keys.value = ' ';
  els.keys.selectionStart = 0;
  els.keys.selectionEnd = 1;
  els.keys.fire('select');
  els.keys.fire('beforeinput', {inputType: 'historyUndo'});
  els.keys.fire('cut');
  assert.deepEqual(socket.sent.filter((m) => m.action === 'down').map((m) => [m.key, m.mods]), [['a', META], ['z', META], ['x', META]]);
});

test('Control-Esc moves out of the page to Give back', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you'}));
  els.canvas.fire('pointerdown', {clientX: 120, clientY: 60, button: 0, buttons: 1, detail: 1});
  socket.sent.length = 0;
  let given = false;
  els.give.focus = () => { given = true; };
  els.keys.fire('keydown', {key: 'Escape', code: 'Escape', ctrlKey: true});
  assert.equal(given, true);
  assert.equal(els.keys.focused, false);
  assert.deepEqual(socket.sent, []);
});

test('a box the page puts up is answered in the panel', async () => {
  const {els, socket} = await panel();
  socket.say(message({owner: 'you', dialog: {kind: 'prompt', message: 'Your name?', default: 'Alex'}}));
  assert.equal(els.dialog.hidden, false);
  assert.equal(els.dialogText.textContent, 'Your name?');
  assert.equal(els.dialogInput.hidden, false);
  assert.equal(els.dialogInput.value, 'Alex');
  els.dialogInput.value = 'Sam';
  socket.sent.length = 0;
  els.dialogOk.fire('click');
  assert.deepEqual(socket.sent, [{type: 'dialog', accept: true, text: 'Sam'}]);
  assert.equal(els.dialogInput.value, '');
  socket.say(message({owner: 'you', dialog: {kind: 'beforeunload', message: ''}}));
  assert.equal(els.dialogTitle.textContent, 'The page asks before you leave it');
  assert.equal(els.dialogOk.textContent, 'Leave');
  socket.say(message({owner: 'you'}));
  assert.equal(els.dialog.hidden, true);
});

test('the card\'s Take over opens the panel and gives the keys to the page when the gateway hands it over', async () => {
  const {browser, els, socket} = await panel();
  browser.hide();
  assert.equal(els.root.hidden, true);
  browser.takeOver();
  assert.equal(els.root.hidden, false);
  const reopened = FakeSocket.all.at(-1);
  reopened.open();
  reopened.say(message({owner: 'you'}));
  assert.equal(els.keys.focused, true);
  assert.deepEqual(reopened.sent, [], 'the card already took it');
  assert.ok(socket);
});

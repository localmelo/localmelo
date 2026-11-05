/**
 * The Browser panel (`browser-live.mjs`): the pure parts (the address, the gateway's `meta` message, the picture's
 * place in its box and the page point under a point on it, the words of each state), and the panel itself on a fake
 * socket and a DOM double: it opens by itself when the chat has a page or melo starts browsing, keeps only the
 * newest picture, follows a tab switch, shows the calm states, retries a lost link, and hides (holding no socket).
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  addressParts,
  layout,
  liveRoute,
  liveUrl,
  parseMeta,
  retryDelay,
  setupBrowser,
  socketUrl,
  toPage,
  view,
} from '../../support/gateway/static/browser-live.mjs';

// ── The pure parts ──

test('Pop out opens the chat\'s live page on this page\'s own origin', () => {
  assert.equal(liveUrl({origin: 'http://127.0.0.1:9000'}, 'a b'), 'http://127.0.0.1:9000/#/live/a%20b');
  assert.equal(liveRoute(new URL(liveUrl({origin: 'http://127.0.0.1:9000'}, 's1')).hash), 's1');
});

test('a #/live address names the chat; anything else names none', () => {
  assert.equal(liveRoute('#/live/abc-1.2:x'), 'abc-1.2:x');
  assert.equal(liveRoute('#/live/'), '');
  assert.equal(liveRoute('#/s/abc'), '');
  assert.equal(liveRoute('#/live/a/b'), '');
  assert.equal(liveRoute(undefined), '');
});

test('the socket address is on the page\'s own origin and follows its scheme', () => {
  assert.equal(socketUrl({protocol: 'http:', host: '127.0.0.1:9000'}, 'a b'), 'ws://127.0.0.1:9000/api/sessions/a%20b/browser/live');
  assert.equal(socketUrl({protocol: 'https:', host: 'x.test'}, 's1', 't2'), 'wss://x.test/api/sessions/s1/browser/live?tab=t2');
});

test('the retry delay grows and stops at thirty seconds', () => {
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6, 40].map(retryDelay), [1, 2, 4, 8, 16, 30, 30, 30]);
  assert.equal(retryDelay(-3), 1);
});

test('an address is the site and what follows it', () => {
  assert.deepEqual(addressParts('https://example.com/'), {host: 'example.com', rest: ''});
  assert.deepEqual(addressParts('https://example.com:8443/a/b?q=1#top'), {host: 'example.com:8443', rest: '/a/b?q=1'});
  assert.deepEqual(addressParts('about:blank'), {host: 'about:blank', rest: ''});
  assert.deepEqual(addressParts('not a url'), {host: 'not a url', rest: ''});
  assert.deepEqual(addressParts(''), {host: '', rest: ''});
  assert.deepEqual(addressParts(null), {host: '', rest: ''});
});

const message = (extra = {}) => JSON.stringify({type: 'meta', state: 'live', tab: 't1', url: 'https://example.com/a', title: 'Example', browsing: false, owner: 'melo', w: 1280, h: 713, offsetTop: 0, scale: 1, ...extra});

test('a meta message is read for what it says and nothing else', () => {
  const meta = parseMeta(message({browsing: true}));
  assert.equal(meta.state, 'live');
  assert.equal(meta.tab, 't1');
  assert.equal(meta.title, 'Example');
  assert.equal(meta.browsing, true);
  assert.equal(meta.owner, 'melo');
  assert.deepEqual([meta.w, meta.h, meta.offsetTop, meta.scale], [1280, 713, 0, 1]);
});

test('a message that is not a meta, or says an unknown state, is ignored', () => {
  assert.equal(parseMeta('{'), null);
  assert.equal(parseMeta('[]'), null);
  assert.equal(parseMeta('null'), null);
  assert.equal(parseMeta(JSON.stringify({type: 'frame', state: 'live'})), null);
  assert.equal(parseMeta(JSON.stringify({type: 'meta', state: 'exploding'})), null);
});

test('the parts of a meta message are cleaned', () => {
  const meta = parseMeta(message({tab: '<script>', owner: 'root', reason: 'because', title: 'x'.repeat(5000), w: -4, h: 'tall', browsing: 'yes'}));
  assert.equal(meta.tab, '');
  assert.equal(meta.owner, 'melo');
  assert.equal(meta.reason, '');
  assert.equal(meta.title.length, 2048);
  assert.equal(meta.browsing, false);
  assert.deepEqual([meta.w, meta.h], [0, 0]);
  assert.equal(parseMeta(message({state: 'closed', reason: 'browser'})).reason, 'browser');
  assert.equal(parseMeta(message({owner: 'you'})).owner, 'you');
  assert.equal(parseMeta(message({scale: 0, offsetTop: -5})).scale, 1);
});

test('the picture is the largest copy of the page view that fits, flush with the top', () => {
  const meta = parseMeta(message());
  // Width is the limit: 400 / 1280.
  assert.deepEqual(layout(meta, {width: 400, height: 300}), {left: 0, top: 0, width: 400, height: 222, k: 0.3125});
  // Height is the limit: the picture is centred across.
  const tall = layout(meta, {width: 1000, height: 200});
  assert.equal(tall.height, 200);
  assert.equal(tall.width, Math.floor(1280 * (200 / 713)));
  assert.equal(tall.left, Math.floor((1000 - tall.width) / 2));
  assert.equal(tall.top, 0);
});

test('there is no place for a picture until its size and the box are known', () => {
  assert.equal(layout(null, {width: 400, height: 300}), null);
  assert.equal(layout(parseMeta(message({w: 0})), {width: 400, height: 300}), null);
  assert.equal(layout(parseMeta(message()), {width: 0, height: 300}), null);
  assert.equal(layout(parseMeta(message()), undefined), null);
});

test('a point on the picture is the page point of the same place, by the frame metadata', () => {
  const meta = parseMeta(message());
  const box = {width: 400, height: 300};
  assert.deepEqual(toPage(meta, box, {x: 0, y: 0}), {x: 0, y: 0});
  assert.deepEqual(toPage(meta, box, {x: 200, y: 111}), {x: 640, y: 355.2});
  assert.deepEqual(toPage(meta, box, {x: 400, y: 100}), {x: 1280, y: 320});
});

test('a letterboxed picture maps from its own edge, not the box\'s', () => {
  const meta = parseMeta(message());
  const box = {width: 1000, height: 200};
  const fit = layout(meta, box);
  const at = toPage(meta, box, {x: fit.left, y: 0});
  assert.deepEqual(at, {x: 0, y: 0});
  const middle = toPage(meta, box, {x: fit.left + fit.width / 2, y: 100});
  assert.ok(Math.abs(middle.x - 640) < 1.5, String(middle.x));
  assert.ok(Math.abs(middle.y - 356.5) < 1.5, String(middle.y));
  // Beside the picture is not on the page.
  assert.equal(toPage(meta, box, {x: fit.left - 2, y: 50}), null);
  assert.equal(toPage(meta, box, {x: fit.left + fit.width + 2, y: 50}), null);
});

test('the space the browser keeps above the page is not the page', () => {
  const meta = parseMeta(message({offsetTop: 40}));
  const box = {width: 1280, height: 800};
  assert.deepEqual(toPage(meta, box, {x: 10, y: 100}), {x: 10, y: 60});
  assert.equal(toPage(meta, box, {x: 10, y: 20}), null);
});

test('a point that is not a number maps to nothing', () => {
  const meta = parseMeta(message());
  assert.equal(toPage(meta, {width: 400, height: 300}, {x: NaN, y: 1}), null);
  assert.equal(toPage(meta, {width: 400, height: 300}, null), null);
  assert.equal(toPage(null, {width: 400, height: 300}, {x: 1, y: 1}), null);
});

// ── The states in words ──

test('before anything is known the panel is idle, with an invitation and no status', () => {
  const v = view(null, 'off');
  assert.equal(v.state, 'idle');
  assert.equal(v.title, 'No page open');
  assert.match(v.note, /you can watch it here/);
  assert.equal(v.status, '');
  assert.equal(v.picture, false);
});

test('a chat that has a page and no picture yet is loading', () => {
  const v = view(null, 'connecting', {known: true});
  assert.equal(v.state, 'loading');
  assert.equal(v.status, 'Opening the page…');
  const named = view(parseMeta(message({state: 'loading', title: '', url: 'https://example.com/x'})), 'open');
  assert.equal(named.title, 'Untitled page');
  assert.equal(named.host, 'example.com');
  assert.equal(named.rest, '/x');
});

test('a live page says Live, and says melo is browsing while a browser step runs', () => {
  const calm = view(parseMeta(message()), 'open', {picture: true});
  assert.equal(calm.state, 'live');
  assert.equal(calm.status, 'Live');
  assert.equal(calm.browsing, false);
  assert.equal(calm.picture, true);
  assert.equal(calm.title, 'Example');
  const busy = view(parseMeta(message({browsing: true})), 'open', {picture: true});
  assert.equal(busy.browsing, true);
  assert.equal(busy.status, 'melo is browsing');
});

test('a page with no picture drawn yet shows its note, not an empty frame', () => {
  const v = view(parseMeta(message()), 'open', {picture: false});
  assert.equal(v.picture, false);
});

test('a lost link keeps the last picture and says it is reconnecting', () => {
  const v = view(parseMeta(message({browsing: true})), 'retry', {picture: true});
  assert.equal(v.state, 'stale');
  assert.equal(v.browsing, false);
  assert.equal(v.status, 'Reconnecting…');
  assert.equal(v.picture, true);
  const early = view(parseMeta(message({state: 'loading'})), 'retry', {picture: false});
  assert.equal(early.state, 'loading');
});

test('a stream that cannot update says so and keeps the picture', () => {
  const v = view(parseMeta(message({state: 'stale'})), 'open', {picture: true});
  assert.equal(v.state, 'stale');
  assert.equal(v.status, 'Not updating');
  assert.match(v.note, /last picture/);
  assert.equal(v.picture, true);
});

test('a closed view says why in a sentence, and shows no picture', () => {
  for (const [reason, words] of [['tab', 'This tab was closed.'], ['browser', "melo's browser stopped."], ['chat', 'This chat ended.'], ['unavailable', 'The live view is not available for this chat.'], ['', 'This tab was closed.']]) {
    const v = view(parseMeta(message({state: 'closed', reason, browsing: true})), 'open', {picture: true});
    assert.equal(v.state, 'closed');
    assert.equal(v.note, words);
    assert.equal(v.picture, false);
    assert.equal(v.browsing, false);
  }
});

test('a page with no title, or one that is called by its address, is untitled, and its address says where it is', () => {
  const none = view(parseMeta(message({title: '', url: 'https://a.test/'})), 'open');
  assert.equal(none.title, 'Untitled page');
  assert.equal(none.host, 'a.test');
  assert.equal(view(parseMeta(message({title: 'a.test/x', url: 'https://a.test/x'})), 'open').title, 'Untitled page');
  assert.equal(view(parseMeta(message({title: 'https://a.test/', url: 'https://a.test/'})), 'open').title, 'Untitled page');
  assert.equal(view(parseMeta(message({title: 'About a.test', url: 'https://a.test/'})), 'open').title, 'About a.test');
  assert.equal(view(parseMeta(message({title: '', url: ''})), 'open').title, 'Untitled page');
});

// ── The panel ──

function node(extra = {}) {
  const attributes = {};
  return {
    hidden: false,
    textContent: '',
    dataset: {},
    style: {},
    setAttribute(key, value) { attributes[key] = String(value); },
    getAttribute(key) { return attributes[key] ?? null; },
    attributes,
    ...extra,
  };
}

class FakeSocket {
  static all = [];
  constructor(url) {
    this.url = url;
    this.closed = null;
    FakeSocket.all.push(this);
  }
  close(code) { this.closed = code ?? 1000; }
  open() { this.onopen?.(); }
  say(data) { this.onmessage?.({data}); }
  drop(code = 1006, reason = '') { this.onclose?.({code, reason}); }
}

function panel({wide = true, state = {available: true, open: true, browsing: true}, popOut = null, bridge = false} = {}) {
  FakeSocket.all = [];
  const els = {
    root: node({hidden: true}),
    chip: node({hidden: true}),
    chipText: node(),
    title: node(),
    host: node(),
    rest: node(),
    status: node(),
    viewport: node({clientWidth: 400, clientHeight: 300}),
    canvas: node({hidden: true, width: 300, height: 150, drawn: [], getContext() { return {drawImage: (bitmap) => els.canvas.drawn.push(bitmap)}; }}),
    note: node(),
    hide: node(),
    pop: node({hidden: true}),
  };
  const timers = [];
  const calls = {api: [], changed: [], decoded: [], denied: 0, popped: []};
  let apiState = state;
  const browser = setupBrowser({
    api: async (path) => {
      calls.api.push(path);
      if (apiState instanceof Error) throw apiState;
      return apiState;
    },
    els,
    openSocket: (url) => new FakeSocket(url),
    decode: (bytes) => {
      calls.decoded.push(bytes);
      return Promise.resolve({width: 1280, height: 713, close() { this.released = true; }, bytes});
    },
    location: {protocol: 'http:', host: 'gw.test'},
    schedule: (fn, ms) => { timers.push({fn, ms, cancelled: false}); return timers.length; },
    cancel: (id) => { timers[id - 1].cancelled = true; },
    wide: () => wide,
    popOut: popOut ?? (bridge ? (session) => calls.popped.push(session) : null),
    changed: (event) => calls.changed.push(event.shown),
    denied: () => { calls.denied += 1; },
  });
  return {browser, els, timers, calls, setApi: (next) => { apiState = next; }};
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

test('a chat melo is browsing in shows the panel and watches it', async () => {
  const {browser, els, calls} = panel();
  await browser.follow('s1');
  assert.deepEqual(calls.api, ['/sessions/s1/browser']);
  assert.equal(els.root.hidden, false);
  assert.deepEqual(calls.changed, [true]);
  assert.equal(FakeSocket.all.length, 1);
  assert.equal(FakeSocket.all[0].url, 'ws://gw.test/api/sessions/s1/browser/live');
  assert.equal(els.chip.hidden, false);
  assert.equal(els.root.dataset.state, 'loading');
});

test('a chat with no page open shows nothing, and no socket, until melo browses', async () => {
  const {browser, els, calls} = panel({state: {available: true, open: false}});
  await browser.follow('s1');
  assert.equal(els.root.hidden, true);
  assert.equal(els.chip.hidden, true);
  assert.equal(FakeSocket.all.length, 0);
  browser.note({type: 'tool.started', data: {name: 'read_file', call_id: 'c0'}});
  assert.equal(els.root.hidden, true);
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  assert.equal(els.root.hidden, false);
  assert.deepEqual(calls.changed, [true]);
  assert.equal(FakeSocket.all.length, 1);
  assert.equal(els.chip.hidden, false);
  assert.equal(els.chip.dataset.browsing, 'true', 'the step is running');
  assert.equal(els.root.dataset.browsing, 'true');
  browser.note({type: 'tool.finished', data: {call_id: 'c1'}});
  assert.equal(els.chip.dataset.browsing, 'false');
  assert.equal(els.root.dataset.browsing, 'false');
});

test('a chat whose browser cannot be watched never shows the panel or the chip', async () => {
  const {browser, els} = panel({state: {available: false, open: true}});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  assert.equal(els.root.hidden, true);
  assert.equal(els.chip.hidden, true);
  assert.equal(FakeSocket.all.length, 0);
});

test('a browser step that starts before the gateway has answered opens the panel when it does', async () => {
  const {browser, els} = panel({state: {available: true, open: false}});
  const following = browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c1'}});
  assert.equal(els.root.hidden, true);
  await following;
  assert.equal(els.root.hidden, false);
});

test('a chat with a page open that melo is not browsing shows the chip, not the panel', async () => {
  const {browser, els} = panel({state: {available: true, open: true, browsing: false}});
  await browser.follow('s1');
  assert.equal(els.root.hidden, true, 'the browsing ended: the panel waits behind the chip');
  assert.equal(els.chip.hidden, false);
  assert.equal(FakeSocket.all.length, 0);
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c1'}});
  assert.equal(els.root.hidden, false, 'melo browses again: it opens');
  const yours = panel({state: {available: true, open: true, browsing: false, yours: true}});
  await yours.browser.follow('s1');
  assert.equal(yours.els.root.hidden, false, 'a page the person took over opens with the chat');
});

test('a browser step in a chat that could not be watched asks the gateway again, so a switch to Chrome shows', async () => {
  const {browser, els, calls, setApi} = panel({state: {available: false, open: false}});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  await settle();
  assert.equal(els.root.hidden, true, 'still Safari: nothing to watch');
  assert.equal(calls.api.length, 2);
  // Settings -> Browser became Chrome: the chat's next turn browses in it.
  setApi({available: true, open: false});
  browser.note({type: 'turn.finished', data: {}});
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c2'}});
  await settle();
  assert.equal(calls.api.length, 3);
  assert.equal(els.root.hidden, false, 'the live view opens by itself');
  assert.equal(els.chip.hidden, false);
  assert.equal(FakeSocket.all.length, 1);
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c3'}});
  await settle();
  assert.equal(calls.api.length, 3, 'a chat that can be watched is not asked again');
});

test('a settings change is checked at once: the chip and panel follow the browser now chosen', async () => {
  const {browser, els, calls, setApi} = panel({state: {available: false, open: false}});
  await browser.follow('s1');
  setApi({available: true, open: false});
  await browser.recheck();
  assert.deepEqual(calls.api, ['/sessions/s1/browser', '/sessions/s1/browser']);
  assert.equal(els.root.hidden, true, 'nothing to show until melo browses');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  assert.equal(els.root.hidden, false, 'no second question: the answer is known');
  const socket = FakeSocket.all[0];
  setApi({available: false, open: false});
  await browser.recheck();
  assert.equal(els.root.hidden, true, 'Safari now: no live view');
  assert.equal(els.chip.hidden, true);
  assert.equal(socket.closed, 1000);
  setApi({available: true, open: true, browsing: true});
  await browser.recheck();
  assert.equal(els.root.hidden, false, 'not hidden by the person, so it may open again');
});

test('the panel that opened by itself goes when the browsing ends, and comes back at the next browser step', async () => {
  const {browser, els, calls} = panel({state: {available: true, open: false}});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  const socket = FakeSocket.all[0];
  browser.note({type: 'tool.finished', data: {call_id: 'c1'}});
  assert.equal(els.root.hidden, false, 'between two steps it stays');
  browser.note({type: 'turn.finished', data: {}});
  assert.equal(els.root.hidden, true);
  assert.equal(socket.closed, 1000, 'nobody watches: the stream stops');
  assert.equal(els.chip.hidden, false, 'the chip brings it back');
  assert.deepEqual(calls.changed, [true, false]);
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c2'}});
  assert.equal(els.root.hidden, false, 'it was not hidden by the person');
  assert.equal(FakeSocket.all.length, 2);
});

test('a panel the person opened, or a page they have, stays when the browsing ends', async () => {
  const opened = panel({state: {available: true, open: true, browsing: false}});
  await opened.browser.follow('s1');
  opened.browser.toggle();
  opened.browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c1'}});
  opened.browser.note({type: 'turn.finished', data: {}});
  assert.equal(opened.els.root.hidden, false);
  const held = panel({state: {available: true, open: false}});
  await held.browser.follow('s1');
  held.browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  FakeSocket.all.at(-1).say(message({owner: 'you'}));
  held.browser.note({type: 'turn.finished', data: {}});
  assert.equal(held.els.root.hidden, false, 'the person has the page');
});

test('a gateway that cannot be asked leaves the panel closed', async () => {
  const {browser, els} = panel({state: new Error('down')});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  assert.equal(els.root.hidden, true);
});

test('the meta message fills the title and address and the pictures are drawn', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.open();
  socket.say(message({browsing: true}));
  assert.equal(els.title.textContent, 'Example');
  assert.equal(els.host.textContent, 'example.com');
  assert.equal(els.rest.textContent, '/a');
  assert.equal(els.root.dataset.state, 'live');
  assert.equal(els.root.dataset.browsing, 'true');
  assert.equal(els.status.textContent, 'melo is browsing');
  assert.equal(els.chipText.textContent, 'melo is browsing');
  assert.equal(els.canvas.hidden, true, 'no picture has come yet');
  socket.say(new ArrayBuffer(8));
  await settle();
  assert.equal(els.canvas.hidden, false);
  assert.equal(els.canvas.drawn.length, 1);
  assert.equal(els.canvas.width, 1280);
  assert.equal(els.canvas.height, 713);
  assert.equal(els.canvas.drawn[0].released, true, 'the bitmap is let go once drawn');
  assert.equal(els.root.dataset.picture, 'true');
  assert.equal(els.note.hidden, true);
  // The picture is laid out in the viewport: 400 wide, its height from the page view's ratio.
  assert.deepEqual([els.canvas.style.left, els.canvas.style.top, els.canvas.style.width, els.canvas.style.height], ['0px', '0px', '400px', '222px']);
});

test('a slow decode drops the pictures in between', async () => {
  FakeSocket.all = [];
  const els = {
    root: node({hidden: true}), chip: node({hidden: true}), chipText: node(), title: node(), host: node(), rest: node(), status: node(),
    viewport: node({clientWidth: 400, clientHeight: 300}),
    canvas: node({hidden: true, width: 1, height: 1, drawn: [], getContext() { return {drawImage: (bitmap) => els.canvas.drawn.push(bitmap.tag)}; }}),
    note: node(), hide: node(), pop: node({hidden: true}),
  };
  const asked = [];
  const release = [];
  const browser = setupBrowser({
    api: async () => ({available: true, open: true, browsing: true}),
    els,
    openSocket: (url) => new FakeSocket(url),
    decode: (bytes) => {
      asked.push(new Uint8Array(bytes)[0]);
      return new Promise((resolve) => release.push(() => resolve({width: 64, height: 36, tag: new Uint8Array(bytes)[0]})));
    },
    location: {protocol: 'http:', host: 'gw.test'},
    schedule: () => 0,
    cancel: () => {},
  });
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message());
  for (const n of [1, 2, 3, 4]) socket.say(new Uint8Array([n]).buffer);
  assert.deepEqual(asked, [1], 'one decode at a time');
  release.shift()();
  await settle();
  assert.deepEqual(asked, [1, 4], 'the pictures between were never decoded');
  release.shift()();
  await settle();
  assert.deepEqual(els.canvas.drawn, [1, 4]);
});

test('a tab switch clears the old picture until the new tab\'s first one comes', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message());
  socket.say(new ArrayBuffer(4));
  await settle();
  assert.equal(els.canvas.hidden, false);
  socket.say(message({tab: 't2', url: 'https://other.test/', title: 'Other'}));
  assert.equal(els.canvas.hidden, true);
  assert.equal(els.title.textContent, 'Other');
  assert.equal(els.host.textContent, 'other.test');
  socket.say(new ArrayBuffer(4));
  await settle();
  assert.equal(els.canvas.hidden, false);
});

test('a same-tab update keeps the picture', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message());
  socket.say(new ArrayBuffer(4));
  await settle();
  socket.say(message({title: 'Renamed', url: 'https://example.com/b'}));
  assert.equal(els.canvas.hidden, false);
  assert.equal(els.title.textContent, 'Renamed');
  assert.equal(els.rest.textContent, '/b');
});

test('idle, loading, stale and closed are told in words, and a closed view drops its picture', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message({state: 'loading'}));
  assert.equal(els.root.dataset.state, 'loading');
  assert.equal(els.note.textContent, 'The first picture is on its way.');
  socket.say(message());
  socket.say(new ArrayBuffer(4));
  await settle();
  socket.say(message({state: 'stale'}));
  assert.equal(els.root.dataset.state, 'stale');
  assert.equal(els.note.dataset.over, 'true');
  assert.equal(els.note.hidden, false);
  assert.equal(els.canvas.hidden, false);
  socket.say(message({state: 'closed', reason: 'tab'}));
  assert.equal(els.root.dataset.state, 'closed');
  assert.equal(els.note.textContent, 'This tab was closed.');
  assert.equal(els.canvas.hidden, true);
  socket.say(message({state: 'idle', tab: '', url: '', title: ''}));
  assert.equal(els.root.dataset.state, 'idle');
  assert.equal(els.title.textContent, 'No page open');
  assert.equal(els.host.textContent, '');
});

test('a bad message changes nothing', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message());
  socket.say('{"type":"meta","state":"nonsense"}');
  socket.say('not json');
  assert.equal(els.root.dataset.state, 'live');
});

test('a lost link keeps the picture, says it is reconnecting, and tries again later and later', async () => {
  const {browser, els, timers} = panel();
  await browser.follow('s1');
  const first = FakeSocket.all[0];
  first.open();
  first.say(message());
  first.say(new ArrayBuffer(4));
  await settle();
  first.drop(1006);
  assert.equal(els.root.dataset.state, 'stale');
  assert.equal(els.status.textContent, 'Reconnecting…');
  assert.equal(els.canvas.hidden, false);
  assert.equal(timers.at(-1).ms, 1000);
  timers.at(-1).fn();
  const second = FakeSocket.all[1];
  assert.ok(second);
  second.drop(1006);
  assert.equal(timers.at(-1).ms, 2000);
  timers.at(-1).fn();
  FakeSocket.all[2].drop(1013);
  assert.equal(timers.at(-1).ms, 4000);
  timers.at(-1).fn();
  const fourth = FakeSocket.all[3];
  fourth.open();
  fourth.say(message());
  assert.equal(els.root.dataset.state, 'live');
  fourth.drop(1006);
  assert.equal(timers.at(-1).ms, 1000, 'a link that was up starts the delays again');
});

test('the gateway ending a closed view on purpose is not retried', async () => {
  const {browser, els, timers} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message({state: 'closed', reason: 'browser'}));
  const before = timers.length;
  socket.drop(1000);
  assert.equal(timers.length, before);
  assert.equal(els.note.textContent, "melo's browser stopped.");
});

test('a refusal closes the view and is not retried', async () => {
  const {browser, els, timers, calls} = panel();
  await browser.follow('s1');
  FakeSocket.all[0].drop(1008, 'refused');
  assert.equal(els.root.dataset.state, 'closed');
  assert.equal(els.note.textContent, 'The live view is not available for this chat.');
  assert.equal(timers.length, 0);
  assert.equal(calls.denied, 0);
});

test('a stream the gateway ended for the token asks the page to check it, then retries', async () => {
  const {browser, timers, calls} = panel();
  await browser.follow('s1');
  FakeSocket.all[0].drop(1008, 'token_rotated');
  assert.equal(calls.denied, 1);
  assert.equal(timers.length, 1);
  assert.equal(browser.state().link, 'retry');
});

test('hiding the panel closes the socket, so the gateway stops the stream, and it stays hidden for the chat', async () => {
  const {browser, els, calls} = panel({state: {available: true, open: false}});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  const socket = FakeSocket.all[0];
  browser.hide();
  assert.equal(socket.closed, 1000);
  assert.equal(els.root.hidden, true);
  assert.deepEqual(calls.changed, [true, false]);
  assert.equal(els.chip.hidden, false, 'the chip brings it back');
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c2'}});
  assert.equal(els.root.hidden, true, 'a hidden panel does not open itself again');
  assert.equal(els.chip.dataset.browsing, 'true', 'but the chip says melo is browsing');
  assert.equal(els.chipText.textContent, 'melo is browsing');
  assert.equal(FakeSocket.all.length, 1);
  browser.toggle();
  assert.equal(els.root.hidden, false);
  assert.equal(FakeSocket.all.length, 2);
  browser.toggle();
  assert.equal(els.root.hidden, true);
  assert.equal(FakeSocket.all[1].closed, 1000);
});

test('the chip says Browser when melo is not browsing, and a finished step clears it', async () => {
  const {browser, els} = panel({state: {available: true, open: false}});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  browser.hide();
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c2'}});
  assert.equal(els.chip.dataset.browsing, 'true');
  browser.note({type: 'tool.finished', data: {call_id: 'c2'}});
  assert.equal(els.chip.dataset.browsing, 'true', 'c1 still runs');
  browser.note({type: 'tool.finished', data: {call_id: 'c1'}});
  assert.equal(els.chip.dataset.browsing, 'false');
  assert.equal(els.chipText.textContent, 'Browser');
  browser.note({type: 'tool.started', data: {name: 'browser_read', call_id: 'c3'}});
  browser.note({type: 'turn.finished', data: {}});
  assert.equal(els.chip.dataset.browsing, 'false');
});

test('on a narrow screen the panel waits to be asked for', async () => {
  const {browser, els, calls} = panel({wide: false, state: {available: true, open: false}});
  await browser.follow('s1');
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  assert.equal(els.root.hidden, true);
  assert.equal(els.chip.hidden, false);
  assert.equal(els.chip.dataset.browsing, 'true');
  assert.equal(FakeSocket.all.length, 0);
  browser.toggle();
  assert.equal(els.root.hidden, false);
  assert.deepEqual(calls.changed, [true]);
  assert.equal(browser.closeSheet(), true, 'Escape closes the sheet');
  assert.equal(els.root.hidden, true);
  assert.equal(browser.closeSheet(), false);
});

test('on a wide screen Escape leaves the docked panel alone', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  assert.equal(browser.closeSheet(), false);
  assert.equal(els.root.hidden, false);
});

test('leaving the chat takes the panel and its socket away, and the same chat keeps them', async () => {
  const {browser, els, calls} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  browser.release('s1');
  assert.equal(els.root.hidden, false, 'the same chat is kept');
  await browser.follow('s1');
  assert.equal(FakeSocket.all.length, 1);
  assert.equal(calls.api.length, 1, 'the chat is not asked about again');
  browser.release('s2');
  assert.equal(socket.closed, 1000);
  assert.equal(els.root.hidden, true);
  assert.deepEqual(calls.changed, [true, false]);
  assert.equal(els.chip.hidden, true);
});

test('another chat starts from nothing and is asked about', async () => {
  const {browser, els, calls, setApi} = panel();
  await browser.follow('s1');
  FakeSocket.all[0].say(message());
  setApi({available: true, open: false});
  await browser.follow('s2');
  assert.deepEqual(calls.api, ['/sessions/s1/browser', '/sessions/s2/browser']);
  assert.equal(els.root.hidden, true);
  assert.equal(FakeSocket.all[0].closed, 1000);
  assert.equal(FakeSocket.all.length, 1);
  assert.equal(els.chip.hidden, true);
});

test('a picture that arrives after the chat was left is not drawn', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  const socket = FakeSocket.all[0];
  socket.say(message());
  socket.say(new ArrayBuffer(4));
  browser.release();
  await settle();
  assert.equal(els.canvas.drawn.length, 0);
  assert.equal(els.canvas.hidden, true);
});

test('the pop-out button is offered where the page can pop out (the app, or a browser window), for a shown panel', async () => {
  const plain = panel();
  await plain.browser.follow('s1');
  assert.equal(plain.els.pop.hidden, true);
  const app = panel({bridge: true});
  await app.browser.follow('s1');
  assert.equal(app.els.pop.hidden, false);
  app.browser.popOut();
  assert.deepEqual(app.calls.popped, ['s1']);
  app.browser.hide();
  assert.equal(app.els.pop.hidden, true);
  app.browser.popOut();
  assert.deepEqual(app.calls.popped, ['s1']);
});

test('the pop-out window shows the panel alone: no chip, no hide, whatever the chat has open', async () => {
  const {browser, els, calls} = panel({state: {available: true, open: false}, bridge: true});
  await browser.follow('s1', {liveOnly: true});
  assert.equal(els.root.hidden, false);
  assert.equal(els.chip.hidden, true);
  assert.equal(els.hide.hidden, true);
  assert.equal(els.pop.hidden, true);
  assert.equal(FakeSocket.all.length, 1);
  browser.hide();
  assert.equal(els.root.hidden, false, 'there is nothing behind it to go back to');
  assert.equal(browser.closeSheet(), false);
  browser.note({type: 'tool.started', data: {name: 'browser_open', call_id: 'c1'}});
  assert.deepEqual(calls.changed, [true]);
});

test('the pop-out window of a chat with no live view says so', async () => {
  const {browser, els} = panel({state: {available: false, open: false}});
  await browser.follow('s1', {liveOnly: true});
  assert.equal(els.root.dataset.state, 'closed');
  assert.equal(els.note.textContent, 'The live view is not available for this chat.');
});

test('the picture follows its box when the panel is resized', async () => {
  const {browser, els} = panel();
  await browser.follow('s1');
  FakeSocket.all[0].say(message());
  els.viewport.clientWidth = 640;
  els.viewport.clientHeight = 200;
  browser.relayout();
  assert.equal(els.canvas.style.height, '200px');
  assert.equal(els.canvas.style.width, String(Math.floor(1280 * (200 / 713))) + 'px');
});

test('a page that is away lets go of its socket, and asks again when it is back', async () => {
  const {browser, els, timers} = panel();
  await browser.follow('s1');
  const first = FakeSocket.all[0];
  first.say(message());
  first.say(new ArrayBuffer(4));
  await settle();
  browser.suspend();
  assert.equal(first.closed, 1000, 'the gateway sees nobody watching');
  assert.equal(els.root.hidden, false, 'the panel stays as it was');
  assert.equal(els.canvas.hidden, false);
  browser.resume();
  assert.equal(FakeSocket.all.length, 2);
  browser.resume();
  assert.equal(FakeSocket.all.length, 2, 'once');
  FakeSocket.all[1].drop(1006);
  browser.suspend();
  assert.equal(timers.at(-1).cancelled, true, 'a retry waiting is dropped too');
  browser.resume();
  assert.equal(FakeSocket.all.length, 3);
});

test('a page that is away is not reconnected by a chat followed again or a lost link', async () => {
  const {browser, timers} = panel();
  await browser.follow('s1');
  browser.suspend();
  await browser.follow('s1');
  browser.toggle();
  browser.toggle();
  assert.equal(FakeSocket.all.length, 1, 'no socket while the page is away');
  assert.equal(timers.length, 0);
});

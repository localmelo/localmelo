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
import {browserLook, mergeBrowsers, readBrowser, setupBrowserCards} from '../../support/gateway/static/browser-card.mjs';
import {createStreams} from '../../support/gateway/static/browser-stream.mjs';

// ── The pure parts ──

test('Pop out opens the source\'s live page on this page\'s own origin, a task\'s with the run it names', () => {
  const origin = {origin: 'http://127.0.0.1:9000'};
  assert.equal(liveUrl(origin, 'a b'), 'http://127.0.0.1:9000/#/live/a%20b');
  assert.deepEqual(liveRoute(new URL(liveUrl(origin, 's1')).hash), {session: 's1'});
  const task = {session: 't9', gen: 'a1b2c3d4', run: 'run-1', owner: 'c1'};
  assert.equal(liveUrl(origin, task), 'http://127.0.0.1:9000/#/live/t9?gen=a1b2c3d4&run=run-1&owner=c1');
  assert.deepEqual(liveRoute(new URL(liveUrl(origin, task)).hash), task);
});

test('a #/live address names the source; anything else names none', () => {
  assert.deepEqual(liveRoute('#/live/abc-1.2:x'), {session: 'abc-1.2:x'});
  assert.deepEqual(liveRoute('#/live/t9?owner=c1&gen=a1b2c3d4&run=r.1&other=1'), {session: 't9', gen: 'a1b2c3d4', run: 'r.1', owner: 'c1'}, 'only the three parts of a source');
  assert.equal(liveRoute('#/live/'), null);
  assert.equal(liveRoute('#/s/abc'), null);
  assert.equal(liveRoute('#/live/a/b'), null);
  assert.equal(liveRoute('#/live/a?gen=<x>'), null);
  assert.equal(liveRoute(undefined), null);
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
  for (const [reason, words] of [['tab', 'This tab was closed.'], ['browser', "melo's browser stopped."], ['chat', 'This chat ended.'], ['unavailable', 'The live view is not available now.'], ['', 'This tab was closed.']]) {
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
  sent = [];
  close(code) { this.closed = code ?? 1000; }
  send(text) { this.sent.push(JSON.parse(text)); }
  open() { this.onopen?.(); }
  say(data) { this.onmessage?.({data}); }
  drop(code = 1006, reason = '') { this.onclose?.({code, reason}); }
}

/**
 * A panel on a fake socket. The panel never opens by itself: with `open` (the default) the chat's header chip is pressed
 * once the chat is followed, as the person does; `open: false` leaves it for the test to press.
 */
function panel({wide = true, state = {available: true, open: true, browsing: true}, popOut = null, bridge = false, open = true} = {}) {
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
  const follow = browser.follow;
  if (open) {
    browser.follow = async (id, options) => {
      await follow(id, options);
      if (!options?.liveOnly) browser.toggle();
    };
  }
  return {browser, els, timers, calls, setApi: (next) => { apiState = next; }};
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

// The state of a chat's browser card, as the page tells the panel.
const card = (extra = {}) => ({session: 's1', gen: 'a1b2c3d4', run: 'run-1', state: 'working', ...extra});
const SOURCE = 'ws://gw.test/api/sessions/s1/browser/live?gen=a1b2c3d4&run=run-1';

test('the panel never opens by itself: the chip opens it, on the source of the chat\'s card', async () => {
  const {browser, els, calls} = panel({open: false});
  await browser.follow('s1');
  assert.deepEqual(calls.api, ['/sessions/s1/browser']);
  assert.equal(els.root.hidden, true);
  assert.equal(els.chip.hidden, false);
  assert.equal(FakeSocket.all.length, 0);
  browser.card(card());
  assert.equal(els.root.hidden, true, 'melo is browsing, and the panel still waits to be asked for');
  assert.equal(FakeSocket.all.length, 0);
  browser.toggle();
  assert.equal(els.root.hidden, false);
  assert.deepEqual(calls.changed, [true]);
  assert.equal(FakeSocket.all.length, 1);
  assert.equal(FakeSocket.all[0].url, SOURCE);
  assert.equal(els.root.dataset.state, 'loading');
});

test('a chat with no page open has no chip until its card says melo browses, in the one word of the state', async () => {
  const {browser, els} = panel({open: false, state: {available: true, open: false}});
  await browser.follow('s1');
  assert.equal(els.chip.hidden, true);
  browser.card(card());
  assert.equal(els.chip.hidden, false);
  assert.equal(els.chip.dataset.browsing, 'true');
  assert.equal(els.chipText.textContent, 'Working');
  const words = ['waiting', 'you', 'done', 'stopped'].map((state) => (browser.card(card({state})), els.chipText.textContent));
  assert.deepEqual(words, ['Waiting for you', "You're in control", 'Browser', 'Browser']);
  assert.equal(els.chip.dataset.browsing, 'false', 'the run is over');
  browser.card(card());
  browser.open(card());
  assert.equal(els.root.hidden, false, 'Open browser shows the card\'s page');
  assert.equal(FakeSocket.all.at(-1).url, SOURCE);
});

test('a chat whose browser cannot be watched, or cannot be asked, shows no chip and Open browser does nothing', async () => {
  for (const state of [{available: false, open: true}, new Error('down')]) {
    const {browser, els} = panel({open: false, state});
    await browser.follow('s1');
    browser.card(card());
    browser.open(card());
    assert.equal(els.root.hidden, true);
    assert.equal(els.chip.hidden, true);
  }
  assert.equal(FakeSocket.all.length, 0);
});

test('a settings change is checked at once: the chip follows the browser now chosen', async () => {
  const {browser, els, calls, setApi} = panel({open: false, state: {available: false, open: false}});
  await browser.follow('s1');
  browser.card(card());
  setApi({available: true, open: false});
  await browser.recheck();
  // Asked when followed, again for the run's new card (the chat was found unwatchable), and by the recheck.
  assert.deepEqual(calls.api, ['/sessions/s1/browser', '/sessions/s1/browser', '/sessions/s1/browser']);
  assert.equal(els.root.hidden, true, 'nothing opens by itself');
  assert.equal(els.chip.hidden, false);
  browser.toggle();
  const socket = FakeSocket.all[0];
  setApi({available: false, open: false});
  await browser.recheck();
  assert.equal(els.root.hidden, true, 'Safari now: no live view');
  assert.equal(els.chip.hidden, true);
  assert.equal(socket.closed, 1000);
});

test('hiding the panel only stops watching: nothing is sent, the chip stays and nothing opens it again', async () => {
  const {browser, els, calls} = panel({open: false});
  await browser.follow('s1');
  browser.card(card());
  browser.toggle();
  const socket = FakeSocket.all[0];
  socket.open();
  browser.hide();
  assert.equal(socket.closed, 1000);
  assert.deepEqual(socket.sent, [], 'closing the panel is not Done or Stop');
  assert.equal(els.root.hidden, true);
  assert.deepEqual(calls.changed, [true, false]);
  assert.equal(els.chip.hidden, false, 'the chip brings it back');
  browser.card(card());
  assert.equal(els.root.hidden, true);
  assert.equal(els.chip.dataset.browsing, 'true');
  assert.equal(FakeSocket.all.length, 1);
  browser.toggle();
  assert.equal(els.root.hidden, false);
  assert.equal(FakeSocket.all.length, 2);
  browser.toggle();
  assert.equal(els.root.hidden, true);
  assert.equal(FakeSocket.all[1].closed, 1000);
});

test('a card that begins while the panel shows the chat\'s browser moves it to the card\'s source, and its end leaves it', async () => {
  const {browser} = panel();
  await browser.follow('s1');
  const first = FakeSocket.all[0];
  assert.equal(first.url, 'ws://gw.test/api/sessions/s1/browser/live');
  browser.card(card());
  assert.equal(first.closed, 1000);
  assert.equal(FakeSocket.all.length, 2);
  assert.equal(FakeSocket.all[1].url, SOURCE);
  browser.card(card({state: 'done'}));
  assert.equal(FakeSocket.all.length, 2, 'the run ended: the panel keeps what it shows');
  assert.equal(FakeSocket.all[1].closed, null);
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
  assert.equal(els.chipText.textContent, 'Browser', 'the chip names the state of the chat\'s card, and there is none');
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
  browser.toggle();
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
  assert.equal(els.note.textContent, 'The live view is not available now.');
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

test('on a narrow screen the panel is a sheet that Escape closes', async () => {
  const {browser, els, calls} = panel({wide: false, open: false, state: {available: true, open: true}});
  await browser.follow('s1');
  assert.equal(els.root.hidden, true);
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
  const {browser, els, calls, setApi} = panel({open: false});
  await browser.follow('s1');
  browser.toggle();
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
  assert.deepEqual(app.calls.popped, [{session: 's1'}]);
  app.browser.hide();
  assert.equal(app.els.pop.hidden, true);
  app.browser.popOut();
  assert.deepEqual(app.calls.popped, [{session: 's1'}]);
});

test('a task\'s browser pops out by its exact run, and its window shows that run alone', async () => {
  const app = panel({bridge: true, open: false});
  await app.browser.follow('c1');
  const task = card({session: 't9', task: true, owner: 'c1'});
  app.browser.card(task);
  await app.browser.open(task);
  assert.equal(app.els.pop.hidden, false, 'a task\'s browser pops out like a chat\'s');
  app.browser.popOut();
  assert.deepEqual(app.calls.popped, [{session: 't9', gen: 'a1b2c3d4', run: 'run-1', owner: 'c1'}]);
  // The window follows the task with the same selector: the probe and the stream both name it, so nothing falls back to a chat.
  const source = {gen: 'a1b2c3d4', run: 'run-1', owner: 'c1'};
  const window = panel({bridge: true});
  await window.browser.follow('t9', {liveOnly: true, pin: source});
  assert.deepEqual(window.calls.api, ['/sessions/t9/browser?gen=a1b2c3d4&run=run-1&owner=c1']);
  assert.equal(FakeSocket.all.at(-1).url, 'ws://gw.test/api/sessions/t9/browser/live?gen=a1b2c3d4&run=run-1&owner=c1');
  assert.equal(window.els.pop.hidden, true, 'the window is the pop-out already');
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
  browser.card(card());
  browser.open(card());
  assert.deepEqual(calls.changed, [true], 'its chat\'s cards and Open browser are not its business');
  assert.equal(FakeSocket.all.length, 1);
});

test('the pop-out window of a chat with no live view says so', async () => {
  const {browser, els} = panel({state: {available: false, open: false}});
  await browser.follow('s1', {liveOnly: true});
  assert.equal(els.root.dataset.state, 'closed');
  assert.equal(els.note.textContent, 'The live view is not available now.');
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

// ── The card, the dock and the panel on one transport ──

const row = (extra = {}) => ({
  id: 7, card_key: 's1:run-1', owner_chat_id: 's1', source_session_id: 's1', source_turn: 1, run_id: 'run-1',
  browser_session_id: 'a1b2c3d4', state: 'working', action: 'press', site: 'example.com', ...extra,
});

test('a card row is read from a stored row or an event frame, and the newest row of each card wins', () => {
  const stored = {...row(), ts: 1};
  const frame = {id: 9, type: 'attention.browser', turn: null, data: row({state: 'waiting', action: ''})};
  assert.equal(readBrowser(stored).state, 'working');
  assert.equal(readBrowser(frame).state, 'waiting');
  assert.equal(readBrowser({id: 3, card_key: 'k', state: 'sleeping'}), null, 'a state the page does not know is not a card');
  assert.equal(readBrowser({id: 3, state: 'working'}), null, 'a card with no key is not a card');
  const merged = mergeBrowsers(mergeBrowsers([], [frame]), [stored, row({id: 12, state: 'done', card_key: 's1:run-2'})]);
  assert.deepEqual(merged.map((card) => [card.key, card.id, card.state]), [['s1:run-1', 9, 'waiting'], ['s1:run-2', 12, 'done']], 'a replayed older row does not undo the newer one');
});

test('a card is worded by its state, and a picture that is not arriving never reads as live', () => {
  const working = readBrowser(row());
  const live = {meta: parseMeta(message()), link: 'open', picture: true};
  assert.deepEqual(
    (({own, word, sub, now, fresh, size, openable}) => ({own, word, sub, now, fresh, size, openable}))(browserLook(working, live, {retained: true})),
    {own: 'working', word: 'Working', sub: 'example.com', now: 'Pressing on the page', fresh: true, size: 'full', openable: true},
  );
  const lost = browserLook(working, {...live, link: 'retry'}, {retained: true});
  assert.deepEqual([lost.own, lost.word, lost.sub, lost.fresh, lost.now], ['stale', 'Not updating', 'Reconnecting…', false, '']);
  const still = browserLook(working, {...live, meta: parseMeta(message({state: 'stale'}))}, {retained: true});
  assert.deepEqual([still.own, still.word, still.sub], ['stale', 'Not updating', 'example.com']);
  const back = browserLook(working, {meta: null, link: 'connecting', picture: false}, {retained: true});
  assert.deepEqual([back.own, back.word, back.fresh], ['working', 'Working', false], 'a picture kept from before dims, and claims nothing');
  assert.equal(browserLook(working, null).note, 'The first picture is on its way.');
  assert.equal(browserLook(working, {meta: parseMeta(message({state: 'closed', reason: 'unavailable'})), link: 'off', picture: false}).openable, false);
  const taken = browserLook(readBrowser(row({state: 'you', action: ''})), live, {retained: true});
  assert.deepEqual([taken.own, taken.word, taken.sub, taken.now], ['you', "You're in control", 'melo is paused', '']);
  const ended = [['done', ''], ['stopped', 'you']].map(([state, by]) => browserLook(readBrowser(row({state, by, action: ''})), null, {retained: true}));
  assert.deepEqual(ended.map((look) => [look.word, look.sub, look.size, look.openable]), [['Done', 'example.com', 'kept', false], ['Stopped', 'by you', 'kept', false]]);
  assert.equal(browserLook(readBrowser(row({state: 'done'})), null).size, 'line', 'after a reload there is no picture to keep');
});

/** A document double: elements that record what a card sets on them. */
function page() {
  const make = (tag) => {
    const attributes = {};
    return {
      tag, hidden: false, textContent: '', className: '', dataset: {}, style: {}, children: [], listeners: {}, tabIndex: 0, drawn: [],
      setAttribute(key, value) { attributes[key] = String(value); },
      getAttribute(key) { return attributes[key] ?? null; },
      append(...items) { this.children.push(...items); },
      addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
      press() { for (const fn of this.listeners.click ?? []) fn({}); },
      getContext() { return {drawImage: (bitmap) => this.drawn.push(bitmap.tag ?? 'copy')}; },
    };
  };
  const all = (root, tag) => [root, ...root.children.flatMap((child) => all(child, tag))].filter((found) => found.tag === tag);
  return {doc: {createElement: make}, make, all};
}

/** The card manager, the panel and one transport on fake sockets, with decodes the test releases by hand. */
function together({state = {available: true, open: true}, acts = {}} = {}) {
  FakeSocket.all = [];
  const decodes = [];
  const streams = createStreams({
    openSocket: (url) => new FakeSocket(url),
    decode: (bytes) => new Promise((resolve) => decodes.push(() => resolve({width: 64, height: 36, tag: new Uint8Array(bytes)[0], close() { this.released = true; }}))),
    location: {protocol: 'http:', host: 'gw.test'},
    schedule: () => 0,
    cancel: () => {},
  });
  const {doc, make, all} = page();
  const els = {
    root: node({hidden: true}), chip: node({hidden: true}), chipText: node(), title: node(), host: node(), rest: node(), status: node(),
    viewport: node({clientWidth: 400, clientHeight: 300}), canvas: make('canvas'), note: node(), hide: node(), pop: node({hidden: true}),
  };
  const browser = setupBrowser({api: async () => state, els, streams, location: {protocol: 'http:', host: 'gw.test'}});
  const sight = new Map();
  let report = null;
  const dock = {root: make('div'), canvas: make('canvas'), mark: make('span'), text: make('span'), jump: make('button'), open: make('button')};
  dock.root.hidden = true;
  const cards = setupBrowserCards({
    doc, streams, open: (card) => browser.open(card), dock, ...acts,
    observe: (onChange) => { report = onChange; return {add: (target) => sight.set(target, null), remove: (target) => sight.delete(target)}; },
  });
  return {
    streams, browser, cards, els, dock, decodes, all,
    see: (key, visible) => report(cards.nodeOf(key), visible),
    open: (key) => all(cards.nodeOf(key), 'button').find((found) => found.className.includes('outline-button')),
  };
}

test('the card and the panel share one socket and one decode, and a late picture of an old source is never painted', async () => {
  const t = together();
  const first = readBrowser(row());
  t.cards.sync([first]);
  t.see('s1:run-1', true);
  await t.browser.follow('s1');
  t.browser.card(first);
  assert.equal(FakeSocket.all.length, 1, 'the card watches its source');
  assert.equal(FakeSocket.all[0].url, SOURCE);
  t.open('s1:run-1').press();
  assert.equal(t.els.root.hidden, false, 'Open browser opens the panel');
  assert.equal(FakeSocket.all.length, 1, 'the panel shares the card\'s socket');
  assert.deepEqual(t.streams.size(), {sources: 1, sinks: 2});
  const [socket] = FakeSocket.all;
  socket.open();
  socket.say(message());
  socket.say(new Uint8Array([1]).buffer);
  assert.equal(t.decodes.length, 1);
  t.decodes.shift()();
  await settle();
  const canvas = t.cards.nodeOf('s1:run-1').children[1].children[0];
  assert.deepEqual([canvas.drawn, t.els.canvas.drawn], [[1], [1]], 'one decode, painted on both');
  assert.equal(t.cards.nodeOf('s1:run-1').dataset.fresh, 'true');
  // A picture of run 1 is still being decoded when run 2 begins and both views move to it.
  socket.say(new Uint8Array([2]).buffer);
  const late = t.decodes.shift();
  const second = readBrowser(row({id: 11, card_key: 's1:run-2', run_id: 'run-2', browser_session_id: 'e5f6a7b8'}));
  t.cards.sync([readBrowser(row({id: 10, state: 'done', action: ''})), second]);
  t.see('s1:run-2', true);
  t.browser.card(second);
  assert.equal(socket.closed, 1000, 'nothing watches the old source any more');
  late();
  await settle();
  assert.deepEqual([canvas.drawn, t.els.canvas.drawn], [[1], [1]], 'the late picture is painted nowhere');
  assert.equal(FakeSocket.all.at(-1).url, 'ws://gw.test/api/sessions/s1/browser/live?gen=e5f6a7b8&run=run-2');
});

test('a task\'s card says who it works for and its goal, and every action names that task\'s run and chat', async () => {
  const stopped = [];
  const given = [];
  const t = together({acts: {stop: async (found) => { stopped.push(found.session); }, done: async (found) => { given.push(found); return false; }}});
  const task = {source_kind: 'task', source_session_id: 'task1', owner_chat_id: 's1', card_key: 'task1:run-1', goal: 'Find ML jobs in Boston'};
  const working = readBrowser(row(task));
  t.cards.sync([working]);
  t.see('task1:run-1', true);
  await t.browser.follow('s1');
  t.browser.card(working);
  const owned = 'ws://gw.test/api/sessions/task1/browser/live?gen=a1b2c3d4&run=run-1&owner=s1';
  assert.deepEqual(FakeSocket.all.map((socket) => socket.url), [owned]);
  t.open('task1:run-1').press();
  await settle();
  assert.equal(FakeSocket.all.length, 1, 'the panel shows the same source on the same socket');
  assert.equal(t.els.pop.hidden, true, 'a task\'s browser stays beside the chat it works for');
  const shown = t.cards.nodeOf('task1:run-1');
  const words = (found) => [found.textContent ?? '', ...found.children.flatMap(words)];
  const text = words(shown).join(' ');
  assert.match(text, /Working for this chat/);
  assert.match(text, /Find ML jobs in Boston/);
  assert.doesNotMatch(text, /task1|P2/, 'never a worker id or a priority code');
  const buttons = () => t.all(shown, 'button').filter((found) => found.className.includes('-button') && !found.hidden);
  assert.deepEqual(buttons().map((found) => found.textContent), ['Open browser', 'Stop']);
  buttons().at(-1).press();
  await settle();
  assert.deepEqual(stopped, ['task1'], 'Stop names the task, not its chat');
  // The person has the page: Done is on the card, for this run; a press that did nothing can be pressed again.
  const yours = readBrowser(row({...task, id: 11, state: 'you', action: ''}));
  t.cards.sync([yours]);
  const done = buttons().find((found) => found.textContent === 'Done');
  done.press();
  await settle();
  assert.deepEqual(given.map((found) => [found.session, found.run, found.owner]), [['task1', 'run-1', 's1']]);
  assert.equal(done.disabled, false);
  // Over: no action is left, and the goal names what it was.
  t.cards.sync([readBrowser(row({...task, id: 12, state: 'done', action: ''}))]);
  assert.deepEqual(buttons().map((found) => found.textContent), []);
  assert.match(words(shown).join(' '), /Find ML jobs in Boston/);
  assert.doesNotMatch(words(shown).join(' '), /Working for this chat/);
});

test('a card watches only while it is on screen, the dock takes over when it leaves, and a finished run lets go', async () => {
  const t = together();
  const working = readBrowser(row());
  t.cards.sync([working]);
  assert.equal(FakeSocket.all.length, 0, 'nothing is watched before the card is seen');
  t.see('s1:run-1', true);
  assert.equal(FakeSocket.all.length, 1);
  assert.equal(t.dock.root.hidden, true);
  FakeSocket.all[0].open();
  FakeSocket.all[0].say(new Uint8Array([5]).buffer);
  t.decodes.shift()();
  await settle();
  t.see('s1:run-1', false);
  assert.equal(t.dock.root.hidden, false, 'the card left the screen: the dock shows it');
  assert.equal(FakeSocket.all.length, 1, 'the dock shares the source\'s socket');
  assert.equal(FakeSocket.all[0].closed, null);
  assert.equal(t.dock.canvas.drawn.length, 1, 'and starts with the picture the card had');
  assert.equal(t.dock.text.textContent, 'Browser · Working · Pressing on the page');
  t.see('s1:run-1', true);
  assert.equal(t.dock.root.hidden, true);
  assert.equal(FakeSocket.all[0].closed, null);
  const same = t.cards.nodeOf('s1:run-1');
  t.cards.sync([readBrowser(row({id: 12, state: 'done', action: ''}))]);
  assert.equal(t.cards.nodeOf('s1:run-1'), same, 'the card is updated in place');
  assert.equal(same.dataset.own, 'done');
  assert.equal(same.dataset.size, 'kept', 'it keeps the last picture it painted');
  assert.equal(FakeSocket.all[0].closed, 1000, 'the run ended: nobody watches');
  t.cards.sync([]);
  assert.equal(t.cards.nodeOf('s1:run-1'), null, 'a card of another chat is let go');
});

test('the gateway ends a run\'s stream just before its card says Done: the card shows the stop, then keeps the last picture', async () => {
  const t = together();
  t.cards.sync([readBrowser(row())]);
  t.see('s1:run-1', true);
  const socket = FakeSocket.all[0];
  socket.open();
  socket.say(message());
  socket.say(new Uint8Array([5]).buffer);
  t.decodes.shift()();
  await settle();
  const node = t.cards.nodeOf('s1:run-1');
  assert.equal(node.children[1].hidden, false, 'the page shows');
  socket.say(message({state: 'closed', tab: '', reason: 'browser'}));
  socket.drop(1000);
  assert.equal(FakeSocket.all.length, 1, 'a view the gateway ended on purpose is not tried again');
  assert.equal(node.children[1].hidden, true, 'for a moment: the browser stopped, and the card says so');
  assert.equal(node.children[2].textContent, "melo's browser stopped.");
  t.cards.sync([readBrowser(row({id: 12, state: 'done', action: ''}))]);
  assert.equal(node.dataset.size, 'kept');
  assert.equal(node.children[1].hidden, false, 'the picture the person already saw stays');
  assert.equal(node.children[2].hidden, true);
});

test('a picture still being decoded when the page changes under it is not painted, nor taken for one of the new page', async () => {
  const changes = {
    'a tab switch': (t) => t.socket.say(message({tab: 't2', url: 'https://other.test/'})),
    'an idle view': (t) => t.socket.say(message({state: 'idle', tab: ''})),
    'a closed view': (t) => t.socket.say(message({state: 'closed', tab: '', reason: 'tab'})),
    'a lost link': (t) => { t.socket.drop(); t.timers.splice(0).forEach((fn) => fn()); },
    'a page that went away and came back': (t) => { t.streams.suspend(); t.streams.resume(); },
  };
  for (const [what, change] of Object.entries(changes)) {
    FakeSocket.all = [];
    const decodes = [];
    const timers = [];
    const streams = createStreams({
      openSocket: (url) => new FakeSocket(url),
      decode: (bytes) => new Promise((resolve) => decodes.push(() => resolve({width: 64, height: 36, tag: new Uint8Array(bytes)[0], close() {}}))),
      location: {protocol: 'http:', host: 'gw.test'},
      schedule: (fn) => timers.push(fn),
      cancel: () => {},
    });
    const canvas = {drawn: [], width: 0, height: 0, getContext() { return {drawImage: (bitmap) => canvas.drawn.push(bitmap.tag)}; }};
    const seen = [];
    streams.watch({session: 's1'}, {canvas, update: (state) => seen.push(state)});
    const t = {streams, timers, socket: FakeSocket.all[0]};
    t.socket.open();
    t.socket.say(message());
    t.socket.say(new Uint8Array([1]).buffer);
    change(t);
    decodes.shift()();
    await settle();
    assert.deepEqual(canvas.drawn, [], `${what}: the old page is not painted`);
    assert.equal(seen.at(-1).picture, false, `${what}: and no picture is promised`);
    // The first picture of what shows now is painted as usual, on the socket that carries it.
    const now = FakeSocket.all.at(-1);
    now.open();
    now.say(message({tab: 't2'}));
    now.say(new Uint8Array([2]).buffer);
    decodes.shift()();
    await settle();
    assert.deepEqual(canvas.drawn, [2], `${what}: the new page is`);
    assert.equal(seen.at(-1).picture, true);
  }
});

test('Open browser works for a chat that was unavailable when it was followed and is watchable now, without a reload', async () => {
  const {browser, els, setApi} = panel({open: false, state: {available: false, open: false}});
  await browser.follow('s1');
  assert.equal(els.chip.hidden, true);
  setApi({available: true, open: true}); // another page saved Settings: this chat's next browser is Chrome
  browser.card(card());
  await settle();
  assert.equal(els.chip.hidden, false, 'the chip follows the browser the run has');
  browser.open(card());
  assert.equal(els.root.hidden, false);
  assert.equal(FakeSocket.all.at(-1).url, SOURCE);
});

test('after a reload an ended card is its words alone: the picture of the page was never kept anywhere', () => {
  const t = together();
  t.cards.sync([readBrowser(row({state: 'done', action: ''}))]);
  const done = t.cards.nodeOf('s1:run-1');
  assert.equal(done.dataset.size, 'line');
  assert.equal(done.dataset.own, 'done');
  assert.equal(done.children[1].hidden, true, 'no picture');
  assert.equal(FakeSocket.all.length, 0, 'and nothing is asked for one');
  t.cards.sync([readBrowser(row({id: 12, card_key: 's1:run-2', run_id: 'run-2', state: 'stopped', action: '', by: 'you'}))]);
  assert.equal(t.cards.nodeOf('s1:run-2').dataset.size, 'line');
  assert.equal(t.cards.nodeOf('s1:run-2').children[1].hidden, true);
});

test('Open browser asks again when the chat was last found unwatchable: it opens if Chrome is watchable now, and does nothing if not', async () => {
  const {browser, els, calls, setApi} = panel({open: false, state: {available: false, open: false}});
  await browser.follow('s1');
  await browser.open(card());
  assert.equal(els.root.hidden, true, 'still Safari: nothing to open');
  assert.equal(FakeSocket.all.length, 0);
  setApi({available: true, open: true});
  await browser.open(card());
  assert.equal(els.root.hidden, false, 'Chrome now: the panel opens without a reload');
  assert.equal(FakeSocket.all.at(-1).url, SOURCE);
  assert.equal(calls.api.length, 3, 'asked when followed and by each click on an unwatchable chat, never again once it is watchable');
  browser.hide();
  await browser.open(card());
  assert.equal(calls.api.length, 3);
});

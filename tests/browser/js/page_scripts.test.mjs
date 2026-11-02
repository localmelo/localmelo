// The page scripts melo sends to a browser, run in a JS engine (jsdom).
//
// The Python wrapper (tests/browser/test_page_scripts.py) exports the scripts
// to the JSON file named by MELO_SCRIPTS and runs this file with `node --test`.
// What a "send" is here: jsdom reports a form that is really submitted as
// "Not implemented: HTMLFormElement.prototype.requestSubmit" (or .submit), so a
// send that reaches the browser is a line in `sent`.

import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';

import {JSDOM, VirtualConsole} from 'jsdom';

const here = dirname(fileURLToPath(import.meta.url));
const scripts = JSON.parse(readFileSync(process.env.MELO_SCRIPTS, 'utf8'));
const cases = JSON.parse(readFileSync(join(here, 'secret_cases.json'), 'utf8'));

const PRELUDE = `
Object.defineProperty(HTMLElement.prototype, 'innerText', {get() { return this.textContent; }, configurable: true});
Element.prototype.getBoundingClientRect = function () { return {width: 10, height: 10, top: 0, left: 0, right: 10, bottom: 10}; };
Element.prototype.scrollIntoView = function () {};
`;

function load(html, {url = 'https://example.com/', before = ''} = {}) {
  const sent = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', (error) => sent.push(String(error.message)));
  const dom = new JSDOM(html, {url, runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole});
  dom.window.eval(PRELUDE);
  if (before) dom.window.eval(before);
  dom.sent = sent;
  return dom;
}

const readPage = (dom) => JSON.parse(dom.window.eval(scripts.page));
const run = (dom, spec) => JSON.parse(dom.window.eval(scripts.actPrefix + JSON.stringify(spec) + ')'));
const sends = (dom) => dom.sent.filter((line) => /HTMLFormElement/.test(line));
const navigations = (dom) => dom.sent.filter((line) => /navigation/i.test(line));

// The spec melo's driver builds from an element it read.
function specFor(page, el, extra = {}) {
  return {
    act: 'click', url: page.url, doc: page.doc, at: el.at, fp: el.fp, kind: el.kind, label: el.label,
    href: el.href || '', submit: el.submit || '', text: '', enter: false, send: false,
    ...(el.attrs ? {attrs: el.attrs} : {}), ...extra,
  };
}

const byLabel = (page, label) => page.elements.find((el) => el.label === label);

const FORM = `
<main><form id="f" action="/save" method="post">
  <label for="q">Search</label><input id="q" name="q" value="cats">
  <input type="hidden" name="csrf" value="tok123">
  <input type="password" name="pw" value="hunter2" aria-label="Password">
  <select name="c" aria-label="Country"><option>France</option><option>Spain</option></select>
  <button id="go">Go</button>
</form><a id="n" href="https://example.com/news" ping="https://tracker.example/p">News</a></main>`;

// ── Reading ──

test('a page is read: elements with fingerprints, the form address and fields, the document and frames', () => {
  const dom = load(FORM + '<iframe src="https://other.org/x"></iframe><iframe></iframe>');
  const page = readPage(dom);
  assert.equal(page.url, 'https://example.com/');
  assert.ok(page.doc && page.doc.length > 0);
  assert.deepEqual(page.frames, ['https://other.org/x']);
  const box = byLabel(page, 'Search');
  assert.equal(box.kind, 'textbox');
  assert.equal(box.value, 'cats');
  assert.equal(box.submit, 'https://example.com/save');
  assert.ok(box.fp.startsWith('input|'));
  assert.deepEqual(box.form, [['q', 'cats', ''], ['csrf', '', 'h'], ['pw', '', 's'], ['c', 'France', '']]);
  const go = byLabel(page, 'Go');
  assert.equal(go.submit, 'https://example.com/save');
  assert.equal(byLabel(page, 'News').href, 'https://example.com/news');
});

test('a secret field is marked and nothing of it is read', () => {
  const dom = load(FORM);
  const page = readPage(dom);
  const secret = page.elements.find((el) => el.sensitive === 'password');
  assert.ok(secret);
  assert.equal(secret.value, '');
  assert.equal(secret.label, '');
  assert.ok(!JSON.stringify(page).includes('hunter2'));
  assert.ok(!JSON.stringify(page).includes('tok123'));
});

test('a field is named by what aria-labelledby, its legend and the text before it say', () => {
  const dom = load(`<span id="l1">Card number</span>
    <input id="a" aria-labelledby="l1" aria-label="Pay">
    <fieldset><legend>Verification code</legend><input id="b" aria-label="Enter it"></fieldset>
    Expiry<input id="c" aria-label="Type here">`);
  const page = readPage(dom);
  const kinds = Object.fromEntries(page.elements.map((el) => [el.attrs.words.split(' ')[0] || el.label, el.sensitive || '']));
  const flagged = page.elements.map((el) => el.sensitive || '');
  assert.deepEqual(flagged, ['card', 'code', 'card'], JSON.stringify(kinds));
});

test('the page script decides as the Python rules do, on every shared case', () => {
  const dom = load('<p></p>');
  const decide = dom.window.eval(`(function () { ${scripts.common}; return secretKind; })()`);
  for (const {why, attrs, kind} of cases) {
    const input = {type: '', autocomplete: '', words: '', inputmode: '', maxlength: 0, masked: false, ...attrs};
    assert.equal(decide(input), kind, why);
  }
});

test('a page that replaced the built-ins it uses is still read with the real ones', () => {
  const dom = load(FORM, {before: `
    Element.prototype.getAttribute = function () { return 'x'; };
    JSON.stringify = function () { return '{"hacked":1}'; };
    Document.prototype.querySelectorAll = function () { return []; };
  `});
  const page = readPage(dom);
  assert.equal(page.url, 'https://example.com/');
  assert.ok(byLabel(page, 'Search'), 'the elements were found');
});

test('a box has the shape of a code by its pattern, its max or its length, with or without a name', () => {
  const dom = load(`<input id="a" type="text" pattern="[0-9]{6}" aria-label="Entry">
    <input id="b" type="number" max="999999" aria-label="Field">
    <input id="c" type="text" pattern="\\d{4,6}" aria-label="Reference">
    <input id="d" type="number" max="100" aria-label="Quantity">
    <input id="e" type="text" maxlength="6" aria-label="Nickname">
    <input id="f" type="text" maxlength="6">
    <input id="g" type="text" pattern="[0-9]*" aria-label="Amount">`);
  const page = readPage(dom);
  assert.deepEqual(page.elements.map((el) => el.sensitive || ''), ['code', 'code', 'code', '', '', 'code', '']);
  const [a, , c, , , , g] = page.elements;
  assert.deepEqual([a.attrs.inputmode, a.attrs.maxlength], ['numeric', 6]);
  assert.deepEqual([c.attrs.inputmode, c.attrs.maxlength], ['numeric', 6]);
  assert.deepEqual([g.attrs.inputmode, g.attrs.maxlength], ['numeric', 0], 'digits only, but no length: not a code');
});

test('a ref is bound to which of its label it is and to the words around it', () => {
  const dom = load('<h2>People</h2><ul><li>Alice <button>Delete</button></li><li>Bob <button>Delete</button></li></ul>');
  const page = readPage(dom);
  const [first, second] = page.elements;
  assert.ok(first.fp.includes('|#1|r:Alice'), first.fp);
  assert.ok(second.fp.includes('|#2|r:Bob'), second.fp);
  const loose = load('<h2>Inbox</h2><button>Archive</button><h2>Sent</h2><button>Archive</button>');
  const [inbox, sent] = readPage(loose).elements;
  assert.ok(inbox.fp.endsWith('|h:Inbox') && sent.fp.endsWith('|h:Sent'), `${inbox.fp} ${sent.fp}`);
});

test('a ref does not act after its row was replaced in place', () => {
  const dom = load('<h2>People</h2><ul><li>Alice <button>Delete</button></li><li>Bob <button>Delete</button></li></ul>');
  const page = readPage(dom);
  const bobs = page.elements[1];
  // The page redraws the list in place: the same tags, labels and places, other people.
  dom.window.document.querySelector('ul').innerHTML = '<li>Carol <button>Delete</button></li><li>Dave <button>Delete</button></li>';
  const clicks = [];
  dom.window.document.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => clicks.push(b.parentNode.textContent)));
  assert.deepEqual(run(dom, specFor(page, bobs)), {ok: false, stale: 'gone'});
  assert.deepEqual(clicks, [], 'nothing was clicked');
  // The same list, untouched, is acted on.
  const same = load('<h2>People</h2><ul><li>Alice <button>Delete</button></li><li>Bob <button>Delete</button></li></ul>');
  const again = readPage(same);
  assert.deepEqual(run(same, specFor(again, again.elements[1])), {ok: true});
});

test('a ref under a heading does not act once the heading above it changed', () => {
  const dom = load('<h2>Inbox</h2><button>Archive</button>');
  const page = readPage(dom);
  dom.window.document.querySelector('h2').textContent = 'Spam';
  assert.deepEqual(run(dom, specFor(page, page.elements[0])), {ok: false, stale: 'gone'});
});

test('a page that breaks the frame the checks come from is read as text and acted on never', () => {
  const breakers = [
    `document.createElement = function () { throw new Error('no'); };`,
    `var make = document.createElement; document.createElement = function (t) { return make.call(document, 'div'); };`,
    `Node.prototype.appendChild = function () { throw new Error('no'); };`,
  ];
  const good = load(FORM);
  const spec = specFor(readPage(good), byLabel(readPage(good), 'Go'), {send: true});
  for (const before of breakers) {
    const dom = load(FORM, {before});
    const page = readPage(dom);
    assert.deepEqual(page.elements, [], before);
    assert.match(page.text, /could not set up its safe page checks/);
    assert.equal(page.url, 'https://example.com/');
    // The act script comes first, with a spec from a normal read of the same page.
    assert.deepEqual(run(dom, spec), {ok: false, stale: 'frame'}, before);
    assert.equal(sends(dom).length, 0);
  }
});

// ── Acting ──

test('a link is clicked, its ping is removed, and an act on another document is refused', () => {
  const dom = load(FORM);
  const page = readPage(dom);
  const news = byLabel(page, 'News');
  assert.deepEqual(run(dom, specFor(page, news, {doc: 'another'})), {ok: false, stale: 'doc'});
  assert.equal(dom.window.document.getElementById('n').getAttribute('ping') !== null, true);
  assert.deepEqual(run(dom, specFor(page, news)), {ok: true});
  assert.equal(dom.window.document.getElementById('n').getAttribute('ping'), null);
});

test('a reloaded page is another document', async () => {
  const first = load(FORM);
  const page = readPage(first);
  await new Promise((resolve) => setTimeout(resolve, 5));
  const second = load(FORM);
  assert.notEqual(readPage(second).doc, page.doc);
  assert.deepEqual(run(second, specFor(page, byLabel(page, 'News'))), {ok: false, stale: 'doc'});
});

test('an element that is not the one that was read is not used', () => {
  const dom = load(FORM);
  const page = readPage(dom);
  const news = byLabel(page, 'News');
  // A new button before it: the same place now holds another element.
  const extra = dom.window.document.createElement('button');
  extra.textContent = 'Buy';
  dom.window.document.querySelector('main').prepend(extra);
  assert.deepEqual(run(dom, specFor(page, news)), {ok: false, stale: 'gone'});
  // The same element moved in the tree: its fingerprint differs.
  const other = load(FORM);
  const read = readPage(other);
  const link = byLabel(read, 'News');
  const wrapper = other.window.document.createElement('div');
  const anchor = other.window.document.getElementById('n');
  anchor.replaceWith(wrapper);
  wrapper.append(anchor);
  assert.deepEqual(run(other, specFor(read, link)), {ok: false, stale: 'gone'});
});

test('where a form goes is read again at the send, and a change since the read is refused', () => {
  for (const change of [
    (d) => d.getElementById('f').setAttribute('action', 'https://evil.example/post'),
    (d) => d.getElementById('go').setAttribute('formaction', 'https://evil.example/post'),
  ]) {
    const dom = load(FORM);
    const page = readPage(dom);
    const go = byLabel(page, 'Go');
    change(dom.window.document);
    assert.deepEqual(run(dom, specFor(page, go, {send: true})), {ok: false, stale: 'dest'});
    assert.equal(sends(dom).length, 0);
  }
});

test('the button of an approved form sends it, once', () => {
  const dom = load(FORM);
  const page = readPage(dom);
  const result = run(dom, specFor(page, byLabel(page, 'Go'), {send: true}));
  assert.deepEqual(result, {ok: true});
  assert.equal(sends(dom).length, 1);
});

test('a click on a button that sends a form nobody asked about is stopped and reported', () => {
  const dom = load(FORM);
  const page = readPage(dom);
  const result = run(dom, specFor(page, byLabel(page, 'Go'), {send: false}));
  assert.deepEqual(result, {ok: true, blocked: ['send']});
  assert.equal(sends(dom).length, 0);
});

test('a submit handler that rewrites the address is stopped when the form goes elsewhere', () => {
  const dom = load(FORM, {before: `
    document.addEventListener('DOMContentLoaded', function () {});
  `});
  dom.window.eval(`document.getElementById('f').addEventListener('submit', function () {
    this.setAttribute('action', 'https://evil.example/post');
  });`);
  const page = readPage(dom);
  const result = run(dom, specFor(page, byLabel(page, 'Go'), {send: true}));
  assert.deepEqual(result, {ok: true, blocked: ['send']});
  assert.equal(sends(dom).length, 0);
});

test('form.submit() and requestSubmit() from the page are stopped while melo acts, and put back after', () => {
  const dom = load(FORM);
  dom.window.eval(`
    window.original = HTMLFormElement.prototype.submit;
    document.getElementById('q').addEventListener('change', function () { this.form.submit(); });
    document.querySelector('select').addEventListener('change', function () { this.form.requestSubmit(); });
  `);
  const page = readPage(dom);
  const typed = run(dom, specFor(page, byLabel(page, 'Search'), {act: 'type', text: 'dogs'}));
  assert.deepEqual(typed, {ok: true, blocked: ['send']});
  const chosen = run(dom, specFor(page, byLabel(page, 'Country'), {act: 'select', text: 'Spain'}));
  assert.deepEqual(chosen, {ok: true, blocked: ['send']});
  assert.equal(sends(dom).length, 0);
  assert.equal(dom.window.eval('HTMLFormElement.prototype.submit === window.original'), true, 'put back');
});

test('typing replaces the text and fires the events a page waits for', () => {
  const dom = load(FORM);
  dom.window.eval(`window.heard = []; ['input', 'change'].forEach(function (t) {
    document.getElementById('q').addEventListener(t, function () { window.heard.push(t + ':' + this.value); }); });`);
  const page = readPage(dom);
  assert.deepEqual(run(dom, specFor(page, byLabel(page, 'Search'), {act: 'type', text: 'dogs'})), {ok: true});
  assert.equal(dom.window.document.getElementById('q').value, 'dogs');
  assert.deepEqual(Array.from(dom.window.eval('window.heard')), ['input:dogs', 'change:dogs']);
});

test('Enter sends the form it is in when that is the form that was approved', () => {
  const dom = load(FORM);
  const page = readPage(dom);
  const box = byLabel(page, 'Search');
  const result = run(dom, specFor(page, box, {act: 'type', text: 'dogs', enter: true, send: true}));
  assert.deepEqual(result, {ok: true});
  assert.equal(sends(dom).length, 1);
});

test('a send approved for the address that was read is refused when the box turns out to be in a form', () => {
  const dom = load('<form action="/x"><input id="q" name="q" aria-label="Find"></form>');
  const page = readPage(dom);
  const result = run(dom, specFor(page, page.elements[0], {act: 'type', text: 'a', enter: true, send: true, submit: ''}));
  assert.deepEqual(result, {ok: false, stale: 'dest'});
  assert.equal(sends(dom).length, 0);
});

test('Enter outside a form is a send the page decides: allowed when approved as that, stopped when not', () => {
  const html = '<input id="q" aria-label="Find"><form id="g" action="/elsewhere"></form>';
  const handler = `document.getElementById('q').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') document.getElementById('g').requestSubmit(); });`;
  for (const [send, count] of [[true, 1], [false, 0]]) {
    const dom = load(html);
    dom.window.eval(handler);
    const page = readPage(dom);
    const box = page.elements[0];
    assert.equal(box.submit, undefined, 'no form around the box');
    const result = run(dom, specFor(page, box, {act: 'type', text: 'a', enter: true, send}));
    assert.deepEqual(result, send ? {ok: true} : {ok: true, blocked: ['send']});
    assert.equal(sends(dom).length, count);
  }
});

test('a field that became protected when it was focused is not typed into', () => {
  const dom = load(FORM);
  dom.window.eval(`document.getElementById('q').addEventListener('focus', function () { this.type = 'password'; });`);
  const page = readPage(dom);
  const result = run(dom, specFor(page, byLabel(page, 'Search'), {act: 'type', text: 'dogs'}));
  assert.equal(result.ok, false);
  assert.ok(['field', 'gone'].includes(result.stale), JSON.stringify(result));
  assert.equal(dom.window.document.getElementById('q').value, 'cats', 'nothing was typed');
});

test('a field that turned protected after the text went in is cleared and reported', () => {
  const dom = load(FORM);
  dom.window.eval(`document.getElementById('q').addEventListener('input', function () { this.type = 'password'; });`);
  const page = readPage(dom);
  const result = run(dom, specFor(page, byLabel(page, 'Search'), {act: 'type', text: 'dogs'}));
  assert.equal(result.ok, false);
  assert.ok(['grew', 'gone', 'field'].includes(result.stale), JSON.stringify(result));
  assert.equal(dom.window.document.getElementById('q').value, '', 'what was typed is cleared');
});

test('a link whose address changes at the press is not clicked', () => {
  const dom = load(FORM);
  dom.window.eval(`document.getElementById('n').addEventListener('mousedown', function () { this.href = 'http://127.0.0.1:9/x'; });`);
  const page = readPage(dom);
  assert.deepEqual(run(dom, specFor(page, byLabel(page, 'News'))), {ok: false, stale: 'dest'});
  assert.equal(navigations(dom).length, 0);
});

test('a link whose address changes at the click is stopped and reported', () => {
  const dom = load(FORM);
  dom.window.eval(`document.getElementById('n').addEventListener('click', function () { this.href = 'http://127.0.0.1:9/x'; });`);
  const page = readPage(dom);
  assert.deepEqual(run(dom, specFor(page, byLabel(page, 'News'))), {ok: true, blocked: ['link']});
  assert.equal(navigations(dom).length, 0);
});

test('an option is chosen by its text, and none or several are not', () => {
  const dom = load(FORM + '<select aria-label="Size"><option>Large</option><option>Large print</option></select>');
  const page = readPage(dom);
  const country = byLabel(page, 'Country');
  assert.deepEqual(run(dom, specFor(page, country, {act: 'select', text: 'spain'})), {ok: true});
  assert.equal(dom.window.document.querySelector('select[aria-label=Country]').value, 'Spain');
  assert.deepEqual(run(dom, specFor(page, country, {act: 'select', text: 'Mars'})), {ok: false, option: 0});
  const size = byLabel(page, 'Size');
  assert.deepEqual(run(dom, specFor(page, size, {act: 'select', text: 'Larg'})), {ok: false, option: 2});
});

test('a click on an element that is not an HTML element is dispatched as a click', () => {
  const dom = load('<svg width="20" height="20"><a href="https://example.com/x"><text>go</text></a></svg>');
  dom.window.eval(`window.clicks = 0; document.querySelector('a').addEventListener('click', function (e) { window.clicks++; e.preventDefault(); });`);
  const page = readPage(dom);
  const link = page.elements[0];
  assert.deepEqual(run(dom, specFor(page, link)), {ok: true});
  assert.equal(dom.window.eval('window.clicks'), 1);
});

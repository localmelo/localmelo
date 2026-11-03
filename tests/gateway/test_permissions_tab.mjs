/**
 * Settings > Permissions (permissions-tab.mjs) in a DOM double: the grants in two groups (everywhere, then chat by
 * chat) in the order the gateway listed them, a row's words, who and when, Revoke (a disabled button while it runs,
 * the row gone after a 204 or a 404, kept with the message after any other error), the empty, reading and failed
 * states, a read that began before a revoke, Simplified Chinese, and the `/grants` command that shares the words.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {useLanguage} from '../../support/gateway/static/i18n.mjs';
import {
  grantChat, grantPlace, grantWho, grantWords, grantedWords, setupPermissionsTab,
} from '../../support/gateway/static/permissions-tab.mjs';
import {createActions} from '../../support/gateway/static/slash-actions.mjs';

class Element {
  constructor(tag) {
    Object.assign(this, {tagName: tag.toUpperCase(), children: [], attributes: {}, handlers: {}, dataset: {}, textContent: '',
      className: '', hidden: false, disabled: false, value: '', parent: null, id: '', tabIndex: 0});
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  append(...nodes) { for (const item of nodes) { item.parent = this; this.children.push(item); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  click() { if (!this.disabled) for (const fn of this.handlers.click ?? []) fn({target: this}); }
  focus() { doc.activeElement = this; }
  text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' ').replace(/\s+/g, ' ').trim(); }
  findAll(test_) { return [...(test_(this) ? [this] : []), ...this.children.flatMap((child) => child.findAll(test_))]; }
}
const doc = {activeElement: null, createElement: (tag) => new Element(tag)};

const settle = () => new Promise((resolve) => setImmediate(resolve));
function deferred() {
  let resolve;
  const promise = new Promise((yes) => { resolve = yes; });
  return {promise, resolve};
}
// 2026-10-28 12:00 local time, a Wednesday.
const NOW = new Date(2026, 9, 28, 12, 0, 0).getTime();
const at = (day, hour, minute = 0, month = 9, year = 2026) => new Date(year, month, day, hour, minute, 0).getTime() / 1000;

const grant = (id, extra = {}) => ({
  id, scope: 'global', session_id: 'main', session_kind: 'main', chat_title: '', kind: 'tool', key: 'shell_exec',
  label: 'Run shell commands', action: 'tool', target: 'shell_exec', by: 'person', granted_at: NOW / 1000 - 600, ...extra,
});
const inChat = (id, session, title, extra = {}) => grant(id, {scope: 'session', session_id: session, session_kind: 'side', chat_title: title,
  kind: 'site', key: 'example.com', label: 'Open example.com', action: 'open', target: 'example.com', ...extra});

function harness({grants = [], fail = {}, hold = {}} = {}) {
  const h = {calls: [], grants, changes: 0, hold};
  const api = async (path, options = {}) => {
    const method = options.method ?? 'GET';
    h.calls.push([method, path]);
    const list = h.grants; // what the gateway had when the request arrived
    const wait = h.hold[`${method} ${path}`];
    await wait;
    const failure = fail[`${method} ${path}`];
    if (failure) throw Object.assign(new Error(failure.message), failure);
    if (method === 'GET') return {grants: list};
    h.grants = h.grants.filter((item) => item.id !== path.split('/')[2]);
    return {};
  };
  h.tab = setupPermissionsTab({doc, api, clock: () => NOW, onChange: () => { h.changes += 1; }});
  h.text = () => h.tab.root.text();
  h.status = () => h.tab.root.children[0];
  h.rows = () => h.tab.root.findAll((node) => node.tagName === 'LI' && node.className === 'st-row');
  h.words = (row) => row.children[0].text(); // a row's words, without its button
  h.titles = () => h.tab.root.findAll((node) => /^H[45]$/.test(node.tagName)).map((node) => node.text());
  h.revoke = (id) => h.rows().find((row) => row.dataset.id === id).findAll((node) => node.tagName === 'BUTTON')[0];
  return h;
}

test('the words of a grant: what it allows, in which chat, who decided and when', () => {
  const words = (extra) => grantWords(grant('x', extra));
  assert.equal(words({}), 'Run shell commands');
  assert.equal(words({target: 'python_exec'}), 'Run Python code');
  assert.equal(words({target: 'file_write'}), 'Write files');
  assert.equal(words({target: 'file_edit'}), 'Edit files');
  assert.equal(words({target: 'browser_open'}), 'Use the tool browser_open');
  assert.equal(words({action: 'open', target: 'example.com'}), 'Open example.com');
  assert.equal(words({action: 'act', target: 'example.com'}), 'Click and choose on example.com');
  assert.equal(words({action: 'read', target: 'docs.example.org'}), 'Read docs.example.org');
  assert.equal(words({action: 'folder', target: '~/Documents'}), 'Read files in ~/Documents');
  assert.equal(words({action: 'window', target: 'Messages'}), 'See Messages windows');
  assert.equal(words({action: 'teleport', label: 'Do something new', target: 'x'}), 'Do something new', 'an action this page does not know reads as the gateway worded it');
  assert.equal(words({action: undefined, label: '', target: 'x'}), 'x');

  assert.equal(grantChat(grant('x')), 'Main chat');
  assert.equal(grantChat(inChat('x', 's', 'Trip planning')), 'Trip planning');
  assert.equal(grantChat(inChat('x', 's', '')), 'Untitled chat');
  assert.equal(grantChat(inChat('x', 's', '  ')), 'Untitled chat');
  assert.equal(grantChat(inChat('x', 's', 'Sort photos', {session_kind: 'task'})), 'Task: Sort photos');
  assert.equal(grantChat(inChat('x', 's', '', {session_kind: 'task'})), 'Untitled task');
  assert.equal(grantChat(inChat('x', 's', '', {session_kind: 'terminal'})), 'Terminal chat');
  assert.equal(grantPlace(grant('x')), 'everywhere');
  assert.equal(grantPlace(inChat('x', 's', 'Trip planning')), 'Trip planning');
  assert.equal(grantWho(grant('x')), 'You');
  assert.equal(grantWho(grant('x', {by: 'auto'})), 'Auto');

  const now = NOW / 1000;
  assert.equal(grantedWords(now - 20, now), 'just now');
  assert.equal(grantedWords(now + 5, now), 'just now', 'a clock a little ahead is not a time in the future');
  assert.equal(grantedWords(now - 12 * 60, now), '12 min ago');
  assert.match(grantedWords(at(28, 8, 5), now), /^today at 8:05\s?AM$/i);
  assert.match(grantedWords(at(27, 21, 30), now), /^yesterday at 9:30\s?PM$/i);
  assert.match(grantedWords(at(20, 9), now), /^Oct 20$/);
  assert.match(grantedWords(at(3, 9, 0, 0, 2025), now), /^Jan 3, 2025$/);
  assert.equal(grantedWords(undefined, now), '');
});

test('the section says it is reading, then lists the grants in two groups in the gateway\'s order', async () => {
  const h = harness({grants: [
    grant('g1', {granted_at: at(28, 11, 40)}),
    grant('g2', {action: 'folder', target: '~/Documents', by: 'auto', granted_at: at(27, 21, 30)}),
    inChat('c1', 'trip', 'Trip planning', {granted_at: at(28, 11, 55)}),
    inChat('c2', 'trip', 'Trip planning', {action: 'act', granted_at: at(28, 10, 0)}),
    inChat('c3', 'chores', '', {action: 'read', by: 'auto', granted_at: at(20, 9)}),
    inChat('c4', 'sort', 'Sort photos', {session_kind: 'task', action: 'tool', target: 'file_write'}),
  ]});
  assert.match(h.text(), /^Reading what you allowed…$/);
  assert.equal(h.status().attributes.role, 'status');
  assert.equal(h.tab.count(), null, 'no count before the list was read');
  await h.tab.show();
  assert.deepEqual(h.calls, [['GET', '/grants']]);
  assert.equal(h.tab.count(), 6);
  assert.equal(h.changes, 1, 'the page is told the list arrived');
  assert.deepEqual(h.titles(), ['Everywhere', 'Per chat', 'Trip planning', 'Untitled chat', 'Task: Sort photos']);
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g1', 'g2', 'c1', 'c2', 'c3', 'c4'], 'the order is the gateway\'s, which /grants numbers');
  const lines = h.rows().map(h.words);
  assert.match(lines[0], /^Run shell commands From the main chat · You · 20 min ago$/);
  assert.match(lines[1], /^Read files in ~\/Documents From the main chat · Auto · yesterday at 9:30\s?PM$/i);
  assert.match(lines[2], /^Open example\.com You · 5 min ago$/, 'under a chat\'s title the row does not repeat the chat');
  assert.match(lines[3], /^Click and choose on example\.com You · today at 10:00\s?AM$/i);
  assert.match(lines[4], /^Read example\.com Auto · Oct 20$/);
  assert.match(lines[5], /^Write files You · 10 min ago$/);
  // The groups are real lists: a list for Everywhere, and one list under each chat's heading.
  const lists = h.tab.root.findAll((node) => node.tagName === 'UL');
  assert.equal(lists.length, 4);
  assert.deepEqual(lists.map((list) => list.children.length), [2, 2, 1, 1]);
  assert.ok(lists.every((list) => list.children.every((item) => item.tagName === 'LI')));
  const chats = h.tab.root.findAll((node) => node.className === 'perm-chat');
  assert.deepEqual(chats.map((box) => box.attributes.role), ['group', 'group', 'group']);
  assert.deepEqual(chats.map((box) => box.dataset.chat), ['trip', 'chores', 'sort']);
  const heading = chats[0].findAll((node) => node.tagName === 'H5')[0];
  assert.equal(chats[0].attributes['aria-labelledby'], heading.id);
  assert.equal(lists[2].attributes['aria-labelledby'], lists[2].parent.children[0].id);
});

test('a group without rows is not shown', async () => {
  const onlyChats = harness({grants: [inChat('c1', 'trip', 'Trip planning')]});
  await onlyChats.tab.show();
  assert.deepEqual(onlyChats.titles(), ['Per chat', 'Trip planning']);
  const onlyEverywhere = harness({grants: [grant('g1')]});
  await onlyEverywhere.tab.show();
  assert.deepEqual(onlyEverywhere.titles(), ['Everywhere']);
});

test('with nothing allowed the section says when something will show here', async () => {
  const h = harness();
  await h.tab.show();
  assert.match(h.text(), /Nothing allowed yet\. When melo asks and you choose to allow for this chat, it shows here\./);
  assert.equal(h.rows().length, 0);
  assert.equal(h.tab.count(), 0);
});

test('Revoke is a real button that names what it revokes, and the request runs with the button disabled', async () => {
  const wait = deferred();
  const h = harness({grants: [grant('g1'), inChat('c1', 'trip', 'Trip planning')], hold: {}});
  await h.tab.show();
  const button = h.revoke('g1');
  assert.equal(button.tagName, 'BUTTON');
  assert.equal(button.textContent, 'Revoke');
  assert.equal(button.type, 'button');
  assert.equal(button.attributes['aria-label'], 'Revoke: Run shell commands');
  h.hold['DELETE /grants/g1'] = wait.promise;
  button.click();
  await settle();
  assert.deepEqual(h.calls.at(-1), ['DELETE', '/grants/g1']);
  assert.equal(button.disabled, true, 'the button waits while the request runs');
  assert.equal(h.rows().length, 2, 'the row stays until the gateway answers');
  button.click();
  await settle();
  assert.equal(h.calls.filter((call) => call[0] === 'DELETE').length, 1, 'a second click does not send a second request');
  wait.resolve();
  await settle();
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['c1']);
  assert.deepEqual(h.titles(), ['Per chat', 'Trip planning'], 'the group goes with its last row');
  assert.equal(h.status().textContent, 'Revoked: Run shell commands. melo asks again next time.');
  assert.equal(h.status().dataset.state, '');
  assert.equal(h.tab.count(), 1);
  assert.equal(h.changes, 2, 'the page is told the count changed');
});

test('a revoked row hands the focus to the next one, and to the section when it was the last', async () => {
  const h = harness({grants: [grant('g1'), grant('g2'), inChat('c1', 'trip', 'Trip planning')]});
  await h.tab.show();
  h.revoke('g2').focus();
  h.revoke('g2').click();
  await settle();
  assert.equal(doc.activeElement, h.revoke('c1'), 'the row that took its place');
  h.revoke('c1').focus();
  h.revoke('c1').click();
  await settle();
  assert.equal(doc.activeElement, h.revoke('g1'), 'the one before when it was the last');
  h.revoke('g1').focus();
  h.revoke('g1').click();
  await settle();
  assert.equal(doc.activeElement, h.tab.root, 'no button left');
  assert.equal(h.tab.root.tabIndex, -1);
  assert.match(h.text(), /Nothing allowed yet/);
});

test('a grant the gateway no longer has is gone: the row goes and there is no complaint', async () => {
  const h = harness({grants: [grant('g1'), grant('g2')], fail: {'DELETE /grants/g1': {status: 404, code: 'not_found', message: 'No such grant'}}});
  await h.tab.show();
  h.revoke('g1').click();
  await settle();
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g2']);
  assert.equal(h.status().textContent, '');
  assert.equal(h.status().dataset.state, '');
  assert.doesNotMatch(h.text(), /No such grant/);
  assert.equal(h.tab.count(), 1);
});

test('any other error keeps the row and says why in the status line, and the button works again', async () => {
  const h = harness({grants: [grant('g1')], fail: {'DELETE /grants/g1': {status: 500, message: 'The gateway could not save this.'}}});
  await h.tab.show();
  const button = h.revoke('g1');
  button.click();
  await settle();
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g1']);
  assert.equal(h.status().textContent, 'The gateway could not save this.');
  assert.equal(h.status().dataset.state, 'error');
  assert.equal(h.status().attributes['aria-live'], 'polite');
  assert.equal(h.revoke('g1').disabled, false);
  assert.equal(h.tab.count(), 1);
  // A message-less failure still says something.
  const bare = harness({grants: [grant('g1')], fail: {'DELETE /grants/g1': {status: 500, message: ''}}});
  await bare.tab.show();
  bare.revoke('g1').click();
  await settle();
  assert.equal(bare.status().textContent, 'Could not revoke it. Try again.');
});

test('a list that cannot be read says so and offers Retry; a failed refresh keeps the list', async () => {
  const fail = {'GET /grants': {status: 500, message: 'The gateway is not answering.'}};
  const h = harness({grants: [grant('g1')], fail});
  await h.tab.show();
  assert.equal(h.status().textContent, 'The gateway is not answering.');
  assert.equal(h.status().dataset.state, 'error');
  assert.equal(h.tab.count(), null);
  const retry = h.tab.root.findAll((node) => node.tagName === 'BUTTON' && node.textContent === 'Retry')[0];
  assert.ok(retry, 'a Retry button');
  delete fail['GET /grants'];
  retry.click();
  assert.match(h.tab.root.children[1].text(), /Reading what you allowed…/, 'it reads again at once');
  await settle();
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g1']);
  assert.equal(h.status().textContent, '');
  assert.equal(h.status().dataset.state, '');
  // Once listed, a refresh that fails leaves the list as it was and tells the person.
  fail['GET /grants'] = {status: 500, message: ''};
  await h.tab.show();
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g1']);
  assert.equal(h.status().textContent, 'Could not read what you allowed.');
});

test('a read that began before a revoke cannot bring the revoked row back, and a reset forgets the list', async () => {
  const slow = deferred();
  const h = harness({grants: [grant('g1'), grant('g2')]});
  await h.tab.show();
  h.hold['GET /grants'] = slow.promise;
  const reading = h.tab.show(); // sees both grants
  await settle();
  delete h.hold['GET /grants'];
  h.revoke('g1').click();
  await settle();
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g2']);
  slow.resolve();
  await reading;
  assert.deepEqual(h.rows().map((row) => row.dataset.id), ['g2'], 'the old answer is dropped');
  h.tab.reset();
  assert.equal(h.tab.count(), null);
  assert.match(h.text(), /^Reading what you allowed…$/);
  // An answer for the visit before the reset changes nothing either.
  const late = deferred();
  h.hold['GET /grants'] = late.promise;
  const before = h.tab.show();
  await settle();
  h.tab.reset();
  late.resolve();
  await before;
  assert.equal(h.tab.count(), null);
  assert.match(h.text(), /^Reading what you allowed…$/);
});

test('the section reads in Simplified Chinese', async () => {
  await useLanguage('zh-Hans');
  try {
    const h = harness({grants: [
      grant('g1'),
      grant('g2', {action: 'window', target: 'Messages', by: 'auto', granted_at: NOW / 1000 - 30}),
      inChat('c1', 'trip', '', {action: 'folder', target: '~/Documents'}),
      inChat('c2', 'sort', '整理照片', {session_kind: 'task', action: 'act'}),
      inChat('c3', 'term', '', {session_kind: 'terminal', action: 'tool', target: 'file_edit', granted_at: at(27, 9)}),
    ]});
    assert.match(h.text(), /^正在读取你允许过的事…$/);
    await h.tab.show();
    assert.deepEqual(h.titles(), ['所有对话', '单个对话', '未命名对话', '任务：整理照片', '终端对话']);
    const lines = h.rows().map(h.words);
    assert.equal(lines[0], '运行 shell 命令 来自主对话 · 你 · 10 分钟前');
    assert.equal(lines[1], '查看 Messages 的窗口 来自主对话 · 自动模式 · 刚刚');
    assert.equal(lines[2], '读取 ~/Documents 里的文件 你 · 10 分钟前');
    assert.equal(lines[3], '在 example.com 上点击和选择 你 · 10 分钟前');
    assert.match(lines[4], /^编辑文件 你 · 昨天 /);
    assert.equal(h.revoke('g1').textContent, '收回');
    assert.equal(h.revoke('g1').attributes['aria-label'], '收回：运行 shell 命令');
    h.revoke('g1').click();
    await settle();
    assert.equal(h.status().textContent, '已收回：运行 shell 命令。下次 melo 会再问你。');
    assert.equal(grantWords(grant('x', {target: 'browser_open'})), '使用工具 browser_open');
    assert.equal(grantPlace(grant('x')), '所有对话');
    const empty = harness();
    await empty.tab.show();
    assert.match(empty.text(), /还没有允许过的事。当 melo 来问你、你选择为这个对话允许时，会显示在这里。/);
  } finally {
    await useLanguage('en');
  }
});

// ── /grants ──

function slashHarness(grants) {
  const calls = [];
  let list = grants;
  const api = async (path, options = {}) => {
    const method = options.method ?? 'GET';
    calls.push([method, path]);
    if (method === 'GET') return {grants: list};
    const id = path.split('/')[2];
    if (!list.some((item) => item.id === id)) throw Object.assign(new Error('No such grant'), {status: 404, code: 'not_found'});
    list = list.filter((item) => item.id !== id);
    return {};
  };
  const actions = createActions({api, state: () => ({})});
  return {calls, actions, grants: () => list};
}
const ORDER = [
  grant('g1'),
  grant('g2', {action: 'folder', target: '~/Documents', by: 'auto'}),
  inChat('c1', 'trip', 'Trip planning'),
  inChat('c2', 'chores', '', {action: 'act'}),
];

test('/grants lists every grant, numbered in the gateway\'s order, with what, where and who', async () => {
  const s = slashHarness(ORDER);
  const rows = await s.actions.grants({args: ''});
  assert.deepEqual(s.calls, [['GET', '/grants']]);
  assert.deepEqual(rows, [
    {text: '1. Run shell commands · everywhere · You', tone: undefined},
    {text: '2. Read files in ~/Documents · everywhere · Auto', tone: undefined},
    {text: '3. Open example.com · Trip planning · You', tone: undefined},
    {text: '4. Click and choose on example.com · Untitled chat · You', tone: undefined},
  ]);
  const empty = await slashHarness([]).actions.grants({args: ''});
  assert.deepEqual(empty, [{text: 'No grants.', tone: 'muted'}]);
});

test('/grants revoke N takes the Nth one back, and says what', async () => {
  const s = slashHarness(ORDER);
  const rows = await s.actions.grants({args: 'revoke 3'});
  assert.deepEqual(s.calls, [['GET', '/grants'], ['DELETE', '/grants/c1']]);
  assert.deepEqual(rows, [{text: 'Revoked: Open example.com.', tone: 'ok'}]);
  assert.deepEqual(s.grants().map((item) => item.id), ['g1', 'g2', 'c2']);
  const again = await s.actions.grants({args: 'revoke   1'});
  assert.deepEqual(again, [{text: 'Revoked: Run shell commands.', tone: 'ok'}]);
  // One that went meanwhile is gone either way.
  const lost = createActions({api: async (path, options = {}) => {
    if (!options.method) return {grants: ORDER};
    throw Object.assign(new Error('No such grant'), {status: 404, code: 'not_found'});
  }, state: () => ({})});
  assert.deepEqual(await lost.grants({args: 'revoke 1'}), [{text: 'Revoked: Run shell commands.', tone: 'ok'}]);
});

test('/grants with anything else says how to use it, without asking the gateway', async () => {
  const s = slashHarness(ORDER);
  for (const args of ['revoke', 'revoke x', 'revoke 1 2', 'revoke -1', 'forget 1', '2', 'revoke 1.5']) {
    await assert.rejects(s.actions.grants({args}), {message: 'Use /grants or /grants revoke N.'}, args);
  }
  assert.deepEqual(s.calls, []);
  // A number the list does not have is named, with where to see the list.
  await assert.rejects(s.actions.grants({args: 'revoke 9'}), {message: 'There is no grant 9. /grants lists them.'});
  await assert.rejects(s.actions.grants({args: 'revoke 0'}), {message: 'There is no grant 0. /grants lists them.'});
  assert.deepEqual(s.calls.filter((call) => call[0] === 'DELETE'), []);
  // Any other error from the gateway reaches the row that shows it.
  const broken = createActions({api: async (path, options = {}) => {
    if (!options.method) return {grants: ORDER};
    throw Object.assign(new Error('The gateway could not save this.'), {status: 500});
  }, state: () => ({})});
  await assert.rejects(broken.grants({args: 'revoke 2'}), {message: 'The gateway could not save this.'});
});

test('/grants reads in Simplified Chinese', async () => {
  await useLanguage('zh-Hans');
  try {
    const s = slashHarness(ORDER);
    const rows = await s.actions.grants({args: ''});
    assert.equal(rows[1].text, '2. 读取 ~/Documents 里的文件 · 所有对话 · 自动模式');
    assert.equal(rows[2].text, '3. 打开 example.com · Trip planning · 你');
    assert.equal((await slashHarness([]).actions.grants({args: ''}))[0].text, '没有授权。');
    assert.equal((await s.actions.grants({args: 'revoke 2'}))[0].text, '已收回：读取 ~/Documents 里的文件。');
    await assert.rejects(s.actions.grants({args: 'revoke'}), {message: '用法：/grants，或 /grants revoke N。'});
    await assert.rejects(s.actions.grants({args: 'revoke 9'}), {message: '没有第 9 项授权。/grants 可以列出它们。'});
  } finally {
    await useLanguage('en');
  }
});
